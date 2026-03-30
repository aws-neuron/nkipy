import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from config import FULL_ATTENTION, LINEAR_ATTENTION, Config, get_config
from kernels.sampling import compute_logits
from kernels.transformer_layer import (
    transformer_layer_full_attn,
    transformer_layer_linear_attn,
)
from nkipy.runtime import DeviceKernel, DeviceTensor
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils import print_log

BUILD_DIR = "./build"


class Qwen35Model:
    def __init__(self, model_weights, config: Config):
        self.config = config
        self.tok_embedding = model_weights.get("tok_embedding")

        # Kernels (compiled lazily)
        self.kernel_cte_full_attn = None
        self.kernel_cte_linear_attn = None
        self.kernel_tkg_full_attn = None
        self.kernel_tkg_linear_attn = None
        self.kernel_cte_greedy_sampling = None
        self.kernel_tkg_greedy_sampling = None

        self.norm_weight = None
        self.lm_head_weight = None

        self._prepare_tensors(model_weights)
        self._prepare_kernels()

    def _prepare_tensors(self, weights):
        t = time.time()
        print_log("Preparing Tensors")

        ws = dist.get_world_size()
        n_local_kv_heads = max(1, self.config.num_kv_heads // ws)
        n_local_v_heads = self.config.linear_num_value_heads // ws
        n_local_k_heads = self.config.linear_num_key_heads // ws
        key_dim_local = n_local_k_heads * self.config.linear_key_head_dim
        value_dim_local = n_local_v_heads * self.config.linear_value_head_dim
        conv_dim_local = key_dim_local * 2 + value_dim_local

        self.layer_tensors = []
        for layer_id in range(self.config.num_layers):
            layer_type = self.config.layer_types[layer_id]
            layer_dict = {}

            # Common MoE weights
            for key in [
                "input_weight",
                "post_attention_weight",
                "router_weight",
                "gate_up_weight",
                "down_weight",
                "shared_gate_proj_weight",
                "shared_up_proj_weight",
                "shared_down_proj_weight",
                "shared_expert_gate_weight",
            ]:
                w = weights.get(f"layers.{layer_id}.{key}")
                layer_dict[key] = DeviceTensor.from_torch(
                    w, f"{key}_L{layer_id}"
                )

            if layer_type == FULL_ATTENTION:
                # Full attention weights + KV cache
                for key in ["qkv_weight", "o_weight", "q_norm_weight", "k_norm_weight"]:
                    w = weights.get(f"layers.{layer_id}.{key}")
                    layer_dict[key] = DeviceTensor.from_torch(
                        w, f"{key}_L{layer_id}"
                    )

                cache_k = np.zeros(
                    (self.config.max_batch_size, self.config.max_seq_len,
                     n_local_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype,
                )
                cache_v = np.zeros_like(cache_k)
                layer_dict["cache_k"] = DeviceTensor.from_numpy(
                    cache_k, f"cache_k_L{layer_id}"
                )
                layer_dict["cache_v"] = DeviceTensor.from_numpy(
                    cache_v, f"cache_v_L{layer_id}"
                )

            else:  # LINEAR_ATTENTION
                for key in [
                    "linear_qkv_weight", "linear_z_weight",
                    "linear_b_weight", "linear_a_weight",
                    "linear_conv_weight", "linear_dt_bias",
                    "linear_A_log", "linear_norm_weight",
                    "linear_out_weight",
                ]:
                    w = weights.get(f"layers.{layer_id}.{key}")
                    layer_dict[key] = DeviceTensor.from_torch(
                        w, f"{key}_L{layer_id}"
                    )

                conv_state = np.zeros(
                    (self.config.max_batch_size, conv_dim_local,
                     self.config.linear_conv_kernel_dim),
                    dtype=self.config.dtype,
                )
                recurrent_state = np.zeros(
                    (self.config.max_batch_size, n_local_v_heads,
                     self.config.linear_key_head_dim,
                     self.config.linear_value_head_dim),
                    dtype=self.config.dtype,
                )
                layer_dict["conv_state"] = DeviceTensor.from_numpy(
                    conv_state, f"conv_state_L{layer_id}"
                )
                layer_dict["recurrent_state"] = DeviceTensor.from_numpy(
                    recurrent_state, f"recurrent_state_L{layer_id}"
                )

            layer_dict["layer_type"] = layer_type
            self.layer_tensors.append(layer_dict)

        self.norm_weight = DeviceTensor.from_torch(
            weights.get("norm_weight"), "norm_weight"
        )
        self.lm_head_weight = DeviceTensor.from_torch(
            weights.get("lm_head_weight"), "lm_head_weight"
        )

        print_log(f"--> Finished Preparing Tensors in {time.time() - t:.2f}s")

    def _find_first_layer_of_type(self, layer_type):
        for i, lt in enumerate(self.layer_tensors):
            if lt["layer_type"] == layer_type:
                return i
        return None

    def _prepare_kernels(self):
        t = time.time()
        print_log("Preparing kernels")

        x_context = DeviceTensor.from_numpy(
            np.empty(
                (self.config.max_batch_size, self.config.context_len,
                 self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "x_context",
        )
        x_token = DeviceTensor.from_numpy(
            np.empty(
                (self.config.max_batch_size, 1, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "x_token",
        )
        start_pos = DeviceTensor.from_numpy(
            np.empty(shape=(1), dtype=np.int32), "start_pos"
        )

        # --- Compile full attention kernels ---
        fa_idx = self._find_first_layer_of_type(FULL_ATTENTION)
        if fa_idx is not None:
            fa = self.layer_tensors[fa_idx]
            fa_common = dict(
                qkv_weight=fa["qkv_weight"],
                o_weight=fa["o_weight"],
                input_weight=fa["input_weight"],
                q_norm_weight=fa["q_norm_weight"],
                k_norm_weight=fa["k_norm_weight"],
                post_attention_weight=fa["post_attention_weight"],
                router_weight=fa["router_weight"],
                gate_up_weight=fa["gate_up_weight"],
                down_weight=fa["down_weight"],
                shared_gate_proj_weight=fa["shared_gate_proj_weight"],
                shared_up_proj_weight=fa["shared_up_proj_weight"],
                shared_down_proj_weight=fa["shared_down_proj_weight"],
                shared_expert_gate_weight=fa["shared_expert_gate_weight"],
                configs=self.config,
                build_dir=BUILD_DIR,
                additional_compiler_args=self.config.additional_compiler_args_nkipy,
            )

            self.kernel_cte_full_attn = DeviceKernel.compile_and_load(
                transformer_layer_full_attn,
                name="cte_full_attn",
                x=x_context,
                start_pos=None,
                cache_k=fa["cache_k"],
                cache_v=fa["cache_v"],
                **fa_common,
            )
            self.kernel_tkg_full_attn = DeviceKernel.compile_and_load(
                transformer_layer_full_attn,
                name="tkg_full_attn",
                x=x_token,
                start_pos=start_pos,
                cache_k=fa["cache_k"],
                cache_v=fa["cache_v"],
                **fa_common,
            )

        # --- Compile linear attention kernels ---
        la_idx = self._find_first_layer_of_type(LINEAR_ATTENTION)
        if la_idx is not None:
            la = self.layer_tensors[la_idx]
            la_common = dict(
                qkv_weight=la["linear_qkv_weight"],
                z_weight=la["linear_z_weight"],
                b_weight=la["linear_b_weight"],
                a_weight=la["linear_a_weight"],
                conv_weight=la["linear_conv_weight"],
                dt_bias=la["linear_dt_bias"],
                A_log=la["linear_A_log"],
                linear_norm_weight=la["linear_norm_weight"],
                out_weight=la["linear_out_weight"],
                input_weight=la["input_weight"],
                post_attention_weight=la["post_attention_weight"],
                router_weight=la["router_weight"],
                gate_up_weight=la["gate_up_weight"],
                down_weight=la["down_weight"],
                shared_gate_proj_weight=la["shared_gate_proj_weight"],
                shared_up_proj_weight=la["shared_up_proj_weight"],
                shared_down_proj_weight=la["shared_down_proj_weight"],
                shared_expert_gate_weight=la["shared_expert_gate_weight"],
                configs=self.config,
                build_dir=BUILD_DIR,
                additional_compiler_args=self.config.additional_compiler_args_nkipy,
            )

            self.kernel_cte_linear_attn = DeviceKernel.compile_and_load(
                transformer_layer_linear_attn,
                name="cte_linear_attn",
                x=x_context,
                start_pos=None,
                conv_state=la["conv_state"],
                recurrent_state=la["recurrent_state"],
                **la_common,
            )
            self.kernel_tkg_linear_attn = DeviceKernel.compile_and_load(
                transformer_layer_linear_attn,
                name="tkg_linear_attn",
                x=x_token,
                start_pos=start_pos,
                conv_state=la["conv_state"],
                recurrent_state=la["recurrent_state"],
                **la_common,
            )

        # --- Compile logits kernels (argmax done on CPU) ---
        ws = dist.get_world_size()
        vocab_per_device = self.lm_head_weight.numpy().shape[1]
        self.vocab_per_device = vocab_per_device

        self.d_logits_ctx = DeviceTensor.from_numpy(
            np.empty((self.config.max_batch_size, vocab_per_device), dtype=np.float32),
            "logits_ctx",
        )
        self.d_logits_tok = DeviceTensor.from_numpy(
            np.empty((self.config.max_batch_size, vocab_per_device), dtype=np.float32),
            "logits_tok",
        )

        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            compute_logits,
            name="cte_logits",
            h=x_context,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )
        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            compute_logits,
            name="tkg_logits",
            h=x_token,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )

        print_log(
            f"--> Finished Kernel Compilation and Loading in {time.time() - t:.2f}s"
        )

    def _run_layer(self, kernel_fa, kernel_la, layer_idx, hidden_states, t_start_pos):
        lt = self.layer_tensors[layer_idx]
        layer_type = lt["layer_type"]

        # Common MoE inputs
        moe_inputs = {
            "input_weight": lt["input_weight"],
            "post_attention_weight": lt["post_attention_weight"],
            "router_weight": lt["router_weight"],
            "gate_up_weight": lt["gate_up_weight"],
            "down_weight": lt["down_weight"],
            "shared_gate_proj_weight": lt["shared_gate_proj_weight"],
            "shared_up_proj_weight": lt["shared_up_proj_weight"],
            "shared_down_proj_weight": lt["shared_down_proj_weight"],
            "shared_expert_gate_weight": lt["shared_expert_gate_weight"],
        }

        if layer_type == FULL_ATTENTION:
            inputs = {
                "x": hidden_states,
                "qkv_weight": lt["qkv_weight"],
                "o_weight": lt["o_weight"],
                "q_norm_weight": lt["q_norm_weight"],
                "k_norm_weight": lt["k_norm_weight"],
                "cache_k.must_alias_input": lt["cache_k"],
                "cache_v.must_alias_input": lt["cache_v"],
                **moe_inputs,
            }
            if t_start_pos is not None:
                inputs["start_pos"] = t_start_pos
            outputs = {
                "output0": hidden_states,
                "cache_k": lt["cache_k"],
                "cache_v": lt["cache_v"],
            }
            kernel_fa(inputs=inputs, outputs=outputs)
        else:
            inputs = {
                "x": hidden_states,
                "qkv_weight": lt["linear_qkv_weight"],
                "z_weight": lt["linear_z_weight"],
                "b_weight": lt["linear_b_weight"],
                "a_weight": lt["linear_a_weight"],
                "conv_weight": lt["linear_conv_weight"],
                "dt_bias": lt["linear_dt_bias"],
                "A_log": lt["linear_A_log"],
                "linear_norm_weight": lt["linear_norm_weight"],
                "out_weight": lt["linear_out_weight"],
                "conv_state.must_alias_input": lt["conv_state"],
                "recurrent_state.must_alias_input": lt["recurrent_state"],
                **moe_inputs,
            }
            if t_start_pos is not None:
                inputs["start_pos"] = t_start_pos
            outputs = {
                "output0": hidden_states,
                "conv_state": lt["conv_state"],
                "recurrent_state": lt["recurrent_state"],
            }
            kernel_la(inputs=inputs, outputs=outputs)

    def _sample_token(self, kernel, hidden_states, d_logits):
        """Run logits kernel on device, then argmax on CPU across all TP ranks."""
        kernel(
            inputs={
                "h": hidden_states,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
            },
            outputs={"output0": d_logits},
        )
        # Read partial logits (this rank's vocab shard) to CPU
        local_logits = torch.from_numpy(d_logits.numpy().astype(np.float32))

        ws = dist.get_world_size()
        if ws > 1:
            # All-gather partial logits from all ranks on CPU
            gathered = [torch.empty_like(local_logits) for _ in range(ws)]
            dist.all_gather(gathered, local_logits)
            all_logits = torch.cat(gathered, dim=-1)  # (B, full_vocab)
        else:
            all_logits = local_logits

        # Argmax on CPU
        next_id = all_logits.argmax(dim=-1, keepdim=True).to(dtype=torch.int)  # (B, 1)
        return next_id

    def generate(self, input_ids):
        context_len = self.config.context_len

        # Reset GDN states (conv_state, recurrent_state) before each generation
        # Unlike KV cache (position-addressed), GDN state is accumulated, so must be zeroed.
        for lt in self.layer_tensors:
            if lt["layer_type"] == LINEAR_ATTENTION:
                lt["conv_state"].write_from_numpy(
                    np.zeros(lt["conv_state"].numpy().shape, dtype=lt["conv_state"].numpy().dtype)
                )
                lt["recurrent_state"].write_from_numpy(
                    np.zeros(lt["recurrent_state"].numpy().shape, dtype=lt["recurrent_state"].numpy().dtype)
                )

        hidden_states = DeviceTensor.from_torch(
            self.tok_embedding[input_ids], "hidden_states"
        )

        # --- Prefill (context phase) ---
        for i in range(self.config.num_layers):
            self._run_layer(
                self.kernel_cte_full_attn,
                self.kernel_cte_linear_attn,
                i,
                hidden_states,
                None,
            )

        next_id_torch = self._sample_token(
            self.kernel_cte_greedy_sampling, hidden_states, self.d_logits_ctx
        )
        yield next_id_torch

        # --- Decode (token-by-token) ---
        for pos in range(context_len, context_len + self.config.max_new_tokens):
            t_start_pos = DeviceTensor.from_numpy(
                np.array([pos], dtype=np.int32)
            )
            hidden_states = DeviceTensor.from_torch(
                self.tok_embedding[next_id_torch], "h0/res1"
            )
            t_res1 = hidden_states

            for i in range(self.config.num_layers):
                self._run_layer(
                    self.kernel_tkg_full_attn,
                    self.kernel_tkg_linear_attn,
                    i,
                    hidden_states,
                    t_start_pos,
                )

            next_id_torch = self._sample_token(
                self.kernel_tkg_greedy_sampling, t_res1, self.d_logits_tok
            )
            yield next_id_torch


def load_model(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"

    dist.init_process_group()
    torch.set_num_threads(128 // dist.get_world_size())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_inputs = tokenizer(args.prompt, return_tensors="np")
    input_ids = model_inputs["input_ids"]
    config = get_config(args.model, input_ids.shape[1], args.max_new_tokens)

    print_log("Loading Model Weights")

    shard_path = os.path.join(
        args.checkpoint, f"shard_{dist.get_rank()}.safetensors"
    )
    weights = load_file(shard_path, device="cpu")

    model = Qwen35Model(weights, config)

    # Warming
    start = time.time()
    print_log("Warming model")
    t = 0
    for id in model.generate(input_ids):
        if t == 1:
            break
        t += 1
    print_log(f"--> Finished warming the model in {time.time() - start:.2f}s")

    return model, input_ids, tokenizer


# EOS tokens for Qwen3.5
EOS_TOKENS = {248044}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument(
        "--checkpoint", default="./qwen3_5_shards"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    args = parser.parse_args()

    model, input_ids, tokenizer = load_model(args)

    dist.barrier()
    start = time.time()
    t = 0
    if dist.get_rank() == 0:
        print(f"\n{args.prompt}", end="")
    for id in model.generate(input_ids):
        if t == 0:
            first_token_time = time.time()
        t += 1
        output_id = id[0].tolist()
        if output_id[-1] in EOS_TOKENS:
            print_log("Found EOS token, stop early")
            break
        if dist.get_rank() == 0:
            print(tokenizer.decode(output_id), end="")
            sys.stdout.flush()

    end_time = time.time()

    ttft = first_token_time - start
    decoding_time = end_time - first_token_time
    tokens_per_second = t / decoding_time
    if dist.get_rank() == 0:
        print(f"\nTime to first token: {ttft:.2f}s")
        print(f"Decoding tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
