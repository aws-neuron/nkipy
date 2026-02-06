import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from config import Config, get_config
from kernels.sampling import greedy_sampling
from kernels.transformer_layer import transformer_layer
from nkipy.runtime import DeviceKernel, DeviceTensor
from parallel_state import initialize_model_parallel
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils import print_log

BUILD_DIR = "./build"
USE_NKI_RMSNORM = True


class Qwen3Model:
    def __init__(self, model_weights, config: Config):
        """Initialize the model with weights and configuration"""
        self.config = config

        # Load and prepare all model resources
        self.tok_embedding = model_weights.get("tok_embedding")

        # Initialize kernels to None - will be loaded lazily
        self.kernel_cte_greedy_sampling = None
        self.kernel_tkg = None
        self.kernel_tkg_greedy_sampling = None
        self.block_wise_moe_layers = []

        self.norm_weight = None
        self.lm_head_weight = None

        # Prepare model resources
        self._prepare_tensors(model_weights)
        # Ensure kernels are compiled
        self._prepare_kernels()

    def _prepare_tensors(self, weights):
        """Prepare all resources needed by the model"""

        t = time.time()
        print_log("Preparing Tensors")

        n_local_kv_heads = max(1, self.config.n_kv_heads // dist.get_world_size())

        cache_k = np.zeros(
            (
                self.config.max_batch_size,
                self.config.max_seq_len,
                n_local_kv_heads,
                self.config.head_dim,
            ),
            dtype=self.config.dtype,
        )
        cache_v = np.zeros(
            (
                self.config.max_batch_size,
                self.config.max_seq_len,
                n_local_kv_heads,
                self.config.head_dim,
            ),
            dtype=self.config.dtype,
        )

        # model weights
        norm_weight = weights.get("norm_weight")
        lm_head_weight = weights.get("lm_head_weight")

        # Prepare layer weights and tensors as class members
        self.layer_tensors = []
        for layer_id in range(self.config.n_layers):
            qkv_weight = weights.get(f"layers.{layer_id}.qkv_weight")
            o_weight = weights.get(f"layers.{layer_id}.o_weight")
            gate_up_weight = weights.get(f"layers.{layer_id}.gate_up_weight")
            down_weight = weights.get(f"layers.{layer_id}.down_weight")
            router_weight = weights.get(f"layers.{layer_id}.router_weight")
            q_norm_weight = weights.get(f"layers.{layer_id}.q_norm_weight")
            k_norm_weight = weights.get(f"layers.{layer_id}.k_norm_weight")
            input_weight = weights.get(f"layers.{layer_id}.input_weight")
            post_attention_weight = weights.get(
                f"layers.{layer_id}.post_attention_weight"
            )

            # Create DeviceTensors for this layer
            self.layer_tensors.append(
                {
                    "qkv_weight": DeviceTensor.from_torch(
                        qkv_weight, f"qkv_weight_L{layer_id}"
                    ),
                    "o_weight": DeviceTensor.from_torch(
                        o_weight, f"o_weight_L{layer_id}"
                    ),
                    "gate_up_weight": DeviceTensor.from_torch(
                        gate_up_weight, f"gate_up_weight_L{layer_id}"
                    ),
                    "down_weight": DeviceTensor.from_torch(
                        down_weight, f"down_weight_L{layer_id}"
                    ),
                    "router_weight": DeviceTensor.from_torch(
                        router_weight, f"router_weight_L{layer_id}"
                    ),
                    "q_norm_weight": DeviceTensor.from_torch(
                        q_norm_weight, f"q_norm_weight_L{layer_id}"
                    ),
                    "k_norm_weight": DeviceTensor.from_torch(
                        k_norm_weight, f"k_norm_weight_L{layer_id}"
                    ),
                    "input_weight": DeviceTensor.from_torch(
                        input_weight, f"input_weight_L{layer_id}"
                    ),
                    "post_attention_weight": DeviceTensor.from_torch(
                        post_attention_weight,
                        f"post_attention_weight_L{layer_id}",
                    ),
                    "cache_k": DeviceTensor.from_numpy(cache_k, f"cache_k_L{layer_id}"),
                    "cache_v": DeviceTensor.from_numpy(cache_v, f"cache_v_L{layer_id}"),
                }
            )

        # Create shared tensors as separate class members using DeviceTensor.from_torch for weights and from_numpy for computed arrays
        self.norm_weight = DeviceTensor.from_torch(norm_weight, "norm_weight")
        self.lm_head_weight = DeviceTensor.from_torch(lm_head_weight, "lm_head_weight")

        print_log(f"--> Finished Preparing Tensors in {time.time() - t:.2f}s")

    def _prepare_kernels(self):
        """Lazily compile and load model kernels using real tensors"""

        t = time.time()
        print_log("Preparing kernels")

        # Create real input tensors for compilation
        x_context = DeviceTensor.from_numpy(
            np.empty(
                shape=(
                    self.config.max_batch_size,
                    self.config.context_len,
                    self.config.hidden_size,
                ),
                dtype=self.config.dtype,
            ),
            "x_context",
        )
        x_token = DeviceTensor.from_numpy(
            np.empty(
                shape=(self.config.max_batch_size, 1, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "x_token",
        )
        start_pos = DeviceTensor.from_numpy(
            np.empty(shape=(1), dtype=np.int32), "start_pos"
        )
        for layer_id in range(self.config.n_layers):
            cte_layer = DeviceKernel.compile_and_load(
                transformer_layer,
                name="cte_layer",
                x=x_context,
                start_pos=None,
                qkv_weight=self.layer_tensors[layer_id]["qkv_weight"],
                o_weight=self.layer_tensors[layer_id]["o_weight"],
                input_weight=self.layer_tensors[layer_id]["input_weight"],
                q_norm_weight=self.layer_tensors[layer_id]["q_norm_weight"],
                k_norm_weight=self.layer_tensors[layer_id]["k_norm_weight"],
                post_attention_weight=self.layer_tensors[layer_id][
                    "post_attention_weight"
                ],
                router_weight=self.layer_tensors[layer_id]["router_weight"],
                gate_up_weight=self.layer_tensors[layer_id]["gate_up_weight"],
                down_weight=self.layer_tensors[layer_id]["down_weight"],
                cache_k=self.layer_tensors[layer_id]["cache_k"],
                cache_v=self.layer_tensors[layer_id]["cache_v"],
                configs=self.config,
                build_dir=BUILD_DIR,
                additional_compiler_args=self.config.additional_compiler_args_nkipy,
            )
            self.block_wise_moe_layers.append(cte_layer)

        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling,
            name="cte_greedy_sampling",
            h=x_context,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            use_nki_rmsnorm=USE_NKI_RMSNORM,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )

        self.kernel_tkg = DeviceKernel.compile_and_load(
            transformer_layer,
            x=x_token,
            start_pos=start_pos,
            qkv_weight=self.layer_tensors[0]["qkv_weight"],
            o_weight=self.layer_tensors[0]["o_weight"],
            input_weight=self.layer_tensors[0]["input_weight"],
            q_norm_weight=self.layer_tensors[0]["q_norm_weight"],
            k_norm_weight=self.layer_tensors[0]["k_norm_weight"],
            cache_k=self.layer_tensors[0]["cache_k"],
            cache_v=self.layer_tensors[0]["cache_v"],
            post_attention_weight=self.layer_tensors[0]["post_attention_weight"],
            router_weight=self.layer_tensors[0]["router_weight"],
            gate_up_weight=self.layer_tensors[0]["gate_up_weight"],
            down_weight=self.layer_tensors[0]["down_weight"],
            configs=self.config,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )

        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling,
            name="tkg_greedy_sampling",
            h=x_token,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            use_nki_rmsnorm=USE_NKI_RMSNORM,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )

        print_log(
            f"--> Finished Kernel Compilation and Loading in {time.time() - t:.2f}s"
        )

    def generate(self, input_ids):
        """Run inference and generate tokens with tensor parallelism (collectives inside kernels)"""
        context_len = self.config.context_len

        hidden_states = DeviceTensor.from_torch(
            self.tok_embedding[input_ids], "hidden_states"
        )

        # Initial position - next_id tensor for storing generated tokens
        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")
        t_start_pos = DeviceTensor.from_numpy(
            np.array([0], dtype=np.int32), "start_pos"
        )

        # Process through all layers (context phase)
        for i in range(self.config.n_layers):
            self.block_wise_moe_layers[i](
                inputs={
                    "x": hidden_states,
                    # Layer i weights
                    "qkv_weight": self.layer_tensors[i]["qkv_weight"],
                    "o_weight": self.layer_tensors[i]["o_weight"],
                    "input_weight": self.layer_tensors[i]["input_weight"],
                    "q_norm_weight": self.layer_tensors[i]["q_norm_weight"],
                    "k_norm_weight": self.layer_tensors[i]["k_norm_weight"],
                    "cache_k.must_alias_input": self.layer_tensors[i]["cache_k"],
                    "cache_v.must_alias_input": self.layer_tensors[i]["cache_v"],
                    "post_attention_weight": self.layer_tensors[i][
                        "post_attention_weight"
                    ],
                    "router_weight": self.layer_tensors[i]["router_weight"],
                    "gate_up_weight": self.layer_tensors[i]["gate_up_weight"],
                    "down_weight": self.layer_tensors[i]["down_weight"],
                },
                outputs={
                    "output0": hidden_states,
                    "cache_k": self.layer_tensors[i]["cache_k"],
                    "cache_v": self.layer_tensors[i]["cache_v"],
                },
            )
        # Generate next token
        self.kernel_cte_greedy_sampling(
            inputs={
                "h": hidden_states,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
            },
            outputs={"output0": next_id},
        )
        next_id_torch = (
            next_id.torch().reshape(self.config.max_batch_size, 1).to(dtype=torch.int)
        )
        yield next_id_torch

        # Generation phase (token by token)
        for pos in range(context_len, context_len + self.config.max_new_tokens):
            # Update the start position for this iteration
            t_start_pos = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))

            hidden_states = DeviceTensor.from_torch(
                self.tok_embedding[next_id_torch], "h0/res1"
            )
            t_res1 = hidden_states  # Output becomes next layer's input

            for i in range(0, self.config.n_layers):
                self.kernel_tkg(
                    inputs={
                        "x": hidden_states,
                        "start_pos": t_start_pos,
                        # Layer i weights
                        "qkv_weight": self.layer_tensors[i]["qkv_weight"],
                        "o_weight": self.layer_tensors[i]["o_weight"],
                        "input_weight": self.layer_tensors[i]["input_weight"],
                        "q_norm_weight": self.layer_tensors[i]["q_norm_weight"],
                        "k_norm_weight": self.layer_tensors[i]["k_norm_weight"],
                        "cache_k.must_alias_input": self.layer_tensors[i]["cache_k"],
                        "cache_v.must_alias_input": self.layer_tensors[i]["cache_v"],
                        "post_attention_weight": self.layer_tensors[i][
                            "post_attention_weight"
                        ],
                        "router_weight": self.layer_tensors[i]["router_weight"],
                        "gate_up_weight": self.layer_tensors[i]["gate_up_weight"],
                        "down_weight": self.layer_tensors[i]["down_weight"],
                    },
                    outputs={
                        "output0": t_res1,
                        "cache_k": self.layer_tensors[i]["cache_k"],
                        "cache_v": self.layer_tensors[i]["cache_v"],
                    },
                )

            self.kernel_tkg_greedy_sampling(
                inputs={
                    "h": t_res1,
                    "norm_weight": self.norm_weight,
                    "lm_head_weight": self.lm_head_weight,
                },
                outputs={"output0": next_id},
            )

            next_id_torch = (
                next_id.torch()
                .reshape(self.config.max_batch_size, 1)
                .to(dtype=torch.int)
            )

            yield next_id_torch


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # disable warning
    os.environ["OMP_NUM_THREADS"] = "1"  # disable warning
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="/kaena/qwen3_shards_30B_A3B_TP8")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    args = parser.parse_args()
    prompt = args.prompt

    initialize_model_parallel()

    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_inputs = tokenizer(prompt, return_tensors="np")
    input_ids = model_inputs["input_ids"]
    config = get_config(model_name, input_ids.shape[1], args.max_new_tokens)

    print_log("Loading Model Weights")
    t = time.time()

    shard_path = os.path.join(args.checkpoint, f"shard_{dist.get_rank()}.safetensors")
    weights = load_file(shard_path, device="cpu")

    # Create and run the model
    model = Qwen3Model(weights, config)

    # warming
    start = time.time()
    print_log("Warming model")
    t = 0
    for id in model.generate(input_ids):
        if t == 1:
            break
        t += 1
    print_log(f"--> Finished warming the model in {time.time() - start:.2f}s")

    dist.barrier()
    # Generate tokens and measure performance
    start = time.time()
    t = 0
    if dist.get_rank() == 0:
        print(f"\n{prompt}", end="")
    for id in model.generate(input_ids):
        if t == 0:
            first_token_time = time.time()
        t += 1
        output_id = id[0].tolist()
        if output_id[-1] in [151643, 151645]:  # EOS or BOS token
            print_log("Found EOS or BOS token, stop early")
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
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted!")
        sys.exit(0)
