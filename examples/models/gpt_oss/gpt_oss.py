import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from config import Config, get_config
from kernels.sampling import greedy_sampling, greedy_sampling_with_embedding
from kernels.transformer_layer import transformer_layer
from nkipy.runtime import DeviceKernel, DeviceTensor
from safetensors.torch import load_file
from transformers import AutoTokenizer
from utils import print_log

# Absolute path: the compiler chdir's into the per-kernel build dir, so a
# relative build dir would double up and the HLO module wouldn't be found.
BUILD_DIR = os.path.abspath("./build")
USE_NKI_RMSNORM = True

# weight names carried per layer (everything except the kv cache)
_LAYER_WEIGHT_KEYS = [
    "qkv_weight",
    "qkv_bias",
    "o_weight",
    "o_bias",
    "sinks",
    "input_weight",
    "post_attention_weight",
    "router_weight",
    "router_bias",
    "gate_up_weight",
    "gate_up_bias",
    "down_weight",
    "down_bias",
]


class GptOssModel:
    def __init__(self, model_weights, config: Config):
        """Initialize the model with weights and configuration."""
        self.config = config
        self.tok_embedding = model_weights.get("tok_embedding")

        # kernels keyed by (phase, sliding) -> compiled DeviceKernel
        self.kernel_layer = {}
        self.kernel_cte_greedy_sampling = None
        self.kernel_cte_greedy_sampling_embed = None
        self.kernel_tkg_greedy_sampling = None
        self.kernel_tkg_greedy_sampling_embed = None

        self.norm_weight = None
        self.lm_head_weight = None
        self.tok_embedding_device = None

        self._prepare_tensors(model_weights)
        self._prepare_kernels()

    def _prepare_tensors(self, weights):
        t = time.time()
        print_log("Preparing Tensors")

        n_local_kv_heads = max(1, self.config.num_kv_heads // dist.get_world_size())

        cache_shape = (
            self.config.max_batch_size,
            self.config.max_seq_len,
            n_local_kv_heads,
            self.config.head_dim,
        )
        cache_k = np.zeros(cache_shape, dtype=self.config.dtype)
        cache_v = np.zeros(cache_shape, dtype=self.config.dtype)

        # Per-layer device tensors.
        self.layer_tensors = []
        for layer_id in range(self.config.num_layers):
            lt = {}
            for key in _LAYER_WEIGHT_KEYS:
                w = weights.get(f"layers.{layer_id}.{key}")
                lt[key] = DeviceTensor.from_torch(w, f"{key}_L{layer_id}")
            lt["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{layer_id}")
            lt["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{layer_id}")
            self.layer_tensors.append(lt)

        self.norm_weight = DeviceTensor.from_torch(
            weights.get("norm_weight"), "norm_weight"
        )
        self.lm_head_weight = DeviceTensor.from_torch(
            weights.get("lm_head_weight"), "lm_head_weight"
        )
        self.tok_embedding_device = DeviceTensor.from_torch(
            self.tok_embedding, "tok_embedding"
        )

        print_log(f"--> Finished Preparing Tensors in {time.time() - t:.2f}s")

    def _compile_layer(self, name, x, start_pos, sliding_window):
        lt = self.layer_tensors[0]
        return DeviceKernel.compile_and_load(
            transformer_layer,
            name=name,
            x=x,
            start_pos=start_pos,
            qkv_weight=lt["qkv_weight"],
            qkv_bias=lt["qkv_bias"],
            o_weight=lt["o_weight"],
            o_bias=lt["o_bias"],
            sinks=lt["sinks"],
            input_weight=lt["input_weight"],
            post_attention_weight=lt["post_attention_weight"],
            router_weight=lt["router_weight"],
            router_bias=lt["router_bias"],
            gate_up_weight=lt["gate_up_weight"],
            gate_up_bias=lt["gate_up_bias"],
            down_weight=lt["down_weight"],
            down_bias=lt["down_bias"],
            cache_k=lt["cache_k"],
            cache_v=lt["cache_v"],
            configs=self.config,
            sliding_window=sliding_window,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )

    def _prepare_kernels(self):
        t = time.time()
        print_log("Preparing kernels")

        x_context = DeviceTensor.from_numpy(
            np.empty(
                (
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
                (self.config.max_batch_size, 1, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "x_token",
        )
        start_pos = DeviceTensor.from_numpy(np.empty((1), dtype=np.int32), "start_pos")

        # gpt-oss alternates sliding-window and full attention per layer, so we
        # compile one transformer-layer kernel per (phase, attention-type) pair.
        for sliding in (False, True):
            sw = self.config.sliding_window if sliding else None
            self.kernel_layer[("cte", sliding)] = self._compile_layer(
                f"cte_layer_{'sw' if sliding else 'full'}", x_context, None, sw
            )
            self.kernel_layer[("tkg", sliding)] = self._compile_layer(
                f"tkg_layer_{'sw' if sliding else 'full'}", x_token, start_pos, sw
            )

        common = dict(
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            use_nki_rmsnorm=USE_NKI_RMSNORM,
            build_dir=BUILD_DIR,
            additional_compiler_args=self.config.additional_compiler_args_nkipy,
        )
        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling, name="cte_greedy_sampling", h=x_context, **common
        )
        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling, name="tkg_greedy_sampling", h=x_token, **common
        )
        self.kernel_cte_greedy_sampling_embed = DeviceKernel.compile_and_load(
            greedy_sampling_with_embedding,
            name="cte_greedy_sampling_embed",
            h=x_context,
            tok_embedding=self.tok_embedding_device,
            **common,
        )
        self.kernel_tkg_greedy_sampling_embed = DeviceKernel.compile_and_load(
            greedy_sampling_with_embedding,
            name="tkg_greedy_sampling_embed",
            h=x_token,
            tok_embedding=self.tok_embedding_device,
            **common,
        )

        print_log(
            f"--> Finished Kernel Compilation and Loading in {time.time() - t:.2f}s"
        )

    def _run_layer(self, phase, i, hidden_states, start_pos):
        lt = self.layer_tensors[i]
        kernel = self.kernel_layer[(phase, self.config.is_sliding(i))]
        inputs = {key: lt[key] for key in _LAYER_WEIGHT_KEYS}
        inputs["x"] = hidden_states
        inputs["cache_k.must_alias_input"] = lt["cache_k"]
        inputs["cache_v.must_alias_input"] = lt["cache_v"]
        if phase == "tkg":
            inputs["start_pos"] = start_pos
        kernel(
            inputs=inputs,
            outputs={
                "output0": hidden_states,
                "cache_k": lt["cache_k"],
                "cache_v": lt["cache_v"],
            },
        )

    def run_prefill(self, input_ids, capture_aux=False):
        """Run the context-encoding (prefill) layer stack.

        Args:
            input_ids: prompt token ids, shape (B, L).
            capture_aux: when True, also return the residual-stream hidden states
                produced by the EAGLE-3 tap layers (``config.aux_layers``), in
                low->mid->high order. Each is a host torch tensor of shape
                (B, L, hidden_size). Used to seed the speculative drafter.

        Returns:
            (hidden_states, aux) where hidden_states is the final-layer device
            tensor and aux is a list of captured host tensors (empty unless
            capture_aux and aux_layers are set).
        """
        hidden_states = DeviceTensor.from_torch(
            self.tok_embedding[input_ids], "hidden_states"
        )

        aux_layers = self.config.aux_layers if capture_aux else None
        aux = []
        for i in range(self.config.num_layers):
            # EAGLE-3 taps the *input* residual stream of each aux layer (i.e. the
            # output of layer i-1), matching vLLM's `hidden_states + residual`
            # captured before running layer i. Snapshot before _run_layer.
            if aux_layers is not None and i in aux_layers:
                aux.append(hidden_states.torch().clone())
            self._run_layer("cte", i, hidden_states, None)

        return hidden_states, aux

    def generate(self, input_ids, double_buffering=True):
        """Run inference and generate tokens."""
        hidden_states, _ = self.run_prefill(input_ids, capture_aux=False)

        if double_buffering:
            yield from self._generate_double_buffered(hidden_states)
        else:
            yield from self._generate_baseline(hidden_states)

    def _run_tkg_layers(self, hidden_states, start_pos):
        for i in range(self.config.num_layers):
            self._run_layer("tkg", i, hidden_states, start_pos)

    # ── Double-buffered decode (fused sampling + on-device embedding) ──────

    def _generate_double_buffered(self, hidden_states):
        context_len = self.config.context_len
        B = self.config.max_batch_size

        next_id_bufs = [
            DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id_0"),
            DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id_1"),
        ]
        decode_hidden = DeviceTensor.from_numpy(
            np.empty((B, 1, self.config.hidden_size), dtype=self.config.dtype),
            "decode_hidden",
        )

        cur_buf = 0
        self.kernel_cte_greedy_sampling_embed(
            inputs={
                "h": hidden_states,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
                "tok_embedding": self.tok_embedding_device,
            },
            outputs={"output0": next_id_bufs[cur_buf], "output1": decode_hidden},
        )

        for pos in range(context_len, context_len + self.config.max_new_tokens):
            prev_buf = cur_buf
            cur_buf = 1 - cur_buf

            t_start_pos = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))
            self._run_tkg_layers(decode_hidden, t_start_pos)

            self.kernel_tkg_greedy_sampling_embed(
                inputs={
                    "h": decode_hidden,
                    "norm_weight": self.norm_weight,
                    "lm_head_weight": self.lm_head_weight,
                    "tok_embedding": self.tok_embedding_device,
                },
                outputs={"output0": next_id_bufs[cur_buf], "output1": decode_hidden},
            )

            next_id_torch = (
                next_id_bufs[prev_buf].torch().reshape(B, 1).to(dtype=torch.int)
            )
            yield next_id_torch

        next_id_torch = next_id_bufs[cur_buf].torch().reshape(B, 1).to(dtype=torch.int)
        yield next_id_torch

    # ── Baseline decode (host embedding lookup, no double buffering) ──────

    def _generate_baseline(self, hidden_states):
        context_len = self.config.context_len
        B = self.config.max_batch_size

        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")

        self.kernel_cte_greedy_sampling(
            inputs={
                "h": hidden_states,
                "norm_weight": self.norm_weight,
                "lm_head_weight": self.lm_head_weight,
            },
            outputs={"output0": next_id},
        )
        next_id_torch = next_id.torch().reshape(B, 1).to(dtype=torch.int)
        yield next_id_torch

        t_start_pos = DeviceTensor.from_numpy(
            np.array([context_len], dtype=np.int32), "start_pos"
        )
        hidden_states = DeviceTensor.from_torch(
            self.tok_embedding[next_id_torch], "h0/res1"
        )

        for pos in range(context_len, context_len + self.config.max_new_tokens):
            t_start_pos.write_from_numpy(np.array([pos], dtype=np.int32))
            hidden_states.write_from_torch(self.tok_embedding[next_id_torch])

            self._run_tkg_layers(hidden_states, t_start_pos)

            self.kernel_tkg_greedy_sampling(
                inputs={
                    "h": hidden_states,
                    "norm_weight": self.norm_weight,
                    "lm_head_weight": self.lm_head_weight,
                },
                outputs={"output0": next_id},
            )

            next_id_torch = next_id.torch().reshape(B, 1).to(dtype=torch.int)
            yield next_id_torch


def _resolve_eos_ids(model_name, tokenizer):
    """Collect stop token ids from the generation config (falls back to tokenizer)."""
    try:
        from transformers import GenerationConfig

        gen = GenerationConfig.from_pretrained(model_name)
        eos = gen.eos_token_id
    except Exception:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def load_model(args):
    """Initialize distributed env, load weights, and build a GptOssModel."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = "localhost:61239"

    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_inputs = tokenizer(args.prompt, return_tensors="np")
    input_ids = model_inputs["input_ids"]
    config = get_config(args.model, input_ids.shape[1], args.max_new_tokens)
    args.eos_ids = _resolve_eos_ids(args.model, tokenizer)

    print_log("Loading Model Weights")
    shard_path = os.path.join(args.checkpoint, f"shard_{dist.get_rank()}.safetensors")
    weights = load_file(shard_path, device="cpu")

    double_buffering = getattr(args, "double_buffering", True)
    model = GptOssModel(weights, config)

    start = time.time()
    print_log("Warming model")
    t = 0
    for _ in model.generate(input_ids, double_buffering=double_buffering):
        if t == 1:
            break
        t += 1
    print_log(f"--> Finished warming the model in {time.time() - start:.2f}s")

    return model, input_ids, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max-new-tokens", type=int, default=16)
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--checkpoint", default="./tmp_gpt-oss-20b")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--no-double-buffering",
        action="store_true",
        help="Disable fused embedding + double-buffered decoding (for perf comparison)",
    )
    args = parser.parse_args()
    args.double_buffering = not args.no_double_buffering

    model, input_ids, tokenizer = load_model(args)

    dist.barrier()
    start = time.time()
    t = 0
    first_token_time = start
    if dist.get_rank() == 0:
        print(f"\n{args.prompt}", end="")
    eos_ids = getattr(args, "eos_ids", set())
    for id in model.generate(input_ids, double_buffering=args.double_buffering):
        if t == 0:
            first_token_time = time.time()
        t += 1
        output_id = id[0].tolist()
        if output_id[-1] in eos_ids:
            print_log("Found special/EOS token, stop early")
            break
        if dist.get_rank() == 0:
            print(tokenizer.decode(output_id), end="")
            sys.stdout.flush()

    end_time = time.time()
    ttft = first_token_time - start
    decoding_time = max(end_time - first_token_time, 1e-6)
    tokens_per_second = t / decoding_time
    if dist.get_rank() == 0:
        print(f"\nTime to first token: {ttft:.2f}s")
        print(f"Decoding tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
