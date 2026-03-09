import os
import time

import numpy as np
import torch
import torch.distributed as dist
from nkipy.runtime import DeviceKernel, DeviceTensor
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .config import Config, get_config
from .utils import print_log

BUILD_DIR = "./build"
USE_NKI_RMSNORM = True


class BaseModel:
    """Base model class shared by Qwen3 and Llama3.

    Subclasses must define:
        - LAYER_WEIGHT_KEYS: list of (weight_key, device_tensor_prefix) for per-layer weights
        - transformer_layer: the kernel function for transformer layers
        - greedy_sampling: the kernel function for sampling
        - _kernel_layer_args(): returns dict of layer tensor args for kernel compilation
    """

    LAYER_WEIGHT_KEYS = []  # Override in subclass

    def __init__(self, model_weights, config: Config, skip_kernels=False):
        self.config = config
        self.tok_embedding = model_weights.get("tok_embedding")

        self.kernel_cte = None
        self.kernel_cte_greedy_sampling = None
        self.kernel_tkg = None
        self.kernel_tkg_greedy_sampling = None

        self.norm_weight = None
        self.lm_head_weight = None

        self._prepare_tensors(model_weights)
        if not skip_kernels:
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

        self.layer_tensors = []
        for layer_id in range(self.config.num_layers):
            layer = {}
            for weight_key, prefix in self.LAYER_WEIGHT_KEYS:
                w = weights.get(f"layers.{layer_id}.{weight_key}")
                layer[weight_key] = DeviceTensor.from_torch(w, f"{prefix}_L{layer_id}")
            layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{layer_id}")
            layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{layer_id}")
            self.layer_tensors.append(layer)

        self.norm_weight = DeviceTensor.from_torch(weights.get("norm_weight"), "norm_weight")
        self.lm_head_weight = DeviceTensor.from_torch(weights.get("lm_head_weight"), "lm_head_weight")

        print_log(f"--> Finished Preparing Tensors in {time.time() - t:.2f}s")

    def _kernel_layer_args(self):
        """Return dict of layer-0 tensor args for kernel compilation. Override in subclass."""
        raise NotImplementedError

    def _kernel_input_keys(self):
        """Return list of (kernel_input_name, layer_tensor_key) pairs for generate().
        Keys ending with '.must_alias_input' are cache aliases.
        Override in subclass."""
        raise NotImplementedError

    def _prepare_kernels(self):
        t = time.time()
        print_log("Preparing kernels")

        x_context = DeviceTensor.from_numpy(
            np.empty(
                shape=(self.config.max_batch_size, self.config.context_len, self.config.hidden_size),
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

        layer_args = self._kernel_layer_args()
        compiler_args = self.config.additional_compiler_args_nkipy

        self.kernel_cte = DeviceKernel.compile_and_load(
            self.transformer_layer,
            name="cte_layer",
            x=x_context,
            start_pos=None,
            **layer_args,
            configs=self.config,
            build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args,
        )

        self.kernel_tkg = DeviceKernel.compile_and_load(
            self.transformer_layer,
            name="tkg_layer",
            x=x_token,
            start_pos=start_pos,
            **layer_args,
            configs=self.config,
            build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args,
        )

        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            self.greedy_sampling,
            name="cte_greedy_sampling",
            h=x_context,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            use_nki_rmsnorm=USE_NKI_RMSNORM,
            build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args,
        )

        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            self.greedy_sampling,
            name="tkg_greedy_sampling",
            h=x_token,
            norm_weight=self.norm_weight,
            lm_head_weight=self.lm_head_weight,
            configs=self.config,
            use_nki_rmsnorm=USE_NKI_RMSNORM,
            build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args,
        )

        print_log(f"--> Finished Kernel Compilation and Loading in {time.time() - t:.2f}s")

    def _build_kernel_inputs(self, layer_idx, hidden_states, start_pos=None):
        """Build the inputs dict for a kernel call on the given layer."""
        inputs = {"x": hidden_states}
        if start_pos is not None:
            inputs["start_pos"] = start_pos
        for kernel_key, layer_key in self._kernel_input_keys():
            inputs[kernel_key] = self.layer_tensors[layer_idx][layer_key]
        return inputs

    def _build_kernel_outputs(self, layer_idx, hidden_states):
        """Build the outputs dict for a kernel call on the given layer."""
        return {
            "output0": hidden_states,
            "cache_k": self.layer_tensors[layer_idx]["cache_k"],
            "cache_v": self.layer_tensors[layer_idx]["cache_v"],
        }

    def generate(self, input_ids):
        context_len = self.config.context_len

        hidden_states = DeviceTensor.from_torch(
            self.tok_embedding[input_ids], "hidden_states"
        )

        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")

        # Context phase
        for i in range(self.config.num_layers):
            self.kernel_cte(
                inputs=self._build_kernel_inputs(i, hidden_states),
                outputs=self._build_kernel_outputs(i, hidden_states),
            )

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

        # Token generation phase
        for pos in range(context_len, context_len + self.config.max_new_tokens):
            t_start_pos = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))

            hidden_states = DeviceTensor.from_torch(
                self.tok_embedding[next_id_torch], "h0/res1"
            )
            t_res1 = hidden_states

            for i in range(self.config.num_layers):
                self.kernel_tkg(
                    inputs=self._build_kernel_inputs(i, hidden_states, t_start_pos),
                    outputs=self._build_kernel_outputs(i, t_res1),
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


def load_model(model_class, args):
    """Load weights, build model, and warmup.

    Expects dist to be already initialized. args must have:
        .model, .checkpoint, .prompt (or .context_len), .max_new_tokens (or .max_tokens)
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Support both standalone (prompt-based context_len) and server (explicit context_len)
    context_len = getattr(args, "context_len", None)
    max_new_tokens = getattr(args, "max_new_tokens", None) or getattr(args, "max_tokens", None)
    if context_len is None:
        model_inputs = tokenizer(args.prompt, return_tensors="np")
        input_ids = model_inputs["input_ids"]
        context_len = input_ids.shape[1]
    else:
        input_ids = None

    config = get_config(args.model, context_len, max_new_tokens)

    shard_path = os.path.join(args.checkpoint, f"shard_{dist.get_rank()}.safetensors")
    print_log("Loading model weights")
    t0 = time.time()
    weights = load_file(shard_path, device="cpu")
    print_log(f"--> load_file completed in {time.time() - t0:.2f}s")

    model = model_class(weights, config)

    # Warmup
    print_log("Warming model")
    if input_ids is None:
        dummy = tokenizer("Hello", return_tensors="np")
        warmup_ids = dummy["input_ids"]
        # Pad to context_len
        seq_len = warmup_ids.shape[1]
        if seq_len < context_len:
            pad = np.full((warmup_ids.shape[0], context_len - seq_len), tokenizer.pad_token_id or 0, dtype=warmup_ids.dtype)
            warmup_ids = np.concatenate([pad, warmup_ids], axis=1)
    else:
        warmup_ids = input_ids

    t0 = time.time()
    for i, _ in enumerate(model.generate(warmup_ids)):
        if i == 1:
            break
    print_log(f"--> Warmup done in {time.time() - t0:.2f}s")

    return model, tokenizer, weights, config, input_ids


def generate_and_print(model, prompt, input_ids, tokenizer, eos_token_ids):
    """Run generation, stream tokens to stdout, and print perf stats."""
    import sys as _sys

    start = time.time()
    t = 0
    if dist.get_rank() == 0:
        print(f"\n{prompt}", end="")
    for id in model.generate(input_ids):
        if t == 0:
            first_token_time = time.time()
        t += 1
        output_id = id[0].tolist()
        if output_id[-1] in eos_token_ids:
            print_log("Found EOS token, stop early")
            break
        if dist.get_rank() == 0:
            print(tokenizer.decode(output_id), end="")
            _sys.stdout.flush()

    end_time = time.time()
    ttft = first_token_time - start
    decoding_time = end_time - first_token_time
    tokens_per_second = t / decoding_time
    if dist.get_rank() == 0:
        print(f"\nTime to first token: {ttft:.2f}s")
        print(f"Decoding tokens per second: {tokens_per_second:.2f}")
