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

    def __init__(self, model_weights=None, config: Config = None, skip_kernels=False):
        self.config = config
        self.tok_embedding = model_weights.get("tok_embedding") if model_weights else None

        self.kernel_cte = None
        self.kernel_cte_greedy_sampling = None
        self.kernel_tkg = None
        self.kernel_tkg_greedy_sampling = None

        self.norm_weight = None
        self.lm_head_weight = None
        self.layer_tensors = []

        if model_weights:
            self._prepare_tensors(model_weights)
        if not skip_kernels and model_weights:
            self._prepare_kernels()

    # --- Tensor allocation ---

    def _init_kv_caches(self):
        """Return zeroed KV cache numpy arrays based on config."""
        cfg = self.config
        n_local_kv_heads = max(1, cfg.num_kv_heads // dist.get_world_size())
        shape = (cfg.max_batch_size, cfg.max_seq_len, n_local_kv_heads, cfg.head_dim)
        return np.zeros(shape, dtype=cfg.dtype), np.zeros(shape, dtype=cfg.dtype)

    def _prepare_tensors(self, weights):
        """Load weight tensors from a CPU weights dict onto device."""
        t = time.time()
        print_log("Preparing Tensors")

        cache_k, cache_v = self._init_kv_caches()

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

    def weight_buffers(self):
        """Yield ``(name, va, size_bytes)`` for each weight tensor on device.

        Used by :mod:`nkipy.p2p` for RDMA registration.  KV caches are
        excluded because they are not transferred between engines.
        """
        for layer_idx, layer in enumerate(self.layer_tensors):
            for key, dt in layer.items():
                if key in ("cache_k", "cache_v"):
                    continue
                va = dt.tensor_ref.va
                size = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
                yield f"layers.{layer_idx}.{key}", va, size
        for attr in ("norm_weight", "lm_head_weight"):
            dt = getattr(self, attr, None)
            if dt is not None:
                va = dt.tensor_ref.va
                size = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
                yield attr, va, size

    def _weight_shapes(self):
        """Return dict of {weight_key: shape} for per-layer weights.

        Shapes are inferred from config and TP degree. Override in subclass
        if the architecture has non-standard weight layouts.
        """
        cfg = self.config
        tp = dist.get_world_size()
        n_local_kv_heads = max(1, cfg.num_kv_heads // tp)
        q_dim = (cfg.num_heads // tp) * cfg.head_dim
        kv_dim = n_local_kv_heads * cfg.head_dim

        shapes = {
            "qkv_weight": (cfg.hidden_size, q_dim + 2 * kv_dim),
            "o_weight": (q_dim, cfg.hidden_size),
            "input_weight": (cfg.hidden_size,),
            "post_attention_weight": (cfg.hidden_size,),
            "q_norm_weight": (cfg.head_dim,),
            "k_norm_weight": (cfg.head_dim,),
        }

        if cfg.num_experts is not None:
            shapes["gate_up_weight"] = (cfg.num_experts, cfg.hidden_size, 2 * cfg.intermediate_size)
            shapes["down_weight"] = (cfg.num_experts, cfg.intermediate_size, cfg.hidden_size)
            shapes["router_weight"] = (cfg.hidden_size, cfg.num_experts)
        else:
            shapes["gate_up_weight"] = (cfg.hidden_size, 2 * cfg.intermediate_size)
            shapes["down_weight"] = (cfg.intermediate_size, cfg.hidden_size)

        return shapes

    def _allocate_empty_tensors(self):
        """Allocate zero-filled device tensors for P2P weight transfer.

        Tensor shapes are inferred from config and TP degree.
        """
        t = time.time()
        print_log("Allocating empty tensors for P2P transfer")

        cfg = self.config
        cache_k, cache_v = self._init_kv_caches()
        shapes = self._weight_shapes()

        self.layer_tensors = []
        for layer_id in range(cfg.num_layers):
            layer = {}
            for weight_key, prefix in self.LAYER_WEIGHT_KEYS:
                buf = np.zeros(shapes[weight_key], dtype=cfg.dtype)
                layer[weight_key] = DeviceTensor.from_numpy(buf, f"{prefix}_L{layer_id}")
            layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{layer_id}")
            layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{layer_id}")
            self.layer_tensors.append(layer)

        tp = dist.get_world_size()
        self.norm_weight = DeviceTensor.from_numpy(
            np.zeros((cfg.hidden_size,), dtype=cfg.dtype), "norm_weight"
        )
        self.lm_head_weight = DeviceTensor.from_numpy(
            np.zeros((cfg.hidden_size, cfg.vocab_size // tp), dtype=cfg.dtype), "lm_head_weight"
        )

        print_log(f"--> Finished Allocating Empty Tensors in {time.time() - t:.2f}s")

    # --- Kernel compilation ---

    def _kernel_layer_args(self):
        """Return dict of layer-0 tensor args for kernel compilation. Override in subclass."""
        raise NotImplementedError

    def _kernel_input_keys(self):
        """Return list of (kernel_input_name, layer_tensor_key) pairs for generate().
        Override in subclass."""
        raise NotImplementedError

    def _kernel_specs(self, x_context, x_token, start_pos, layer_args, norm_weight, lm_head_weight):
        """Return list of (attr_name, compile_name, kernel_fn, kwargs) for all kernels."""
        cfg = self.config
        return [
            ("kernel_cte", "cte_layer", self.transformer_layer,
             dict(x=x_context, start_pos=None, **layer_args, configs=cfg)),
            ("kernel_tkg", "tkg_layer", self.transformer_layer,
             dict(x=x_token, start_pos=start_pos, **layer_args, configs=cfg)),
            ("kernel_cte_greedy_sampling", "cte_greedy_sampling", self.greedy_sampling,
             dict(h=x_context, norm_weight=norm_weight, lm_head_weight=lm_head_weight,
                  configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM)),
            ("kernel_tkg_greedy_sampling", "tkg_greedy_sampling", self.greedy_sampling,
             dict(h=x_token, norm_weight=norm_weight, lm_head_weight=lm_head_weight,
                  configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM)),
        ]

    def _prepare_kernels(self):
        t = time.time()
        print_log("Preparing kernels")

        cfg = self.config
        x_context = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, cfg.context_len, cfg.hidden_size), dtype=cfg.dtype), "x_context")
        x_token = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, 1, cfg.hidden_size), dtype=cfg.dtype), "x_token")
        start_pos = DeviceTensor.from_numpy(np.empty((1,), dtype=np.int32), "start_pos")

        compiler_args = cfg.additional_compiler_args_nkipy
        for attr, name, kernel_fn, kwargs in self._kernel_specs(
            x_context, x_token, start_pos, self._kernel_layer_args(),
            self.norm_weight, self.lm_head_weight,
        ):
            setattr(self, attr, DeviceKernel.compile_and_load(
                kernel_fn, name=name, build_dir=BUILD_DIR,
                additional_compiler_args=compiler_args, **kwargs,
            ))

        print_log(f"--> Finished Kernel Compilation and Loading in {time.time() - t:.2f}s")

    def _compile_kernels(self):
        """Compile kernels to NEFF without loading onto device (no neuron cores needed).

        Returns a kernel_cache dict: {name: (neff_path, cache_key)} compatible
        with the format used by server.py sleep/wake_up.
        """
        t = time.time()
        print_log("Compiling kernels (compile-only, no device)")

        cfg = self.config
        tp = dist.get_world_size()

        x_context = np.empty((cfg.max_batch_size, cfg.context_len, cfg.hidden_size), dtype=cfg.dtype)
        x_token = np.empty((cfg.max_batch_size, 1, cfg.hidden_size), dtype=cfg.dtype)
        start_pos = np.empty((1,), dtype=np.int32)

        shapes = self._weight_shapes()
        cache_k, cache_v = self._init_kv_caches()
        layer_args = {k: np.empty(shapes[k], dtype=cfg.dtype) for k, _ in self.LAYER_WEIGHT_KEYS}
        layer_args["cache_k"] = cache_k
        layer_args["cache_v"] = cache_v

        norm_weight = np.empty((cfg.hidden_size,), dtype=cfg.dtype)
        lm_head_weight = np.empty((cfg.hidden_size, cfg.vocab_size // tp), dtype=cfg.dtype)

        compiler_args = cfg.additional_compiler_args_nkipy
        kernel_cache = {}
        for attr, name, kernel_fn, kwargs in self._kernel_specs(
            x_context, x_token, start_pos, layer_args, norm_weight, lm_head_weight,
        ):
            neff_path, cache_key = DeviceKernel.compile_only(
                kernel_fn, name=name, build_dir=BUILD_DIR,
                additional_compiler_args=compiler_args, **kwargs,
            )
            kernel_cache[attr] = (neff_path, cache_key)

        print_log(f"--> Finished Kernel Compilation (no device) in {time.time() - t:.2f}s")
        return kernel_cache

    # --- Inference ---

    def _build_kernel_inputs(self, layer_idx, hidden_states, start_pos=None):
        inputs = {"x": hidden_states}
        if start_pos is not None:
            inputs["start_pos"] = start_pos
        for kernel_key, layer_key in self._kernel_input_keys():
            inputs[kernel_key] = self.layer_tensors[layer_idx][layer_key]
        return inputs

    def _build_kernel_outputs(self, layer_idx, hidden_states):
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
            inputs={"h": hidden_states, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
            outputs={"output0": next_id},
        )
        next_id_torch = next_id.torch().reshape(self.config.max_batch_size, 1).to(dtype=torch.int)
        yield next_id_torch

        # Token generation phase
        for pos in range(context_len, context_len + self.config.max_new_tokens):
            t_start_pos = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))
            hidden_states = DeviceTensor.from_torch(self.tok_embedding[next_id_torch], "h0/res1")
            t_res1 = hidden_states

            for i in range(self.config.num_layers):
                self.kernel_tkg(
                    inputs=self._build_kernel_inputs(i, hidden_states, t_start_pos),
                    outputs=self._build_kernel_outputs(i, t_res1),
                )

            self.kernel_tkg_greedy_sampling(
                inputs={"h": t_res1, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
                outputs={"output0": next_id},
            )
            next_id_torch = next_id.torch().reshape(self.config.max_batch_size, 1).to(dtype=torch.int)
            yield next_id_torch


# --- Model loading ---

def _make_warmup_ids(tokenizer, context_len, input_ids=None):
    """Create padded input IDs for warmup."""
    if input_ids is not None:
        return input_ids
    dummy = tokenizer("Hello", return_tensors="np")["input_ids"]
    seq_len = dummy.shape[1]
    if seq_len < context_len:
        pad = np.full((dummy.shape[0], context_len - seq_len), tokenizer.pad_token_id or 0, dtype=dummy.dtype)
        dummy = np.concatenate([pad, dummy], axis=1)
    return dummy


def _warmup(model, warmup_ids):
    """Run two generation steps to warm up compiled kernels."""
    t0 = time.time()
    for i, _ in enumerate(model.generate(warmup_ids)):
        if i == 1:
            break
    print_log(f"--> Warmup done in {time.time() - t0:.2f}s")


def load_model(model_class, args):
    """Load weights, build model, and warmup.

    Expects dist to be already initialized. args must have:
        .model, .checkpoint (or None for P2P), .prompt (or .context_len),
        .max_new_tokens (or .max_tokens)

    When checkpoint is None, returns a pre-initialized model with empty device
    tensors and compiled kernels (ready for P2P weight transfer on wake_up).
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    context_len = getattr(args, "context_len", None)
    max_new_tokens = getattr(args, "max_new_tokens", None) or getattr(args, "max_tokens", None)
    if context_len is None:
        model_inputs = tokenizer(args.prompt, return_tensors="np")
        input_ids = model_inputs["input_ids"]
        context_len = input_ids.shape[1]
    else:
        input_ids = None

    config = get_config(args.model, context_len, max_new_tokens)
    checkpoint = getattr(args, "checkpoint", None)

    if checkpoint is not None:
        shard_path = os.path.join(checkpoint, f"shard_{dist.get_rank()}.safetensors")
        print_log("Loading model weights")
        t0 = time.time()
        weights = load_file(shard_path, device="cpu")
        print_log(f"--> load_file completed in {time.time() - t0:.2f}s")

        model = model_class(weights, config)

        print_log("Warming model")
        _warmup(model, _make_warmup_ids(tokenizer, context_len, input_ids))
    else:
        print_log("No checkpoint, compiling kernels for P2P transfer (no device needed)")
        weights = None
        model = model_class(config=config, skip_kernels=True)
        model.kernel_cache = model._compile_kernels()

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
