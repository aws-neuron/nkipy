# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Llama3 model for Neuron (supports TinyLlama)."""

import logging
import time

import numpy as np
import torch
import torch.distributed as dist
from nkipy.runtime import DeviceKernel, DeviceTensor

from .config import Config
from .kernels.transformer_layer import transformer_layer
from .kernels.sampling import greedy_sampling

logger = logging.getLogger(__name__)
BUILD_DIR = "./build"
USE_NKI_RMSNORM = False

LAYER_WEIGHT_KEYS = [
    ("qkv_weight", "qkv_weight"),
    ("o_weight", "o_weight"),
    ("gate_up_weight", "gate_up_weight"),
    ("down_weight", "down_weight"),
    ("input_weight", "input_weight"),
    ("post_attention_weight", "post_attention_weight"),
]

_KERNEL_INPUT_KEYS = [
    ("qkv_weight", "qkv_weight"),
    ("o_weight", "o_weight"),
    ("input_weight", "input_weight"),
    ("cache_k.must_alias_input", "cache_k"),
    ("cache_v.must_alias_input", "cache_v"),
    ("post_attention_weight", "post_attention_weight"),
    ("gate_up_weight", "gate_up_weight"),
    ("down_weight", "down_weight"),
]


class Llama3Model:
    """NKIPy Llama3/TinyLlama model running on Neuron cores."""

    def __init__(self, model_weights=None, config: Config = None, skip_kernels=False):
        self.config = config
        # Keep tok_embedding as CPU tensor for compatibility (used during inference)
        self.tok_embedding = model_weights.get("tok_embedding") if model_weights else None
        # tok_embedding_device is DeviceTensor version for P2P transfer
        self.tok_embedding_device = None

        self.kernel_cte = None
        self.kernel_tkg = None
        self.kernel_cte_greedy_sampling = None
        self.kernel_tkg_greedy_sampling = None
        self.norm_weight = None
        self.lm_head_weight = None
        self.layer_tensors = []

        if model_weights:
            self._prepare_tensors(model_weights)
            if not skip_kernels and config.context_len is not None:
                self._prepare_kernels()

    def _prepare_tensors(self, weights):
        t = time.time()
        logger.info("Preparing tensors")
        cfg = self.config
        n_local_kv_heads = max(1, cfg.num_kv_heads // dist.get_world_size())
        cache_shape = (cfg.max_batch_size, cfg.max_seq_len, n_local_kv_heads, cfg.head_dim)
        cache_k = np.zeros(cache_shape, dtype=cfg.dtype)
        cache_v = np.zeros(cache_shape, dtype=cfg.dtype)

        self.layer_tensors = []
        for lid in range(cfg.num_layers):
            layer = {}
            for wk, prefix in LAYER_WEIGHT_KEYS:
                layer[wk] = DeviceTensor.from_torch(weights[f"layers.{lid}.{wk}"], f"{prefix}_L{lid}")
            layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{lid}")
            layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{lid}")
            self.layer_tensors.append(layer)

        self.norm_weight = DeviceTensor.from_torch(weights["norm_weight"], "norm_weight")
        self.lm_head_weight = DeviceTensor.from_torch(weights["lm_head_weight"], "lm_head_weight")

        # Convert tok_embedding to DeviceTensor for P2P transfer
        if self.tok_embedding is not None:
            self.tok_embedding_device = DeviceTensor.from_torch(self.tok_embedding, "tok_embedding")

        logger.info("Tensors prepared in %.2fs", time.time() - t)

    def _load_weights_into_existing_tensors(self, weights):
        """Load weights from checkpoint into existing DeviceTensors (for wake_up from checkpoint).

        This populates already-allocated tensors without recreating them.
        Used after _allocate_empty_tensors() has created the tensor structure.
        """
        t = time.time()
        logger.info("Loading weights into existing tensors")

        # Load layer weights
        for lid in range(self.config.num_layers):
            layer = self.layer_tensors[lid]
            for wk, _ in LAYER_WEIGHT_KEYS:
                weight_key = f"layers.{lid}.{wk}"
                if weight_key in weights:
                    layer[wk].write_from_torch(weights[weight_key])

        # Load model head weights
        if "norm_weight" in weights:
            self.norm_weight.write_from_torch(weights["norm_weight"])
        if "lm_head_weight" in weights:
            self.lm_head_weight.write_from_torch(weights["lm_head_weight"])

        # Load tok_embedding
        if "tok_embedding" in weights:
            self.tok_embedding_device.write_from_torch(weights["tok_embedding"])
            # Also set the CPU version
            self.tok_embedding = weights["tok_embedding"]

        logger.info("Weights loaded into existing tensors in %.2fs", time.time() - t)

    def _allocate_empty_tensors(self):
        """Allocate device tensors WITHOUT initialization (for P2P receive).

        Key optimization: Allocate device memory without writing zeros.
        P2P RDMA will directly write weights to uninitialized memory,
        avoiding double-write that causes memory fragmentation and slow spike_reset().
        """
        cfg = self.config
        tp = dist.get_world_size()
        n_local_kv_heads = max(1, cfg.num_kv_heads // tp)
        n_local_heads = cfg.num_heads // tp
        cache_shape = (cfg.max_batch_size, cfg.max_seq_len, n_local_kv_heads, cfg.head_dim)

        # Cache tensors still need zeros (they're not filled by P2P)
        cache_k = np.zeros(cache_shape, dtype=cfg.dtype)
        cache_v = np.zeros(cache_shape, dtype=cfg.dtype)

        qkv_dim = (n_local_heads + 2 * n_local_kv_heads) * cfg.head_dim
        shapes = {
            "qkv_weight": (cfg.hidden_size, qkv_dim),
            "o_weight": (n_local_heads * cfg.head_dim, cfg.hidden_size),
            "gate_up_weight": (cfg.hidden_size, 2 * cfg.intermediate_size),
            "down_weight": (cfg.intermediate_size, cfg.hidden_size),
            "input_weight": (cfg.hidden_size,),
            "post_attention_weight": (cfg.hidden_size,),
        }

        # OPTIMIZATION: Allocate weight tensors first (uninitialized),
        # then cache tensors (zeros) to avoid memory fragmentation.
        # Interleaving uninitialized + zero-writes causes NRT cleanup issues.
        self.layer_tensors = []
        for lid in range(cfg.num_layers):
            layer = {}
            for wk, prefix in LAYER_WEIGHT_KEYS:
                # Use allocate_uninitialized instead of from_numpy(zeros)
                # P2P RDMA will fill these directly
                layer[wk] = DeviceTensor.allocate_uninitialized(
                    shapes[wk], cfg.dtype, f"{prefix}_L{lid}")
            # Cache tensors will be added in second pass
            self.layer_tensors.append(layer)

        # Second pass: allocate cache tensors (need zeros, not P2P-filled)
        for lid in range(cfg.num_layers):
            layer = self.layer_tensors[lid]
            layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{lid}")
            layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{lid}")

        # Model head weights also uninitialized (filled by P2P)
        self.norm_weight = DeviceTensor.allocate_uninitialized(
            (cfg.hidden_size,), cfg.dtype, "norm_weight")
        self.lm_head_weight = DeviceTensor.allocate_uninitialized(
            (cfg.hidden_size, cfg.vocab_size // tp), cfg.dtype, "lm_head_weight")

        # Token embedding (uninitialized, filled by P2P) - NOT sharded across TP
        self.tok_embedding_device = DeviceTensor.allocate_uninitialized(
            (cfg.vocab_size, cfg.hidden_size), cfg.dtype, "tok_embedding")

    def weight_buffers(self):
        """Yield (name, va, size_bytes) for all weight tensors (for P2P)."""
        for lid, layer in enumerate(self.layer_tensors):
            for key, dt in layer.items():
                if key in ("cache_k", "cache_v"):
                    continue
                va = dt.tensor_ref.va
                size = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
                yield f"layers.{lid}.{key}", va, size
        for attr in ("norm_weight", "lm_head_weight", "tok_embedding_device"):
            dt = getattr(self, attr, None)
            if dt is not None:
                va = dt.tensor_ref.va
                size = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
                # Use "tok_embedding" as the name for compatibility
                name = "tok_embedding" if attr == "tok_embedding_device" else attr
                yield name, va, size

    def _prepare_kernels(self):
        t = time.time()
        logger.info("Compiling kernels")
        cfg = self.config
        compiler_args = cfg.additional_compiler_args_nkipy
        L0 = self.layer_tensors[0]

        x_ctx = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, cfg.context_len, cfg.hidden_size), dtype=cfg.dtype), "x_ctx")
        x_tok = DeviceTensor.from_numpy(
            np.empty((cfg.max_batch_size, 1, cfg.hidden_size), dtype=cfg.dtype), "x_tok")
        sp = DeviceTensor.from_numpy(np.empty((1,), dtype=np.int32), "start_pos")

        layer_args = {wk: L0[wk] for wk, _ in LAYER_WEIGHT_KEYS}
        layer_args["cache_k"] = L0["cache_k"]
        layer_args["cache_v"] = L0["cache_v"]

        self.kernel_cte = DeviceKernel.compile_and_load(
            transformer_layer, name="cte_layer", x=x_ctx, start_pos=None,
            **layer_args, configs=cfg, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        self.kernel_tkg = DeviceKernel.compile_and_load(
            transformer_layer, name="tkg_layer", x=x_tok, start_pos=sp,
            **layer_args, configs=cfg, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        self.kernel_cte_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling, name="cte_greedy_sampling", h=x_ctx,
            norm_weight=self.norm_weight, lm_head_weight=self.lm_head_weight,
            configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        self.kernel_tkg_greedy_sampling = DeviceKernel.compile_and_load(
            greedy_sampling, name="tkg_greedy_sampling", h=x_tok,
            norm_weight=self.norm_weight, lm_head_weight=self.lm_head_weight,
            configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM, build_dir=BUILD_DIR,
            additional_compiler_args=compiler_args)

        logger.info("Kernels compiled in %.2fs", time.time() - t)

    def _compile_kernels(self):
        """Compile kernels to NEFF without loading onto device. Returns kernel_cache dict."""
        cfg = self.config
        tp = dist.get_world_size()
        compiler_args = cfg.additional_compiler_args_nkipy

        n_local_kv_heads = max(1, cfg.num_kv_heads // tp)
        n_local_heads = cfg.num_heads // tp
        qkv_dim = (n_local_heads + 2 * n_local_kv_heads) * cfg.head_dim
        cache_shape = (cfg.max_batch_size, cfg.max_seq_len, n_local_kv_heads, cfg.head_dim)

        shapes = {
            "qkv_weight": (cfg.hidden_size, qkv_dim),
            "o_weight": (n_local_heads * cfg.head_dim, cfg.hidden_size),
            "gate_up_weight": (cfg.hidden_size, 2 * cfg.intermediate_size),
            "down_weight": (cfg.intermediate_size, cfg.hidden_size),
            "input_weight": (cfg.hidden_size,),
            "post_attention_weight": (cfg.hidden_size,),
        }

        x_ctx = np.empty((cfg.max_batch_size, cfg.context_len, cfg.hidden_size), dtype=cfg.dtype)
        x_tok = np.empty((cfg.max_batch_size, 1, cfg.hidden_size), dtype=cfg.dtype)
        sp = np.empty((1,), dtype=np.int32)

        layer_args = {wk: np.empty(shapes[wk], dtype=cfg.dtype) for wk, _ in LAYER_WEIGHT_KEYS}
        layer_args["cache_k"] = np.zeros(cache_shape, dtype=cfg.dtype)
        layer_args["cache_v"] = np.zeros(cache_shape, dtype=cfg.dtype)

        norm_w = np.empty((cfg.hidden_size,), dtype=cfg.dtype)
        lm_head_w = np.empty((cfg.hidden_size, cfg.vocab_size // tp), dtype=cfg.dtype)

        specs = [
            ("kernel_cte", "cte_layer", transformer_layer,
             dict(x=x_ctx, start_pos=None, **layer_args, configs=cfg)),
            ("kernel_tkg", "tkg_layer", transformer_layer,
             dict(x=x_tok, start_pos=sp, **layer_args, configs=cfg)),
            ("kernel_cte_greedy_sampling", "cte_greedy_sampling", greedy_sampling,
             dict(h=x_ctx, norm_weight=norm_w, lm_head_weight=lm_head_w,
                  configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM)),
            ("kernel_tkg_greedy_sampling", "tkg_greedy_sampling", greedy_sampling,
             dict(h=x_tok, norm_weight=norm_w, lm_head_weight=lm_head_w,
                  configs=cfg, use_nki_rmsnorm=USE_NKI_RMSNORM)),
        ]

        kernel_cache = {}
        for attr, name, kernel_fn, kwargs in specs:
            neff_path, cache_key = DeviceKernel.compile_only(
                kernel_fn, name=name, build_dir=BUILD_DIR,
                additional_compiler_args=compiler_args, **kwargs)
            kernel_cache[attr] = (neff_path, cache_key)
        return kernel_cache

    def _build_inputs(self, lid, hidden, start_pos=None):
        inputs = {"x": hidden}
        if start_pos is not None:
            inputs["start_pos"] = start_pos
        for kk, lk in _KERNEL_INPUT_KEYS:
            inputs[kk] = self.layer_tensors[lid][lk]
        return inputs

    def _build_outputs(self, lid, hidden):
        return {
            "output0": hidden,
            "cache_k": self.layer_tensors[lid]["cache_k"],
            "cache_v": self.layer_tensors[lid]["cache_v"],
        }

    def generate(self, input_ids):
        """Yield one token tensor per step (prefill + decode)."""
        cfg = self.config
        hidden = DeviceTensor.from_torch(self.tok_embedding[input_ids], "hidden_states")
        next_id = DeviceTensor.from_numpy(np.array([[0]], dtype=np.uint32), "next_id")

        # Prefill
        for i in range(cfg.num_layers):
            self.kernel_cte(inputs=self._build_inputs(i, hidden),
                            outputs=self._build_outputs(i, hidden))

        self.kernel_cte_greedy_sampling(
            inputs={"h": hidden, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
            outputs={"output0": next_id})
        next_id_torch = next_id.torch().reshape(cfg.max_batch_size, 1).to(dtype=torch.int)
        yield next_id_torch

        # Decode
        for pos in range(cfg.context_len, cfg.context_len + cfg.max_new_tokens):
            t_sp = DeviceTensor.from_numpy(np.array([pos], dtype=np.int32))
            hidden = DeviceTensor.from_torch(self.tok_embedding[next_id_torch], "h0/res1")

            for i in range(cfg.num_layers):
                self.kernel_tkg(inputs=self._build_inputs(i, hidden, t_sp),
                                outputs=self._build_outputs(i, hidden))

            self.kernel_tkg_greedy_sampling(
                inputs={"h": hidden, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
                outputs={"output0": next_id})
            next_id_torch = next_id.torch().reshape(cfg.max_batch_size, 1).to(dtype=torch.int)
            yield next_id_torch
