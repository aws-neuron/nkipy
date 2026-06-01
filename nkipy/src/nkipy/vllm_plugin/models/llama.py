# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Llama3 model for Neuron (supports TinyLlama)."""

import logging
import os
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


def _tp_embedding_lookup(shard, input_ids):
    """Hidden-dim-parallel embedding lookup with all-gather across TP ranks.

    Each rank holds tok_embedding[:, hidden_start:hidden_end].
    Local lookup produces a partial hidden vector; all_gather reconstructs
    the full hidden_size dimension.
    """
    partial = shard[input_ids.long()]                   # [*, hidden_per_rank]
    world = dist.get_world_size()
    if world == 1:
        return partial
    chunks = [torch.empty_like(partial) for _ in range(world)]
    dist.all_gather(chunks, partial.contiguous())
    return torch.cat(chunks, dim=-1)                    # [*, hidden_size]


class Llama3Model:
    """NKIPy Llama3/TinyLlama model running on Neuron cores."""

    def __init__(self, model_weights=None, config: Config = None, skip_kernels=False):
        self.config = config
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

    def _shard_tok_embedding(self, embedding):
        """Shard tok_embedding along hidden dimension for this TP rank.

        If the embedding is already sharded (shape[1] == hidden_size // tp),
        returns it as-is. Otherwise slices from the full embedding.
        """
        tp = dist.get_world_size()
        rank = dist.get_rank()
        expected_cols = self.config.hidden_size // tp
        if embedding.shape[1] == expected_cols:
            return embedding.contiguous()
        cols_per_rank = embedding.shape[1] // tp
        start = rank * cols_per_rank
        end = start + cols_per_rank
        return embedding[:, start:end].contiguous()

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

        if self.tok_embedding is not None:
            self.tok_embedding = self._shard_tok_embedding(self.tok_embedding)
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

        # Load tok_embedding (shard from full embedding)
        if "tok_embedding" in weights:
            shard = self._shard_tok_embedding(weights["tok_embedding"])
            self.tok_embedding_device.write_from_torch(shard)
            self.tok_embedding = shard

        logger.info("Weights loaded into existing tensors in %.2fs", time.time() - t)

    def _allocate_empty_tensors(self):
        """Allocate device tensors from a contiguous arena (for P2P receive).

        All weight tensors are sliced from a single large device allocation.
        This enables single-MR RDMA registration (1 ibv_reg_mr instead of 483),
        reducing receiver MR registration from ~3.2s to ~0.01s.
        """
        from spike import get_spike_singleton

        cfg = self.config
        tp = dist.get_world_size()
        n_local_kv_heads = max(1, cfg.num_kv_heads // tp)
        n_local_heads = cfg.num_heads // tp
        cache_shape = (cfg.max_batch_size, cfg.max_seq_len, n_local_kv_heads, cfg.head_dim)

        cache_k = np.zeros(cache_shape, dtype=cfg.dtype)
        cache_v = np.zeros(cache_shape, dtype=cfg.dtype)

        qkv_dim = (n_local_heads + 2 * n_local_kv_heads) * cfg.head_dim
        itemsize = np.dtype(cfg.dtype).itemsize
        shapes = {
            "qkv_weight": (cfg.hidden_size, qkv_dim),
            "o_weight": (n_local_heads * cfg.head_dim, cfg.hidden_size),
            "gate_up_weight": (cfg.hidden_size, 2 * cfg.intermediate_size),
            "down_weight": (cfg.intermediate_size, cfg.hidden_size),
            "input_weight": (cfg.hidden_size,),
            "post_attention_weight": (cfg.hidden_size,),
        }

        # Compute per-tensor sizes in allocation order (same as weight_buffers)
        alloc_plan = []
        for lid in range(cfg.num_layers):
            for wk, prefix in LAYER_WEIGHT_KEYS:
                size = int(np.prod(shapes[wk])) * itemsize
                alloc_plan.append((lid, wk, prefix, shapes[wk], size))
        # norm_weight, lm_head_weight, tok_embedding
        norm_shape = (cfg.hidden_size,)
        lm_head_shape = (cfg.hidden_size, cfg.vocab_size // tp)
        cols_per_rank = cfg.hidden_size // tp
        tok_shape = (cfg.vocab_size, cols_per_rank)
        alloc_plan.append((None, "norm_weight", None, norm_shape, int(np.prod(norm_shape)) * itemsize))
        alloc_plan.append((None, "lm_head_weight", None, lm_head_shape, int(np.prod(lm_head_shape)) * itemsize))
        alloc_plan.append((None, "tok_embedding", None, tok_shape, int(np.prod(tok_shape)) * itemsize))

        total_bytes = sum(size for *_, size in alloc_plan)
        spike = get_spike_singleton()

        # Sender: contiguous arena for single-MR registration (never does spike_reset).
        # Receiver: individual allocations to avoid ibv_reg_mr regression after NRT reinit.
        use_arena = os.environ.get("NKIPY_CHECKPOINT") is not None
        if use_arena:
            self._weight_arena = spike.allocate_tensor(size=total_bytes, core_id=0, name="weight_arena")

        offset = 0
        self.layer_tensors = [{} for _ in range(cfg.num_layers)]
        for lid, wk, prefix, shape, size in alloc_plan:
            name = f"{prefix}_L{lid}" if lid is not None else wk
            if use_arena:
                tensor_ref = spike.slice_from_tensor(self._weight_arena, offset=offset, size=size)
            else:
                tensor_ref = spike.allocate_tensor(size=size, core_id=0, name=name)
            dt = DeviceTensor(tensor_ref=tensor_ref, shape=shape, dtype=cfg.dtype, name=name)
            if lid is not None:
                self.layer_tensors[lid][wk] = dt
            elif wk == "norm_weight":
                self.norm_weight = dt
            elif wk == "lm_head_weight":
                self.lm_head_weight = dt
            elif wk == "tok_embedding":
                self.tok_embedding_device = dt
            offset += size

        # Cache tensors (not part of P2P) — allocated separately
        for lid in range(cfg.num_layers):
            layer = self.layer_tensors[lid]
            layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{lid}")
            layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{lid}")

    @classmethod
    def compute_weight_sizes(cls, config) -> list:
        """Compute weight buffer sizes from config without allocating tensors."""
        cfg = config
        tp = dist.get_world_size()
        n_local_kv_heads = max(1, cfg.num_kv_heads // tp)
        n_local_heads = cfg.num_heads // tp
        itemsize = np.dtype(cfg.dtype).itemsize

        qkv_dim = (n_local_heads + 2 * n_local_kv_heads) * cfg.head_dim
        shapes = {
            "qkv_weight": (cfg.hidden_size, qkv_dim),
            "o_weight": (n_local_heads * cfg.head_dim, cfg.hidden_size),
            "gate_up_weight": (cfg.hidden_size, 2 * cfg.intermediate_size),
            "down_weight": (cfg.intermediate_size, cfg.hidden_size),
            "input_weight": (cfg.hidden_size,),
            "post_attention_weight": (cfg.hidden_size,),
        }

        sizes = []
        for lid in range(cfg.num_layers):
            for wk, _ in LAYER_WEIGHT_KEYS:
                sizes.append(int(np.prod(shapes[wk])) * itemsize)
        # norm_weight, lm_head_weight, tok_embedding
        sizes.append(cfg.hidden_size * itemsize)
        sizes.append(cfg.hidden_size * (cfg.vocab_size // tp) * itemsize)
        cols_per_rank = cfg.hidden_size // tp
        sizes.append(cfg.vocab_size * cols_per_rank * itemsize)
        return sizes

    def weight_buffers(self):
        """Yield (name, va, size_bytes) for all weight tensors (for P2P)."""
        for name, va, size, _ in self._iter_weights():
            yield name, va, size

    def weight_buffers_with_tensors(self):
        """Yield (name, va, size_bytes, device_tensor) for host-staged DMA."""
        return list(self._iter_weights())

    def _iter_weights(self):
        for lid, layer in enumerate(self.layer_tensors):
            for key, dt in layer.items():
                if key in ("cache_k", "cache_v"):
                    continue
                va = dt.tensor_ref.va
                size = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
                yield f"layers.{lid}.{key}", va, size, dt
        for attr in ("norm_weight", "lm_head_weight", "tok_embedding_device"):
            dt = getattr(self, attr, None)
            if dt is not None:
                va = dt.tensor_ref.va
                size = int(np.prod(dt.shape) * np.dtype(dt.dtype).itemsize)
                name = "tok_embedding" if attr == "tok_embedding_device" else attr
                yield name, va, size, dt

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

    def embed_tokens(self, input_ids):
        """Hidden-dim-parallel embedding lookup with all-gather."""
        if self.tok_embedding is None and self.tok_embedding_device is not None:
            self.tok_embedding = self.tok_embedding_device.torch()
        return _tp_embedding_lookup(
            self.tok_embedding, torch.as_tensor(input_ids, dtype=torch.long),
        )

    def generate(self, input_ids):
        """Yield one token tensor per step (prefill + decode)."""
        cfg = self.config
        hidden = DeviceTensor.from_torch(self.embed_tokens(input_ids), "hidden_states")
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
            hidden = DeviceTensor.from_torch(self.embed_tokens(next_id_torch), "h0/res1")

            for i in range(cfg.num_layers):
                self.kernel_tkg(inputs=self._build_inputs(i, hidden, t_sp),
                                outputs=self._build_outputs(i, hidden))

            self.kernel_tkg_greedy_sampling(
                inputs={"h": hidden, "norm_weight": self.norm_weight, "lm_head_weight": self.lm_head_weight},
                outputs={"output0": next_id})
            next_id_torch = next_id.torch().reshape(cfg.max_batch_size, 1).to(dtype=torch.int)
            yield next_id_torch
