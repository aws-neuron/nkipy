"""Paged sparse-attention device kernel launcher for DSV4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nkipy_serving.ops.attention.sparse_mla import (
    D_BLOCK,
    K_TILE,
    P_MAX,
    _sparse_attn_batched_paged_entry,
    _sparse_attn_batched_paged_multiK_entry,
)
from nkipy_serving.runtime.device_tensor import dtype_like as _dtype_like
from nkipy_serving.runtime.device_tensor import sample_like as _sample_like
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

_DEVICE_PAGED_ATTENTION_KERNEL_CACHE: dict[tuple, Any] = {}


def _compile_and_load_with_lock(*args: Any, **kwargs: Any) -> Any:
    from nkipy_serving.attention.deepseek_v4 import kernels as dsv4_kernels

    return dsv4_kernels.compile_and_load_with_lock(*args, **kwargs)


def run_sparse_attention_paged_device(
    *,
    q_scaled_t: Any,
    kv_hbm: Any,
    topk_t: Any,
    mask: Any,
    sink: Any,
    output: Any,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Run paged sparse attention with already-device-resident inputs.

    This device entry point does not upload, transpose, mask, or scale on CPU;
    callers must provide
    kernel-ready device tensors:

    - ``q_scaled_t``: ``[tokens, head_dim, num_heads]`` bf16
    - ``kv_hbm``: ``[num_kv_slots, head_dim]`` bf16
    - ``topk_t``: ``[K, tokens]`` int32, safe-clamped for invalid slots
    - ``mask``: ``[tokens, K]`` numeric 0/1 bf16
    - ``sink``: ``[1, num_heads]`` fp32
    - ``output``: ``[tokens, num_heads, head_dim]`` fp32

    Returns the same ``output`` handle after execution.
    """
    q_shape = tuple(int(dim) for dim in getattr(q_scaled_t, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_hbm, "shape"))
    topk_shape = tuple(int(dim) for dim in getattr(topk_t, "shape"))
    mask_shape = tuple(int(dim) for dim in getattr(mask, "shape"))
    sink_shape = tuple(int(dim) for dim in getattr(sink, "shape"))
    out_shape = tuple(int(dim) for dim in getattr(output, "shape"))

    if len(q_shape) != 3:
        raise ValueError(f"q_scaled_t must be [tokens, head_dim, heads], got {q_shape}")
    if len(kv_shape) != 2:
        raise ValueError(f"kv_hbm must be [num_slots, head_dim], got {kv_shape}")
    if len(topk_shape) != 2:
        raise ValueError(f"topk_t must be [K, tokens], got {topk_shape}")
    if len(mask_shape) != 2:
        raise ValueError(f"mask must be [tokens, K], got {mask_shape}")
    if len(sink_shape) != 2:
        raise ValueError(f"sink must be [1, heads], got {sink_shape}")

    tokens, head_dim, num_heads = q_shape
    k, topk_tokens = topk_shape
    if topk_tokens != tokens:
        raise ValueError(f"topk_t tokens={topk_tokens} must match q tokens={tokens}")
    if mask_shape != (tokens, k):
        raise ValueError(f"mask must be [{tokens}, {k}], got {mask_shape}")
    if sink_shape != (1, num_heads):
        raise ValueError(f"sink must be [1, {num_heads}], got {sink_shape}")
    if kv_shape[1] != head_dim:
        raise ValueError(f"kv head_dim={kv_shape[1]} must match q head_dim={head_dim}")
    if out_shape != (tokens, num_heads, head_dim):
        raise ValueError(
            f"output must be [{tokens}, {num_heads}, {head_dim}], got {out_shape}"
        )
    if k % K_TILE:
        raise NotImplementedError(f"K={k} must be a multiple of K_TILE={K_TILE}")
    if head_dim % D_BLOCK:
        raise NotImplementedError(f"head_dim={head_dim} not a multiple of {D_BLOCK}")
    if num_heads > P_MAX:
        raise ValueError(f"num_heads={num_heads} must be <= {P_MAX}")

    cache = (
        _DEVICE_PAGED_ATTENTION_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    # Multi-tile variant subsumes the single-tile path (n_k==1 runs the
    # same outer loop body once) but compiles a different NEFF; key by
    # the entrypoint's name so they don't collide.
    multi_tile = k > K_TILE
    entry = (
        _sparse_attn_batched_paged_multiK_entry
        if multi_tile
        else _sparse_attn_batched_paged_entry
    )
    cache_key = (
        "sparse_attention_paged_device",
        entry.__name__,
        q_shape,
        kv_shape,
        str(_dtype_like(q_scaled_t)),
        str(_dtype_like(kv_hbm)),
        topk_shape,
        str(_dtype_like(topk_t)),
        mask_shape,
        str(_dtype_like(mask)),
        sink_shape,
        str(_dtype_like(sink)),
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            entry,
            _sample_like(q_scaled_t),
            _sample_like(kv_hbm),
            _sample_like(topk_t),
            _sample_like(mask),
            _sample_like(sink),
            name=f"dsv4_sparse_attention_paged_t{tokens}_d{head_dim}_k{k}",
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "q_T": q_scaled_t,
            "kv_hbm": kv_hbm,
            "topk_T": topk_t,
            "mask": mask,
            "sink": sink,
        },
        outputs={"output0": output},
    )
    return output
