"""Shared device utilities for model executors.

Provides device tensor allocation and KV cache management shared across
all model executors. Uses lazy imports from nkipy.runtime so that
non-device processes (scheduler, tokenizer) never trigger NRT init.
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.models.reload_utils import (
    overwrite_device_tensor,
)
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls


def _get_device_tensor_cls():
    """Lazy import of DeviceTensor — cached by Python's import system."""
    return get_device_tensor_cls()


def _get_device_kernel_cls():
    """Lazy import of DeviceKernel — cached by Python's import system."""
    from nkipy.runtime import DeviceKernel

    return DeviceKernel


# ---------------------------------------------------------------------------
# Device tensor allocation
# ---------------------------------------------------------------------------


def alloc_device_scratch(shape: tuple[int, ...], dtype: np.dtype, *, name: str):
    """Allocate output scratch once; kernels fully overwrite these buffers."""
    return _get_device_tensor_cls().from_numpy(np.empty(shape, dtype=dtype), name=name)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


def allocate_device_kv_cache(
    *,
    num_hidden_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    dtype: np.dtype,
) -> list[object]:
    """Allocate persistent device KV caches for NKI attention backend.

    Shape per layer: [2, num_blocks, num_kv_heads, block_size, head_dim]
    The caller should pass num_blocks = kv_pool.num_blocks + 1 (extra scratch block).
    """
    DeviceTensor = _get_device_tensor_cls()
    shape = (2, num_blocks, num_kv_heads, block_size, head_dim)
    return [
        DeviceTensor.from_numpy(np.zeros(shape, dtype=dtype), name=f"kv_cache_L{i}")
        for i in range(num_hidden_layers)
    ]


def pre_allocate_kv_cache_zeros(
    *,
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Pre-allocate zeros array for flush_cache to avoid per-call allocation."""
    return np.zeros(
        (2, num_blocks, num_kv_heads, block_size, head_dim),
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Generated kernel source loading
# ---------------------------------------------------------------------------


def load_generated_kernel_fn(
    *,
    build_dir: str,
    mod_name: str,
    fn_name: str,
    source: str,
):
    """Write generated Python source to disk and load the named function.

    Used by model executors that generate all-layers-in-one-graph kernel
    source at compile time (unrolled layer parameters).
    """
    import importlib.util
    import os
    import sys

    gen_dir = os.path.join(build_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    py_path = os.path.join(gen_dir, f"{mod_name}.py")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(source)

    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load generated module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    fn = getattr(module, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Missing generated kernel function: {fn_name}")
    return fn


def flush_device_kv_cache(
    kv_cache_dev: list[object],
    kv_cache_zeros: np.ndarray,
    kv_pool,
) -> None:
    """Clear CPU-side KV pool and zero-out device KV caches."""
    kv_pool.clear()
    for kv_tensor in kv_cache_dev:
        overwrite_device_tensor(kv_tensor, kv_cache_zeros)


# ---------------------------------------------------------------------------
# Compiler-arg joining
# ---------------------------------------------------------------------------


def join_compiler_args(*args: str) -> str:
    """Merge multiple compiler-arg strings into one, dropping empties."""
    cleaned: list[str] = []
    for arg in args:
        if not arg:
            continue
        stripped = str(arg).strip()
        if stripped:
            cleaned.append(stripped)
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# NKI tile-plan sample arrays (for compile-time shape specialization)
# ---------------------------------------------------------------------------


def nki_tile_plan_sample_arrays(
    *,
    max_num_prefill_tiles: int,
    max_num_decode_tiles: int,
    block_size: int,
) -> tuple[np.ndarray, ...]:
    """Create fixed-shape sample tile-plan tensors for tracing/compilation."""
    _B_P = 128
    _LARGE_Q_TILE = 128
    _LARGE_KV_TILE = 1024
    if block_size <= 0:
        raise RuntimeError(f"block_size must be > 0, got {block_size}")
    if _LARGE_KV_TILE % block_size != 0:
        raise RuntimeError(
            "block_size must divide 1024 for NKI attention tiling. "
            f"Got block_size={block_size}"
        )
    kv_blocks_per_tile = _LARGE_KV_TILE // block_size
    P = int(max_num_prefill_tiles)
    D = int(max_num_decode_tiles)

    p_tqi = np.zeros((P, _LARGE_Q_TILE), dtype=np.int32)
    p_tbt = np.zeros((P, kv_blocks_per_tile), dtype=np.int32)
    p_tm = np.zeros((_B_P, P, _LARGE_Q_TILE // _B_P, _LARGE_KV_TILE), dtype=np.uint8)
    p_ndls = np.zeros((1, 1), dtype=np.int32)
    p_qup = np.zeros((P, 1), dtype=np.uint8)
    p_lti = np.zeros((_B_P, 2), dtype=np.int32)

    d_tqi = np.zeros((D, 1), dtype=np.int32)
    d_tbt = np.zeros((D, kv_blocks_per_tile), dtype=np.int32)
    d_tm = np.zeros((_B_P, D, _LARGE_KV_TILE // _B_P), dtype=np.uint8)
    d_ndls = np.zeros((1, 1), dtype=np.int32)
    d_qup = np.zeros((D, 1), dtype=np.uint8)
    d_lti = np.zeros((_B_P, 2), dtype=np.int32)

    return (
        p_tqi,
        p_tbt,
        p_tm,
        p_ndls,
        p_qup,
        p_lti,
        d_tqi,
        d_tbt,
        d_tm,
        d_ndls,
        d_qup,
        d_lti,
    )
