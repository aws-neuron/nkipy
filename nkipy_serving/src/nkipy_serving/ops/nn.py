"""Shared numpy math utilities for model implementations.

Pure-numpy helpers used across multiple model executors:
RMS norm, RoPE, SiLU, MLP, and TP sharding utilities.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def apply_rms_norm(
    x: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-6,
    compute_dtype=np.float32,
) -> np.ndarray:
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    scale = scale.astype(compute_dtype)
    eps = np.float32(eps)
    inv_rms = np.float32(1.0) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    out = x * inv_rms * scale.reshape((1,) * (x.ndim - 1) + (-1,))
    return out.astype(original_dtype)


def apply_head_rms_norm(
    x: np.ndarray,
    scale: np.ndarray,
    eps: float,
    compute_dtype=np.float32,
) -> np.ndarray:
    """RMS norm applied per-head: x is [T, num_heads, head_dim], scale is [head_dim]."""
    original_dtype = x.dtype
    x = x.astype(compute_dtype)
    scale = scale.astype(compute_dtype)
    eps = np.float32(eps)
    inv_rms = np.float32(1.0) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    out = x * inv_rms * scale.reshape(1, 1, -1)
    return out.astype(original_dtype)


# ---------------------------------------------------------------------------
# RoPE (standard, no YaRN)
# ---------------------------------------------------------------------------


def build_rope_cache_for_positions(
    positions: np.ndarray,
    head_dim: int,
    theta: float,
    dtype: np.dtype = ml_dtypes.bfloat16,
) -> tuple[np.ndarray, np.ndarray]:
    if head_dim % 2 != 0:
        raise RuntimeError(f"head_dim must be even for RoPE, got {head_dim}")
    half_dim = head_dim // 2
    inv_freq = np.float32(1.0) / (
        np.float32(theta)
        ** (np.arange(half_dim, dtype=np.float32) / np.float32(half_dim))
    )
    freqs = positions.astype(np.float32).reshape((-1, 1)) * inv_freq.reshape((1, -1))
    return np.cos(freqs).astype(dtype), np.sin(freqs).astype(dtype)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half_dim = x.shape[-1] // 2
    x1 = x[:, :, :half_dim]
    x2 = x[:, :, half_dim : 2 * half_dim]
    cos_v = cos.astype(x.dtype).reshape((cos.shape[0], 1, cos.shape[1]))
    sin_v = sin.astype(x.dtype).reshape((sin.shape[0], 1, sin.shape[1]))
    out1 = x1 * cos_v - x2 * sin_v
    out2 = x1 * sin_v + x2 * cos_v
    if x.shape[-1] == 2 * half_dim:
        return np.concatenate((out1, out2), axis=-1).astype(x.dtype)
    tail = x[:, :, 2 * half_dim :]
    return np.concatenate((out1, out2, tail), axis=-1).astype(x.dtype)


# ---------------------------------------------------------------------------
# Activation + MLP
# ---------------------------------------------------------------------------


def silu(x: np.ndarray) -> np.ndarray:
    original_dtype = x.dtype
    x = x.astype(np.float32)
    out = np.float32(0.5) * x * (np.float32(1.0) + np.tanh(np.float32(0.5) * x))
    return out.astype(original_dtype)


def mlp_block(
    hidden_states: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    hidden_dtype = hidden_states.dtype
    gate = (hidden_states @ w_gate).astype(hidden_dtype)
    up = (hidden_states @ w_up).astype(hidden_dtype)
    fused = silu(gate) * up
    return (fused @ w_down).astype(hidden_dtype)


# ---------------------------------------------------------------------------
# TP sharding utilities
# ---------------------------------------------------------------------------


def validate_tp_runtime(tp_degree: int, tp_rank: int, tp_world_size: int) -> None:
    if tp_degree <= 0:
        raise RuntimeError(f"tp_degree must be > 0, got {tp_degree}")
    if tp_world_size != tp_degree:
        raise RuntimeError(
            "tp_world_size must equal tp_degree. "
            f"Got tp_world_size={tp_world_size}, tp_degree={tp_degree}"
        )
    if tp_rank < 0 or tp_rank >= tp_world_size:
        raise RuntimeError(
            f"tp_rank out of range: tp_rank={tp_rank}, tp_world_size={tp_world_size}"
        )


def require_divisible(value: int, divisor: int, field_name: str) -> int:
    if value % divisor != 0:
        raise RuntimeError(f"{field_name} must be divisible by {divisor}, got {value}")
    return value // divisor


def kv_head_indices_for_rank(
    global_num_kv_heads: int,
    tp_degree: int,
    tp_rank: int,
) -> tuple[int, ...]:
    if global_num_kv_heads % tp_degree == 0:
        local_num_kv_heads = global_num_kv_heads // tp_degree
        start = tp_rank * local_num_kv_heads
        return tuple(range(start, start + local_num_kv_heads))
    if tp_degree % global_num_kv_heads == 0:
        ranks_per_kv_head = tp_degree // global_num_kv_heads
        kv_head = tp_rank // ranks_per_kv_head
        return (int(kv_head),)
    raise RuntimeError(
        "Unsupported KV-head TP mapping. "
        f"num_kv_heads={global_num_kv_heads}, tp_degree={tp_degree}"
    )


def select_head_columns(
    matrix: np.ndarray,
    head_indices: tuple[int, ...],
    head_dim: int,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected rank-2 matrix, got shape={matrix.shape}")
    reshaped = matrix.reshape(matrix.shape[0], -1, head_dim)
    out = reshaped[:, list(head_indices), :].reshape(matrix.shape[0], -1)
    return np.asarray(out, dtype=matrix.dtype)


def select_head_rows(
    matrix: np.ndarray,
    head_indices: tuple[int, ...],
    head_dim: int,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected rank-2 matrix, got shape={matrix.shape}")
    reshaped = matrix.reshape(-1, head_dim, matrix.shape[1])
    out = reshaped[list(head_indices), :, :].reshape(-1, matrix.shape[1])
    return np.asarray(out, dtype=matrix.dtype)
