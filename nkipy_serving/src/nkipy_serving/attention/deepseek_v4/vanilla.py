"""CPU oracle helpers for DeepSeek-V4 sparse attention."""

from __future__ import annotations

import numpy as np

from nkipy_serving.attention.deepseek_v4.types import (
    Dsv4AttentionMetadata,
)


def dsv4_vanilla_update_kv_cache(
    kv: np.ndarray,
    kv_cache: np.ndarray,
    slot_mapping: np.ndarray,
) -> None:
    """Write DSV4 attention KV rows into a flat cache by global slot ID."""
    if kv.ndim != 2:
        raise ValueError(f"kv must be [total_tokens, head_dim], got {kv.shape}")
    if kv_cache.ndim != 2:
        raise ValueError(
            f"kv_cache must be [num_slots, head_dim], got {kv_cache.shape}"
        )
    if kv.shape[1] != kv_cache.shape[1]:
        raise ValueError(
            f"head_dim mismatch: kv={kv.shape[1]}, kv_cache={kv_cache.shape[1]}"
        )
    slots = np.asarray(slot_mapping, dtype=np.int64).reshape(-1)
    if slots.shape != (kv.shape[0],):
        raise ValueError(f"slot_mapping must be [{kv.shape[0]}], got {slots.shape}")
    if np.any(slots < 0) or np.any(slots >= kv_cache.shape[0]):
        raise ValueError("slot_mapping contains values outside the flat KV cache range")
    kv_cache[slots] = kv


def dsv4_vanilla_sparse_attention_core(
    q: np.ndarray,
    kv_cache: np.ndarray,
    metadata: Dsv4AttentionMetadata,
    attn_sink: np.ndarray,
    softmax_scale: float,
) -> np.ndarray:
    """Reference DSV4 sparse attention over global cache slots."""
    if q.ndim != 3:
        raise ValueError(
            f"q must be [total_tokens, num_heads, head_dim], got {q.shape}"
        )
    if kv_cache.ndim != 2:
        raise ValueError(
            f"kv_cache must be [num_slots, head_dim], got {kv_cache.shape}"
        )
    if q.shape[0] != metadata.total_tokens:
        raise ValueError(
            f"q total_tokens={q.shape[0]} does not match metadata={metadata.total_tokens}"
        )
    if q.shape[2] != kv_cache.shape[1]:
        raise ValueError(
            f"head_dim mismatch: q={q.shape[2]}, kv_cache={kv_cache.shape[1]}"
        )
    if attn_sink.shape != (q.shape[1],):
        raise ValueError(
            f"attn_sink must be [num_heads={q.shape[1]}], got {attn_sink.shape}"
        )

    topk = np.asarray(metadata.sparse.topk_indices, dtype=np.int64)
    valid = topk >= 0
    if np.any(topk[valid] >= kv_cache.shape[0]):
        raise ValueError("topk_indices contain slots outside kv_cache")

    from nkipy_serving.attention.deepseek_v4.kernels import (
        gather_kv_and_mask,
        sparse_attention_oracle,
    )

    gathered, valid_mask = gather_kv_and_mask(kv_cache, topk)
    out = sparse_attention_oracle(
        q,
        gathered,
        valid_mask,
        attn_sink,
        float(softmax_scale),
    )
    return out.astype(q.dtype, copy=False)


def dsv4_vanilla_attn_fn(
    q: np.ndarray,
    kv: np.ndarray,
    kv_cache: np.ndarray,
    metadata: Dsv4AttentionMetadata,
    attn_sink: np.ndarray,
    softmax_scale: float,
) -> np.ndarray:
    """KV cache update + DSV4 sparse attention oracle."""
    dsv4_vanilla_update_kv_cache(kv, kv_cache, metadata.base.slot_mapping)
    return dsv4_vanilla_sparse_attention_core(
        q,
        kv_cache,
        metadata,
        attn_sink,
        softmax_scale,
    )


__all__ = [
    "dsv4_vanilla_attn_fn",
    "dsv4_vanilla_sparse_attention_core",
    "dsv4_vanilla_update_kv_cache",
]
