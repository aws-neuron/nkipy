"""Model-agnostic attention graph functions shared by eager executors.

These wrap NKI paged attention and vanilla attention with the KV-cache
update + Q-padding plumbing needed to run them as Fragment stages. They
do not take model-specific parameters (no sink, no query/key norm), so
every model whose attention shape matches ``(q, k, v, kv_cache, ...)``
plus the standard tile-plan inputs can reuse them.
"""

from __future__ import annotations

import numpy as np


def nki_attn_fn(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    kv_cache: np.ndarray,
    slot_mapping: np.ndarray,
    p_tqi: np.ndarray,
    p_tbt: np.ndarray,
    p_tm: np.ndarray,
    p_ndls: np.ndarray,
    p_qup: np.ndarray,
    p_lti: np.ndarray,
    d_tqi: np.ndarray,
    d_tbt: np.ndarray,
    d_tm: np.ndarray,
    d_ndls: np.ndarray,
    d_qup: np.ndarray,
    d_lti: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """KV cache update + NKI paged attention (no sink)."""
    from nkipy_serving.attention.nki_blocksparse_flash_attention import (
        NKI_MIN_Q_SEQLEN,
        nki_attention_unified,
        nki_update_kv_cache_core,
    )

    token_bucket = int(q.shape[0])
    attn_bucket = max(token_bucket, int(NKI_MIN_Q_SEQLEN))
    softmax_scale = 1.0 / (float(head_dim) ** 0.5)

    kv_cache = nki_update_kv_cache_core(k, v, kv_cache, slot_mapping)

    if attn_bucket > token_bucket:
        pad_t = attn_bucket - token_bucket
        from nkipy.core.tensor_apis import zeros

        q_pad = zeros((pad_t, num_heads, head_dim), dtype=q.dtype)
        k_pad = zeros((pad_t, num_kv_heads, head_dim), dtype=k.dtype)
        v_pad = zeros((pad_t, num_kv_heads, head_dim), dtype=v.dtype)
        q_attn = np.concatenate((q, q_pad), axis=0)
        k_attn = np.concatenate((k, k_pad), axis=0)
        v_attn = np.concatenate((v, v_pad), axis=0)
    else:
        q_attn, k_attn, v_attn = q, k, v

    context_attn = nki_attention_unified(
        q_attn,
        k_attn,
        v_attn,
        kv_cache,
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
        softmax_scale=softmax_scale,
    )
    return context_attn[:token_bucket] if attn_bucket > token_bucket else context_attn


def cpu_attn_fn(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    kv_cache: np.ndarray,
    attn_metadata,
) -> np.ndarray:
    """KV cache update + vanilla attention (CPU reference, no sink)."""
    from nkipy_serving.attention.vanilla import (
        vanilla_attention_core,
        vanilla_update_kv_cache,
    )

    vanilla_update_kv_cache(k, v, kv_cache, attn_metadata.slot_mapping)
    return vanilla_attention_core(q, kv_cache, attn_metadata)
