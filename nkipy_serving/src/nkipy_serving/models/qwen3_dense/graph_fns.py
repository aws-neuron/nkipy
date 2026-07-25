"""Device-compilable sub-functions (traceable by nkipy DeviceKernel).

These use only supported numpy ops: matmul, element-wise, reshape,
concatenate, reductions. No //, %, int() on tensors, no data-dependent
loops. Each can be passed to DeviceKernel.compile_and_load().
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.models.common.attn_fns import (
    cpu_attn_fn as cpu_attn_fn,
)
from nkipy_serving.models.common.attn_fns import (
    nki_attn_fn as nki_attn_fn,
)
from nkipy_serving.ops.nn import (
    apply_head_rms_norm as _apply_head_rms_norm,
)
from nkipy_serving.ops.nn import (
    apply_rms_norm as _apply_rms_norm,
)
from nkipy_serving.ops.nn import (
    apply_rope as _apply_rope,
)
from nkipy_serving.ops.nn import (
    mlp_block as _mlp_block,
)


def embedding_fn(input_ids: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Embedding lookup. [total_tokens] -> [total_tokens, hidden_size]."""
    return embeddings[input_ids]


def pre_attn_fn(
    hidden: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-attention: norm -> QKV projection -> head norms -> RoPE.

    Args:
        hidden: [total_tokens, hidden_size]
        cos, sin: [total_tokens, head_dim//2] precomputed RoPE
    Returns:
        (Q, K, V) each with appropriate shapes:
        Q: [total_tokens, num_heads, head_dim]
        K: [total_tokens, num_kv_heads, head_dim]
        V: [total_tokens, num_kv_heads, head_dim]
    """
    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, input_norm, eps=rms_norm_eps)
    total_tokens = normed.shape[0]
    q = (normed @ w_q).astype(hidden_dtype).reshape(total_tokens, num_heads, head_dim)
    k = (
        (normed @ w_k)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    v = (
        (normed @ w_v)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    q = _apply_head_rms_norm(q, q_norm, eps=rms_norm_eps)
    k = _apply_head_rms_norm(k, k_norm, eps=rms_norm_eps)
    q = _apply_rope(q, cos=cos, sin=sin)
    k = _apply_rope(k, cos=cos, sin=sin)
    return q, k, v


def post_attn_fn(
    hidden: np.ndarray,
    context: np.ndarray,
    w_o: np.ndarray,
    post_attn_norm: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    num_heads: int,
    head_dim: int,
    rms_norm_eps: float = 1e-6,
    tp_degree: int = 1,
) -> np.ndarray:
    """Post-attention: output proj + residual + norm + MLP + residual.

    Args:
        hidden: [total_tokens, hidden_size] (residual input)
        context: [total_tokens, num_heads, head_dim] (attention output)
        tp_degree: If > 1, all-reduce after attn proj and MLP.
    Returns:
        [total_tokens, hidden_size]
    """
    hidden_dtype = hidden.dtype
    total_tokens = hidden.shape[0]
    attn_out = (
        context.reshape(total_tokens, num_heads * head_dim).astype(hidden_dtype) @ w_o
    ).astype(hidden_dtype)
    if tp_degree > 1:
        import nkipy.distributed.collectives as cc

        attn_out = cc.all_reduce(
            attn_out,
            replica_groups=[list(range(tp_degree))],
            reduce_op=np.add,
        )
    hidden = (hidden + attn_out).astype(hidden_dtype)
    post_normed = _apply_rms_norm(hidden, post_attn_norm, eps=rms_norm_eps)
    mlp_out = _mlp_block(post_normed, w_gate=w_gate, w_up=w_up, w_down=w_down)
    if tp_degree > 1:
        import nkipy.distributed.collectives as cc

        mlp_out = cc.all_reduce(
            mlp_out,
            replica_groups=[list(range(tp_degree))],
            reduce_op=np.add,
        )
    return (hidden + mlp_out).astype(hidden_dtype)


def transformer_layer_nki_fn(
    hidden: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    kv_cache: np.ndarray,
    slot_mapping: np.ndarray,
    # Prefill tile plan (fixed-shape).
    p_tqi: np.ndarray,
    p_tbt: np.ndarray,
    p_tm: np.ndarray,
    p_ndls: np.ndarray,
    p_qup: np.ndarray,
    p_lti: np.ndarray,
    # Decode tile plan (fixed-shape).
    d_tqi: np.ndarray,
    d_tbt: np.ndarray,
    d_tm: np.ndarray,
    d_ndls: np.ndarray,
    d_qup: np.ndarray,
    d_lti: np.ndarray,
    # Post-attn weights.
    w_o: np.ndarray,
    post_attn_norm: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    tp_degree: int,
) -> np.ndarray:
    """Single-layer graph: pre_attn -> kv_update -> NKI attention -> post_attn.

    This is intended to be compiled as a single DeviceKernel per token_bucket.
    """
    from nkipy_serving.attention.nki_blocksparse_flash_attention import (
        NKI_MIN_Q_SEQLEN,
        nki_attention_unified,
        nki_update_kv_cache_core,
    )

    token_bucket = int(hidden.shape[0])
    attn_bucket = max(token_bucket, int(NKI_MIN_Q_SEQLEN))
    softmax_scale = 1.0 / (float(head_dim) ** 0.5)

    q, k, v = pre_attn_fn(
        hidden,
        input_norm=input_norm,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        q_norm=q_norm,
        k_norm=k_norm,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
    )

    # KV cache update (paged). Updated cache is used immediately by attention.
    kv_cache = nki_update_kv_cache_core(k, v, kv_cache, slot_mapping)

    # NKI attention kernel requires seqlen_q >= 128; pad Q/K/V as needed.
    if attn_bucket > token_bucket:
        pad_t = attn_bucket - token_bucket
        # Important: do not use `np.zeros(...)` here. DeviceKernel tracing only
        # converts *function arguments* (np.ndarray) into HLO parameters, so
        # raw numpy arrays created inside the traced function would leak into
        # the HLO operand list and crash serialization.
        #
        # Use NKIPy tensor_apis instead so the zeros are traceable.
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
    context = (
        context_attn[:token_bucket] if attn_bucket > token_bucket else context_attn
    )

    return post_attn_fn(
        hidden,
        context=context,
        w_o=w_o,
        post_attn_norm=post_attn_norm,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        num_heads=num_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        tp_degree=tp_degree,
    )
