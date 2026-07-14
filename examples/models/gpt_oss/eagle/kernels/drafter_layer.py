"""Decoder layers for the P-EAGLE drafter.

Two layer shapes share most of the math:

  * The **fusion midlayer** (layer 0) receives ``cat(embeds, hidden)`` of width
    ``2*hidden``. It normalizes the two halves separately (``input_layernorm`` on
    the embedding half, ``hidden_norm`` on the hidden half), re-concatenates them
    as the attention input, and keeps the (un-normalized, or normed-before-
    residual) hidden half as the residual. Its QKV projections take ``2*hidden``.
  * **Plain layers** are standard Llama decoder layers operating on ``hidden``.

Both use rotate-halves llama3 RoPE and plain SwiGLU.
"""

import neuronxcc.nki.language as nl
import numpy as np
from nkipy.core import tensor_apis

from .rmsnorm import rmsnorm_kernel
from .rope import apply_rotary_emb_kernel, compute_cos_sin_cache
from .softmax import softmax_kernel


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def _mlp(x, gate_proj, up_proj, down_proj):
    """Plain Llama SwiGLU MLP. Weights stored in x @ W form (hidden, inter)."""
    gate = _silu(np.matmul(x, gate_proj))
    up = np.matmul(x, up_proj)
    return np.matmul(gate * up, down_proj)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=2)


def _attention_cached(
    attn_input,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    n_heads,
    n_kv_heads,
    head_dim,
    rope_inv_freq,
    rope_attention_scaling,
    cache_k,
    cache_v,
    start_pos,
):
    """KV-cached self-attention over absolute positions.

    Mirrors the base gpt-oss attention decode path (kernels/attention.py) but for
    the drafter: no sinks, no bias, no GQA all-reduce (the drafter is replicated).

    When ``start_pos`` is None: prefill mode — process the full ``S`` positions and
    write them to cache slots ``0..S-1``. Otherwise: append ``S`` new positions at
    absolute offsets ``start_pos + [0..S-1]``, scatter their K/V into the cache, and
    attend to the whole cache under a causal mask over absolute positions.
    """
    is_prefill = start_pos is None
    B, S, _ = attn_input.shape

    xq = np.matmul(attn_input, q_proj).reshape(B, S, n_heads, head_dim)
    xk = np.matmul(attn_input, k_proj).reshape(B, S, n_kv_heads, head_dim)
    xv = np.matmul(attn_input, v_proj).reshape(B, S, n_kv_heads, head_dim)

    max_seq_len = cache_k.shape[1]
    freqs_cos, freqs_sin = compute_cos_sin_cache(
        rope_inv_freq, max_seq_len, rope_attention_scaling, dtype=nl.bfloat16
    )
    if is_prefill:
        freqs_cos = freqs_cos[0:S]
        freqs_sin = freqs_sin[0:S]
    else:
        query_pos = start_pos + np.arange(S, dtype=np.int32)
        freqs_cos = tensor_apis.constant(freqs_cos)
        freqs_sin = tensor_apis.constant(freqs_sin)
        freqs_cos = freqs_cos[query_pos]
        freqs_sin = freqs_sin[query_pos]
    xq, xk = apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin)

    # KV cache update.
    if is_prefill:
        cache_k[:, :S] = xk
        cache_v[:, :S] = xv
    else:
        cache_k[:, query_pos] = xk
        cache_v[:, query_pos] = xv

    n_rep = n_heads // n_kv_heads
    keys = _repeat_kv(cache_k, n_rep)
    values = _repeat_kv(cache_v, n_rep)

    # BSHD -> BHSD
    xq = xq.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    k_seq_len = keys.shape[2]
    scores = (xq @ keys.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    scores = scores.astype(nl.bfloat16)

    # Causal mask over absolute positions (compile-time constant).
    NEG = -100000.0
    full_mask = np.triu(np.ones((k_seq_len, k_seq_len)) * NEG, k=1).astype(scores.dtype)
    causal_mask = tensor_apis.constant(full_mask)
    if is_prefill:
        scores = scores + np.expand_dims(causal_mask[:S, :k_seq_len], axis=[0, 1])
    else:
        # One mask row per query token: query at start_pos+i attends to keys
        # <= start_pos+i (block-causal for the S committed+ptd positions).
        scores = scores + np.expand_dims(causal_mask[query_pos, :k_seq_len], axis=[0, 1])

    weights = softmax_kernel(scores)
    out = weights @ values  # BHSD
    out = out.transpose(0, 2, 1, 3).reshape(B, S, n_heads * head_dim)
    return np.matmul(out, o_proj)


def drafter_layer_cached(
    x,
    weights,
    cfg_norm_eps,
    n_heads,
    n_kv_heads,
    head_dim,
    rope_inv_freq,
    rope_attention_scaling,
    cache_k,
    cache_v,
    start_pos,
    is_fusion,
):
    """KV-cached variant of :func:`drafter_layer`.

    Same layer math, but attention reads/writes a persistent per-layer KV cache
    and attends over absolute positions (``start_pos`` runtime offset), so the new
    positions see the full prompt + accepted-token context. ``start_pos=None``
    runs prefill (writes cache slots ``0..S-1``).
    """
    if is_fusion:
        hidden_size = x.shape[-1] // 2
        embeds = x[:, :, :hidden_size]
        hidden = x[:, :, hidden_size:]
        residual = hidden
        hidden_n = rmsnorm_kernel(hidden, weights["hidden_norm"], cfg_norm_eps)
        embeds_n = rmsnorm_kernel(embeds, weights["input_layernorm"], cfg_norm_eps)
        attn_input = np.concatenate([embeds_n, hidden_n], axis=-1)
    else:
        residual = x
        attn_input = rmsnorm_kernel(x, weights["input_layernorm"], cfg_norm_eps)

    attn_out = _attention_cached(
        attn_input,
        weights["q_proj"],
        weights["k_proj"],
        weights["v_proj"],
        weights["o_proj"],
        n_heads,
        n_kv_heads,
        head_dim,
        rope_inv_freq,
        rope_attention_scaling,
        cache_k,
        cache_v,
        start_pos,
    )

    h = residual + attn_out
    residual = h
    h = rmsnorm_kernel(h, weights["post_attention_layernorm"], cfg_norm_eps)
    h = _mlp(h, weights["gate_proj"], weights["up_proj"], weights["down_proj"])
    return residual + h
