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

from .rmsnorm import rmsnorm_kernel
from .rope import apply_rotary_emb_kernel
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


def _attention(
    attn_input,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    n_heads,
    n_kv_heads,
    head_dim,
    freqs_cos,
    freqs_sin,
    attn_mask,
):
    """Self-attention over the drafter's K parallel positions (no GQA bias/sink).

    attn_mask is an additive (S, S) cross-depth causal mask (compile-time const).
    """
    B, S, _ = attn_input.shape

    xq = np.matmul(attn_input, q_proj).reshape(B, S, n_heads, head_dim)
    xk = np.matmul(attn_input, k_proj).reshape(B, S, n_kv_heads, head_dim)
    xv = np.matmul(attn_input, v_proj).reshape(B, S, n_kv_heads, head_dim)

    xq, xk = apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin)

    n_rep = n_heads // n_kv_heads
    keys = _repeat_kv(xk, n_rep)
    values = _repeat_kv(xv, n_rep)

    # BSHD -> BHSD
    xq = xq.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    scores = (xq @ keys.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    scores = scores.astype(nl.bfloat16)
    scores = scores + np.expand_dims(attn_mask, axis=[0, 1])

    weights = softmax_kernel(scores)
    out = weights @ values  # BHSD

    out = out.transpose(0, 2, 1, 3).reshape(B, S, n_heads * head_dim)
    return np.matmul(out, o_proj)


def drafter_layer(
    x,
    weights,
    cfg_norm_eps,
    n_heads,
    n_kv_heads,
    head_dim,
    freqs_cos,
    freqs_sin,
    attn_mask,
    is_fusion,
):
    """Run one drafter decoder layer.

    `weights` is a dict of this layer's numpy/device arrays. For the fusion
    midlayer it additionally contains ``hidden_norm`` and `x` is ``2*hidden`` wide;
    for plain layers `x` is ``hidden`` wide.
    """
    if is_fusion:
        hidden_size = x.shape[-1] // 2
        embeds = x[:, :, :hidden_size]
        hidden = x[:, :, hidden_size:]

        # norm_before_residual is False for this checkpoint: residual is the raw
        # hidden half, hidden_norm is applied only to the attention input.
        residual = hidden
        hidden_n = rmsnorm_kernel(hidden, weights["hidden_norm"], cfg_norm_eps)
        embeds_n = rmsnorm_kernel(embeds, weights["input_layernorm"], cfg_norm_eps)
        attn_input = np.concatenate([embeds_n, hidden_n], axis=-1)
    else:
        residual = x
        attn_input = rmsnorm_kernel(x, weights["input_layernorm"], cfg_norm_eps)

    attn_out = _attention(
        attn_input,
        weights["q_proj"],
        weights["k_proj"],
        weights["v_proj"],
        weights["o_proj"],
        n_heads,
        n_kv_heads,
        head_dim,
        freqs_cos,
        freqs_sin,
        attn_mask,
    )

    h = residual + attn_out
    residual = h
    h = rmsnorm_kernel(h, weights["post_attention_layernorm"], cfg_norm_eps)
    h = _mlp(h, weights["gate_proj"], weights["up_proj"], weights["down_proj"])
    return residual + h
