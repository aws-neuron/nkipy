"""Multi-head attention for the PixArt DiT.

Two variants, both bidirectional (no causal mask), stateless (no KV cache), and
without RoPE — this is an image/text encoder-style attention, unlike the
autoregressive decode attention in the qwen3 example:

* ``self_attention_kernel`` — Q, K, V all come from the image latent tokens.
* ``cross_attention_kernel`` — Q comes from the image tokens, K/V from the
  projected T5 caption tokens (``context``).

Both use separate to_q / to_k / to_v / to_out projections to match the
diffusers PixArt ``Attention`` state dict.
"""

import numpy as np

from .softmax import softmax_kernel


def _mha(q, k, v, n_heads, head_dim, mask_bias=None):
    """Scaled dot-product multi-head attention.

    Args:
        q: (B, Lq, n_heads*head_dim)
        k, v: (B, Lk, n_heads*head_dim)
        mask_bias: optional (B, 1, 1, Lk) additive bias applied to scores before
            softmax (0 for keep, large-negative for masked key positions).
    Returns:
        (B, Lq, n_heads*head_dim)
    """
    B, Lq, _ = q.shape
    Lk = k.shape[1]

    # (B, L, H, d) -> (B, H, L, d)
    q = q.reshape(B, Lq, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, Lk, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, Lk, n_heads, head_dim).transpose(0, 2, 1, 3)

    # scores: (B, H, Lq, Lk)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    scores = scores.astype(np.float32)
    if mask_bias is not None:
        scores = scores + mask_bias.astype(np.float32)
    weights = softmax_kernel(scores).astype(v.dtype)

    out = weights @ v  # (B, H, Lq, d)
    out = out.transpose(0, 2, 1, 3).reshape(B, Lq, n_heads * head_dim)
    return out


def self_attention_kernel(
    x,
    q_weight, q_bias,
    k_weight, k_bias,
    v_weight, v_bias,
    out_weight, out_bias,
    n_heads,
    head_dim,
):
    """Bidirectional self-attention over image latent tokens.

    Weights are (hidden, hidden); biases are (hidden,) or None.
    """
    q = np.matmul(x, q_weight)
    k = np.matmul(x, k_weight)
    v = np.matmul(x, v_weight)
    if q_bias is not None:
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

    out = _mha(q, k, v, n_heads, head_dim)

    out = np.matmul(out, out_weight)
    if out_bias is not None:
        out = out + out_bias
    return out


def cross_attention_kernel(
    x,
    context,
    q_weight, q_bias,
    k_weight, k_bias,
    v_weight, v_bias,
    out_weight, out_bias,
    n_heads,
    head_dim,
    mask_bias=None,
):
    """Cross-attention: queries from image tokens ``x``, keys/values from
    the projected caption tokens ``context`` (B, Ltext, hidden).

    ``mask_bias`` (B, 1, 1, Ltext) masks out T5 padding tokens.
    """
    q = np.matmul(x, q_weight)
    k = np.matmul(context, k_weight)
    v = np.matmul(context, v_weight)
    if q_bias is not None:
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

    out = _mha(q, k, v, n_heads, head_dim, mask_bias=mask_bias)

    out = np.matmul(out, out_weight)
    if out_bias is not None:
        out = out + out_bias
    return out
