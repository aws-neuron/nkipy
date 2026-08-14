"""Multi-head attention for the PixArt DiT.

Two variants, both bidirectional (no causal mask), stateless (no KV cache), and
without RoPE — this is an image/text encoder-style attention, unlike the
autoregressive decode attention in the qwen3 example:

* ``self_attention_kernel`` — Q, K, V all come from the image latent tokens.
  The O(L^2) score matrix over 4096 patches is the DiT's dominant cost. Two
  backends are supported (``backend`` arg):
    - "naive" (default): hand-rolled numpy softmax. Portable and CPU-testable.
    - "cte": nki-library's tuned ``attention_cte`` context/prefill kernel,
      ~3.7x faster at seq=4096 on trn2 (device-only, no CPU backend).
* ``cross_attention_kernel`` — Q comes from the image tokens, K/V from the
  projected T5 caption tokens (``context``). Always the plain hand-rolled
  softmax: the key length (T5 tokens, ~300) is small, so it is not a bottleneck
  and the padding mask is easiest to apply here.

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


# Cache of pre-traced attention_cte ops, keyed by (bh, seq, head_dim, scale,
# dtype). ``attention_cte`` is expensive to specialize, and a DiT stacks 28
# identically-shaped self-attentions; re-specializing per call makes the whole
# graph's compile time explode (>10min for 28 blocks). ``wrap_nki_kernel``
# pre-traces once into a reusable NKICustomOp so all blocks share one compile.
_CTE_OP_CACHE = {}


def _get_cte_op(bh, seq, head_dim, scale, dtype):
    key = (bh, seq, head_dim, float(scale), np.dtype(dtype).name)
    op = _CTE_OP_CACHE.get(key)
    if op is None:
        from nkilib.core.attention.attention_cte import attention_cte
        from nkipy.core.nki_op import wrap_nki_kernel

        # Example operands for tracing: q (bh,seq,d), k (bh,d,seq), v (bh,seq,d).
        q_ex = np.zeros((bh, seq, head_dim), dtype=dtype)
        k_ex = np.zeros((bh, head_dim, seq), dtype=dtype)
        v_ex = np.zeros((bh, seq, head_dim), dtype=dtype)
        op = wrap_nki_kernel(
            attention_cte,
            operands=[q_ex, k_ex, v_ex],
            is_nki_beta_3_version=True,
            kernel_kwargs={"scale": float(scale), "causal_mask": False},
        )
        _CTE_OP_CACHE[key] = op
    return op


def _mha_cte(q, k, v, n_heads, head_dim):
    """Bidirectional multi-head attention via nki-library ``attention_cte``.

    Inputs q/k/v are (B, L, n_heads*head_dim). ``attention_cte`` operates per
    (batch*head) with layout q (bh, seq, d), k (bh, d, seq), v (bh, seq, d), so
    we fold the head axis into the batch axis and lay out K transposed. The
    kernel is pre-traced once and reused across all DiT blocks (see
    ``_get_cte_op``). Imported lazily so the "naive" backend has no nkilib
    dependency and stays importable on CPU / machines without neuronx-cc.
    """
    B, L, _ = q.shape
    scale = np.float32(1.0 / np.sqrt(head_dim))
    bh = B * n_heads

    # (B, L, H, d) -> (B, H, L, d) -> (B*H, L, d)
    q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3).reshape(bh, L, head_dim)
    v = v.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3).reshape(bh, L, head_dim)
    # k needs (B*H, d, seq)
    k = k.reshape(B, L, n_heads, head_dim).transpose(0, 2, 3, 1).reshape(bh, head_dim, L)

    op = _get_cte_op(bh, L, head_dim, scale, q.dtype)
    out = op(q, k, v)  # (bh, L, d)

    # (B*H, L, d) -> (B, H, L, d) -> (B, L, H*d)
    out = out.reshape(B, n_heads, L, head_dim).transpose(0, 2, 1, 3).reshape(B, L, n_heads * head_dim)
    return out


def self_attention_kernel(
    x,
    q_weight, q_bias,
    k_weight, k_bias,
    v_weight, v_bias,
    out_weight, out_bias,
    n_heads,
    head_dim,
    backend="naive",
):
    """Bidirectional self-attention over image latent tokens.

    Weights are (hidden, hidden); biases are (hidden,) or None.
    ``backend`` selects the attention core: "naive" (hand-rolled softmax,
    default, CPU-testable) or "cte" (nki-library ``attention_cte``, device-only).
    """
    q = np.matmul(x, q_weight)
    k = np.matmul(x, k_weight)
    v = np.matmul(x, v_weight)
    if q_bias is not None:
        q = q + q_bias
        k = k + k_bias
        v = v + v_bias

    if backend == "cte":
        out = _mha_cte(q, k, v, n_heads, head_dim)
    else:
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
