"""T5-XXL encoder for PixArt text conditioning, on Trainium.

The encoder is a 24-layer bidirectional transformer (d_model=4096, 64 heads x 64,
d_ff=10240, gated-GELU). It differs from the DiT/LLM attention in three ways
that matter for correctness:

* **No attention scaling** — T5 folds the 1/sqrt(d) into the weight init, so
  scores are raw q@k^T.
* **RMSNorm without mean subtraction and without bias** (``t5_layernorm``).
* **Relative position bias** — a learned (num_buckets, n_heads) table added to
  the attention scores, shared across all layers (only layer 0 owns it). Bucket
  indices depend only on (i - j) for a fixed sequence length, so they are a
  comptime constant here and the bias is gathered with a one-hot matmul.

Token embedding lookup is done on host (a memory gather, as in the qwen3
example); this kernel takes the embedded tokens and runs all 24 layers.
"""

import numpy as np
from nkipy.core import tensor_apis


# ── relative position bias (comptime) ──────────────────────────────────────


def _relative_position_bucket(relative_position, bidirectional=True,
                              num_buckets=32, max_distance=128):
    """numpy port of T5Attention._relative_position_bucket (comptime)."""
    relative_buckets = np.zeros_like(relative_position)
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(np.int64) * num_buckets
        relative_position = np.abs(relative_position)
    else:
        relative_position = -np.minimum(relative_position, np.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    # log() of the (masked-out) small positions can hit 0; guard to avoid a
    # benign divide-by-zero warning — those entries are discarded by np.where.
    safe = np.maximum(relative_position, max_exact).astype(np.float64)
    rel_if_large = max_exact + (
        np.log(safe / max_exact)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int64)
    rel_if_large = np.minimum(rel_if_large, np.full_like(rel_if_large, num_buckets - 1))
    relative_buckets += np.where(is_small, relative_position, rel_if_large)
    return relative_buckets


def position_bias(bias_table, seq_len, n_heads, num_buckets=32, max_distance=128):
    """Build (1, n_heads, seq_len, seq_len) relative position bias.

    ``bias_table`` is the runtime (num_buckets, n_heads) weight. The bucket
    index matrix is a comptime constant; the gather is done as a one-hot matmul
    so it lowers cleanly.
    """
    ctx = np.arange(seq_len)[:, None]
    mem = np.arange(seq_len)[None, :]
    rel = mem - ctx  # (seq, seq) = memory - query
    buckets = _relative_position_bucket(rel, bidirectional=True,
                                        num_buckets=num_buckets,
                                        max_distance=max_distance)  # (seq, seq)
    # one-hot (seq*seq, num_buckets) @ table (num_buckets, n_heads).
    # The one-hot is a comptime constant; promote it to a runtime tensor so it
    # composes with the runtime bias_table (matches the qwen3 mask pattern).
    onehot = np.zeros((seq_len * seq_len, num_buckets), dtype=np.float32)
    onehot[np.arange(seq_len * seq_len), buckets.reshape(-1)] = 1
    onehot = tensor_apis.constant(onehot.astype(bias_table.dtype))
    bias = np.matmul(onehot, bias_table)  # (seq*seq, n_heads)
    bias = bias.reshape(seq_len, seq_len, n_heads).transpose(2, 0, 1)  # (h, sq, sk)
    return np.expand_dims(bias, axis=0)  # (1, h, sq, sk)


# ── primitives ──────────────────────────────────────────────────────────────


def t5_layernorm(x, weight, eps=1e-6):
    """T5 RMSNorm: no mean subtraction, no bias. Computed in fp32."""
    dtype = x.dtype
    xf = x.astype(np.float32)
    var = np.mean(np.square(xf), axis=-1, keepdims=True)
    xf = xf / np.sqrt(var + eps)
    return (xf.astype(dtype)) * weight


def _gelu_new(x):
    xf = x.astype(np.float32)
    inner = np.sqrt(2.0 / np.pi) * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + np.tanh(inner))).astype(x.dtype)


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def t5_self_attention(x, q_w, k_w, v_w, o_w, pos_bias, mask_bias, n_heads, head_dim):
    """Bidirectional self-attention, NO 1/sqrt(d) scaling (T5 convention).

    q_w/k_w/v_w/o_w are (d_model, d_model); ``x`` is (B, S, d_model).
    pos_bias: (1, n_heads, S, S). mask_bias: (B, 1, 1, S).
    """
    B, S, _ = x.shape
    q = np.matmul(x, q_w).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = np.matmul(x, k_w).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = np.matmul(x, v_w).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)).astype(np.float32)  # no scaling
    scores = scores + pos_bias.astype(np.float32)
    if mask_bias is not None:
        scores = scores + mask_bias.astype(np.float32)
    weights = _softmax(scores).astype(v.dtype)

    out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, S, n_heads * head_dim)
    return np.matmul(out, o_w)


def t5_ffn(x, wi0, wi1, wo):
    """Gated-GELU FFN: wo @ (gelu(wi0 @ x) * (wi1 @ x)). No biases."""
    h_gelu = _gelu_new(np.matmul(x, wi0))
    h_lin = np.matmul(x, wi1)
    return np.matmul(h_gelu * h_lin, wo)


def t5_encoder(inputs_embeds, attention_mask, configs, **weights):
    """Run the full T5 encoder.

    Args:
        inputs_embeds: (B, S, d_model) host-gathered token embeddings.
        attention_mask: (B, S) 1 for real tokens, 0 for padding.
        configs: object with t5_* attributes (see PixArtT5Config).
        **weights: flat T5 weights, keys ``t5_b{layer}_*`` and shared
            ``t5_rel_bias``, ``t5_final_ln`` (see t5_weight_layout).
    Returns:
        (B, S, d_model) encoder hidden states.
    """
    from .t5_weight_layout import regroup_t5_weights

    shared, blocks = regroup_t5_weights(weights, configs.t5_num_layers)

    n_heads = configs.t5_num_heads
    head_dim = configs.t5_d_kv
    eps = configs.t5_eps

    B, S, _ = inputs_embeds.shape

    # relative position bias (shared across layers), comptime bucket indices
    pos_bias = position_bias(shared["t5_rel_bias"], S, n_heads)

    # padding mask -> additive bias (B, 1, 1, S)
    mask_bias = (1.0 - attention_mask.astype(np.float32)) * np.float32(-1e9)
    mask_bias = np.expand_dims(mask_bias, axis=[1, 2])

    x = inputs_embeds
    for w in blocks:
        h = t5_layernorm(x, w["ln0"], eps)
        x = x + t5_self_attention(h, w["q"], w["k"], w["v"], w["o"],
                                  pos_bias, mask_bias, n_heads, head_dim)
        h = t5_layernorm(x, w["ln1"], eps)
        x = x + t5_ffn(h, w["wi0"], w["wi1"], w["wo"])

    x = t5_layernorm(x, shared["t5_final_ln"], eps)
    return x
