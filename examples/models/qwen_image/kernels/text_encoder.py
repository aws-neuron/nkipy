"""Qwen2.5 text-encoder forward for Qwen-Image, on device.

Qwen-Image conditions on the last hidden state of a Qwen2.5-VL model run
**text-only** (no image tokens), so the vision tower is unused and the encoder
is a standard Qwen2.5 decoder LM: 28 layers, hidden 3584, 28 query heads / 4 KV
heads (GQA), SwiGLU MLP (intermediate 18944), RMSNorm, RoPE (theta 1e6),
head_dim 128. Differences vs the qwen3 example's kernels:

* q/k/v are **separate** projections **with bias** (not a fused qkv, no QK-norm);
* we run a single **prefill** pass over the whole prompt and return the *last
  hidden state* (before the LM head) — this is an encoder, not a generator, so
  there is no KV cache / decode loop and no sampling.

Attention is causal (it is a decoder LM; diffusers runs it with the default
causal mask). Weights arrive as a flat ``**weights`` dict keyed by
``text_weight_layout.py``. RoPE cos/sin and the causal mask are comptime numpy
constants baked into the graph (same pattern as the MMDiT / qwen3 kernels).

TP: q/k/v/o and the MLP shard exactly like the MMDiT denoiser (heads /
intermediate), with an all-reduce after o_proj and down_proj. ``local_heads`` /
``local_kv_heads`` / ``all_reduce_fn`` come from the config.
"""

import numpy as np

from .rmsnorm import rmsnorm_kernel
from .softmax import softmax_kernel


def _rope_tables(seq_len, head_dim, theta, dtype):
    """Comptime cos/sin for standard (half-split) RoPE. (seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, head_dim)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)


def _apply_rope(x, cos, sin):
    """x: (B, H, S, d); cos/sin: (S, d). Half-split rotation (Qwen2 convention)."""
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    cos = cos.reshape((1, 1) + cos.shape)
    sin = sin.reshape((1, 1) + sin.shape)
    return x * cos + rot * sin


def _swiglu(x, gate_w, up_w, down_w, all_reduce_fn):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). gate/up column-parallel, down row."""
    g = np.matmul(x, gate_w)
    u = np.matmul(x, up_w)
    gf = g.astype(np.float32)
    act = (gf * (1.0 / (1.0 + np.exp(-gf)))).astype(g.dtype)
    h = act * u
    out = np.matmul(h, down_w)
    return all_reduce_fn(out)


def _layer(x, w, i, cos, sin, causal_bias, n_heads, n_kv_heads, head_dim, eps,
           local_heads, local_kv_heads, all_reduce_fn):
    """One Qwen2.5 decoder layer (pre-norm attention + pre-norm SwiGLU)."""
    B, S, _ = x.shape
    hq = local_heads if local_heads is not None else n_heads
    hkv = local_kv_heads if local_kv_heads is not None else n_kv_heads

    # ── attention ──
    h = rmsnorm_kernel(x, w[f"l{i}.attn_norm"], eps=eps)
    q = np.matmul(h, w[f"l{i}.q_w"]) + w[f"l{i}.q_b"]
    k = np.matmul(h, w[f"l{i}.k_w"]) + w[f"l{i}.k_b"]
    v = np.matmul(h, w[f"l{i}.v_w"]) + w[f"l{i}.v_b"]

    q = q.reshape(B, S, hq, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, hkv, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, hkv, head_dim).transpose(0, 2, 1, 3)

    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    # GQA: repeat KV heads
    n_rep = hq // hkv
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)

    scores = np.matmul(q, k.transpose(0, 1, 3, 2)).astype(np.float32)
    scores = scores / np.float32(np.sqrt(head_dim))
    scores = scores + causal_bias  # (1,1,S,S) comptime
    attn = softmax_kernel(scores).astype(v.dtype)
    o = np.matmul(attn, v)  # (B, hq, S, d)
    o = o.transpose(0, 2, 1, 3).reshape(B, S, hq * head_dim)

    o = np.matmul(o, w[f"l{i}.o_w"])
    o = all_reduce_fn(o)
    x = x + o

    # ── SwiGLU MLP ──
    h = rmsnorm_kernel(x, w[f"l{i}.mlp_norm"], eps=eps)
    ff = _swiglu(h, w[f"l{i}.gate_w"], w[f"l{i}.up_w"], w[f"l{i}.down_w"],
                 all_reduce_fn=all_reduce_fn)
    x = x + ff
    return x


def text_encoder_forward(hidden, configs, **weights):
    """Prefill forward returning the last hidden state (B, S, hidden).

    Args:
        hidden: (B, S, hidden) token embeddings (host does the embedding lookup
            from the input ids; the embedding table is huge and the lookup is
            data-dependent, so it stays on host).
        configs: text-encoder config (see TextEncoderConfig).
    Returns:
        (B, S, hidden) last hidden state (post final norm), matching
        ``hidden_states[-1]`` of the diffusers encoder.
    """
    n_heads = configs.num_heads
    n_kv_heads = configs.num_kv_heads
    head_dim = configs.head_dim
    eps = configs.rms_norm_eps
    tp = getattr(configs, "tp_size", 1) or 1
    local_heads = n_heads // tp if tp > 1 else None
    local_kv_heads = max(1, n_kv_heads // tp) if tp > 1 else None
    all_reduce_fn = configs.all_reduce_fn

    B, S, _ = hidden.shape
    cos, sin = _rope_tables(S, head_dim, configs.rope_theta, configs.dtype)
    # causal mask (comptime): upper triangle -> large negative
    causal = np.triu(np.ones((S, S), dtype=np.float32) * -1e9, k=1)
    causal_bias = causal.reshape(1, 1, S, S)

    x = hidden
    for i in range(configs.num_layers):
        x = _layer(x, weights, i, cos, sin, causal_bias, n_heads, n_kv_heads,
                   head_dim, eps, local_heads, local_kv_heads, all_reduce_fn)

    x = rmsnorm_kernel(x, weights["final_norm"], eps=eps)
    return x
