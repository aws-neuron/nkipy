"""Example graphs built with tensor_ir."""

import numpy as np

from nkigen_lite.tensor_ir.ir import Builder, Value, run
from nkigen_lite.core import DType, Graph


def softmax(b: Builder, x: Value, axis: int = -1) -> Value:
    if axis < 0:
        axis = x.type.rank + axis
    m = b.reduce(x, axis=axis, kind="max", keepdims=True)
    shifted = b.sub(x, m)
    e = b.exp(shifted)
    s = b.reduce(e, axis=axis, kind="sum", keepdims=True)
    return b.div(e, s)


def layer_norm(
    b: Builder, x: Value, weight: Value, bias: Value,
    axis: int = -1, eps: float = 1e-5,
) -> Value:
    if axis < 0:
        axis = x.type.rank + axis
    mean = b.reduce(x, axis=axis, kind="mean", keepdims=True)
    centered = b.sub(x, mean)
    var = b.reduce(b.mul(centered, centered), axis=axis, kind="mean", keepdims=True)
    eps_val = b.constant(eps, var.type.shape, var.type.dtype)
    inv_std = b.rsqrt(b.add(var, eps_val))
    normed = b.mul(centered, inv_std)
    return b.add(b.mul(normed, weight), bias)


def build_rmsnorm() -> Graph:
    b = Builder("rmsnorm")
    x = b.add_input("x", (2, 128, 768), DType.F32)
    w = b.add_input("w", (768,), DType.F32)

    # x^2
    xsq = b.mul(x, x)
    # mean(x^2, axis=-1, keepdims=True)
    mean_sq = b.reduce(xsq, axis=2, keepdims=True, kind="mean")
    eps = b.constant(1e-5, mean_sq.type.shape, DType.F32)
    rstd = b.rsqrt(b.add(mean_sq, eps))
    normed = b.mul(x, rstd)       # (2,128,768) * (2,128,1) broadcasts
    out = b.mul(normed, w)          # (2,128,768) * (768,) broadcasts
    b.set_outputs({"result": out})
    return b.graph


def build_attention() -> Graph:
    B, H, S, D = 2, 8, 128, 64
    b = Builder("attention")
    q = b.add_input("q", (B, H, S, D), DType.F32)
    k = b.add_input("k", (B, H, S, D), DType.F32)
    v = b.add_input("v", (B, H, S, D), DType.F32)

    # scores = Q @ K^T / sqrt(D)
    kt = b.transpose(k, (0, 1, 3, 2))
    scores = b.matmul(q, kt)
    scale = b.constant(1.0 / (D ** 0.5), scores.type.shape, DType.F32)
    scores_scaled = b.mul(scores, scale)

    # softmax
    probs = softmax(b, scores_scaled, axis=-1)

    # out = probs @ V
    out = b.matmul(probs, v)
    b.set_outputs({"result": out})
    return b.graph


def build_tiled_sum() -> Graph:
    """Demonstrate for_loop: accumulate over 128 iterations."""
    ROWS, COLS = 4, 128

    b = Builder("tiled_sum")
    zero = b.constant(0.0, (ROWS, 1), DType.F32)

    def body(lb, _i, acc):
        one = lb.constant(1.0, (ROWS, 1), DType.F32)
        return lb.add(acc, one)

    (result,) = b.for_loop(trip_count=COLS, init=[zero], body_fn=body)
    b.set_outputs({"sum": result})
    return b.graph


def build_rope() -> Graph:
    """Rotary Position Embedding (RoPE) as used in Qwen3.

    Precomputes cos/sin frequency tables (compile-time constants in nkipy),
    then applies the rotation:
        x_out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
        x_out[..., half:] = x[..., :half] * sin + x[..., half:] * cos
    """
    B, S, H, D = 1, 16, 4, 64
    half = D // 2

    b = Builder("rope")
    xq = b.add_input("xq", (B, S, H, D), DType.F32)  # query
    # cos/sin caches: (S, half) — precomputed from freqs
    freqs_cos = b.add_input("freqs_cos", (S, half), DType.F32)
    freqs_sin = b.add_input("freqs_sin", (S, half), DType.F32)

    # broadcast cos/sin to (B, S, H, half) via reshape + broadcast
    fc = b.reshape(freqs_cos, (1, S, 1, half))
    fc = b.broadcast_to(fc, (B, S, H, half))
    fs = b.reshape(freqs_sin, (1, S, 1, half))
    fs = b.broadcast_to(fs, (B, S, H, half))

    # split query into two halves
    xq0, xq1 = b.split(xq, 2, axis=3)  # each (B, S, H, half)

    # rotate: out0 = xq0 * cos - xq1 * sin
    #         out1 = xq0 * sin + xq1 * cos
    out0 = b.sub(b.mul(xq0, fc), b.mul(xq1, fs))
    out1 = b.add(b.mul(xq0, fs), b.mul(xq1, fc))

    # reassemble
    xq_out = b.concat([out0, out1], axis=3)
    b.set_outputs({"xq_out": xq_out})
    return b.graph


def build_causal_attention() -> Graph:
    """Scaled dot-product attention with causal masking (Qwen3-style).

    Demonstrates: matmul, comparison ops, where (masking), softmax.
    """
    B, H, S, D = 1, 4, 32, 64

    b = Builder("causal_attention")
    q = b.add_input("q", (B, H, S, D), DType.F32)
    k = b.add_input("k", (B, H, S, D), DType.F32)
    v = b.add_input("v", (B, H, S, D), DType.F32)

    # scores = Q @ K^T / sqrt(D)
    kt = b.transpose(k, (0, 1, 3, 2))
    scores = b.matmul(q, kt)  # (B, H, S, S)
    scale = b.constant(1.0 / (D ** 0.5), scores.type.shape, DType.F32)
    scores = b.mul(scores, scale)

    # causal mask: mask[i,j] = true where j > i  (upper triangle)
    # build row indices (0..S) and col indices (0..S), broadcast to (S, S)
    row_idx = b.add_input("row_idx", (S, 1), DType.F32)
    col_idx = b.add_input("col_idx", (1, S), DType.F32)
    row_bc = b.broadcast_to(row_idx, (S, S))
    col_bc = b.broadcast_to(col_idx, (S, S))
    mask_2d = b.greater(col_bc, row_bc)  # True where j > i

    # broadcast mask to (B, H, S, S)
    mask_4d = b.reshape(mask_2d, (1, 1, S, S))
    mask_4d = b.broadcast_to(mask_4d, (B, H, S, S))

    # apply mask: where(mask, -1e5, scores)
    neg_inf = b.constant(-1e5, scores.type.shape, DType.F32)
    scores = b.where(mask_4d, neg_inf, scores)

    # softmax + output
    probs = softmax(b, scores, axis=-1)
    out = b.matmul(probs, v)
    b.set_outputs({"result": out})
    return b.graph


def build_qkv_proj() -> Graph:
    """QKV projection with grouped query attention split (Qwen3-style).

    Projects input x through a single weight matrix, then splits into Q, K, V
    with different head counts (GQA: fewer KV heads than Q heads).
    """
    B, S, D = 1, 16, 256
    n_heads, n_kv_heads, head_dim = 8, 2, 32
    # weight columns: Q + K + V
    q_dim = n_heads * head_dim       # 256
    k_dim = n_kv_heads * head_dim    # 64
    v_dim = n_kv_heads * head_dim    # 64
    total = q_dim + k_dim + v_dim    # 384

    b = Builder("qkv_proj")
    x = b.add_input("x", (B, S, D), DType.F32)
    w = b.add_input("w", (D, total), DType.F32)

    # single fused projection
    qkv = b.matmul(x, w)  # (B, S, 384)

    # split into Q, K, V with uneven sizes
    xq, xk, xv = b.split(qkv, [q_dim, k_dim, v_dim], axis=2)

    # reshape to per-head layout: (B, S, n_heads, head_dim)
    xq = b.reshape(xq, (B, S, n_heads, head_dim))
    xk = b.reshape(xk, (B, S, n_kv_heads, head_dim))
    xv = b.reshape(xv, (B, S, n_kv_heads, head_dim))

    b.set_outputs({"xq": xq, "xk": xk, "xv": xv})
    return b.graph


def build_feedforward() -> Graph:
    """SwiGLU feed-forward network as used in Qwen3.

    Performs:
        gate, up = split(x @ gate_up_weight, 2)
        out = (gate * sigmoid(gate)) * up   # SiLU(gate) * up
        out = out @ down_weight
    """
    B, S, D = 1, 16, 256
    intermediate = 512  # gate_up projects to 2 * intermediate

    b = Builder("feedforward")
    x = b.add_input("x", (B, S, D), DType.F32)
    gate_up_w = b.add_input("gate_up_w", (D, intermediate * 2), DType.F32)
    down_w = b.add_input("down_w", (intermediate, D), DType.F32)

    # fused gate + up projection
    mm = b.matmul(x, gate_up_w)  # (B, S, 2 * intermediate)
    gate, up = b.split(mm, 2, axis=2)  # each (B, S, intermediate)

    # SiLU(gate) = gate * sigmoid(gate)
    silu = b.mul(gate, b.sigmoid(gate))

    # gated output
    x0 = b.mul(silu, up)

    # down projection
    out = b.matmul(x0, down_w)
    b.set_outputs({"result": out})
    return b.graph


if __name__ == "__main__":
    np.random.seed(42)

    print("==== RMSNorm ====")
    g = build_rmsnorm()
    print(g.dump())
    outs = run(g, {
        "x": np.random.randn(2, 128, 768).astype(np.float32),
        "w": np.ones(768, dtype=np.float32),
    })
    r = outs["result"]
    print(f"Output shape: {r.shape}, mean: {r.mean():.6f}, std: {r.std():.6f}")

    print("\n==== Attention ====")
    g = build_attention()
    print(g.dump())
    outs = run(g, {
        "q": np.random.randn(2, 8, 128, 64).astype(np.float32),
        "k": np.random.randn(2, 8, 128, 64).astype(np.float32),
        "v": np.random.randn(2, 8, 128, 64).astype(np.float32),
    })
    r = outs["result"]
    print(f"Output shape: {r.shape}, mean: {r.mean():.6f}, std: {r.std():.6f}")

    print("\n==== Tiled Sum (for_loop) ====")
    g = build_tiled_sum()
    print(g.dump())
    outs = run(g, {})
    print(f"Result (should be 128 everywhere): {outs['sum'].flatten()}")

    print("\n==== RoPE (Rotary Position Embedding) ====")
    g = build_rope()
    print(g.dump())
    B, S, H, D = 1, 16, 4, 64
    half = D // 2
    # Compute cos/sin frequency table (same as qwen3 rope.py)
    base = 1000000
    freqs = 1.0 / (base ** (np.arange(0, D, 2)[: half] / D))
    t = np.arange(S, dtype=np.float32)
    freqs = np.outer(t, freqs)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    xq = np.random.randn(B, S, H, D).astype(np.float32)
    outs = run(g, {"xq": xq, "freqs_cos": cos_cache, "freqs_sin": sin_cache})
    r = outs["xq_out"]
    # RoPE should preserve norms approximately
    print(f"Output shape: {r.shape}")
    print(f"Input norm:  {np.linalg.norm(xq):.4f}")
    print(f"Output norm: {np.linalg.norm(r):.4f}")

    print("\n==== Causal Attention ====")
    g = build_causal_attention()
    print(g.dump())
    B, H, S, D = 1, 4, 32, 64
    row_idx = np.arange(S, dtype=np.float32).reshape(S, 1)
    col_idx = np.arange(S, dtype=np.float32).reshape(1, S)
    outs = run(g, {
        "q": np.random.randn(B, H, S, D).astype(np.float32),
        "k": np.random.randn(B, H, S, D).astype(np.float32),
        "v": np.random.randn(B, H, S, D).astype(np.float32),
        "row_idx": row_idx,
        "col_idx": col_idx,
    })
    r = outs["result"]
    print(f"Output shape: {r.shape}, mean: {r.mean():.6f}, std: {r.std():.6f}")

    print("\n==== QKV Projection (GQA split) ====")
    g = build_qkv_proj()
    print(g.dump())
    B, S, D = 1, 16, 256
    outs = run(g, {
        "x": np.random.randn(B, S, D).astype(np.float32),
        "w": np.random.randn(D, 384).astype(np.float32) * 0.02,
    })
    print(f"Q shape: {outs['xq'].shape}  (expect (1, 16, 8, 32))")
    print(f"K shape: {outs['xk'].shape}  (expect (1, 16, 2, 32))")
    print(f"V shape: {outs['xv'].shape}  (expect (1, 16, 2, 32))")

    print("\n==== Feed-Forward (SwiGLU) ====")
    g = build_feedforward()
    print(g.dump())
    B, S, D, intermediate = 1, 16, 256, 512
    x_ff = np.random.randn(B, S, D).astype(np.float32)
    outs = run(g, {
        "x": x_ff,
        "gate_up_w": np.random.randn(D, intermediate * 2).astype(np.float32) * 0.02,
        "down_w": np.random.randn(intermediate, D).astype(np.float32) * 0.02,
    })
    r = outs["result"]
    print(f"Output shape: {r.shape}  (expect (1, 16, 256))")
    print(f"mean: {r.mean():.6f}, std: {r.std():.6f}")
