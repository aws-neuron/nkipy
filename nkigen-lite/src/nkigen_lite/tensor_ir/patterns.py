"""
Tensor layout solver graph patterns for ML workloads.

Uses nkigen_lite.tensor_ir.ir.Builder for graph construction (auto shape inference,
unified reduce opcode, broadcasting validation).

Pattern builders for: RMSNorm, Softmax, FFN, Attention, LayerNorm, GQA,
RoPE, residual connections, KV-cache, SwiGLU, projections, cross-entropy,
DeltaNet, cross-lane reduce,
fused scale/bias/activation, matmul+epilogue, and rank-change examples.
"""
from __future__ import annotations

from nkigen_lite.core import Graph
from nkigen_lite.tensor_ir.ir import Builder


def _graph(b: Builder) -> Graph:
    return b.graph


# ---------------------------------------------------------------------------
# Normalization patterns
# ---------------------------------------------------------------------------


def build_rmsnorm(shape: tuple[int, ...]) -> Graph:
    b = Builder(f"rmsnorm_{shape}")
    x = b.add_input("x", shape)
    w = b.add_input("w", (shape[-1],))

    sq = b.mul(x, x)
    mean_sq = b.reduce(sq, axis=-1, kind="mean", keepdims=True)
    eps = b.constant(1e-5, mean_sq.type.shape)
    added = b.add(mean_sq, eps)
    rstd = b.rsqrt(added)
    normed = b.mul(x, rstd)
    out = b.mul(normed, w)
    b.set_outputs({"output": out})
    return _graph(b)


def build_layernorm(shape: tuple[int, ...]) -> Graph:
    b = Builder(f"layernorm_{shape}")
    x = b.add_input("x", shape)
    gamma = b.add_input("gamma", (shape[-1],))
    beta = b.add_input("beta", (shape[-1],))

    mean = b.reduce(x, axis=-1, kind="mean", keepdims=True)
    centered = b.sub(x, mean)
    sq = b.mul(centered, centered)
    var = b.reduce(sq, axis=-1, kind="mean", keepdims=True)
    eps = b.constant(1e-5, var.type.shape)
    var_eps = b.add(var, eps)
    rstd = b.rsqrt(var_eps)
    normed = b.mul(centered, rstd)
    scaled = b.mul(normed, gamma)
    out = b.add(scaled, beta)
    out.name = "output_out"
    b.set_outputs({"output": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Softmax / cross-entropy
# ---------------------------------------------------------------------------


def build_softmax(shape: tuple[int, ...]) -> Graph:
    b = Builder(f"softmax_{shape}")
    x = b.add_input("x", shape)

    max_v = b.reduce(x, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(x, max_v)
    exp_v = b.exp(shifted)
    sum_v = b.reduce(exp_v, axis=-1, kind="sum", keepdims=True)
    inv = b.reciprocal(sum_v)
    out = b.mul(exp_v, inv)
    b.set_outputs({"probs": out})
    return _graph(b)


def build_cross_entropy_loss(B: int, S: int, V: int) -> Graph:
    b = Builder(f"ce_loss_B{B}_S{S}_V{V}")
    logits = b.add_input("logits", (B, S, V))

    max_v = b.reduce(logits, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(logits, max_v)
    exp_v = b.exp(shifted)
    sum_v = b.reduce(exp_v, axis=-1, kind="sum", keepdims=True)
    log_sum = b.log(sum_v)
    log_softmax = b.sub(shifted, log_sum)
    log_softmax.name = "log_softmax_out"
    b.set_outputs({"log_softmax": log_softmax})
    return _graph(b)


# ---------------------------------------------------------------------------
# FFN / gating patterns
# ---------------------------------------------------------------------------


def build_ffn(shape: tuple[int, ...], intermediate: int = 512) -> Graph:
    D = shape[-1]
    b = Builder(f"ffn_{shape}")
    x = b.add_input("x", shape)
    gate_up_w = b.add_input("gate_up_w", (D, intermediate * 2))
    down_w = b.add_input("down_w", (intermediate, D))

    mm1 = b.matmul(x, gate_up_w)
    # Rename for test compatibility
    mm1.name = "mm_gate_up_out"

    half_shape = shape[:-1] + (intermediate,)
    starts_gate = (0,) * len(shape)
    stops_gate = shape[:-1] + (intermediate,)
    gate = b.slice(mm1, starts_gate, stops_gate)

    starts_up = (0,) * (len(shape) - 1) + (intermediate,)
    stops_up = shape[:-1] + (intermediate * 2,)
    up = b.slice(mm1, starts_up, stops_up)

    sig = b.sigmoid(gate)
    silu = b.mul(gate, sig)
    gated = b.mul(silu, up)

    out = b.matmul(gated, down_w)
    b.set_outputs({"output": out})
    return _graph(b)


def build_swiglu_gate(shape: tuple[int, ...], intermediate: int) -> Graph:
    D = shape[-1]
    b = Builder(f"swiglu_{shape}_I{intermediate}")
    x = b.add_input("x", shape)
    W_gate = b.add_input("W_gate", (D, intermediate))
    W_up = b.add_input("W_up", (D, intermediate))

    gate_proj = b.matmul(x, W_gate)
    gate_proj.name = "gate_proj_out"
    up_proj = b.matmul(x, W_up)

    sig = b.sigmoid(gate_proj)
    silu = b.mul(gate_proj, sig)
    out = b.mul(silu, up_proj)
    b.set_outputs({"gated": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Attention patterns
# ---------------------------------------------------------------------------


def build_attention(shape: tuple[int, ...]) -> Graph:
    """shape = (..., S, D)"""
    b = Builder(f"attention_{shape}")
    q = b.add_input("q", shape)
    k = b.add_input("k", shape)
    v = b.add_input("v", shape)

    rank = len(shape)
    S = shape[-2]
    D = shape[-1]

    perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
    kt = b.transpose(k, perm)

    scores = b.matmul(q, kt)
    scores.name = "scores_out"

    scaled = b.mul(scores, scores)

    max_v = b.reduce(scaled, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(scaled, max_v)
    exp_v = b.exp(shifted)
    sum_v = b.reduce(exp_v, axis=-1, kind="sum", keepdims=True)
    inv = b.reciprocal(sum_v)
    probs = b.mul(exp_v, inv)

    out = b.matmul(probs, v)
    out.name = "output_out"
    b.set_outputs({"output": out})
    return _graph(b)


def build_gqa_attention(B: int, H_q: int, H_kv: int, S: int, D: int) -> Graph:
    groups = H_q // H_kv
    b = Builder(f"gqa_B{B}_Hq{H_q}_Hkv{H_kv}_S{S}_D{D}")
    q = b.add_input("q", (B, H_q, S, D))
    k = b.add_input("k", (B, H_kv, S, D))
    v = b.add_input("v", (B, H_kv, S, D))

    # Expand KV heads: (B, H_kv, S, D) → (B, H_kv, 1, S, D) → broadcast → (B, H_kv, groups, S, D) → reshape → (B, H_q, S, D)
    k_5d = b.reshape(k, (B, H_kv, 1, S, D))
    k_bcast = b.broadcast_to(k_5d, (B, H_kv, groups, S, D))
    k_expanded = b.reshape(k_bcast, (B, H_q, S, D))

    v_5d = b.reshape(v, (B, H_kv, 1, S, D))
    v_bcast = b.broadcast_to(v_5d, (B, H_kv, groups, S, D))
    v_expanded = b.reshape(v_bcast, (B, H_q, S, D))

    kt = b.transpose(k_expanded, (0, 1, 3, 2))
    scores = b.matmul(q, kt)
    scores.name = "scores_out"

    max_v = b.reduce(scores, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(scores, max_v)
    exp_v = b.exp(shifted)
    sum_v = b.reduce(exp_v, axis=-1, kind="sum", keepdims=True)
    inv = b.reciprocal(sum_v)
    probs = b.mul(exp_v, inv)

    out = b.matmul(probs, v_expanded)
    out.name = "output_out"
    b.set_outputs({"output": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Position encoding / sequence ops
# ---------------------------------------------------------------------------


def build_rope(shape: tuple[int, ...]) -> Graph:
    D = shape[-1]
    half_D = D // 2
    b = Builder(f"rope_{shape}")
    x = b.add_input("x", shape)
    cos = b.add_input("cos", shape[:-1] + (half_D,))
    sin = b.add_input("sin", shape[:-1] + (half_D,))

    rank = len(shape)
    starts_x1 = (0,) * rank
    stops_x1 = shape[:-1] + (half_D,)
    x1 = b.slice(x, starts_x1, stops_x1)

    starts_x2 = (0,) * (rank - 1) + (half_D,)
    stops_x2 = shape
    x2 = b.slice(x, starts_x2, stops_x2)

    x1_cos = b.mul(x1, cos)
    x2_sin = b.mul(x2, sin)
    out1 = b.sub(x1_cos, x2_sin)

    x2_cos = b.mul(x2, cos)
    x1_sin = b.mul(x1, sin)
    out2 = b.add(x2_cos, x1_sin)

    result = b.concat([out1, out2], axis=-1)
    result.name = "concat_rope_out"
    b.set_outputs({"rope": result})
    return _graph(b)




# ---------------------------------------------------------------------------
# Residual / projection patterns
# ---------------------------------------------------------------------------


def build_residual_add(shape: tuple[int, ...]) -> Graph:
    D = shape[-1]
    b = Builder(f"residual_{shape}")
    x = b.add_input("x", shape)
    W = b.add_input("W", (D, D))

    proj = b.matmul(x, W)
    act = b.gelu(proj)
    out = b.add(x, act)
    out.name = "residual_add_out"
    b.set_outputs({"residual": out})
    return _graph(b)


def build_multi_head_projection(B: int, S: int, D: int, H: int) -> Graph:
    D_h = D // H
    b = Builder(f"mhp_B{B}_S{S}_D{D}_H{H}")
    x = b.add_input("x", (B, S, D))
    W_qkv = b.add_input("W_qkv", (D, 3 * D))

    qkv = b.matmul(x, W_qkv)
    qkv.name = "qkv_proj_out"

    starts_q = (0, 0, 0)
    stops_q = (B, S, D)
    q = b.slice(qkv, starts_q, stops_q)

    starts_k = (0, 0, D)
    stops_k = (B, S, 2 * D)
    k = b.slice(qkv, starts_k, stops_k)

    starts_v = (0, 0, 2 * D)
    stops_v = (B, S, 3 * D)
    v = b.slice(qkv, starts_v, stops_v)

    q_split = b.reshape(q, (B, S, H, D_h))
    q_mh = b.transpose(q_split, (0, 2, 1, 3))
    q_mh.name = "q_reshape_out"
    k_split = b.reshape(k, (B, S, H, D_h))
    k_mh = b.transpose(k_split, (0, 2, 1, 3))
    v_split = b.reshape(v, (B, S, H, D_h))
    v_mh = b.transpose(v_split, (0, 2, 1, 3))

    b.set_outputs({"q": q_mh, "k": k_mh, "v": v_mh})
    return _graph(b)


def build_output_projection(B: int, H: int, S: int, D_h: int, D: int) -> Graph:
    b = Builder(f"out_proj_B{B}_H{H}_S{S}_Dh{D_h}_D{D}")
    attn_out = b.add_input("attn_out", (B, H, S, D_h))
    W_o = b.add_input("W_o", (D, D))

    reshaped = b.reshape(attn_out, (B, S, D))
    out = b.matmul(reshaped, W_o)
    out.name = "out_proj_out"
    b.set_outputs({"output": out})
    return _graph(b)


def build_full_attention(B: int, S: int, D: int, H: int) -> Graph:
    """Full multi-head attention: QKV projection → attention → output projection.

    x @ W_qkv → slice → reshape+transpose → Q@K^T → softmax → @V
    → transpose+reshape → @ W_o → output
    """
    D_h = D // H
    b = Builder(f"full_mha_B{B}_S{S}_D{D}_H{H}")

    x = b.add_input("x", (B, S, D))
    W_qkv = b.add_input("W_qkv", (D, 3 * D))
    W_o = b.add_input("W_o", (D, D))

    # --- QKV projection ---
    qkv = b.matmul(x, W_qkv)
    qkv.name = "qkv_proj"
    q_flat = b.slice(qkv, (0, 0, 0), (B, S, D))
    k_flat = b.slice(qkv, (0, 0, D), (B, S, 2 * D))
    v_flat = b.slice(qkv, (0, 0, 2 * D), (B, S, 3 * D))

    # --- Multi-head reshape: (B,S,D) → (B,S,H,D_h) → (B,H,S,D_h) ---
    q = b.transpose(b.reshape(q_flat, (B, S, H, D_h)), (0, 2, 1, 3))
    q.name = "q_heads"
    k = b.transpose(b.reshape(k_flat, (B, S, H, D_h)), (0, 2, 1, 3))
    k.name = "k_heads"
    v = b.transpose(b.reshape(v_flat, (B, S, H, D_h)), (0, 2, 1, 3))
    v.name = "v_heads"

    # --- Attention: Q @ K^T → softmax → @ V ---
    kt = b.transpose(k, (0, 1, 3, 2))
    scores = b.matmul(q, kt)
    scores.name = "attn_scores"

    max_s = b.reduce(scores, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(scores, max_s)
    exp_s = b.exp(shifted)
    sum_s = b.reduce(exp_s, axis=-1, kind="sum", keepdims=True)
    probs = b.mul(exp_s, b.reciprocal(sum_s))
    probs.name = "attn_probs"

    attn_out = b.matmul(probs, v)
    attn_out.name = "attn_out"

    # --- Output projection: (B,H,S,D_h) → (B,S,H,D_h) → (B,S,D) → @ W_o ---
    attn_t = b.transpose(attn_out, (0, 2, 1, 3))
    attn_flat = b.reshape(attn_t, (B, S, D))
    attn_flat.name = "attn_flat"
    out = b.matmul(attn_flat, W_o)
    out.name = "out_proj"

    b.set_outputs({"output": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# KV-cache
# ---------------------------------------------------------------------------


def build_kv_cache_update(B: int, H: int, S_cached: int, S_new: int, D: int) -> Graph:
    b = Builder(f"kv_cache_B{B}_H{H}_S{S_cached}+{S_new}_D{D}")
    cached_k = b.add_input("cached_k", (B, H, S_cached, D))
    new_k = b.add_input("new_k", (B, H, S_new, D))

    out = b.concat([cached_k, new_k], axis=2)
    b.set_outputs({"kv_concat": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Elementwise / activation patterns
# ---------------------------------------------------------------------------


def build_fused_scale_bias_activation(shape: tuple[int, ...]) -> Graph:
    b = Builder(f"fused_scale_bias_act_{shape}")
    x = b.add_input("x", shape)
    scale = b.add_input("scale", (shape[-1],))
    bias = b.add_input("bias", (shape[-1],))

    scaled = b.mul(x, scale)
    biased = b.add(scaled, bias)
    out = b.gelu(biased)
    b.set_outputs({"activated": out})
    return _graph(b)


def build_matmul_with_epilogue(shape: tuple[int, ...], N: int = 256) -> Graph:
    D = shape[-1]
    b = Builder(f"matmul_epilogue_{shape}")
    x = b.add_input("x", shape)
    W = b.add_input("W", (D, N))
    bias = b.add_input("bias", (N,))

    mm = b.matmul(x, W)
    mm.name = "linear_out"
    biased = b.add(mm, bias)
    out = b.relu(biased)
    out.name = "relu_out"
    b.set_outputs({"output": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Reduction patterns
# ---------------------------------------------------------------------------


def build_cross_lane_reduce(shape: tuple[int, ...]) -> Graph:
    b = Builder(f"cross_lane_reduce_{shape}")
    x = b.add_input("x", shape)

    out = b.reduce(x, axis=0, kind="sum", keepdims=True)
    b.set_outputs({"p_reduce": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# QK normalization
# ---------------------------------------------------------------------------


def build_qk_norm(B: int, S: int, H: int, D: int) -> Graph:
    """Per-head RMSNorm applied to Q and K after projection (Qwen3-style)."""
    b = Builder(f"qk_norm_B{B}_S{S}_H{H}_D{D}")
    q = b.add_input("q", (B, S, H, D))
    k = b.add_input("k", (B, S, H, D))
    q_norm_w = b.add_input("q_norm_w", (D,))
    k_norm_w = b.add_input("k_norm_w", (D,))

    # RMSNorm on Q: norm over head_dim (last axis)
    q_sq = b.mul(q, q)
    q_mean_sq = b.reduce(q_sq, axis=-1, kind="mean", keepdims=True)
    q_eps = b.constant(1e-5, q_mean_sq.type.shape)
    q_rstd = b.rsqrt(b.add(q_mean_sq, q_eps))
    q_normed = b.mul(q, q_rstd)
    q_out = b.mul(q_normed, q_norm_w)

    # RMSNorm on K: norm over head_dim (last axis)
    k_sq = b.mul(k, k)
    k_mean_sq = b.reduce(k_sq, axis=-1, kind="mean", keepdims=True)
    k_eps = b.constant(1e-5, k_mean_sq.type.shape)
    k_rstd = b.rsqrt(b.add(k_mean_sq, k_eps))
    k_normed = b.mul(k, k_rstd)
    k_out = b.mul(k_normed, k_norm_w)

    b.set_outputs({"q_normed": q_out, "k_normed": k_out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Full transformer layer
# ---------------------------------------------------------------------------


def build_transformer_layer(B: int, S: int, D: int, H: int, intermediate: int) -> Graph:
    """Full transformer block: norm → attention → residual → norm → FFN → residual."""
    D_h = D // H
    b = Builder(f"transformer_B{B}_S{S}_D{D}_H{H}_I{intermediate}")
    x = b.add_input("x", (B, S, D))
    attn_norm_w = b.add_input("attn_norm_w", (D,))
    W_qkv = b.add_input("W_qkv", (D, 3 * D))
    W_o = b.add_input("W_o", (D, D))
    ffn_norm_w = b.add_input("ffn_norm_w", (D,))
    gate_up_w = b.add_input("gate_up_w", (D, intermediate * 2))
    down_w = b.add_input("down_w", (intermediate, D))

    # --- Pre-attention RMSNorm ---
    sq = b.mul(x, x)
    mean_sq = b.reduce(sq, axis=-1, kind="mean", keepdims=True)
    eps1 = b.constant(1e-5, mean_sq.type.shape)
    rstd1 = b.rsqrt(b.add(mean_sq, eps1))
    norm_x = b.mul(b.mul(x, rstd1), attn_norm_w)

    # --- QKV projection + reshape to multi-head ---
    qkv = b.matmul(norm_x, W_qkv)
    qkv.name = "qkv_proj_out"

    q = b.slice(qkv, (0, 0, 0), (B, S, D))
    k = b.slice(qkv, (0, 0, D), (B, S, 2 * D))
    v = b.slice(qkv, (0, 0, 2 * D), (B, S, 3 * D))

    q_mh = b.reshape(q, (B, H, S, D_h))
    k_mh = b.reshape(k, (B, H, S, D_h))
    v_mh = b.reshape(v, (B, H, S, D_h))

    # --- Attention: Q @ K^T, softmax, @ V ---
    kt = b.transpose(k_mh, (0, 1, 3, 2))
    scores = b.matmul(q_mh, kt)
    scores.name = "attn_scores_out"

    max_v = b.reduce(scores, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(scores, max_v)
    exp_v = b.exp(shifted)
    sum_v = b.reduce(exp_v, axis=-1, kind="sum", keepdims=True)
    inv = b.reciprocal(sum_v)
    probs = b.mul(exp_v, inv)

    attn_out = b.matmul(probs, v_mh)

    # --- Output projection ---
    attn_flat = b.reshape(attn_out, (B, S, D))
    h1 = b.matmul(attn_flat, W_o)

    # --- Residual after attention ---
    z = b.add(x, h1)

    # --- Pre-FFN RMSNorm ---
    z_sq = b.mul(z, z)
    z_mean_sq = b.reduce(z_sq, axis=-1, kind="mean", keepdims=True)
    eps2 = b.constant(1e-5, z_mean_sq.type.shape)
    rstd2 = b.rsqrt(b.add(z_mean_sq, eps2))
    norm_z = b.mul(b.mul(z, rstd2), ffn_norm_w)

    # --- SwiGLU FFN ---
    mm1 = b.matmul(norm_z, gate_up_w)
    gate = b.slice(mm1, (0, 0, 0), (B, S, intermediate))
    up = b.slice(mm1, (0, 0, intermediate), (B, S, intermediate * 2))
    sig = b.sigmoid(gate)
    silu = b.mul(gate, sig)
    gated = b.mul(silu, up)
    ffn_out = b.matmul(gated, down_w)

    # --- Residual after FFN ---
    out = b.add(z, ffn_out)
    out.name = "transformer_out"
    b.set_outputs({"output": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Linear attention (DeltaNet)
# ---------------------------------------------------------------------------


def build_linear_attention_deltanet(B: int = 1, H: int = 4, L: int = 64, D: int = 32) -> Graph:
    b = Builder(f"deltanet_B{B}_H{H}_L{L}_D{D}")

    q = b.add_input("q", (B, H, L, D))
    k = b.add_input("k", (B, H, L, D))
    v = b.add_input("v", (B, H, L, D))
    beta_logits = b.add_input("beta_logits", (B, H, L))

    k_sq = b.mul(k, k)
    k_sum = b.reduce(k_sq, axis=-1, kind="sum", keepdims=True)
    k_inv_norm = b.rsqrt(k_sum)
    k_normed = b.mul(k, k_inv_norm)

    beta_expanded = b.reshape(beta_logits, (B, H, L, 1))
    beta = b.sigmoid(beta_expanded)

    gated_v = b.mul(v, beta)
    out = b.mul(q, gated_v)
    b.set_outputs({"qkv_interact": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# Elementwise rank merge/split examples
# ---------------------------------------------------------------------------


def build_elementwise_rank_change(B: int = 2, S: int = 64, D: int = 128, N: int = 256, O: int = 32) -> Graph:
    b = Builder(f"elementwise_rank_change_B{B}_S{S}_D{D}_N{N}_O{O}")

    x = b.add_input("x", (B, S, D))
    W = b.add_input("W", (D, N))
    V = b.add_input("V", (B, N, O))

    proj = b.matmul(x, W)
    act = b.relu(proj)
    act2 = b.mul(act, act)
    out = b.matmul(act2, V)
    b.set_outputs({"output": out})
    return _graph(b)


def build_elementwise_merge_for_utilization(B: int = 4, S: int = 32, D: int = 64, N: int = 128) -> Graph:
    b = Builder(f"elementwise_merge_B{B}_S{S}_D{D}_N{N}")

    x = b.add_input("x", (B, S, D))
    W = b.add_input("W", (D, N))
    proj = b.matmul(x, W)

    a = b.gelu(proj)
    bias = b.add_input("bias", (N,))
    added = b.add(a, bias)
    scale = b.add_input("scale", (N,))
    c = b.mul(added, scale)
    d = b.relu(c)

    W2 = b.add_input("W2", (N, D))
    out = b.matmul(d, W2)
    b.set_outputs({"output": out})
    return _graph(b)


def build_elementwise_split_for_batched_mm(S: int = 128, D: int = 128, N: int = 64, B_out: int = 2, O: int = 32) -> Graph:
    assert S % B_out == 0
    S_split = S // B_out
    b = Builder(f"elementwise_split_S{S}_D{D}_N{N}_B{B_out}_O{O}")

    x = b.add_input("x", (S, D))
    W = b.add_input("W", (D, N))
    proj = b.matmul(x, W)

    reshaped = b.reshape(proj, (B_out, S_split, N))
    act = b.relu(reshaped)
    act2 = b.gelu(act)

    K = b.add_input("K", (B_out, N, O))
    out = b.matmul(act2, K)
    b.set_outputs({"output": out})
    return _graph(b)


# ---------------------------------------------------------------------------
# GPT-2 layer (LayerNorm + MHA + FFN with GELU)
# ---------------------------------------------------------------------------


def build_gpt2_layer(B: int, S: int, D: int, H: int) -> Graph:
    """GPT-2 transformer block: LN → MHA → residual → LN → FFN(GELU) → residual.

    Uses pre-norm LayerNorm (not RMSNorm), GELU activation (not SwiGLU),
    and standard 4×D intermediate size.
    """
    D_h = D // H
    intermediate = 4 * D
    b = Builder(f"gpt2_B{B}_S{S}_D{D}_H{H}")

    x = b.add_input("x", (B, S, D))
    ln1_gamma = b.add_input("ln1_gamma", (D,))
    ln1_beta = b.add_input("ln1_beta", (D,))
    W_qkv = b.add_input("W_qkv", (D, 3 * D))
    W_o = b.add_input("W_o", (D, D))
    ln2_gamma = b.add_input("ln2_gamma", (D,))
    ln2_beta = b.add_input("ln2_beta", (D,))
    W_fc = b.add_input("W_fc", (D, intermediate))
    W_proj = b.add_input("W_proj", (intermediate, D))

    # --- LayerNorm 1 ---
    mean1 = b.reduce(x, axis=-1, kind="mean", keepdims=True)
    centered1 = b.sub(x, mean1)
    var1 = b.reduce(b.mul(centered1, centered1), axis=-1, kind="mean", keepdims=True)
    eps1 = b.constant(1e-5, var1.type.shape)
    rstd1 = b.rsqrt(b.add(var1, eps1))
    norm1 = b.add(b.mul(b.mul(centered1, rstd1), ln1_gamma), ln1_beta)
    norm1.name = "ln1_out"

    # --- QKV projection + multi-head split ---
    qkv = b.matmul(norm1, W_qkv)
    qkv.name = "qkv_out"
    q = b.slice(qkv, (0, 0, 0), (B, S, D))
    k = b.slice(qkv, (0, 0, D), (B, S, 2 * D))
    v = b.slice(qkv, (0, 0, 2 * D), (B, S, 3 * D))
    q_mh = b.reshape(q, (B, H, S, D_h))
    k_mh = b.reshape(k, (B, H, S, D_h))
    v_mh = b.reshape(v, (B, H, S, D_h))

    # --- Attention: Q @ K^T → softmax → @ V ---
    kt = b.transpose(k_mh, (0, 1, 3, 2))
    scores = b.matmul(q_mh, kt)
    scores.name = "attn_scores"

    # Softmax
    max_s = b.reduce(scores, axis=-1, kind="max", keepdims=True)
    shifted = b.sub(scores, max_s)
    exp_s = b.exp(shifted)
    sum_s = b.reduce(exp_s, axis=-1, kind="sum", keepdims=True)
    probs = b.mul(exp_s, b.reciprocal(sum_s))
    probs.name = "attn_probs"

    attn_out = b.matmul(probs, v_mh)
    attn_out.name = "attn_out"

    # --- Output projection + residual ---
    attn_flat = b.reshape(attn_out, (B, S, D))
    h = b.matmul(attn_flat, W_o)
    h.name = "attn_proj_out"
    residual1 = b.add(x, h)
    residual1.name = "residual1"

    # --- LayerNorm 2 ---
    mean2 = b.reduce(residual1, axis=-1, kind="mean", keepdims=True)
    centered2 = b.sub(residual1, mean2)
    var2 = b.reduce(b.mul(centered2, centered2), axis=-1, kind="mean", keepdims=True)
    eps2 = b.constant(1e-5, var2.type.shape)
    rstd2 = b.rsqrt(b.add(var2, eps2))
    norm2 = b.add(b.mul(b.mul(centered2, rstd2), ln2_gamma), ln2_beta)
    norm2.name = "ln2_out"

    # --- FFN: linear → GELU → linear ---
    fc1 = b.matmul(norm2, W_fc)
    fc1.name = "fc1_out"
    act = b.gelu(fc1)
    fc2 = b.matmul(act, W_proj)
    fc2.name = "fc2_out"

    # --- Residual ---
    out = b.add(residual1, fc2)
    out.name = "gpt2_out"
    b.set_outputs({"output": out})
    return _graph(b)
