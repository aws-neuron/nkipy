import numpy as np

from nkipy.core import tensor_apis


def clamped_swiglu(gate, up, alpha, limit):
    """gpt-oss clamped SwiGLU gating.

    Mirrors GptOssExperts._apply_gate:
        gate = clamp(gate, max=limit)
        up   = clamp(up, -limit, limit)
        glu  = gate * sigmoid(alpha * gate)
        out  = (up + 1) * glu
    """
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    glu = gate * (1.0 / (1.0 + np.exp(-alpha * gate)))
    return (up + 1.0) * glu


def feedforward_kernel(
    x, gate_up_weight, gate_up_bias, down_weight, down_bias, alpha, limit
):
    """Single-expert feed-forward with clamped SwiGLU and biases.

    `gate_up_weight`/`gate_up_bias` are pre-arranged at weight-prep time so the
    gate half comes first and the up half second (de-interleaved from the HF
    layout), letting us split in half here.
    """
    mm_gup = np.matmul(x, gate_up_weight) + gate_up_bias

    xg, x_up = np.split(mm_gup, 2, axis=-1)

    gated = clamped_swiglu(xg, x_up, alpha, limit)

    return np.matmul(gated, down_weight) + down_bias


def moe_batched(
    norm_z,
    router_weight,
    router_bias,
    gate_up_weight,
    gate_up_bias,
    down_weight,
    down_bias,
    top_k,
    alpha,
    limit,
):
    """Batched MoE over all (B*L) tokens with a single gather + batched matmul.

    Replaces the per-(token, expert) Python loop. For N = B*L tokens and top_k
    selected experts, all N*top_k expert feed-forwards are expressed as batched
    ops so they trace into a handful of large HLO ops instead of N*top_k tiny
    GEMV chains.

    Shapes (E = num_experts, D = hidden, I = intermediate per rank):
        norm_z:         (B, L, D)
        gate_up_weight: (E, D, 2I)   gate_up_bias: (E, 2I)
        down_weight:    (E, I, D)    down_bias:    (E, D)
    Returns (B, L, D) expert output (pre-residual, pre-all-reduce).
    """
    B, L, D = norm_z.shape
    N = B * L
    flat = norm_z.reshape(N, D)  # (N, D)

    # Router: top_k on raw logits (+bias), then softmax over the selected logits.
    router_logits = np.matmul(flat, router_weight) + router_bias  # (N, E)
    top_vals, top_idx = tensor_apis.topk(router_logits, k=top_k, axis=-1)  # (N, k)
    top_w = _softmax_lastdim(top_vals)  # (N, k)

    # Gather the selected experts' weights. Flatten (N, k) selection to (N*k,)
    # so a single take() pulls all expert matrices at once.
    sel = top_idx.reshape(N * top_k)  # (N*k,)
    gu_w = np.take(gate_up_weight, sel, axis=0)  # (N*k, D, 2I)
    gu_b = np.take(gate_up_bias, sel, axis=0)  # (N*k, 2I)
    dn_w = np.take(down_weight, sel, axis=0)  # (N*k, I, D)
    dn_b = np.take(down_bias, sel, axis=0)  # (N*k, D)

    # Repeat each token's input for its top_k experts: (N, D) -> (N*k, 1, D).
    x = flat.reshape(N, 1, D)
    x = np.broadcast_to(x, (N, top_k, D)).reshape(N * top_k, 1, D)

    # Batched gate/up projection: (N*k,1,D) @ (N*k,D,2I) -> (N*k,1,2I)
    gup = np.matmul(x, gu_w) + gu_b.reshape(N * top_k, 1, -1)
    xg, x_up = np.split(gup, 2, axis=-1)
    gated = clamped_swiglu(xg, x_up, alpha, limit)  # (N*k,1,I)

    # Batched down projection: (N*k,1,I) @ (N*k,I,D) -> (N*k,1,D)
    expert_out = np.matmul(gated, dn_w) + dn_b.reshape(N * top_k, 1, -1)
    expert_out = expert_out.reshape(N, top_k, D)

    # Weighted combine over the top_k experts.
    combined = np.sum(top_w.reshape(N, top_k, 1) * expert_out, axis=1)  # (N, D)
    return combined.reshape(B, L, D)


def moe_dense_masked(
    norm_z,
    router_weight,
    router_bias,
    gate_up_weight,
    gate_up_bias,
    down_weight,
    down_bias,
    top_k,
    alpha,
    limit,
):
    """Dense MoE: run ALL E experts as one batched GEMM, then mask to top_k.

    No gather and no data-dependent shapes -- the expert dimension is a static
    batch dim, so both projections are dense (E,N)-batched matmuls that fill the
    PE array. Costs E/top_k x the weight traffic (all experts, every token), so
    it only makes sense if PE-density wins beat the extra HBM reads -- expected
    to lose at N=1 but worth measuring as the ceiling of "fully dense".

    Shapes as in moe_batched; E = num_experts.
    """
    B, L, D = norm_z.shape
    N = B * L
    E = gate_up_weight.shape[0]
    I = down_weight.shape[1]
    flat = norm_z.reshape(N, D)

    router_logits = np.matmul(flat, router_weight) + router_bias  # (N, E)
    top_vals, top_idx = tensor_apis.topk(router_logits, k=top_k, axis=-1)
    top_w = _softmax_lastdim(top_vals)  # (N, k)
    # Scatter the top_k softmax weights back to a dense (N, E) gate (0 elsewhere).
    gate = _scatter_topk(top_w, top_idx, N, E)  # (N, E)

    # All experts, all tokens: (E,1,N,D) broadcast -> matmul with (E,1,D,2I).
    # Use expert as the leading batch dim so weights are not gathered/repeated.
    x = flat.reshape(1, N, D)
    x = np.broadcast_to(x, (E, N, D))  # (E, N, D)
    gup = np.matmul(x, gate_up_weight) + gate_up_bias.reshape(E, 1, 2 * I)  # (E,N,2I)
    xg, x_up = np.split(gup, 2, axis=-1)
    gated = clamped_swiglu(xg, x_up, alpha, limit)  # (E, N, I)
    eo = np.matmul(gated, down_weight) + down_bias.reshape(E, 1, D)  # (E, N, D)

    # Weight each expert's output by the (masked) router gate and sum over E.
    gate_t = gate.transpose(1, 0).reshape(E, N, 1)  # (E, N, 1)
    out = np.sum(gate_t * eo, axis=0)  # (N, D)
    return out.reshape(B, L, D)


def _scatter_topk(top_w, top_idx, N, E):
    """Dense (N, E) gate from (N, k) weights at (N, k) expert indices."""
    # one-hot via compare-against-iota: onehot[n,k,e] = (idx[n,k] == e)
    ids = np.arange(E).reshape(1, 1, E)  # (1,1,E) constant
    onehot = (top_idx.reshape(N, -1, 1) == ids).astype(top_w.dtype)  # (N,k,E)
    return np.sum(onehot * top_w.reshape(N, -1, 1), axis=1)  # (N, E)


def _softmax_lastdim(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
