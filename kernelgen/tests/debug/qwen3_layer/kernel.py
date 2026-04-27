"""
Qwen3 Transformer Decoder Layer (inlined for readability).

All sub-kernels (RMSNorm, RoPE, softmax, SiLU) are inlined so the full
data flow is visible in one place.

Shape convention:
  - 2D projections use (BS, hidden_size) where BS = batch * seq_len.
    This flattens batch and sequence into one "token" dimension so the
    matmuls are plain 2D.  The reshape to (batch, seq_len, n_heads, head_dim)
    recovers the sequence dimension when needed for multi-head attention.
  - 3D attention tensors are (BH, seq_len, X) where BH = batch * n_heads.
    The partition dimension for these is dim 1 (seq_len), NOT dim 0 (BH).
"""
import numpy as np
from nkipy_kernelgen import trace, knob

# ----------------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------------
batch = 2
seq_len = 128
hidden_size = 256
n_heads = 2
head_dim = hidden_size // n_heads      # 128
intermediate_size = 256
half_dim = head_dim // 2               # 64
eps = 1e-6
scale = 1.0 / np.sqrt(head_dim).item()

# Derived (flattened dimensions)
BS = batch * seq_len                   # 256  (tokens = batch * seq_len)
BH = batch * n_heads                   # 4    (heads  = batch * n_heads)

# ----------------------------------------------------------------
# Tile sizes
# ----------------------------------------------------------------
matmul_tile_2d = [128, 128]
matmul_reduction_2d = [128]
attn_tile = [1, 128, 128]              # (BH, seq_len, seq_len/head_dim)
attn_reduction = [128]
rope_tile = [1, 128, 64]               # (BH, seq_len, half_dim)
elem_tile_2d = [128, 128]


@trace(input_specs=[
    # hidden_states: (BS, hidden_size) = (256, 256)
    #   BS = batch * seq_len, flattened for 2D matmul projections.
    #   Reshape to (batch, seq_len, n_heads, head_dim) recovers seq_len
    #   for multi-head attention.
    ((BS, hidden_size), "f32"),
    # RMSNorm weights — (hidden_size, 1) so broadcast is over the free dim
    # ([P, 1] pattern), which maps to nisa.tensor_scalar_arith.
    ((hidden_size, 1), "f32"),                             # ln1_weight
    ((hidden_size, 1), "f32"),                             # ln2_weight
    # Attention projection weights
    ((hidden_size, hidden_size), "f32"),                   # w_q
    ((hidden_size, hidden_size), "f32"),                   # w_k
    ((hidden_size, hidden_size), "f32"),                   # w_v
    ((hidden_size, hidden_size), "f32"),                   # w_o
    # RoPE frequencies (position-dependent, broadcast over BH)
    ((1, seq_len, half_dim), "f32"),                       # freqs_cos
    ((1, seq_len, half_dim), "f32"),                       # freqs_sin
    # FFN weights
    ((hidden_size, intermediate_size), "f32"),             # w_gate
    ((hidden_size, intermediate_size), "f32"),             # w_up
    ((intermediate_size, hidden_size), "f32"),             # w_down
])
def qwen3_layer(hidden_states,
                ln1_weight, ln2_weight,
                w_q, w_k, w_v, w_o,
                freqs_cos, freqs_sin,
                w_gate, w_up, w_down):

    residual = hidden_states                                  # (BS, hidden_size)

    # ================================================================
    # 1. Pre-attention RMSNorm
    #    norm(x) = x / sqrt(mean(x^2) + eps) * weight
    # ================================================================
    x_fp32 = hidden_states.astype(np.float32)
    w_fp32 = ln1_weight.astype(np.float32)

    sq = np.square(x_fp32)                                    # (256, 256)
    knob.knob(sq, mem_space="Sbuf", tile_size=elem_tile_2d)

    sum_sq = np.sum(sq, axis=-1, keepdims=True)               # (256, 1)
    knob.knob(sum_sq, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])

    mean_sq = sum_sq * np.float32(1.0 / hidden_size)          # (256, 1)
    knob.knob(mean_sq, mem_space="Sbuf", tile_size=[128, 1])

    normed = x_fp32 / np.sqrt(mean_sq + eps)                  # (256, 256)
    knob.knob(normed, mem_space="Sbuf", tile_size=elem_tile_2d)

    normed = normed * w_fp32                                   # (256, 256)
    knob.knob(normed, mem_space="Sbuf", tile_size=elem_tile_2d)

    # ================================================================
    # 2. QKV projections (2D matmuls on flattened BS dimension)
    #    SharedHbm = sub-kernel boundary (results flow through reshape)
    # ================================================================
    q = np.matmul(normed, w_q)                                # (256, 256)
    knob.knob(q, mem_space="SharedHbm", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    k = np.matmul(normed, w_k)                                # (256, 256)
    knob.knob(k, mem_space="SharedHbm", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    v = np.matmul(normed, w_v)                                # (256, 256)
    knob.knob(v, mem_space="SharedHbm", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    # ================================================================
    # 3. Reshape to multi-head format
    #    (BS, hidden) -> (batch, seq_len, n_heads, head_dim)
    #    -> transpose to (batch, n_heads, seq_len, head_dim)
    #    -> flatten to (BH, seq_len, head_dim)
    # ================================================================
    q = np.reshape(q, (batch, seq_len, n_heads, head_dim))    # (2, 128, 2, 128)
    q = np.transpose(q, (0, 2, 1, 3))                        # (2, 2, 128, 128)
    q = np.reshape(q, (BH, seq_len, head_dim))                # (4, 128, 128)

    k = np.reshape(k, (batch, seq_len, n_heads, head_dim))
    k = np.transpose(k, (0, 2, 1, 3))
    k = np.reshape(k, (BH, seq_len, head_dim))                # (4, 128, 128)

    v = np.reshape(v, (batch, seq_len, n_heads, head_dim))
    v = np.transpose(v, (0, 2, 1, 3))
    v = np.reshape(v, (BH, seq_len, head_dim))                # (4, 128, 128)
    # V is a sub-kernel boundary: keep in SharedHbm so the 4D transpose
    # intermediate stays in HBM (avoids a 4D SBUF alloc that legalize-layout
    # cannot tile).
    knob.knob(v, mem_space="SharedHbm", tile_size=attn_tile)

    # ================================================================
    # 4. RoPE on Q and K (not V)
    #    Split head_dim in half, rotate: [x0, x1] -> [x0*cos - x1*sin,
    #                                                  x0*sin + x1*cos]
    #    freqs_cos/sin are (1, 128, 64) — broadcast over BH dim
    # ================================================================
    # --- RoPE on Q ---
    q0 = q[:, :, :half_dim]                                   # (4, 128, 64)
    q1 = q[:, :, half_dim:]                                   # (4, 128, 64)

    q0_cos = q0 * freqs_cos                                    # (4, 128, 64)
    knob.knob(q0_cos, mem_space="SharedHbm", tile_size=rope_tile)
    q1_sin = q1 * freqs_sin                                    # (4, 128, 64)
    knob.knob(q1_sin, mem_space="SharedHbm", tile_size=rope_tile)
    q_rot0 = q0_cos - q1_sin                                   # (4, 128, 64)
    knob.knob(q_rot0, mem_space="SharedHbm", tile_size=rope_tile)

    q0_sin = q0 * freqs_sin                                    # (4, 128, 64)
    knob.knob(q0_sin, mem_space="SharedHbm", tile_size=rope_tile)
    q1_cos = q1 * freqs_cos                                    # (4, 128, 64)
    knob.knob(q1_cos, mem_space="SharedHbm", tile_size=rope_tile)
    q_rot1 = q0_sin + q1_cos                                   # (4, 128, 64)
    knob.knob(q_rot1, mem_space="SharedHbm", tile_size=rope_tile)

    q = np.concatenate([q_rot0, q_rot1], axis=-1)             # (4, 128, 128)
    knob.knob(q, mem_space="SharedHbm", tile_size=attn_tile)

    # --- RoPE on K ---
    k0 = k[:, :, :half_dim]                                   # (4, 128, 64)
    k1 = k[:, :, half_dim:]                                   # (4, 128, 64)

    k0_cos = k0 * freqs_cos                                    # (4, 128, 64)
    knob.knob(k0_cos, mem_space="SharedHbm", tile_size=rope_tile)
    k1_sin = k1 * freqs_sin                                    # (4, 128, 64)
    knob.knob(k1_sin, mem_space="SharedHbm", tile_size=rope_tile)
    k_rot0 = k0_cos - k1_sin                                   # (4, 128, 64)
    knob.knob(k_rot0, mem_space="SharedHbm", tile_size=rope_tile)

    k0_sin = k0 * freqs_sin                                    # (4, 128, 64)
    knob.knob(k0_sin, mem_space="SharedHbm", tile_size=rope_tile)
    k1_cos = k1 * freqs_cos                                    # (4, 128, 64)
    knob.knob(k1_cos, mem_space="SharedHbm", tile_size=rope_tile)
    k_rot1 = k0_sin + k1_cos                                   # (4, 128, 64)
    knob.knob(k_rot1, mem_space="SharedHbm", tile_size=rope_tile)

    k = np.concatenate([k_rot0, k_rot1], axis=-1)             # (4, 128, 128)
    knob.knob(k, mem_space="SharedHbm", tile_size=attn_tile)

    # K^T for attention scores
    k_t = np.transpose(k, (0, 2, 1))                          # (4, 128, 128)
    knob.knob(k_t, mem_space="SharedHbm", tile_size=attn_tile)

    # ================================================================
    # 5. Scaled dot-product attention
    #    scores = (Q @ K^T) * scale
    #    weights = softmax(scores)
    #    context = weights @ V
    # ================================================================
    scores = np.matmul(q, k_t)                                # (4, 128, 128)
    knob.knob(scores, mem_space="Sbuf", tile_size=attn_tile, reduction_tile=attn_reduction)

    scores = scores * scale                                    # (4, 128, 128)
    knob.knob(scores, mem_space="Sbuf", tile_size=attn_tile, partition_dim=1)

    # --- softmax (numerically stable) ---
    scores_fp32 = scores.astype(np.float32)

    s_max = np.max(scores_fp32, axis=-1, keepdims=True)       # (4, 128, 1)
    knob.knob(s_max, mem_space="Sbuf", tile_size=[1, 128],
              reduction_tile=[128], partition_dim=1)

    shifted = scores_fp32 - s_max                              # (4, 128, 128)
    knob.knob(shifted, mem_space="Sbuf", tile_size=attn_tile, partition_dim=1)

    exp_s = np.exp(shifted)                                    # (4, 128, 128)
    knob.knob(exp_s, mem_space="Sbuf", tile_size=attn_tile, partition_dim=1)

    sum_exp = np.sum(exp_s, axis=-1, keepdims=True)            # (4, 128, 1)
    knob.knob(sum_exp, mem_space="Sbuf", tile_size=[1, 128],
              reduction_tile=[128], partition_dim=1)

    attn_weights = exp_s / sum_exp                             # (4, 128, 128)
    knob.knob(attn_weights, mem_space="SharedHbm", tile_size=attn_tile)

    # --- context = attn_weights @ V ---
    context = np.matmul(attn_weights, v)                       # (4, 128, 128)
    knob.knob(context, mem_space="SharedHbm", tile_size=attn_tile, reduction_tile=attn_reduction)

    # ================================================================
    # 6. Concat heads + output projection
    #    (BH, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim)
    #    -> transpose to (batch, seq_len, n_heads, head_dim)
    #    -> flatten to (BS, hidden_size)
    # ================================================================
    context = np.reshape(context, (batch, n_heads, seq_len, head_dim))
    context = np.transpose(context, (0, 2, 1, 3))
    context = np.reshape(context, (BS, hidden_size))           # (256, 256)

    attn_out = np.matmul(context, w_o)                         # (256, 256)
    knob.knob(attn_out, mem_space="Sbuf", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    # ================================================================
    # 7. First residual connection
    # ================================================================
    hidden_states = residual + attn_out                        # (256, 256)
    knob.knob(hidden_states, mem_space="Sbuf", tile_size=elem_tile_2d)

    residual = hidden_states

    # ================================================================
    # 8. Post-attention RMSNorm
    # ================================================================
    x_fp32 = hidden_states.astype(np.float32)
    w_fp32 = ln2_weight.astype(np.float32)

    sq = np.square(x_fp32)                                     # (256, 256)
    knob.knob(sq, mem_space="Sbuf", tile_size=elem_tile_2d)

    sum_sq = np.sum(sq, axis=-1, keepdims=True)                # (256, 1)
    knob.knob(sum_sq, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])

    mean_sq = sum_sq * np.float32(1.0 / hidden_size)           # (256, 1)
    knob.knob(mean_sq, mem_space="Sbuf", tile_size=[128, 1])

    normed = x_fp32 / np.sqrt(mean_sq + eps)                   # (256, 256)
    knob.knob(normed, mem_space="Sbuf", tile_size=elem_tile_2d)

    normed = normed * w_fp32                                    # (256, 256)
    knob.knob(normed, mem_space="Sbuf", tile_size=elem_tile_2d)

    # ================================================================
    # 9. SwiGLU FFN
    #    gate = SiLU(normed @ w_gate)
    #    up   = normed @ w_up
    #    out  = (gate * up) @ w_down
    # ================================================================
    gate = np.matmul(normed, w_gate)                           # (256, 256)
    knob.knob(gate, mem_space="Sbuf", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    up = np.matmul(normed, w_up)                               # (256, 256)
    knob.knob(up, mem_space="Sbuf", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    # --- SiLU(gate) = gate * sigmoid(gate) ---
    neg_gate = -gate                                           # (256, 256)
    knob.knob(neg_gate, mem_space="Sbuf", tile_size=elem_tile_2d)

    exp_neg = np.exp(neg_gate)                                 # (256, 256)
    knob.knob(exp_neg, mem_space="Sbuf", tile_size=elem_tile_2d)

    one_plus = exp_neg + 1.0                                   # (256, 256)
    knob.knob(one_plus, mem_space="Sbuf", tile_size=elem_tile_2d)

    sigmoid = 1.0 / one_plus                                   # (256, 256)
    knob.knob(sigmoid, mem_space="Sbuf", tile_size=elem_tile_2d)

    gate = gate * sigmoid                                      # (256, 256)
    knob.knob(gate, mem_space="Sbuf", tile_size=elem_tile_2d)

    # --- gated output ---
    gated = gate * up                                          # (256, 256)
    knob.knob(gated, mem_space="Sbuf", tile_size=elem_tile_2d)

    ffn_out = np.matmul(gated, w_down)                         # (256, 256)
    knob.knob(ffn_out, mem_space="Sbuf", tile_size=matmul_tile_2d, reduction_tile=matmul_reduction_2d)

    # ================================================================
    # 10. Second residual connection
    # ================================================================
    output = residual + ffn_out                                # (256, 256)
    knob.knob(output, mem_space="SharedHbm", tile_size=elem_tile_2d)

    return output                                              # (256, 256)
