"""A single PixArt-Sigma DiT block.

Order (diffusers BasicTransformerBlock, ``ada_norm_single`` mode):

    1. self-attention:  norm1(x) modulated by (shift_msa, scale_msa),
       output gated by gate_msa, residual add.
    2. cross-attention:  NO norm2 in this mode (PixArt); Q from x, K/V from
       the caption context, residual add.
    3. feed-forward:  norm2(x) modulated by (shift_mlp, scale_mlp), output
       gated by gate_mlp, residual add.

norm1 / norm2 are non-affine LayerNorm; all conditioning enters through adaLN.

Weights arrive as a flat dict ``w`` keyed by the short canonical names defined
in ``weight_layout.py`` (e.g. ``q_w``, ``cq_w``, ``ff0_w``, ``sst``).
"""

from .attention import cross_attention_kernel, self_attention_kernel
from .feedforward import feedforward_kernel
from .layernorm import layernorm_kernel
from .modulation import block_modulation, modulate


def dit_block(x, context, w, timestep, n_heads, head_dim, hidden_size, eps,
              cross_mask_bias=None):
    """Run one DiT block.

    Args:
        x: (B, L, hidden) image latent tokens.
        context: (B, Ltext, hidden) projected caption tokens.
        w: dict of this block's weights (short canonical keys).
        timestep: (B, 6*hidden) shared adaLN projection.
        cross_mask_bias: (B, 1, 1, Ltext) additive mask for T5 padding tokens.
    Returns:
        (B, L, hidden)
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block_modulation(
        w["sst"], timestep, hidden_size
    )

    # 1. self-attention
    h = layernorm_kernel(x, eps=eps)
    h = modulate(h, shift_msa, scale_msa)
    attn = self_attention_kernel(
        h,
        w["q_w"], w.get("q_b"),
        w["k_w"], w.get("k_b"),
        w["v_w"], w.get("v_b"),
        w["o_w"], w.get("o_b"),
        n_heads, head_dim,
    )
    x = x + gate_msa * attn

    # 2. cross-attention (no norm in ada_norm_single mode)
    cross = cross_attention_kernel(
        x, context,
        w["cq_w"], w.get("cq_b"),
        w["ck_w"], w.get("ck_b"),
        w["cv_w"], w.get("cv_b"),
        w["co_w"], w.get("co_b"),
        n_heads, head_dim,
        mask_bias=cross_mask_bias,
    )
    x = x + cross

    # 3. feed-forward (reuses norm2)
    h = layernorm_kernel(x, eps=eps)
    h = modulate(h, shift_mlp, scale_mlp)
    ff = feedforward_kernel(
        h,
        w["ff0_w"], w.get("ff0_b"),
        w["ff2_w"], w.get("ff2_b"),
    )
    x = x + gate_mlp * ff

    return x
