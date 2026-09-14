"""A single Qwen-Image MMDiT dual-stream block.

Mirrors diffusers ``QwenImageTransformerBlock`` (still-image path, no
``zero_cond_t`` / ``modulate_index``). Both the image stream (``img``) and the
text stream (``txt``) carry their own norms, modulation, and MLP; they meet only
in the joint attention.

Order:
    1. modulation: img_mod/txt_mod = Linear(SiLU(temb)) -> 6*dim, split into
       (mod1, mod2), each (shift, scale, gate).
    2. norm1 + modulate both streams -> joint attention -> gated residual.
    3. norm2 + modulate both streams -> per-stream MLP -> gated residual.

norm1/norm2 are non-affine LayerNorm; all conditioning enters through the
modulation. Weights arrive as a flat dict ``w`` keyed by the canonical names in
``weight_layout.py``.
"""

import numpy as np

from .attention import joint_attention_kernel
from .feedforward import feedforward_kernel
from .layernorm import layernorm_kernel


def _silu(x):
    xf = x.astype(np.float32)
    return (xf * (1.0 / (1.0 + np.exp(-xf)))).astype(x.dtype)


def _modulation(temb, mod_w, mod_b, hidden):
    """SiLU -> Linear(dim, 6*dim), returning the two (shift, scale, gate) triples.

    Args:
        temb: (B, hidden) timestep conditioning.
        mod_w: (hidden, 6*hidden), mod_b: (6*hidden,).
    Returns:
        (mod1, mod2), each a tuple (shift, scale, gate) of (B, 1, hidden).
    """
    params = np.matmul(_silu(temb), mod_w) + mod_b  # (B, 6*hidden)
    mod1, mod2 = np.split(params, 2, axis=-1)  # each (B, 3*hidden)

    def _split3(m):
        shift, scale, gate = np.split(m, 3, axis=-1)  # each (B, hidden)
        return (
            np.expand_dims(shift, 1),
            np.expand_dims(scale, 1),
            np.expand_dims(gate, 1),
        )

    return _split3(mod1), _split3(mod2)


def _apply_mod(x, shift, scale):
    return x * (1 + scale) + shift


def mmdit_block(img, txt, w, temb, n_heads, head_dim, hidden, eps,
                img_cos, img_sin, txt_cos, txt_sin, txt_mask_bias=None,
                local_heads=None, all_reduce_fn=None):
    """Run one MMDiT block.

    Args:
        img: (B, Limg, hidden) image tokens.
        txt: (B, Ltext, hidden) text tokens.
        w: dict of this block's weights (canonical keys). Under tensor
            parallelism the attention/MLP weights are the sharded slices; the
            replicated modulation/norms keep full hidden.
        temb: (B, hidden) timestep conditioning.
        img_cos/img_sin/txt_cos/txt_sin: RoPE tables (see rope3d).
        txt_mask_bias: optional (B, 1, 1, Ltext) additive text-padding bias.
        local_heads: attention heads owned by this rank (None -> n_heads / TP=1).
        all_reduce_fn: optional collective summing the row-parallel outputs
            (attention out-proj and MLP down-proj) across ranks.
    Returns:
        (img, txt) updated streams (full hidden; replicated across ranks).
    """
    (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = _modulation(
        temb, w["img_mod_w"], w["img_mod_b"], hidden
    )
    (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = _modulation(
        temb, w["txt_mod_w"], w["txt_mod_b"], hidden
    )

    # 1. norm1 + modulate -> joint attention -> gated residual
    img_mod = _apply_mod(layernorm_kernel(img, eps=eps), img_shift1, img_scale1)
    txt_mod = _apply_mod(layernorm_kernel(txt, eps=eps), txt_shift1, txt_scale1)

    img_attn, txt_attn = joint_attention_kernel(
        img_mod, txt_mod,
        w["iq_w"], w["iq_b"], w["ik_w"], w["ik_b"], w["iv_w"], w["iv_b"],
        w["io_w"], w["io_b"], w["iq_g"], w["ik_g"],
        w["tq_w"], w["tq_b"], w["tk_w"], w["tk_b"], w["tv_w"], w["tv_b"],
        w["to_w"], w["to_b"], w["tq_g"], w["tk_g"],
        n_heads, head_dim, eps,
        img_cos, img_sin, txt_cos, txt_sin,
        txt_mask_bias=txt_mask_bias,
        local_heads=local_heads, all_reduce_fn=all_reduce_fn,
    )
    img = img + img_gate1 * img_attn
    txt = txt + txt_gate1 * txt_attn

    # 2. norm2 + modulate -> per-stream MLP -> gated residual
    img_mod2 = _apply_mod(layernorm_kernel(img, eps=eps), img_shift2, img_scale2)
    img_ff = feedforward_kernel(img_mod2, w["iff0_w"], w["iff0_b"], w["iff2_w"], w["iff2_b"],
                                all_reduce_fn=all_reduce_fn)
    img = img + img_gate2 * img_ff

    txt_mod2 = _apply_mod(layernorm_kernel(txt, eps=eps), txt_shift2, txt_scale2)
    txt_ff = feedforward_kernel(txt_mod2, w["tff0_w"], w["tff0_b"], w["tff2_w"], w["tff2_b"],
                                all_reduce_fn=all_reduce_fn)
    txt = txt + txt_gate2 * txt_ff

    return img, txt
