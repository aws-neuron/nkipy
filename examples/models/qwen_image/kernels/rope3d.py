"""3D RoPE for the Qwen-Image MMDiT (``QwenEmbedRope``).

Qwen-Image applies rotary position embedding over a 3D (frame, height, width)
grid for the image tokens and a 1D range for the text tokens, then rotates the
attention queries/keys of *both* streams before the joint attention.

Two things differ from the qwen3 example's RoPE:

* **Interleaved complex convention.** diffusers builds per-position unit-magnitude
  complex frequencies and multiplies ``view_as_complex(x reshaped to (...,-1,2))``
  by them (``apply_rotary_emb_qwen`` with ``use_real=False``). So dimension pair
  ``(2i, 2i+1)`` of the head is one complex number rotated by angle
  ``theta_i``, rather than qwen3's ``(i, i+half)`` half-split.
* **3D axes with ``scale_rope`` centering.** ``head_dim`` is partitioned into
  ``axes_dims_rope`` (frame, H, W); the H/W position indices are centered around
  zero (negative half then positive half), the frame axis is not.

Positions are comptime constants (grid size is known at trace time), so the
angle table is built in numpy and baked into the graph; only the runtime rotate
(``apply_rotary_emb``) touches tensors. We keep everything real-valued
(cos/sin), which matches ``x_out = x*cos + rotate(x)*sin`` — the real form of the
complex multiply — and needs no complex dtype on device.
"""

import numpy as np


def _axis_freqs(index, dim, theta):
    """Per-position angles for one RoPE axis.

    Args:
        index: (P,) integer positions.
        dim: this axis' slice of head_dim (must be even); yields dim//2 angles.
        theta: RoPE base.
    Returns:
        (P, dim//2) angles = outer(index, 1/theta**(arange(0,dim,2)/dim)).
    """
    assert dim % 2 == 0
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    return np.outer(index.astype(np.float64), inv_freq)  # (P, dim//2)


def compute_rope_freqs(frame, height, width, txt_len, axes_dims, theta=10000.0,
                       scale_rope=True, max_pos=4096):
    """Build the (image, text) RoPE angle tables (comptime constant).

    Mirrors ``QwenEmbedRope._compute_video_freqs`` + the text slice in
    ``forward``. Returns per-position angles (not yet cos/sin) so callers can
    materialise cos/sin at the desired dtype.

    Args:
        frame, height, width: image grid (frame is 1 for a still image). These
            are the RoPE grid extents; the pipeline passes H_latent//patch,
            W_latent//patch here.
        txt_len: number of text tokens.
        axes_dims: (d_frame, d_h, d_w) partition of head_dim, each even.
        theta: RoPE base.
        scale_rope: center the H/W position indices around zero (Qwen default).
        max_pos: size of the cached position range (diffusers uses 4096).

    Returns:
        vid_angles: (frame*height*width, head_dim//2)
        txt_angles: (txt_len, head_dim//2)
    """
    d_f, d_h, d_w = axes_dims

    # Positive index range [0..max_pos) and the negative (centering) range,
    # matching diffusers: neg index = arange(max_pos).flip(0) * -1 - 1.
    pos_index = np.arange(max_pos)
    neg_index = np.arange(max_pos)[::-1] * -1 - 1  # [..., -3, -2, -1]

    pos_f = _axis_freqs(pos_index, d_f, theta)  # (max_pos, d_f//2)
    pos_h = _axis_freqs(pos_index, d_h, theta)
    pos_w = _axis_freqs(pos_index, d_w, theta)
    neg_h = _axis_freqs(neg_index, d_h, theta)
    neg_w = _axis_freqs(neg_index, d_w, theta)

    # frame axis: positions [0..frame), broadcast over (frame, H, W)
    fr = pos_f[:frame].reshape(frame, 1, 1, -1)
    fr = np.broadcast_to(fr, (frame, height, width, d_f // 2))

    if scale_rope:
        h_ang = np.concatenate([neg_h[-(height - height // 2):], pos_h[:height // 2]], axis=0)
        w_ang = np.concatenate([neg_w[-(width - width // 2):], pos_w[:width // 2]], axis=0)
    else:
        h_ang = pos_h[:height]
        w_ang = pos_w[:width]
    hh = np.broadcast_to(h_ang.reshape(1, height, 1, -1), (frame, height, width, d_h // 2))
    ww = np.broadcast_to(w_ang.reshape(1, 1, width, -1), (frame, height, width, d_w // 2))

    vid_angles = np.concatenate([fr, hh, ww], axis=-1).reshape(frame * height * width, -1)

    # text tokens occupy positions after the image grid center
    if scale_rope:
        max_vid_index = max(height // 2, width // 2)
    else:
        max_vid_index = max(height, width)
    txt_f = pos_f[max_vid_index:max_vid_index + txt_len]
    txt_h = pos_h[max_vid_index:max_vid_index + txt_len]
    txt_w = pos_w[max_vid_index:max_vid_index + txt_len]
    txt_angles = np.concatenate([txt_f, txt_h, txt_w], axis=-1)  # (txt_len, head_dim//2)

    return vid_angles, txt_angles


def cos_sin_from_angles(angles, dtype=np.float32):
    """Turn (S, head_dim//2) angles into interleaved (S, head_dim) cos/sin.

    Each angle governs a dimension *pair*; interleaving to (cos0,cos0,cos1,...)
    lets ``apply_rotary_emb`` operate on the (2i, 2i+1) real/imag pairs without
    reshaping. Returned as comptime constants.
    """
    cos = np.cos(angles)
    sin = np.sin(angles)
    cos = np.repeat(cos, 2, axis=-1).astype(dtype)  # (S, head_dim)
    sin = np.repeat(sin, 2, axis=-1).astype(dtype)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """Rotate q/k with interleaved RoPE (real form of the complex multiply).

    Args:
        x: (B, H, S, head_dim) queries or keys.
        cos, sin: (S, head_dim) interleaved tables from ``cos_sin_from_angles``.

    For complex z = (x_even + i*x_odd) rotated by (cos + i*sin):
        out_even = x_even*cos - x_odd*sin
        out_odd  = x_even*sin + x_odd*cos
    which equals ``x*cos + rotate_half_interleaved(x)*sin`` where
    ``rotate_half_interleaved([a,b]) = [-b, a]`` per pair.
    """
    orig_dtype = x.dtype
    xf = x.astype(np.float32)
    cos = cos.astype(np.float32).reshape((1, 1) + cos.shape)
    sin = sin.astype(np.float32).reshape((1, 1) + sin.shape)

    x_even = xf[..., 0::2]
    x_odd = xf[..., 1::2]
    # interleave [-x_odd, x_even] back to full width. Build it with a stack +
    # reshape rather than assigning into strided slices of an empty array: the
    # strided-slice assignment lowers to a stride-2 scatter that dominated the
    # whole block (~75 ms/block, ~95% of the cte-backend cost); stack+reshape
    # lowers to a cheap concat + contiguous reshape.
    rot = np.stack([-x_odd, x_even], axis=-1).reshape(xf.shape)

    out = xf * cos + rot * sin
    return out.astype(orig_dtype)
