"""Final output layer + unpatchify for the PixArt DiT."""

import numpy as np

from .layernorm import layernorm_kernel
from .modulation import final_modulation


def final_layer(x, embedded_timestep, w, hidden_size, eps):
    """norm_out -> adaLN modulate -> proj_out.

    Args:
        x: (B, L, hidden) tokens out of the last block.
        embedded_timestep: (B, hidden).
        w: dict with ``scale_shift_table`` (2, hidden), ``proj_out.weight``
           (hidden, patch_size**2 * out_channels), ``proj_out.bias``.
    Returns:
        (B, L, patch_size**2 * out_channels)
    """
    shift, scale = final_modulation(w["scale_shift_table"], embedded_timestep, hidden_size)
    h = layernorm_kernel(x, eps=eps)
    h = h * (1 + scale) + shift
    out = np.matmul(h, w["proj_out.weight"]) + w["proj_out.bias"]
    return out


def unpatchify(x, patch_size, out_channels, gh, gw):
    """(B, gh*gw, patch_size**2 * out_channels) -> (B, out_channels, gh*p, gw*p).

    Mirrors diffusers' einsum('nhwpqc->nchpwq').
    """
    B = x.shape[0]
    p = patch_size
    c = out_channels
    x = x.reshape(B, gh, gw, p, p, c)
    x = x.transpose(0, 5, 1, 3, 2, 4)  # n c h p w q
    x = x.reshape(B, c, gh * p, gw * p)
    return x
