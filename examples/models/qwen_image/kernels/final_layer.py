"""Final output head for the Qwen-Image MMDiT.

``norm_out`` (``AdaLayerNormContinuous``): non-affine LayerNorm modulated by a
(scale, shift) derived from ``temb`` via SiLU-Linear, applied to the image
stream only. ``proj_out``: Linear(inner_dim -> patch_size**2 * out_channels).

diffusers' ``QwenImageTransformer2DModel.forward`` returns the proj_out result
directly (shape (B, Limg, patch**2 * out_ch)); the pipeline unpatchifies it into
(B, out_ch, H, W). ``unpatchify`` here mirrors that pipeline step for M4.
"""

import numpy as np

from .layernorm import layernorm_kernel


def _silu(x):
    xf = x.astype(np.float32)
    return (xf * (1.0 / (1.0 + np.exp(-xf)))).astype(x.dtype)


def final_layer(img, temb, norm_lin_w, norm_lin_b, proj_w, proj_b, eps=1e-6):
    """AdaLayerNormContinuous(norm_out) + proj_out on the image stream.

    Args:
        img: (B, Limg, inner_dim) image tokens from the last block.
        temb: (B, inner_dim) timestep conditioning.
        norm_lin_w: (inner_dim, 2*inner_dim), norm_lin_b: (2*inner_dim,).
        proj_w: (inner_dim, patch**2 * out_ch), proj_b: (patch**2 * out_ch,).
    Returns:
        (B, Limg, patch**2 * out_ch).
    """
    emb = np.matmul(_silu(temb), norm_lin_w) + norm_lin_b  # (B, 2*inner_dim)
    scale, shift = np.split(emb, 2, axis=-1)  # each (B, inner_dim)
    x = layernorm_kernel(img, eps=eps)
    x = x * (1 + np.expand_dims(scale, 1)) + np.expand_dims(shift, 1)
    return np.matmul(x, proj_w) + proj_b


def unpatchify(x, patch_size, out_channels, gh, gw):
    """(B, gh*gw, patch**2 * out_ch) -> (B, out_ch, gh*patch, gw*patch).

    Mirrors the QwenImagePipeline unpatchify: tokens are row-major over the
    (gh, gw) patch grid, each carrying a (out_ch, patch, patch) block.
    """
    B = x.shape[0]
    p = patch_size
    x = x.reshape(B, gh, gw, out_channels, p, p)
    x = x.transpose(0, 3, 1, 4, 2, 5)  # (B, out_ch, gh, p, gw, p)
    return x.reshape(B, out_channels, gh * p, gw * p)
