"""Input embeddings for the Qwen-Image MMDiT.

Three pieces run before the transformer blocks:

1. ``img_in`` — a Linear projecting the patchified latent (in_channels=64) to
   ``inner_dim`` (3072). The latent is already patchified by the pipeline, so
   this is a plain matmul (no strided patch-embed conv).
2. ``txt_norm`` (RMSNorm over joint_attention_dim) + ``txt_in`` (Linear to
   inner_dim) — normalize and project the Qwen2.5-VL text embeddings.
3. ``time_text_embed`` (``QwenTimestepProjEmbeddings``) — sinusoidal timestep
   embedding (256 ch, flip_sin_to_cos, scale 1000) → SiLU-MLP to ``inner_dim``.
   This ``temb`` conditions every block's modulation and the final layer.

The sinusoidal frequency table is a comptime numpy constant; only the timestep
values are runtime tensors.
"""

import numpy as np

from .rmsnorm import rmsnorm_kernel


def _silu(x):
    xf = x.astype(np.float32)
    return (xf * (1.0 / (1.0 + np.exp(-xf)))).astype(x.dtype)


def img_in_kernel(latent, weight, bias):
    """Project patchified latent (B, Limg, in_channels) -> (B, Limg, inner_dim)."""
    return np.matmul(latent, weight) + bias


def txt_in_kernel(text, norm_weight, in_weight, in_bias, eps=1e-6):
    """RMSNorm(text) then Linear -> (B, Ltext, inner_dim).

    Args:
        text: (B, Ltext, joint_attention_dim).
        norm_weight: (joint_attention_dim,) RMSNorm gain.
        in_weight: (joint_attention_dim, inner_dim), in_bias: (inner_dim,).
    """
    x = rmsnorm_kernel(text, norm_weight, eps=eps)
    return np.matmul(x, in_weight) + in_bias


def timestep_embedding(timesteps, dim=256, max_period=10000, scale=1000.0,
                       flip_sin_to_cos=True, downscale_freq_shift=0.0):
    """Sinusoidal timestep embedding matching diffusers ``get_timestep_embedding``.

    Qwen-Image uses ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0,
    scale=1000)``: the frequency argument is scaled by 1000, embeddings are
    ``[sin, cos]`` then flipped to ``[cos, sin]``.

    ``timesteps`` is a runtime (B,) tensor; the frequency table is comptime.
    Returns (B, dim).
    """
    half = dim // 2
    exponent = -np.log(max_period) * np.arange(half, dtype=np.float32)
    exponent = exponent / (half - downscale_freq_shift)
    freqs = np.exp(exponent)  # (half,)

    args = np.expand_dims(timesteps.astype(np.float32), -1) * np.expand_dims(freqs, 0)
    args = scale * args
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)  # (B, dim)
    if flip_sin_to_cos:
        emb = np.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    return emb


def time_text_embed_kernel(timesteps, proj_weight, proj_bias, emb_weight, emb_bias,
                           dtype):
    """QwenTimestepProjEmbeddings: sinusoid -> Linear -> SiLU -> Linear.

    Args:
        timesteps: (B,) diffusion timestep.
        proj_weight: (256, inner_dim), proj_bias: (inner_dim,)  [linear_1]
        emb_weight: (inner_dim, inner_dim), emb_bias: (inner_dim,)  [linear_2]
    Returns:
        temb: (B, inner_dim) conditioning embedding.
    """
    sin_emb = timestep_embedding(timesteps).astype(dtype)  # (B, 256)
    h = np.matmul(sin_emb, proj_weight) + proj_bias
    h = _silu(h)
    temb = np.matmul(h, emb_weight) + emb_bias
    return temb
