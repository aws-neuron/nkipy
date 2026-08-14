"""Full PixArt-Sigma DiT forward pass, compiled as a single device kernel.

Given a (N, C, H, W) latent (N = 2B, already duplicated for classifier-free
guidance), a (N,) timestep, and (N, Ltext, caption_channels) T5 caption
embeddings, produce the predicted noise (N, C, H, W).

All weights arrive as flat ``**weights`` kwargs (the tracer turns each top-level
ndarray / expanded kwarg into an HLO parameter but does not recurse into dicts
or lists). ``regroup_weights`` rebuilds the shared/per-block structure.

PixArt uses "learn sigma": proj_out has ``2*C`` channels (noise + variance); the
sampler only needs the first ``C``, so we slice the model output here.
"""

import numpy as np
from config import Config

from .dit_block import dit_block
from .embeddings import (
    adaln_single_kernel,
    caption_projection_kernel,
    get_2d_sincos_pos_embed,
    patch_embed_kernel,
)
from .final_layer import final_layer, unpatchify
from .weight_layout import regroup_weights


def _dit_core(latent, timestep, caption, caption_mask, shared, blocks, configs):
    """Shared DiT denoiser body. Returns predicted noise (N, in_channels, H, W).

    ``latent``/``timestep``/``caption`` cover the full CFG batch (N = 2B).
    """
    C = configs.in_channels
    p = configs.patch_size
    hidden = configs.hidden_size
    eps = configs.norm_eps
    n_heads = configs.num_heads
    head_dim = configs.head_dim

    N, _, H, W = latent.shape
    gh, gw = H // p, W // p

    # ── patch embed (+ positional) ──
    # base_size is the model's *native* grid (native_sample_size // patch_size),
    # held fixed regardless of the runtime grid, matching diffusers' PatchEmbed.
    base_size = configs.native_sample_size // p
    pos_embed = get_2d_sincos_pos_embed(
        hidden, gh, base_size=base_size,
        interpolation_scale=configs.interpolation_scale,
    )  # comptime constant
    x = patch_embed_kernel(
        latent, shared["pos_embed.proj.weight"], shared["pos_embed.proj.bias"],
        p, pos_embed,
    )

    # ── shared adaLN-single timestep conditioning ──
    timestep_proj, embedded_timestep = adaln_single_kernel(
        timestep,
        shared["adaln.time_proj.weight"], shared["adaln.time_proj.bias"],
        shared["adaln.time_emb.weight"], shared["adaln.time_emb.bias"],
        shared["adaln.linear.weight"], shared["adaln.linear.bias"],
        hidden, configs.dtype,
    )

    # ── caption projection (T5 -> hidden) ──
    context = caption_projection_kernel(
        caption,
        shared["caption.w1"], shared["caption.b1"],
        shared["caption.w2"], shared["caption.b2"],
    )

    # cross-attention mask: (N, Ltext) {0,1} -> (N, 1, 1, Ltext) additive bias
    cross_mask_bias = (1.0 - caption_mask.astype(configs.dtype)) * np.float32(-1e9)
    cross_mask_bias = np.expand_dims(cross_mask_bias, axis=[1, 2])

    # ── transformer blocks ──
    for w in blocks:
        x = dit_block(x, context, w, timestep_proj, n_heads, head_dim, hidden, eps,
                      cross_mask_bias=cross_mask_bias)

    # ── final layer + unpatchify ──
    x = final_layer(x, embedded_timestep, shared, hidden, eps)
    out = unpatchify(x, p, configs.out_channels, gh, gw)

    # learn-sigma: keep only the noise-prediction channels
    noise_pred = out[:, :C]
    return noise_pred


def dit_forward(latent, timestep, caption, caption_mask, configs: Config, **weights):
    """Bare DiT denoiser: returns predicted noise (N, in_channels, H, W).

    ``**weights`` are flat tensors (the tracer turns each top-level ndarray /
    expanded kwarg into an HLO parameter); ``regroup_weights`` rebuilds the
    shared/per-block structure. Kept for validation against the diffusers
    baseline (see evaluate.py); the fused ``denoise_step`` is used for sampling.
    """
    shared, blocks = regroup_weights(weights, configs.num_layers)
    return _dit_core(latent, timestep, caption, caption_mask, shared, blocks, configs)


def denoise_step(
    latents,          # (B, C, H, W)  current latent (single, not CFG-duplicated)
    timestep,         # (2B,) timestep for the CFG batch
    caption,          # (2B, Ltext, cc)
    caption_mask,     # (2B, Ltext)
    prev_x0,          # (B, C, H, W) previous-step x0 prediction (multistep state)
    coeffs,           # (8,) host-precomputed scalar step coefficients
    configs: Config,
    **weights,
):
    """Fused one sampling step, keeping the latent resident on device.

    Runs the DiT denoiser on the CFG batch, applies classifier-free guidance,
    converts the predicted noise to an x0 prediction, and performs the
    DPM-Solver++ update — all on device. The host only supplies the per-step
    scalar coefficients (derived from the fixed sigma schedule).

    ``coeffs`` layout (see pixart.py ``dpm_coeffs``):
        [0] guidance_scale
        [1] alpha_t_s      (=1/sqrt(sigma_s^2+1) at current sigma)   -- convert
        [2] sigma_t_s      (=sigma_s*alpha_t_s)                       -- convert
        [3] c_sample       coefficient on the latent in the update
        [4] c_D0           coefficient on D0 (current x0)
        [5] c_D1           coefficient on D1 (x0 finite-difference term)
        [6] r0             h_0/h ratio for the 2nd-order term (0 for 1st order)
        [7] second_order   1.0 to enable the D1 term, else 0.0

    Returns (prev_sample, x0_cur):
        prev_sample: (B, C, H, W) latent for the next step
        x0_cur:      (B, C, H, W) this step's x0 prediction (feeds next step)
    """
    shared, blocks = regroup_weights(weights, configs.num_layers)

    # duplicate the latent for the uncond/cond CFG batch
    latent_in = np.concatenate([latents, latents], axis=0)
    noise = _dit_core(latent_in, timestep, caption, caption_mask, shared, blocks, configs)

    # classifier-free guidance
    noise = noise.astype(np.float32)
    noise_uncond, noise_cond = np.split(noise, 2, axis=0)
    guidance_scale = coeffs[0]
    model_output = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    # convert epsilon -> x0 prediction:  x0 = (sample - sigma_t * eps) / alpha_t
    sample = latents.astype(np.float32)
    x0_cur = (sample - coeffs[2] * model_output) / coeffs[1]

    # DPM-Solver++ update (midpoint). For a first-order step second_order=0.
    D0 = x0_cur
    D1 = (1.0 / coeffs[6]) * (x0_cur - prev_x0.astype(np.float32))
    prev_sample = (
        coeffs[3] * sample
        + coeffs[4] * D0
        + coeffs[7] * (coeffs[5] * D1)
    )

    return prev_sample.astype(configs.dtype), x0_cur.astype(configs.dtype)
