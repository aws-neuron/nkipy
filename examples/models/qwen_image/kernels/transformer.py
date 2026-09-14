"""Full Qwen-Image MMDiT forward pass.

Given a patchified latent (B, Limg, in_channels), text embeddings (B, Ltext,
joint_attention_dim), a (B,) timestep, and the image grid ``img_shape`` =
(frame, gh, gw), produce ``proj_out`` (B, Limg, patch**2 * out_channels) —
matching diffusers ``QwenImageTransformer2DModel.forward`` (which returns before
unpatchify; the pipeline unpatchifies).

All weights arrive as flat ``**weights`` kwargs (the tracer turns each top-level
ndarray into an HLO parameter but does not recurse into dicts); ``regroup_weights``
rebuilds the shared/per-block structure. The RoPE tables are comptime constants
built from the (fixed) image grid + text length.
"""

import numpy as np
from config import Config

from .embeddings import img_in_kernel, time_text_embed_kernel, txt_in_kernel
from .final_layer import final_layer, unpatchify
from .mmdit_block import mmdit_block
from .rope3d import compute_rope_freqs, cos_sin_from_angles
from .weight_layout import regroup_weights


def _core(latent, text, timestep, img_shape, shared, blocks, configs,
          txt_mask_bias=None):
    """Shared MMDiT body. Returns proj_out (B, Limg, patch**2 * out_channels).

    Tensor parallelism (``configs.tp_size`` > 1) shards the attention heads and
    MLP intermediate across ranks; ``configs.all_reduce_fn`` sums the
    row-parallel outputs. The blocks receive already-sharded weights (from
    ``tp.shard_*`` in the driver) and ``local_heads`` = n_heads // tp_size.
    """
    hidden = configs.hidden_size
    n_heads = configs.num_heads
    head_dim = configs.head_dim
    eps = configs.norm_eps
    frame, gh, gw = img_shape

    tp_size = getattr(configs, "tp_size", 1) or 1
    local_heads = n_heads // tp_size if tp_size > 1 else None
    all_reduce_fn = configs.all_reduce_fn
    Ltext = text.shape[1]

    # ── input projections ──
    img = img_in_kernel(latent, shared["img_in.weight"], shared["img_in.bias"])
    txt = txt_in_kernel(
        text, shared["txt_norm.weight"],
        shared["txt_in.weight"], shared["txt_in.bias"], eps=1e-6,
    )

    # ── timestep conditioning ──
    temb = time_text_embed_kernel(
        timestep,
        shared["time.proj.weight"], shared["time.proj.bias"],
        shared["time.emb.weight"], shared["time.emb.bias"],
        configs.dtype,
    )

    # ── RoPE tables (comptime) ──
    vid_ang, txt_ang = compute_rope_freqs(
        frame, gh, gw, Ltext, configs.axes_dims_rope,
        theta=configs.rope_theta, scale_rope=True,
    )
    img_cos, img_sin = cos_sin_from_angles(vid_ang, dtype=configs.dtype)
    txt_cos, txt_sin = cos_sin_from_angles(txt_ang, dtype=configs.dtype)

    # ── transformer blocks ──
    for w in blocks:
        img, txt = mmdit_block(
            img, txt, w, temb, n_heads, head_dim, hidden, eps,
            img_cos, img_sin, txt_cos, txt_sin, txt_mask_bias=txt_mask_bias,
            local_heads=local_heads, all_reduce_fn=all_reduce_fn,
        )

    # ── final layer (image stream only) ──
    out = final_layer(
        img, temb,
        shared["norm_out.linear.weight"], shared["norm_out.linear.bias"],
        shared["proj_out.weight"], shared["proj_out.bias"], eps=1e-6,
    )
    return out


def qwenimage_forward(latent, text, timestep, img_shape, configs: Config,
                      text_mask=None, unpatch=False, **weights):
    """MMDiT denoiser.

    Args:
        latent: (B, Limg, in_channels) patchified latent.
        text: (B, Ltext, joint_attention_dim) text embeddings.
        timestep: (B,) diffusion timestep.
        img_shape: (frame, gh, gw) patch grid for RoPE / unpatchify.
        text_mask: optional (B, Ltext) {0,1} mask for text padding.
        unpatch: if True, unpatchify to (B, out_channels, gh*p, gw*p); otherwise
            return the raw proj_out (B, Limg, patch**2 * out_channels), matching
            the diffusers transformer output.
    """
    shared, blocks = regroup_weights(weights, configs.num_layers)

    txt_mask_bias = None
    if text_mask is not None:
        txt_mask_bias = (1.0 - text_mask.astype(configs.dtype)) * np.float32(-1e9)
        txt_mask_bias = np.expand_dims(txt_mask_bias, axis=[1, 2])  # (B,1,1,Ltext)

    out = _core(latent, text, timestep, img_shape, shared, blocks, configs,
                txt_mask_bias=txt_mask_bias)

    if unpatch:
        frame, gh, gw = img_shape
        out = unpatchify(out, configs.patch_size, configs.out_channels, gh, gw)
    return out


def denoise_step(
    latents,       # (B, Limg, in_channels)  current packed latent (resident)
    cond_text,     # (B, Ltext, joint_dim)   conditional text embeddings
    neg_text,      # (B, Ltext, joint_dim)   unconditional text embeddings
    timestep,      # (B,) current flow-match timestep (already /1000)
    coeffs,        # (2,) [true_cfg_scale, dt]  host-precomputed scalars
    img_shape, configs: Config,
    cond_mask=None, neg_mask=None, **weights,
):
    """Fused one flow-match sampling step, keeping the latent resident on device.

    Runs the MMDiT denoiser on the CFG batch (cond + uncond stacked), applies
    Qwen-Image "true CFG" (norm-rescaled guidance), and performs the
    FlowMatchEuler update ``prev = sample + dt * model_output`` — all on device.
    The host supplies only the per-step scalars (``true_cfg_scale``, ``dt``);
    both are functions of the fixed sigma schedule.

    Model output and packed latents share the same shape (in_channels ==
    patch**2 * out_channels == 64), so the step stays in packed space (unpatchify
    happens once, on host, after the loop).

    Returns prev_sample: (B, Limg, in_channels) latent for the next step.
    """
    shared, blocks = regroup_weights(weights, configs.num_layers)

    # stack cond + uncond so the denoiser runs once over a 2B batch
    latent_in = np.concatenate([latents, latents], axis=0)
    text_in = np.concatenate([cond_text, neg_text], axis=0)
    ts_in = np.concatenate([timestep, timestep], axis=0)

    txt_mask_bias = None
    if cond_mask is not None:
        mask = np.concatenate([cond_mask, neg_mask], axis=0)
        txt_mask_bias = (1.0 - mask.astype(configs.dtype)) * np.float32(-1e9)
        txt_mask_bias = np.expand_dims(txt_mask_bias, axis=[1, 2])

    noise = _core(latent_in, text_in, ts_in, img_shape, shared, blocks, configs,
                  txt_mask_bias=txt_mask_bias)
    noise = noise.astype(np.float32)
    noise_cond, noise_uncond = np.split(noise, 2, axis=0)

    # true CFG: combine then rescale to the conditional norm (per-token, over ch)
    cfg_scale = coeffs[0]
    comb = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
    cond_norm = np.sqrt(np.sum(noise_cond * noise_cond, axis=-1, keepdims=True))
    comb_norm = np.sqrt(np.sum(comb * comb, axis=-1, keepdims=True))
    model_output = comb * (cond_norm / comb_norm)

    # FlowMatchEuler (non-stochastic): prev = sample + dt * model_output
    sample = latents.astype(np.float32)
    prev_sample = sample + coeffs[1] * model_output
    return prev_sample.astype(configs.dtype)
