"""Qwen-Image VAE decoder (latents -> pixels) on Trainium.

``AutoencoderKLQwenImage`` is a WAN-style 3D **causal video** VAE. For
text-to-image we decode a **single frame** (T=1), and at T=1 the entire temporal
machinery collapses to a plain 2D conv decoder (verified numerically to 0.0 diff
against diffusers ``vae.decode``):

* each ``QwenImageCausalConv3d`` (3x3x3, causal left-pad of 2 in time) sees a
  T=1 input, so the two leading temporal taps multiply zero-padding and only the
  **last** temporal tap contributes -> a 2D conv with weight ``w[:, :, -1]``. The
  weight extraction bakes this collapse in, so the kernel just calls ``conv2d``.
* ``feat_cache`` stays ``None`` for the single (first) frame, and the
  ``upsample3d`` resamplers' temporal ``time_conv`` is on the "Rep" first-frame
  path and is **skipped** entirely. Spatial upsampling is nearest-2x + conv3x3.

Decoder structure (diffusers ``QwenImageDecoder3d``, Qwen-Image config
base_dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2):
    conv_in (3x3)
    mid_block: resnet -> attention (1x1 conv, single head) -> resnet
    4x up_block: (num_res_blocks+1) residual blocks, then (all but last)
        nearest-2x upsample + conv3x3
    norm_out (RMS over channels) -> SiLU -> conv_out (3x3)

Nonlinearity is SiLU. Norm is ``QwenImageRMS_norm``: L2-normalize over the
channel dim, then ``* sqrt(dim) * gamma`` (no bias in this checkpoint). Weights
arrive flat via ``**weights`` (see ``vae_weight_layout``). Runs in fp32 (the
decoder is numerically sensitive and one-shot at the end of sampling); conv is
the weak path on the PE array, so this is correctness-first, not a perf win.
"""

import numpy as np
from nkipy.core import tensor_apis


def _silu(x):
    # sigmoid via clip to avoid overflow on large-magnitude activations
    return x * (1 / (1 + np.exp(-np.clip(x, -30, 30))))


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def rms_norm(x, gamma, eps=1e-12):
    """QwenImageRMS_norm over (B, C, H, W): L2-normalize channels * sqrt(C) * gamma.

    ``F.normalize(x, dim=1)`` == x / sqrt(sum_c x^2 + eps); the layer then scales
    by ``sqrt(dim)`` and the per-channel ``gamma`` (no bias in this checkpoint).
    Computed in fp32.
    """
    dtype = x.dtype
    B, C, H, W = x.shape
    xf = x.astype(np.float32)
    denom = np.sqrt(np.sum(np.square(xf), axis=1, keepdims=True) + eps)
    normalized = xf / denom
    scale = np.float32(np.sqrt(C))
    g = gamma.astype(np.float32).reshape(1, C, 1, 1)
    return (normalized * scale * g).astype(dtype)


def conv2d(x, w, b, stride=1, padding=1):
    out = tensor_apis.conv2d(x, w, stride=stride, padding=padding)
    return out + b.reshape(1, -1, 1, 1)


def resnet_block(x, w, prefix):
    """QwenImageResidualBlock: (RMS-SiLU-conv3x3) x2 + shortcut.

    The shortcut is a 1x1 conv when in/out channels differ, else identity
    (diffusers uses ``nn.Identity`` -> no shortcut weight in the flat dict).
    """
    if prefix + "conv_shortcut.weight" in w:
        shortcut = conv2d(x, w[prefix + "conv_shortcut.weight"],
                          w[prefix + "conv_shortcut.bias"], padding=0)
    else:
        shortcut = x

    h = rms_norm(x, w[prefix + "norm1.gamma"])
    h = _silu(h)
    h = conv2d(h, w[prefix + "conv1.weight"], w[prefix + "conv1.bias"], padding=1)
    h = rms_norm(h, w[prefix + "norm2.gamma"])
    h = _silu(h)
    h = conv2d(h, w[prefix + "conv2.weight"], w[prefix + "conv2.bias"], padding=1)
    return shortcut + h


def attention(x, w, prefix):
    """QwenImageAttentionBlock: single-head spatial self-attention.

    RMS-norm, then 1x1-conv qkv (a per-pixel linear), attention over H*W, 1x1-conv
    proj, residual. The 1x1 convs are applied as matmuls over the (HW, C) layout.
    """
    B, C, H, W = x.shape
    h = rms_norm(x, w[prefix + "norm.gamma"])
    # (B, C, H, W) -> (B, HW, C)
    seq = h.reshape(B, C, H * W).transpose(0, 2, 1)
    # to_qkv: 1x1 conv (out 3C) == linear on channels. Weight stored (in, 3C).
    qkv = np.matmul(seq, w[prefix + "to_qkv.weight"]) + w[prefix + "to_qkv.bias"]
    q, k, v = np.split(qkv, 3, axis=-1)

    scores = (q @ k.transpose(0, 2, 1)).astype(np.float32) / np.float32(np.sqrt(C))
    attn = _softmax(scores).astype(v.dtype)
    out = attn @ v  # (B, HW, C)
    out = np.matmul(out, w[prefix + "proj.weight"]) + w[prefix + "proj.bias"]

    out = out.transpose(0, 2, 1).reshape(B, C, H, W)
    return x + out


def upsample_nearest(x):
    """Nearest-exact 2x spatial upsampling via repeat (QwenImageUpsample)."""
    x = np.repeat(x, 2, axis=2)
    x = np.repeat(x, 2, axis=3)
    return x


def vae_decode(latents, configs, **weights):
    """Decode (1, z_dim, h, w) latents to (1, 3, H, W) pixels in [-1, 1].

    ``latents`` must already be denormalized (``z * std + mean``) on the host,
    matching ``QwenImagePipeline`` (the driver does this before calling). The
    single-frame temporal dim is squeezed out on the host so the kernel is pure
    2D; output is the T=1 frame (1, 3, H, W).
    """
    from .vae_weight_layout import regroup_vae_weights

    w = regroup_vae_weights(weights)
    nrb = configs.vae_num_res_blocks
    n_up = len(configs.vae_dim_mult)

    # post_quant_conv (1x1) then conv_in (3x3)
    x = conv2d(latents, w["post_quant.weight"], w["post_quant.bias"], padding=0)
    x = conv2d(x, w["conv_in.weight"], w["conv_in.bias"], padding=1)

    # mid block: resnet, attention, resnet
    x = resnet_block(x, w, "mid.resnets.0.")
    x = attention(x, w, "mid.attentions.0.")
    x = resnet_block(x, w, "mid.resnets.1.")

    # up blocks: (nrb + 1) residual blocks, then spatial upsample (all but last)
    for i in range(n_up):
        for j in range(nrb + 1):
            x = resnet_block(x, w, f"up.{i}.resnets.{j}.")
        if f"up.{i}.upsample.weight" in w:
            x = upsample_nearest(x)
            x = conv2d(x, w[f"up.{i}.upsample.weight"],
                       w[f"up.{i}.upsample.bias"], padding=1)

    # output head: RMS-norm -> SiLU -> conv_out (3x3)
    x = rms_norm(x, w["norm_out.gamma"])
    x = _silu(x)
    x = conv2d(x, w["conv_out.weight"], w["conv_out.bias"], padding=1)
    return x
