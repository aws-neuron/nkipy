"""SD VAE decoder (latents -> pixels) on Trainium.

Structure (diffusers AutoencoderKL decoder, SD config):
    post_quant_conv (1x1)
    conv_in (3x3)
    mid_block: resnet -> spatial self-attention -> resnet
    4x UpDecoderBlock2D: (layers_per_block+1) resnets, then nearest-2x upsample
        + 3x3 conv (all but the last block)
    GroupNorm(32) -> SiLU -> conv_out (3x3)

Everything is convolutional, which is the weakest path on the matmul-oriented PE
array (nkipy conv2d has no perf tuning) — this is a one-shot correctness-first
offload, run once at the end of sampling.

Weights arrive flat via ``**weights`` (see vae_weight_layout).
"""

import numpy as np
from nkipy.core import tensor_apis


def _silu(x):
    # sigmoid via clip to avoid overflow warnings on large-magnitude activations
    return x * (1 / (1 + np.exp(-np.clip(x, -30, 30))))


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def group_norm(x, weight, bias, num_groups=32, eps=1e-6):
    """GroupNorm over (B, C, H, W). Computed in fp32."""
    dtype = x.dtype
    B, C, H, W = x.shape
    xf = x.astype(np.float32).reshape(B, num_groups, C // num_groups, H, W)
    mean = np.mean(xf, axis=(2, 3, 4), keepdims=True)
    var = np.mean(np.square(xf - mean), axis=(2, 3, 4), keepdims=True)
    xf = (xf - mean) / np.sqrt(var + eps)
    xf = xf.reshape(B, C, H, W)
    w = weight.astype(np.float32).reshape(1, C, 1, 1)
    b = bias.astype(np.float32).reshape(1, C, 1, 1)
    return (xf * w + b).astype(dtype)


def conv2d(x, w, b, stride=1, padding=1):
    out = tensor_apis.conv2d(x, w, stride=stride, padding=padding)
    return out + b.reshape(1, -1, 1, 1)


def resnet_block(x, w, prefix, num_groups, eps):
    """diffusers ResnetBlock2D: GN-SiLU-conv3x3 -> GN-SiLU-conv3x3 + shortcut."""
    h = group_norm(x, w[prefix + "norm1.weight"], w[prefix + "norm1.bias"], num_groups, eps)
    h = _silu(h)
    h = conv2d(h, w[prefix + "conv1.weight"], w[prefix + "conv1.bias"], padding=1)
    h = group_norm(h, w[prefix + "norm2.weight"], w[prefix + "norm2.bias"], num_groups, eps)
    h = _silu(h)
    h = conv2d(h, w[prefix + "conv2.weight"], w[prefix + "conv2.bias"], padding=1)

    if prefix + "conv_shortcut.weight" in w:
        x = conv2d(x, w[prefix + "conv_shortcut.weight"],
                   w[prefix + "conv_shortcut.bias"], padding=0)
    return x + h


def spatial_attention(x, w, prefix, num_groups, eps):
    """Single-head spatial self-attention over H*W (diffusers Attention, mid-block)."""
    B, C, H, W = x.shape
    h = group_norm(x, w[prefix + "group_norm.weight"], w[prefix + "group_norm.bias"],
                   num_groups, eps)
    # (B, C, H, W) -> (B, HW, C)
    seq = h.reshape(B, C, H * W).transpose(0, 2, 1)
    q = np.matmul(seq, w[prefix + "to_q.weight"]) + w[prefix + "to_q.bias"]
    k = np.matmul(seq, w[prefix + "to_k.weight"]) + w[prefix + "to_k.bias"]
    v = np.matmul(seq, w[prefix + "to_v.weight"]) + w[prefix + "to_v.bias"]

    scores = (q @ k.transpose(0, 2, 1)).astype(np.float32) / np.float32(np.sqrt(C))
    attn = _softmax(scores).astype(v.dtype)
    out = attn @ v  # (B, HW, C)
    out = np.matmul(out, w[prefix + "to_out.weight"]) + w[prefix + "to_out.bias"]

    out = out.transpose(0, 2, 1).reshape(B, C, H, W)
    return x + out


def upsample_nearest(x):
    """Nearest-neighbour 2x upsampling via repeat (matches diffusers Upsample2D)."""
    x = np.repeat(x, 2, axis=2)
    x = np.repeat(x, 2, axis=3)
    return x


def vae_decode(latents, configs, **weights):
    """Decode (B, 4, h, w) latents to (B, 3, H, W) pixels in [-1, 1]-ish range.

    ``latents`` should already be divided by the scaling factor (done on host).
    """
    from .vae_weight_layout import regroup_vae_weights

    w = regroup_vae_weights(weights)
    ng = configs.vae_norm_groups
    eps = configs.vae_eps
    lpb = configs.vae_layers_per_block
    block_out = configs.vae_block_out_channels  # e.g. [128,256,512,512]
    n_up = len(block_out)

    # post_quant_conv (1x1) then conv_in (3x3)
    x = conv2d(latents, w["post_quant.weight"], w["post_quant.bias"], padding=0)
    x = conv2d(x, w["conv_in.weight"], w["conv_in.bias"], padding=1)

    # mid block: resnet, attention, resnet
    x = resnet_block(x, w, "mid.resnets.0.", ng, eps)
    x = spatial_attention(x, w, "mid.attentions.0.", ng, eps)
    x = resnet_block(x, w, "mid.resnets.1.", ng, eps)

    # up blocks (diffusers iterates reversed block_out; decoder blocks are stored
    # in that reversed order already)
    for i in range(n_up):
        for j in range(lpb + 1):
            x = resnet_block(x, w, f"up.{i}.resnets.{j}.", ng, eps)
        if f"up.{i}.upsample.weight" in w:  # all but the last block upsample
            x = upsample_nearest(x)
            x = conv2d(x, w[f"up.{i}.upsample.weight"],
                       w[f"up.{i}.upsample.bias"], padding=1)

    # output head
    x = group_norm(x, w["norm_out.weight"], w["norm_out.bias"], ng, eps)
    x = _silu(x)
    x = conv2d(x, w["conv_out.weight"], w["conv_out.bias"], padding=1)
    return x
