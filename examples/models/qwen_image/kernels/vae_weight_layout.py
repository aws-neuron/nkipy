"""Flat weight-key scheme for the Qwen-Image VAE decoder kernel.

Keys mirror the ``QwenImageDecoder3d`` module tree with short prefixes:
    conv_in.*, conv_out.*, norm_out.gamma
    mid.resnets.{0,1}.*, mid.attentions.0.*
    up.{i}.resnets.{j}.*, up.{i}.upsample.{weight,bias}

Conv weights are collapsed from 3D (out, in, kt, kh, kw) to 2D (out, in, kh, kw)
during extraction (the T=1 last-temporal-tap collapse; see ``kernels/vae.py``),
so the kernel consumes plain (out, in, kh, kw) directly via nkipy conv2d. The
attention ``to_qkv``/``proj`` are 1x1 convs stored as (in, out) matmul weights.
``regroup_vae_weights`` is an identity pass-through (keys are already flat) so
the kernel and prep share one naming source of truth.
"""


def regroup_vae_weights(flat):
    # keys are already the canonical flat names; nothing to regroup
    return dict(flat)
