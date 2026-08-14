"""Flat weight-key scheme for the VAE decoder kernel.

Keys mirror the module tree with short prefixes:
    post_quant.{weight,bias}, conv_in.*, conv_out.*, norm_out.*
    mid.resnets.{0,1}.*, mid.attentions.0.*
    up.{i}.resnets.{j}.*, up.{i}.upsample.{weight,bias}

Conv weights are kept in (out, in, kh, kw) layout (nkipy conv2d consumes them
directly); attention linears are transposed to (in, out) for ``x @ W`` in prep.
``regroup_vae_weights`` is an identity pass-through (keys are already flat) so
the kernel and prep share one naming source of truth.
"""


def regroup_vae_weights(flat):
    # keys are already the canonical flat names; nothing to regroup
    return dict(flat)
