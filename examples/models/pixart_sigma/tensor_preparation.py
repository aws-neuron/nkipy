"""Download a diffusers PixArt-Sigma checkpoint and repack its transformer
weights into the flat canonical key scheme used by the NKIPy kernels.

Two conversions matter:

* ``nn.Linear`` weights are stored ``(out, in)`` but the kernels compute
  ``x @ W`` with ``W`` shaped ``(in, out)``, so every linear weight is
  transposed here. The patch-embed conv weight is left as ``(hidden, C, p, p)``
  because ``patch_embed_kernel`` reshapes it itself.
* Everything is saved into a single ``weights.safetensors`` under the output
  directory, keyed by ``kernels/weight_layout.py`` names.

Run:
    python tensor_preparation.py \
        --model-name PixArt-alpha/PixArt-Sigma-XL-2-1024-MS \
        --output-dir ./tmp_pixart_sigma
"""

import argparse
import os

import torch
from kernels.weight_layout import block_key
from safetensors.torch import save_file


def _t(sd, key):
    """Fetch a linear weight and transpose (out, in) -> (in, out)."""
    return sd[key].t().contiguous()


def _b(sd, key):
    return sd[key].contiguous() if key in sd else None


def convert_transformer(sd, num_layers):
    """Map a diffusers PixArtTransformer2DModel state_dict to canonical keys."""
    out = {}

    # ── patch embed (conv weight kept as-is) ──
    out["pos_embed.proj.weight"] = sd["pos_embed.proj.weight"].contiguous()
    out["pos_embed.proj.bias"] = sd["pos_embed.proj.bias"].contiguous()

    # ── adaLN-single ──
    # emb.timestep_embedder is a Linear-SiLU-Linear MLP (linear_1, linear_2)
    out["adaln.time_proj.weight"] = _t(sd, "adaln_single.emb.timestep_embedder.linear_1.weight")
    out["adaln.time_proj.bias"] = _b(sd, "adaln_single.emb.timestep_embedder.linear_1.bias")
    out["adaln.time_emb.weight"] = _t(sd, "adaln_single.emb.timestep_embedder.linear_2.weight")
    out["adaln.time_emb.bias"] = _b(sd, "adaln_single.emb.timestep_embedder.linear_2.bias")
    out["adaln.linear.weight"] = _t(sd, "adaln_single.linear.weight")
    out["adaln.linear.bias"] = _b(sd, "adaln_single.linear.bias")

    # ── caption projection (PixArtAlphaTextProjection: linear_1, linear_2) ──
    out["caption.w1"] = _t(sd, "caption_projection.linear_1.weight")
    out["caption.b1"] = _b(sd, "caption_projection.linear_1.bias")
    out["caption.w2"] = _t(sd, "caption_projection.linear_2.weight")
    out["caption.b2"] = _b(sd, "caption_projection.linear_2.bias")

    # ── final layer ──
    out["scale_shift_table"] = sd["scale_shift_table"].contiguous()
    out["proj_out.weight"] = _t(sd, "proj_out.weight")
    out["proj_out.bias"] = _b(sd, "proj_out.bias")

    # ── transformer blocks ──
    for i in range(num_layers):
        p = f"transformer_blocks.{i}."
        m = {
            "sst": sd[p + "scale_shift_table"].contiguous(),
            # self-attention (attn1); to_out is a ModuleList -> to_out.0
            "q_w": _t(sd, p + "attn1.to_q.weight"), "q_b": _b(sd, p + "attn1.to_q.bias"),
            "k_w": _t(sd, p + "attn1.to_k.weight"), "k_b": _b(sd, p + "attn1.to_k.bias"),
            "v_w": _t(sd, p + "attn1.to_v.weight"), "v_b": _b(sd, p + "attn1.to_v.bias"),
            "o_w": _t(sd, p + "attn1.to_out.0.weight"), "o_b": _b(sd, p + "attn1.to_out.0.bias"),
            # cross-attention (attn2)
            "cq_w": _t(sd, p + "attn2.to_q.weight"), "cq_b": _b(sd, p + "attn2.to_q.bias"),
            "ck_w": _t(sd, p + "attn2.to_k.weight"), "ck_b": _b(sd, p + "attn2.to_k.bias"),
            "cv_w": _t(sd, p + "attn2.to_v.weight"), "cv_b": _b(sd, p + "attn2.to_v.bias"),
            "co_w": _t(sd, p + "attn2.to_out.0.weight"), "co_b": _b(sd, p + "attn2.to_out.0.bias"),
            # feed-forward: GELU-approximate wraps a Linear in .proj; net.2 is Linear
            "ff0_w": _t(sd, p + "ff.net.0.proj.weight"), "ff0_b": _b(sd, p + "ff.net.0.proj.bias"),
            "ff2_w": _t(sd, p + "ff.net.2.weight"), "ff2_b": _b(sd, p + "ff.net.2.bias"),
        }
        for short, tensor in m.items():
            if tensor is not None:
                out[block_key(i, short)] = tensor

    return out


def convert_t5(sd, num_layers):
    """Map a HF T5 encoder state_dict to canonical flat keys (transposed linears)."""
    from kernels.t5_weight_layout import t5_block_key

    def tt(k):
        return sd[k].t().contiguous()

    out = {}
    out["t5_rel_bias"] = sd[
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ].contiguous()  # (num_buckets, n_heads), used as-is
    out["t5_final_ln"] = sd["encoder.final_layer_norm.weight"].contiguous()
    for i in range(num_layers):
        p = f"encoder.block.{i}."
        m = {
            "ln0": sd[p + "layer.0.layer_norm.weight"].contiguous(),
            "q": tt(p + "layer.0.SelfAttention.q.weight"),
            "k": tt(p + "layer.0.SelfAttention.k.weight"),
            "v": tt(p + "layer.0.SelfAttention.v.weight"),
            "o": tt(p + "layer.0.SelfAttention.o.weight"),
            "ln1": sd[p + "layer.1.layer_norm.weight"].contiguous(),
            "wi0": tt(p + "layer.1.DenseReluDense.wi_0.weight"),
            "wi1": tt(p + "layer.1.DenseReluDense.wi_1.weight"),
            "wo": tt(p + "layer.1.DenseReluDense.wo.weight"),
        }
        for short, tensor in m.items():
            out[t5_block_key(i, short)] = tensor
    return out


def convert_vae(sd, num_up_blocks, layers_per_block):
    """Map a diffusers AutoencoderKL decoder state_dict to flat VAE keys.

    Conv weights kept as (out,in,kh,kw); attention linears transposed to (in,out).
    """
    out = {}

    def conv(dst, src):
        out[dst + ".weight"] = sd[src + ".weight"].contiguous()
        out[dst + ".bias"] = sd[src + ".bias"].contiguous()

    def gn(dst, src):
        out[dst + ".weight"] = sd[src + ".weight"].contiguous()
        out[dst + ".bias"] = sd[src + ".bias"].contiguous()

    def lin(dst, src):
        out[dst + ".weight"] = sd[src + ".weight"].t().contiguous()
        out[dst + ".bias"] = sd[src + ".bias"].contiguous()

    conv("post_quant", "post_quant_conv")
    conv("conv_in", "decoder.conv_in")
    conv("conv_out", "decoder.conv_out")
    gn("norm_out", "decoder.conv_norm_out")

    # mid block
    for j in (0, 1):
        _convert_resnet(sd, out, f"mid.resnets.{j}.", f"decoder.mid_block.resnets.{j}.")
    ap = "decoder.mid_block.attentions.0."
    gn("mid.attentions.0.group_norm", ap + "group_norm")
    lin("mid.attentions.0.to_q", ap + "to_q")
    lin("mid.attentions.0.to_k", ap + "to_k")
    lin("mid.attentions.0.to_v", ap + "to_v")
    lin("mid.attentions.0.to_out", ap + "to_out.0")

    # up blocks
    for i in range(num_up_blocks):
        for j in range(layers_per_block + 1):
            _convert_resnet(
                sd, out, f"up.{i}.resnets.{j}.",
                f"decoder.up_blocks.{i}.resnets.{j}.",
            )
        up_key = f"decoder.up_blocks.{i}.upsamplers.0.conv"
        if up_key + ".weight" in sd:
            out[f"up.{i}.upsample.weight"] = sd[up_key + ".weight"].contiguous()
            out[f"up.{i}.upsample.bias"] = sd[up_key + ".bias"].contiguous()
    return out


def _convert_resnet(sd, out, dst, src):
    for part in ("norm1", "norm2", "conv1", "conv2"):
        out[dst + part + ".weight"] = sd[src + part + ".weight"].contiguous()
        out[dst + part + ".bias"] = sd[src + part + ".bias"].contiguous()
    if src + "conv_shortcut.weight" in sd:
        out[dst + "conv_shortcut.weight"] = sd[src + "conv_shortcut.weight"].contiguous()
        out[dst + "conv_shortcut.bias"] = sd[src + "conv_shortcut.bias"].contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--output-dir", default="./tmp_pixart_sigma")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--t5", action="store_true",
                        help="Also convert the T5 encoder (for on-device text encoding)")
    parser.add_argument("--vae", action="store_true",
                        help="Also convert the VAE decoder (for on-device decoding)")
    args = parser.parse_args()

    from diffusers import PixArtTransformer2DModel

    print(f"[prep] loading {args.model_name} transformer")
    model = PixArtTransformer2DModel.from_pretrained(
        args.model_name, subfolder="transformer"
    )
    sd = model.state_dict()
    num_layers = model.config.num_layers

    print(f"[prep] converting {num_layers} blocks")
    out = convert_transformer(sd, num_layers)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in out.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "weights.safetensors")
    save_file(out, path)
    print(f"[prep] wrote {len(out)} tensors to {path}")

    if args.t5:
        from transformers import T5EncoderModel

        print("[prep] loading T5 encoder")
        enc = T5EncoderModel.from_pretrained(args.model_name, subfolder="text_encoder")
        t5_sd = enc.state_dict()
        t5_out = convert_t5(t5_sd, enc.config.num_layers)
        t5_out = {k: (v.to(dtype) if v.is_floating_point() else v)
                  for k, v in t5_out.items()}
        t5_path = os.path.join(args.output_dir, "t5_weights.safetensors")
        save_file(t5_out, t5_path)
        print(f"[prep] wrote {len(t5_out)} T5 tensors to {t5_path}")

    if args.vae:
        from diffusers import AutoencoderKL

        print("[prep] loading VAE")
        vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
        vae_out = convert_vae(
            vae.state_dict(),
            len(vae.config.block_out_channels),
            vae.config.layers_per_block,
        )
        # VAE decode is numerically sensitive; keep it in fp32 regardless of --dtype
        vae_out = {k: (v.to(torch.float32) if v.is_floating_point() else v)
                   for k, v in vae_out.items()}
        vae_path = os.path.join(args.output_dir, "vae_weights.safetensors")
        save_file(vae_out, vae_path)
        print(f"[prep] wrote {len(vae_out)} VAE tensors to {vae_path}")


if __name__ == "__main__":
    main()
