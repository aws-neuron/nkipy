"""Map a diffusers ``QwenImageTransformer2DModel`` state dict to the flat
canonical key scheme (``kernels/weight_layout.py``).

Torch ``nn.Linear`` weights are stored as (out, in); our numpy kernels use the
``x @ W`` convention, so linear weights are transposed to (in, out). RMSNorm /
LayerNorm gains and all biases pass through unchanged.

Kept dependency-light (numpy/torch only) so it can run against an in-memory
model — the driver, the reduced-config tests, and ``tests/test_tp_device.py``
all call these functions directly on a loaded diffusers module.
"""

import numpy as np
import torch

from kernels.tp import shard_heads, shard_intermediate
from kernels.weight_layout import BLOCK_KEYS, SHARED_KEYS, block_key


def _np(p):
    """torch tensor -> numpy, routing bf16 through fp32 (torch can't .numpy() bf16)."""
    p = p.detach().cpu()
    if p.dtype == torch.bfloat16:
        p = p.to(torch.float32)
    return p.numpy()


def _t(p):
    """torch Linear weight (out,in) -> numpy (in,out)."""
    return _np(p).T.copy()


def _v(p):
    return _np(p).copy()


def _block_weights(sd, i):
    """Extract one block's weights (canonical short keys) from a state dict.

    ``sd`` keys are the ``transformer_blocks.{i}.`` submodule names.
    """
    def g(name):
        return sd[f"transformer_blocks.{i}.{name}"]

    return {
        "img_mod_w": _t(g("img_mod.1.weight")), "img_mod_b": _v(g("img_mod.1.bias")),
        "txt_mod_w": _t(g("txt_mod.1.weight")), "txt_mod_b": _v(g("txt_mod.1.bias")),
        "iq_w": _t(g("attn.to_q.weight")), "iq_b": _v(g("attn.to_q.bias")),
        "ik_w": _t(g("attn.to_k.weight")), "ik_b": _v(g("attn.to_k.bias")),
        "iv_w": _t(g("attn.to_v.weight")), "iv_b": _v(g("attn.to_v.bias")),
        "io_w": _t(g("attn.to_out.0.weight")), "io_b": _v(g("attn.to_out.0.bias")),
        "iq_g": _v(g("attn.norm_q.weight")), "ik_g": _v(g("attn.norm_k.weight")),
        "tq_w": _t(g("attn.add_q_proj.weight")), "tq_b": _v(g("attn.add_q_proj.bias")),
        "tk_w": _t(g("attn.add_k_proj.weight")), "tk_b": _v(g("attn.add_k_proj.bias")),
        "tv_w": _t(g("attn.add_v_proj.weight")), "tv_b": _v(g("attn.add_v_proj.bias")),
        "to_w": _t(g("attn.to_add_out.weight")), "to_b": _v(g("attn.to_add_out.bias")),
        "tq_g": _v(g("attn.norm_added_q.weight")), "tk_g": _v(g("attn.norm_added_k.weight")),
        "iff0_w": _t(g("img_mlp.net.0.proj.weight")), "iff0_b": _v(g("img_mlp.net.0.proj.bias")),
        "iff2_w": _t(g("img_mlp.net.2.weight")), "iff2_b": _v(g("img_mlp.net.2.bias")),
        "tff0_w": _t(g("txt_mlp.net.0.proj.weight")), "tff0_b": _v(g("txt_mlp.net.0.proj.bias")),
        "tff2_w": _t(g("txt_mlp.net.2.weight")), "tff2_b": _v(g("txt_mlp.net.2.bias")),
    }


def extract_flat_weights(model, num_layers, dtype=np.float32):
    """Return a flat ``{key: ndarray}`` dict for ``qwenimage_forward``.

    Args:
        model: a diffusers ``QwenImageTransformer2DModel`` (or any module whose
            ``named_parameters`` follow the same naming).
        num_layers: number of transformer blocks to extract.
    """
    sd = dict(model.named_parameters())

    flat = {
        "img_in.weight": _t(sd["img_in.weight"]), "img_in.bias": _v(sd["img_in.bias"]),
        "txt_norm.weight": _v(sd["txt_norm.weight"]),
        "txt_in.weight": _t(sd["txt_in.weight"]), "txt_in.bias": _v(sd["txt_in.bias"]),
        "time.proj.weight": _t(sd["time_text_embed.timestep_embedder.linear_1.weight"]),
        "time.proj.bias": _v(sd["time_text_embed.timestep_embedder.linear_1.bias"]),
        "time.emb.weight": _t(sd["time_text_embed.timestep_embedder.linear_2.weight"]),
        "time.emb.bias": _v(sd["time_text_embed.timestep_embedder.linear_2.bias"]),
        "norm_out.linear.weight": _t(sd["norm_out.linear.weight"]),
        "norm_out.linear.bias": _v(sd["norm_out.linear.bias"]),
        "proj_out.weight": _t(sd["proj_out.weight"]), "proj_out.bias": _v(sd["proj_out.bias"]),
    }

    for i in range(num_layers):
        for short, val in _block_weights(sd, i).items():
            flat[block_key(i, short)] = val

    if dtype is not None:
        flat = {k: v.astype(dtype) for k, v in flat.items()}
    return flat


# per-block key -> (kind, axis) sharding rule. Column-parallel (axis 1) shards
# the output dim; row-parallel (axis 0) shards the input dim. Keys not listed
# (modulation, QK-norm gains) are replicated on every rank.
_HEAD_COL = ("head", 1)   # q/k/v: shard output (head) dim + bias
_HEAD_ROW = ("head", 0)   # o: shard input (head) dim; bias replicated
_INT_COL = ("inter", 1)   # ff0: shard intermediate dim + bias
_INT_ROW = ("inter", 0)   # ff2: shard intermediate (input) dim; bias replicated

_SHARD_RULES = {
    "iq_w": _HEAD_COL, "ik_w": _HEAD_COL, "iv_w": _HEAD_COL, "io_w": _HEAD_ROW,
    "tq_w": _HEAD_COL, "tk_w": _HEAD_COL, "tv_w": _HEAD_COL, "to_w": _HEAD_ROW,
    "iff0_w": _INT_COL, "iff2_w": _INT_ROW,
    "tff0_w": _INT_COL, "tff2_w": _INT_ROW,
}
# bias that rides along with a column-parallel weight (sharded the same way)
_BIAS_OF = {
    "iq_w": "iq_b", "ik_w": "ik_b", "iv_w": "iv_b", "io_w": "io_b",
    "tq_w": "tq_b", "tk_w": "tk_b", "tv_w": "tv_b", "to_w": "to_b",
    "iff0_w": "iff0_b", "iff2_w": "iff2_b", "tff0_w": "tff0_b", "tff2_w": "tff2_b",
}


def shard_flat_weights(flat, rank, tp_size, num_layers, n_heads, head_dim):
    """Slice a full flat weight dict into ``rank``'s tensor-parallel shard.

    Attention q/k/v/o shard by heads, MLP ff0/ff2 by intermediate; everything
    else (modulation, norms, QK-norm gains, shared input/output projections) is
    replicated. Row-parallel biases (o, ff2) stay full — they are added once
    after the all-reduce (see the kernels). Returns a new flat dict.
    """
    if tp_size <= 1:
        return dict(flat)

    out = {}
    # shared weights are replicated
    for k in SHARED_KEYS:
        if k in flat:
            out[k] = flat[k]

    for i in range(num_layers):
        for short in BLOCK_KEYS:
            key = block_key(i, short)
            if key not in flat:
                continue
            rule = _SHARD_RULES.get(short)
            if rule is None:
                # replicated (modulation weights/biases, QK-norm gains, and the
                # biases already handled alongside their weight)
                if short not in _BIAS_OF.values():
                    out[key] = flat[key]
                continue
            kind, axis = rule
            w = flat[key]
            b_short = _BIAS_OF[short]
            b = flat.get(block_key(i, b_short))
            if kind == "head":
                sw, sb = shard_heads(w, b, rank, tp_size, n_heads, head_dim, axis)
            else:
                sw, sb = shard_intermediate(w, b, rank, tp_size, axis)
            out[key] = sw
            if axis == 1:
                # column-parallel: bias sharded with the weight
                out[block_key(i, b_short)] = sb
            elif b is not None:
                # row-parallel: keep the full replicated bias
                out[block_key(i, b_short)] = b
    return out


# ── text encoder (Qwen2.5 LM) ────────────────────────────────────────────────

def extract_text_encoder_weights(lm, num_layers, dtype=np.float32):
    """Flatten a Qwen2.5 text-model (``te.model.language_model``) to flat keys.

    ``lm`` is the ``Qwen2_5_VLTextModel``; we take its decoder layers + final
    norm. The embedding table / LM head are left on host.
    """
    from kernels.text_weight_layout import layer_key

    sd = dict(lm.named_parameters())
    out = {"final_norm": _v(sd["norm.weight"])}
    for i in range(num_layers):
        p = f"layers.{i}."
        out[layer_key(i, "attn_norm")] = _v(sd[p + "input_layernorm.weight"])
        out[layer_key(i, "mlp_norm")] = _v(sd[p + "post_attention_layernorm.weight"])
        out[layer_key(i, "q_w")] = _t(sd[p + "self_attn.q_proj.weight"])
        out[layer_key(i, "q_b")] = _v(sd[p + "self_attn.q_proj.bias"])
        out[layer_key(i, "k_w")] = _t(sd[p + "self_attn.k_proj.weight"])
        out[layer_key(i, "k_b")] = _v(sd[p + "self_attn.k_proj.bias"])
        out[layer_key(i, "v_w")] = _t(sd[p + "self_attn.v_proj.weight"])
        out[layer_key(i, "v_b")] = _v(sd[p + "self_attn.v_proj.bias"])
        out[layer_key(i, "o_w")] = _t(sd[p + "self_attn.o_proj.weight"])
        out[layer_key(i, "gate_w")] = _t(sd[p + "mlp.gate_proj.weight"])
        out[layer_key(i, "up_w")] = _t(sd[p + "mlp.up_proj.weight"])
        out[layer_key(i, "down_w")] = _t(sd[p + "mlp.down_proj.weight"])
    if dtype is not None:
        out = {k: v.astype(dtype) for k, v in out.items()}
    return out


# ── VAE decoder (AutoencoderKLQwenImage, T=1 2D collapse) ────────────────────

def _conv3d_to_2d(p):
    """Collapse a causal-conv3d weight (out, in, kt, kh, kw) to 2D (out, in, kh,
    kw) by taking the last temporal tap. At T=1 the causal left-padding zeros the
    earlier taps, so only ``w[:, :, -1]`` contributes (verified 0.0 diff)."""
    return _np(p)[:, :, -1, :, :].copy()


def _conv1x1_to_linear(p):
    """A 1x1 ``nn.Conv2d`` weight (out, in, 1, 1) -> (in, out) matmul weight."""
    return _np(p)[:, :, 0, 0].T.copy()


def extract_vae_decoder_weights(vae, dtype=np.float32):
    """Flatten an ``AutoencoderKLQwenImage`` decoder to the flat 2D key scheme.

    Collapses every 3D causal conv to a 2D conv (last temporal tap); the
    attention 1x1 convs become (in, out) matmul weights. ``post_quant_conv`` (a
    1x1 conv applied to the latent before the decoder proper) is included so the
    kernel input is the raw (denormalized) latent. The encoder / quant_conv are
    unused for decoding.
    """
    sd = dict(vae.named_parameters())
    dec = "decoder."
    cfg_mult = vae.config.dim_mult
    nrb = vae.config.num_res_blocks

    out = {}

    def conv3(dst, src):
        out[dst + ".weight"] = _conv3d_to_2d(sd[src + ".weight"])
        out[dst + ".bias"] = _v(sd[src + ".bias"])

    # post_quant_conv (1x1x1 causal-conv3d) applied to the latent before the
    # decoder proper -> 2D 1x1 conv
    out["post_quant.weight"] = _conv3d_to_2d(sd["post_quant_conv.weight"])
    out["post_quant.bias"] = _v(sd["post_quant_conv.bias"])

    conv3("conv_in", dec + "conv_in")
    conv3("conv_out", dec + "conv_out")
    out["norm_out.gamma"] = _v(sd[dec + "norm_out.gamma"])

    # mid block: resnets 0/1 + attention 0
    for j in (0, 1):
        _extract_resnet(sd, out, f"mid.resnets.{j}.", dec + f"mid_block.resnets.{j}.")
    ap = dec + "mid_block.attentions.0."
    out["mid.attentions.0.norm.gamma"] = _v(sd[ap + "norm.gamma"])
    out["mid.attentions.0.to_qkv.weight"] = _conv1x1_to_linear(sd[ap + "to_qkv.weight"])
    out["mid.attentions.0.to_qkv.bias"] = _v(sd[ap + "to_qkv.bias"])
    out["mid.attentions.0.proj.weight"] = _conv1x1_to_linear(sd[ap + "proj.weight"])
    out["mid.attentions.0.proj.bias"] = _v(sd[ap + "proj.bias"])

    # up blocks
    n_up = len(cfg_mult)
    for i in range(n_up):
        up = dec + f"up_blocks.{i}."
        for j in range(nrb + 1):
            _extract_resnet(sd, out, f"up.{i}.resnets.{j}.", up + f"resnets.{j}.")
        # spatial upsampler conv (resample.1); time_conv is skipped at T=1
        usw = up + "upsamplers.0.resample.1.weight"
        if usw in sd:
            out[f"up.{i}.upsample.weight"] = _np(sd[usw]).copy()  # already 2D conv
            out[f"up.{i}.upsample.bias"] = _v(sd[up + "upsamplers.0.resample.1.bias"])

    if dtype is not None:
        out = {k: v.astype(dtype) for k, v in out.items()}
    return out


def _extract_resnet(sd, out, dst, src):
    """One QwenImageResidualBlock -> flat 2D keys (norm gammas, conv1/conv2, and
    an optional 1x1 conv_shortcut)."""
    out[dst + "norm1.gamma"] = _v(sd[src + "norm1.gamma"])
    out[dst + "norm2.gamma"] = _v(sd[src + "norm2.gamma"])
    out[dst + "conv1.weight"] = _conv3d_to_2d(sd[src + "conv1.weight"])
    out[dst + "conv1.bias"] = _v(sd[src + "conv1.bias"])
    out[dst + "conv2.weight"] = _conv3d_to_2d(sd[src + "conv2.weight"])
    out[dst + "conv2.bias"] = _v(sd[src + "conv2.bias"])
    if src + "conv_shortcut.weight" in sd:
        # shortcut is a 1x1x1 causal-conv3d -> 2D 1x1 conv (out, in, 1, 1)
        out[dst + "conv_shortcut.weight"] = _conv3d_to_2d(sd[src + "conv_shortcut.weight"])
        out[dst + "conv_shortcut.bias"] = _v(sd[src + "conv_shortcut.bias"])


def shard_text_encoder_weights(flat, rank, tp_size, num_layers, n_heads,
                               n_kv_heads, head_dim):
    """Slice the flat text-encoder weights into ``rank``'s TP shard.

    Megatron-style, mirroring the denoiser (``shard_flat_weights``): attention
    q/o shard by query heads, k/v shard by KV heads (GQA), and the SwiGLU MLP
    shards by intermediate (gate/up column-parallel, down row-parallel). q/k/v
    biases ride with their column-parallel weight; the row-parallel o/down have
    no bias in Qwen2.5. RMSNorm gains (attn_norm/mlp_norm/final_norm) replicate.

    Requires ``tp_size`` to divide both ``n_heads`` and ``n_kv_heads`` (so
    ``tp_size <= n_kv_heads``; the 4-KV-head encoder shards up to TP=4).
    """
    from kernels.text_weight_layout import SHARED_KEYS, layer_key
    from kernels.tp import shard_heads, shard_intermediate

    if tp_size <= 1:
        return dict(flat)
    if n_heads % tp_size or n_kv_heads % tp_size:
        raise ValueError(
            f"text-encoder TP={tp_size} must divide n_heads={n_heads} and "
            f"n_kv_heads={n_kv_heads} (KV heads cap TP at {n_kv_heads})")

    out = {k: flat[k] for k in SHARED_KEYS if k in flat}
    for i in range(num_layers):
        # replicated norms
        for s in ("attn_norm", "mlp_norm"):
            out[layer_key(i, s)] = flat[layer_key(i, s)]
        # attention: q/o by query heads, k/v by KV heads
        for w_short, b_short, nh in (("q_w", "q_b", n_heads),
                                     ("k_w", "k_b", n_kv_heads),
                                     ("v_w", "v_b", n_kv_heads)):
            sw, sb = shard_heads(flat[layer_key(i, w_short)],
                                 flat[layer_key(i, b_short)],
                                 rank, tp_size, nh, head_dim, axis=1)
            out[layer_key(i, w_short)] = sw
            out[layer_key(i, b_short)] = sb
        ow, _ = shard_heads(flat[layer_key(i, "o_w")], None,
                            rank, tp_size, n_heads, head_dim, axis=0)
        out[layer_key(i, "o_w")] = ow
        # SwiGLU MLP: gate/up column-parallel, down row-parallel
        for w_short in ("gate_w", "up_w"):
            sw, _ = shard_intermediate(flat[layer_key(i, w_short)], None,
                                       rank, tp_size, axis=1)
            out[layer_key(i, w_short)] = sw
        dw, _ = shard_intermediate(flat[layer_key(i, "down_w")], None,
                                   rank, tp_size, axis=0)
        out[layer_key(i, "down_w")] = dw
    return out
