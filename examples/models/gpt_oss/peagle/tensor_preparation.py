#!/usr/bin/env python3
"""Convert the P-EAGLE drafter checkpoint into a replicated safetensors file.

The drafter is small (4 layers, ~3.6 GB bf16) and not quantized. Unlike the
TP-sharded target, the drafter is **replicated** on every rank: the target's
captured aux hidden states are already all-reduced (full) on every rank, so each
rank can run the identical full drafter forward and produce identical draft
tokens with no extra collectives. Prep is therefore just a transpose into the
``x @ W`` form the kernels expect.

Output (single file ``drafter.safetensors``):
  shared: embed_tokens, fc_weight, mask_hidden, norm_weight, lm_head_weight, d2t
  midlayer (fusion): q/k/v/o_proj, input_weight, hidden_norm_weight,
    post_attention_weight, gate/up/down_proj
  layers.{i} (plain, i=1..N-1): same minus hidden_norm_weight
"""

import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


def _attn_block(get, prefix, dtype):
    """Transpose one attention block into fused x@W form (no sharding)."""
    return {
        "q_proj": get(f"{prefix}.self_attn.q_proj.weight").to(dtype).T.contiguous(),
        "k_proj": get(f"{prefix}.self_attn.k_proj.weight").to(dtype).T.contiguous(),
        "v_proj": get(f"{prefix}.self_attn.v_proj.weight").to(dtype).T.contiguous(),
        "o_proj": get(f"{prefix}.self_attn.o_proj.weight").to(dtype).T.contiguous(),
    }


def _mlp_block(get, prefix, dtype):
    return {
        "gate_proj": get(f"{prefix}.mlp.gate_proj.weight").to(dtype).T.contiguous(),
        "up_proj": get(f"{prefix}.mlp.up_proj.weight").to(dtype).T.contiguous(),
        "down_proj": get(f"{prefix}.mlp.down_proj.weight").to(dtype).T.contiguous(),
    }


def build(get, config, dtype):
    n_layers = config.num_hidden_layers
    out = {}

    # Shared tensors.
    out["embed_tokens"] = get("embed_tokens.weight").to(dtype)
    out["fc_weight"] = get("fc.weight").to(dtype).T.contiguous()  # (3H, H) -> x@W
    out["mask_hidden"] = get("mask_hidden").to(dtype).reshape(1, -1).contiguous()
    out["norm_weight"] = get("norm.weight").to(dtype)
    out["lm_head_weight"] = (
        get("lm_head.weight").to(dtype).T.contiguous()
    )  # (H, draft_vocab)
    out["d2t"] = get("d2t").to(torch.int32)

    # Fusion midlayer (layer 0).
    out.update(
        {f"midlayer.{k}": v for k, v in _attn_block(get, "midlayer", dtype).items()}
    )
    out.update(
        {f"midlayer.{k}": v for k, v in _mlp_block(get, "midlayer", dtype).items()}
    )
    out["midlayer.input_weight"] = get("midlayer.input_layernorm.weight").to(dtype)
    out["midlayer.hidden_norm_weight"] = get("midlayer.hidden_norm.weight").to(dtype)
    out["midlayer.post_attention_weight"] = get(
        "midlayer.post_attention_layernorm.weight"
    ).to(dtype)

    # Plain layers 1..N-1.
    for i in range(1, n_layers):
        p = f"layers.{i}"
        out.update({f"{p}.{k}": v for k, v in _attn_block(get, p, dtype).items()})
        out.update({f"{p}.{k}": v for k, v in _mlp_block(get, p, dtype).items()})
        out[f"{p}.input_weight"] = get(f"{p}.input_layernorm.weight").to(dtype)
        out[f"{p}.post_attention_weight"] = get(
            f"{p}.post_attention_layernorm.weight"
        ).to(dtype)

    return {k: v.contiguous() for k, v in out.items()}


def prepare(model_name, output_dir, dtype=torch.bfloat16):
    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(model_name)
    handle = safe_open(
        os.path.join(model_name, "model.safetensors"), framework="pt", device="cpu"
    )

    def get(key):
        return handle.get_tensor(key)

    print(f"[1/1] Converting drafter `{model_name}` (replicated)...")
    shard = build(get, config, dtype)
    path = os.path.join(output_dir, "drafter.safetensors")
    save_file(shard, path)
    print(f"  - wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the P-EAGLE drafter into a replicated bf16 safetensors."
    )
    parser.add_argument(
        "--model-name", required=True, help="path to P-EAGLE checkpoint"
    )
    parser.add_argument("--output-dir", default="tmp_p-eagle")
    parser.add_argument("--dtype", choices=["f32", "bf16"], default="bf16")
    args = parser.parse_args()
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    prepare(args.model_name, args.output_dir, dtype=dtype)
