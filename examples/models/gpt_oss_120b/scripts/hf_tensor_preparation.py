#!/usr/bin/env python3
"""Pre-shard a HuggingFace ``gpt_oss`` checkpoint (e.g. ``openai/gpt-oss-120b``)
into the ``shard_{rank}.safetensors`` files the runtime loads.

The HF checkpoint stores the MoE expert weights in MXFP4 (``*_blocks`` / ``*_scales``)
and uses ``model.layers.*`` names; this script dequantizes to bf16, remaps to the
internal layout, and applies the tensor-parallel sharding for the chosen degree.
Output is byte-for-byte equivalent to the legacy OpenAI-checkpoint path
(``openai_tensor_preparation.py``), which it supersedes as the default.

Usage:
    python scripts/hf_tensor_preparation.py \
        --model-dir /path/to/openai/gpt-oss-120b \
        --output-dir ./gpt-oss-120b-bf16-TP8 \
        --world-size 8 --num-layers 36 --head-dim 64
"""
import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm

# MXFP4 codebook (nibble value -> fp value); the stored exponent is a per-block
# power-of-two scale applied via ldexp.
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


class HFCheckpoint:
    """Random-access reader over a sharded HF safetensors checkpoint that
    dequantizes MXFP4 MoE weights on demand."""

    def __init__(self, path: str, dtype: torch.dtype):
        self.path = path
        self.dtype = dtype
        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            self.weight_map = json.load(open(index_path))["weight_map"]
        else:
            # single-file checkpoint
            self.weight_map = {}
            for fname in os.listdir(path):
                if fname.endswith(".safetensors"):
                    with safe_open(os.path.join(path, fname), framework="pt") as f:
                        for k in f.keys():
                            self.weight_map[k] = fname

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        with safe_open(
            os.path.join(self.path, self.weight_map[name]), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(name)

    def get_mxfp4(self, base: str) -> torch.Tensor:
        """Dequantize the ``base_blocks`` / ``base_scales`` MXFP4 pair to a dense
        tensor. Layout: blocks (..., G, B) uint8 nibble-packed, scales (..., G)
        biased exponents. Returns (..., G*B*2) in ``self.dtype``."""
        blocks = self.get(base + "_blocks")
        scales = self.get(base + "_scales").to(torch.int32) - 127
        assert blocks.shape[:-1] == scales.shape, (
            f"{base}: {blocks.shape=} vs {scales.shape=}"
        )
        lut = torch.tensor(FP4_VALUES, dtype=self.dtype)
        *prefix, G, B = blocks.shape
        rows = math.prod(prefix) * G
        blk = blocks.reshape(rows, B)
        exp = scales.reshape(rows, 1)
        out = torch.empty(rows, B * 2, dtype=self.dtype)
        out[:, 0::2] = lut[(blk & 0x0F).long()]
        out[:, 1::2] = lut[(blk >> 4).long()]
        torch.ldexp(out, exp, out=out)
        return out.reshape(*prefix, G, B * 2).reshape(*prefix, G * B * 2)


def _all_chunks(t: torch.Tensor, world_size: int, dim: int):
    """Split into per-rank contiguous chunks (each tensor is read/dequantized
    once and sliced for all ranks — avoids re-dequantizing MXFP4 per rank)."""
    return [c.contiguous() for c in t.chunk(world_size, dim=dim)]


def add_layer_to_shards(ckpt: HFCheckpoint, shards: list, layer: int, world_size: int):
    """Remap one HF layer's tensors and append each rank's slice to shards[rank]."""
    p = f"model.layers.{layer}"
    L = f"layers.{layer}"
    dt = ckpt.dtype

    # ---- attention: shard Q/K/V heads on the output dim, concat, transpose ----
    qc = _all_chunks(ckpt.get(f"{p}.self_attn.q_proj.weight").to(dt), world_size, 0)
    kc = _all_chunks(ckpt.get(f"{p}.self_attn.k_proj.weight").to(dt), world_size, 0)
    vc = _all_chunks(ckpt.get(f"{p}.self_attn.v_proj.weight").to(dt), world_size, 0)
    qbc = _all_chunks(ckpt.get(f"{p}.self_attn.q_proj.bias").to(dt), world_size, 0)
    kbc = _all_chunks(ckpt.get(f"{p}.self_attn.k_proj.bias").to(dt), world_size, 0)
    vbc = _all_chunks(ckpt.get(f"{p}.self_attn.v_proj.bias").to(dt), world_size, 0)
    oc = _all_chunks(ckpt.get(f"{p}.self_attn.o_proj.weight").to(dt), world_size, 1)
    o_bias = ckpt.get(f"{p}.self_attn.o_proj.bias").to(dt)
    sinkc = _all_chunks(ckpt.get(f"{p}.self_attn.sinks").to(dt), world_size, 0)
    attn_norm = ckpt.get(f"{p}.input_layernorm.weight").to(dt)
    mlp_norm = ckpt.get(f"{p}.post_attention_layernorm.weight").to(dt)

    # ---- MoE gate/up: dequant once (E, 2I interleaved, H); per-rank shard I
    # (keeping gate/up interleave pairs), transpose, de-interleave to [gate|up].
    gu = ckpt.get_mxfp4(f"{p}.mlp.experts.gate_up_proj")  # (E, 2I, H)
    E, twoI, H = gu.shape
    I = twoI // 2
    guc = _all_chunks(gu.reshape(E, I, 2, H), world_size, 1)
    gub = ckpt.get(f"{p}.mlp.experts.gate_up_proj_bias").to(dt)  # (E, 2I)
    gubc = _all_chunks(gub.reshape(E, I, 2), world_size, 1)

    # ---- MoE down: dequant once (E, H, I); per-rank shard I on input dim, transpose.
    dn = ckpt.get_mxfp4(f"{p}.mlp.experts.down_proj")  # (E, H, I)
    dnc = _all_chunks(dn, world_size, 2)
    down_bias = ckpt.get(f"{p}.mlp.experts.down_proj_bias").to(dt)

    router_weight = ckpt.get(f"{p}.mlp.router.weight").to(dt).T.contiguous()
    router_bias = ckpt.get(f"{p}.mlp.router.bias").to(dt)

    for rank in range(world_size):
        s = shards[rank]
        s[f"{L}.qkv_weight"] = torch.cat([qc[rank], kc[rank], vc[rank]], 0).T.contiguous()
        s[f"{L}.qkv_bias"] = torch.cat([qbc[rank], kbc[rank], vbc[rank]], 0)
        s[f"{L}.o_weight"] = oc[rank].T.contiguous()
        s[f"{L}.o_bias"] = o_bias
        s[f"{L}.attn_sinks"] = sinkc[rank]
        s[f"{L}.attn_norm_weight"] = attn_norm
        s[f"{L}.mlp_norm_weight"] = mlp_norm

        gu_s = guc[rank].reshape(E, (I // world_size) * 2, H).transpose(1, 2)
        s[f"{L}.gate_up_weight"] = torch.cat([gu_s[..., ::2], gu_s[..., 1::2]], -1).contiguous()
        gub_s = gubc[rank].reshape(E, (I // world_size) * 2)
        s[f"{L}.gate_up_bias"] = torch.cat([gub_s[..., ::2], gub_s[..., 1::2]], -1).contiguous()

        s[f"{L}.down_weight"] = dnc[rank].transpose(1, 2).contiguous()
        s[f"{L}.down_bias"] = down_bias
        s[f"{L}.router_weight"] = router_weight
        s[f"{L}.router_bias"] = router_bias


def preshard_hf_model(model_dir, output_dir, world_size, head_dim, num_layers,
                      n_kv_heads=8, dtype=torch.bfloat16):
    os.makedirs(output_dir, exist_ok=True)
    ckpt = HFCheckpoint(model_dir, dtype)
    print(f"[1/3] Loaded HF checkpoint index from `{model_dir}` "
          f"({len(ckpt.weight_map)} tensors).")

    # Global weights (each read/dequantized once, sliced across ranks).
    norm_weight = ckpt.get("model.norm.weight").to(dtype)
    tok_embedding = ckpt.get("model.embed_tokens.weight").to(dtype)
    lm_head_c = _all_chunks(ckpt.get("lm_head.weight").to(dtype), world_size, 0)
    shards = [{} for _ in range(world_size)]
    for rank in range(world_size):
        shards[rank]["norm_weight"] = norm_weight
        shards[rank]["tok_embedding"] = tok_embedding
        shards[rank]["lm_head_weight"] = lm_head_c[rank].T.contiguous()

    print(f"[2/3] Remapping {num_layers} layers across {world_size} shards "
          f"(dequantizing MXFP4 MoE weights once per tensor)...")
    for layer in tqdm(range(num_layers), desc="Layers"):
        add_layer_to_shards(ckpt, shards, layer, world_size)

    print(f"[3/3] Saving {world_size} shards...")
    for rank in tqdm(range(world_size), desc="Saving"):
        save_file(shards[rank], os.path.join(output_dir, f"shard_{rank}.safetensors"))
    print(f"Done! {world_size} shards written to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-shard a HuggingFace gpt_oss checkpoint for tensor parallelism."
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to the HuggingFace gpt_oss checkpoint directory")
    parser.add_argument("--output-dir", default="gpt-oss-120b-bf16-TP8")
    parser.add_argument("--world-size", type=int, required=True,
                        help="Number of tensor-parallel ranks")
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="bf16")
    args = parser.parse_args()
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    preshard_hf_model(args.model_dir, args.output_dir, args.world_size, args.head_dim,
                      args.num_layers, n_kv_heads=args.n_kv_heads, dtype=dtype)
