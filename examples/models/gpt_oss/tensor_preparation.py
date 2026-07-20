#!/usr/bin/env python3
"""Download/convert gpt-oss weights into per-rank safetensors shards for NKIPy.

gpt-oss ships its experts MXFP4-quantized (``*_blocks`` / ``*_scales``). We
dequantize them to bf16 at prep time (the chosen, simplest approach) so the NKI
kernels operate purely on bf16, mirroring the Qwen3 example.

Each rank's shard contains, per layer:
  * ``qkv_weight`` / ``qkv_bias``  - fused Q,K,V projection (x @ W form)
  * ``o_weight`` / ``o_bias``      - output projection
  * ``sinks``                      - per-head attention sink logits
  * ``input_weight`` / ``post_attention_weight`` - RMSNorm gains
  * ``router_weight`` / ``router_bias``
  * ``gate_up_weight`` / ``gate_up_bias`` - de-interleaved [gate | up]
  * ``down_weight`` / ``down_bias``
plus shared ``tok_embedding``, ``norm_weight`` and ``lm_head_weight``.
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.integrations.mxfp4 import convert_moe_packed_tensors


class LazyWeights:
    """Read tensors by name from a sharded safetensors checkpoint on demand."""

    def __init__(self, model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            # Single-file checkpoint.
            self.weight_map = None
        self.model_dir = model_dir
        self._handles = {}
        self._single = os.path.join(model_dir, "model.safetensors")

    def _handle(self, key):
        fname = self.weight_map[key] if self.weight_map else "model.safetensors"
        path = os.path.join(self.model_dir, fname)
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return self._handles[path]

    def get(self, key):
        return self._handle(key).get_tensor(key)

    def has(self, key):
        if self.weight_map is not None:
            return key in self.weight_map
        return key in self._handle(key).keys()


def chunk_rank(t, dim, rank, world_size):
    return t.chunk(world_size, dim=dim)[rank].contiguous()


def shard_kv(weight, head_dim, n_kv_heads, rank, world_size):
    """Shard a K/V projection (out=n_kv_heads*head_dim, in).

    When the per-rank slice would be smaller than a single head, replicate whole
    heads across rank groups instead (mirrors the Qwen3 example).
    """
    out_features = weight.shape[0]
    if out_features // world_size >= head_dim and out_features % world_size == 0:
        return chunk_rank(weight, 0, rank, world_size)
    # Replicate: pick the kv head this rank maps to.
    w = weight.reshape(n_kv_heads, head_dim, weight.shape[1])
    head_index = (n_kv_heads * rank) // world_size
    return w[head_index].contiguous()


def build_shard(weights, config, rank, world_size, dtype):
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    out = {}

    # Shared (non-sharded) tensors.
    out["norm_weight"] = weights.get("model.norm.weight").to(dtype)
    out["tok_embedding"] = weights.get("model.embed_tokens.weight").to(dtype)
    # lm_head: colwise shard along vocab, store transposed (hidden, vocab_local).
    lm_head = weights.get("lm_head.weight").to(dtype)
    out["lm_head_weight"] = chunk_rank(lm_head, 0, rank, world_size).T.contiguous()

    for layer_id in range(n_layers):
        p = f"model.layers.{layer_id}"

        # ---- Attention projections (nn.Linear: y = x @ W.T) ----
        q_w = weights.get(f"{p}.self_attn.q_proj.weight").to(dtype)
        k_w = weights.get(f"{p}.self_attn.k_proj.weight").to(dtype)
        v_w = weights.get(f"{p}.self_attn.v_proj.weight").to(dtype)
        o_w = weights.get(f"{p}.self_attn.o_proj.weight").to(dtype)
        q_b = weights.get(f"{p}.self_attn.q_proj.bias").to(dtype)
        k_b = weights.get(f"{p}.self_attn.k_proj.bias").to(dtype)
        v_b = weights.get(f"{p}.self_attn.v_proj.bias").to(dtype)
        o_b = weights.get(f"{p}.self_attn.o_proj.bias").to(dtype)
        sinks = weights.get(f"{p}.self_attn.sinks").to(dtype)

        # Q: colwise (shard heads). K/V: shard heads w/ replication fallback.
        q_w_s = chunk_rank(q_w, 0, rank, world_size)
        q_b_s = chunk_rank(q_b, 0, rank, world_size)
        k_w_s = shard_kv(k_w, head_dim, n_kv_heads, rank, world_size)
        v_w_s = shard_kv(v_w, head_dim, n_kv_heads, rank, world_size)
        k_b_s = shard_kv(
            k_b.reshape(-1, 1), head_dim, n_kv_heads, rank, world_size
        ).reshape(-1)
        v_b_s = shard_kv(
            v_b.reshape(-1, 1), head_dim, n_kv_heads, rank, world_size
        ).reshape(-1)

        # Fuse into x @ W form: W stored as (hidden, out_local).
        qkv_weight = torch.cat([q_w_s.T, k_w_s.T, v_w_s.T], dim=-1).contiguous()
        qkv_bias = torch.cat([q_b_s, k_b_s, v_b_s], dim=-1).contiguous()
        out[f"layers.{layer_id}.qkv_weight"] = qkv_weight
        out[f"layers.{layer_id}.qkv_bias"] = qkv_bias

        # o_proj: rowwise (shard along input = heads*head_dim).
        out[f"layers.{layer_id}.o_weight"] = chunk_rank(
            o_w, 1, rank, world_size
        ).T.contiguous()
        # o_bias is replicated; keep it only on rank 0 (added once post all-reduce).
        out[f"layers.{layer_id}.o_bias"] = (
            o_b if rank == 0 else torch.zeros_like(o_b)
        ).contiguous()

        # sinks: per-head, shard along heads to match Q.
        out[f"layers.{layer_id}.sinks"] = chunk_rank(sinks, 0, rank, world_size)

        # ---- RMSNorm gains (replicated) ----
        out[f"layers.{layer_id}.input_weight"] = weights.get(
            f"{p}.input_layernorm.weight"
        ).to(dtype)
        out[f"layers.{layer_id}.post_attention_weight"] = weights.get(
            f"{p}.post_attention_layernorm.weight"
        ).to(dtype)

        # ---- Router (replicated) ----
        router_w = weights.get(f"{p}.mlp.router.weight").to(
            dtype
        )  # (n_experts, hidden)
        router_b = weights.get(f"{p}.mlp.router.bias").to(dtype)
        out[f"layers.{layer_id}.router_weight"] = (
            router_w.T.contiguous()
        )  # (hidden, n_experts)
        out[f"layers.{layer_id}.router_bias"] = router_b.contiguous()

        # ---- Experts (MXFP4 -> bf16, de-interleave gate/up, shard along inter) ----
        gu_blocks = weights.get(f"{p}.mlp.experts.gate_up_proj_blocks")
        gu_scales = weights.get(f"{p}.mlp.experts.gate_up_proj_scales")
        gu_bias = weights.get(f"{p}.mlp.experts.gate_up_proj_bias").to(dtype)
        dn_blocks = weights.get(f"{p}.mlp.experts.down_proj_blocks")
        dn_scales = weights.get(f"{p}.mlp.experts.down_proj_scales")
        dn_bias = weights.get(f"{p}.mlp.experts.down_proj_bias").to(dtype)

        # gate_up: (E, hidden, 2*inter) with gate/up INTERLEAVED on last dim.
        gate_up = convert_moe_packed_tensors(gu_blocks, gu_scales).to(dtype)
        E, hidden, two_inter = gate_up.shape
        gate_up = gate_up.reshape(E, hidden, two_inter // 2, 2)
        gate = gate_up[..., 0]  # (E, hidden, inter)
        up = gate_up[..., 1]
        gate = chunk_rank(gate, 2, rank, world_size)
        up = chunk_rank(up, 2, rank, world_size)
        # Store [gate | up] so the kernel can split in half.
        out[f"layers.{layer_id}.gate_up_weight"] = torch.cat(
            [gate, up], dim=-1
        ).contiguous()

        gu_bias = gu_bias.reshape(E, two_inter // 2, 2)
        gate_b = chunk_rank(gu_bias[..., 0], 1, rank, world_size)
        up_b = chunk_rank(gu_bias[..., 1], 1, rank, world_size)
        out[f"layers.{layer_id}.gate_up_bias"] = torch.cat(
            [gate_b, up_b], dim=-1
        ).contiguous()

        # down: (E, inter, hidden), shard along inter (dim 1).
        down = convert_moe_packed_tensors(dn_blocks, dn_scales).to(dtype)
        out[f"layers.{layer_id}.down_weight"] = chunk_rank(down, 1, rank, world_size)
        # down bias replicated -> rank 0 only (added once post all-reduce).
        out[f"layers.{layer_id}.down_bias"] = (
            dn_bias if rank == 0 else torch.zeros_like(dn_bias)
        ).contiguous()

    return out


def preshard_model(model_name, output_dir, world_size, dtype=torch.bfloat16):
    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(model_name)
    weights = LazyWeights(model_name)

    print(
        f"[1/2] Sharding `{model_name}` into {world_size} ranks (dequantizing MXFP4)..."
    )
    for rank in range(world_size):
        shard = build_shard(weights, config, rank, world_size, dtype)
        path = os.path.join(output_dir, f"shard_{rank}.safetensors")
        save_file(shard, path)
        print(f"  - wrote {path}")
        del shard

    print(f"[2/2] Done! {world_size} shards saved in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-shard gpt-oss into per-rank bf16 safetensors for NKIPy."
    )
    parser.add_argument(
        "--model-name", required=True, help="HF repo or local path to gpt-oss"
    )
    parser.add_argument("--output-dir", default="sharded_gpt_oss")
    parser.add_argument(
        "--world-size", type=int, required=True, help="Number of tensor-parallel ranks"
    )
    parser.add_argument(
        "--dtype", choices=["f32", "bf16"], default="bf16", help="Output dtype"
    )
    args = parser.parse_args()
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    preshard_model(args.model_name, args.output_dir, args.world_size, dtype=dtype)
