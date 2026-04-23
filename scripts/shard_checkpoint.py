#!/usr/bin/env python3
"""Pre-shard a HuggingFace model checkpoint for tensor-parallel inference.

Loads the full model onto CPU, applies a TP sharding plan (colwise / rowwise
splits per weight), post-processes into the nkipy weight format (merged QKV,
transposed projections, combined expert tensors), and writes one safetensors
file per rank.

Usage:
    python scripts/shard_checkpoint.py \
        --model-name Qwen/Qwen3-30B-A3B \
        --world-size 32 --head-dim 128 --shard-embed \
        --output-dir /tmp/qwen3_30B_A3B_TP32

    python scripts/shard_checkpoint.py \
        --model-name Qwen/Qwen3-30B-A3B \
        --world-size 8 --head-dim 128 --dtype bf16 \
        --output-dir /tmp/qwen3_30B_A3B_TP8
"""

import argparse
import fnmatch
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

# TP plan: pattern -> split style
# For 2D tensors: colwise=dim0, rowwise=dim1
# For 3D expert tensors (transformers 5.0+): colwise=dim1, rowwise=dim2
BASE_MODEL_TP_PLAN = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    # Pre-5.0 transformers: separate expert tensors (2D)
    "layers.*.mlp.experts.*.gate_proj": "colwise",
    "layers.*.mlp.experts.*.up_proj": "colwise",
    "layers.*.mlp.experts.*.down_proj": "rowwise",
    # Transformers 5.0+: combined expert tensors (3D)
    "layers.*.mlp.experts.gate_up_proj": "colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    # Non-MoE models
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
    "lm_head": "colwise",
}


def _get_split_style(name: str, tp_plan: dict):
    for pat, style in tp_plan.items():
        if fnmatch.fnmatch(name, f"*{pat}*"):
            return style
    return None


def _get_split_dim(name: str, tensor: torch.Tensor, tp_plan: dict):
    style = _get_split_style(name, tp_plan)
    if style is None:
        return None
    ndim = tensor.ndim
    if ndim == 2:
        return 0 if style == "colwise" else 1
    elif ndim == 3:
        return 1 if style == "colwise" else 2
    return None


def _build_shard(state_items, rank, world_size, head_dim, tp_plan=None):
    if tp_plan is None:
        tp_plan = BASE_MODEL_TP_PLAN
    shard = {}
    for name, tensor in state_items:
        dim = _get_split_dim(name, tensor, tp_plan)

        if dim is None or tensor.numel() == 1 or tensor.size(dim) % world_size != 0:
            shard[name] = tensor
            continue

        # KV projections: replicate when shard would be smaller than one head
        if ("k_proj" in name or "v_proj" in name) and (
            tensor.shape[dim] // world_size < head_dim
        ):
            n_kv_heads = tensor.shape[0] // head_dim
            t = tensor.reshape(-1, head_dim, tensor.shape[1])
            head_index = int(np.floor(n_kv_heads * rank / world_size))
            shard[name] = t[head_index]
            continue

        # Transformers 5.0+ gate_up_proj: shard gate and up halves separately
        if "experts.gate_up_proj" in name and tensor.ndim == 3:
            intermediate = tensor.shape[1] // 2
            gate = tensor[:, :intermediate, :]
            up = tensor[:, intermediate:, :]
            shard[name] = torch.cat(
                [gate.chunk(world_size, dim=1)[rank],
                 up.chunk(world_size, dim=1)[rank]], dim=1,
            )
            continue

        shard[name] = tensor.chunk(world_size, dim=dim)[rank]

    return shard


def _post_process_shard(shard, config):
    """Rename and reshape weights into the nkipy weight format."""
    processed = {}
    n_layers = config.num_hidden_layers

    if "model.norm.weight" in shard:
        processed["norm_weight"] = shard["model.norm.weight"]
    if "lm_head.weight" in shard:
        processed["lm_head_weight"] = shard["lm_head.weight"].T
    if "model.embed_tokens.weight" in shard:
        processed["tok_embedding"] = shard["model.embed_tokens.weight"]

    for layer_id in range(n_layers):
        pfx = f"model.layers.{layer_id}"
        out = f"layers.{layer_id}"

        # QKV merge
        q = shard.get(f"{pfx}.self_attn.q_proj.weight")
        k = shard.get(f"{pfx}.self_attn.k_proj.weight")
        v = shard.get(f"{pfx}.self_attn.v_proj.weight")
        o = shard.get(f"{pfx}.self_attn.o_proj.weight")
        if q is not None and k is not None and v is not None:
            processed[f"{out}.qkv_weight"] = torch.cat([q.T, k.T, v.T], dim=-1)
        if o is not None:
            processed[f"{out}.o_weight"] = o.T

        # Norm weights
        if f"{pfx}.self_attn.q_norm.weight" in shard:
            processed[f"{out}.q_norm_weight"] = shard[f"{pfx}.self_attn.q_norm.weight"]
            processed[f"{out}.k_norm_weight"] = shard[f"{pfx}.self_attn.k_norm.weight"]
            processed[f"{out}.input_weight"] = shard[f"{pfx}.input_layernorm.weight"]
            processed[f"{out}.post_attention_weight"] = shard[f"{pfx}.post_attention_layernorm.weight"]

        # Router
        if f"{pfx}.mlp.gate.weight" in shard:
            processed[f"{out}.router_weight"] = shard[f"{pfx}.mlp.gate.weight"].T

        # MoE experts — transformers 5.0+ combined format
        gate_up_key = f"{pfx}.mlp.experts.gate_up_proj"
        down_key = f"{pfx}.mlp.experts.down_proj"
        if gate_up_key in shard:
            processed[f"{out}.gate_up_weight"] = shard[gate_up_key].transpose(1, 2)
            processed[f"{out}.down_weight"] = shard[down_key].transpose(1, 2)
        else:
            # Pre-5.0 format: separate per-expert tensors
            num_experts = 0
            while f"{pfx}.mlp.experts.{num_experts}.gate_proj.weight" in shard:
                num_experts += 1
            if num_experts > 0:
                gate_up_list, down_list = [], []
                for eid in range(num_experts):
                    g = shard.get(f"{pfx}.mlp.experts.{eid}.gate_proj.weight")
                    u = shard.get(f"{pfx}.mlp.experts.{eid}.up_proj.weight")
                    d = shard.get(f"{pfx}.mlp.experts.{eid}.down_proj.weight")
                    if g is not None and u is not None:
                        gate_up_list.append(torch.cat([g.T, u.T], dim=-1))
                    if d is not None:
                        down_list.append(d.T)
                if gate_up_list:
                    processed[f"{out}.gate_up_weight"] = torch.stack(gate_up_list)
                if down_list:
                    processed[f"{out}.down_weight"] = torch.stack(down_list)

        # Non-MoE MLP (dense models)
        gate = shard.get(f"{pfx}.mlp.gate_proj.weight")
        up = shard.get(f"{pfx}.mlp.up_proj.weight")
        down = shard.get(f"{pfx}.mlp.down_proj.weight")
        if gate is not None and up is not None:
            processed[f"{out}.gate_up_weight"] = torch.cat([gate.T, up.T], dim=-1)
        if down is not None and f"{out}.down_weight" not in processed:
            processed[f"{out}.down_weight"] = down.T

    return processed


def shard_checkpoint(
    model_name: str,
    output_dir: str,
    world_size: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    shard_embed: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    tp_plan = dict(BASE_MODEL_TP_PLAN)
    if shard_embed:
        tp_plan["embed_tokens"] = "rowwise"

    print(f"[1/3] Loading full model `{model_name}` onto CPU …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    config = model.config
    state_items = list(model.state_dict().items())

    print(f"[2/3] Splitting, post-processing, and saving {world_size} shards …")
    for rank in range(world_size):
        shard = _build_shard(state_items, rank, world_size, head_dim, tp_plan)
        processed = _post_process_shard(shard, config)
        for k in processed:
            processed[k] = processed[k].contiguous()
        path = os.path.join(output_dir, f"shard_{rank}.safetensors")
        save_file(processed, path)
        del shard, processed

    print(f"[3/3] Done! {world_size} shards saved in {output_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-shard a HuggingFace model into per-rank safetensors for TP inference",
    )
    parser.add_argument(
        "--model-name", required=True, help="HF repo or local path, e.g. Qwen/Qwen3-30B-A3B",
    )
    parser.add_argument("--output-dir", default="sharded_safetensors")
    parser.add_argument(
        "--world-size", type=int, required=True, help="Number of tensor-parallel ranks",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension")
    parser.add_argument(
        "--dtype", choices=["f32", "f16", "bf16"], default="bf16", help="Data type to load/save",
    )
    parser.add_argument(
        "--shard-embed", action="store_true",
        help="Shard embed_tokens along hidden dim (rowwise) instead of replicating",
    )
    args = parser.parse_args()

    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    shard_checkpoint(
        args.model_name, args.output_dir, args.world_size, args.head_dim, dtype,
        shard_embed=args.shard_embed,
    )


if __name__ == "__main__":
    main()
