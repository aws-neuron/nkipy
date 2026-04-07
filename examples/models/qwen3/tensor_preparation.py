#!/usr/bin/env python3
import torch

from ..common.tensor_preparation import main

base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.experts.*.gate_proj": "colwise",
    "layers.*.mlp.experts.*.up_proj": "colwise",
    "layers.*.mlp.experts.*.down_proj": "rowwise",
    "layers.*.mlp.experts.gate_up_proj": "colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
    "lm_head": "colwise",
}


def post_process_shard(shard, rank, config):
    processed = {}
    n_layers = config.num_hidden_layers

    if "model.norm.weight" in shard:
        processed["norm_weight"] = shard["model.norm.weight"]
    if "lm_head.weight" in shard:
        processed["lm_head_weight"] = shard["lm_head.weight"].T
    if "model.embed_tokens.weight" in shard:
        processed["tok_embedding"] = shard["model.embed_tokens.weight"]

    for layer_id in range(n_layers):
        q = shard.get(f"model.layers.{layer_id}.self_attn.q_proj.weight")
        k = shard.get(f"model.layers.{layer_id}.self_attn.k_proj.weight")
        v = shard.get(f"model.layers.{layer_id}.self_attn.v_proj.weight")
        o = shard.get(f"model.layers.{layer_id}.self_attn.o_proj.weight")

        if q is not None and k is not None and v is not None:
            processed[f"layers.{layer_id}.qkv_weight"] = torch.cat([q.T, k.T, v.T], axis=-1)
        if o is not None:
            processed[f"layers.{layer_id}.o_weight"] = o.T

        if f"model.layers.{layer_id}.self_attn.q_norm.weight" in shard:
            processed[f"layers.{layer_id}.q_norm_weight"] = shard[f"model.layers.{layer_id}.self_attn.q_norm.weight"]
            processed[f"layers.{layer_id}.k_norm_weight"] = shard[f"model.layers.{layer_id}.self_attn.k_norm.weight"]
            processed[f"layers.{layer_id}.input_weight"] = shard[f"model.layers.{layer_id}.input_layernorm.weight"]
            processed[f"layers.{layer_id}.post_attention_weight"] = shard[f"model.layers.{layer_id}.post_attention_layernorm.weight"]

        if f"model.layers.{layer_id}.mlp.gate.weight" in shard:
            processed[f"layers.{layer_id}.router_weight"] = shard[f"model.layers.{layer_id}.mlp.gate.weight"].T

        gate_up_key = f"model.layers.{layer_id}.mlp.experts.gate_up_proj"
        down_key = f"model.layers.{layer_id}.mlp.experts.down_proj"

        if gate_up_key in shard:
            processed[f"layers.{layer_id}.gate_up_weight"] = shard[gate_up_key].transpose(1, 2)
            processed[f"layers.{layer_id}.down_weight"] = shard[down_key].transpose(1, 2)
        else:
            num_experts = 0
            while f"model.layers.{layer_id}.mlp.experts.{num_experts}.gate_proj.weight" in shard:
                num_experts += 1

            if num_experts > 0:
                gate_up_weights = []
                down_weights = []
                for eid in range(num_experts):
                    gate = shard.get(f"model.layers.{layer_id}.mlp.experts.{eid}.gate_proj.weight")
                    up = shard.get(f"model.layers.{layer_id}.mlp.experts.{eid}.up_proj.weight")
                    down = shard.get(f"model.layers.{layer_id}.mlp.experts.{eid}.down_proj.weight")
                    if gate is not None and up is not None:
                        gate_up_weights.append(torch.cat([gate.T, up.T], axis=-1))
                    if down is not None:
                        down_weights.append(down.T)
                if gate_up_weights:
                    processed[f"layers.{layer_id}.gate_up_weight"] = torch.stack(gate_up_weights)
                if down_weights:
                    processed[f"layers.{layer_id}.down_weight"] = torch.stack(down_weights)

    return processed


if __name__ == "__main__":
    """
    Usage:
    python tensor_preparation.py --model-name Qwen/Qwen3-30B-A3B --world-size 8 --head-dim 128 --output-dir tmp_qwen3_30B_A3B_TP8
    """
    main(
        description="Pre-shard a Qwen3 model into safetensors using a custom TP plan.",
        tp_plan=base_model_tp_plan,
        post_process_shard=post_process_shard,
    )
