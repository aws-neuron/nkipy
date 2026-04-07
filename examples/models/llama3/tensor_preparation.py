#!/usr/bin/env python3
import torch

from ..common.tensor_preparation import main

base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
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

        input_ln = f"model.layers.{layer_id}.input_layernorm.weight"
        post_ln = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        if input_ln in shard:
            processed[f"layers.{layer_id}.input_weight"] = shard[input_ln]
        if post_ln in shard:
            processed[f"layers.{layer_id}.post_attention_weight"] = shard[post_ln]

        gate = shard.get(f"model.layers.{layer_id}.mlp.gate_proj.weight")
        up = shard.get(f"model.layers.{layer_id}.mlp.up_proj.weight")
        down = shard.get(f"model.layers.{layer_id}.mlp.down_proj.weight")

        if gate is not None and up is not None:
            processed[f"layers.{layer_id}.gate_up_weight"] = torch.cat([gate.T, up.T], axis=-1)
        if down is not None:
            processed[f"layers.{layer_id}.down_weight"] = down.T

    return processed


if __name__ == "__main__":
    """
    Usage:
    python tensor_preparation.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --world-size 8 --head-dim 64 --output-dir tmp_tinyllama_TP8
    """
    main(
        description="Pre-shard a Llama3 model into safetensors using a custom TP plan.",
        tp_plan=base_model_tp_plan,
        post_process_shard=post_process_shard,
    )
