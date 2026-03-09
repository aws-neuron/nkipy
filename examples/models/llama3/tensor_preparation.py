#!/usr/bin/env python3
import fnmatch
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

# TP plan for Llama3 (dense model, no MoE)
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

_STATE_ITEMS = None
_WORLD_SIZE = None
_OUTPUT_DIR = None
_CONFIG = None


def get_split_style(name: str):
    for pat, style in base_model_tp_plan.items():
        if fnmatch.fnmatch(name, f"*{pat}*"):
            return style
    return None


def get_split_dim(name: str, tensor: torch.Tensor):
    style = get_split_style(name)
    if style is None:
        return None
    if tensor.ndim == 2:
        return 0 if style == "colwise" else 1
    return None


def build_and_save_shard(rank, head_dim):
    shard = {}
    for name, tensor in _STATE_ITEMS:
        t = tensor
        dim = get_split_dim(name, tensor)

        if dim is None or t.numel() == 1 or t.size(dim) % _WORLD_SIZE != 0:
            shard[name] = t
            continue

        # KV head replication when sharding below head_dim
        if ("k_proj" in name or "v_proj" in name) and (
            t.shape[dim] // _WORLD_SIZE < head_dim
        ):
            n_kv_heads = tensor.shape[0] // head_dim
            tensor = tensor.reshape(-1, head_dim, tensor.shape[1])
            head_index = np.floor(n_kv_heads * rank / _WORLD_SIZE).astype(int)
            part = tensor[head_index]
            shard[name] = part
            continue

        part = t.chunk(_WORLD_SIZE, dim=dim)[rank]
        shard[name] = part

    processed_shard = post_process_shard(shard, rank)

    for name, part in processed_shard.items():
        processed_shard[name] = part.contiguous()

    path = os.path.join(_OUTPUT_DIR, f"shard_{rank}.safetensors")
    save_file(processed_shard, path)
    del shard, processed_shard


def post_process_shard(shard, rank):
    processed = {}
    n_layers = _CONFIG.num_hidden_layers

    if "model.norm.weight" in shard:
        processed["norm_weight"] = shard["model.norm.weight"]

    if "lm_head.weight" in shard:
        processed["lm_head_weight"] = shard["lm_head.weight"].T

    if "model.embed_tokens.weight" in shard:
        processed["tok_embedding"] = shard["model.embed_tokens.weight"]

    for layer_id in range(n_layers):
        q_weight = shard.get(f"model.layers.{layer_id}.self_attn.q_proj.weight")
        k_weight = shard.get(f"model.layers.{layer_id}.self_attn.k_proj.weight")
        v_weight = shard.get(f"model.layers.{layer_id}.self_attn.v_proj.weight")
        o_weight = shard.get(f"model.layers.{layer_id}.self_attn.o_proj.weight")

        if q_weight is not None and k_weight is not None and v_weight is not None:
            qkv_weight = torch.cat([q_weight.T, k_weight.T, v_weight.T], axis=-1)
            processed[f"layers.{layer_id}.qkv_weight"] = qkv_weight

        if o_weight is not None:
            processed[f"layers.{layer_id}.o_weight"] = o_weight.T

        # Layer norms
        input_ln_key = f"model.layers.{layer_id}.input_layernorm.weight"
        post_attn_ln_key = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        if input_ln_key in shard:
            processed[f"layers.{layer_id}.input_weight"] = shard[input_ln_key]
        if post_attn_ln_key in shard:
            processed[f"layers.{layer_id}.post_attention_weight"] = shard[post_attn_ln_key]

        # Dense FFN: gate_proj + up_proj combined, down_proj
        gate_weight = shard.get(f"model.layers.{layer_id}.mlp.gate_proj.weight")
        up_weight = shard.get(f"model.layers.{layer_id}.mlp.up_proj.weight")
        down_weight = shard.get(f"model.layers.{layer_id}.mlp.down_proj.weight")

        if gate_weight is not None and up_weight is not None:
            gate_up_weight = torch.cat([gate_weight.T, up_weight.T], axis=-1)
            processed[f"layers.{layer_id}.gate_up_weight"] = gate_up_weight

        if down_weight is not None:
            processed[f"layers.{layer_id}.down_weight"] = down_weight.T

    return processed


def preshard_model(
    model_name: str,
    output_dir: str,
    world_size: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
):
    global _STATE_ITEMS, _WORLD_SIZE, _OUTPUT_DIR, _CONFIG

    os.makedirs(output_dir, exist_ok=True)
    _WORLD_SIZE = world_size
    _OUTPUT_DIR = output_dir

    print(f"[1/3] Loading full model `{model_name}` onto CPU…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    _CONFIG = model.config
    _STATE_ITEMS = list(model.state_dict().items())

    print(f"[2/3] Splitting, post-processing, and saving {_WORLD_SIZE} shards...")

    for rank in range(_WORLD_SIZE):
        build_and_save_shard(rank, head_dim)

    print(f"[3/3] Done! {_WORLD_SIZE} post-processed shards saved in {_OUTPUT_DIR}.")


if __name__ == "__main__":
    """
    Usage:
    python tensor_preparation.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --world-size 8 --head-dim 64 --output-dir tmp_tinyllama_TP8
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-shard a Llama3 model into safetensors using a custom TP plan."
    )
    parser.add_argument(
        "--model-name", required=True, help="HF repo or local path, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--output-dir", default="sharded_safetensors")
    parser.add_argument(
        "--world-size", type=int, required=True, help="Number of tensor-parallel ranks"
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16", "bf16"],
        default="bf16",
        help="Data type to load/save",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="The head dim size")

    args = parser.parse_args()
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[
        args.dtype
    ]

    preshard_model(
        args.model_name,
        args.output_dir,
        args.world_size,
        args.head_dim,
        dtype=dtype,
    )
