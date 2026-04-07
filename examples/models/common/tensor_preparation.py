#!/usr/bin/env python3
import fnmatch
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

_STATE_ITEMS = None
_WORLD_SIZE = None
_OUTPUT_DIR = None
_CONFIG = None


def get_split_style(name, tp_plan):
    for pat, style in tp_plan.items():
        if fnmatch.fnmatch(name, f"*{pat}*"):
            return style
    return None


def get_split_dim(name, tensor, tp_plan):
    style = get_split_style(name, tp_plan)
    if style is None:
        return None
    if tensor.ndim == 2:
        return 0 if style == "colwise" else 1
    elif tensor.ndim == 3:
        return 1 if style == "colwise" else 2
    return None


def build_and_save_shard(rank, head_dim, tp_plan, post_process_shard):
    shard = {}
    for name, tensor in _STATE_ITEMS:
        t = tensor
        dim = get_split_dim(name, tensor, tp_plan)

        if dim is None or t.numel() == 1 or t.size(dim) % _WORLD_SIZE != 0:
            shard[name] = t
            continue

        if ("k_proj" in name or "v_proj" in name) and (
            t.shape[dim] // _WORLD_SIZE < head_dim
        ):
            n_kv_heads = tensor.shape[0] // head_dim
            tensor = tensor.reshape(-1, head_dim, tensor.shape[1])
            head_index = np.floor(n_kv_heads * rank / _WORLD_SIZE).astype(int)
            shard[name] = tensor[head_index]
            continue

        if "experts.gate_up_proj" in name and t.ndim == 3:
            intermediate = t.shape[1] // 2
            gate_part = t[:, :intermediate, :].chunk(_WORLD_SIZE, dim=1)[rank]
            up_part = t[:, intermediate:, :].chunk(_WORLD_SIZE, dim=1)[rank]
            shard[name] = torch.cat([gate_part, up_part], dim=1)
            continue

        shard[name] = t.chunk(_WORLD_SIZE, dim=dim)[rank]

    processed_shard = post_process_shard(shard, rank, _CONFIG)

    for name, part in processed_shard.items():
        processed_shard[name] = part.contiguous()

    path = os.path.join(_OUTPUT_DIR, f"shard_{rank}.safetensors")
    save_file(processed_shard, path)
    del shard, processed_shard


def preshard_model(
    model_name,
    output_dir,
    world_size,
    head_dim,
    tp_plan,
    post_process_shard,
    dtype=torch.float32,
):
    global _STATE_ITEMS, _WORLD_SIZE, _OUTPUT_DIR, _CONFIG

    os.makedirs(output_dir, exist_ok=True)
    _WORLD_SIZE = world_size
    _OUTPUT_DIR = output_dir

    print(f"[1/3] Loading full model `{model_name}` onto CPU…")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cpu", torch_dtype=dtype, low_cpu_mem_usage=False,
    )

    _CONFIG = model.config
    _STATE_ITEMS = list(model.state_dict().items())

    print(f"[2/3] Splitting, post-processing, and saving {_WORLD_SIZE} shards...")
    for rank in range(_WORLD_SIZE):
        build_and_save_shard(rank, head_dim, tp_plan, post_process_shard)

    print(f"[3/3] Done! {_WORLD_SIZE} post-processed shards saved in {_OUTPUT_DIR}.")


def main(description, tp_plan, post_process_shard):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-name", required=True, help="HF repo or local path")
    parser.add_argument("--output-dir", default="sharded_safetensors")
    parser.add_argument("--world-size", type=int, required=True, help="Number of tensor-parallel ranks")
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="bf16")
    parser.add_argument("--head-dim", type=int, default=128)

    args = parser.parse_args()
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    preshard_model(
        args.model_name, args.output_dir, args.world_size, args.head_dim,
        tp_plan, post_process_shard, dtype=dtype,
    )
