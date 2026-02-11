#!/usr/bin/env python3
import fnmatch

# Ensure fork start method for shared-memory inheritance
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

# TP plan: pattern → split style
# For 2D tensors: colwise=dim0, rowwise=dim1
# For 3D expert tensors (transformers 5.0+): colwise=dim1, rowwise=dim2
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",  # shard till head dim is reached, then replicate
    "layers.*.self_attn.v_proj": "colwise",  # shard till head dim is reached, then replicate
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

# Globals inherited via fork
_STATE_ITEMS = None
_WORLD_SIZE = None
_OUTPUT_DIR = None
_CONFIG = None  # Store model config for post-processing


def get_split_style(name: str):
    """Return the split style for this parameter name, or None to replicate."""
    for pat, style in base_model_tp_plan.items():
        if fnmatch.fnmatch(name, f"*{pat}*"):
            return style
    return None


def get_split_dim(name: str, tensor: torch.Tensor):
    """Return the dimension to split along for this parameter, or None to replicate.

    For 2D tensors: colwise=0, rowwise=1
    For 3D tensors (transformers 5.0+ expert format): colwise=1, rowwise=2
    """
    style = get_split_style(name)
    if style is None:
        return None

    ndim = tensor.ndim
    if ndim == 2:
        return 0 if style == "colwise" else 1
    elif ndim == 3:
        # 3D expert tensors: (num_experts, out_features, in_features)
        return 1 if style == "colwise" else 2
    else:
        return None


def build_and_save_shard(rank, head_dim):
    """Construct shard dict for given rank and write to safetensors."""
    shard = {}
    for name, tensor in _STATE_ITEMS:
        t = tensor
        dim = get_split_dim(name, tensor)

        # Case 1: Don't shard this tensor (including router weight)
        if dim is None or t.numel() == 1 or t.size(dim) % _WORLD_SIZE != 0:
            shard[name] = t
            continue

        # Case 2: Shard KV, but --
        if ("k_proj" in name or "v_proj" in name) and (
            t.shape[dim] // _WORLD_SIZE < head_dim
        ):
            n_kv_heads = tensor.shape[0] // head_dim

            tensor = tensor.reshape(-1, head_dim, tensor.shape[1])
            head_index = np.floor(n_kv_heads * rank / _WORLD_SIZE).astype(int)
            part = tensor[head_index]
            shard[name] = part
            continue

        # Case 3: Special handling for transformers 5.0+ gate_up_proj
        # Shape: (num_experts, 2*intermediate, hidden), first half=gate, second=up
        # We need to shard gate and up separately, then concatenate
        if "experts.gate_up_proj" in name and t.ndim == 3:
            intermediate = t.shape[1] // 2
            gate = t[:, :intermediate, :]  # (num_experts, intermediate, hidden)
            up = t[:, intermediate:, :]  # (num_experts, intermediate, hidden)

            # Shard each on dim 1
            gate_part = gate.chunk(_WORLD_SIZE, dim=1)[rank]
            up_part = up.chunk(_WORLD_SIZE, dim=1)[rank]

            # Concatenate back: (num_experts, 2*intermediate/world_size, hidden)
            part = torch.cat([gate_part, up_part], dim=1)
            shard[name] = part
            continue

        # Case 4: shard normally
        part = t.chunk(_WORLD_SIZE, dim=dim)[rank]
        shard[name] = part

    # Process shard to new format before saving
    processed_shard = post_process_shard(shard, rank)

    # make all contiguous
    for name, part in processed_shard.items():
        processed_shard[name] = part.contiguous()

    path = os.path.join(_OUTPUT_DIR, f"shard_{rank}.safetensors")
    save_file(processed_shard, path)
    del shard, processed_shard


def post_process_shard(shard, rank):
    """Transform the shard weights into the target format"""
    processed = {}
    n_layers = _CONFIG.num_hidden_layers

    # Process norm weight and lm_head
    if "model.norm.weight" in shard:
        processed["norm_weight"] = shard["model.norm.weight"]

    if "lm_head.weight" in shard:
        processed["lm_head_weight"] = shard["lm_head.weight"].T

    # Process token embeddings if they exist
    if "model.embed_tokens.weight" in shard:
        processed["tok_embedding"] = shard["model.embed_tokens.weight"]

    # Process each layer
    for layer_id in range(n_layers):
        # Get attention weights from shard
        q_weight = shard.get(f"model.layers.{layer_id}.self_attn.q_proj.weight")
        k_weight = shard.get(f"model.layers.{layer_id}.self_attn.k_proj.weight")
        v_weight = shard.get(f"model.layers.{layer_id}.self_attn.v_proj.weight")
        o_weight = shard.get(f"model.layers.{layer_id}.self_attn.o_proj.weight")

        # If the weights are present in this shard, combine them
        if q_weight is not None and k_weight is not None and v_weight is not None:
            qkv_weight = torch.cat([q_weight.T, k_weight.T, v_weight.T], axis=-1)
            processed[f"layers.{layer_id}.qkv_weight"] = qkv_weight

        if o_weight is not None:
            processed[f"layers.{layer_id}.o_weight"] = o_weight.T

        # Get normalization weights (not sharded, so will be in every shard)
        if f"model.layers.{layer_id}.self_attn.q_norm.weight" in shard:
            processed[f"layers.{layer_id}.q_norm_weight"] = shard[
                f"model.layers.{layer_id}.self_attn.q_norm.weight"
            ]
            processed[f"layers.{layer_id}.k_norm_weight"] = shard[
                f"model.layers.{layer_id}.self_attn.k_norm.weight"
            ]
            processed[f"layers.{layer_id}.input_weight"] = shard[
                f"model.layers.{layer_id}.input_layernorm.weight"
            ]
            processed[f"layers.{layer_id}.post_attention_weight"] = shard[
                f"model.layers.{layer_id}.post_attention_layernorm.weight"
            ]

        # Get router weight (not sharded, so will be in every shard)
        if f"model.layers.{layer_id}.mlp.gate.weight" in shard:
            processed[f"layers.{layer_id}.router_weight"] = shard[
                f"model.layers.{layer_id}.mlp.gate.weight"
            ].T

        # Process MoE weights - handle both old and new transformers formats
        # Check for transformers 5.0+ format (combined 3D tensors)
        gate_up_proj_key = f"model.layers.{layer_id}.mlp.experts.gate_up_proj"
        down_proj_key = f"model.layers.{layer_id}.mlp.experts.down_proj"

        if gate_up_proj_key in shard:
            # Transformers 5.0+ format: combined 3D expert tensors
            # gate_up_proj shape: (num_experts, 2*intermediate, hidden)
            # down_proj shape: (num_experts, hidden, intermediate)
            gate_up_proj = shard[gate_up_proj_key]
            down_proj = shard[down_proj_key]

            # Transpose to match expected output format:
            # gate_up_weight: (num_experts, hidden, 2*intermediate)
            # down_weight: (num_experts, intermediate, hidden)
            processed[f"layers.{layer_id}.gate_up_weight"] = gate_up_proj.transpose(
                1, 2
            )
            processed[f"layers.{layer_id}.down_weight"] = down_proj.transpose(1, 2)
        else:
            # Pre-5.0 format: separate expert tensors
            # First, find out how many experts we have
            num_experts = 0
            while (
                f"model.layers.{layer_id}.mlp.experts.{num_experts}.gate_proj.weight"
                in shard
            ):
                num_experts += 1

            # If it's an MoE layer, process the expert weights
            if num_experts > 0:
                gate_up_weights = []
                down_weights = []

                # For each expert, collect the sharded gate_proj and up_proj weights
                for expert_id in range(num_experts):
                    gate_weight = shard.get(
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                    )
                    up_weight = shard.get(
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                    )
                    down_weight = shard.get(
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
                    )

                    if gate_weight is not None and up_weight is not None:
                        # 2D tensors: gate_weight shape (intermediate, hidden)
                        # After .T: (hidden, intermediate)
                        # Concatenate to get (hidden, 2*intermediate)
                        expert_gate_up = torch.cat(
                            [gate_weight.T, up_weight.T], axis=-1
                        )
                        gate_up_weights.append(expert_gate_up)

                    if down_weight is not None:
                        # 2D tensor: down_weight shape (hidden, intermediate)
                        # After .T: (intermediate, hidden)
                        down_weights.append(down_weight.T)

                # Only save if we have collected weights for this shard
                if gate_up_weights:
                    gate_up_weight = torch.stack(gate_up_weights)
                    processed[f"layers.{layer_id}.gate_up_weight"] = gate_up_weight

                if down_weights:
                    down_weight = torch.stack(down_weights)
                    processed[f"layers.{layer_id}.down_weight"] = down_weight

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

    # Store the model config for later use
    _CONFIG = model.config

    # capture state items once for in-memory slicing
    _STATE_ITEMS = list(model.state_dict().items())

    print(f"[2/3] Splitting, post-processing, and saving {_WORLD_SIZE} shards...")

    for rank in range(_WORLD_SIZE):
        build_and_save_shard(rank, head_dim)

    print(f"[3/3] Done! {_WORLD_SIZE} post-processed shards saved in {_OUTPUT_DIR}.")


if __name__ == "__main__":
    """
    ~20 mins - 1h, we have 500G to read, process, then write, so depending on disk speed, maybe even slower

    Usage:
    python tensor_preparation.py --model-name Qwen/Qwen3-30B-A3B --world-size 8 --head-dim 128 --output-dir tmp_qwen3_30B_A3B_TP8
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre‐shard a HuggingFace model into safetensors using a custom TP plan and specific key format."
    )
    parser.add_argument(
        "--model-name", required=True, help="HF repo or local path, e.g. Qwen/Qwen-3B"
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
