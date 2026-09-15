#!/usr/bin/env python3
"""Pre-shard Qwen3.5-35B-A3B weights for tensor-parallel inference on Trainium.

Handles the hybrid architecture:
- Full attention layers: Q (with gate), K, V, O projections + QK norms
- Linear attention layers: GatedDeltaNet projections, conv, state params
- MoE layers: routed experts + shared expert
"""

import fnmatch
import os

import numpy as np
import torch
from safetensors.torch import save_file

# TP plan
base_model_tp_plan = {
    # Full attention
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    # Linear attention (GatedDeltaNet)
    "layers.*.linear_attn.in_proj_qkv": "colwise",
    "layers.*.linear_attn.in_proj_z": "colwise",
    "layers.*.linear_attn.in_proj_b": "colwise",
    "layers.*.linear_attn.in_proj_a": "colwise",
    "layers.*.linear_attn.out_proj": "rowwise",
    # MoE routed experts (3D tensors)
    "layers.*.mlp.experts.gate_up_proj": "colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    # Shared expert
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
    # LM head
    "lm_head": "colwise",
}

_STATE_ITEMS = None
_WORLD_SIZE = None
_OUTPUT_DIR = None
_CONFIG = None
_LAYER_TYPES = None


def get_split_style(name: str):
    for pat, style in base_model_tp_plan.items():
        if fnmatch.fnmatch(name, f"*{pat}*"):
            return style
    return None


def get_split_dim(name: str, tensor: torch.Tensor):
    style = get_split_style(name)
    if style is None:
        return None
    ndim = tensor.ndim
    if ndim == 2:
        return 0 if style == "colwise" else 1
    elif ndim == 3:
        return 1 if style == "colwise" else 2
    else:
        return None


def shard_1d_by_heads(tensor, rank, world_size):
    """Shard a 1D tensor (e.g., dt_bias, A_log) evenly across ranks."""
    n = tensor.shape[0]
    chunk_size = n // world_size
    return tensor[rank * chunk_size : (rank + 1) * chunk_size]


def shard_conv_weight(tensor, rank, world_size):
    """Shard conv1d weight: (channels, 1, kernel_size) -> shard channels."""
    n_channels = tensor.shape[0]
    chunk_size = n_channels // world_size
    return tensor[rank * chunk_size : (rank + 1) * chunk_size]


def build_and_save_shard(rank, head_dim):
    shard = {}
    for name, tensor in _STATE_ITEMS:
        t = tensor
        dim = get_split_dim(name, tensor)

        # Special: 1D params sharded by head count
        if any(
            k in name
            for k in ["linear_attn.dt_bias", "linear_attn.A_log"]
        ):
            shard[name] = shard_1d_by_heads(t, rank, _WORLD_SIZE)
            continue

        # Special: in_proj_qkv needs interleaved Q/K/V sharding
        # Weight shape (conv_dim, hidden) where conv_dim = key_dim*2 + value_dim
        # Naive colwise shard gives rank 0 ALL of Q, rank 1 ALL of K, etc.
        # Correct: each rank gets its portion of Q, K, AND V.
        if "linear_attn.in_proj_qkv" in name:
            key_dim = _CONFIG.linear_num_key_heads * _CONFIG.linear_key_head_dim
            value_dim = _CONFIG.linear_num_value_heads * _CONFIG.linear_value_head_dim
            q_part = t[:key_dim]
            k_part = t[key_dim : key_dim * 2]
            v_part = t[key_dim * 2 :]
            q_local = q_part.chunk(_WORLD_SIZE, dim=0)[rank]
            k_local = k_part.chunk(_WORLD_SIZE, dim=0)[rank]
            v_local = v_part.chunk(_WORLD_SIZE, dim=0)[rank]
            shard[name] = torch.cat([q_local, k_local, v_local], dim=0)
            continue

        # Special: conv1d weight needs same interleaved sharding as in_proj_qkv
        # Weight shape (conv_dim, 1, kernel_size), channels match QKV ordering
        if "linear_attn.conv1d.weight" in name:
            key_dim = _CONFIG.linear_num_key_heads * _CONFIG.linear_key_head_dim
            value_dim = _CONFIG.linear_num_value_heads * _CONFIG.linear_value_head_dim
            q_ch = t[:key_dim]
            k_ch = t[key_dim : key_dim * 2]
            v_ch = t[key_dim * 2 :]
            q_local = q_ch.chunk(_WORLD_SIZE, dim=0)[rank]
            k_local = k_ch.chunk(_WORLD_SIZE, dim=0)[rank]
            v_local = v_ch.chunk(_WORLD_SIZE, dim=0)[rank]
            shard[name] = torch.cat([q_local, k_local, v_local], dim=0)
            continue

        # Don't shard
        if dim is None or t.numel() == 1 or t.size(dim) % _WORLD_SIZE != 0:
            shard[name] = t
            continue

        # KV projection: shard till head dim, then replicate
        if ("k_proj" in name or "v_proj" in name) and (
            t.shape[dim] // _WORLD_SIZE < head_dim
        ):
            n_kv_heads = tensor.shape[0] // head_dim
            tensor_r = tensor.reshape(-1, head_dim, tensor.shape[1])
            head_index = np.floor(n_kv_heads * rank / _WORLD_SIZE).astype(int)
            part = tensor_r[head_index]
            shard[name] = part
            continue

        # 3D expert gate_up_proj: shard gate and up separately
        if "experts.gate_up_proj" in name and t.ndim == 3:
            intermediate = t.shape[1] // 2
            gate = t[:, :intermediate, :]
            up = t[:, intermediate:, :]
            gate_part = gate.chunk(_WORLD_SIZE, dim=1)[rank]
            up_part = up.chunk(_WORLD_SIZE, dim=1)[rank]
            part = torch.cat([gate_part, up_part], dim=1)
            shard[name] = part
            continue

        # Normal shard
        part = t.chunk(_WORLD_SIZE, dim=dim)[rank]
        shard[name] = part

    processed_shard = post_process_shard(shard, rank)

    for name, part in processed_shard.items():
        processed_shard[name] = part.contiguous()

    path = os.path.join(_OUTPUT_DIR, f"shard_{rank}.safetensors")
    save_file(processed_shard, path)
    del shard, processed_shard


def post_process_shard(shard, rank):
    """Transform shard weights into the target format for inference."""
    processed = {}
    n_layers = _CONFIG.num_hidden_layers

    # Token embeddings and final norm/head
    if "model.norm.weight" in shard:
        processed["norm_weight"] = shard["model.norm.weight"]
    if "lm_head.weight" in shard:
        processed["lm_head_weight"] = shard["lm_head.weight"].T
    if "model.embed_tokens.weight" in shard:
        processed["tok_embedding"] = shard["model.embed_tokens.weight"]

    for layer_id in range(n_layers):
        prefix = f"model.layers.{layer_id}"
        layer_type = _LAYER_TYPES[layer_id]

        # --- Common: layer norms ---
        if f"{prefix}.input_layernorm.weight" in shard:
            processed[f"layers.{layer_id}.input_weight"] = shard[
                f"{prefix}.input_layernorm.weight"
            ]
            processed[f"layers.{layer_id}.post_attention_weight"] = shard[
                f"{prefix}.post_attention_layernorm.weight"
            ]

        # --- Full attention layer ---
        if layer_type == "full_attention":
            q_weight = shard.get(f"{prefix}.self_attn.q_proj.weight")
            k_weight = shard.get(f"{prefix}.self_attn.k_proj.weight")
            v_weight = shard.get(f"{prefix}.self_attn.v_proj.weight")
            o_weight = shard.get(f"{prefix}.self_attn.o_proj.weight")

            if q_weight is not None and k_weight is not None and v_weight is not None:
                # q_weight includes gate (2x head_dim per head)
                qkv_weight = torch.cat(
                    [q_weight.T, k_weight.T, v_weight.T], axis=-1
                )
                processed[f"layers.{layer_id}.qkv_weight"] = qkv_weight

            if o_weight is not None:
                processed[f"layers.{layer_id}.o_weight"] = o_weight.T

            if f"{prefix}.self_attn.q_norm.weight" in shard:
                processed[f"layers.{layer_id}.q_norm_weight"] = shard[
                    f"{prefix}.self_attn.q_norm.weight"
                ]
                processed[f"layers.{layer_id}.k_norm_weight"] = shard[
                    f"{prefix}.self_attn.k_norm.weight"
                ]

        # --- Linear attention layer (GatedDeltaNet) ---
        elif layer_type == "linear_attention":
            # QKV projection
            w = shard.get(f"{prefix}.linear_attn.in_proj_qkv.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_qkv_weight"] = w.T

            # Z projection (gate for RMSNormGated)
            w = shard.get(f"{prefix}.linear_attn.in_proj_z.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_z_weight"] = w.T

            # Beta projection
            w = shard.get(f"{prefix}.linear_attn.in_proj_b.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_b_weight"] = w.T

            # Alpha projection
            w = shard.get(f"{prefix}.linear_attn.in_proj_a.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_a_weight"] = w.T

            # Conv1d weight: (channels, 1, kernel_size) -> (channels, kernel_size)
            w = shard.get(f"{prefix}.linear_attn.conv1d.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_conv_weight"] = w.squeeze(1)

            # dt_bias
            w = shard.get(f"{prefix}.linear_attn.dt_bias")
            if w is not None:
                processed[f"layers.{layer_id}.linear_dt_bias"] = w

            # A_log
            w = shard.get(f"{prefix}.linear_attn.A_log")
            if w is not None:
                processed[f"layers.{layer_id}.linear_A_log"] = w

            # RMSNormGated weight
            w = shard.get(f"{prefix}.linear_attn.norm.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_norm_weight"] = w

            # Output projection
            w = shard.get(f"{prefix}.linear_attn.out_proj.weight")
            if w is not None:
                processed[f"layers.{layer_id}.linear_out_weight"] = w.T

        # --- MoE ---
        # Router weight
        if f"{prefix}.mlp.gate.weight" in shard:
            processed[f"layers.{layer_id}.router_weight"] = shard[
                f"{prefix}.mlp.gate.weight"
            ].T

        # Routed experts (transformers 5.0+ 3D format)
        gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
        down_key = f"{prefix}.mlp.experts.down_proj"
        if gate_up_key in shard:
            processed[f"layers.{layer_id}.gate_up_weight"] = shard[
                gate_up_key
            ].transpose(1, 2)
            processed[f"layers.{layer_id}.down_weight"] = shard[
                down_key
            ].transpose(1, 2)
        else:
            # Pre-5.0 format: separate expert tensors
            num_experts = 0
            while f"{prefix}.mlp.experts.{num_experts}.gate_proj.weight" in shard:
                num_experts += 1

            if num_experts > 0:
                gate_up_weights = []
                down_weights = []
                for expert_id in range(num_experts):
                    gate_w = shard.get(
                        f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight"
                    )
                    up_w = shard.get(
                        f"{prefix}.mlp.experts.{expert_id}.up_proj.weight"
                    )
                    down_w = shard.get(
                        f"{prefix}.mlp.experts.{expert_id}.down_proj.weight"
                    )
                    if gate_w is not None and up_w is not None:
                        gate_up_weights.append(
                            torch.cat([gate_w.T, up_w.T], axis=-1)
                        )
                    if down_w is not None:
                        down_weights.append(down_w.T)

                if gate_up_weights:
                    processed[f"layers.{layer_id}.gate_up_weight"] = torch.stack(
                        gate_up_weights
                    )
                if down_weights:
                    processed[f"layers.{layer_id}.down_weight"] = torch.stack(
                        down_weights
                    )

        # Shared expert
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            key = f"{prefix}.mlp.shared_expert.{proj_name}.weight"
            if key in shard:
                processed[f"layers.{layer_id}.shared_{proj_name}_weight"] = shard[
                    key
                ].T

        # Shared expert gate
        key = f"{prefix}.mlp.shared_expert_gate.weight"
        if key in shard:
            processed[f"layers.{layer_id}.shared_expert_gate_weight"] = shard[key].T

    return processed


def preshard_model(
    model_name: str,
    output_dir: str,
    world_size: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
):
    global _STATE_ITEMS, _WORLD_SIZE, _OUTPUT_DIR, _CONFIG, _LAYER_TYPES

    os.makedirs(output_dir, exist_ok=True)
    _WORLD_SIZE = world_size
    _OUTPUT_DIR = output_dir

    print(f"[1/3] Loading full model `{model_name}` onto CPU...")

    from transformers import AutoConfig, AutoModelForCausalLM

    # Load config to get layer types
    hf_config = AutoConfig.from_pretrained(model_name)
    text_cfg = (
        hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    )
    _LAYER_TYPES = list(text_cfg.layer_types)
    _CONFIG = text_cfg

    # Try loading as CausalLM first (text-only)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            dtype=dtype,
            low_cpu_mem_usage=False,
        )
    except Exception:
        # Fallback: load full multimodal model and extract language model
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            model_name,
            device_map="cpu",
            dtype=dtype,
            low_cpu_mem_usage=False,
        )
        model = full_model.language_model

    _STATE_ITEMS = list(model.state_dict().items())

    print(f"[2/3] Splitting, post-processing, and saving {_WORLD_SIZE} shards...")

    for rank in range(_WORLD_SIZE):
        build_and_save_shard(rank, head_dim)

    print(f"[3/3] Done! {_WORLD_SIZE} post-processed shards saved in {_OUTPUT_DIR}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-shard Qwen3.5-35B-A3B for tensor-parallel inference."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="HF repo or local path, e.g. Qwen/Qwen3.5-35B-A3B",
    )
    parser.add_argument("--output-dir", default="qwen3_5_shards")
    parser.add_argument(
        "--world-size", type=int, required=True, help="Number of tensor-parallel ranks"
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16", "bf16"],
        default="bf16",
        help="Data type to load/save",
    )
    parser.add_argument("--head-dim", type=int, default=256, help="The head dim size")

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
