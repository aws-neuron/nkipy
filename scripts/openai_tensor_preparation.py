#!/usr/bin/env python3
import os
import fnmatch
import glob
import torch
import numpy as np
from safetensors.torch import save_file
from safetensors import safe_open
from tqdm.auto import tqdm

# Ensure fork start method for shared-memory inheritance
import multiprocessing as mp

mp.set_start_method("fork", force=True)

# TP plan for OpenAI model: pattern → split style
# Shard attention on the head, and shard MLP on the intermediate
openai_model_tp_plan = {
    # Attention weights - shard on head dimension
    "block.*.attn.qkv.weight": "colwise",  # shard on output dim (head)
    "block.*.attn.qkv.bias": "colwise",  # shard on output dim (head)
    "block.*.attn.out.weight": "rowwise",  # shard on input dim
    "block.*.attn.out.bias": "replicate",  # don't shard bias
    "block.*.attn.sinks": "colwise",
    # MLP weights - shard on intermediate dimension
    "block.*.mlp.mlp1_weight": "colwise",  # shard on intermediate dim
    "block.*.mlp.mlp1_bias": "rowwise",  # shard on intermediate dim
    "block.*.mlp.mlp2_weight": "rowwise",  # shard on input dim (intermediate)
    "block.*.mlp.mlp2_bias": "replicate",  # don't shard output bias
    # MLP gate (router) - replicate
    "block.*.mlp.gate.weight": "replicate",
    "block.*.mlp.gate.bias": "replicate",
    # Normalization - replicate
    "block.*.attn.norm.scale": "replicate",
    "block.*.mlp.norm.scale": "replicate",
    # Global weights
    "embedding.weight": "replicate",  # shard on vocab dimension
    "unembedding.weight": "colwise",  # shard on input dimension
    "norm.scale": "replicate",
}

# Globals inherited via fork
_STATE_ITEMS = None
_WORLD_SIZE = None
_OUTPUT_DIR = None
_NUM_LAYERS = None
_HEAD_DIM = None
_DTYPE = None


def get_split_dim(name: str):
    """Return the dimension to split along for this parameter name, or None to replicate."""
    shard_axis = None
    for pat, style in openai_model_tp_plan.items():
        if fnmatch.fnmatch(name, pat):
            if style == "colwise":
                shard_axis = 0
            elif style == "rowwise":
                shard_axis = 1
            else:  # replicate
                shard_axis = None
            break
    return shard_axis


def load_single_safetensors_file(file_path):
    """Load tensors from a single safetensors file."""
    tensors = []
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype != _DTYPE:
                tensor = tensor.to(_DTYPE)
            tensors.append((key, tensor))
    return tensors


def build_and_save_shard(args):
    """Construct shard dict for given rank and write to safetensors."""
    rank, num_threads = args

    # Limit PyTorch threads for this process
    torch.set_num_threads(num_threads)

    shard = {}
    for name, tensor in _STATE_ITEMS:
        t = tensor
        dim = get_split_dim(name)

        # Case 1: Don't shard this tensor (replicate)
        if dim is None or t.numel() == 1 or t.size(dim) % _WORLD_SIZE != 0:
            shard[name] = t
            continue

        # Case 2: Special handling for attention QKV - shard on head dimension
        if "attn.qkv" in name and dim == 0:
            # For QKV, we need to handle different head counts for Q, K, V
            # Assuming format: [q_heads*head_dim + k_heads*head_dim + v_heads*head_dim, hidden_dim]

            # STEP 1: Split QKV tensor into Q, K, V components
            # Based on the comment, this appears to be [64+8+8, 64, 2880] format
            # where 64 is q_heads*head_dim, 8 is k_heads*head_dim, 8 is v_heads*head_dim
            total_dim = t.size(0)

            # Calculate head counts (assuming the pattern from comments)
            # This is a heuristic - in practice, these would be model-specific parameters
            if total_dim % _HEAD_DIM == 0:
                # Assume Q gets most heads, K and V get fewer (typical in GQA)
                # For now, assume equal distribution and adjust if needed
                q_heads = 64
                kv_heads = 8

                q_dim = q_heads * _HEAD_DIM
                k_dim = kv_heads * _HEAD_DIM
                v_dim = kv_heads * _HEAD_DIM

                # Split the tensor
                q_tensor = t[:q_dim]
                k_tensor = t[q_dim : q_dim + k_dim]
                v_tensor = t[q_dim + k_dim : q_dim + k_dim + v_dim]

                # STEP 2: Shard each component separately
                # Shard Q normally if possible
                if q_heads % _WORLD_SIZE == 0:
                    q_part = q_tensor.chunk(_WORLD_SIZE, dim=0)[rank]
                else:
                    # If can't shard evenly, replicate
                    q_part = q_tensor

                # For K and V, use head selection logic if can't shard evenly
                if kv_heads >= _WORLD_SIZE and kv_heads % _WORLD_SIZE == 0:
                    k_part = k_tensor.chunk(_WORLD_SIZE, dim=0)[rank]
                    v_part = v_tensor.chunk(_WORLD_SIZE, dim=0)[rank]
                else:
                    # Use head selection for K and V
                    if kv_heads > 0:
                        k_reshaped = k_tensor.reshape(
                            kv_heads, _HEAD_DIM, *k_tensor.shape[1:]
                        )
                        v_reshaped = v_tensor.reshape(
                            kv_heads, _HEAD_DIM, *v_tensor.shape[1:]
                        )

                        head_index = int(np.floor(kv_heads * rank / _WORLD_SIZE))
                        head_index = min(head_index, kv_heads - 1)  # Ensure valid index

                        k_part = k_reshaped[head_index].reshape(
                            _HEAD_DIM, *k_tensor.shape[1:]
                        )
                        v_part = v_reshaped[head_index].reshape(
                            _HEAD_DIM, *k_tensor.shape[1:]
                        )
                    else:
                        k_part = k_tensor
                        v_part = v_tensor

                # STEP 3: Concatenate the sharded QKV tensor together
                part = torch.cat([q_part, k_part, v_part], dim=0)
                shard[name] = part
            else:
                raise AssertionError("Unexpected tensor shape for attn.qkv")
            continue

        if "qkv.bias" in name and dim == 0:
            if t.size(dim) // _WORLD_SIZE >= _HEAD_DIM:
                # Normal sharding
                part = t.chunk(_WORLD_SIZE, dim=dim)[rank]
                shard[name] = part
            else:
                raise AssertionError("Unexpected tensor shape for attn.qkv")

        # Case 3: Special handling for MoE weights - never shard expert dimension
        if "mlp.mlp" in name and len(t.shape) == 3:
            # For MoE weights [experts, intermediate, hidden], only shard intermediate dimension
            if dim == 0:  # This would be colwise sharding on intermediate
                # Shard on intermediate dimension (dim=1), keep experts intact
                experts, intermediate, hidden = t.shape
                intermediate_per_shard = intermediate // _WORLD_SIZE
                start_idx = rank * intermediate_per_shard
                end_idx = (rank + 1) * intermediate_per_shard
                part = t[
                    :, start_idx:end_idx, :
                ]  # Keep all experts, shard intermediate
                shard[name] = part
            elif dim == 1:  # This would be rowwise sharding on input (intermediate)
                # Shard on intermediate dimension (dim=2 for input), keep experts intact
                experts, intermediate, hidden = t.shape
                hidden_per_shard = hidden // _WORLD_SIZE
                start_idx = rank * hidden_per_shard
                end_idx = (rank + 1) * hidden_per_shard
                part = t[:, :, start_idx:end_idx]  # Keep all experts, shard hidden
                shard[name] = part
            else:
                # Fallback to normal sharding
                part = t.chunk(_WORLD_SIZE, dim=dim)[rank]
                shard[name] = part
        else:
            # Case 4: shard normally for non-MoE tensors
            part = t.chunk(_WORLD_SIZE, dim=dim)[rank]
            shard[name] = part

    # Process shard to new format before saving
    processed_shard = post_process_shard(shard, rank)

    # Make all tensors contiguous
    for name, part in processed_shard.items():
        processed_shard[name] = part.contiguous()

    path = os.path.join(_OUTPUT_DIR, f"shard_{rank}.safetensors")
    save_file(processed_shard, path)
    del shard, processed_shard


def post_process_shard(shard, rank):
    """Transform the shard weights into the target format for OpenAI model"""
    processed = {}

    # Process global weights
    if "norm.scale" in shard:
        processed["norm_weight"] = shard["norm.scale"]

    if "unembedding.weight" in shard:
        processed["lm_head_weight"] = shard["unembedding.weight"].T

    if "embedding.weight" in shard:
        processed["tok_embedding"] = shard["embedding.weight"]

    # Process each layer
    for layer_id in range(_NUM_LAYERS):
        layer_prefix = f"block.{layer_id}"

        # Attention weights
        qkv_weight = shard.get(f"{layer_prefix}.attn.qkv.weight")
        qkv_bias = shard.get(f"{layer_prefix}.attn.qkv.bias")
        out_weight = shard.get(f"{layer_prefix}.attn.out.weight")
        out_bias = shard.get(f"{layer_prefix}.attn.out.bias")

        if qkv_weight is not None:
            # Transpose QKV weight to [hidden, head] format
            processed[f"layers.{layer_id}.qkv_weight"] = qkv_weight.T
        if qkv_bias is not None:
            processed[f"layers.{layer_id}.qkv_bias"] = qkv_bias
        if out_weight is not None:
            # Transpose output weight to [intermediate, hidden] format
            processed[f"layers.{layer_id}.o_weight"] = out_weight.T
        if out_bias is not None:
            processed[f"layers.{layer_id}.o_bias"] = out_bias

        # Attention normalization and sinks
        attn_norm = shard.get(f"{layer_prefix}.attn.norm.scale")
        attn_sinks = shard.get(f"{layer_prefix}.attn.sinks")

        if attn_norm is not None:
            processed[f"layers.{layer_id}.attn_norm_weight"] = attn_norm
        if attn_sinks is not None:
            processed[f"layers.{layer_id}.attn_sinks"] = attn_sinks

        # MLP weights
        mlp1_weight = shard.get(f"{layer_prefix}.mlp.mlp1_weight")
        mlp1_bias = shard.get(f"{layer_prefix}.mlp.mlp1_bias")
        mlp2_weight = shard.get(f"{layer_prefix}.mlp.mlp2_weight")
        mlp2_bias = shard.get(f"{layer_prefix}.mlp.mlp2_bias")

        if mlp1_weight is not None:
            # Handle MLP1: [32, intermediate_shard*2(interleaved), 2880] -> [32, 2880, intermediate_shard*2 (not interleaved)]
            # The intermediate dimension contains gate and up weights that need to be combined
            # Following the original tensor_preparation.py pattern: gate_up = [gate.T, up.T] concatenated
            assert len(mlp1_weight.shape) == 3
            experts, intermediate_shard, hidden = mlp1_weight.shape
            # Transpose to [experts, hidden, intermediate_shard]
            transposed = mlp1_weight.transpose(1, 2)  # [32, 2880, intermediate_shard]

            # interleaved to non interleaved
            gate_up_weight = torch.cat(
                [transposed[..., ::2], transposed[..., 1::2]], dim=-1
            )  # [32, 2880, intermediate_shard*2]
            processed[f"layers.{layer_id}.gate_up_weight"] = gate_up_weight

        if mlp1_bias is not None:
            # Handle MLP1 bias: create gate_up bias format
            assert len(mlp1_bias.shape) == 2
            # Concatenate gate and up biases: [experts, intermediate_shard*2]
            # interleaved to non interleaved
            gate_up_bias = torch.cat(
                [mlp1_bias[..., ::2], mlp1_bias[..., 1::2]], dim=-1
            )
            processed[f"layers.{layer_id}.gate_up_bias"] = gate_up_bias

        if mlp2_weight is not None:
            # Transpose to [experts, intermediate, hidden_shard]
            processed[f"layers.{layer_id}.down_weight"] = mlp2_weight.transpose(1, 2)

        if mlp2_bias is not None:
            processed[f"layers.{layer_id}.down_bias"] = mlp2_bias

        # MLP gate (router) and normalization
        gate_weight = shard.get(f"{layer_prefix}.mlp.gate.weight")
        gate_bias = shard.get(f"{layer_prefix}.mlp.gate.bias")
        mlp_norm = shard.get(f"{layer_prefix}.mlp.norm.scale")

        if gate_weight is not None:
            processed[f"layers.{layer_id}.router_weight"] = gate_weight.T
        if gate_bias is not None:
            processed[f"layers.{layer_id}.router_bias"] = gate_bias
        if mlp_norm is not None:
            processed[f"layers.{layer_id}.mlp_norm_weight"] = mlp_norm

    return processed


def preshard_openai_model(
    model_dir: str,
    output_dir: str,
    world_size: int,
    head_dim: int,
    num_layers: int,
    dtype: torch.dtype = torch.bfloat16,
    num_workers: int = None,
    threads_per_worker: int = 1,
):
    global _STATE_ITEMS, _WORLD_SIZE, _OUTPUT_DIR, _NUM_LAYERS, _HEAD_DIM, _DTYPE

    os.makedirs(output_dir, exist_ok=True)
    _WORLD_SIZE = world_size
    _OUTPUT_DIR = output_dir
    _NUM_LAYERS = num_layers
    _HEAD_DIM = head_dim
    _DTYPE = dtype

    print(f"[1/3] Loading OpenAI model from directory `{model_dir}`…")

    # Find all safetensors files in the directory
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in directory: {model_dir}")
    
    print(f"Found {len(safetensors_files)} safetensors files to load")
    
    # Determine number of workers for loading files
    load_workers = min(mp.cpu_count(), len(safetensors_files))
    
    # Load all safetensors files in parallel
    state_items = []
    with mp.Pool(processes=load_workers) as pool:
        file_tensors = list(
            tqdm(
                pool.imap(load_single_safetensors_file, safetensors_files),
                total=len(safetensors_files),
                desc="Loading files",
            )
        )
    
    # Combine all tensors from all files
    for tensors in file_tensors:
        state_items.extend(tensors)
    
    print(f"Loaded {len(state_items)} tensors from {len(safetensors_files)} files")
    
    _STATE_ITEMS = state_items

    # Determine number of workers if not specified
    if num_workers is None:
        num_workers = min(mp.cpu_count(), world_size)

    print(
        f"[2/3] Splitting, post-processing, and saving {_WORLD_SIZE} shards using {num_workers} workers..."
    )

    # Create a pool with the desired number of processes
    with mp.Pool(processes=num_workers) as pool:
        args_list = [(rank, threads_per_worker) for rank in range(_WORLD_SIZE)]

        # Map the worker function with arguments
        list(
            tqdm(
                pool.imap(build_and_save_shard, args_list),
                total=_WORLD_SIZE,
                desc="Sharding progress",
            )
        )

    print(f"[3/3] Done! {_WORLD_SIZE} post-processed shards saved in {_OUTPUT_DIR}.")


if __name__ == "__main__":
    """
    Usage:
    python openai_tensor_preparation.py --model-dir openai-20b-bf16/ --world-size 8 --num-layers 24 --head-dim 64 --output-dir openai_20b_shards_TP8
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre‐shard OpenAI model safetensors files from a directory using a custom TP plan."
    )
    parser.add_argument(
        "--model-dir", required=True, help="Path to directory containing OpenAI model safetensors files"
    )
    parser.add_argument("--output-dir", default="openai_sharded_safetensors")
    parser.add_argument(
        "--world-size", type=int, required=True, help="Number of tensor-parallel ranks"
    )
    parser.add_argument(
        "--num-layers", type=int, required=True, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16", "bf16"],
        default="bf16",
        help="Data type to load/save",
    )
    parser.add_argument("--head-dim", type=int, default=64, help="The head dim size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Number of PyTorch threads per worker",
    )

    args = parser.parse_args()
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[
        args.dtype
    ]

    preshard_openai_model(
        args.model_dir,
        args.output_dir,
        args.world_size,
        args.head_dim,
        args.num_layers,
        dtype=dtype,
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
    )
