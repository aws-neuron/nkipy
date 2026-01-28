#!/usr/bin/env python3
"""Download and convert Qwen3 model weights for Trainium."""

import argparse
import os

import torch
from config import get_config
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer


def download_and_convert_qwen3_weights(
    model_name: str,
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Download Qwen3 model from HuggingFace and convert to our format.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save converted weights
        dtype: Data type for weights (default: bfloat16)
    """
    print(f"Downloading {model_name} from HuggingFace...")

    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Converting weights...")
    os.makedirs(output_dir, exist_ok=True)

    converted_weights = {}

    # Token embedding
    converted_weights["tok_embedding"] = model.embed_tokens.weight

    # Final layer norm
    converted_weights["norm_weight"] = model.norm.weight

    # Process each layer
    for layer_id, layer in enumerate(model.layers):
        prefix = f"layers.{layer_id}"

        # Fuse Q, K, V projections
        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        converted_weights[f"{prefix}.qkv_weight"] = qkv_weight.T

        # Q and K normalization weights
        converted_weights[f"{prefix}.q_norm_weight"] = layer.self_attn.q_norm.weight
        converted_weights[f"{prefix}.k_norm_weight"] = layer.self_attn.k_norm.weight

        # Output projection
        converted_weights[f"{prefix}.o_weight"] = layer.self_attn.o_proj.weight.T

        # Layer norms
        converted_weights[f"{prefix}.input_layernorm_weight"] = (
            layer.input_layernorm.weight
        )
        converted_weights[f"{prefix}.post_attention_layernorm_weight"] = (
            layer.post_attention_layernorm.weight
        )

        # Fuse gate and up projections
        gate_weight = layer.mlp.gate_proj.weight
        up_weight = layer.mlp.up_proj.weight
        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0).T
        converted_weights[f"{prefix}.gate_up_weight"] = gate_up_weight

        # Down projection
        converted_weights[f"{prefix}.down_weight"] = layer.mlp.down_proj.weight.T

    # Make all tensors contiguous
    for name in converted_weights:
        converted_weights[name] = converted_weights[name].contiguous()

    # Save weights
    output_path = os.path.join(output_dir, "qwen3_weights.safetensors")
    save_file(converted_weights, output_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Weights saved to {output_path}")
    print(f"Tokenizer saved to {output_dir}")

    # Print weight shapes
    print("\nWeight shapes:")
    for name, weight in converted_weights.items():
        print(f"  {name}: {weight.shape}")

    return output_path


def load_qwen3_weights(weights_path: str) -> dict:
    """Load converted Qwen3 weights from safetensors file.

    Args:
        weights_path: Path to the safetensors file

    Returns:
        Dictionary of weight tensors
    """
    print(f"Loading weights from {weights_path}")
    weights = load_file(weights_path, device="cpu")
    return dict(weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert Qwen3 weights")
    parser.add_argument(
        "--model-size",
        choices=["0.6b", "8b"],
        default="0.6b",
        help="Model size (default: 0.6b)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: use config default)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type for weights (default: bfloat16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    config = get_config(args.model_size)
    output_dir = args.output_dir or config.weights_dir

    download_and_convert_qwen3_weights(
        model_name=config.model_name,
        output_dir=output_dir,
        dtype=dtype_map[args.dtype],
    )
