#!/usr/bin/env python3
import argparse
import os

import torch
from config import DEFAULT_MODEL_NAME, DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_FILENAME
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer


def download_and_convert_qwen3_weights(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_WEIGHTS_DIR,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Download Qwen3 model from HuggingFace and convert to our format
    """
    print(f"Downloading {model_name} from HuggingFace...")

    # Download model and tokenizer
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Converting weights to our format...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert weights
    converted_weights = {}

    # Token embedding
    converted_weights["tok_embedding"] = model.embed_tokens.weight

    # Final layer norm
    converted_weights["norm_weight"] = model.norm.weight

    # Process each layer
    for layer_id, layer in enumerate(model.layers):
        layer_prefix = f"layers.{layer_id}"

        # Attention weights - fuse Q, K, V projections
        q_weight = layer.self_attn.q_proj.weight  # [hidden_size, hidden_size]
        k_weight = layer.self_attn.k_proj.weight  # [kv_hidden_size, hidden_size]
        v_weight = layer.self_attn.v_proj.weight  # [kv_hidden_size, hidden_size]

        # Concatenate QKV weights: [q_hidden + kv_hidden + kv_hidden, hidden_size]
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        converted_weights[f"{layer_prefix}.qkv_weight"] = (
            qkv_weight.T
        )  # Transpose for our format

        # Q and K normalization weights
        converted_weights[f"{layer_prefix}.q_norm_weight"] = (
            layer.self_attn.q_norm.weight
        )
        converted_weights[f"{layer_prefix}.k_norm_weight"] = (
            layer.self_attn.k_norm.weight
        )

        # Output projection
        converted_weights[f"{layer_prefix}.o_weight"] = layer.self_attn.o_proj.weight.T

        # Layer norms
        converted_weights[f"{layer_prefix}.input_layernorm_weight"] = (
            layer.input_layernorm.weight
        )
        converted_weights[f"{layer_prefix}.post_attention_layernorm_weight"] = (
            layer.post_attention_layernorm.weight
        )

        # MLP weights - fuse gate and up projections
        gate_weight = layer.mlp.gate_proj.weight  # [intermediate_size, hidden_size]
        up_weight = layer.mlp.up_proj.weight  # [intermediate_size, hidden_size]

        # Concatenate gate and up weights: [hidden_size, 2 * intermediate_size]
        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0).T
        converted_weights[f"{layer_prefix}.gate_up_weight"] = gate_up_weight

        # Down projection
        converted_weights[f"{layer_prefix}.down_weight"] = layer.mlp.down_proj.weight.T

        # Note: Qwen3 doesn't use bias by default, so we skip bias weights

    # Make all tensors contiguous before saving
    for name in converted_weights:
        converted_weights[name] = converted_weights[name].contiguous()

    # Save converted weights
    output_path = os.path.join(output_dir, DEFAULT_WEIGHTS_FILENAME)
    save_file(converted_weights, output_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save config info
    config_info = {
        "model_name": model_name,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "intermediate_size": model.config.intermediate_size,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rms_norm_eps": model.config.rms_norm_eps,
        "rope_theta": model.config.rope_theta,
    }

    import json

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"Weights saved to {output_path}")
    print(f"Tokenizer saved to {output_dir}")
    print(f"Config saved to {os.path.join(output_dir, 'model_config.json')}")

    # Print weight shapes for verification
    print("Weight shapes:")
    for name, weight in converted_weights.items():
        print(f"  {name}: {weight.shape}")

    return output_path


def load_qwen3_weights(weights_path: str) -> dict:
    """Load converted Qwen3 weights from safetensors file"""
    from safetensors.torch import load_file

    print(f"Loading weights from {weights_path}")
    weights = load_file(weights_path, device="cpu")

    # Convert to our expected format (torch tensors)
    converted = {}
    for name, tensor in weights.items():
        converted[name] = tensor

    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert Qwen3 weights")
    parser.add_argument(
        "--model-name", default=DEFAULT_MODEL_NAME, help="HuggingFace model name"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_WEIGHTS_DIR,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type for weights",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    download_and_convert_qwen3_weights(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dtype=dtype_map[args.dtype],
    )
