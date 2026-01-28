"""Qwen3-Embedding model configurations for 0.6B and 8B variants."""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from neuronxcc.nki.language import bfloat16

# Build directory for compiled kernels (local to this example)
BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")


@dataclass
class Qwen3Config:
    """Configuration for Qwen3-Embedding models.

    Supports both 0.6B and 8B model variants with appropriate defaults.
    """

    # Model architecture
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 151669

    # Normalization
    rms_norm_eps: float = 1e-6

    # RoPE (Rotary Position Embedding)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Runtime configuration
    max_batch_size: int = 1
    max_model_len: int = 128
    dtype: np.dtype = bfloat16

    # Model source
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    weights_dir: str = "tmp_qwen3_weights"
    weights_filename: str = "qwen3_weights.safetensors"
    weights_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # Validate GQA setup
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible "
            f"by num_key_value_heads ({self.num_key_value_heads})"
        )
        # Set default weights path if not provided
        if self.weights_path is None:
            self.weights_path = os.path.join(self.weights_dir, self.weights_filename)


# Pre-configured model instances
QWEN3_0_6B_CONFIG = Qwen3Config(
    # 0.6B model defaults (already set in dataclass)
)

QWEN3_8B_CONFIG = Qwen3Config(
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=12288,
    vocab_size=151665,
    max_position_embeddings=40960,
    max_model_len=128,
    model_name="Qwen/Qwen3-Embedding-8B",
    weights_dir="tmp_qwen3_weights_8b",
)


def get_config(model_size: str = "0.6b", **overrides) -> Qwen3Config:
    """Get a model configuration by size.

    Args:
        model_size: Model size, either "0.6b" or "8b"
        **overrides: Override any config parameters

    Returns:
        Qwen3Config instance

    Example:
        config = get_config("8b", max_model_len=4096)
    """
    model_size = model_size.lower()

    if model_size in ("0.6b", "0.6"):
        base = QWEN3_0_6B_CONFIG
    elif model_size in ("8b", "8"):
        base = QWEN3_8B_CONFIG
    else:
        raise ValueError(f"Unknown model size: {model_size}. Use '0.6b' or '8b'.")

    # Create a new config with overrides
    if overrides:
        from dataclasses import asdict

        config_dict = asdict(base)
        config_dict.update(overrides)
        # Reset weights_path to trigger recalculation if weights_dir changed
        if "weights_dir" in overrides and "weights_path" not in overrides:
            config_dict["weights_path"] = None
        return Qwen3Config(**config_dict)

    return base
