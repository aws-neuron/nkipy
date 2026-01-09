"""Qwen3-Embedding-0.6B configuration"""

import os
from dataclasses import dataclass

import numpy as np
from neuronxcc.nki.language import bfloat16

# Build directory for compiled kernels
_BUILD_DIR = "/tmp/build"


def get_build_dir():
    """Get the build directory for compiled kernels"""
    return _BUILD_DIR


# Default constants
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_WEIGHTS_DIR = "tmp_qwen3_weights"
DEFAULT_WEIGHTS_FILENAME = "qwen3_weights.safetensors"


@dataclass
class Qwen3Config:
    """Configuration for Qwen3-Embedding-0.6B model"""

    # Model architecture
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # Grouped Query Attention
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
    model_name: str = DEFAULT_MODEL_NAME
    weights_path: str = None  # Will use default if None

    def __post_init__(self):
        # Validate GQA setup
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        )

        # Set default weights path if not provided
        if self.weights_path is None:
            self.weights_path = os.path.join(
                DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_FILENAME
            )
