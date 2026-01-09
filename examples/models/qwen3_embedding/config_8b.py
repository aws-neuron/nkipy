"""Qwen3-Embedding-8B configuration"""

import os
from dataclasses import dataclass

from config import Qwen3Config

# Default constants for 8B model
DEFAULT_MODEL_NAME_8B = "Qwen/Qwen3-Embedding-8B"
DEFAULT_WEIGHTS_DIR_8B = "tmp_qwen3_weights_8b"
DEFAULT_WEIGHTS_FILENAME = "qwen3_weights.safetensors"


@dataclass
class Qwen3Config_8B(Qwen3Config):
    """Configuration for Qwen3-Embedding-8B model (inherits from 0.6B base config)"""

    # Override 8B-specific architecture parameters
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    intermediate_size: int = 12288
    vocab_size: int = 151665
    max_position_embeddings: int = 40960

    # Override runtime configuration
    max_model_len: int = 8 * 1024

    # Override model source
    model_name: str = DEFAULT_MODEL_NAME_8B

    def __post_init__(self):
        # Call parent's __post_init__ for validation
        super().__post_init__()

        # Override weights path for 8B model if not provided
        if self.weights_path == os.path.join(
            "tmp_qwen3_weights", "qwen3_weights.safetensors"
        ):
            self.weights_path = os.path.join(
                DEFAULT_WEIGHTS_DIR_8B, DEFAULT_WEIGHTS_FILENAME
            )
