"""Qwen3 Embedding Model for Trainium"""

from .config import Qwen3Config
from .config_8b import Qwen3Config_8B
from .embedding_utils import (
    get_detailed_instruct,
    last_token_pool,
    normalize_embeddings,
)
from .model import Qwen3EmbeddingModel
from .prepare_weights import download_and_convert_qwen3_weights, load_qwen3_weights

__all__ = [
    # Configs
    "Qwen3Config",
    "Qwen3Config_8B",
    # Model
    "Qwen3EmbeddingModel",
    # Weight utilities
    "download_and_convert_qwen3_weights",
    "load_qwen3_weights",
    # Embedding utilities
    "normalize_embeddings",
    "last_token_pool",
    "get_detailed_instruct",
]
