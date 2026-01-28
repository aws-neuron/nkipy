"""Qwen3 Embedding Model for Trainium."""

from .config import QWEN3_0_6B_CONFIG, QWEN3_8B_CONFIG, Qwen3Config, get_config
from .embedding_utils import (
    get_detailed_instruct,
    last_token_pool,
    normalize_embeddings,
)
from .model import Qwen3EmbeddingModel
from .prepare_weights import download_and_convert_qwen3_weights, load_qwen3_weights

__all__ = [
    # Config
    "Qwen3Config",
    "QWEN3_0_6B_CONFIG",
    "QWEN3_8B_CONFIG",
    "get_config",
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
