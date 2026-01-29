"""Qwen3 model configurations and registry.

Single source of truth for all supported Qwen3 dense models.
All Qwen3 dense models have 8 KV heads, which allows TP=1, 2, 4, or 8.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a Qwen3 model."""

    model_id: str
    hidden_size: int
    num_layers: int
    num_kv_heads: int = 8  # All Qwen3 dense models have 8 KV heads
    recommended_tp: List[int] = None  # Recommended TP degrees

    def __post_init__(self):
        if self.recommended_tp is None:
            self.recommended_tp = [1]


# Qwen3 Dense Model Registry
QWEN3_MODELS: Dict[str, ModelConfig] = {
    # Small models - single device
    "Qwen/Qwen3-0.6B": ModelConfig(
        model_id="Qwen/Qwen3-0.6B",
        hidden_size=1024,
        num_layers=28,
        recommended_tp=[1],
    ),
    "Qwen/Qwen3-1.7B": ModelConfig(
        model_id="Qwen/Qwen3-1.7B",
        hidden_size=1536,
        num_layers=28,
        recommended_tp=[1, 2],
    ),
    # Medium models - single or distributed
    "Qwen/Qwen3-4B": ModelConfig(
        model_id="Qwen/Qwen3-4B",
        hidden_size=2560,
        num_layers=36,
        recommended_tp=[1, 2, 4],
    ),
    # Large models - distributed
    "Qwen/Qwen3-8B": ModelConfig(
        model_id="Qwen/Qwen3-8B",
        hidden_size=4096,
        num_layers=36,
        recommended_tp=[4, 8],
    ),
    "Qwen/Qwen3-14B": ModelConfig(
        model_id="Qwen/Qwen3-14B",
        hidden_size=5120,
        num_layers=40,
        recommended_tp=[8],
    ),
    "Qwen/Qwen3-32B": ModelConfig(
        model_id="Qwen/Qwen3-32B",
        hidden_size=5120,
        num_layers=64,
        recommended_tp=[8, 16],
    ),
}


def get_all_models() -> List[str]:
    """Return all supported model IDs."""
    return list(QWEN3_MODELS.keys())


def get_single_device_models() -> List[str]:
    """Return models suitable for TP=1 (single device)."""
    return [
        model_id
        for model_id, config in QWEN3_MODELS.items()
        if 1 in config.recommended_tp
    ]


def get_distributed_models() -> List[str]:
    """Return models that can run with TP>1 (distributed)."""
    return [
        model_id
        for model_id, config in QWEN3_MODELS.items()
        if any(tp > 1 for tp in config.recommended_tp)
    ]


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Return configuration for a specific model, or None if not found."""
    return QWEN3_MODELS.get(model_id)


def is_valid_model(model_id: str) -> bool:
    """Check if a model ID is in the registry."""
    return model_id in QWEN3_MODELS
