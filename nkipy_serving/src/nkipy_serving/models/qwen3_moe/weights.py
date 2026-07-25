"""Qwen3 MoE weight loading, TP/EP sharding, and KV metadata.

Backend-agnostic: NO nkipy imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

from nkipy_serving.models.qwen3_moe.config import (
    Qwen3MoeModelConfig,
    Qwen3MoeWeights,
)
from nkipy_serving.models.reload_utils import resolve_model_snapshot_path
from nkipy_serving.ops.nn import (
    kv_head_indices_for_rank as _kv_head_indices_for_rank,
)
from nkipy_serving.ops.nn import (
    require_divisible as _require_divisible,
)
from nkipy_serving.ops.nn import (
    validate_tp_runtime as _validate_tp_runtime,
)

# ---------------------------------------------------------------------------
# HF config reader
# ---------------------------------------------------------------------------


def _load_model_config(snapshot_path: Path) -> dict[str, object]:
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Missing config.json under {snapshot_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config.json payload under {snapshot_path}")
    return cfg


def _validate_qwen3_moe_hf_config(model_id: str, cfg: dict[str, object]) -> None:
    if cfg.get("model_type") != "qwen3_moe":
        raise RuntimeError(
            "Unsupported HF checkpoint for qwen3-moe runtime. "
            f"Expected config.model_type='qwen3_moe', got {cfg.get('model_type')!r}. "
            f"model_id={model_id}"
        )
    arch = cfg.get("architectures")
    if arch is not None:
        if not isinstance(arch, list) or not all(isinstance(x, str) for x in arch):
            raise RuntimeError(
                f"Invalid HF config.architectures: expected list[str], got {type(arch)}"
            )
        if "Qwen3MoeForCausalLM" not in arch:
            raise RuntimeError(
                "Unsupported HF checkpoint for qwen3-moe runtime. "
                f"Expected 'Qwen3MoeForCausalLM' in config.architectures, got {arch}."
            )


def _resolve_snapshot_and_config(
    config: Qwen3MoeModelConfig,
) -> tuple[Path, dict[str, object]]:
    snapshot_path = resolve_model_snapshot_path(
        config.hf_model_id,
        revision=config.hf_revision,
        local_files_only=config.hf_local_files_only,
    )
    cfg = _load_model_config(snapshot_path)
    _validate_qwen3_moe_hf_config(config.hf_model_id, cfg)
    return snapshot_path, cfg


class _SafeTensorReader:
    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path
        index_path = snapshot_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise RuntimeError(
                f"Missing model.safetensors.index.json under {snapshot_path}"
            )
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError(f"Invalid safetensors index file: {index_path}")
        self.weight_map = {str(k): str(v) for k, v in weight_map.items()}
        self._handles: dict[str, object] = {}

    def _resolve_file(self, key: str) -> Path:
        filename = self.weight_map.get(key)
        if filename is None:
            raise KeyError(key)
        return self.snapshot_path / filename

    def _get_handle(self, path: Path):
        cache_key = str(path)
        h = self._handles.get(cache_key)
        if h is None:
            # Import ml_dtypes before opening so numpy recognizes bfloat16.
            _ = ml_dtypes.bfloat16
            h = safe_open(str(path), framework="numpy", device="cpu")
            self._handles[cache_key] = h
        return h

    def get_tensor(self, key: str) -> np.ndarray:
        path = self._resolve_file(key)
        h = self._get_handle(path)
        return h.get_tensor(key)

    def get_slice(self, key: str):
        path = self._resolve_file(key)
        h = self._get_handle(path)
        return h.get_slice(key)

    def close(self) -> None:
        self._handles.clear()


# ---------------------------------------------------------------------------
# KV metadata (scheduler + worker init)
# ---------------------------------------------------------------------------


def get_qwen3_moe_kv_metadata(
    config: Qwen3MoeModelConfig,
) -> tuple[int, int, int, np.dtype]:
    """Return (num_kv_heads_per_rank, head_dim, num_layers, dtype)."""
    _validate_tp_runtime(config.tp_degree, config.tp_rank, config.tp_world_size)

    snapshot_path = resolve_model_snapshot_path(
        config.hf_model_id,
        revision=config.hf_revision,
        local_files_only=config.hf_local_files_only,
    )
    cfg = _load_model_config(snapshot_path)
    _validate_qwen3_moe_hf_config(config.hf_model_id, cfg)

    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    num_layers = int(cfg["num_hidden_layers"])
    if config.hf_num_hidden_layers is not None:
        num_layers = min(num_layers, int(config.hf_num_hidden_layers))

    # Support both tp=4 (1 unique KV head per rank) and tp=8 (KV head replication).
    if config.attention_backend == "NKIBlockSparseFlashAttention":
        if num_kv_heads == config.tp_degree:
            num_kv_heads = 1
        elif config.tp_degree % num_kv_heads == 0:
            num_kv_heads = 1
        else:
            raise RuntimeError(
                "Qwen3 MoE NKI attention requires num_key_value_heads to divide "
                "tp_degree or equal tp_degree. "
                f"Got num_key_value_heads={num_kv_heads}, tp_degree={config.tp_degree}."
            )
    else:
        if num_kv_heads % config.tp_degree != 0:
            raise RuntimeError(
                f"num_key_value_heads ({num_kv_heads}) must be divisible by "
                f"tp_degree ({config.tp_degree})"
            )
        num_kv_heads = num_kv_heads // config.tp_degree

    return int(num_kv_heads), int(head_dim), int(num_layers), ml_dtypes.bfloat16


def init_qwen3_moe_weights(config: Qwen3MoeModelConfig) -> Qwen3MoeWeights:
    """Load HF config and derive TP-local metadata (no heavy weight materialization)."""
    _, weights = _load_qwen3_moe_weights(config)
    return weights


def _load_qwen3_moe_weights(
    config: Qwen3MoeModelConfig,
) -> tuple[Path, Qwen3MoeWeights]:
    _validate_tp_runtime(config.tp_degree, config.tp_rank, config.tp_world_size)

    snapshot_path, cfg = _resolve_snapshot_and_config(config)

    hidden_size = int(cfg["hidden_size"])
    head_dim = int(cfg["head_dim"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    n_layers = int(cfg["num_hidden_layers"])
    if config.hf_num_hidden_layers is not None:
        n_layers = min(n_layers, int(config.hf_num_hidden_layers))
    vocab_size = int(cfg["vocab_size"])
    moe_intermediate_size = int(cfg["moe_intermediate_size"])
    num_experts = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000000.0))

    local_num_heads = _require_divisible(
        n_heads, config.tp_degree, "num_attention_heads"
    )
    local_vocab = _require_divisible(vocab_size, config.tp_degree, "vocab_size")
    local_intermediate = _require_divisible(
        moe_intermediate_size, config.tp_degree, "moe_intermediate_size"
    )
    local_num_experts = _require_divisible(
        num_experts, config.ep_degree, "num_experts by ep_degree"
    )

    # KV heads: support both tp=4 and tp=8 via _kv_head_indices_for_rank.
    kv_indices = _kv_head_indices_for_rank(n_kv_heads, config.tp_degree, config.tp_rank)
    local_kv = len(kv_indices)

    return snapshot_path, Qwen3MoeWeights(
        model_id=config.hf_model_id,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        experts_per_token=top_k,
        rms_norm_eps=eps,
        rope_theta=rope_theta,
        dtype=ml_dtypes.bfloat16,
        tp_degree=config.tp_degree,
        tp_rank=config.tp_rank,
        num_heads=local_num_heads,
        num_kv_heads=local_kv,
        local_vocab_size=local_vocab,
        lm_head_vocab_offset=config.tp_rank * local_vocab,
        local_intermediate_size=local_intermediate,
        ep_degree=config.ep_degree,
        ep_rank=config.ep_rank,
        local_num_experts=local_num_experts,
    )
