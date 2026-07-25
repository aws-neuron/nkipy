"""Qwen3-dense weight dataclasses, TP sharding, random init, and HF loading.

This file is backend-agnostic: NO nkipy imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

from nkipy_serving.models.qwen3_dense.config import Qwen3DenseModelConfig
from nkipy_serving.models.reload_utils import resolve_model_snapshot_path
from nkipy_serving.ops.nn import (
    kv_head_indices_for_rank as _kv_head_indices_for_rank,
)
from nkipy_serving.ops.nn import (
    require_divisible as _require_divisible,
)
from nkipy_serving.ops.nn import (
    select_head_columns as _select_head_columns,
)
from nkipy_serving.ops.nn import (
    select_head_rows as _select_head_rows,
)
from nkipy_serving.ops.nn import (
    validate_tp_runtime as _validate_tp_runtime,
)

# ---------------------------------------------------------------------------
# Weight dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen3DenseLayerWeights:
    input_norm: np.ndarray
    post_attn_norm: np.ndarray
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


@dataclass(frozen=True)
class Qwen3DenseWeights:
    embeddings: np.ndarray
    layers: tuple[Qwen3DenseLayerWeights, ...]
    lm_head: np.ndarray
    final_norm: np.ndarray
    num_heads: int
    num_kv_heads: int
    global_num_heads: int
    global_num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    global_intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    lm_head_vocab_offset: int
    local_vocab_size: int
    tp_degree: int
    tp_rank: int
    rope_theta: float
    rms_norm_eps: float


def _shard_weights_for_tp(
    weights: Qwen3DenseWeights,
    tp_degree: int,
    tp_rank: int,
    tp_world_size: int,
) -> Qwen3DenseWeights:
    _validate_tp_runtime(
        tp_degree=tp_degree, tp_rank=tp_rank, tp_world_size=tp_world_size
    )
    if tp_degree == 1:
        return Qwen3DenseWeights(
            embeddings=weights.embeddings,
            layers=weights.layers,
            lm_head=weights.lm_head,
            final_norm=weights.final_norm,
            num_heads=weights.num_heads,
            num_kv_heads=weights.num_kv_heads,
            global_num_heads=weights.num_heads,
            global_num_kv_heads=weights.num_kv_heads,
            head_dim=weights.head_dim,
            hidden_size=weights.hidden_size,
            intermediate_size=weights.intermediate_size,
            global_intermediate_size=weights.intermediate_size,
            num_hidden_layers=weights.num_hidden_layers,
            vocab_size=weights.vocab_size,
            lm_head_vocab_offset=0,
            local_vocab_size=weights.vocab_size,
            tp_degree=1,
            tp_rank=0,
            rope_theta=weights.rope_theta,
            rms_norm_eps=weights.rms_norm_eps,
        )

    local_num_heads = _require_divisible(
        weights.num_heads, tp_degree, field_name="num_heads"
    )
    local_intermediate_size = _require_divisible(
        weights.intermediate_size, tp_degree, field_name="intermediate_size"
    )
    local_vocab_size = _require_divisible(
        weights.vocab_size, tp_degree, field_name="vocab_size"
    )
    local_num_kv_head_indices = _kv_head_indices_for_rank(
        global_num_kv_heads=weights.num_kv_heads,
        tp_degree=tp_degree,
        tp_rank=tp_rank,
    )
    local_num_kv_heads = len(local_num_kv_head_indices)
    if local_num_heads % local_num_kv_heads != 0:
        raise RuntimeError(
            "Invalid local GQA ratio after TP sharding. "
            f"local_num_heads={local_num_heads}, local_num_kv_heads={local_num_kv_heads}"
        )

    q_head_start = tp_rank * local_num_heads
    q_head_indices = tuple(range(q_head_start, q_head_start + local_num_heads))
    inter_start = tp_rank * local_intermediate_size
    inter_end = inter_start + local_intermediate_size
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size

    sharded_layers: list[Qwen3DenseLayerWeights] = []
    for layer in weights.layers:
        sharded_layers.append(
            Qwen3DenseLayerWeights(
                input_norm=layer.input_norm,
                post_attn_norm=layer.post_attn_norm,
                w_q=_select_head_columns(layer.w_q, q_head_indices, weights.head_dim),
                w_k=_select_head_columns(
                    layer.w_k, local_num_kv_head_indices, weights.head_dim
                ),
                w_v=_select_head_columns(
                    layer.w_v, local_num_kv_head_indices, weights.head_dim
                ),
                w_o=_select_head_rows(layer.w_o, q_head_indices, weights.head_dim),
                q_norm=layer.q_norm,
                k_norm=layer.k_norm,
                w_gate=np.asarray(
                    layer.w_gate[:, inter_start:inter_end], dtype=layer.w_gate.dtype
                ),
                w_up=np.asarray(
                    layer.w_up[:, inter_start:inter_end], dtype=layer.w_up.dtype
                ),
                w_down=np.asarray(
                    layer.w_down[inter_start:inter_end, :], dtype=layer.w_down.dtype
                ),
            )
        )

    sharded_lm_head = np.asarray(
        weights.lm_head[vocab_start:vocab_end, :], dtype=weights.lm_head.dtype
    )
    return Qwen3DenseWeights(
        embeddings=weights.embeddings,
        layers=tuple(sharded_layers),
        lm_head=sharded_lm_head,
        final_norm=weights.final_norm,
        num_heads=local_num_heads,
        num_kv_heads=local_num_kv_heads,
        global_num_heads=weights.num_heads,
        global_num_kv_heads=weights.num_kv_heads,
        head_dim=weights.head_dim,
        hidden_size=weights.hidden_size,
        intermediate_size=local_intermediate_size,
        global_intermediate_size=weights.intermediate_size,
        num_hidden_layers=weights.num_hidden_layers,
        vocab_size=weights.vocab_size,
        lm_head_vocab_offset=vocab_start,
        local_vocab_size=local_vocab_size,
        tp_degree=tp_degree,
        tp_rank=tp_rank,
        rope_theta=weights.rope_theta,
        rms_norm_eps=weights.rms_norm_eps,
    )


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------


def _randn(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float
) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _init_random_weights(config: Qwen3DenseModelConfig) -> Qwen3DenseWeights:
    rng = np.random.default_rng(config.seed)
    hidden = config.hidden_size
    vocab = config.vocab_size
    weight_dtype = ml_dtypes.bfloat16
    num_layers = config.num_hidden_layers
    if num_layers <= 0:
        raise RuntimeError(f"num_hidden_layers must be > 0, got {num_layers}")
    intermediate = (
        config.intermediate_size if config.intermediate_size is not None else hidden * 4
    )
    if intermediate <= 0:
        raise RuntimeError(f"intermediate_size must be > 0, got {intermediate}")

    layers: list[Qwen3DenseLayerWeights] = []
    for _ in range(num_layers):
        layers.append(
            Qwen3DenseLayerWeights(
                input_norm=np.ones((hidden,), dtype=weight_dtype),
                post_attn_norm=np.ones((hidden,), dtype=weight_dtype),
                w_q=_randn(rng, (hidden, hidden), scale=0.02).astype(weight_dtype),
                w_k=_randn(rng, (hidden, hidden), scale=0.02).astype(weight_dtype),
                w_v=_randn(rng, (hidden, hidden), scale=0.02).astype(weight_dtype),
                w_o=_randn(rng, (hidden, hidden), scale=0.02).astype(weight_dtype),
                q_norm=np.ones((hidden,), dtype=weight_dtype),
                k_norm=np.ones((hidden,), dtype=weight_dtype),
                w_gate=_randn(rng, (hidden, intermediate), scale=0.02).astype(
                    weight_dtype
                ),
                w_up=_randn(rng, (hidden, intermediate), scale=0.02).astype(
                    weight_dtype
                ),
                w_down=_randn(rng, (intermediate, hidden), scale=0.02).astype(
                    weight_dtype
                ),
            )
        )

    return Qwen3DenseWeights(
        embeddings=_randn(rng, (vocab, hidden), scale=0.02).astype(weight_dtype),
        layers=tuple(layers),
        lm_head=_randn(rng, (vocab, hidden), scale=0.02).astype(weight_dtype),
        final_norm=np.ones((hidden,), dtype=weight_dtype),
        num_heads=1,
        num_kv_heads=1,
        global_num_heads=1,
        global_num_kv_heads=1,
        head_dim=hidden,
        hidden_size=hidden,
        intermediate_size=intermediate,
        global_intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        vocab_size=vocab,
        lm_head_vocab_offset=0,
        local_vocab_size=vocab,
        tp_degree=1,
        tp_rank=0,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
    )


# ---------------------------------------------------------------------------
# HF weight loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HFQwen3LayerWeights:
    input_layernorm: np.ndarray
    post_attention_layernorm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    gate_proj: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray


@dataclass(frozen=True)
class _HFQwen3DenseWeights:
    model_id: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    embeddings: np.ndarray
    layers: tuple[_HFQwen3LayerWeights, ...]
    final_norm: np.ndarray
    lm_head: np.ndarray


def _to_bfloat16(array: np.ndarray) -> np.ndarray:
    if str(array.dtype) == "bfloat16":
        return array
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=ml_dtypes.bfloat16)
    raise RuntimeError(
        f"Unsupported tensor dtype for Qwen3 weight: {array.dtype}. "
        "Expected floating-point tensor."
    )


class _SafeTensorReader:
    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path
        index_path = snapshot_path / "model.safetensors.index.json"
        self.weight_map: dict[str, str] | None = None
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict):
                raise RuntimeError(f"Invalid safetensors index file: {index_path}")
            self.weight_map = {str(k): str(v) for k, v in weight_map.items()}
        else:
            single = snapshot_path / "model.safetensors"
            if not single.exists():
                raise RuntimeError(
                    f"Cannot find model.safetensors(.index.json) under {snapshot_path}"
                )
        self._handles: dict[str, object] = {}

    def _resolve_file(self, key: str) -> Path:
        if self.weight_map is None:
            return self.snapshot_path / "model.safetensors"
        filename = self.weight_map.get(key)
        if filename is None:
            raise KeyError(key)
        return self.snapshot_path / filename

    def _get_handle(self, path: Path):
        cache_key = str(path)
        handle = self._handles.get(cache_key)
        if handle is None:
            handle = safe_open(str(path), framework="np", device="cpu")
            self._handles[cache_key] = handle
        return handle

    def has_key(self, key: str) -> bool:
        if self.weight_map is None:
            handle = self._get_handle(self.snapshot_path / "model.safetensors")
            return key in list(handle.keys())
        return key in self.weight_map

    def load_tensor(self, key: str) -> np.ndarray:
        _ = ml_dtypes.bfloat16
        path = self._resolve_file(key)
        handle = self._get_handle(path)
        return handle.get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


def _load_model_config(snapshot_path: Path) -> dict[str, object]:
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Missing config.json under {snapshot_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid config.json payload under {snapshot_path}")
    return config


def _validate_qwen3_dense_hf_config(model_id: str, config: dict[str, object]) -> None:
    """Fail fast if the HF checkpoint does not look like a supported Qwen3-dense model."""
    model_type = config.get("model_type")
    if model_type != "qwen3":
        raise RuntimeError(
            "Unsupported HF checkpoint for Qwen3 dense runtime. "
            f"Expected config.model_type='qwen3', got {model_type!r}. model_id={model_id}"
        )

    architectures = config.get("architectures")
    if architectures is not None:
        if not isinstance(architectures, list) or not all(
            isinstance(x, str) for x in architectures
        ):
            raise RuntimeError(
                "Invalid HF config field 'architectures'. "
                f"Expected list[str], got {type(architectures)}. model_id={model_id}"
            )
        if "Qwen3ForCausalLM" not in architectures:
            raise RuntimeError(
                "Unsupported HF checkpoint for Qwen3 dense runtime. "
                "Expected 'Qwen3ForCausalLM' in config.architectures, "
                f"got {architectures}. model_id={model_id}"
            )

    unsupported_keys = [
        k for k in config.keys() if "moe" in k.lower() or "expert" in k.lower()
    ]
    if unsupported_keys:
        raise RuntimeError(
            "Unsupported HF checkpoint for Qwen3 dense runtime (MoE-looking config). "
            f"Unexpected keys={unsupported_keys}. model_id={model_id}"
        )


def _load_hf_weights(
    model_id: str,
    revision: str | None = None,
    local_files_only: bool = True,
    max_layers: int | None = None,
) -> _HFQwen3DenseWeights:
    model_path = Path(model_id).expanduser()
    if not model_path.exists() and not model_id.startswith("Qwen/Qwen3-"):
        raise RuntimeError(f"Unsupported model for Qwen3 loader: {model_id}")

    snapshot_path = resolve_model_snapshot_path(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    config = _load_model_config(snapshot_path)
    _validate_qwen3_dense_hf_config(model_id, config)

    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab_size = int(config["vocab_size"])
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
    rope_theta = float(config.get("rope_theta", 1000000.0))
    hidden_act = str(config.get("hidden_act", "silu"))

    if max_layers is None:
        target_layers = num_hidden_layers
    else:
        if max_layers <= 0:
            raise RuntimeError(f"max_layers must be > 0, got {max_layers}")
        if max_layers > num_hidden_layers:
            raise RuntimeError(
                f"max_layers {max_layers} exceeds model layer count {num_hidden_layers}"
            )
        target_layers = int(max_layers)

    reader = _SafeTensorReader(snapshot_path)
    try:
        embeddings = _to_bfloat16(reader.load_tensor("model.embed_tokens.weight"))
        layers: list[_HFQwen3LayerWeights] = []
        for layer_idx in range(target_layers):
            prefix = f"model.layers.{layer_idx}"
            q_norm_key = f"{prefix}.self_attn.q_norm.weight"
            k_norm_key = f"{prefix}.self_attn.k_norm.weight"
            layer = _HFQwen3LayerWeights(
                input_layernorm=np.asarray(
                    reader.load_tensor(f"{prefix}.input_layernorm.weight"),
                    dtype=ml_dtypes.bfloat16,
                ),
                post_attention_layernorm=np.asarray(
                    reader.load_tensor(f"{prefix}.post_attention_layernorm.weight"),
                    dtype=ml_dtypes.bfloat16,
                ),
                q_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.self_attn.q_proj.weight")
                ),
                k_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.self_attn.k_proj.weight")
                ),
                v_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.self_attn.v_proj.weight")
                ),
                o_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.self_attn.o_proj.weight")
                ),
                q_norm=(
                    np.asarray(reader.load_tensor(q_norm_key), dtype=ml_dtypes.bfloat16)
                    if reader.has_key(q_norm_key)
                    else np.ones((head_dim,), dtype=ml_dtypes.bfloat16)
                ),
                k_norm=(
                    np.asarray(reader.load_tensor(k_norm_key), dtype=ml_dtypes.bfloat16)
                    if reader.has_key(k_norm_key)
                    else np.ones((head_dim,), dtype=ml_dtypes.bfloat16)
                ),
                gate_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.mlp.gate_proj.weight")
                ),
                up_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.mlp.up_proj.weight")
                ),
                down_proj=_to_bfloat16(
                    reader.load_tensor(f"{prefix}.mlp.down_proj.weight")
                ),
            )
            layers.append(layer)

        final_norm = np.asarray(
            reader.load_tensor("model.norm.weight"), dtype=ml_dtypes.bfloat16
        )
        lm_head_key = "lm_head.weight"
        if reader.has_key(lm_head_key):
            lm_head = _to_bfloat16(reader.load_tensor(lm_head_key))
        elif tie_word_embeddings:
            lm_head = embeddings
        else:
            raise RuntimeError(
                f"lm_head.weight missing for untied model checkpoint: {model_id}"
            )
    finally:
        reader.close()

    if embeddings.shape != (vocab_size, hidden_size):
        raise RuntimeError(
            f"Embedding shape mismatch: {embeddings.shape} vs {(vocab_size, hidden_size)}"
        )
    if final_norm.shape != (hidden_size,):
        raise RuntimeError(
            f"Final norm shape mismatch: {final_norm.shape} vs {(hidden_size,)}"
        )
    if lm_head.shape != (vocab_size, hidden_size):
        raise RuntimeError(
            f"lm_head shape mismatch: {lm_head.shape} vs {(vocab_size, hidden_size)}"
        )
    if hidden_act != "silu":
        raise RuntimeError(
            f"Unsupported hidden_act for runtime: {hidden_act}. Expected: silu"
        )

    return _HFQwen3DenseWeights(
        model_id=model_id,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=target_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        hidden_act=hidden_act,
        embeddings=embeddings,
        layers=tuple(layers),
        final_norm=final_norm,
        lm_head=lm_head,
    )


def _init_hf_weights(config: Qwen3DenseModelConfig) -> Qwen3DenseWeights:
    if not config.hf_model_id:
        raise RuntimeError("hf_model_id is required to load HF Qwen3 weights")
    hf = _load_hf_weights(
        model_id=config.hf_model_id,
        revision=config.hf_revision,
        local_files_only=config.hf_local_files_only,
        max_layers=config.hf_num_hidden_layers,
    )
    layers = tuple(
        Qwen3DenseLayerWeights(
            input_norm=np.asarray(layer.input_layernorm, dtype=ml_dtypes.bfloat16),
            post_attn_norm=np.asarray(
                layer.post_attention_layernorm, dtype=ml_dtypes.bfloat16
            ),
            w_q=np.asarray(layer.q_proj.T, dtype=ml_dtypes.bfloat16),
            w_k=np.asarray(layer.k_proj.T, dtype=ml_dtypes.bfloat16),
            w_v=np.asarray(layer.v_proj.T, dtype=ml_dtypes.bfloat16),
            w_o=np.asarray(layer.o_proj.T, dtype=ml_dtypes.bfloat16),
            q_norm=np.asarray(layer.q_norm, dtype=ml_dtypes.bfloat16),
            k_norm=np.asarray(layer.k_norm, dtype=ml_dtypes.bfloat16),
            w_gate=np.asarray(layer.gate_proj.T, dtype=ml_dtypes.bfloat16),
            w_up=np.asarray(layer.up_proj.T, dtype=ml_dtypes.bfloat16),
            w_down=np.asarray(layer.down_proj.T, dtype=ml_dtypes.bfloat16),
        )
        for layer in hf.layers
    )
    return Qwen3DenseWeights(
        embeddings=np.asarray(hf.embeddings, dtype=ml_dtypes.bfloat16),
        layers=layers,
        lm_head=np.asarray(hf.lm_head, dtype=ml_dtypes.bfloat16),
        final_norm=np.asarray(hf.final_norm, dtype=ml_dtypes.bfloat16),
        num_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        global_num_heads=hf.num_attention_heads,
        global_num_kv_heads=hf.num_key_value_heads,
        head_dim=hf.head_dim,
        hidden_size=hf.hidden_size,
        intermediate_size=hf.intermediate_size,
        global_intermediate_size=hf.intermediate_size,
        num_hidden_layers=hf.num_hidden_layers,
        vocab_size=hf.vocab_size,
        lm_head_vocab_offset=0,
        local_vocab_size=hf.vocab_size,
        tp_degree=1,
        tp_rank=0,
        rope_theta=hf.rope_theta,
        rms_norm_eps=hf.rms_norm_eps,
    )


def get_qwen3_dense_kv_metadata(
    config: Qwen3DenseModelConfig,
) -> tuple[int, int, int, np.dtype]:
    """Return KV cache metadata without loading model weights.

    Returns (num_kv_heads, head_dim, num_layers, dtype) for the scheduler
    to allocate its KV pool stub.  For HF models, reads config.json from
    the (already-cached) snapshot.  For random-weight configs, derives
    the values from hidden_size and num_hidden_layers.

    For TP, we validate and apply sharding to num_kv_heads. The NKI attention
    backend currently requires 1 KV head per rank (i.e. num_key_value_heads
    equals tp_degree), so we fail fast if that is not satisfied.
    """
    dtype = ml_dtypes.bfloat16

    if config.hf_model_id:
        snapshot_path = resolve_model_snapshot_path(
            config.hf_model_id,
            revision=config.hf_revision,
            local_files_only=config.hf_local_files_only,
        )
        hf_config = _load_model_config(snapshot_path)
        _validate_qwen3_dense_hf_config(config.hf_model_id, hf_config)
        num_kv_heads = int(hf_config["num_key_value_heads"])
        head_dim = int(hf_config["head_dim"])
        num_layers = int(hf_config["num_hidden_layers"])
        if config.hf_num_hidden_layers is not None:
            num_layers = min(num_layers, config.hf_num_hidden_layers)
    else:
        # Random-weight config: single head, head_dim = hidden_size
        num_kv_heads = 1
        head_dim = config.hidden_size
        num_layers = config.num_hidden_layers

    # Apply TP sharding (num_kv_heads returned is per-rank).
    if config.tp_degree > 1:
        if config.attention_backend == "NKIBlockSparseFlashAttention":
            if num_kv_heads != config.tp_degree:
                raise RuntimeError(
                    "NKIBlockSparseFlashAttention requires "
                    "num_key_value_heads == tp_degree (1 KV head per rank). "
                    f"Got num_key_value_heads={num_kv_heads}, tp_degree={config.tp_degree}."
                )
            num_kv_heads = 1
        else:
            if num_kv_heads % config.tp_degree != 0:
                raise RuntimeError(
                    f"num_kv_heads ({num_kv_heads}) must be divisible by "
                    f"tp_degree ({config.tp_degree})"
                )
            num_kv_heads = num_kv_heads // config.tp_degree

    return num_kv_heads, head_dim, num_layers, dtype


def init_qwen3_dense_weights(config: Qwen3DenseModelConfig) -> Qwen3DenseWeights:
    if config.kv_cache_block_size <= 0:
        raise RuntimeError(
            f"kv_cache_block_size must be > 0, got {config.kv_cache_block_size}"
        )
    if config.tp_degree > 1 and not config.hf_model_id:
        raise RuntimeError(
            "tp_degree > 1 requires hf_model_id-backed Qwen3 weights in this runtime"
        )
    if config.hf_model_id:
        base = _init_hf_weights(config)
    else:
        base = _init_random_weights(config)
    return _shard_weights_for_tp(
        base,
        tp_degree=config.tp_degree,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
    )
