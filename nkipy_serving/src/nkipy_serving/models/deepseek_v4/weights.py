"""DeepSeek-V4-Flash HF config reader and rank-filtered weight metadata.

Backend-agnostic: NO nkipy imports.

Scope: read `config.json`, validate, derive per-rank shapes, and return a
`DeepseekV4Weights` metadata object. No tensor materialization.

The HF checkpoint for V4-Flash publishes weights as FP8 E4M3 with UE8M0
[128,128] scales already, so no MXFP4 conversion step is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ml_dtypes

from nkipy_serving.models.deepseek_v4.config import (
    DeepseekV4ModelConfig,
    DeepseekV4Weights,
)
from nkipy_serving.models.deepseek_v4.rank_layout import (
    coord_for_rank,
    local_expert_ids,
    validate_v4_rank_layout,
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


def _load_model_config(snapshot_path: Path) -> dict[str, Any]:
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Missing config.json under {snapshot_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid config.json payload under {snapshot_path}")
    return cfg


def _validate_deepseek_v4_hf_config(model_id: str, cfg: dict[str, Any]) -> None:
    if cfg.get("model_type") != "deepseek_v4":
        raise RuntimeError(
            "Unsupported HF checkpoint for deepseek-v4 runtime. "
            f"Expected config.model_type='deepseek_v4', got {cfg.get('model_type')!r}. "
            f"model_id={model_id}"
        )
    arch = cfg.get("architectures")
    if arch is not None:
        if not isinstance(arch, list) or not all(isinstance(x, str) for x in arch):
            raise RuntimeError(
                f"Invalid HF config.architectures: expected list[str], got {type(arch)}"
            )
        if "DeepseekV4ForCausalLM" not in arch:
            raise RuntimeError(
                "Unsupported HF checkpoint for deepseek-v4 runtime. "
                f"Expected 'DeepseekV4ForCausalLM' in config.architectures, got {arch}."
            )

    quant = cfg.get("quantization_config")
    if not isinstance(quant, dict):
        raise RuntimeError(
            "HF checkpoint is missing quantization_config; V4 runtime expects "
            "FP8 E4M3 + UE8M0 block scales."
        )
    if quant.get("quant_method") != "fp8" or quant.get("fmt") != "e4m3":
        raise RuntimeError(
            "Unsupported quantization for V4 runtime. "
            f"Expected fp8/e4m3, got {quant.get('quant_method')!r}/{quant.get('fmt')!r}."
        )
    if quant.get("scale_fmt") != "ue8m0":
        raise RuntimeError(
            f"Unsupported scale format: {quant.get('scale_fmt')!r}. Expected 'ue8m0'."
        )


def _compress_ratios(
    cfg: dict[str, Any],
    num_hidden_layers: int,
) -> tuple[int, ...]:
    raw = cfg.get("compress_ratios")
    if not isinstance(raw, list):
        raise RuntimeError("HF config missing compress_ratios array")
    # HF publishes `num_hidden_layers + num_nextn_predict_layers` entries; the
    # trailing entries describe MTP. Truncate to main-layer count here.
    ratios = tuple(int(x) for x in raw[:num_hidden_layers])
    if len(ratios) != num_hidden_layers:
        raise RuntimeError(
            f"compress_ratios has {len(raw)} entries, need at least {num_hidden_layers}"
        )
    for layer_idx, ratio in enumerate(ratios):
        if ratio not in (0, 4, 128):
            raise RuntimeError(
                f"Unsupported compress_ratio {ratio} at layer {layer_idx}; "
                "V4 expects 0 (full), 4 (c4a), or 128 (c128a)."
            )
    return ratios


def get_deepseek_v4_kv_metadata(
    config: DeepseekV4ModelConfig,
) -> tuple[int, int, int, Any]:
    """Return (num_kv_heads_per_rank, head_dim, num_layers, dtype).

    Used for scheduler bookkeeping. The real V4 KV layout is heterogeneous per
    layer (full / c4a / c128a / sliding). The `num_kv_heads_per_rank` here is
    the main attention head count; indexer KV is tracked separately.
    """
    _validate_tp_runtime(config.tp_degree, config.tp_rank, config.tp_world_size)

    snapshot_path = resolve_model_snapshot_path(
        config.hf_model_id,
        revision=config.hf_revision,
        local_files_only=config.hf_local_files_only,
    )
    cfg = _load_model_config(snapshot_path)
    _validate_deepseek_v4_hf_config(config.hf_model_id, cfg)

    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    num_layers = int(cfg["num_hidden_layers"])
    if config.hf_num_hidden_layers is not None:
        num_layers = min(num_layers, int(config.hf_num_hidden_layers))

    kv_indices = _kv_head_indices_for_rank(
        num_kv_heads, config.tp_degree, config.tp_rank
    )
    local_kv = len(kv_indices)
    return int(local_kv), int(head_dim), int(num_layers), ml_dtypes.bfloat16


def init_deepseek_v4_weights(config: DeepseekV4ModelConfig) -> DeepseekV4Weights:
    """Load HF config and derive per-rank metadata (no tensor materialization)."""
    _, weights = _load_deepseek_v4_weights(config)
    return weights


def _load_deepseek_v4_weights(
    config: DeepseekV4ModelConfig,
) -> tuple[Path, DeepseekV4Weights]:
    _validate_tp_runtime(config.tp_degree, config.tp_rank, config.tp_world_size)

    # Derive world-size from logical axes; validate the V4 layout.
    world_size = config.tp_degree * config.ep_degree * config.replica_degree
    validate_v4_rank_layout(
        tp_degree=config.tp_degree,
        ep_degree=config.ep_degree,
        replica_degree=config.replica_degree,
        world_size=world_size,
    )
    global_rank = (
        config.request_lane_rank * config.tp_degree + config.tp_rank
    )  # caller-supplied attention-lane × TP column
    coord = coord_for_rank(
        rank=global_rank,
        tp_degree=config.tp_degree,
        ep_degree=config.ep_degree,
        replica_degree=config.replica_degree,
    )

    snapshot_path = resolve_model_snapshot_path(
        config.hf_model_id,
        revision=config.hf_revision,
        local_files_only=config.hf_local_files_only,
    )
    cfg = _load_model_config(snapshot_path)
    _validate_deepseek_v4_hf_config(config.hf_model_id, cfg)

    hidden_size = int(cfg["hidden_size"])
    head_dim = int(cfg["head_dim"])
    qk_rope_head_dim = int(cfg["qk_rope_head_dim"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    n_layers = int(cfg["num_hidden_layers"])
    if config.hf_num_hidden_layers is not None:
        n_layers = min(n_layers, int(config.hf_num_hidden_layers))
    vocab_size = int(cfg["vocab_size"])
    moe_intermediate_size = int(cfg["moe_intermediate_size"])
    num_routed_experts = int(cfg["n_routed_experts"])
    num_shared_experts = int(cfg["n_shared_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    num_hash_layers = int(cfg["num_hash_layers"])
    routed_scaling_factor = float(cfg["routed_scaling_factor"])
    swiglu_limit = float(cfg["swiglu_limit"])
    scoring_func = str(cfg["scoring_func"])
    topk_method = str(cfg["topk_method"])
    q_lora_rank = int(cfg["q_lora_rank"])
    o_lora_rank = int(cfg["o_lora_rank"])
    o_groups = int(cfg["o_groups"])
    hc_mult = int(cfg["hc_mult"])
    hc_sinkhorn_iters = int(cfg["hc_sinkhorn_iters"])
    hc_eps = float(cfg["hc_eps"])
    index_n_heads = int(cfg["index_n_heads"])
    index_head_dim = int(cfg["index_head_dim"])
    index_topk = int(cfg["index_topk"])
    sliding_window = int(cfg["sliding_window"])
    compress_rope_theta = float(cfg["compress_rope_theta"])
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    max_position = int(cfg["max_position_embeddings"])
    rope_scaling = cfg.get("rope_scaling", {}) or {}
    rope_scaling_factor = int(rope_scaling.get("factor", 1))
    rope_original = int(
        rope_scaling.get("original_max_position_embeddings", max_position)
    )
    rope_beta_fast = int(rope_scaling.get("beta_fast", 32))
    rope_beta_slow = int(rope_scaling.get("beta_slow", 1))
    num_nextn = int(cfg.get("num_nextn_predict_layers", 0))

    compress_ratios = _compress_ratios(cfg, n_layers)

    local_num_heads = _require_divisible(
        n_heads, config.tp_degree, "num_attention_heads"
    )
    local_vocab = _require_divisible(vocab_size, config.tp_degree, "vocab_size")
    local_intermediate = _require_divisible(
        moe_intermediate_size, config.tp_degree, "moe_intermediate_size"
    )
    local_num_experts = _require_divisible(
        num_routed_experts, config.ep_degree, "n_routed_experts by ep_degree"
    )
    kv_indices = _kv_head_indices_for_rank(n_kv_heads, config.tp_degree, config.tp_rank)
    local_kv = len(kv_indices)
    expert_ids = local_expert_ids(
        num_routed_experts,
        config.ep_degree,
        coord.row_in_replica,
    )

    return snapshot_path, DeepseekV4Weights(
        model_id=config.hf_model_id,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        moe_intermediate_size=moe_intermediate_size,
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        experts_per_token=top_k,
        num_hash_layers=num_hash_layers,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
        scoring_func=scoring_func,
        topk_method=topk_method,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        hc_eps=hc_eps,
        compress_ratios=compress_ratios,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        compress_rope_theta=compress_rope_theta,
        rope_theta=rope_theta,
        rope_scaling_factor=rope_scaling_factor,
        rope_original_max_position=rope_original,
        rope_beta_fast=rope_beta_fast,
        rope_beta_slow=rope_beta_slow,
        max_position_embeddings=max_position,
        num_nextn_predict_layers=num_nextn,
        rms_norm_eps=rms_eps,
        dtype=ml_dtypes.bfloat16,
        tp_degree=config.tp_degree,
        tp_rank=config.tp_rank,
        ep_degree=config.ep_degree,
        ep_rank=config.ep_rank,
        replica_degree=config.replica_degree,
        replica_rank=coord.replica,
        attention_dp_degree=config.attention_dp_degree,
        attention_lane=coord.attn_lane,
        dsv4_prepared_weight_dir=config.dsv4_prepared_weight_dir,
        dsv4_prepared_weight_local_dir=config.dsv4_prepared_weight_local_dir,
        local_num_attention_heads=local_num_heads,
        local_num_kv_heads=local_kv,
        local_vocab_size=local_vocab,
        lm_head_vocab_offset=config.tp_rank * local_vocab,
        local_moe_intermediate_size=local_intermediate,
        local_num_routed_experts=local_num_experts,
        local_expert_ids=expert_ids,
        quant_fmt="e4m3",
        quant_scale_fmt="ue8m0",
        quant_weight_block_size=(128, 128),
    )
