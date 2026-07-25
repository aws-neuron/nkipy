"""DeepSeek-V4-Flash model configuration and weight metadata dataclasses.

Backend-agnostic: NO nkipy imports.

The current runtime contract is documented in ``docs/models/deepseek_v4.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeepseekV4ModelConfig:
    hf_model_id: str
    hf_revision: str | None = None
    hf_local_files_only: bool = True
    hf_num_hidden_layers: int | None = None
    nkipy_compiler_args: str = ""
    kv_cache_block_size: int = 32
    attention_backend: str = "Dsv4SparseAttention"
    dsv4_prepared_weight_dir: str | None = None
    dsv4_prepared_weight_local_dir: str | None = None

    # Dense / attention TP within a row.
    tp_degree: int = 8
    tp_rank: int = 0
    tp_world_size: int = 8

    # MoE expert parallelism.
    ep_degree: int = 8
    ep_rank: int = 0

    # Logical axes for V4 DP-attention.
    # attention_dp_degree = request lanes (one per TP row).
    # moe_tp_degree = 1 on V4 (experts not TP-of-experts sharded).
    # replica_degree = total_workers / (tp_degree * ep_degree); experts are
    # replicated across replicas, attention lanes split across replicas too.
    attention_dp_degree: int = 16
    attention_tp_degree: int = 1
    moe_tp_degree: int = 1
    replica_degree: int = 2

    # Per-rank request-lane identity (populated by the coordinator at launch).
    request_lane_rank: int = 0
    request_lane_world_size: int = 16

    # Product runtime is target-only until device-resident MTP state is ready.
    dsv4_disable_mtp: bool = True


@dataclass(frozen=True)
class DeepseekV4Weights:
    """Lightweight metadata for executor + scheduler.

    No CPU materialization of weights here. The executor reads bytes directly
    into device buffers and preserves FP8 E4M3 + UE8M0 scale bytes.
    """

    model_id: str
    vocab_size: int
    hidden_size: int
    head_dim: int
    qk_rope_head_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int

    # MoE.
    moe_intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    experts_per_token: int
    num_hash_layers: int
    routed_scaling_factor: float
    swiglu_limit: float
    scoring_func: str
    topk_method: str

    # mHC.
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    # Attention compression per layer (compress_ratio, cache_kind tag).
    # Length = num_hidden_layers. 0 = full attention KV; 4 = c4a; 128 = c128a.
    compress_ratios: tuple[int, ...]

    # Sparse indexer (c4a layers only; ignored on other layers).
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    sliding_window: int
    compress_rope_theta: float

    # RoPE.
    rope_theta: float
    rope_scaling_factor: int
    rope_original_max_position: int
    rope_beta_fast: int
    rope_beta_slow: int
    max_position_embeddings: int

    # MTP.
    num_nextn_predict_layers: int

    rms_norm_eps: float
    dtype: Any

    # TP/EP/replica metadata (per-rank).
    tp_degree: int
    tp_rank: int
    ep_degree: int
    ep_rank: int
    replica_degree: int
    replica_rank: int
    attention_dp_degree: int
    attention_lane: int

    # Per-rank derived shapes.
    local_num_attention_heads: int
    local_num_kv_heads: int
    local_vocab_size: int
    lm_head_vocab_offset: int
    local_moe_intermediate_size: int
    local_num_routed_experts: int
    dsv4_prepared_weight_dir: str | None = None
    dsv4_prepared_weight_local_dir: str | None = None
    local_expert_ids: tuple[int, ...] = field(default_factory=tuple)

    # Quantization (FP8 E4M3 + UE8M0 block [128,128]).
    # On Trn2, `nisa.nc_matmul` accepts fp8 inputs natively (POC 2).
    quant_fmt: str = "e4m3"
    quant_scale_fmt: str = "ue8m0"
    quant_weight_block_size: tuple[int, int] = (128, 128)

    @property
    def num_kv_heads(self) -> int:
        """Generic ModelRunner alias used by non-DSV4 model metadata."""
        return int(self.num_key_value_heads)
