"""Qwen3 MoE model configuration and weight metadata dataclasses.

Backend-agnostic: NO nkipy imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Model config (runtime-facing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen3MoeModelConfig:
    hf_model_id: str
    hf_revision: str | None = None
    hf_local_files_only: bool = True
    hf_num_hidden_layers: int | None = None
    nkipy_compiler_args: str = ""
    kv_cache_block_size: int = 32
    attention_backend: str = "NKIBlockSparseFlashAttention"
    tp_degree: int = 4
    tp_rank: int = 0
    tp_world_size: int = 4
    ep_degree: int = 1
    ep_rank: int = 0


@dataclass(frozen=True)
class Qwen3MoeWeights:
    """Lightweight metadata for executor + scheduler.

    We intentionally do not keep full CPU copies of model weights here; the
    executor loads and uploads weights directly to device tensors.
    """

    model_id: str
    vocab_size: int
    hidden_size: int
    head_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    moe_intermediate_size: int
    num_experts: int
    experts_per_token: int
    rms_norm_eps: float
    rope_theta: float
    dtype: np.dtype
    # TP metadata (per-rank).
    tp_degree: int
    tp_rank: int
    num_heads: int
    num_kv_heads: int
    local_vocab_size: int
    lm_head_vocab_offset: int
    local_intermediate_size: int
    # EP metadata.
    ep_degree: int
    ep_rank: int
    local_num_experts: int
