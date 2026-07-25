"""Qwen3-dense model configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3DenseModelConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    seed: int = 0
    num_hidden_layers: int = 1
    intermediate_size: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hf_model_id: str | None = None
    hf_revision: str | None = None
    hf_local_files_only: bool = True
    hf_num_hidden_layers: int | None = None
    nkipy_compiler_args: str = ""
    kv_cache_block_size: int = 32
    attention_backend: str = "NKIBlockSparseFlashAttention"
    tp_degree: int = 1
    tp_rank: int = 0
    tp_world_size: int = 1
