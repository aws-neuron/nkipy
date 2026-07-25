"""Shared resource dataclasses for the DSV4 NEFF runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.variants import (
    GraphVariantName,
    QkvVariantName,
)

KernelCache = dict[tuple[Any, ...], Any]


@dataclass(frozen=True)
class Dsv4ProductBucket:
    """Product executor handles selected at token-bucket granularity."""

    token_bucket: int
    max_requests: int
    last_token_indices_host: np.ndarray
    last_token_indices_dev: Any
    owner_ids_host: np.ndarray
    owner_ids_dev: Any
    input_ids_host: np.ndarray
    input_ids_dev: Any
    vocab_range_host: np.ndarray
    vocab_range_dev: Any
    freq_positions_host: np.ndarray
    freq_positions_dev: Any
    attention_dp_lane_start_host: np.ndarray
    attention_dp_lane_start_dev: Any
    attention_dp_token_start_host: np.ndarray
    attention_dp_token_start_dev: Any
    attention_dp_token_count_host: np.ndarray
    attention_dp_token_count_dev: Any
    kernel_caches: dict[GraphVariantName, KernelCache]
    head_hidden_output: Any
    head_top1_values: Any
    head_top1_indices: Any
    attention_outputs: tuple[Any, ...]
    moe_prefill_outputs: tuple[Any, ...]
    moe_prefill_ep_outputs: tuple[Any, ...]
    moe_prefill_tp_outputs: tuple[Any, ...]
    moe_decode_outputs: tuple[Any, ...]
    moe_decode_ep_outputs: tuple[Any, ...]
    moe_decode_tp_outputs: tuple[Any, ...]
    scratch_outputs: dict[tuple[str, tuple[tuple[int, ...], str]], Any]


@dataclass(frozen=True)
class _TensorSpec:
    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class _AttentionOutCollectiveSpec:
    rows: int
    bsz: int
    seqlen: int
    batch_size: int
    start: int
    size: int
    reduce_rows: int
    is_decode: bool = False


_PRODUCT_REQUIRED_GRAPH_KEYS: tuple[str, ...] = ()
_PRODUCT_KERNEL_CACHE_FIELDS: tuple[str, ...] = (
    "embedding_hc_mhc_pre_from_ids_kernels",
    "hash_moe_dispatch_no_bias_kernels",
    "learned_moe_dispatch_no_bias_kernels",
    "learned_moe_dispatch_with_bias_kernels",
    "attention_inverse_rope_tail_flat_kernels",
    "attention_out_dp_flat_kernels",
    "attention_inverse_rope_out_dp_flat_kernels",
    "dp_attention_all_reduce_kernels",
    "dp_attention_all_reduce_post_pre_kernels",
    "dp_attention_unpad_post_pre_kernels",
    "sequence_hidden_pad_kernels",
    "dp_attention_hash_moe_dispatch_no_bias_kernels",
    "dp_attention_learned_moe_dispatch_no_bias_kernels",
    "dp_attention_learned_moe_dispatch_with_bias_kernels",
    "dp_attention_moe_blockwise_kernels",
    QkvVariantName.QKV_QUANT,
    QkvVariantName.QKV_WRITE_KV_CACHE,
    QkvVariantName.INDEXER_COMPRESSOR_TABLE,
    QkvVariantName.INDEXER_COMPRESSOR_TABLE_WRITE_SWA_STATE,
    QkvVariantName.INDEXER_ALL_KV_TOPK_PREP,
    QkvVariantName.INDEXER_ALL_KV_TOPK_PREP_WRITE_SWA_STATE,
    QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_PREP,
    QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE,
    QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_PREP,
    QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE,
    QkvVariantName.EMPTY_INDEXER_COMPRESSOR_TOPK,
    QkvVariantName.TOKEN_TOPK_PREP,
    QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP,
    QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP_WRITE_SWA_STATE,
    QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_PREP,
    QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE,
    QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_PREP,
    QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE,
    QkvVariantName.COMPRESSOR_TABLE,
    "compressor_post_qdq_freq_table_kernels",
    "compressor_decode_pool_post_qdq_freq_table_kernels",
    "indexer_sparse_attention_prep_static_kernels",
    "shared_expert_add_restore_post_pre_kernels",
    "shared_expert_add_restore_head_select_kernels",
    "shared_expert_add_restore_head_top1_kernels",
)
_PRODUCT_PREALLOCATED_OUTPUT_FIELDS: tuple[str, ...] = (
    "attention_outputs",
    "moe_prefill_outputs",
    "moe_prefill_ep_outputs",
    "moe_prefill_tp_outputs",
    "moe_decode_outputs",
    "moe_decode_ep_outputs",
    "moe_decode_tp_outputs",
)
