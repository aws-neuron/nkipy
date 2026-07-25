"""Neutral DSV4 graph variant identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GraphVariantName = str


class QkvVariantName:
    COMPRESSOR_TOKEN_TOPK_PREP_WRITE_SWA_STATE: GraphVariantName = (
        "compressor_token_topk_prep_write_swa_state"
    )
    COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE: GraphVariantName = (
        "compressor_decode_post_qdq_token_topk_write_swa_state_cache"
    )
    COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_PREP: GraphVariantName = (
        "compressor_decode_post_qdq_token_topk_prep"
    )
    COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE: GraphVariantName = (
        "compressor_prefill_post_qdq_token_topk_write_swa_state_cache"
    )
    COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_PREP: GraphVariantName = (
        "compressor_prefill_post_qdq_token_topk_prep"
    )
    COMPRESSOR_TOKEN_TOPK_PREP: GraphVariantName = "compressor_token_topk_prep"
    TOKEN_TOPK_PREP: GraphVariantName = "token_topk_prep"
    INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE: GraphVariantName = (
        "indexer_all_kv_prefill_post_qdq_topk_write_swa_state_cache"
    )
    INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_PREP: GraphVariantName = (
        "indexer_all_kv_prefill_post_qdq_topk_prep"
    )
    INDEXER_ALL_KV_TOPK_PREP_WRITE_SWA_STATE: GraphVariantName = (
        "indexer_all_kv_topk_prep_write_swa_state"
    )
    INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE: GraphVariantName = (
        "indexer_all_kv_decode_post_qdq_topk_write_swa_state_cache"
    )
    INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_PREP: GraphVariantName = (
        "indexer_all_kv_decode_post_qdq_topk_prep"
    )
    INDEXER_ALL_KV_TOPK_PREP: GraphVariantName = "indexer_all_kv_topk_prep"
    INDEXER_COMPRESSOR_TABLE_WRITE_SWA_STATE: GraphVariantName = (
        "indexer_compressor_table_write_swa_state"
    )
    INDEXER_COMPRESSOR_TABLE: GraphVariantName = "indexer_compressor_table"
    EMPTY_INDEXER_COMPRESSOR_TOPK: GraphVariantName = "empty_indexer_compressor_topk"
    COMPRESSOR_TABLE: GraphVariantName = "compressor_table"
    QKV_WRITE_KV_CACHE: GraphVariantName = "qkv_write_kv_cache"
    QKV_QUANT: GraphVariantName = "qkv_quant"


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: GraphVariantName
    family: str

    def is_name(self, name: GraphVariantName) -> bool:
        return self.name == name


@dataclass(frozen=True, slots=True)
class VariantInputs:
    values: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VariantOutputs:
    tensors: dict[str, Any | None] | None
    flat_kv: bool


_VARIANT_FAMILIES: dict[GraphVariantName, str] = {
    QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP_WRITE_SWA_STATE: "compressor_token_topk",
    QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE: (
        "compressor_token_topk"
    ),
    QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_PREP: (
        "compressor_token_topk"
    ),
    QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE: (
        "compressor_token_topk"
    ),
    QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_PREP: (
        "compressor_token_topk"
    ),
    QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP: "compressor_token_topk",
    QkvVariantName.TOKEN_TOPK_PREP: "compressor_token_topk",
    QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE: (
        "indexer_all_kv"
    ),
    QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_PREP: "indexer_all_kv",
    QkvVariantName.INDEXER_ALL_KV_TOPK_PREP_WRITE_SWA_STATE: "indexer_all_kv",
    QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE: (
        "indexer_all_kv"
    ),
    QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_PREP: "indexer_all_kv",
    QkvVariantName.INDEXER_ALL_KV_TOPK_PREP: "indexer_all_kv",
    QkvVariantName.INDEXER_COMPRESSOR_TABLE_WRITE_SWA_STATE: "indexer_table",
    QkvVariantName.INDEXER_COMPRESSOR_TABLE: "indexer_table",
    QkvVariantName.EMPTY_INDEXER_COMPRESSOR_TOPK: "indexer_table",
    QkvVariantName.COMPRESSOR_TABLE: "base",
    QkvVariantName.QKV_WRITE_KV_CACHE: "base",
    QkvVariantName.QKV_QUANT: "base",
}

VARIANT_SPECS: dict[GraphVariantName, VariantSpec] = {
    name: VariantSpec(name=name, family=family)
    for name, family in _VARIANT_FAMILIES.items()
}


def variant_spec(name: GraphVariantName) -> VariantSpec:
    try:
        return VARIANT_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown DSV4 QKV graph variant: {name}") from exc
