"""QKV graph variant selection for DSV4 attention."""

from __future__ import annotations

from nkipy_serving.models.deepseek_v4.variants import (
    GraphVariantName,
    QkvVariantName,
)


def _select_qkv_variant_name(
    *,
    use_qkv_compressor_token_topk_prep_write_swa_state: bool,
    use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache: bool,
    use_qkv_compressor_decode_post_qdq_token_topk_prep: bool,
    use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache: bool,
    use_qkv_compressor_prefill_post_qdq_token_topk_prep: bool,
    use_qkv_compressor_token_topk_prep: bool,
    use_qkv_token_topk_prep: bool,
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache: bool,
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep: bool,
    use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state: bool,
    use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache: bool,
    use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep: bool,
    use_qkv_indexer_compressor_all_kv_topk_prep: bool,
    use_qkv_indexer_compressor_table_write_swa_state: bool,
    use_qkv_indexer_compressor_table: bool,
    use_qkv_empty_indexer_compressor_topk: bool,
    use_qkv_compressor_table: bool,
) -> GraphVariantName:
    if use_qkv_compressor_token_topk_prep_write_swa_state:
        return QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP_WRITE_SWA_STATE
    if use_qkv_compressor_decode_post_qdq_token_topk_write_swa_state_cache:
        return (
            QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE
        )
    if use_qkv_compressor_decode_post_qdq_token_topk_prep:
        return QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_PREP
    if use_qkv_compressor_prefill_post_qdq_token_topk_write_swa_state_cache:
        return (
            QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE
        )
    if use_qkv_compressor_prefill_post_qdq_token_topk_prep:
        return QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_PREP
    if use_qkv_compressor_token_topk_prep:
        return QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP
    if use_qkv_token_topk_prep:
        return QkvVariantName.TOKEN_TOPK_PREP
    if use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_state_cache:
        return QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE
    if use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep:
        return QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_PREP
    if use_qkv_indexer_compressor_all_kv_topk_prep_write_swa_state:
        return QkvVariantName.INDEXER_ALL_KV_TOPK_PREP_WRITE_SWA_STATE
    if use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_state_cache:
        return QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE
    if use_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep:
        return QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_PREP
    if use_qkv_indexer_compressor_all_kv_topk_prep:
        return QkvVariantName.INDEXER_ALL_KV_TOPK_PREP
    if use_qkv_indexer_compressor_table_write_swa_state:
        return QkvVariantName.INDEXER_COMPRESSOR_TABLE_WRITE_SWA_STATE
    if use_qkv_indexer_compressor_table:
        return QkvVariantName.INDEXER_COMPRESSOR_TABLE
    if use_qkv_empty_indexer_compressor_topk:
        return QkvVariantName.EMPTY_INDEXER_COMPRESSOR_TOPK
    if use_qkv_compressor_table:
        return QkvVariantName.COMPRESSOR_TABLE
    return QkvVariantName.QKV_QUANT
