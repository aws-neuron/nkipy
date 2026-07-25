"""Typed execution capability flags derived from the graph function table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Dsv4ExecutionCapabilities:
    attention_shape_helpers_alias: bool = False
    require_fused_attention_qkv_table: bool = False
    require_flat_swa_kv: bool = False
    require_fused_swa_kv_write: bool = False
    require_fused_sparse_attention_prep: bool = False
    require_fused_inverse_rope_out: bool = False
    require_precomputed_compressor_kv_score: bool = False
    prefix_two_token_flats_aliases_prefix: bool = False
    require_fused_compressor_post_qdq: bool = False
    require_precomputed_empty_indexer_topk: bool = False
    require_precomputed_indexer_qw: bool = False
    indexer_sparse_prep_accepts_positions: bool = False

    @classmethod
    def from_graph_fns(cls, fns: Mapping[str, Any]) -> "Dsv4ExecutionCapabilities":
        return cls(
            attention_shape_helpers_alias=bool(
                fns.get("_product_attention_shape_helpers_alias", False)
            ),
            require_fused_attention_qkv_table=bool(
                fns.get("_product_require_fused_attention_qkv_table", False)
            ),
            require_flat_swa_kv=bool(fns.get("_product_require_flat_swa_kv", False)),
            require_fused_swa_kv_write=bool(
                fns.get("_product_require_fused_swa_kv_write", False)
            ),
            require_fused_sparse_attention_prep=bool(
                fns.get("_product_require_fused_sparse_attention_prep", False)
            ),
            require_fused_inverse_rope_out=bool(
                fns.get("_product_require_fused_inverse_rope_out", False)
            ),
            require_precomputed_compressor_kv_score=bool(
                fns.get("_product_require_precomputed_compressor_kv_score", False)
            ),
            prefix_two_token_flats_aliases_prefix=bool(
                fns.get("_product_prefix_two_token_flats_aliases_prefix", False)
            ),
            require_fused_compressor_post_qdq=bool(
                fns.get("_product_require_fused_compressor_post_qdq", False)
            ),
            require_precomputed_empty_indexer_topk=bool(
                fns.get("_product_require_precomputed_empty_indexer_topk", False)
            ),
            require_precomputed_indexer_qw=bool(
                fns.get("_product_require_precomputed_indexer_qw", False)
            ),
            indexer_sparse_prep_accepts_positions=bool(
                fns.get("_product_indexer_sparse_prep_accepts_positions", False)
            ),
        )


__all__ = ["Dsv4ExecutionCapabilities"]
