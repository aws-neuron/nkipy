"""Descriptor-driven registry for traceable DSV4 NEFF graph functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nkipy_serving.fragment_jit import jit
from nkipy_serving.models.deepseek_v4.graph_types import (
    Dsv4GraphFns,
    _with_dsv4_fp8_compiler_arg,
)
from nkipy_serving.models.deepseek_v4.neff_graphs import attention as graph_attention
from nkipy_serving.models.deepseek_v4.neff_graphs import common as graph_common
from nkipy_serving.models.deepseek_v4.neff_graphs import moe as graph_moe


@dataclass(frozen=True, slots=True)
class _GraphDescriptor:
    key: str
    fn: Callable[..., Any]
    name: str
    collective: bool = False


_GRAPH_DESCRIPTORS: tuple[_GraphDescriptor, ...] = (
    _GraphDescriptor("embedding_hc", graph_common.embedding_hc_fn, "embedding_hc"),
    _GraphDescriptor(
        "vocab_parallel_embedding_hc",
        graph_common.vocab_parallel_embedding_hc_fn,
        "vocab_parallel_embedding_hc",
        collective=True,
    ),
    _GraphDescriptor("linear", graph_common.linear_fn, "linear"),
    _GraphDescriptor(
        "attention_zero_like",
        graph_common.attention_zero_like_fn,
        "attention_zero_like",
    ),
    _GraphDescriptor("two_linear", graph_common.two_linear_fn, "two_linear"),
    _GraphDescriptor(
        "gate_scores_no_bias",
        graph_common.gate_scores_no_bias_fn,
        "gate_scores_no_bias",
    ),
    _GraphDescriptor(
        "gate_scores_with_bias",
        graph_common.gate_scores_with_bias_fn,
        "gate_scores_with_bias",
    ),
    _GraphDescriptor("mhc_pre_gemm", graph_common.mhc_pre_gemm_fn, "mhc_pre_gemm"),
    _GraphDescriptor(
        "mhc_pre_mix",
        graph_common.mhc_pre_mix_sinkhorn_fn,
        "mhc_pre_mix_sinkhorn",
    ),
    _GraphDescriptor("mhc_pre_apply", graph_common.mhc_pre_apply_fn, "mhc_pre_apply"),
    _GraphDescriptor("mhc_pre", graph_common.mhc_pre_fn, "mhc_pre"),
    _GraphDescriptor("mhc_post", graph_common.mhc_post_fn, "mhc_post"),
    _GraphDescriptor("hc_head", graph_common.hc_head_fn, "hc_head"),
    _GraphDescriptor(
        "attention_out",
        graph_moe.attention_out_proj_fn,
        "attn_out",
        collective=True,
    ),
    _GraphDescriptor(
        "attention_out_flat",
        graph_attention.attention_out_proj_flat_fn,
        "attn_out_flat",
        collective=True,
    ),
    _GraphDescriptor(
        "inverse_rope_tail",
        graph_moe.inverse_rope_tail_fn,
        "inverse_rope_tail",
    ),
    _GraphDescriptor(
        "inverse_rope_tail_flat",
        graph_attention.inverse_rope_tail_flat_fn,
        "inverse_rope_tail_flat",
    ),
    _GraphDescriptor(
        "attention_qkv_quant",
        graph_attention.attention_qkv_quant_fn,
        "attn_qkv_quant",
    ),
    _GraphDescriptor(
        "q_scale_transpose",
        graph_attention.q_scale_transpose_fn,
        "q_scale_transpose",
    ),
    _GraphDescriptor(
        "attention_kv_flatten",
        graph_attention.attention_kv_flatten_fn,
        "attention_kv_flatten",
    ),
    _GraphDescriptor(
        "attention_kv_tail_window",
        graph_attention.attention_kv_tail_window_fn,
        "attention_kv_tail_window",
    ),
    _GraphDescriptor(
        "attention_kv_request_tail_window",
        graph_attention.attention_kv_request_tail_window_fn,
        "attention_kv_request_tail_window",
    ),
    _GraphDescriptor(
        "attention_sink_2d",
        graph_attention.attention_sink_2d_fn,
        "attention_sink_2d",
    ),
    _GraphDescriptor(
        "compressor_norm_2d",
        graph_attention.compressor_norm_2d_fn,
        "compressor_norm_2d",
    ),
    _GraphDescriptor(
        "attention_unpad_reshape",
        graph_attention.attention_unpad_reshape_fn,
        "attention_unpad_reshape",
    ),
    _GraphDescriptor(
        "attention_hidden_reshape",
        graph_attention.attention_hidden_reshape_fn,
        "attention_hidden_reshape",
    ),
    _GraphDescriptor(
        "head_hidden_flatten",
        graph_attention.head_hidden_flatten_fn,
        "head_hidden_flatten",
    ),
    _GraphDescriptor(
        "head_hidden_flatten_pad",
        graph_common.head_hidden_flatten_pad_fn,
        "head_hidden_flatten_pad",
    ),
    _GraphDescriptor("pad_flat_rows", graph_common.pad_flat_rows_fn, "pad_flat_rows"),
    _GraphDescriptor("pad_topk_rows", graph_common.pad_topk_rows_fn, "pad_topk_rows"),
    _GraphDescriptor(
        "sequence_hidden_pad",
        graph_common.sequence_hidden_pad_fn,
        "sequence_hidden_pad",
    ),
    _GraphDescriptor(
        "dp_attention_lane_slice",
        graph_attention.dp_attention_lane_slice_fn,
        "dp_attention_lane_slice",
    ),
    _GraphDescriptor(
        "dp_attention_lane_scatter",
        graph_attention.dp_attention_lane_scatter_fn,
        "dp_attention_lane_scatter",
    ),
    _GraphDescriptor(
        "dp_attention_all_reduce",
        graph_common.dp_attention_all_reduce_fn,
        "dp_attention_all_reduce",
        collective=True,
    ),
    _GraphDescriptor(
        "dp_attention_flatten_pad",
        graph_attention.dp_attention_flatten_pad_fn,
        "dp_attention_flatten_pad",
    ),
    _GraphDescriptor(
        "dp_attention_unpad_reshape",
        graph_common.dp_attention_unpad_reshape_fn,
        "dp_attention_unpad_reshape",
    ),
    _GraphDescriptor("topk_concat", graph_common.topk_concat_fn, "topk_concat"),
    _GraphDescriptor(
        "window_topk_from_tokens",
        graph_common.window_topk_from_tokens_fn,
        "window_topk_from_tokens",
    ),
    _GraphDescriptor(
        "compressed_topk_no_indexer_from_tokens",
        graph_common.compressed_topk_no_indexer_from_tokens_fn,
        "compressed_topk_no_indexer_from_tokens",
    ),
    _GraphDescriptor(
        "invalid_topk_from_tokens",
        graph_common.invalid_topk_from_tokens_fn,
        "invalid_topk_from_tokens",
    ),
    _GraphDescriptor(
        "topk_sparse_attention_prep",
        graph_common.topk_sparse_attention_prep_fn,
        "topk_sparse_attention_prep",
    ),
    _GraphDescriptor(
        "indexer_q_transform",
        graph_common.indexer_q_transform_fn,
        "indexer_q_transform",
    ),
    _GraphDescriptor(
        "indexer_q_reshape",
        graph_moe.indexer_q_reshape_fn,
        "indexer_q_reshape",
    ),
    _GraphDescriptor(
        "indexer_score_qw_prep",
        graph_common.indexer_score_qw_prep_fn,
        "indexer_score_qw_prep",
    ),
    _GraphDescriptor(
        "indexer_score_reshape",
        graph_moe.indexer_score_reshape_fn,
        "indexer_score_reshape",
    ),
    _GraphDescriptor(
        "indexer_project_qw_prep",
        graph_common.indexer_project_qw_prep_fn,
        "indexer_project_qw_prep",
    ),
    _GraphDescriptor(
        "indexer_project_qw_prep_from_freq_table",
        graph_common.indexer_project_qw_prep_from_freq_table_fn,
        "indexer_project_qw_prep_from_freq_table",
    ),
    _GraphDescriptor("decode_pool", graph_moe.decode_pool_fn, "decode_pool"),
    _GraphDescriptor(
        "decode_overlap_pool",
        graph_moe.decode_overlap_pool_fn,
        "decode_overlap_pool",
    ),
    _GraphDescriptor("router_tail", graph_moe.router_tail_fn, "router_tail"),
    _GraphDescriptor("topk_rebase", graph_common.topk_rebase_fn, "topk_rebase"),
    _GraphDescriptor(
        "causal_mask_add", graph_moe.causal_mask_add_fn, "causal_mask_add"
    ),
    _GraphDescriptor(
        "indexer_project",
        graph_common.indexer_project_fn,
        "indexer_project",
    ),
    _GraphDescriptor("swiglu", graph_moe.swiglu_fn, "swiglu"),
    _GraphDescriptor(
        "shared_expert_add",
        graph_moe.shared_expert_add_fn,
        "shared_expert_add",
        collective=True,
    ),
    _GraphDescriptor(
        "moe_hidden_flatten",
        graph_moe.moe_hidden_flatten_fn,
        "moe_hidden_flatten",
    ),
    _GraphDescriptor(
        "moe_hidden_flatten_pad",
        graph_moe.moe_hidden_flatten_pad_fn,
        "moe_hidden_flatten_pad",
    ),
    _GraphDescriptor(
        "moe_routed_unpad",
        graph_moe.moe_routed_unpad_fn,
        "moe_routed_unpad",
    ),
    _GraphDescriptor(
        "compressor_kv_score_bf16",
        graph_common.compressor_kv_score_bf16_fn,
        "compressor_kv_score_bf16",
    ),
    _GraphDescriptor(
        "prefix_two_token_flats",
        graph_moe.prefix_two_token_flats_fn,
        "prefix_two_token_flats",
    ),
    _GraphDescriptor("cast_bf16", graph_common.cast_bf16_fn, "cast_bf16"),
    _GraphDescriptor(
        "compressor_qdq_bf16",
        graph_common.compressor_qdq_bf16_fn,
        "compressor_qdq_bf16",
    ),
    _GraphDescriptor("topk_linearize", graph_moe.topk_linearize_fn, "topk_linearize"),
    _GraphDescriptor("hash_route", graph_moe.hash_route_fn, "hash_route"),
    _GraphDescriptor(
        "compressor_pool",
        graph_common.compressor_pool_fn,
        "compressor_pool",
    ),
    _GraphDescriptor(
        "overlap_transform",
        graph_common.overlap_transform_fn,
        "overlap_transform",
    ),
    _GraphDescriptor("fp8_act_qdq", graph_common.fp8_act_qdq_fn, "fp8_act_qdq"),
    _GraphDescriptor("topk_idx", graph_moe.topk_idx_fn, "topk_idx"),
)


def build_dsv4_graph_fns(
    *,
    build_dir: str | Path | None = None,
    compiler_args: str = "",
    name_prefix: str = "dsv4_sampled",
    cc_enabled: bool = False,
    rank_id: int | None = None,
    world_size: int | None = None,
) -> Dsv4GraphFns:
    """Build the traceable trace-function callable table."""
    build_dir_s = str(build_dir) if build_dir is not None else None
    compiler_args = _with_dsv4_fp8_compiler_arg(compiler_args)

    def _frag(
        fn: Callable[..., Any],
        name: str,
        *,
        collective: bool = False,
    ) -> Callable[..., Any]:
        return jit(
            fn,
            device=True,
            name=f"{name_prefix}_{name}",
            build_dir=build_dir_s,
            additional_compiler_args=compiler_args,
            cc_enabled=bool(cc_enabled) if collective else None,
            rank_id=rank_id if collective else None,
            world_size=world_size if collective else None,
        )

    return {
        descriptor.key: _frag(
            descriptor.fn,
            descriptor.name,
            collective=descriptor.collective,
        )
        for descriptor in _GRAPH_DESCRIPTORS
    }


__all__ = ["build_dsv4_graph_fns"]
