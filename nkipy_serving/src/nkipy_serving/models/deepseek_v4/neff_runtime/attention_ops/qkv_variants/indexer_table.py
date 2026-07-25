"""Indexer-table QKV variant runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _decode_positions_1d_array,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.setup import (
    Dsv4CompressedAttentionQkvResult,
    Dsv4CompressedAttentionQkvSetup,
    _decode_owner_pos_aliases,
)
from nkipy_serving.models.deepseek_v4.variants import QkvVariantName


def _run_compressed_attention_indexer_table_qkv(
    *,
    qkv_setup: Dsv4CompressedAttentionQkvSetup,
    x: np.ndarray,
    attn: Any,
    device_layer_state: Any,
    owner_ids: np.ndarray,
    owner_ids_dev: Any | None,
    device_token_positions: Any | None,
    qkv_positions: np.ndarray,
    qkv_positions_input: Any,
    freqs_cos: Any,
    freqs_sin: Any,
    active_bucket: int,
    win: int,
    ratio: int,
    start_pos: int,
    bsz: int,
    seqlen: int,
    prefill_device_primary: bool,
) -> Dsv4CompressedAttentionQkvResult:
    variant = qkv_setup.variant
    qkv_outputs = qkv_setup.outputs
    compressor_wkv = qkv_setup.compressor_wkv
    compressor_wgate = qkv_setup.compressor_wgate
    compressor_ape = qkv_setup.compressor_ape
    indexer_obj = qkv_setup.indexer_obj
    indexer_compressor = qkv_setup.indexer_compressor
    indexer_compressor_wkv = qkv_setup.indexer_compressor_wkv
    indexer_compressor_wgate = qkv_setup.indexer_compressor_wgate
    indexer_freqs_cos = qkv_setup.indexer_freqs_cos
    indexer_freqs_sin = qkv_setup.indexer_freqs_sin
    token_topk_offset = int(qkv_setup.token_topk_offset)
    qkv_indexer_compressor_table_write_swa_state = (
        qkv_setup.qkv_indexer_compressor_table_write_swa_state
    )
    qkv_indexer_compressor_table = qkv_setup.qkv_indexer_compressor_table
    qkv_empty_indexer_compressor_topk = qkv_setup.qkv_empty_indexer_compressor_topk

    if variant.name == QkvVariantName.INDEXER_COMPRESSOR_TABLE_WRITE_SWA_STATE:
        spec = device_layer_state.compressor.spec
        idx_spec = device_layer_state.indexer.spec
        idx_comp_ape = getattr(indexer_compressor, "ape")
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        q_dev, kv_dev, idx_q_t_dev, idx_w_dev = (
            qkv_indexer_compressor_table_write_swa_state(
                x,
                attn.wq_a,
                attn.q_norm,
                attn.wq_b,
                attn.wkv,
                attn.kv_norm,
                compressor_wkv,
                compressor_wgate,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
                indexer_obj.wq_b,
                indexer_obj.weights_proj,
                device_layer_state.swa_kv_cache,
                device_layer_state.compressor.kv_score_state,
                device_layer_state.indexer.kv_score_state,
                decode_owner_ids_dev if decode_owner_ids_dev is not None else owner_ids,
                compressor_ape,
                idx_comp_ape,
                freqs_cos,
                freqs_sin,
                indexer_freqs_cos,
                indexer_freqs_sin,
                (
                    decode_positions_1d
                    if decode_positions_1d is not None
                    else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
                ),
                n_heads=int(attn.n_heads),
                head_dim=int(attn.head_dim),
                rope_head_dim=int(attn.rope_head_dim),
                eps=float(attn.eps),
                block_size=64,
                fp8_max=240.0,
                q_softmax_scale=float(attn.softmax_scale),
                q_token_bucket=int(active_bucket),
                kv_token_bucket=int(active_bucket),
                indexer_score_scale=float(
                    indexer_obj.softmax_scale * indexer_obj.n_heads**-0.5
                ),
                indexer_n_heads=int(indexer_obj.n_heads),
                indexer_head_dim=int(indexer_obj.head_dim),
                indexer_rope_head_dim=int(indexer_obj.rope_head_dim),
                indexer_block_size=32,
                indexer_fp8_max=240.0,
                window_size=int(win),
                ratio=int(ratio),
                start_pos=int(start_pos),
                compressor_ring_size=int(spec.ring_size),
                indexer_compressor_ring_size=int(idx_spec.ring_size),
                **(
                    {"_nkipy_output_tensors": qkv_outputs}
                    if qkv_outputs is not None
                    else {}
                ),
            )
        )
        return Dsv4CompressedAttentionQkvResult(
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            indexer_precomputed_qw=(idx_q_t_dev, idx_w_dev),
            indexer_precomputed_compressor_state_written=True,
            compressor_state_swa_write_fused=True,
            token_topk_offset=int(token_topk_offset),
        )

    if variant.name == QkvVariantName.INDEXER_COMPRESSOR_TABLE:
        (
            q_dev,
            kv_dev,
            comp_kv_dev,
            comp_score_dev,
            idx_comp_kv_dev,
            idx_comp_score_dev,
            idx_q_t_dev,
            idx_w_dev,
        ) = qkv_indexer_compressor_table(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            indexer_obj.wq_b,
            indexer_obj.weights_proj,
            freqs_cos,
            freqs_sin,
            indexer_freqs_cos,
            indexer_freqs_sin,
            qkv_positions_input,
            n_heads=int(attn.n_heads),
            head_dim=int(attn.head_dim),
            rope_head_dim=int(attn.rope_head_dim),
            eps=float(attn.eps),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(attn.softmax_scale),
            q_token_bucket=int(active_bucket),
            kv_token_bucket=int(active_bucket),
            indexer_score_scale=float(
                indexer_obj.softmax_scale * indexer_obj.n_heads**-0.5
            ),
            indexer_n_heads=int(indexer_obj.n_heads),
            indexer_head_dim=int(indexer_obj.head_dim),
            indexer_rope_head_dim=int(indexer_obj.rope_head_dim),
            indexer_block_size=32,
            indexer_fp8_max=240.0,
            window_size=int(win),
            **(
                {"_nkipy_output_tensors": qkv_outputs}
                if qkv_outputs is not None
                else {}
            ),
        )
        table_executor = getattr(qkv_indexer_compressor_table, "__self__", None)
        table_offset, table_bucketed, _table_cseq = getattr(
            table_executor,
            "_product_last_qkv_compiled_offset",
            (int(token_topk_offset), False, int(seqlen)),
        )
        q_shape = tuple(int(dim) for dim in getattr(q_dev, "shape", ()))
        attention_rows = int(q_shape[0]) if q_shape else int(active_bucket)
        return Dsv4CompressedAttentionQkvResult(
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            indexer_precomputed_compressor_kv_score=(
                idx_comp_kv_dev,
                idx_comp_score_dev,
            ),
            indexer_precomputed_qw=(idx_q_t_dev, idx_w_dev),
            bucketed_prefill_done=bool(table_bucketed),
            bucketed_kv_primary=kv_dev if table_bucketed else None,
            token_topk_offset=int(table_offset) or int(token_topk_offset),
            attention_rows=int(attention_rows),
        )

    if variant.name == QkvVariantName.EMPTY_INDEXER_COMPRESSOR_TOPK:
        if int(start_pos) == 0 and prefill_device_primary:
            empty_indexer_offset = int(win)
        elif int(start_pos) == 0:
            empty_indexer_offset = int(seqlen)
        else:
            empty_indexer_offset = int(win)
        (
            q_dev,
            kv_dev,
            comp_kv_dev,
            comp_score_dev,
            idx_comp_kv_dev,
            idx_comp_score_dev,
            idx_topk_t_dev,
            idx_mask_dev,
        ) = qkv_empty_indexer_compressor_topk(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            freqs_cos,
            freqs_sin,
            qkv_positions_input,
            n_heads=int(attn.n_heads),
            head_dim=int(attn.head_dim),
            rope_head_dim=int(attn.rope_head_dim),
            eps=float(attn.eps),
            block_size=64,
            fp8_max=240.0,
            q_softmax_scale=float(attn.softmax_scale),
            q_token_bucket=int(active_bucket),
            kv_token_bucket=int(active_bucket),
            window_size=int(win),
            ratio=int(ratio),
            offset=int(empty_indexer_offset),
            start_pos=int(start_pos),
            max_c_len=0,
            rows=int(active_bucket),
            k_tile=int(K_TILE),
            **(
                {"_nkipy_output_tensors": qkv_outputs}
                if qkv_outputs is not None
                else {}
            ),
        )
        return Dsv4CompressedAttentionQkvResult(
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            indexer_precomputed_compressor_kv_score=(
                idx_comp_kv_dev,
                idx_comp_score_dev,
            ),
            indexer_precomputed_empty_topk=(idx_topk_t_dev, idx_mask_dev),
            token_topk_offset=int(token_topk_offset),
        )

    return Dsv4CompressedAttentionQkvResult(
        handled=False,
        token_topk_offset=int(token_topk_offset),
    )
