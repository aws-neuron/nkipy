"""QKV variant dispatcher for compressed DSV4 attention."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.base import (
    _run_compressed_attention_base_qkv,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.compressor_token_topk import (
    _run_compressed_attention_token_topk_qkv,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.indexer_all_kv import (
    _run_compressed_attention_indexer_all_kv_qkv,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.indexer_table import (
    _run_compressed_attention_indexer_table_qkv,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.setup import (
    Dsv4CompressedAttentionQkvResult,
    Dsv4CompressedAttentionQkvSetup,
)


def _run_compressed_attention_qkv(
    *,
    qkv_setup: Dsv4CompressedAttentionQkvSetup,
    fns: Dsv4GraphFns,
    x: np.ndarray,
    attn: Any,
    build_dir: str | None,
    device_layer_state: Any,
    owner_ids: np.ndarray,
    owner_ids_dev: Any | None,
    device_token_positions: Any | None,
    qkv_positions: np.ndarray,
    qkv_positions_input: Any,
    freqs: np.ndarray | None,
    freqs_cos: Any,
    freqs_sin: Any,
    qkv_fuses_q_scale: bool,
    active_bucket: int,
    win: int,
    ratio: int,
    start_pos: int,
    bsz: int,
    seqlen: int,
    prefill_device_primary: bool,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None,
) -> Dsv4CompressedAttentionQkvResult:
    variant = qkv_setup.variant
    qkv_outputs = qkv_setup.outputs
    qkv_outputs_flat_kv = bool(qkv_setup.qkv_outputs_flat_kv)
    compressor_wkv = qkv_setup.compressor_wkv
    compressor_wgate = qkv_setup.compressor_wgate
    compressor_ape = qkv_setup.compressor_ape
    compressor_norm_weight = qkv_setup.compressor_norm_weight
    compressor_freqs_cos = qkv_setup.compressor_freqs_cos
    compressor_freqs_sin = qkv_setup.compressor_freqs_sin
    indexer_obj = qkv_setup.indexer_obj
    indexer_compressor = qkv_setup.indexer_compressor
    indexer_compressor_wkv = qkv_setup.indexer_compressor_wkv
    indexer_compressor_wgate = qkv_setup.indexer_compressor_wgate
    indexer_freqs_cos = qkv_setup.indexer_freqs_cos
    indexer_freqs_sin = qkv_setup.indexer_freqs_sin
    compressor_prefill_state_tail_len = int(qkv_setup.compressor_prefill_state_tail_len)
    indexer_prefill_state_tail_len = int(qkv_setup.indexer_prefill_state_tail_len)
    compressed_kv_len = int(qkv_setup.compressed_kv_len)
    indexer_k = int(qkv_setup.indexer_k)
    token_topk_offset = int(qkv_setup.token_topk_offset)
    token_topk_max_c_len = int(qkv_setup.token_topk_max_c_len)
    use_qkv_compressor_prefill_post_qdq_token_topk_bucketed = bool(
        qkv_setup.use_qkv_compressor_prefill_post_qdq_token_topk_bucketed
    )
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed = bool(
        qkv_setup.use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed
    )
    qkv_compressor_token_topk_prep_write_swa_state = (
        qkv_setup.qkv_compressor_token_topk_prep_write_swa_state
    )
    qkv_compressor_decode_post_qdq_token_topk_prep = (
        qkv_setup.qkv_compressor_decode_post_qdq_token_topk_prep
    )
    qkv_compressor_prefill_post_qdq_token_topk_prep = (
        qkv_setup.qkv_compressor_prefill_post_qdq_token_topk_prep
    )
    qkv_compressor_token_topk_prep = qkv_setup.qkv_compressor_token_topk_prep
    qkv_token_topk_prep = qkv_setup.qkv_token_topk_prep
    qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep = (
        qkv_setup.qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep
    )
    qkv_indexer_compressor_all_kv_topk_prep_write_swa_state = (
        qkv_setup.qkv_indexer_compressor_all_kv_topk_prep_write_swa_state
    )
    qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep = (
        qkv_setup.qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep
    )
    qkv_indexer_compressor_all_kv_topk_prep = (
        qkv_setup.qkv_indexer_compressor_all_kv_topk_prep
    )
    token_topk_qkv = _run_compressed_attention_token_topk_qkv(
        variant=variant,
        use_qkv_compressor_prefill_post_qdq_token_topk_bucketed=(
            use_qkv_compressor_prefill_post_qdq_token_topk_bucketed
        ),
        qkv_compressor_token_topk_prep_write_swa_state=(
            qkv_compressor_token_topk_prep_write_swa_state
        ),
        qkv_compressor_decode_post_qdq_token_topk_prep=(
            qkv_compressor_decode_post_qdq_token_topk_prep
        ),
        qkv_compressor_prefill_post_qdq_token_topk_prep=(
            qkv_compressor_prefill_post_qdq_token_topk_prep
        ),
        qkv_compressor_token_topk_prep=qkv_compressor_token_topk_prep,
        qkv_token_topk_prep=qkv_token_topk_prep,
        x=x,
        attn=attn,
        build_dir=build_dir,
        device_layer_state=device_layer_state,
        owner_ids=owner_ids,
        owner_ids_dev=owner_ids_dev,
        device_token_positions=device_token_positions,
        qkv_positions=qkv_positions,
        qkv_positions_input=qkv_positions_input,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        compressor_freqs_cos=compressor_freqs_cos,
        compressor_freqs_sin=compressor_freqs_sin,
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_ape=compressor_ape,
        compressor_norm_weight=compressor_norm_weight,
        qkv_outputs=qkv_outputs,
        qkv_outputs_flat_kv=bool(qkv_outputs_flat_kv),
        active_bucket=int(active_bucket),
        win=int(win),
        ratio=int(ratio),
        token_topk_offset=int(token_topk_offset),
        start_pos=int(start_pos),
        token_topk_max_c_len=int(token_topk_max_c_len),
        bsz=int(bsz),
        seqlen=int(seqlen),
        compressor_prefill_state_tail_len=int(compressor_prefill_state_tail_len),
    )
    if token_topk_qkv.handled:
        return Dsv4CompressedAttentionQkvResult(
            q_dev=token_topk_qkv.q_dev,
            kv_dev=token_topk_qkv.kv_dev,
            qr_dev=token_topk_qkv.qr_dev,
            topk_t_dev=token_topk_qkv.topk_t_dev,
            mask_dev=token_topk_qkv.mask_dev,
            precomputed_compressor_kv_score=(
                token_topk_qkv.precomputed_compressor_kv_score
            ),
            precomputed_compressor_prefill_scatter_rows=(
                token_topk_qkv.precomputed_compressor_prefill_scatter_rows
            ),
            precomputed_compressor_decode_scatter_rows=(
                token_topk_qkv.precomputed_compressor_decode_scatter_rows
            ),
            compressor_state_swa_write_fused=bool(
                token_topk_qkv.compressor_state_swa_write_fused
            ),
            bucketed_prefill_done=bool(token_topk_qkv.bucketed_prefill_done),
            bucketed_kv_primary=token_topk_qkv.bucketed_kv_primary,
            token_topk_offset=int(token_topk_qkv.token_topk_offset),
        )

    indexer_all_kv_qkv = _run_compressed_attention_indexer_all_kv_qkv(
        variant=variant,
        use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed=(
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed
        ),
        qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep=(
            qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep
        ),
        qkv_indexer_compressor_all_kv_topk_prep_write_swa_state=(
            qkv_indexer_compressor_all_kv_topk_prep_write_swa_state
        ),
        qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep=(
            qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep
        ),
        qkv_indexer_compressor_all_kv_topk_prep=(
            qkv_indexer_compressor_all_kv_topk_prep
        ),
        fns=fns,
        x=x,
        attn=attn,
        build_dir=build_dir,
        device_layer_state=device_layer_state,
        owner_ids=owner_ids,
        owner_ids_dev=owner_ids_dev,
        device_token_positions=device_token_positions,
        qkv_positions=qkv_positions,
        qkv_positions_input=qkv_positions_input,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        compressor_freqs_cos=compressor_freqs_cos,
        compressor_freqs_sin=compressor_freqs_sin,
        indexer_freqs_cos=indexer_freqs_cos,
        indexer_freqs_sin=indexer_freqs_sin,
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_ape=compressor_ape,
        compressor_norm_weight=compressor_norm_weight,
        indexer_compressor_wkv=indexer_compressor_wkv,
        indexer_compressor_wgate=indexer_compressor_wgate,
        indexer_compressor=indexer_compressor,
        indexer_obj=indexer_obj,
        qkv_outputs=qkv_outputs,
        active_bucket=int(active_bucket),
        win=int(win),
        ratio=int(ratio),
        start_pos=int(start_pos),
        seqlen=int(seqlen),
        bsz=int(bsz),
        compressed_kv_len=int(compressed_kv_len),
        indexer_k=int(indexer_k),
        prefill_device_primary=bool(prefill_device_primary),
        compressor_prefill_state_tail_len=int(compressor_prefill_state_tail_len),
        indexer_prefill_state_tail_len=int(indexer_prefill_state_tail_len),
        attention_scratch=attention_scratch,
    )
    if indexer_all_kv_qkv.handled:
        return Dsv4CompressedAttentionQkvResult(
            q_dev=indexer_all_kv_qkv.q_dev,
            kv_dev=indexer_all_kv_qkv.kv_dev,
            qr_dev=indexer_all_kv_qkv.qr_dev,
            topk_t_dev=indexer_all_kv_qkv.topk_t_dev,
            mask_dev=indexer_all_kv_qkv.mask_dev,
            precomputed_compressor_kv_score=(
                indexer_all_kv_qkv.precomputed_compressor_kv_score
            ),
            precomputed_compressor_prefill_scatter_rows=(
                indexer_all_kv_qkv.precomputed_compressor_prefill_scatter_rows
            ),
            precomputed_compressor_decode_scatter_rows=(
                indexer_all_kv_qkv.precomputed_compressor_decode_scatter_rows
            ),
            compressor_state_swa_write_fused=bool(
                indexer_all_kv_qkv.compressor_state_swa_write_fused
            ),
            deferred_indexer_state=indexer_all_kv_qkv.deferred_indexer_state,
            bucketed_prefill_done=bool(indexer_all_kv_qkv.bucketed_prefill_done),
            bucketed_kv_primary=indexer_all_kv_qkv.bucketed_kv_primary,
            token_topk_offset=int(token_topk_offset),
            all_kv_offset=int(indexer_all_kv_qkv.all_kv_offset),
            attention_rows=indexer_all_kv_qkv.attention_rows,
        )

    indexer_table_qkv = _run_compressed_attention_indexer_table_qkv(
        qkv_setup=qkv_setup,
        x=x,
        attn=attn,
        device_layer_state=device_layer_state,
        owner_ids=owner_ids,
        owner_ids_dev=owner_ids_dev,
        device_token_positions=device_token_positions,
        qkv_positions=qkv_positions,
        qkv_positions_input=qkv_positions_input,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        active_bucket=int(active_bucket),
        win=int(win),
        ratio=int(ratio),
        start_pos=int(start_pos),
        bsz=int(bsz),
        seqlen=int(seqlen),
        prefill_device_primary=bool(prefill_device_primary),
    )
    if indexer_table_qkv.handled:
        return indexer_table_qkv

    return _run_compressed_attention_base_qkv(
        qkv_setup=qkv_setup,
        fns=fns,
        x=x,
        attn=attn,
        freqs=freqs,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        qkv_positions_input=qkv_positions_input,
        qkv_fuses_q_scale=bool(qkv_fuses_q_scale),
        active_bucket=int(active_bucket),
    )
