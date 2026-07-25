"""Indexer/all-KV QKV variant runner."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.graph_types import Dsv4GraphFns
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _decode_positions_1d_array,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.compressor import (
    Dsv4DeferredIndexerState,
    _run_compressor,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.qkv_variants.setup import (
    Dsv4CompressedAttentionQkvResult,
    _compressor_kernel_kwargs,
    _decode_owner_pos_aliases,
)
from nkipy_serving.models.deepseek_v4.variants import (
    QkvVariantName,
    VariantSpec,
)


def _compressed_attention_all_kv_topk_kwargs(
    attn: Any,
    *,
    active_bucket: int,
    win: int,
    ratio: int,
    all_kv_offset: int,
    start_pos: int,
    compressed_kv_len: int,
    indexer_k: int,
    qkv_outputs: dict[str, Any | None] | None,
) -> dict[str, Any]:
    kwargs = dict(
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
        offset=int(all_kv_offset),
        start_pos=int(start_pos),
        kv_len=int(compressed_kv_len),
        k=int(indexer_k),
        rows=int(active_bucket),
        k_tile=int(K_TILE),
    )
    if qkv_outputs is not None:
        kwargs["_nkipy_output_tensors"] = qkv_outputs
    return kwargs


def _run_compressed_attention_indexer_all_kv_qkv(
    *,
    variant: VariantSpec,
    use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed: bool,
    qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep: Any,
    qkv_indexer_compressor_all_kv_topk_prep_write_swa_state: Any,
    qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep: Any,
    qkv_indexer_compressor_all_kv_topk_prep: Any,
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
    freqs_cos: Any,
    freqs_sin: Any,
    compressor_freqs_cos: Any,
    compressor_freqs_sin: Any,
    indexer_freqs_cos: Any,
    indexer_freqs_sin: Any,
    compressor_wkv: Any,
    compressor_wgate: Any,
    compressor_ape: Any,
    compressor_norm_weight: Any,
    indexer_compressor_wkv: Any,
    indexer_compressor_wgate: Any,
    indexer_compressor: Any,
    indexer_obj: Any,
    qkv_outputs: dict[str, Any | None] | None,
    active_bucket: int,
    win: int,
    ratio: int,
    start_pos: int,
    seqlen: int,
    bsz: int,
    compressed_kv_len: int,
    indexer_k: int,
    prefill_device_primary: bool,
    compressor_prefill_state_tail_len: int,
    indexer_prefill_state_tail_len: int,
    attention_scratch: Callable[[str, tuple[int, ...], Any], Any] | None,
) -> Dsv4CompressedAttentionQkvResult:
    def _step_all_kv_offset() -> int:
        if int(start_pos) != 0 or prefill_device_primary:
            return int(win)
        return int(seqlen)

    def _all_kv_kwargs(offset: int) -> dict[str, Any]:
        return _compressed_attention_all_kv_topk_kwargs(
            attn,
            active_bucket=int(active_bucket),
            win=int(win),
            ratio=int(ratio),
            all_kv_offset=int(offset),
            start_pos=int(start_pos),
            compressed_kv_len=int(compressed_kv_len),
            indexer_k=int(indexer_k),
            qkv_outputs=qkv_outputs,
        )

    def _prefill_positions_input() -> Any:
        if (
            int(start_pos) == 0
            and use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed
        ):
            return qkv_positions
        return qkv_positions_input

    if (
        variant.name
        == QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE
    ):
        all_kv_offset = _step_all_kv_offset()
        spec = device_layer_state.compressor.spec
        idx_spec = device_layer_state.indexer.spec
        idx_comp_ape = getattr(indexer_compressor, "ape")
        idx_comp_norm = getattr(indexer_compressor, "norm_weight")
        (
            q_dev,
            kv_dev,
            _comp_kv_dev,
            _comp_score_dev,
            _idx_comp_kv_dev,
            _idx_comp_score_dev,
            topk_t_dev,
            mask_dev,
            _comp_rows_dev,
            _idx_comp_rows_dev,
        ) = qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            compressor_ape,
            compressor_norm_weight,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            idx_comp_ape,
            idx_comp_norm,
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            indexer_freqs_cos,
            indexer_freqs_sin,
            _prefill_positions_input(),
            **_all_kv_kwargs(all_kv_offset),
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_overlap=bool(spec.overlap),
            **_compressor_kernel_kwargs(
                indexer_compressor, prefix="indexer_compressor"
            ),
            indexer_compressor_overlap=bool(idx_spec.overlap),
            write_swa_state_cache=True,
            swa_kv_cache=device_layer_state.swa_kv_cache,
            kv_score_state=device_layer_state.compressor.kv_score_state,
            compressed_kv_cache=device_layer_state.compressor.compressed_kv_cache,
            indexer_kv_score_state=device_layer_state.indexer.kv_score_state,
            indexer_compressed_kv_cache=(
                device_layer_state.indexer.compressed_kv_cache
            ),
            owner_ids=owner_ids_dev if owner_ids_dev is not None else owner_ids,
            compressor_ring_size=int(spec.ring_size),
            compressor_state_tail_len=int(compressor_prefill_state_tail_len),
            indexer_compressor_ring_size=int(idx_spec.ring_size),
            indexer_compressor_state_tail_len=int(indexer_prefill_state_tail_len),
            max_c_len=int(spec.max_compressed_len),
            indexer_max_c_len=int(idx_spec.max_compressed_len),
        )
        if use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed:
            all_kv_offset, all_kv_bucketed, _all_kv_cseq = (
                qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep.__self__._product_last_qkv_compiled_offset
            )
        else:
            all_kv_bucketed = False
        q_shape = tuple(int(dim) for dim in getattr(q_dev, "shape", ()))
        attention_rows = int(q_shape[0]) if q_shape else int(active_bucket)
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            compressor_state_swa_write_fused=True,
            bucketed_prefill_done=bool(all_kv_bucketed),
            bucketed_kv_primary=kv_dev if all_kv_bucketed else None,
            all_kv_offset=int(all_kv_offset),
            attention_rows=int(attention_rows),
        )

    if variant.name == QkvVariantName.INDEXER_ALL_KV_PREFILL_POST_QDQ_TOPK_PREP:
        all_kv_offset = _step_all_kv_offset()
        idx_comp_ape = getattr(indexer_compressor, "ape")
        idx_comp_norm = getattr(indexer_compressor, "norm_weight")
        (
            q_dev,
            kv_dev,
            comp_kv_dev,
            comp_score_dev,
            idx_comp_kv_dev,
            idx_comp_score_dev,
            topk_t_dev,
            mask_dev,
            comp_rows_dev,
            idx_comp_rows_dev,
        ) = qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            compressor_ape,
            compressor_norm_weight,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            idx_comp_ape,
            idx_comp_norm,
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            indexer_freqs_cos,
            indexer_freqs_sin,
            _prefill_positions_input(),
            **_all_kv_kwargs(all_kv_offset),
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_overlap=bool(attn.compressor.overlap),
            **_compressor_kernel_kwargs(
                indexer_compressor, prefix="indexer_compressor"
            ),
            indexer_compressor_overlap=bool(indexer_compressor.overlap),
        )
        if use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed:
            all_kv_offset, all_kv_bucketed, all_kv_cseq = (
                qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep.__self__._product_last_qkv_compiled_offset
            )
        q_shape = tuple(int(dim) for dim in getattr(q_dev, "shape", ()))
        attention_rows = int(q_shape[0]) if q_shape else int(active_bucket)
        if (
            use_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_bucketed
            and int(all_kv_cseq) >= 2 * int(ratio) - 1
        ):
            from nkipy_serving.ops.deepseek_v4.compressor_state import (
                run_compressor_prefill_dual_state_cache_swa_scatter_device as _bucketed_dual_scatter,
            )

            _bucketed_dual_scatter(
                attn.compressor,
                swa_kv_cache=device_layer_state.swa_kv_cache,
                swa_rows=kv_dev,
                swa_start_pos=int(start_pos),
                swa_bsz=int(bsz),
                swa_seqlen=int(seqlen),
                kv=comp_kv_dev,
                score=comp_score_dev,
                scatter_rows=comp_rows_dev,
                indexer_compressor=indexer_obj.compressor,
                indexer_kv=idx_comp_kv_dev,
                indexer_score=idx_comp_score_dev,
                indexer_scatter_rows=idx_comp_rows_dev,
                bsz=int(bsz),
                seqlen=int(all_kv_cseq) if all_kv_bucketed else int(seqlen),
                clen=(
                    int(all_kv_cseq) // int(ratio)
                    if all_kv_bucketed
                    else int(seqlen) // int(ratio)
                ),
                device_state=device_layer_state.compressor,
                indexer_device_state=device_layer_state.indexer,
                window_size=int(win),
                build_dir=build_dir,
                real_seqlen=int(seqlen),
                owner_ids=owner_ids,
            )
            return Dsv4CompressedAttentionQkvResult(
                handled=True,
                q_dev=q_dev,
                kv_dev=kv_dev,
                qr_dev=None,
                topk_t_dev=topk_t_dev,
                mask_dev=mask_dev,
                compressor_state_swa_write_fused=True,
                bucketed_prefill_done=bool(all_kv_bucketed),
                bucketed_kv_primary=kv_dev,
                all_kv_offset=int(all_kv_offset),
                attention_rows=int(attention_rows),
            )
        deferred_indexer_state = None
        can_defer_indexer_prefill_to_main_swa = (
            int(start_pos) == 0
            and not isinstance(comp_rows_dev, np.ndarray)
            and not isinstance(idx_comp_kv_dev, np.ndarray)
            and not isinstance(idx_comp_score_dev, np.ndarray)
            and not isinstance(idx_comp_rows_dev, np.ndarray)
            and hasattr(device_layer_state.indexer.kv_score_state, "tensor_ref")
            and hasattr(device_layer_state.indexer.compressed_kv_cache, "tensor_ref")
        )
        if can_defer_indexer_prefill_to_main_swa:
            deferred_indexer_state = Dsv4DeferredIndexerState(
                compressor=indexer_obj.compressor,
                kv=idx_comp_kv_dev,
                score=idx_comp_score_dev,
                device_state=device_layer_state.indexer,
                prefill_scatter_rows=idx_comp_rows_dev,
            )
        else:
            _run_compressor(
                fns,
                indexer_obj.compressor,
                x,
                start_pos,
                build_dir=build_dir,
                device_state=device_layer_state.indexer,
                owner_ids=owner_ids,
                owner_ids_dev=owner_ids_dev,
                token_positions=device_token_positions,
                attention_scratch=attention_scratch,
                precomputed_kv_score=(idx_comp_kv_dev, idx_comp_score_dev),
                precomputed_prefill_scatter_rows=idx_comp_rows_dev,
            )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            precomputed_compressor_prefill_scatter_rows=comp_rows_dev,
            indexer_precomputed_compressor_kv_score=(
                idx_comp_kv_dev,
                idx_comp_score_dev,
            ),
            deferred_indexer_state=deferred_indexer_state,
            all_kv_offset=int(all_kv_offset),
            attention_rows=int(attention_rows),
        )

    if variant.name == QkvVariantName.INDEXER_ALL_KV_TOPK_PREP_WRITE_SWA_STATE:
        all_kv_offset = int(win)
        spec = device_layer_state.compressor.spec
        idx_spec = device_layer_state.indexer.spec
        idx_comp_ape = getattr(indexer_compressor, "ape")
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        q_dev, kv_dev, topk_t_dev, mask_dev = (
            qkv_indexer_compressor_all_kv_topk_prep_write_swa_state(
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
                device_layer_state.swa_kv_cache,
                device_layer_state.compressor.kv_score_state,
                device_layer_state.indexer.kv_score_state,
                (
                    decode_owner_ids_dev
                    if decode_owner_ids_dev is not None
                    else owner_ids
                ),
                compressor_ape,
                idx_comp_ape,
                freqs_cos,
                freqs_sin,
                (
                    decode_positions_1d
                    if decode_positions_1d is not None
                    else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
                ),
                **_all_kv_kwargs(all_kv_offset),
                compressor_ring_size=int(spec.ring_size),
                indexer_compressor_ring_size=int(idx_spec.ring_size),
            )
        )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            compressor_state_swa_write_fused=True,
            all_kv_offset=int(all_kv_offset),
        )

    if (
        variant.name
        == QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_WRITE_SWA_STATE_CACHE
    ):
        all_kv_offset = _step_all_kv_offset()
        spec = device_layer_state.compressor.spec
        idx_spec = device_layer_state.indexer.spec
        idx_comp_ape = getattr(indexer_compressor, "ape")
        idx_comp_norm = getattr(indexer_compressor, "norm_weight")
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        decode_owner_ids = (
            decode_owner_ids_dev if decode_owner_ids_dev is not None else owner_ids
        )
        decode_end_positions = (
            decode_positions_1d
            if decode_positions_1d is not None
            else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
        )
        q_dev, kv_dev, topk_t_dev, mask_dev = (
            qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep(
                x,
                attn.wq_a,
                attn.q_norm,
                attn.wq_b,
                attn.wkv,
                attn.kv_norm,
                compressor_wkv,
                compressor_wgate,
                device_layer_state.compressor.kv_score_state,
                decode_owner_ids,
                decode_end_positions,
                compressor_ape,
                compressor_norm_weight,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
                device_layer_state.indexer.kv_score_state,
                idx_comp_ape,
                idx_comp_norm,
                freqs_cos,
                freqs_sin,
                compressor_freqs_cos,
                compressor_freqs_sin,
                indexer_freqs_cos,
                indexer_freqs_sin,
                qkv_positions_input,
                **_all_kv_kwargs(all_kv_offset),
                **_compressor_kernel_kwargs(attn.compressor),
                compressor_state_width=int(spec.state_width),
                compressor_ring_size=int(spec.ring_size),
                compressor_overlap=bool(spec.overlap),
                **_compressor_kernel_kwargs(
                    indexer_compressor,
                    prefix="indexer_compressor",
                ),
                indexer_compressor_state_width=int(idx_spec.state_width),
                indexer_compressor_ring_size=int(idx_spec.ring_size),
                indexer_compressor_overlap=bool(idx_spec.overlap),
                write_swa_state_cache=True,
                swa_kv_cache=device_layer_state.swa_kv_cache,
                compressed_kv_cache=device_layer_state.compressor.compressed_kv_cache,
                indexer_compressed_kv_cache=(
                    device_layer_state.indexer.compressed_kv_cache
                ),
                max_c_len=int(spec.max_compressed_len),
                indexer_max_c_len=int(idx_spec.max_compressed_len),
            )
        )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            compressor_state_swa_write_fused=True,
            all_kv_offset=int(all_kv_offset),
        )

    if variant.name == QkvVariantName.INDEXER_ALL_KV_DECODE_POST_QDQ_TOPK_PREP:
        all_kv_offset = _step_all_kv_offset()
        spec = device_layer_state.compressor.spec
        idx_spec = device_layer_state.indexer.spec
        idx_comp_ape = getattr(indexer_compressor, "ape")
        idx_comp_norm = getattr(indexer_compressor, "norm_weight")
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        decode_owner_ids = (
            decode_owner_ids_dev if decode_owner_ids_dev is not None else owner_ids
        )
        decode_end_positions = (
            decode_positions_1d
            if decode_positions_1d is not None
            else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
        )
        (
            q_dev,
            kv_dev,
            comp_kv_dev,
            comp_score_dev,
            idx_comp_kv_dev,
            idx_comp_score_dev,
            topk_t_dev,
            mask_dev,
            comp_rows_dev,
            idx_comp_rows_dev,
        ) = qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            device_layer_state.compressor.kv_score_state,
            decode_owner_ids,
            decode_end_positions,
            compressor_ape,
            compressor_norm_weight,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            device_layer_state.indexer.kv_score_state,
            idx_comp_ape,
            idx_comp_norm,
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            indexer_freqs_cos,
            indexer_freqs_sin,
            qkv_positions_input,
            **_all_kv_kwargs(all_kv_offset),
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_state_width=int(spec.state_width),
            compressor_ring_size=int(spec.ring_size),
            compressor_overlap=bool(spec.overlap),
            **_compressor_kernel_kwargs(
                indexer_compressor, prefix="indexer_compressor"
            ),
            indexer_compressor_state_width=int(idx_spec.state_width),
            indexer_compressor_ring_size=int(idx_spec.ring_size),
            indexer_compressor_overlap=bool(idx_spec.overlap),
        )
        deferred_indexer_state = None
        can_defer_indexer_boundary_to_main_swa = (
            int(start_pos) != 0
            and int(seqlen) == 1
            and int(ratio) > 0
            and (int(start_pos) + 1) % int(ratio) == 0
            and not isinstance(idx_comp_kv_dev, np.ndarray)
            and not isinstance(idx_comp_score_dev, np.ndarray)
            and not isinstance(idx_comp_rows_dev, np.ndarray)
            and hasattr(device_layer_state.indexer.kv_score_state, "tensor_ref")
            and hasattr(device_layer_state.indexer.compressed_kv_cache, "tensor_ref")
        )
        if can_defer_indexer_boundary_to_main_swa:
            deferred_indexer_state = Dsv4DeferredIndexerState(
                compressor=indexer_obj.compressor,
                kv=idx_comp_kv_dev,
                score=idx_comp_score_dev,
                device_state=device_layer_state.indexer,
                decode_scatter_rows=idx_comp_rows_dev,
            )
        else:
            _run_compressor(
                fns,
                indexer_obj.compressor,
                x,
                start_pos,
                build_dir=build_dir,
                device_state=device_layer_state.indexer,
                owner_ids=owner_ids,
                owner_ids_dev=owner_ids_dev,
                token_positions=device_token_positions,
                attention_scratch=attention_scratch,
                precomputed_kv_score=(idx_comp_kv_dev, idx_comp_score_dev),
                precomputed_decode_scatter_rows=idx_comp_rows_dev,
            )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            precomputed_compressor_decode_scatter_rows=comp_rows_dev,
            indexer_precomputed_compressor_kv_score=(
                idx_comp_kv_dev,
                idx_comp_score_dev,
            ),
            indexer_precomputed_compressor_decode_scatter_rows=idx_comp_rows_dev,
            deferred_indexer_state=deferred_indexer_state,
            all_kv_offset=int(all_kv_offset),
        )

    if variant.name == QkvVariantName.INDEXER_ALL_KV_TOPK_PREP:
        all_kv_offset = _step_all_kv_offset()
        (
            q_dev,
            kv_dev,
            comp_kv_dev,
            comp_score_dev,
            idx_comp_kv_dev,
            idx_comp_score_dev,
            topk_t_dev,
            mask_dev,
        ) = qkv_indexer_compressor_all_kv_topk_prep(
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
            **_all_kv_kwargs(all_kv_offset),
        )
        deferred_indexer_state = None
        can_defer_indexer_state_to_main_swa = bool(
            int(start_pos) != 0
            and int(seqlen) == 1
            and int(ratio) > 0
            and (int(start_pos) + 1) % int(ratio) != 0
            and not isinstance(idx_comp_kv_dev, np.ndarray)
            and not isinstance(idx_comp_score_dev, np.ndarray)
            and hasattr(device_layer_state.indexer.kv_score_state, "tensor_ref")
        )
        if can_defer_indexer_state_to_main_swa:
            deferred_indexer_state = Dsv4DeferredIndexerState(
                compressor=indexer_obj.compressor,
                kv=idx_comp_kv_dev,
                score=idx_comp_score_dev,
                device_state=device_layer_state.indexer,
            )
        else:
            _run_compressor(
                fns,
                indexer_obj.compressor,
                x,
                start_pos,
                build_dir=build_dir,
                device_state=device_layer_state.indexer,
                owner_ids=owner_ids,
                owner_ids_dev=owner_ids_dev,
                token_positions=device_token_positions,
                attention_scratch=attention_scratch,
                precomputed_kv_score=(idx_comp_kv_dev, idx_comp_score_dev),
            )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            indexer_precomputed_compressor_kv_score=(
                idx_comp_kv_dev,
                idx_comp_score_dev,
            ),
            deferred_indexer_state=deferred_indexer_state,
            all_kv_offset=int(all_kv_offset),
        )

    return Dsv4CompressedAttentionQkvResult(handled=False)
