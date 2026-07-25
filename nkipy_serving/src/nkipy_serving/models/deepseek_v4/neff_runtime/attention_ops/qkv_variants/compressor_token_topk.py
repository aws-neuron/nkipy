"""Compressor/token-top-k QKV variant runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.constants import K_TILE
from nkipy_serving.models.deepseek_v4.neff_runtime.attention_ops.common import (
    _decode_positions_1d_array,
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


def _compressed_attention_token_topk_kwargs(
    attn: Any,
    *,
    active_bucket: int,
    win: int,
    ratio: int,
    token_topk_offset: int,
    start_pos: int,
    token_topk_max_c_len: int,
    qkv_outputs_flat_kv: bool,
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
        window_size=int(win),
        ratio=int(ratio),
        offset=int(token_topk_offset),
        start_pos=int(start_pos),
        max_c_len=int(token_topk_max_c_len),
        rows=int(active_bucket),
        k_tile=int(K_TILE),
        kv_token_bucket=int(active_bucket) if qkv_outputs_flat_kv else 0,
        return_qr=False,
    )
    if qkv_outputs is not None:
        kwargs["_nkipy_output_tensors"] = qkv_outputs
    return kwargs


def _run_compressed_attention_token_topk_qkv(
    *,
    variant: VariantSpec,
    use_qkv_compressor_prefill_post_qdq_token_topk_bucketed: bool,
    qkv_compressor_token_topk_prep_write_swa_state: Any,
    qkv_compressor_decode_post_qdq_token_topk_prep: Any,
    qkv_compressor_prefill_post_qdq_token_topk_prep: Any,
    qkv_compressor_token_topk_prep: Any,
    qkv_token_topk_prep: Any,
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
    compressor_wkv: Any,
    compressor_wgate: Any,
    compressor_ape: Any,
    compressor_norm_weight: Any,
    qkv_outputs: dict[str, Any | None] | None,
    qkv_outputs_flat_kv: bool,
    active_bucket: int,
    win: int,
    ratio: int,
    token_topk_offset: int,
    start_pos: int,
    token_topk_max_c_len: int,
    bsz: int,
    seqlen: int,
    compressor_prefill_state_tail_len: int,
) -> Dsv4CompressedAttentionQkvResult:
    qkv_kwargs = _compressed_attention_token_topk_kwargs(
        attn,
        active_bucket=int(active_bucket),
        win=int(win),
        ratio=int(ratio),
        token_topk_offset=int(token_topk_offset),
        start_pos=int(start_pos),
        token_topk_max_c_len=int(token_topk_max_c_len),
        qkv_outputs_flat_kv=bool(qkv_outputs_flat_kv),
        qkv_outputs=qkv_outputs,
    )

    if variant.name == QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP_WRITE_SWA_STATE:
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        q_dev, kv_dev, topk_t_dev, mask_dev = (
            qkv_compressor_token_topk_prep_write_swa_state(
                x,
                attn.wq_a,
                attn.q_norm,
                attn.wq_b,
                attn.wkv,
                attn.kv_norm,
                compressor_wkv,
                compressor_wgate,
                device_layer_state.swa_kv_cache,
                device_layer_state.compressor.kv_score_state,
                decode_owner_ids_dev if decode_owner_ids_dev is not None else owner_ids,
                compressor_ape,
                freqs_cos,
                freqs_sin,
                (
                    decode_positions_1d
                    if decode_positions_1d is not None
                    else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
                ),
                **qkv_kwargs,
                compressor_ring_size=int(device_layer_state.compressor.spec.ring_size),
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
            token_topk_offset=int(token_topk_offset),
        )

    if (
        variant.name
        == QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE
    ):
        spec = device_layer_state.compressor.spec
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        q_dev, kv_dev, topk_t_dev, mask_dev = (
            qkv_compressor_decode_post_qdq_token_topk_prep(
                x,
                attn.wq_a,
                attn.q_norm,
                attn.wq_b,
                attn.wkv,
                attn.kv_norm,
                compressor_wkv,
                compressor_wgate,
                device_layer_state.compressor.kv_score_state,
                decode_owner_ids_dev if decode_owner_ids_dev is not None else owner_ids,
                (
                    decode_positions_1d
                    if decode_positions_1d is not None
                    else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
                ),
                compressor_ape,
                compressor_norm_weight,
                freqs_cos,
                freqs_sin,
                compressor_freqs_cos,
                compressor_freqs_sin,
                qkv_positions_input,
                **qkv_kwargs,
                **_compressor_kernel_kwargs(attn.compressor),
                compressor_state_width=int(spec.state_width),
                compressor_ring_size=int(spec.ring_size),
                compressor_overlap=bool(spec.overlap),
                write_swa_state_cache=True,
                compressed_cache_stride=int(spec.max_compressed_len),
                swa_kv_cache=device_layer_state.swa_kv_cache,
                compressed_kv_cache=device_layer_state.compressor.compressed_kv_cache,
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
            token_topk_offset=int(token_topk_offset),
        )

    if variant.name == QkvVariantName.COMPRESSOR_DECODE_POST_QDQ_TOKEN_TOPK_PREP:
        spec = device_layer_state.compressor.spec
        decode_owner_ids_dev, decode_positions_1d = _decode_owner_pos_aliases(
            bsz=int(bsz),
            owner_ids_dev=owner_ids_dev,
            device_token_positions=device_token_positions,
        )
        (
            q_dev,
            kv_dev,
            topk_t_dev,
            mask_dev,
            comp_kv_dev,
            comp_score_dev,
            comp_rows_dev,
        ) = qkv_compressor_decode_post_qdq_token_topk_prep(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            compressor_wkv,
            compressor_wgate,
            device_layer_state.compressor.kv_score_state,
            decode_owner_ids_dev if decode_owner_ids_dev is not None else owner_ids,
            (
                decode_positions_1d
                if decode_positions_1d is not None
                else _decode_positions_1d_array(qkv_positions, bsz=int(bsz))
            ),
            compressor_ape,
            compressor_norm_weight,
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            qkv_positions_input,
            **qkv_kwargs,
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_state_width=int(spec.state_width),
            compressor_ring_size=int(spec.ring_size),
            compressor_overlap=bool(spec.overlap),
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
            token_topk_offset=int(token_topk_offset),
        )

    if use_qkv_compressor_prefill_post_qdq_token_topk_bucketed:
        from nkipy_serving.ops.deepseek_v4.compressor_state import (
            run_compressor_prefill_state_cache_swa_scatter_device as _bkt_single_scatter,
        )

        (
            q_dev,
            kv_dev,
            topk_t_dev,
            mask_dev,
            _comp_kv_dev_b,
            _comp_score_dev_b,
            comp_rows_dev_b,
        ) = qkv_compressor_prefill_post_qdq_token_topk_prep(
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
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            qkv_positions_input,
            **qkv_kwargs,
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_overlap=bool(attn.compressor.overlap),
            write_swa_state_cache=False,
        )
        token_topk_offset, tt_bucketed, tt_cseq = (
            qkv_compressor_prefill_post_qdq_token_topk_prep.__self__._product_last_qkv_compiled_offset
        )
        _bkt_single_scatter(
            attn.compressor,
            swa_kv_cache=device_layer_state.swa_kv_cache,
            swa_rows=kv_dev,
            swa_start_pos=int(start_pos),
            swa_bsz=int(bsz),
            swa_seqlen=int(seqlen),
            kv=_comp_kv_dev_b,
            score=_comp_score_dev_b,
            scatter_rows=comp_rows_dev_b,
            bsz=int(bsz),
            seqlen=int(tt_cseq) if tt_bucketed else int(seqlen),
            clen=(
                int(tt_cseq) // int(ratio) if tt_bucketed else int(seqlen) // int(ratio)
            ),
            device_state=device_layer_state.compressor,
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
            bucketed_prefill_done=bool(tt_bucketed),
            bucketed_kv_primary=kv_dev,
            token_topk_offset=int(token_topk_offset),
        )

    if (
        variant.name
        == QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_WRITE_SWA_STATE_CACHE
    ):
        spec = device_layer_state.compressor.spec
        (
            q_dev,
            kv_dev,
            topk_t_dev,
            mask_dev,
            _comp_kv_dev,
            _comp_score_dev,
            _comp_rows_dev,
        ) = qkv_compressor_prefill_post_qdq_token_topk_prep(
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
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            qkv_positions_input,
            **qkv_kwargs,
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_overlap=bool(spec.overlap),
            write_swa_state_cache=True,
            swa_kv_cache=device_layer_state.swa_kv_cache,
            kv_score_state=device_layer_state.compressor.kv_score_state,
            compressed_kv_cache=device_layer_state.compressor.compressed_kv_cache,
            owner_ids=owner_ids_dev if owner_ids_dev is not None else owner_ids,
            compressor_ring_size=int(spec.ring_size),
            compressor_state_tail_len=int(compressor_prefill_state_tail_len),
        )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            compressor_state_swa_write_fused=True,
            token_topk_offset=int(token_topk_offset),
        )

    if variant.name == QkvVariantName.COMPRESSOR_PREFILL_POST_QDQ_TOKEN_TOPK_PREP:
        (
            q_dev,
            kv_dev,
            topk_t_dev,
            mask_dev,
            comp_kv_dev,
            comp_score_dev,
            comp_rows_dev,
        ) = qkv_compressor_prefill_post_qdq_token_topk_prep(
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
            freqs_cos,
            freqs_sin,
            compressor_freqs_cos,
            compressor_freqs_sin,
            qkv_positions_input,
            **qkv_kwargs,
            **_compressor_kernel_kwargs(attn.compressor),
            compressor_overlap=bool(attn.compressor.overlap),
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
            token_topk_offset=int(token_topk_offset),
        )

    if variant.name == QkvVariantName.COMPRESSOR_TOKEN_TOPK_PREP:
        q_dev, kv_dev, topk_t_dev, mask_dev, comp_kv_dev, comp_score_dev = (
            qkv_compressor_token_topk_prep(
                x,
                attn.wq_a,
                attn.q_norm,
                attn.wq_b,
                attn.wkv,
                attn.kv_norm,
                compressor_wkv,
                compressor_wgate,
                freqs_cos,
                freqs_sin,
                qkv_positions_input,
                **qkv_kwargs,
            )
        )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=None,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            precomputed_compressor_kv_score=(comp_kv_dev, comp_score_dev),
            token_topk_offset=int(token_topk_offset),
        )

    if variant.name == QkvVariantName.TOKEN_TOPK_PREP:
        q_dev, kv_dev, qr_dev, topk_t_dev, mask_dev = qkv_token_topk_prep(
            x,
            attn.wq_a,
            attn.q_norm,
            attn.wq_b,
            attn.wkv,
            attn.kv_norm,
            freqs_cos,
            freqs_sin,
            qkv_positions_input,
            **qkv_kwargs,
        )
        return Dsv4CompressedAttentionQkvResult(
            handled=True,
            q_dev=q_dev,
            kv_dev=kv_dev,
            qr_dev=qr_dev,
            topk_t_dev=topk_t_dev,
            mask_dev=mask_dev,
            token_topk_offset=int(token_topk_offset),
        )

    return Dsv4CompressedAttentionQkvResult(
        handled=False,
        token_topk_offset=int(token_topk_offset),
    )
