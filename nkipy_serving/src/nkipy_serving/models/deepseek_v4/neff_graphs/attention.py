"""DeepSeek-V4 attention QKV/output trace functions.

Merged from attention_qkv + attention_out. Pure HLO-traceable; bodies byte-identical
to pre-merge. Imports cross-cutting primitives from graphs.common only."""

from __future__ import annotations

import numpy as np

from nkipy_serving.models.deepseek_v4.neff_graphs.common import (
    _all_reduce_last_dim_preserve_shape,
    _apply_interleaved_rope,
    _linear_out_in,
    compressor_decode_pool_post_qdq_from_state_plus_current_freq_table_fn,
    compressor_kv_score_bf16_fn,
    compressor_kv_score_token_topk_prep_decode_from_positions_fn,
    compressor_kv_score_token_topk_prep_fn,
    compressor_prefill_post_qdq_from_token_slabs_fn,
    indexer_compressor_kv_score_project_qw_prep_from_freq_table_fn,
    indexer_sparse_attention_prep_all_kv_decode_from_positions_fn,
    indexer_sparse_attention_prep_all_kv_static_fn,
    kv_rope_quant_fn,
    mhc_post_pre_fn,
    pad_flat_rows_fn,
    topk_tokens_concat_pad_sparse_attention_prep_decode_from_positions_fn,
    topk_tokens_concat_pad_sparse_attention_prep_fn,
)
from nkipy_serving.ops.nn import apply_rms_norm


def attention_kv_flatten_fn(
    kv: np.ndarray,  # [bsz, seqlen, d]
    *,
    total_tokens: int,
    head_dim: int,
) -> np.ndarray:
    """Flatten attention KV rows to the backend scatter layout."""
    return np.reshape(kv, (int(total_tokens), int(head_dim)))


def q_scale_transpose_fn(
    q: np.ndarray,  # [bsz, seqlen, h, d]  fp32 from attention_qkv_quant
    *,
    softmax_scale: float,
    token_bucket: int,
) -> np.ndarray:
    """Pre-scale + bf16 cast + transpose to the sparse-attention kernel layout.

    The paged-sparse-attention kernels want ``q_scaled_t [N_q, d, h]``
    bf16 where ``N_q == token_bucket`` (static-shape kernel). Folding this into
    a device fragment lets the q produced by ``attention_qkv_quant`` chain
    directly into attention without a download.

    Input ``q`` is ``[bsz, seqlen, h, d]`` fp32. ``bsz * seqlen`` must be
    ``<= token_bucket``; the trailing rows are zero-padded so the output
    shape is fixed per bucket (one NEFF per bucket instead of per seqlen).
    """
    import ml_dtypes as _ml

    bsz, seqlen, h, d = q.shape
    n_tokens = int(bsz) * int(seqlen)
    bucket = int(token_bucket)
    if n_tokens > bucket:
        raise ValueError(f"q has {n_tokens} tokens but token_bucket={bucket}")
    scaled = (q.astype(np.float32) * np.float32(softmax_scale)).reshape(
        n_tokens,
        int(h),
        int(d),
    )
    if n_tokens < bucket:
        pad = np.zeros((bucket - n_tokens, int(h), int(d)), dtype=np.float32)
        scaled = np.concatenate((scaled, pad), axis=0)
    return scaled.astype(_ml.bfloat16).transpose(0, 2, 1)


def attention_qkv_proj_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bsz, seqlen, _ = x.shape
    q_low = _linear_out_in(x, wq_a)
    qr = apply_rms_norm(q_low, q_norm, eps=eps)
    q = _linear_out_in(qr, wq_b).reshape(bsz, seqlen, int(n_heads), int(head_dim))
    inv_rms = np.float32(1.0) / np.sqrt(
        np.mean(q * q, axis=-1, keepdims=True) + np.float32(eps)
    )
    q = q * inv_rms
    rd = int(rope_head_dim)
    q = np.concatenate(
        (
            q[..., : int(head_dim) - rd],
            _apply_interleaved_rope(q[..., int(head_dim) - rd :], cos, sin),
        ),
        axis=-1,
    )

    kv = _linear_out_in(x, wkv)
    kv = apply_rms_norm(kv, kv_norm, eps=eps)
    kv = np.concatenate(
        (
            kv[..., : int(head_dim) - rd],
            _apply_interleaved_rope(kv[..., int(head_dim) - rd :], cos, sin),
        ),
        axis=-1,
    )
    return q, kv, qr


def attention_qkv_quant_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QKV projection + RoPE + FP8 qdq on KV non-rope legs, one fragment.

    Trace-time composition of ``attention_qkv_proj_fn`` followed by
    ``kv_rope_quant_fn`` keeps all intermediates on device.
    """
    q, kv, qr = attention_qkv_proj_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos,
        sin,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        eps=eps,
    )
    kv_q = kv_rope_quant_fn(
        kv,
        rope_head_dim=rope_head_dim,
        block_size=block_size,
        fp8_max=fp8_max,
    )
    return q, kv_q, qr


def attention_qkv_quant_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QKV projection/quant using device-resident RoPE tables."""
    bsz, seqlen, _ = x.shape
    pos = positions.astype(np.int32).reshape(-1)
    n_tokens = int(bsz) * int(seqlen)
    if int(pos.shape[0]) > n_tokens:
        pos = pos[:n_tokens]
    if int(pos.shape[0]) == int(seqlen):
        pos = pos.reshape(int(seqlen))
    else:
        pos = pos.reshape(int(bsz), int(seqlen))
    cos = cos_table[pos]
    sin = sin_table[pos]
    return attention_qkv_quant_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos,
        sin,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
    )


def attention_qkv_quant_scaled_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QKV projection/quant with attention-ready scaled/transposed Q output."""
    q, kv, qr = attention_qkv_quant_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
    )
    return (
        q_scale_transpose_fn(
            q,
            softmax_scale=float(q_softmax_scale),
            token_bucket=int(q_token_bucket),
        ),
        kv,
        qr,
    )


def attention_qkv_quant_scaled_kvflat_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QKV table path with attention-ready Q and SWA cache-ready flat KV."""
    q_scaled, kv, qr = attention_qkv_quant_scaled_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
    )
    kv_flat = attention_kv_flatten_fn(
        kv,
        total_tokens=int(x.shape[0]) * int(x.shape[1]),
        head_dim=int(head_dim),
    )
    return (
        q_scaled,
        pad_flat_rows_fn(kv_flat, rows=int(kv_token_bucket)),
        qr,
    )


def attention_qkv_quant_scaled_kvflat_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SWA-attention QKV prologue without materializing the unused QR output."""
    q_scaled, kv, _qr = attention_qkv_quant_scaled_kvflat_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
    )
    return q_scaled, kv


def attention_qkv_quant_scaled_kv_cache_write_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    kv_cache: np.ndarray,
    slot_mapping: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
) -> np.ndarray:
    """SWA-only QKV prologue that also scatters flat KV into the paged cache."""
    q_scaled, kv, _qr = attention_qkv_quant_scaled_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
    )
    real_tokens = int(x.shape[0]) * int(x.shape[1])
    kv_flat = attention_kv_flatten_fn(
        kv,
        total_tokens=real_tokens,
        head_dim=int(head_dim),
    )
    slots = slot_mapping.astype(np.int32).reshape(-1)
    if int(slots.shape[0]) > real_tokens:
        slots = slots[:real_tokens]
    kv_cache[slots] = kv_flat.astype(kv_cache.dtype)
    return q_scaled


def attention_qkv_compressor_kv_score_scaled_kvflat_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compressed-attention QKV prologue plus compressor input projection."""
    q_scaled, kv, qr = attention_qkv_quant_scaled_kvflat_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
    )
    comp_kv, comp_score = compressor_kv_score_bf16_fn(
        x,
        compressor_wkv,
        compressor_wgate,
    )
    return q_scaled, kv, qr, comp_kv, comp_score


def attention_qkv_indexer_compressor_qw_prep_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    indexer_wq_b: np.ndarray,
    indexer_weights_proj: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    indexer_cos_table: np.ndarray,
    indexer_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    indexer_score_scale: float,
    indexer_n_heads: int,
    indexer_head_dim: int,
    indexer_rope_head_dim: int,
    indexer_block_size: int,
    indexer_fp8_max: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compressed-attention and indexer prologue without exposing QR."""
    q_scaled, kv, qr, comp_kv, comp_score = (
        attention_qkv_compressor_kv_score_scaled_kvflat_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    idx_kv, idx_score, idx_q_t, idx_w_flat = (
        indexer_compressor_kv_score_project_qw_prep_from_freq_table_fn(
            x,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            qr,
            indexer_wq_b,
            indexer_weights_proj,
            indexer_cos_table,
            indexer_sin_table,
            positions,
            score_scale=float(indexer_score_scale),
            n_heads=int(indexer_n_heads),
            head_dim=int(indexer_head_dim),
            rope_head_dim=int(indexer_rope_head_dim),
            block_size=int(indexer_block_size),
            fp8_max=float(indexer_fp8_max),
        )
    )
    return (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        idx_q_t,
        idx_w_flat,
    )


def attention_qkv_indexer_compressor_qw_prep_write_swa_dual_state_decode_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    indexer_wq_b: np.ndarray,
    indexer_weights_proj: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    indexer_kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    compressor_ape: np.ndarray,
    indexer_compressor_ape: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    indexer_cos_table: np.ndarray,
    indexer_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    indexer_score_scale: float,
    indexer_n_heads: int,
    indexer_head_dim: int,
    indexer_rope_head_dim: int,
    indexer_block_size: int,
    indexer_fp8_max: float,
    window_size: int,
    ratio: int,
    start_pos: int,
    compressor_ring_size: int,
    indexer_compressor_ring_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode table prologue that also writes SWA and dual ring state."""
    if int(start_pos) <= 0 or int(x.shape[1]) != 1:
        raise RuntimeError("table dual SWA/state write fusion requires decode x")
    (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        idx_q_t,
        idx_w_flat,
    ) = attention_qkv_indexer_compressor_qw_prep_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        indexer_wq_b,
        indexer_weights_proj,
        cos_table,
        sin_table,
        indexer_cos_table,
        indexer_sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
        indexer_score_scale=float(indexer_score_scale),
        indexer_n_heads=int(indexer_n_heads),
        indexer_head_dim=int(indexer_head_dim),
        indexer_rope_head_dim=int(indexer_rope_head_dim),
        indexer_block_size=int(indexer_block_size),
        indexer_fp8_max=float(indexer_fp8_max),
    )
    n_new = int(x.shape[0]) * int(x.shape[1])
    kv_rows = kv[:n_new]
    owners = owner_ids.astype(np.int32).reshape(-1)[:n_new]
    pos = positions.astype(np.int32).reshape(-1)[:n_new]

    swa_offsets = pos - (
        (pos.astype(np.float32) / np.float32(window_size)).astype(np.int32)
        * np.int32(window_size)
    )
    swa_rows = owners * np.int32(window_size) + swa_offsets
    swa_kv_cache[swa_rows] = kv_rows.astype(swa_kv_cache.dtype)

    ratio_i = int(ratio)
    ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
        * np.int32(ratio_i)
    )

    ring_i = int(compressor_ring_size)
    ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        * np.int32(ring_i)
    )
    state_rows = owners * np.int32(ring_i) + ring_offsets
    width = int(comp_kv.shape[-1])
    comp_kv_rows = comp_kv.astype(np.float32).reshape(n_new, width)
    comp_score_rows = comp_score.astype(np.float32).reshape(n_new, width)
    ape_rows = compressor_ape[ape_offsets].astype(np.float32).reshape(n_new, width)
    kv_score_state[state_rows, :width] = comp_kv_rows.astype(kv_score_state.dtype)
    kv_score_state[state_rows, width : 2 * width] = (comp_score_rows + ape_rows).astype(
        kv_score_state.dtype
    )

    indexer_ring_i = int(indexer_compressor_ring_size)
    indexer_ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(indexer_ring_i)).astype(np.int32)
        * np.int32(indexer_ring_i)
    )
    indexer_state_rows = owners * np.int32(indexer_ring_i) + indexer_ring_offsets
    indexer_ratio_i = int(indexer_compressor_ape.shape[0])
    indexer_ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(indexer_ratio_i)).astype(np.int32)
        * np.int32(indexer_ratio_i)
    )
    indexer_width = int(idx_kv.shape[-1])
    idx_kv_rows = idx_kv.astype(np.float32).reshape(n_new, indexer_width)
    idx_score_rows = idx_score.astype(np.float32).reshape(n_new, indexer_width)
    idx_ape_rows = (
        indexer_compressor_ape[indexer_ape_offsets]
        .astype(np.float32)
        .reshape(n_new, indexer_width)
    )
    indexer_kv_score_state[indexer_state_rows, :indexer_width] = idx_kv_rows.astype(
        indexer_kv_score_state.dtype
    )
    indexer_kv_score_state[
        indexer_state_rows,
        indexer_width : 2 * indexer_width,
    ] = (idx_score_rows + idx_ape_rows).astype(indexer_kv_score_state.dtype)
    return q_scaled, kv, idx_q_t, idx_w_flat


def attention_qkv_indexer_compressor_all_kv_topk_prep_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Attention/indexer prologue for the ``index_topk >= kv_len`` path.

    In this bucket the indexer score values do not affect the selected
    compressed KV set because every compressed row is selected. Fuse QKV,
    both compressor input projections, and sparse-attention prep without
    materializing indexer Q/W score inputs.
    """
    q_scaled, kv = attention_qkv_quant_scaled_kvflat_no_qr_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
    )
    comp_kv, comp_score = compressor_kv_score_bf16_fn(
        x,
        compressor_wkv,
        compressor_wgate,
    )
    idx_kv, idx_score = compressor_kv_score_bf16_fn(
        x,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
    )
    topk_t, mask = indexer_sparse_attention_prep_all_kv_static_fn(
        x,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        kv_len=int(kv_len),
        k=int(k),
        ratio=int(ratio),
        offset=int(offset),
        prefill=int(start_pos) == 0,
        window_size=int(window_size),
        start_pos=int(start_pos),
        rows=int(rows),
        k_tile=int(k_tile),
    )
    return q_scaled, kv, comp_kv, comp_score, idx_kv, idx_score, topk_t, mask


def attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    indexer_compressor_ape: np.ndarray,
    indexer_compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    indexer_compressor_cos_table: np.ndarray,
    indexer_compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
    compressor_head_dim: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
    indexer_compressor_head_dim: int,
    indexer_compressor_rope_head_dim: int,
    indexer_compressor_block_size: int,
    indexer_compressor_fp8_max: float,
    indexer_compressor_rotate: bool,
    indexer_compressor_overlap: bool,
    indexer_compressor_eps: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """All-KV indexer prefill prologue plus both compressor post-QDQ tails."""
    if int(start_pos) != 0:
        raise RuntimeError("prefill compressor post-QDQ fusion requires start_pos=0")
    (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
    ) = attention_qkv_indexer_compressor_all_kv_topk_prep_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        kv_len=int(kv_len),
        k=int(k),
        rows=int(rows),
        k_tile=int(k_tile),
    )
    cutoff = int(x.shape[1]) - (int(x.shape[1]) % int(ratio))
    comp_rows = compressor_prefill_post_qdq_from_token_slabs_fn(
        comp_kv,
        comp_score,
        compressor_ape,
        compressor_norm_weight,
        compressor_cos_table,
        compressor_sin_table,
        positions,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        cutoff=int(cutoff),
        ratio=int(ratio),
        head_dim=int(compressor_head_dim),
        rope_head_dim=int(compressor_rope_head_dim),
        block_size=int(compressor_block_size),
        fp8_max=float(compressor_fp8_max),
        rotate=bool(compressor_rotate),
        overlap=bool(compressor_overlap),
        eps=float(compressor_eps),
    )
    idx_rows = compressor_prefill_post_qdq_from_token_slabs_fn(
        idx_kv,
        idx_score,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        indexer_compressor_cos_table,
        indexer_compressor_sin_table,
        positions,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        cutoff=int(cutoff),
        ratio=int(ratio),
        head_dim=int(indexer_compressor_head_dim),
        rope_head_dim=int(indexer_compressor_rope_head_dim),
        block_size=int(indexer_compressor_block_size),
        fp8_max=float(indexer_compressor_fp8_max),
        rotate=bool(indexer_compressor_rotate),
        overlap=bool(indexer_compressor_overlap),
        eps=float(indexer_compressor_eps),
    )
    return (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
        comp_rows,
        idx_rows,
    )


def _mod_i32(values: np.ndarray, modulus: int) -> np.ndarray:
    modulus_i = np.int32(modulus)
    return values - (
        (values.astype(np.float32) / np.float32(modulus_i)).astype(np.int32) * modulus_i
    )


def _write_prefill_swa_state_cache_rows_fn(
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    compressed_kv_cache: np.ndarray,
    kv_rows: np.ndarray,
    comp_kv: np.ndarray,
    comp_score: np.ndarray,
    comp_rows: np.ndarray,
    owner_ids: np.ndarray,
    positions: np.ndarray,
    ape: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    head_dim: int,
    window_size: int,
    ring_size: int,
    state_tail_len: int,
    max_c_len: int,
) -> None:
    """Scatter prefill SWA, compressor tail state, and compressed rows."""
    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    n_tokens = bsz_i * seqlen_i
    head_dim_i = int(head_dim)
    kv_active = kv_rows[:n_tokens].astype(np.float32)
    owners = owner_ids.astype(np.int32).reshape(-1)[:n_tokens]
    pos = positions.astype(np.int32).reshape(-1)[:n_tokens]

    kv_rect = kv_active.reshape(bsz_i, seqlen_i, head_dim_i)
    owner_rect = owners.reshape(bsz_i, seqlen_i)
    pos_rect = pos.reshape(bsz_i, seqlen_i)

    swa_len = min(seqlen_i, int(window_size))
    swa_kv = kv_rect[:, seqlen_i - swa_len :, :].reshape(
        bsz_i * swa_len,
        head_dim_i,
    )
    swa_owners = owner_rect[:, seqlen_i - swa_len :].reshape(-1)
    swa_pos = pos_rect[:, seqlen_i - swa_len :].reshape(-1)
    swa_offsets = _mod_i32(swa_pos, int(window_size))
    swa_rows = swa_owners * np.int32(window_size) + swa_offsets
    swa_kv_cache[swa_rows] = swa_kv.astype(swa_kv_cache.dtype)

    keep = int(state_tail_len)
    if keep > 0:
        width = int(comp_kv.shape[-1])
        state_kv = (
            comp_kv.astype(np.float32)
            .reshape(
                bsz_i,
                seqlen_i,
                width,
            )[:, seqlen_i - keep :, :]
            .reshape(bsz_i * keep, width)
        )
        state_score = (
            comp_score.astype(np.float32)
            .reshape(
                bsz_i,
                seqlen_i,
                width,
            )[:, seqlen_i - keep :, :]
            .reshape(bsz_i * keep, width)
        )
        state_owners = owner_rect[:, seqlen_i - keep :].reshape(-1)
        state_pos = pos_rect[:, seqlen_i - keep :].reshape(-1)
        ring_offsets = _mod_i32(state_pos, int(ring_size))
        state_rows = state_owners * np.int32(ring_size) + ring_offsets
        ape_offsets = _mod_i32(state_pos, int(ape.shape[0]))
        ape_rows = ape[ape_offsets].astype(np.float32).reshape(bsz_i * keep, width)
        kv_score_state[state_rows, :width] = state_kv.astype(kv_score_state.dtype)
        kv_score_state[state_rows, width : 2 * width] = (state_score + ape_rows).astype(
            kv_score_state.dtype
        )

    clen = int(comp_rows.shape[0]) // bsz_i
    cache_owners = owner_rect[:, 0:1]
    cache_cols = np.arange(clen, dtype=np.int32).reshape(1, clen)
    cache_rows = (cache_owners * np.int32(max_c_len) + cache_cols).reshape(-1)
    compressed_kv_cache[cache_rows] = comp_rows.astype(compressed_kv_cache.dtype)


def attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_dual_state_cache_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    compressed_kv_cache: np.ndarray,
    owner_ids: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    indexer_kv_score_state: np.ndarray,
    indexer_compressed_kv_cache: np.ndarray,
    indexer_compressor_ape: np.ndarray,
    indexer_compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    indexer_compressor_cos_table: np.ndarray,
    indexer_compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
    compressor_head_dim: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
    compressor_ring_size: int,
    compressor_state_tail_len: int,
    max_c_len: int,
    indexer_compressor_head_dim: int,
    indexer_compressor_rope_head_dim: int,
    indexer_compressor_block_size: int,
    indexer_compressor_fp8_max: float,
    indexer_compressor_rotate: bool,
    indexer_compressor_overlap: bool,
    indexer_compressor_eps: float,
    indexer_compressor_ring_size: int,
    indexer_compressor_state_tail_len: int,
    indexer_max_c_len: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """All-KV prefill prologue plus direct SWA/dual-state/cache writes."""
    (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
        comp_rows,
        idx_rows,
    ) = attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        compressor_ape,
        compressor_norm_weight,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        cos_table,
        sin_table,
        compressor_cos_table,
        compressor_sin_table,
        indexer_compressor_cos_table,
        indexer_compressor_sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        kv_len=int(kv_len),
        k=int(k),
        rows=int(rows),
        k_tile=int(k_tile),
        compressor_head_dim=int(compressor_head_dim),
        compressor_rope_head_dim=int(compressor_rope_head_dim),
        compressor_block_size=int(compressor_block_size),
        compressor_fp8_max=float(compressor_fp8_max),
        compressor_rotate=bool(compressor_rotate),
        compressor_overlap=bool(compressor_overlap),
        compressor_eps=float(compressor_eps),
        indexer_compressor_head_dim=int(indexer_compressor_head_dim),
        indexer_compressor_rope_head_dim=int(indexer_compressor_rope_head_dim),
        indexer_compressor_block_size=int(indexer_compressor_block_size),
        indexer_compressor_fp8_max=float(indexer_compressor_fp8_max),
        indexer_compressor_rotate=bool(indexer_compressor_rotate),
        indexer_compressor_overlap=bool(indexer_compressor_overlap),
        indexer_compressor_eps=float(indexer_compressor_eps),
    )
    _write_prefill_swa_state_cache_rows_fn(
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        kv,
        comp_kv,
        comp_score,
        comp_rows,
        owner_ids,
        positions,
        compressor_ape,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        head_dim=int(head_dim),
        window_size=int(window_size),
        ring_size=int(compressor_ring_size),
        state_tail_len=int(compressor_state_tail_len),
        max_c_len=int(max_c_len),
    )
    _write_prefill_swa_state_cache_rows_fn(
        swa_kv_cache,
        indexer_kv_score_state,
        indexer_compressed_kv_cache,
        kv,
        idx_kv,
        idx_score,
        idx_rows,
        owner_ids,
        positions,
        indexer_compressor_ape,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        head_dim=int(head_dim),
        window_size=int(window_size),
        ring_size=int(indexer_compressor_ring_size),
        state_tail_len=int(indexer_compressor_state_tail_len),
        max_c_len=int(indexer_max_c_len),
    )
    return (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
        comp_rows,
        idx_rows,
    )


def attention_qkv_indexer_compressor_all_kv_topk_prep_decode_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Decode variant using runtime positions for the SWA window tail."""
    del start_pos
    if int(x.shape[1]) != 1:
        raise RuntimeError("all-KV dynamic-position indexer prep is decode-only")
    q_scaled, kv = attention_qkv_quant_scaled_kvflat_no_qr_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
    )
    comp_kv, comp_score = compressor_kv_score_bf16_fn(
        x,
        compressor_wkv,
        compressor_wgate,
    )
    idx_kv, idx_score = compressor_kv_score_bf16_fn(
        x,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
    )
    topk_t, mask = indexer_sparse_attention_prep_all_kv_decode_from_positions_fn(
        x,
        positions,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        kv_len=int(kv_len),
        k=int(k),
        ratio=int(ratio),
        offset=int(offset),
        prefill=False,
        window_size=int(window_size),
        start_pos=0,
        rows=int(rows),
        k_tile=int(k_tile),
    )
    return q_scaled, kv, comp_kv, comp_score, idx_kv, idx_score, topk_t, mask


def attention_qkv_indexer_compressor_all_kv_topk_write_swa_dual_state_decode_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    indexer_kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    compressor_ape: np.ndarray,
    indexer_compressor_ape: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
    compressor_ring_size: int,
    indexer_compressor_ring_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All-KV decode prologue that also writes SWA and dual ring state."""
    if int(start_pos) <= 0 or int(x.shape[1]) != 1:
        raise RuntimeError("all-KV dual SWA/state write fusion requires decode x")
    (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
    ) = attention_qkv_indexer_compressor_all_kv_topk_prep_decode_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        kv_len=int(kv_len),
        k=int(k),
        rows=int(rows),
        k_tile=int(k_tile),
    )
    n_new = int(x.shape[0]) * int(x.shape[1])
    kv_rows = kv[:n_new]
    owners = owner_ids.astype(np.int32).reshape(-1)[:n_new]
    pos = positions.astype(np.int32).reshape(-1)[:n_new]

    swa_offsets = pos - (
        (pos.astype(np.float32) / np.float32(window_size)).astype(np.int32)
        * np.int32(window_size)
    )
    swa_rows = owners * np.int32(window_size) + swa_offsets
    swa_kv_cache[swa_rows] = kv_rows.astype(swa_kv_cache.dtype)

    ratio_i = int(ratio)
    ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
        * np.int32(ratio_i)
    )

    ring_i = int(compressor_ring_size)
    ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        * np.int32(ring_i)
    )
    state_rows = owners * np.int32(ring_i) + ring_offsets
    width = int(comp_kv.shape[-1])
    comp_kv_rows = comp_kv.astype(np.float32).reshape(n_new, width)
    comp_score_rows = comp_score.astype(np.float32).reshape(n_new, width)
    ape_rows = compressor_ape[ape_offsets].astype(np.float32).reshape(n_new, width)
    kv_score_state[state_rows, :width] = comp_kv_rows.astype(kv_score_state.dtype)
    kv_score_state[state_rows, width : 2 * width] = (comp_score_rows + ape_rows).astype(
        kv_score_state.dtype
    )

    indexer_ring_i = int(indexer_compressor_ring_size)
    indexer_ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(indexer_ring_i)).astype(np.int32)
        * np.int32(indexer_ring_i)
    )
    indexer_state_rows = owners * np.int32(indexer_ring_i) + indexer_ring_offsets
    indexer_ratio_i = int(indexer_compressor_ape.shape[0])
    indexer_ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(indexer_ratio_i)).astype(np.int32)
        * np.int32(indexer_ratio_i)
    )
    indexer_width = int(idx_kv.shape[-1])
    idx_kv_rows = idx_kv.astype(np.float32).reshape(n_new, indexer_width)
    idx_score_rows = idx_score.astype(np.float32).reshape(n_new, indexer_width)
    idx_ape_rows = (
        indexer_compressor_ape[indexer_ape_offsets]
        .astype(np.float32)
        .reshape(n_new, indexer_width)
    )
    indexer_kv_score_state[indexer_state_rows, :indexer_width] = idx_kv_rows.astype(
        indexer_kv_score_state.dtype
    )
    indexer_kv_score_state[
        indexer_state_rows,
        indexer_width : 2 * indexer_width,
    ] = (idx_score_rows + idx_ape_rows).astype(indexer_kv_score_state.dtype)
    return q_scaled, kv, topk_t, mask


def attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    indexer_kv_score_state: np.ndarray,
    indexer_compressor_ape: np.ndarray,
    indexer_compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    indexer_compressor_cos_table: np.ndarray,
    indexer_compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
    compressor_head_dim: int,
    compressor_state_width: int,
    compressor_ring_size: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
    indexer_compressor_head_dim: int,
    indexer_compressor_state_width: int,
    indexer_compressor_ring_size: int,
    indexer_compressor_rope_head_dim: int,
    indexer_compressor_block_size: int,
    indexer_compressor_fp8_max: float,
    indexer_compressor_rotate: bool,
    indexer_compressor_overlap: bool,
    indexer_compressor_eps: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """All-KV indexer decode prologue plus both current-row post-QDQ tails."""
    if int(start_pos) <= 0 or int(x.shape[1]) != 1:
        raise RuntimeError("all-KV decode compressor post-QDQ fusion requires decode x")
    (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
    ) = attention_qkv_indexer_compressor_all_kv_topk_prep_decode_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        kv_len=int(kv_len),
        k=int(k),
        rows=int(rows),
        k_tile=int(k_tile),
    )
    comp_rows = compressor_decode_pool_post_qdq_from_state_plus_current_freq_table_fn(
        kv_score_state,
        comp_kv,
        comp_score,
        owner_ids,
        end_positions,
        compressor_ape,
        compressor_norm_weight,
        compressor_cos_table,
        compressor_sin_table,
        end_positions,
        bsz=int(x.shape[0]),
        ratio=int(ratio),
        head_dim=int(compressor_head_dim),
        state_width=int(compressor_state_width),
        ring_size=int(compressor_ring_size),
        overlap=bool(compressor_overlap),
        source_token_positions=True,
        compress_ratio=int(ratio),
        start_pos=1,
        seqlen=1,
        rope_head_dim=int(compressor_rope_head_dim),
        block_size=int(compressor_block_size),
        fp8_max=float(compressor_fp8_max),
        rotate=bool(compressor_rotate),
        eps=float(compressor_eps),
    )
    idx_rows = compressor_decode_pool_post_qdq_from_state_plus_current_freq_table_fn(
        indexer_kv_score_state,
        idx_kv,
        idx_score,
        owner_ids,
        end_positions,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        indexer_compressor_cos_table,
        indexer_compressor_sin_table,
        positions,
        bsz=int(x.shape[0]),
        ratio=int(ratio),
        head_dim=int(indexer_compressor_head_dim),
        state_width=int(indexer_compressor_state_width),
        ring_size=int(indexer_compressor_ring_size),
        overlap=bool(indexer_compressor_overlap),
        source_token_positions=True,
        compress_ratio=int(ratio),
        start_pos=1,
        seqlen=1,
        rope_head_dim=int(indexer_compressor_rope_head_dim),
        block_size=int(indexer_compressor_block_size),
        fp8_max=float(indexer_compressor_fp8_max),
        rotate=bool(indexer_compressor_rotate),
        eps=float(indexer_compressor_eps),
    )
    return (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
        comp_rows,
        idx_rows,
    )


def attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_dual_state_cache_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    compressed_kv_cache: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    indexer_kv_score_state: np.ndarray,
    indexer_compressed_kv_cache: np.ndarray,
    indexer_compressor_ape: np.ndarray,
    indexer_compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    indexer_compressor_cos_table: np.ndarray,
    indexer_compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    kv_len: int,
    k: int,
    rows: int,
    k_tile: int,
    compressor_head_dim: int,
    compressor_state_width: int,
    compressor_ring_size: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
    indexer_compressor_head_dim: int,
    indexer_compressor_state_width: int,
    indexer_compressor_ring_size: int,
    indexer_compressor_rope_head_dim: int,
    indexer_compressor_block_size: int,
    indexer_compressor_fp8_max: float,
    indexer_compressor_rotate: bool,
    indexer_compressor_overlap: bool,
    indexer_compressor_eps: float,
    max_c_len: int,
    indexer_max_c_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All-KV boundary decode prologue plus SWA, dual state, and cache writes."""
    (
        q_scaled,
        kv,
        comp_kv,
        comp_score,
        idx_kv,
        idx_score,
        topk_t,
        mask,
        comp_rows,
        idx_rows,
    ) = attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        kv_score_state,
        owner_ids,
        end_positions,
        compressor_ape,
        compressor_norm_weight,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        indexer_kv_score_state,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        cos_table,
        sin_table,
        compressor_cos_table,
        compressor_sin_table,
        indexer_compressor_cos_table,
        indexer_compressor_sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        kv_token_bucket=int(kv_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        kv_len=int(kv_len),
        k=int(k),
        rows=int(rows),
        k_tile=int(k_tile),
        compressor_head_dim=int(compressor_head_dim),
        compressor_state_width=int(compressor_state_width),
        compressor_ring_size=int(compressor_ring_size),
        compressor_rope_head_dim=int(compressor_rope_head_dim),
        compressor_block_size=int(compressor_block_size),
        compressor_fp8_max=float(compressor_fp8_max),
        compressor_rotate=bool(compressor_rotate),
        compressor_overlap=bool(compressor_overlap),
        compressor_eps=float(compressor_eps),
        indexer_compressor_head_dim=int(indexer_compressor_head_dim),
        indexer_compressor_state_width=int(indexer_compressor_state_width),
        indexer_compressor_ring_size=int(indexer_compressor_ring_size),
        indexer_compressor_rope_head_dim=int(indexer_compressor_rope_head_dim),
        indexer_compressor_block_size=int(indexer_compressor_block_size),
        indexer_compressor_fp8_max=float(indexer_compressor_fp8_max),
        indexer_compressor_rotate=bool(indexer_compressor_rotate),
        indexer_compressor_overlap=bool(indexer_compressor_overlap),
        indexer_compressor_eps=float(indexer_compressor_eps),
    )

    n_new = int(x.shape[0]) * int(x.shape[1])
    if int(kv_token_bucket) > 0:
        kv_rows = kv[:n_new]
    else:
        kv_rows = attention_kv_flatten_fn(
            kv,
            total_tokens=n_new,
            head_dim=int(head_dim),
        )
    owners = owner_ids.astype(np.int32).reshape(-1)[:n_new]
    pos = end_positions.astype(np.int32).reshape(-1)[:n_new]

    swa_offsets = pos - (
        (pos.astype(np.float32) / np.float32(window_size)).astype(np.int32)
        * np.int32(window_size)
    )
    swa_rows = owners * np.int32(window_size) + swa_offsets
    swa_kv_cache[swa_rows] = kv_rows.astype(swa_kv_cache.dtype)

    ratio_i = int(ratio)
    ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
        * np.int32(ratio_i)
    )
    ring_i = int(compressor_ring_size)
    ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        * np.int32(ring_i)
    )
    state_rows = owners * np.int32(ring_i) + ring_offsets
    width = int(comp_kv.shape[-1])
    comp_kv_rows = comp_kv.astype(np.float32).reshape(n_new, width)
    comp_score_rows = comp_score.astype(np.float32).reshape(n_new, width)
    ape_rows = compressor_ape[ape_offsets].astype(np.float32).reshape(n_new, width)
    kv_score_state[state_rows, :width] = comp_kv_rows.astype(kv_score_state.dtype)
    kv_score_state[state_rows, width : 2 * width] = (comp_score_rows + ape_rows).astype(
        kv_score_state.dtype
    )

    cache_rows = owners * np.int32(max_c_len) + (
        pos.astype(np.float32) / np.float32(ratio_i)
    ).astype(np.int32)
    compressed_kv_cache[cache_rows] = comp_rows.astype(compressed_kv_cache.dtype)

    indexer_ratio_i = int(indexer_compressor_ape.shape[0])
    indexer_ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(indexer_ratio_i)).astype(np.int32)
        * np.int32(indexer_ratio_i)
    )
    indexer_ring_i = int(indexer_compressor_ring_size)
    indexer_ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(indexer_ring_i)).astype(np.int32)
        * np.int32(indexer_ring_i)
    )
    indexer_state_rows = owners * np.int32(indexer_ring_i) + indexer_ring_offsets
    indexer_width = int(idx_kv.shape[-1])
    idx_kv_rows = idx_kv.astype(np.float32).reshape(n_new, indexer_width)
    idx_score_rows = idx_score.astype(np.float32).reshape(n_new, indexer_width)
    idx_ape_rows = (
        indexer_compressor_ape[indexer_ape_offsets]
        .astype(np.float32)
        .reshape(n_new, indexer_width)
    )
    indexer_kv_score_state[indexer_state_rows, :indexer_width] = idx_kv_rows.astype(
        indexer_kv_score_state.dtype
    )
    indexer_kv_score_state[
        indexer_state_rows,
        indexer_width : 2 * indexer_width,
    ] = (idx_score_rows + idx_ape_rows).astype(indexer_kv_score_state.dtype)

    indexer_cache_rows = owners * np.int32(indexer_max_c_len) + (
        pos.astype(np.float32) / np.float32(indexer_ratio_i)
    ).astype(np.int32)
    indexer_compressed_kv_cache[indexer_cache_rows] = idx_rows.astype(
        indexer_compressed_kv_cache.dtype
    )
    return q_scaled, kv, topk_t, mask


def attention_qkv_token_topk_prep_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """No-indexer compressed-attention prologue.

    Fuses QKV table projection/q-scale with token-derived sparse-attention
    prep. Sparse attention itself remains a hard backend boundary.
    """
    q_scaled, kv, qr = attention_qkv_quant_scaled_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
    )
    topk_t, mask = topk_tokens_concat_pad_sparse_attention_prep_fn(
        x,
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        max_c_len=int(max_c_len),
        rows=int(rows),
        k_tile=int(k_tile),
    )
    if int(kv_token_bucket) > 0:
        kv = pad_flat_rows_fn(
            attention_kv_flatten_fn(
                kv,
                total_tokens=int(x.shape[0]) * int(x.shape[1]),
                head_dim=int(head_dim),
            ),
            rows=int(kv_token_bucket),
        )
    return q_scaled, kv, qr, topk_t, mask


def attention_qkv_token_topk_prep_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """No-indexer compressed-attention prologue without materialized QR output."""
    q_scaled, kv, _qr, topk_t, mask = attention_qkv_token_topk_prep_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        max_c_len=int(max_c_len),
        rows=int(rows),
        k_tile=int(k_tile),
        kv_token_bucket=int(kv_token_bucket),
    )
    return q_scaled, kv, topk_t, mask


def attention_qkv_token_topk_prep_decode_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode no-indexer prologue with runtime-position top-k.

    ``start_pos`` matches the static variant signature but is not used. Decode
    top-k is derived from ``positions`` so
    product serving does not compile one NEFF per decode step.
    """
    del start_pos
    q_scaled, kv, _qr = attention_qkv_quant_scaled_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
    )
    topk_t, mask = (
        topk_tokens_concat_pad_sparse_attention_prep_decode_from_positions_fn(
            x,
            positions,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
    )
    if int(kv_token_bucket) > 0:
        kv = pad_flat_rows_fn(
            attention_kv_flatten_fn(
                kv,
                total_tokens=int(x.shape[0]) * int(x.shape[1]),
                head_dim=int(head_dim),
            ),
            rows=int(kv_token_bucket),
        )
    return q_scaled, kv, topk_t, mask


def attention_qkv_compressor_kv_score_token_topk_prep_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """No-indexer QKV/top-k prologue plus compressor input projection."""
    q_scaled, kv, topk_t, mask = attention_qkv_token_topk_prep_no_qr_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        cos_table,
        sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        max_c_len=int(max_c_len),
        rows=int(rows),
        k_tile=int(k_tile),
        kv_token_bucket=int(kv_token_bucket),
    )
    comp_kv, comp_score = compressor_kv_score_bf16_fn(
        x,
        compressor_wkv,
        compressor_wgate,
    )
    return q_scaled, kv, topk_t, mask, comp_kv, comp_score


def attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_prep_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int,
    compressor_head_dim: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """No-indexer prefill prologue plus compressor pool/post-QDQ rows."""
    if int(start_pos) != 0:
        raise RuntimeError("prefill compressor post-QDQ fusion requires start_pos=0")
    q_scaled, kv, topk_t, mask, comp_kv, comp_score = (
        attention_qkv_compressor_kv_score_token_topk_prep_no_qr_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    cutoff = int(x.shape[1]) - (int(x.shape[1]) % int(ratio))
    comp_rows = compressor_prefill_post_qdq_from_token_slabs_fn(
        comp_kv,
        comp_score,
        compressor_ape,
        compressor_norm_weight,
        compressor_cos_table,
        compressor_sin_table,
        positions,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        cutoff=int(cutoff),
        ratio=int(ratio),
        head_dim=int(compressor_head_dim),
        rope_head_dim=int(compressor_rope_head_dim),
        block_size=int(compressor_block_size),
        fp8_max=float(compressor_fp8_max),
        rotate=bool(compressor_rotate),
        overlap=bool(compressor_overlap),
        eps=float(compressor_eps),
    )
    return q_scaled, kv, topk_t, mask, comp_kv, comp_score, comp_rows


def attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_write_swa_state_cache_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    compressed_kv_cache: np.ndarray,
    owner_ids: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int,
    compressor_head_dim: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
    compressor_ring_size: int,
    compressor_state_tail_len: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """No-indexer prefill prologue plus direct SWA/state/cache writes."""
    (
        q_scaled,
        kv,
        topk_t,
        mask,
        comp_kv,
        comp_score,
        comp_rows,
    ) = attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_prep_no_qr_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        compressor_cos_table,
        compressor_sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        max_c_len=int(max_c_len),
        rows=int(rows),
        k_tile=int(k_tile),
        kv_token_bucket=int(kv_token_bucket),
        compressor_head_dim=int(compressor_head_dim),
        compressor_rope_head_dim=int(compressor_rope_head_dim),
        compressor_block_size=int(compressor_block_size),
        compressor_fp8_max=float(compressor_fp8_max),
        compressor_rotate=bool(compressor_rotate),
        compressor_overlap=bool(compressor_overlap),
        compressor_eps=float(compressor_eps),
    )
    _write_prefill_swa_state_cache_rows_fn(
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        kv,
        comp_kv,
        comp_score,
        comp_rows,
        owner_ids,
        positions,
        compressor_ape,
        bsz=int(x.shape[0]),
        seqlen=int(x.shape[1]),
        head_dim=int(head_dim),
        window_size=int(window_size),
        ring_size=int(compressor_ring_size),
        state_tail_len=int(compressor_state_tail_len),
        max_c_len=int(max_c_len),
    )
    return q_scaled, kv, topk_t, mask, comp_kv, comp_score, comp_rows


def attention_qkv_compressor_kv_score_token_topk_prep_decode_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode no-indexer prologue plus compressor projection."""
    q_scaled, kv, topk_t, mask = (
        attention_qkv_token_topk_prep_decode_no_qr_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    comp_kv, comp_score = compressor_kv_score_bf16_fn(
        x,
        compressor_wkv,
        compressor_wgate,
    )
    return q_scaled, kv, topk_t, mask, comp_kv, comp_score


def attention_qkv_compressor_kv_score_token_topk_prep_write_swa_state_decode_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    compressor_ape: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int,
    compressor_ring_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode no-indexer prologue that also writes SWA and compressor state."""
    if int(start_pos) <= 0 or int(x.shape[1]) != 1:
        raise RuntimeError("decode SWA/state write fusion requires decode x")
    q_scaled, kv, topk_t, mask, comp_kv, comp_score = (
        attention_qkv_compressor_kv_score_token_topk_prep_decode_no_qr_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    n_new = int(x.shape[0]) * int(x.shape[1])
    if int(kv_token_bucket) > 0:
        kv_rows = kv[:n_new]
    else:
        kv_rows = attention_kv_flatten_fn(
            kv,
            total_tokens=n_new,
            head_dim=int(head_dim),
        )

    owners = owner_ids.astype(np.int32).reshape(-1)[:n_new]
    pos = positions.astype(np.int32).reshape(-1)[:n_new]

    swa_offsets = pos - (
        (pos.astype(np.float32) / np.float32(window_size)).astype(np.int32)
        * np.int32(window_size)
    )
    swa_rows = owners * np.int32(window_size) + swa_offsets
    swa_kv_cache[swa_rows] = kv_rows.astype(swa_kv_cache.dtype)

    ring_i = int(compressor_ring_size)
    ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        * np.int32(ring_i)
    )
    state_rows = owners * np.int32(ring_i) + ring_offsets
    ratio_i = int(ratio)
    ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
        * np.int32(ratio_i)
    )
    width = int(comp_kv.shape[-1])
    comp_kv_rows = comp_kv.astype(np.float32).reshape(n_new, width)
    comp_score_rows = comp_score.astype(np.float32).reshape(n_new, width)
    ape_rows = compressor_ape[ape_offsets].astype(np.float32).reshape(n_new, width)
    kv_score_state[state_rows, :width] = comp_kv_rows.astype(kv_score_state.dtype)
    kv_score_state[state_rows, width : 2 * width] = (comp_score_rows + ape_rows).astype(
        kv_score_state.dtype
    )
    return q_scaled, kv, topk_t, mask


def attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_prep_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int,
    compressor_head_dim: int,
    compressor_state_width: int,
    compressor_ring_size: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Decode no-indexer prologue plus current-row compressor pool/post-QDQ."""
    if int(start_pos) <= 0 or int(x.shape[1]) != 1:
        raise RuntimeError("decode compressor post-QDQ fusion requires decode x")
    q_scaled, kv, topk_t, mask, comp_kv, comp_score = (
        attention_qkv_compressor_kv_score_token_topk_prep_decode_no_qr_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    comp_rows = compressor_decode_pool_post_qdq_from_state_plus_current_freq_table_fn(
        kv_score_state,
        comp_kv,
        comp_score,
        owner_ids,
        end_positions,
        compressor_ape,
        compressor_norm_weight,
        compressor_cos_table,
        compressor_sin_table,
        positions,
        bsz=int(x.shape[0]),
        ratio=int(ratio),
        head_dim=int(compressor_head_dim),
        state_width=int(compressor_state_width),
        ring_size=int(compressor_ring_size),
        overlap=bool(compressor_overlap),
        source_token_positions=True,
        compress_ratio=int(ratio),
        start_pos=1,
        seqlen=1,
        rope_head_dim=int(compressor_rope_head_dim),
        block_size=int(compressor_block_size),
        fp8_max=float(compressor_fp8_max),
        rotate=bool(compressor_rotate),
        eps=float(compressor_eps),
    )
    return q_scaled, kv, topk_t, mask, comp_kv, comp_score, comp_rows


def attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_write_swa_state_cache_no_qr_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    swa_kv_cache: np.ndarray,
    kv_score_state: np.ndarray,
    compressed_kv_cache: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
    compressor_ape: np.ndarray,
    compressor_norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    compressor_cos_table: np.ndarray,
    compressor_sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
    kv_token_bucket: int,
    compressor_head_dim: int,
    compressor_state_width: int,
    compressor_ring_size: int,
    compressor_rope_head_dim: int,
    compressor_block_size: int,
    compressor_fp8_max: float,
    compressor_rotate: bool,
    compressor_overlap: bool,
    compressor_eps: float,
    compressed_cache_stride: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode no-indexer prologue plus boundary SWA/state/cache writes."""
    (
        q_scaled,
        kv,
        topk_t,
        mask,
        comp_kv,
        comp_score,
        comp_rows,
    ) = attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_prep_no_qr_from_freq_table_fn(
        x,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        compressor_wkv,
        compressor_wgate,
        kv_score_state,
        owner_ids,
        end_positions,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        compressor_cos_table,
        compressor_sin_table,
        positions,
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        eps=float(eps),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        q_softmax_scale=float(q_softmax_scale),
        q_token_bucket=int(q_token_bucket),
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        max_c_len=int(max_c_len),
        rows=int(rows),
        k_tile=int(k_tile),
        kv_token_bucket=int(kv_token_bucket),
        compressor_head_dim=int(compressor_head_dim),
        compressor_state_width=int(compressor_state_width),
        compressor_ring_size=int(compressor_ring_size),
        compressor_rope_head_dim=int(compressor_rope_head_dim),
        compressor_block_size=int(compressor_block_size),
        compressor_fp8_max=float(compressor_fp8_max),
        compressor_rotate=bool(compressor_rotate),
        compressor_overlap=bool(compressor_overlap),
        compressor_eps=float(compressor_eps),
    )

    n_new = int(x.shape[0]) * int(x.shape[1])
    if int(kv_token_bucket) > 0:
        kv_rows = kv[:n_new]
    else:
        kv_rows = attention_kv_flatten_fn(
            kv,
            total_tokens=n_new,
            head_dim=int(head_dim),
        )
    owners = owner_ids.astype(np.int32).reshape(-1)[:n_new]
    pos = positions.astype(np.int32).reshape(-1)[:n_new]

    swa_offsets = pos - (
        (pos.astype(np.float32) / np.float32(window_size)).astype(np.int32)
        * np.int32(window_size)
    )
    swa_rows = owners * np.int32(window_size) + swa_offsets
    swa_kv_cache[swa_rows] = kv_rows.astype(swa_kv_cache.dtype)

    ring_i = int(compressor_ring_size)
    ring_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        * np.int32(ring_i)
    )
    state_rows = owners * np.int32(ring_i) + ring_offsets
    ratio_i = int(ratio)
    ape_offsets = pos - (
        (pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
        * np.int32(ratio_i)
    )
    width = int(comp_kv.shape[-1])
    comp_kv_rows = comp_kv.astype(np.float32).reshape(n_new, width)
    comp_score_rows = comp_score.astype(np.float32).reshape(n_new, width)
    ape_rows = compressor_ape[ape_offsets].astype(np.float32).reshape(n_new, width)
    kv_score_state[state_rows, :width] = comp_kv_rows.astype(kv_score_state.dtype)
    kv_score_state[state_rows, width : 2 * width] = (comp_score_rows + ape_rows).astype(
        kv_score_state.dtype
    )

    cpos = (pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
    cache_stride = (
        int(compressed_cache_stride)
        if int(compressed_cache_stride) > 0
        else int(max_c_len)
    )
    cache_rows = owners * np.int32(cache_stride) + cpos
    compressed_kv_cache[cache_rows] = comp_rows.astype(compressed_kv_cache.dtype)
    return q_scaled, kv, topk_t, mask


def attention_qkv_empty_indexer_compressor_token_topk_prep_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Attention prologue plus empty-compressed-indexer token-topk path."""
    q_scaled, kv, _qr, comp_kv, comp_score = (
        attention_qkv_compressor_kv_score_scaled_kvflat_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    idx_kv, idx_score, topk_t, mask = compressor_kv_score_token_topk_prep_fn(
        x,
        indexer_compressor_wkv,
        indexer_compressor_wgate,
        window_size=int(window_size),
        ratio=int(ratio),
        offset=int(offset),
        start_pos=int(start_pos),
        max_c_len=int(max_c_len),
        rows=int(rows),
        k_tile=int(k_tile),
    )
    return q_scaled, kv, comp_kv, comp_score, idx_kv, idx_score, topk_t, mask


def attention_qkv_empty_indexer_compressor_token_topk_prep_decode_from_freq_table_fn(
    x: np.ndarray,
    wq_a: np.ndarray,
    q_norm: np.ndarray,
    wq_b: np.ndarray,
    wkv: np.ndarray,
    kv_norm: np.ndarray,
    compressor_wkv: np.ndarray,
    compressor_wgate: np.ndarray,
    indexer_compressor_wkv: np.ndarray,
    indexer_compressor_wgate: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
    block_size: int,
    fp8_max: float,
    q_softmax_scale: float,
    q_token_bucket: int,
    kv_token_bucket: int,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Decode variant using runtime device positions for empty indexer top-k."""
    q_scaled, kv, _qr, comp_kv, comp_score = (
        attention_qkv_compressor_kv_score_scaled_kvflat_from_freq_table_fn(
            x,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor_wkv,
            compressor_wgate,
            cos_table,
            sin_table,
            positions,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            rope_head_dim=int(rope_head_dim),
            eps=float(eps),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
            q_softmax_scale=float(q_softmax_scale),
            q_token_bucket=int(q_token_bucket),
            kv_token_bucket=int(kv_token_bucket),
        )
    )
    idx_kv, idx_score, topk_t, mask = (
        compressor_kv_score_token_topk_prep_decode_from_positions_fn(
            x,
            indexer_compressor_wkv,
            indexer_compressor_wgate,
            positions,
            window_size=int(window_size),
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
            rows=int(rows),
            k_tile=int(k_tile),
        )
    )
    return q_scaled, kv, comp_kv, comp_score, idx_kv, idx_score, topk_t, mask


def attention_kv_tail_window_fn(
    kv: np.ndarray,  # [bsz, seqlen, d]
    *,
    window_size: int,
    head_dim: int,
) -> np.ndarray:
    """Flatten only each request's final SWA window for long-prefill mirroring."""
    bsz, seqlen, _ = kv.shape
    tail = min(int(seqlen), int(window_size))
    return np.reshape(kv[:, int(seqlen) - tail :, :], (int(bsz) * tail, int(head_dim)))


def attention_kv_request_tail_window_fn(
    kv: np.ndarray,  # [bsz, seqlen, d]
    *,
    request_index: int,
    window_size: int,
    head_dim: int,
) -> np.ndarray:
    """Flatten one request's final SWA window for chunked device mirroring."""
    bsz, seqlen, _ = kv.shape
    req = int(request_index)
    if req < 0 or req >= int(bsz):
        raise ValueError(f"request_index={req} outside batch size {int(bsz)}")
    tail = min(int(seqlen), int(window_size))
    return np.reshape(kv[req : req + 1, int(seqlen) - tail :, :], (tail, int(head_dim)))


def attention_sink_2d_fn(
    sink: np.ndarray,  # [h]
    *,
    n_heads: int,
) -> np.ndarray:
    """Reshape attention sink weights to sparse-attention kernel layout."""
    return np.reshape(sink, (1, int(n_heads)))


def compressor_norm_2d_fn(
    norm_weight: np.ndarray,  # [d]
    *,
    width: int,
) -> np.ndarray:
    """Reshape compressor norm weights to post-pool kernel layout."""
    return np.reshape(norm_weight, (1, int(width)))


def attention_unpad_reshape_fn(
    out: np.ndarray,  # [token_bucket, h, d]  fp32 from sparse attention
    *,
    bsz: int,
    seqlen: int,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Drop static-bucket padding and restore ``[b, s, h, d]`` on device."""
    n_tokens = int(bsz) * int(seqlen)
    return np.reshape(
        out[:n_tokens],
        (int(bsz), int(seqlen), int(n_heads), int(head_dim)),
    )


def attention_hidden_reshape_fn(
    x: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    hidden_size: int,
) -> np.ndarray:
    """Restore attention output to the mHC contract ``[bsz, seqlen, hidden]``.

    The input may already be bucket-padded. Flatten first, then drop trailing
    bucket rows so collectives can run on static shapes while residual math
    keeps the scheduler's real token shape.
    """
    hidden = int(hidden_size)
    n_tokens = int(bsz) * int(seqlen)
    flat = np.reshape(x, (-1, hidden))
    return np.reshape(flat[:n_tokens], (int(bsz), int(seqlen), hidden))


def inverse_rope_tail_flat_fn(
    o: np.ndarray,  # [token_bucket, h, d]
    cos: np.ndarray,  # [token_bucket, rd//2]
    sin: np.ndarray,  # [token_bucket, rd//2]
    *,
    rope_head_dim: int,
) -> np.ndarray:
    """Inverse RoPE for flat sparse-attention output rows."""
    rd = int(rope_head_dim)
    head = o[..., :-rd].astype(np.float32)
    tail = o[..., -rd:].astype(np.float32)
    original_dtype = tail.dtype
    half = tail.shape[-1] // 2
    pair = tail.reshape(tail.shape[0], tail.shape[1], half, 2)
    x0 = pair[..., 0]
    x1 = pair[..., 1]
    cos_v = cos.astype(np.float32).reshape(cos.shape[0], 1, cos.shape[1])
    sin_v = -sin.astype(np.float32).reshape(sin.shape[0], 1, sin.shape[1])
    y0 = x0 * cos_v - x1 * sin_v
    y1 = x0 * sin_v + x1 * cos_v
    out = np.concatenate((y0[..., None], y1[..., None]), axis=-1)
    tail_out = out.reshape(tail.shape).astype(original_dtype)
    return np.concatenate((head, tail_out), axis=-1).astype(o.dtype)


def inverse_rope_tail_flat_from_freq_table_fn(
    o: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    rope_head_dim: int,
) -> np.ndarray:
    """Inverse RoPE using device-resident frequency tables and row positions."""
    pos = positions.astype(np.int32).reshape(-1)
    rows = int(o.shape[0])
    if int(pos.shape[0]) > rows:
        pos = pos[:rows]
    # Clamp positions into the frequency table before the indirect gather. Real
    # token positions are always in range, so this is numerically inert for the
    # standalone path; it guards padded/garbage rows on the attention-into-layer
    # fused path (where positions come straight from the backend step inputs)
    # from indexing ``cos_table`` out of bounds (device gather faults otherwise).
    max_pos = int(cos_table.shape[0])
    pos = np.minimum(np.maximum(pos, np.int32(0)), np.int32(max_pos - 1))
    cos = cos_table[pos]
    sin = sin_table[pos]
    return inverse_rope_tail_flat_fn(
        o,
        cos,
        sin,
        rope_head_dim=int(rope_head_dim),
    )


def attention_out_proj_flat_fn(
    o: np.ndarray,  # [token_bucket, h, d]
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    *,
    n_groups: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Attention output projection on static flat token rows."""
    rows, n_heads, head_dim = o.shape
    groups = int(n_groups)
    heads_per_group = n_heads // groups
    group_dim = heads_per_group * head_dim
    rank = wo_a.shape[0] // groups
    o_g = o.reshape(int(rows), groups, group_dim).astype(np.float32)
    wo_a_g = wo_a.reshape(groups, rank, group_dim).astype(np.float32)
    parts = []
    for gi in range(groups):
        parts.append(o_g[:, gi, :] @ wo_a_g[gi].T)
    projected = _linear_out_in(np.concatenate(parts, axis=-1), wo_b)
    if int(tp_degree) <= 1:
        return projected

    tp_groups = (
        [list(group) for group in tp_replica_groups]
        if tp_replica_groups
        else [list(range(int(tp_degree)))]
    )
    return _all_reduce_last_dim_preserve_shape(
        projected,
        replica_groups=tp_groups,
    )


def attention_out_proj_flat_hidden_fn(
    o: np.ndarray,  # [token_bucket, h, d]
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    *,
    n_groups: int,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Attention output projection directly restored to sampled hidden shape."""
    projected = attention_out_proj_flat_fn(
        o,
        wo_a,
        wo_b,
        n_groups=int(n_groups),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    return attention_hidden_reshape_fn(
        projected,
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
    )


def attention_out_proj_dp_flat_fn(
    o: np.ndarray,  # [token_bucket, h, d]
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    *,
    n_groups: int,
    bsz: int,
    seqlen: int,
    batch_size: int,
    start: int,
    size: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Project attention output directly into the DP-attention reduce buffer."""
    hidden = attention_out_proj_flat_hidden_fn(
        o,
        wo_a,
        wo_b,
        n_groups=int(n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    return dp_attention_lane_scatter_flatten_pad_fn(
        hidden,
        batch_size=int(batch_size),
        start=int(start),
        size=int(size),
        rows=int(rows),
        hidden_size=int(hidden_size),
    )


def attention_out_proj_dp_flat_dynamic_start_fn(
    o: np.ndarray,  # [token_bucket, h, d]
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    lane_start: np.ndarray,  # [1] int32 runtime DP lane start
    *,
    n_groups: int,
    bsz: int,
    seqlen: int,
    batch_size: int,
    size: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Project attention output into DP-flat rows with runtime lane start."""
    hidden = attention_out_proj_flat_hidden_fn(
        o,
        wo_a,
        wo_b,
        n_groups=int(n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    return dp_attention_lane_scatter_flatten_pad_dynamic_start_fn(
        hidden,
        lane_start,
        batch_size=int(batch_size),
        size=int(size),
        rows=int(rows),
        hidden_size=int(hidden_size),
    )


def attention_inverse_rope_out_proj_dp_flat_from_freq_table_fn(
    o: np.ndarray,
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    rope_head_dim: int,
    n_groups: int,
    bsz: int,
    seqlen: int,
    batch_size: int,
    start: int,
    size: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Fuse inverse RoPE tail with attention output projection into DP-flat rows."""
    restored = inverse_rope_tail_flat_from_freq_table_fn(
        o,
        cos_table,
        sin_table,
        positions,
        rope_head_dim=int(rope_head_dim),
    )
    return attention_out_proj_dp_flat_fn(
        restored,
        wo_a,
        wo_b,
        n_groups=int(n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        batch_size=int(batch_size),
        start=int(start),
        size=int(size),
        rows=int(rows),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )


def attention_inverse_rope_out_proj_dp_flat_dynamic_start_from_freq_table_fn(
    o: np.ndarray,
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    lane_start: np.ndarray,
    *,
    rope_head_dim: int,
    n_groups: int,
    bsz: int,
    seqlen: int,
    batch_size: int,
    size: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Inverse RoPE + attention output projection with runtime DP lane start."""
    restored = inverse_rope_tail_flat_from_freq_table_fn(
        o,
        cos_table,
        sin_table,
        positions,
        rope_head_dim=int(rope_head_dim),
    )
    return attention_out_proj_dp_flat_dynamic_start_fn(
        restored,
        wo_a,
        wo_b,
        lane_start,
        n_groups=int(n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        batch_size=int(batch_size),
        size=int(size),
        rows=int(rows),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )


def attention_inverse_rope_active_out_proj_dp_flat_dynamic_start_from_freq_table_fn(
    o: np.ndarray,
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    lane_start: np.ndarray,
    *,
    rope_head_dim: int,
    n_groups: int,
    bsz: int,
    seqlen: int,
    batch_size: int,
    size: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Inverse RoPE + output projection using only active token rows.

    Product sparse attention may write active rows into a canonical bucket
    buffer. Slice before inverse-RoPE/projection so canonical decode shapes do
    not project padded rows.
    """
    n_tokens = int(bsz) * int(seqlen)
    active = o[:n_tokens]
    pos = positions.astype(np.int32).reshape(-1)
    if int(pos.shape[0]) > n_tokens:
        pos = pos[:n_tokens]
    restored = inverse_rope_tail_flat_from_freq_table_fn(
        active,
        cos_table,
        sin_table,
        pos,
        rope_head_dim=int(rope_head_dim),
    )
    return attention_out_proj_dp_flat_dynamic_start_fn(
        restored,
        wo_a,
        wo_b,
        lane_start,
        n_groups=int(n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        batch_size=int(batch_size),
        size=int(size),
        rows=int(rows),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )


def attention_out_proj_dp_flat_dynamic_token_range_fn(
    o: np.ndarray,
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    token_start: np.ndarray,
    token_count: np.ndarray,
    *,
    n_groups: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Project attention output and scatter by runtime flat token range."""

    projected = attention_out_proj_flat_fn(
        o,
        wo_a,
        wo_b,
        n_groups=int(n_groups),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    return dp_attention_token_range_scatter_flatten_pad_fn(
        projected,
        token_start,
        token_count,
        rows=int(rows),
        hidden_size=int(hidden_size),
    )


def attention_inverse_rope_out_proj_dp_flat_dynamic_token_range_from_freq_table_fn(
    o: np.ndarray,
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    token_start: np.ndarray,
    token_count: np.ndarray,
    *,
    rope_head_dim: int,
    n_groups: int,
    rows: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Inverse-RoPE + output projection scattered by runtime token range."""

    restored = inverse_rope_tail_flat_from_freq_table_fn(
        o,
        cos_table,
        sin_table,
        positions,
        rope_head_dim=int(rope_head_dim),
    )
    return attention_out_proj_dp_flat_dynamic_token_range_fn(
        restored,
        wo_a,
        wo_b,
        token_start,
        token_count,
        n_groups=int(n_groups),
        rows=int(rows),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )


def attention_out_proj_mhc_post_pre_fn(
    o: np.ndarray,  # [token_bucket, h, d]
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    n_groups: int,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    moe_replica_groups: tuple = (),
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse attention output projection into the following mHC post/pre stage."""
    out = attention_out_proj_flat_hidden_fn(
        o,
        wo_a,
        wo_b,
        n_groups=int(n_groups),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    return mhc_post_pre_fn(
        out,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )


def head_hidden_flatten_fn(
    hidden: np.ndarray,  # [bsz, seqlen, hidden_size]
    *,
    n_tokens: int,
    hidden_size: int,
) -> np.ndarray:
    """Flatten sampled-head hidden rows for the logits processor."""
    return np.reshape(hidden, (int(n_tokens), int(hidden_size)))


def dp_attention_lane_slice_fn(
    x: np.ndarray,  # [batch, seqlen, ...]
    *,
    start: int,
    size: int,
) -> np.ndarray:
    """Slice the current rank's DP-attention lane from a full superstep."""
    begin = int(start)
    end = begin + int(size)
    return x[begin:end]


def dp_attention_lane_scatter_fn(
    x: np.ndarray,  # [lane_batch, seqlen, hidden]
    *,
    batch_size: int,
    start: int,
    size: int,
) -> np.ndarray:
    """Scatter a lane-local attention result into full-batch row positions."""
    total = int(batch_size)
    begin = int(start)
    lane_size = int(size)
    end = begin + lane_size
    if begin < 0 or lane_size < 0 or end > total:
        raise RuntimeError(
            "invalid DP-attention lane scatter range: "
            f"start={begin}, size={lane_size}, batch={total}"
        )
    tail_shape = tuple(int(dim) for dim in x.shape[1:])
    parts = []
    if begin:
        parts.append(np.zeros((begin, *tail_shape), dtype=x.dtype))
    if lane_size:
        parts.append(x[:lane_size])
    if end < total:
        parts.append(np.zeros((total - end, *tail_shape), dtype=x.dtype))
    if not parts:
        return np.zeros((total, *tail_shape), dtype=x.dtype)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def dp_attention_lane_scatter_flatten_pad_fn(
    x: np.ndarray,  # [lane_batch, seqlen, hidden]
    *,
    batch_size: int,
    start: int,
    size: int,
    rows: int,
    hidden_size: int,
) -> np.ndarray:
    """Scatter lane-local attention rows directly into the flat reduce buffer."""
    total = int(batch_size)
    begin = int(start)
    lane_size = int(size)
    hidden = int(hidden_size)
    target = int(rows)
    if begin < 0 or lane_size < 0 or begin + lane_size > total:
        raise RuntimeError(
            "invalid DP-attention lane scatter/flatten range: "
            f"start={begin}, size={lane_size}, batch={total}"
        )
    flat = np.reshape(x[:lane_size], (-1, hidden))
    seq = int(flat.shape[0]) // lane_size if lane_size else 0
    full_rows = total * seq
    if target < full_rows:
        raise ValueError(
            "DP-attention flat reduce rows cannot be smaller than full batch rows: "
            f"rows={target}, batch_rows={full_rows}"
        )
    offset = begin * seq
    end = offset + int(flat.shape[0])
    parts = []
    if offset:
        parts.append(np.zeros((offset, hidden), dtype=flat.dtype))
    parts.append(flat)
    if end < target:
        parts.append(np.zeros((target - end, hidden), dtype=flat.dtype))
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def dp_attention_lane_scatter_flatten_pad_dynamic_start_fn(
    x: np.ndarray,  # [lane_batch, seqlen, hidden]
    lane_start: np.ndarray,  # [1] int32
    *,
    batch_size: int,
    size: int,
    rows: int,
    hidden_size: int,
) -> np.ndarray:
    """Scatter lane-local rows into the flat reduce buffer with runtime start."""
    total = int(batch_size)
    lane_size = int(size)
    hidden = int(hidden_size)
    target = int(rows)
    if lane_size <= 0:
        raise RuntimeError(f"lane size must be positive, got {lane_size}")
    flat = np.reshape(x[:lane_size], (-1, hidden))
    seq = int(flat.shape[0]) // lane_size
    full_rows = total * seq
    if target < full_rows:
        raise ValueError(
            "DP-attention flat reduce rows cannot be smaller than full batch rows: "
            f"rows={target}, batch_rows={full_rows}"
        )
    start_v = lane_start.astype(np.int32).reshape(-1)[:1].reshape(1, 1)
    offset = start_v * np.int32(seq)
    row = np.arange(target, dtype=np.int32).reshape(target, 1)
    col = np.arange(hidden, dtype=np.int32).reshape(1, hidden)
    local_row = row - offset
    flat_rows = int(flat.shape[0])
    valid = (local_row >= np.int32(0)) & (local_row < np.int32(flat_rows))
    safe_row = np.where(valid, local_row, np.int32(0))
    # NKIPy tracing supports one tensor index; flatten source rows so the
    # dynamic gather uses a single fused row/column index tensor.
    flat_1d = np.reshape(flat, (-1,))
    gathered = flat_1d[safe_row * np.int32(hidden) + col]
    zero = np.zeros((target, hidden), dtype=flat.dtype)
    return np.where(valid, gathered, zero)


def dp_attention_token_range_scatter_flatten_pad_fn(
    x: np.ndarray,  # [compile_tokens, hidden]
    token_start: np.ndarray,  # [1] int32 runtime flat token start
    token_count: np.ndarray,  # [1] int32 runtime active token count
    *,
    rows: int,
    hidden_size: int,
) -> np.ndarray:
    """Scatter a lane-local flat active prefix into flat DP reduce rows."""

    hidden = int(hidden_size)
    target = int(rows)
    flat = np.reshape(x, (-1, hidden))
    start_v = token_start.astype(np.int32).reshape(-1)[:1].reshape(1, 1)
    count_v = token_count.astype(np.int32).reshape(-1)[:1].reshape(1, 1)
    row = np.arange(target, dtype=np.int32).reshape(target, 1)
    col = np.arange(hidden, dtype=np.int32).reshape(1, hidden)
    local_row = row - start_v
    valid = (local_row >= np.int32(0)) & (local_row < count_v)
    safe_row = np.where(valid, local_row, np.int32(0))
    flat_1d = np.reshape(flat, (-1,))
    gathered = flat_1d[safe_row * np.int32(hidden) + col]
    zero = np.zeros((target, hidden), dtype=flat.dtype)
    return np.where(valid, gathered, zero)


def dp_attention_flat_zero_fn(
    x: np.ndarray,
    *,
    rows: int,
    hidden_size: int,
) -> np.ndarray:
    """Create a flat float32 zero reduce buffer for empty DP-attention lanes."""
    zero = np.sum(x.astype(np.float32)) * np.float32(0.0)
    return np.zeros((int(rows), int(hidden_size)), dtype=np.float32) + zero


def dp_attention_flatten_pad_fn(
    x: np.ndarray,
    *,
    rows: int,
    hidden_size: int,
) -> np.ndarray:
    """Flatten DP-attention rows and avoid singleton-row rank collapse."""
    target = int(rows)
    hidden = int(hidden_size)
    flat = np.reshape(x, (-1, hidden))
    n_rows = int(flat.shape[0])
    if n_rows > target:
        raise ValueError(f"input rows {n_rows} exceed target rows {target}")
    if n_rows == target:
        return flat
    pad = np.zeros((target - n_rows, hidden), dtype=flat.dtype)
    return np.concatenate((flat, pad), axis=0)


def sparse_attention_paged_swa_decode_fn(
    q_scaled_t: np.ndarray,  # [bucket, head_dim, n_heads] bf16
    kv_hbm: np.ndarray,  # [num_kv_slots, head_dim] bf16
    positions: np.ndarray,  # [bucket, 1] int32, absolute token pos
    block_tables_per_token: np.ndarray,  # [bucket, max_blocks] int32
    sink: np.ndarray,  # [1, n_heads] fp32
    *,
    block_size: int,
    window_size: int,
    max_k: int,
) -> np.ndarray:
    """Paged sliding-window sparse attention with in-fragment slot derivation.

    Graph-fragment transcription of ``_sparse_attn_batched_paged_swa_multiK_kernel``
    (the production decode attention kernel). Derives the sliding-window global
    KV-cache slots per query from ``positions`` + ``block_tables_per_token``,
    gathers the selected KV rows from the flat paged cache via a tensor-indexed
    gather (supported in the fragment IR — see
    ``vocab_parallel_embedding_local_fn``), and runs a sink-aware softmax.

    ``q_scaled_t`` is ALREADY multiplied by ``softmax_scale`` and transposed to
    ``[N_q, head_dim, n_heads]`` by ``q_scale_transpose_fn``; the scale is
    therefore NOT re-applied here (the production kernel never scales the qk
    accumulator either). Returns ``[bucket, n_heads, head_dim]`` fp32.
    """
    n_q, head_dim, n_heads = q_scaled_t.shape
    max_blocks = int(block_tables_per_token.shape[1])
    bs = int(block_size)
    eff_win = min(int(window_size), int(max_k))
    mk = int(max_k)

    # q back to oracle layout [N_q, h, d] fp32; the softmax_scale is already
    # baked into q_scaled_t, so it is intentionally not re-applied below.
    q = q_scaled_t.transpose(0, 2, 1).astype(np.float32)  # [N_q, h, d]

    # Per-query sliding-window length and window start.
    pos = positions.astype(np.int32).reshape(n_q)  # [N_q]
    cur_len = np.minimum(pos + np.int32(1), np.int32(eff_win))  # [N_q]
    start_pos = pos - cur_len + np.int32(1)  # [N_q]

    # Logical positions for the max_k window slots, with validity mask.
    k_range = np.arange(mk, dtype=np.int32)[None, :]  # [1, max_k]
    logical = start_pos[:, None] + k_range  # [N_q, max_k]
    valid = k_range < cur_len[:, None]  # [N_q, max_k] bool

    # block_idx = logical // block_size, block_offset = logical % block_size.
    # Integer ``//``/``%`` lower to a ``floor`` on int operands which the
    # hardware rejects; do the division in fp32 (the only floor is the
    # float->int cast, on a float operand) and recover the offset by
    # subtraction. A small epsilon guards against fp32 representing e.g.
    # 32/16 as 1.9999998 and flooring to 1.
    logical_f = logical.astype(np.float32)
    block_idx = (logical_f * np.float32(1.0 / float(bs)) + np.float32(1e-4)).astype(
        np.int32
    )  # floor on float
    block_offset = logical - block_idx * np.int32(bs)  # [N_q, max_k]
    safe_block_idx = np.minimum(
        np.maximum(block_idx, np.int32(0)), np.int32(max_blocks - 1)
    )

    # Resolve to flat cache slots through the per-token block table. Use an
    # explicit FLAT linear gather (row * max_blocks + col) instead of
    # ``take_along_axis``: the 2-D fancy-index form can lower to a device gather
    # whose descriptor reads out of bounds, so flatten the [N_q, max_blocks]
    # table and index it 1-D with a fully-clamped flat index.
    bt_flat = block_tables_per_token.astype(np.int32).reshape(-1)  # [N_q*max_blocks]
    row_off = np.arange(n_q, dtype=np.int32)[:, None] * np.int32(max_blocks)  # [N_q, 1]
    flat_idx = row_off + safe_block_idx  # [N_q, max_k]
    flat_idx = np.minimum(
        np.maximum(flat_idx, np.int32(0)),
        np.int32(n_q * max_blocks - 1),
    )
    block_id = np.take(bt_flat, flat_idx.reshape(-1)).reshape(n_q, mk)
    slot = block_id * np.int32(bs) + block_offset  # [N_q, max_k]
    # Clamp the resolved slot into the flat KV cache before the indirect gather:
    # invalid window slots (and any stale/garbage block-table entries on padded
    # rows) must never index out of ``kv_hbm`` or the device gather faults
    # (NRT_EXEC_OOB). Invalid rows are zeroed afterwards via ``valid_f`` anyway.
    num_kv_slots = int(kv_hbm.shape[0])
    slot = np.minimum(np.maximum(slot, np.int32(0)), np.int32(num_kv_slots - 1))
    safe_slot = np.where(valid, slot, np.int32(0))  # invalid -> slot 0

    # Numeric validity mask (1.0 valid / 0.0 invalid). Used as a multiplicative
    # KV zero-out and an additive score bias, mirroring the production kernel's
    # ``(mask - 1) * 1e9`` masking (avoids an inf literal the HLO emitter cannot
    # serialize while being numerically identical to ``where(valid, ., -inf)``).
    valid_f = valid.astype(np.float32)  # [N_q, max_k]

    # Gather KV; zero invalid rows so the masked softmax is bit-faithful to
    # ``gather_kv_and_mask`` + ``sparse_mla_attention_oracle``.
    gathered = (
        kv_hbm[safe_slot.reshape(-1)]
        .reshape(
            n_q,
            mk,
            head_dim,
        )
        .astype(np.float32)
    )  # [N_q, max_k, d]
    gathered = gathered * valid_f[:, :, None]

    # Scores (scale already in q), masked, sink-aware softmax. Expressed as
    # batched matmuls (np.einsum is not supported by the fragment tracer):
    #   scores[n] = q[n] @ gathered[n].T   -> [h, max_k]
    #   out[n]    = p[n] @ gathered[n]      -> [h, d]
    gathered_t = np.transpose(gathered, (0, 2, 1))  # [N_q, d, max_k]
    scores = np.matmul(q, gathered_t)  # [N_q, h, max_k]
    score_bias = (valid_f - np.float32(1.0)) * np.float32(1e9)  # 0 / -1e9
    scores = scores + score_bias[:, None, :]
    # any_valid[n] == True iff query n has >=1 valid window slot.
    any_valid_f = np.max(valid_f, axis=-1, keepdims=True)  # [N_q, 1]
    m = np.max(scores, axis=-1, keepdims=True)  # [N_q, h, 1]
    e = np.exp(scores - m)
    e = e * valid_f[:, None, :]
    sink_e = np.exp(sink.astype(np.float32)[0][None, :, None] - m)  # [N_q, h, 1]
    denom = np.sum(e, axis=-1, keepdims=True) + sink_e
    p = e / denom

    out = np.matmul(p, gathered)  # [N_q, h, d]
    out = out * any_valid_f[:, :, None]  # zero no-slot queries
    return out.astype(np.float32)


__all__ = [
    "_mod_i32",
    "_write_prefill_swa_state_cache_rows_fn",
    "attention_hidden_reshape_fn",
    "attention_inverse_rope_active_out_proj_dp_flat_dynamic_start_from_freq_table_fn",
    "attention_inverse_rope_out_proj_dp_flat_dynamic_start_from_freq_table_fn",
    "attention_inverse_rope_out_proj_dp_flat_dynamic_token_range_from_freq_table_fn",
    "attention_inverse_rope_out_proj_dp_flat_from_freq_table_fn",
    "attention_kv_flatten_fn",
    "attention_kv_request_tail_window_fn",
    "attention_kv_tail_window_fn",
    "attention_out_proj_dp_flat_dynamic_start_fn",
    "attention_out_proj_dp_flat_dynamic_token_range_fn",
    "attention_out_proj_dp_flat_fn",
    "attention_out_proj_flat_fn",
    "attention_out_proj_flat_hidden_fn",
    "attention_out_proj_mhc_post_pre_fn",
    "attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_prep_no_qr_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_decode_post_qdq_token_topk_write_swa_state_cache_no_qr_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_prep_no_qr_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_prefill_post_qdq_token_topk_write_swa_state_cache_no_qr_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_scaled_kvflat_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_token_topk_prep_decode_no_qr_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_token_topk_prep_no_qr_from_freq_table_fn",
    "attention_qkv_compressor_kv_score_token_topk_prep_write_swa_state_decode_no_qr_from_freq_table_fn",
    "attention_qkv_empty_indexer_compressor_token_topk_prep_decode_from_freq_table_fn",
    "attention_qkv_empty_indexer_compressor_token_topk_prep_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_prep_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_decode_post_qdq_topk_write_swa_dual_state_cache_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_prep_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_prefill_post_qdq_topk_write_swa_dual_state_cache_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_topk_prep_decode_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_topk_prep_from_freq_table_fn",
    "attention_qkv_indexer_compressor_all_kv_topk_write_swa_dual_state_decode_from_freq_table_fn",
    "attention_qkv_indexer_compressor_qw_prep_from_freq_table_fn",
    "attention_qkv_indexer_compressor_qw_prep_write_swa_dual_state_decode_from_freq_table_fn",
    "attention_qkv_proj_fn",
    "attention_qkv_quant_fn",
    "attention_qkv_quant_from_freq_table_fn",
    "attention_qkv_quant_scaled_from_freq_table_fn",
    "attention_qkv_quant_scaled_kv_cache_write_no_qr_from_freq_table_fn",
    "attention_qkv_quant_scaled_kvflat_from_freq_table_fn",
    "attention_qkv_quant_scaled_kvflat_no_qr_from_freq_table_fn",
    "attention_qkv_token_topk_prep_decode_no_qr_from_freq_table_fn",
    "attention_qkv_token_topk_prep_from_freq_table_fn",
    "attention_qkv_token_topk_prep_no_qr_from_freq_table_fn",
    "attention_sink_2d_fn",
    "attention_unpad_reshape_fn",
    "compressor_norm_2d_fn",
    "dp_attention_flat_zero_fn",
    "dp_attention_flatten_pad_fn",
    "dp_attention_lane_scatter_flatten_pad_dynamic_start_fn",
    "dp_attention_lane_scatter_flatten_pad_fn",
    "dp_attention_lane_scatter_fn",
    "dp_attention_lane_slice_fn",
    "head_hidden_flatten_fn",
    "inverse_rope_tail_flat_fn",
    "inverse_rope_tail_flat_from_freq_table_fn",
    "q_scale_transpose_fn",
    "sparse_attention_paged_swa_decode_fn",
]
