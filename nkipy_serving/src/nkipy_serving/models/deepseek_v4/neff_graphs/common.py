"""Traceable NumPy trace-function primitives + cross-cutting helpers (leaf layer).

Merged from the former shared/core/common modules plus the cross-domain helpers
promoted here so attention.py and moe.py depend only on this module. Bodies are
byte-identical to their pre-merge form."""

from __future__ import annotations

import numpy as np

from nkipy_serving.models.deepseek_v4.eager_ops import (
    hc_split_sinkhorn,
    sqrtsoftplus,
)
from nkipy_serving.ops.nn import apply_rms_norm
from nkipy_serving.ops.vocab_parallel_embedding import (
    vocab_parallel_embedding_no_sp_dynamic_range_fn,
    vocab_parallel_embedding_no_sp_fn,
)


def _all_reduce_last_dim_preserve_shape(
    x: np.ndarray,
    *,
    replica_groups: list[list[int]],
) -> np.ndarray:
    """Run all-reduce on 2D rows and restore the caller's logical shape."""
    original_shape = tuple(int(dim) for dim in x.shape)
    if len(original_shape) > 2:
        flat = np.reshape(x, (-1, original_shape[-1]))
    else:
        flat = x

    import nkipy.distributed.collectives as cc

    reduced = cc.all_reduce(
        flat,
        replica_groups=replica_groups,
        reduce_op=np.add,
    )
    if len(original_shape) > 2:
        return np.reshape(reduced, original_shape)
    return reduced


def _apply_interleaved_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Interleaved RoPE using real cos/sin tensors instead of complex values."""
    original_dtype = x.dtype
    xf = x.astype(np.float32)
    half = xf.shape[-1] // 2
    pair = xf.reshape(*xf.shape[:-1], half, 2)
    x0 = pair[..., 0]
    x1 = pair[..., 1]
    cos_f = cos.astype(np.float32)
    sin_f = sin.astype(np.float32)
    if inverse:
        sin_f = -sin_f
    if xf.ndim == 2:
        if cos_f.ndim != 2:
            raise RuntimeError(
                f"RoPE cos/sin expects rank-2 for rank-2 tensor, got {cos_f.shape}"
            )
        cos_v = cos_f
        sin_v = sin_f
    elif xf.ndim == 3:
        if cos_f.ndim == 2:
            cos_v = cos_f.reshape(1, cos_f.shape[0], cos_f.shape[1])
            sin_v = sin_f.reshape(1, sin_f.shape[0], sin_f.shape[1])
        elif cos_f.ndim == 3:
            cos_v = cos_f
            sin_v = sin_f
        else:
            raise RuntimeError(
                f"RoPE cos/sin expects rank-2 or rank-3, got {cos_f.shape}"
            )
    elif xf.ndim == 4:
        if cos_f.ndim == 2:
            cos_v = cos_f.reshape(1, cos_f.shape[0], 1, cos_f.shape[1])
            sin_v = sin_f.reshape(1, sin_f.shape[0], 1, sin_f.shape[1])
        elif cos_f.ndim == 3:
            cos_v = cos_f.reshape(cos_f.shape[0], cos_f.shape[1], 1, cos_f.shape[2])
            sin_v = sin_f.reshape(sin_f.shape[0], sin_f.shape[1], 1, sin_f.shape[2])
        else:
            raise RuntimeError(
                f"RoPE cos/sin expects rank-2 or rank-3, got {cos_f.shape}"
            )
    else:
        raise RuntimeError(f"RoPE expects rank-3 or rank-4 tensor, got {xf.shape}")
    y0 = x0 * cos_v - x1 * sin_v
    y1 = x0 * sin_v + x1 * cos_v
    out = np.concatenate((y0[..., None], y1[..., None]), axis=-1)
    return out.reshape(xf.shape).astype(original_dtype)


def _linear_out_in(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``x @ w.T`` for V4 weights stored as ``[out, in]``."""
    y = x.astype(np.float32) @ w.astype(np.float32).T
    return np.reshape(y, (*x.shape[:-1], int(w.shape[0])))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x.astype(np.float32)))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    xf = x.astype(np.float32)
    m = xf.max(axis=axis, keepdims=True)
    e = np.exp(xf - m)
    return e / e.sum(axis=axis, keepdims=True)


def cast_bf16_fn(x: np.ndarray) -> np.ndarray:
    """Cast a tensor to bf16 on device.

    One-op fragment that lets the post-qdq chain consume the pool kernel's
    fp32 DeviceTensor output without a host round-trip.
    """
    import ml_dtypes as _ml

    return x.astype(_ml.bfloat16)


def compressed_topk_no_indexer_decode_from_positions_fn(
    x: np.ndarray,  # [b, 1, ...]
    positions: np.ndarray,  # [b] or padded flat positions
    *,
    ratio: int,
    offset: int,
    max_c_len: int,
) -> np.ndarray:
    """Decode compressed top-k using runtime device positions."""
    b = int(x.shape[0])
    r = np.int32(ratio)
    off = np.int32(offset)
    w = int(max_c_len)
    pos = positions.astype(np.int32).reshape(-1)
    sp = pos[:1].reshape(1, 1)
    sp1 = sp + np.int32(1)
    n = (sp1.astype(np.float32) * (np.float32(1.0) / np.float32(r))).astype(np.int32)
    idx = np.arange(w, dtype=np.int32).reshape(1, w)
    row = np.where(idx < n, idx + off, np.int32(-1))
    anchor = x[:, :1, :1].astype(np.int32) * np.int32(0)
    return np.broadcast_to(row.reshape(1, 1, w), (b, 1, w)) + anchor


def compressed_topk_no_indexer_from_tokens_fn(
    x: np.ndarray,  # [b, s, ...]
    *,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
) -> np.ndarray:
    """Build no-indexer compressed top-k on device from token dimensions."""
    b, s = x.shape[:2]
    r = int(ratio)
    off = np.int32(offset)
    sp = int(start_pos)
    if sp > 0:
        w = int(max_c_len)
        n = (sp + 1) // r
        idx = np.arange(w, dtype=np.int32)
        row = np.where(idx < np.int32(n), idx + off, np.int32(-1))
        anchor = x[:, :1, :1].astype(np.int32) * np.int32(0)
        return np.broadcast_to(row.reshape(1, 1, w), (int(b), 1, w)) + anchor

    c_len = int(s) // r
    cols = np.arange(c_len, dtype=np.int32)
    matrix = np.broadcast_to(cols[None, :], (int(s), c_len)).copy()
    row = (np.arange(1, int(s) + 1, dtype=np.int32) // np.int32(r))[:, None]
    matrix = np.where(matrix >= row, np.int32(-1), matrix + off)
    anchor = x[:, :, :1].astype(np.int32) * np.int32(0)
    return np.broadcast_to(matrix[None, :, :], (int(b), int(s), c_len)) + anchor


def compressor_decode_pool_from_state_plus_current_fn(
    kv_score_state: np.ndarray,  # [owners * ring_size, 2 * state_width]
    kv_new: np.ndarray,  # [bsz, state_width] current-token KV slab
    score_new: np.ndarray,  # [bsz, state_width] current-token score slab
    owner_ids: np.ndarray,  # [bsz] int32
    end_positions: np.ndarray,  # [bsz] int32
    ape: np.ndarray,  # [ratio, state_width]
    *,
    ratio: int,
    head_dim: int,
    state_width: int,
    ring_size: int,
    overlap: bool,
) -> np.ndarray:
    """Decode-pool as if the current row had already been ring-scattered.

    Product decode prologues can read old state and the current compressor
    projection in one pure trace function. The actual mutable state write still
    happens afterward through the dedicated NKI scatter kernel, so this helper
    substitutes only the newest row that would otherwise be read from state.
    """
    bsz_i = int(owner_ids.shape[0])
    ratio_i = int(ratio)
    d_i = int(head_dim)
    width_i = int(state_width)
    ring_i = int(ring_size)
    owners = owner_ids.astype(np.int32).reshape(bsz_i, 1)
    end_pos = end_positions.astype(np.int32).reshape(bsz_i, 1)
    offs = np.arange(ratio_i, dtype=np.int32).reshape(1, ratio_i)
    current_mask = (offs == np.int32(ratio_i - 1)).reshape(1, ratio_i, 1)

    def ring_rows(pos: np.ndarray) -> np.ndarray:
        safe_pos = np.where(pos >= np.int32(0), pos, np.zeros_like(pos))
        quot = (safe_pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        rem = safe_pos - quot * np.int32(ring_i)
        return owners * np.int32(ring_i) + rem

    ape_pos = end_pos - (
        (end_pos.astype(np.float32) / np.float32(ratio_i)).astype(np.int32)
        * np.int32(ratio_i)
    )
    cur_kv_full = kv_new.astype(np.float32).reshape(bsz_i, width_i)
    cur_score_full = score_new.astype(np.float32).reshape(bsz_i, width_i) + ape[
        ape_pos.reshape(-1)
    ].astype(np.float32).reshape(bsz_i, width_i)

    if bool(overlap):
        prev_pos = end_pos - np.int32(2 * ratio_i - 1) + offs
        cur_pos = end_pos - np.int32(ratio_i - 1) + offs
        prev_rows = ring_rows(prev_pos)
        cur_rows = ring_rows(cur_pos)
        prev_state = (
            kv_score_state[prev_rows.reshape(-1)]
            .reshape(
                bsz_i,
                ratio_i,
                int(kv_score_state.shape[-1]),
            )
            .astype(np.float32)
        )
        cur_state = (
            kv_score_state[cur_rows.reshape(-1)]
            .reshape(
                bsz_i,
                ratio_i,
                int(kv_score_state.shape[-1]),
            )
            .astype(np.float32)
        )
        prev_valid = (prev_pos >= np.int32(0)).reshape(bsz_i, ratio_i, 1)
        kv_prev = np.where(
            prev_valid,
            prev_state[:, :, :d_i],
            np.zeros((bsz_i, ratio_i, d_i), dtype=np.float32),
        )
        score_prev = np.where(
            prev_valid,
            prev_state[:, :, width_i : width_i + d_i],
            np.full((bsz_i, ratio_i, d_i), -1e9, dtype=np.float32),
        )
        kv_cur = cur_state[:, :, d_i : 2 * d_i]
        score_cur = cur_state[:, :, width_i + d_i : width_i + 2 * d_i]
        kv_cur = np.where(
            current_mask,
            cur_kv_full[:, None, d_i : 2 * d_i],
            kv_cur,
        )
        score_cur = np.where(
            current_mask,
            cur_score_full[:, None, d_i : 2 * d_i],
            score_cur,
        )
        kv_parts = np.concatenate((kv_prev, kv_cur), axis=1)
        score_parts = np.concatenate((score_prev, score_cur), axis=1)
    else:
        cur_pos = end_pos - np.int32(ratio_i - 1) + offs
        rows = ring_rows(cur_pos)
        state = (
            kv_score_state[rows.reshape(-1)]
            .reshape(
                bsz_i,
                ratio_i,
                int(kv_score_state.shape[-1]),
            )
            .astype(np.float32)
        )
        if ratio_i == 1:
            kv_parts = cur_kv_full[:, None, :d_i]
            score_parts = cur_score_full[:, None, :d_i]
        else:
            # The newest boundary row is always the final offset.  Slice/concat
            # avoids a large tensorselect that Neuron 2.24 can fail to legalize
            # for ratio=128 no-indexer decode write-cache product graphs.
            kv_parts = np.concatenate(
                (
                    state[:, : ratio_i - 1, :d_i],
                    cur_kv_full[:, None, :d_i],
                ),
                axis=1,
            )
            score_parts = np.concatenate(
                (
                    state[:, : ratio_i - 1, width_i : width_i + d_i],
                    cur_score_full[:, None, :d_i],
                ),
                axis=1,
            )

    score_max = score_parts.max(axis=1, keepdims=True)
    weights = np.exp(score_parts - score_max)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (kv_parts * weights).sum(axis=1).astype(np.float32)


def compressor_decode_pool_post_qdq_from_state_plus_current_freq_table_fn(
    kv_score_state: np.ndarray,
    kv_new: np.ndarray,
    score_new: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
    ape: np.ndarray,
    norm_weight: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    bsz: int,
    ratio: int,
    head_dim: int,
    state_width: int,
    ring_size: int,
    overlap: bool,
    source_token_positions: bool = False,
    compress_ratio: int = 1,
    start_pos: int = 0,
    seqlen: int = 0,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
    rotate: bool,
    eps: float,
) -> np.ndarray:
    """Fuse decode compressor pool/post-QDQ using old state plus current row."""
    pooled = compressor_decode_pool_from_state_plus_current_fn(
        kv_score_state,
        kv_new,
        score_new,
        owner_ids,
        end_positions,
        ape,
        ratio=int(ratio),
        head_dim=int(head_dim),
        state_width=int(state_width),
        ring_size=int(ring_size),
        overlap=bool(overlap),
    )
    return compressor_post_qdq_from_freq_table_fn(
        pooled,
        norm_weight,
        cos_table,
        sin_table,
        positions,
        bsz=int(bsz),
        clen=1,
        source_token_positions=bool(source_token_positions),
        compress_ratio=int(compress_ratio),
        start_pos=int(start_pos),
        seqlen=int(seqlen),
        rope_head_dim=int(rope_head_dim),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        rotate=bool(rotate),
        eps=float(eps),
    )


def compressor_kv_score_bf16_fn(
    x: np.ndarray,  # [bsz, seqlen, dim]
    wkv: np.ndarray,  # [width, dim]
    wgate: np.ndarray,  # [width, dim]
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse ``two_linear(x, wkv, wgate)`` + reshape + bf16 cast on device.

    Returns ``(kv_flat, score_flat)`` each ``[bsz*seqlen, width] bf16``.
    Chains directly into ``run_write_kv_score_state_device`` /
    ``run_prefill_pool_from_slab_device`` without a host round-trip.
    """
    import ml_dtypes as _ml

    kv = _linear_out_in(x, wkv)  # fp32 [bsz, seqlen, width]
    score = _linear_out_in(x, wgate)  # fp32 [bsz, seqlen, width]
    bsz, seqlen, width = kv.shape
    n = int(bsz) * int(seqlen)
    kv_flat = kv.reshape(n, width).astype(_ml.bfloat16)
    score_flat = score.reshape(n, width).astype(_ml.bfloat16)
    return kv_flat, score_flat


def compressor_kv_score_token_topk_prep_decode_from_positions_fn(
    x: np.ndarray,  # [bsz, 1, dim]
    wkv: np.ndarray,  # [width, dim]
    wgate: np.ndarray,  # [width, dim]
    positions: np.ndarray,  # [bsz] or padded flat positions
    *,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode empty-compressor prologue with runtime-position top-k."""
    del start_pos
    kv_bf, score_bf = compressor_kv_score_bf16_fn(x, wkv, wgate)
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
    return kv_bf, score_bf, topk_t, mask


def compressor_kv_score_token_topk_prep_fn(
    x: np.ndarray,  # [bsz, seqlen, dim]
    wkv: np.ndarray,  # [width, dim]
    wgate: np.ndarray,  # [width, dim]
    *,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fused empty-compressed-KV indexer prologue.

    The indexer path must always mirror fresh compressor KV/score state, even
    when compressed KV length is zero and no score-dependent indexer work is
    needed. This function fuses that state input projection with the static
    token-derived sparse-attention prep used by the empty compressed side.
    """
    kv_bf, score_bf = compressor_kv_score_bf16_fn(x, wkv, wgate)
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
    return kv_bf, score_bf, topk_t, mask


def compressor_pool_fn(
    kv: np.ndarray,  # [b, s', r, d]  fp32
    score: np.ndarray,  # [b, s', r, d]  fp32 (score + ape already added)
) -> np.ndarray:
    """Softmax-weighted pool over the ratio axis.

    Replaces Compressor.forward's
        w = softmax(score_r, axis=2); kv_pool = (kv_r * w).sum(axis=2)
    with a single HLO-traceable graph-fn. All dense ops; no gather, no
    top-k, no bit-manipulation — fully decidable shapes at trace time.
    """
    # Score may contain -inf from invalid overlap positions; softmax handles
    # that naturally (exp(-inf) = 0). Use -1e9 instead of -inf if the caller
    # wants to stay strictly HLO-safe; softmax's internal shift by max keeps
    # either variant numerically equivalent here.
    w = _softmax(score.astype(np.float32), axis=2)
    return (kv.astype(np.float32) * w).sum(axis=2)


def compressor_post_pool_freqs_from_table_fn(
    cos_table: np.ndarray,  # [max_seq_len, rd//2] fp32
    sin_table: np.ndarray,  # [max_seq_len, rd//2] fp32
    positions: np.ndarray,  # [clen] or [bsz*clen] int32
    *,
    bsz: int,
    clen: int,
    source_token_positions: bool = False,
    compress_ratio: int = 1,
    start_pos: int = 0,
    seqlen: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather post-pool compressor RoPE rows from device-resident tables."""
    bsz_i = int(bsz)
    clen_i = int(clen)
    n_flat = bsz_i * clen_i
    width = int(cos_table.shape[-1])
    pos = positions.astype(np.int32).reshape(-1)
    if bool(source_token_positions):
        ratio_i = int(compress_ratio)
        if ratio_i <= 0:
            raise RuntimeError(f"compress_ratio must be positive, got {ratio_i}")
        if int(start_pos) == 0:
            seqlen_i = int(seqlen)
            if seqlen_i <= 0:
                raise RuntimeError(
                    "seqlen must be positive when deriving prefill compressor "
                    "positions from token positions"
                )
            needed = bsz_i * seqlen_i
            if int(pos.shape[0]) < needed:
                raise RuntimeError(
                    "compressor token positions too short for prefill: "
                    f"got {int(pos.shape[0])}, need {needed}"
                )
            cutoff = clen_i * ratio_i
            if cutoff > seqlen_i:
                raise RuntimeError(
                    "compressor clen/ratio exceeds seqlen: "
                    f"clen={clen_i}, ratio={ratio_i}, seqlen={seqlen_i}"
                )
            pos = pos[:needed].reshape(bsz_i, seqlen_i)[:, :cutoff:ratio_i]
            pos = pos.reshape(-1)
        else:
            if int(pos.shape[0]) < bsz_i:
                raise RuntimeError(
                    "compressor token positions too short for decode: "
                    f"got {int(pos.shape[0])}, need {bsz_i}"
                )
            pos = pos[:bsz_i] - np.int32(ratio_i - 1)
    cos = cos_table[pos]
    sin = sin_table[pos]
    if int(pos.shape[0]) == clen_i:
        cos = np.broadcast_to(
            cos.reshape(1, clen_i, width),
            (bsz_i, clen_i, width),
        ).reshape(n_flat, width)
        sin = np.broadcast_to(
            sin.reshape(1, clen_i, width),
            (bsz_i, clen_i, width),
        ).reshape(n_flat, width)
    elif int(pos.shape[0]) == n_flat:
        cos = cos.reshape(n_flat, width)
        sin = sin.reshape(n_flat, width)
    else:
        raise RuntimeError(
            "compressor post-pool frequency positions must be [clen] or "
            f"[bsz*clen], got {int(pos.shape[0])} for bsz={bsz_i}, clen={clen_i}"
        )
    return cos.astype(np.float32), sin.astype(np.float32)


def compressor_post_qdq_from_freq_table_fn(
    kv_pool: np.ndarray,  # [bsz * clen, d] fp32
    norm_weight: np.ndarray,  # [d] fp32
    cos_table: np.ndarray,  # [max_seq_len, rd//2] fp32
    sin_table: np.ndarray,  # [max_seq_len, rd//2] fp32
    positions: np.ndarray,  # token positions or compressed positions
    *,
    bsz: int,
    clen: int,
    source_token_positions: bool = False,
    compress_ratio: int = 1,
    start_pos: int = 0,
    seqlen: int = 0,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
    rotate: bool,
    eps: float,
) -> np.ndarray:
    """Fuse compressor post-pool table gather, RMS/RoPE/Hadamard, and qdq.

    This is the product serving tail after the dedicated compressor pool
    kernels and before compressed-KV scatter. It fuses cast, compressor norm,
    post-pool input prep, and compressor qDQ into one bucket-shaped DeviceKernel.
    """
    rd = int(rope_head_dim)
    x_bf = cast_bf16_fn(kv_pool).astype(np.float32)
    normed = apply_rms_norm(
        x_bf,
        norm_weight.astype(np.float32),
        eps=float(eps),
    )
    cos, sin = compressor_post_pool_freqs_from_table_fn(
        cos_table,
        sin_table,
        positions,
        bsz=int(bsz),
        clen=int(clen),
        source_token_positions=bool(source_token_positions),
        compress_ratio=int(compress_ratio),
        start_pos=int(start_pos),
        seqlen=int(seqlen),
    )
    post = q_partial_rope_fn(normed, cos, sin, rope_head_dim=rd)
    if bool(rotate):
        post = hadamard_fn(post)
    return compressor_qdq_bf16_fn(
        post,
        rope_head_dim=rd,
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        rotate=bool(rotate),
    )


def compressor_prefill_post_qdq_from_token_slabs_fn(
    kv_flat: np.ndarray,  # [bsz * seqlen, width] bf16/fp32
    score_flat: np.ndarray,  # [bsz * seqlen, width] bf16/fp32
    ape: np.ndarray,  # [ratio, width] fp32/bf16
    norm_weight: np.ndarray,  # [head_dim] fp32/bf16
    cos_table: np.ndarray,  # [max_seq_len, rd//2] fp32
    sin_table: np.ndarray,  # [max_seq_len, rd//2] fp32
    positions: np.ndarray,  # token positions [bsz * seqlen]
    *,
    bsz: int,
    seqlen: int,
    cutoff: int,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
    rotate: bool,
    overlap: bool,
    eps: float,
) -> np.ndarray:
    """Fuse prefill compressor pool with post-pool RMS/RoPE/qDQ.

    The QKV prologues already materialize ``kv_flat`` and ``score_flat`` on
    device. This helper consumes those slabs directly and emits the bf16 rows
    that the compressed-KV scatter kernel writes to cache.
    """
    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    ratio_i = int(ratio)
    cutoff_i = int(cutoff)
    if ratio_i <= 0 or cutoff_i <= 0 or cutoff_i % ratio_i != 0:
        raise RuntimeError(
            "prefill compressor fusion requires positive cutoff divisible by ratio"
        )
    groups = cutoff_i // ratio_i
    width = int(kv_flat.shape[-1])
    kv = kv_flat.reshape(bsz_i, seqlen_i, width)[:, :cutoff_i, :].astype(np.float32)
    score = score_flat.reshape(bsz_i, seqlen_i, width)[:, :cutoff_i, :].astype(
        np.float32
    )
    kv_group = kv.reshape(bsz_i, groups, ratio_i, width)
    score_group = score.reshape(bsz_i, groups, ratio_i, width) + ape.reshape(
        1,
        1,
        ratio_i,
        width,
    ).astype(np.float32)
    if bool(overlap):
        kv_group = overlap_transform_fn(
            kv_group,
            ratio=ratio_i,
            head_dim=int(head_dim),
            fill_value=0.0,
        )
        score_group = overlap_transform_fn(
            score_group,
            ratio=ratio_i,
            head_dim=int(head_dim),
            fill_value=-1e9,
        )
    pooled = compressor_pool_fn(kv_group, score_group).reshape(
        bsz_i * groups,
        int(head_dim),
    )
    return compressor_post_qdq_from_freq_table_fn(
        pooled,
        norm_weight,
        cos_table,
        sin_table,
        positions,
        bsz=bsz_i,
        clen=int(groups),
        source_token_positions=True,
        compress_ratio=ratio_i,
        start_pos=0,
        seqlen=seqlen_i,
        rope_head_dim=int(rope_head_dim),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
        rotate=bool(rotate),
        eps=float(eps),
    )


def compressor_qdq_bf16_fn(
    post: np.ndarray,  # [..., d] fp32
    *,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
    rotate: bool,
) -> np.ndarray:
    """Compressor post-pool FP8 qdq fused with the bf16 scatter cast.

    On the rotate path the whole tensor is qdq'd (Hadamard spans full
    width). On the no-rotate path only the non-RoPE head is qdq'd and the
    RoPE tail is passed through — matches ``kv_rope_quant_fn``'s
    convention for the attention path. Returns bf16 shaped like ``post``.
    """
    import ml_dtypes as _ml

    if bool(rotate):
        q = fp8_act_qdq_fn(post, block_size=int(block_size), fp8_max=float(fp8_max))
    else:
        q = kv_rope_quant_fn(
            post,
            rope_head_dim=int(rope_head_dim),
            block_size=int(block_size),
            fp8_max=float(fp8_max),
        )
    return q.astype(_ml.bfloat16)


def dp_attention_all_reduce_fn(
    x: np.ndarray,
    *,
    replica_groups: tuple = (),
) -> np.ndarray:
    """Gather lane-scattered attention outputs across the replica's EP rows."""
    groups = [list(group) for group in replica_groups]
    if not groups or all(len(group) <= 1 for group in groups):
        return x
    return _all_reduce_last_dim_preserve_shape(x, replica_groups=groups)


def dp_attention_unpad_reshape_fn(
    x: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    hidden_size: int,
) -> np.ndarray:
    """Drop DP-attention reduce padding and restore ``[bsz, seqlen, hidden]``."""
    n_tokens = int(bsz) * int(seqlen)
    return np.reshape(
        x[:n_tokens],
        (int(bsz), int(seqlen), int(hidden_size)),
    )


def dp_attention_unpad_reshape_mhc_post_pre_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse DP-attention unpad/reshape into the following mHC post/pre stage."""
    out = dp_attention_unpad_reshape_fn(
        x,
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
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


def fp8_act_qdq_fn(
    x: np.ndarray,  # [..., n] fp32
    *,
    block_size: int,
    fp8_max: float,
) -> np.ndarray:
    """FP8 quant-dequant round-trip along the last dim (QAT simulation).

    Reproduces ``fp8_act_quant_inplace(x, block_size, scale_fmt="ue8m0")``
    using HLO-legal primitives (no bit-manipulation). The scale round-
    to-power-of-two uses the natural-log identity (NKIPy's HLO path
    supports ``log`` and ``exp`` but not ``log2``/``exp2``). The FP8
    round casts through ``float8_e4m3`` (Trn2-native IEEE FP8_EXP4,
    range ±240), *not* ``float8_e4m3fn`` — Trn2's compiler refuses to
    lower F8E4M3FN converts.

    Callers pass ``fp8_max=240.0`` to match the Trn2 FP8 range. At
    activation scales the mantissa truncation character is the same
    as OCP e4m3fn, so the QAT simulation is faithful; the exponent
    range is only relevant near ±448 which activations never reach.

    Returns same shape as ``x``, fp32 output after the qdq round trip.
    Invalid/all-zero blocks clip to a minimum amax of 1e-4.
    """
    xf = x.astype(np.float32)
    shape = xf.shape
    n = shape[-1]
    nb = n // int(block_size)
    flat = xf.reshape(*shape[:-1], nb, block_size)
    amax = np.maximum(np.abs(flat).max(axis=-1, keepdims=True), np.float32(1e-4))
    log2 = np.float32(np.log(2.0))
    ratio = amax / np.float32(fp8_max)
    k = np.ceil(np.log(ratio) / log2)
    scale = np.exp(k * log2)
    scaled = np.clip(flat / scale, -np.float32(fp8_max), np.float32(fp8_max))
    # Use float8_e4m3 (Trn2 IEEE FP8_EXP4) rather than float8_e4m3fn —
    # see module docstring.
    import ml_dtypes as _ml

    rounded = scaled.astype(_ml.float8_e4m3).astype(np.float32)
    dequant = (rounded * scale).reshape(shape)
    return dequant


def gate_scores_no_bias_fn(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    score_func: str,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _linear_out_in(x, weight)
    if score_func == "softmax":
        orig = _softmax(scores, axis=-1)
    elif score_func == "sigmoid":
        orig = _sigmoid(scores)
    else:
        orig = sqrtsoftplus(scores)
    # Return two distinct HLO values. Returning ``orig, orig`` can produce
    # aliased output memloc names in neuronx-cc for small hash-gate shapes.
    return orig, orig + np.zeros_like(orig)


def gate_scores_with_bias_fn(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    score_func: str,
) -> tuple[np.ndarray, np.ndarray]:
    orig, _ = gate_scores_no_bias_fn(x, weight, score_func=score_func)
    return orig, orig + bias.astype(np.float32)


def hadamard_fn(x: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard transform along the last axis, normalized by 1/sqrt(d).

    Matches ``eager_ops.hadamard_transform``. Traces as reshape + add/sub +
    stack stages (butterflies of size 2, 4, ..., d); each stage is a fixed
    shape so it lowers cleanly to HLO. ``d`` must be a power of 2.
    """
    original_dtype = x.dtype
    xf = x.astype(np.float32)
    d = int(xf.shape[-1])
    if d & (d - 1):
        raise RuntimeError(f"Hadamard last-dim must be power of 2, got {d}")
    out = xf
    step = 1
    while step < d:
        shape = out.shape
        out = out.reshape(*shape[:-1], d // (2 * step), 2, step)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = np.stack((a + b, a - b), axis=-2).reshape(shape)
        step *= 2
    out = out * np.float32(d**-0.5)
    return out.astype(original_dtype)


def hc_head_flatten_pad_fn(
    x: np.ndarray,
    hc_head_fn_weight: np.ndarray,
    hc_head_scale: np.ndarray,
    hc_head_base: np.ndarray,
    *,
    norm_eps: float,
    hc_eps: float,
    n_tokens: int,
    hidden_size: int,
    rows: int,
) -> np.ndarray:
    """Run sampled-head HC and write directly to flat padded logits rows."""
    hidden = hc_head_fn(
        x,
        hc_head_fn_weight,
        hc_head_scale,
        hc_head_base,
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    return head_hidden_flatten_pad_fn(
        hidden,
        n_tokens=int(n_tokens),
        hidden_size=int(hidden_size),
        rows=int(rows),
    )


def hc_head_fn(
    x: np.ndarray,
    hc_head_fn_weight: np.ndarray,
    hc_head_scale: np.ndarray,
    hc_head_base: np.ndarray,
    *,
    norm_eps: float,
    hc_eps: float,
) -> np.ndarray:
    dtype = x.dtype
    shape = x.shape
    xf = np.reshape(x, (shape[0], shape[1], -1)).astype(np.float32)
    rsqrt = np.float32(1.0) / np.sqrt(
        np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(norm_eps)
    )
    mixes = (xf @ hc_head_fn_weight.astype(np.float32).T) * rsqrt
    pre = _sigmoid(
        mixes * hc_head_scale.astype(np.float32) + hc_head_base.astype(np.float32)
    ) + np.float32(hc_eps)
    y = (pre[..., None] * x.astype(np.float32)).sum(axis=2)
    return y.astype(dtype)


def head_hidden_flatten_pad_fn(
    hidden: np.ndarray,  # [bsz, seqlen, hidden_size]
    *,
    n_tokens: int,
    hidden_size: int,
    rows: int,
) -> np.ndarray:
    """Flatten sampled-head rows and pad directly to a static product bucket."""
    flat = np.reshape(hidden, (int(n_tokens), int(hidden_size)))
    return pad_flat_rows_fn(flat, rows=int(rows))


def indexer_all_kv_topk_decode_from_positions_fn(
    x: np.ndarray,  # [bsz, 1, hidden]
    positions: np.ndarray,  # [bsz] or padded flat positions
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    offset: int,
) -> np.ndarray:
    """Build all-KV decode top-k with invalid padding from runtime positions."""
    if int(seqlen) != 1:
        raise RuntimeError("dynamic-position all-KV top-k is decode-only")
    if int(k) != int(kv_len):
        raise RuntimeError(
            "all-KV decode top-k requires k == kv_len, got "
            f"k={int(k)} kv_len={int(kv_len)}"
        )
    b = int(bsz)
    r = np.int32(ratio)
    off = np.int32(offset)
    width = int(kv_len)
    pos = positions.astype(np.int32).reshape(-1)
    sp = pos[:1].reshape(1, 1)
    sp1 = sp + np.int32(1)
    n = (sp1.astype(np.float32) * (np.float32(1.0) / np.float32(r))).astype(np.int32)
    idx = np.arange(width, dtype=np.int32).reshape(1, width)
    row = np.where(idx < n, idx + off, np.int32(-1))
    anchor = x[:, :1, :1].astype(np.int32) * np.int32(0)
    return np.broadcast_to(row.reshape(1, 1, width), (b, 1, width)) + anchor


def indexer_all_kv_topk_static_fn(
    x: np.ndarray,  # [bsz, seqlen, hidden]
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    offset: int,
    prefill: bool,
) -> np.ndarray:
    """Build compressed top-k when every compressed KV row is selected.

    For ``k == kv_len`` the indexer score only changes ordering, not the
    selected set. Sparse attention is permutation-invariant over the selected
    KV rows, so this path emits the whole compressed range and applies the
    same causal rebase/mask as the score-based path.
    """
    if int(k) != int(kv_len):
        raise RuntimeError(
            "all-KV indexer top-k requires k == kv_len, got "
            f"k={int(k)} kv_len={int(kv_len)}"
        )
    b = int(bsz)
    s = int(seqlen)
    t = int(kv_len)
    base = np.arange(t, dtype=np.int32).reshape(1, 1, t)
    anchor = x[:, :, :1].astype(np.int32) * np.int32(0)
    topk = np.broadcast_to(base, (b, s, t)) + anchor
    return topk_rebase_static_fn(
        topk,
        seqlen=s,
        ratio=int(ratio),
        offset=int(offset),
        prefill=bool(prefill),
    )


def indexer_compressor_kv_score_project_qw_prep_from_freq_table_fn(
    x: np.ndarray,
    wkv: np.ndarray,
    wgate: np.ndarray,
    qr: np.ndarray,
    wq_b: np.ndarray,
    weights_proj: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    score_scale: float,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse indexer compressor projection with table-backed Q/W prep.

    Returns compressor input slabs followed by indexer score inputs:
    ``kv_bf [B*S, W]``, ``score_bf [B*S, W]``, ``q_T [B*S, d, h]``,
    and ``w_flat [B*S, h]``.
    """
    kv_bf, score_bf = compressor_kv_score_bf16_fn(x, wkv, wgate)
    q_t, w_flat = indexer_project_qw_prep_from_freq_table_fn(
        qr,
        wq_b,
        x,
        weights_proj,
        cos_table,
        sin_table,
        positions,
        score_scale=float(score_scale),
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
    )
    return kv_bf, score_bf, q_t, w_flat


def indexer_project_fn(
    qr: np.ndarray,
    wq_b: np.ndarray,
    x: np.ndarray,
    weights_proj: np.ndarray,
    *,
    score_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = _linear_out_in(qr, wq_b)
    w = _linear_out_in(x.astype(np.float32), weights_proj) * np.float32(score_scale)
    return q, w


def indexer_project_qw_prep_fn(
    qr: np.ndarray,
    wq_b: np.ndarray,
    x: np.ndarray,
    weights_proj: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    score_scale: float,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project indexer q/weights and prepare NKI score-kernel inputs.

    Fuses the former project -> q reshape -> q transform -> q/w prep chain.
    The outputs match ``indexer_score_qw_prep_fn``: ``q_T [B, d, h]`` bf16 and
    ``w_flat [B, h]`` fp32.
    """
    q_flat, weights = indexer_project_fn(
        qr,
        wq_b,
        x,
        weights_proj,
        score_scale=float(score_scale),
    )
    bsz, seqlen, _ = q_flat.shape
    q = np.reshape(
        q_flat,
        (int(bsz), int(seqlen), int(n_heads), int(head_dim)),
    )
    q_dev = indexer_q_transform_fn(
        q,
        cos,
        sin,
        rope_head_dim=int(rope_head_dim),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
    )
    return indexer_score_qw_prep_fn(q_dev, weights)


def indexer_project_qw_prep_from_freq_table_fn(
    qr: np.ndarray,
    wq_b: np.ndarray,
    x: np.ndarray,
    weights_proj: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    positions: np.ndarray,
    *,
    score_scale: float,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Indexer project/Q-transform/QW-prep using device-resident RoPE tables."""
    pos = positions.astype(np.int32).reshape(-1)
    token_count = int(x.shape[0]) * int(x.shape[1])
    if int(pos.shape[0]) > token_count:
        pos = pos[:token_count]
    cos = cos_table[pos]
    sin = sin_table[pos]
    return indexer_project_qw_prep_fn(
        qr,
        wq_b,
        x,
        weights_proj,
        cos,
        sin,
        score_scale=float(score_scale),
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        rope_head_dim=int(rope_head_dim),
        block_size=int(block_size),
        fp8_max=float(fp8_max),
    )


def indexer_q_transform_fn(
    q: np.ndarray,  # [..., head_dim]
    cos: np.ndarray,  # [s, rope_head_dim // 2]
    sin: np.ndarray,  # [s, rope_head_dim // 2]
    *,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
) -> np.ndarray:
    """Fused indexer q transform.

    Chains ``q_partial_rope_fn`` → ``hadamard_fn`` → ``fp8_act_qdq_fn`` in
    a single traceable fragment so intermediate q tensors never round-trip
    to host. Replaces three separate trace-function calls
    (``q_partial_rope`` + ``hadamard`` + ``fp8_act_qdq``) with one.
    """
    q_roped = q_partial_rope_fn(q, cos, sin, rope_head_dim=rope_head_dim)
    q_had = hadamard_fn(q_roped)
    return fp8_act_qdq_fn(q_had, block_size=block_size, fp8_max=fp8_max)


def indexer_score_qw_prep_fn(
    q: np.ndarray,  # [bsz, seqlen, h, d] fp32
    w: np.ndarray,  # [bsz, seqlen, h] fp32
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare q/weights for the ``indexer_score_from_cache`` NKI kernel.

    Produces:
    - ``q_T [B=bsz*seqlen, d, h]`` bf16, the kernel's ``d``-on-partition layout.
    - ``w_flat [B, h]`` fp32, flattened per-token indexer weights.

    Fuses the host reshape/transpose/cast pair that used to round-trip each
    layer's ``indexer_q_transform`` output through numpy before re-uploading.
    """
    import ml_dtypes as _ml

    bsz, seqlen, h, d = q.shape
    n = int(bsz) * int(seqlen)
    q_bf = q.reshape(n, h, d).astype(_ml.bfloat16)
    q_T = q_bf.transpose(0, 2, 1)  # [B, d, h]
    w_flat = w.reshape(n, int(w.shape[-1])).astype(np.float32)  # [B, h]
    return q_T, w_flat


def indexer_sparse_attention_prep_all_kv_decode_from_positions_fn(
    x: np.ndarray,  # [bsz, 1, hidden]
    positions: np.ndarray,  # [bsz] or padded flat positions
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    offset: int,
    prefill: bool,
    window_size: int,
    start_pos: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode all-KV sparse-prep with runtime-position SWA top-k."""
    del start_pos
    if bool(prefill) or int(seqlen) != 1:
        raise RuntimeError("all-KV dynamic-position indexer prep is decode-only")
    topk_win = window_topk_decode_from_positions_fn(
        x,
        positions,
        window_size=int(window_size),
    )
    topk_comp = indexer_all_kv_topk_decode_from_positions_fn(
        x,
        positions,
        bsz=int(bsz),
        seqlen=int(seqlen),
        kv_len=int(kv_len),
        k=int(k),
        ratio=int(ratio),
        offset=int(offset),
    )
    return topk_concat_pad_sparse_attention_prep_fn(
        topk_win,
        topk_comp,
        rows=int(rows),
        k_tile=int(k_tile),
    )


def indexer_sparse_attention_prep_all_kv_static_fn(
    x: np.ndarray,  # [bsz, seqlen, hidden]
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    offset: int,
    prefill: bool,
    window_size: int,
    start_pos: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse-attention prep for the ``index_topk >= kv_len`` path."""
    topk_win = window_topk_from_tokens_fn(
        x,
        window_size=int(window_size),
        start_pos=int(start_pos),
    )
    topk_comp = indexer_all_kv_topk_static_fn(
        x,
        bsz=int(bsz),
        seqlen=int(seqlen),
        kv_len=int(kv_len),
        k=int(k),
        ratio=int(ratio),
        offset=int(offset),
        prefill=bool(prefill),
    )
    return topk_concat_pad_sparse_attention_prep_fn(
        topk_win,
        topk_comp,
        rows=int(rows),
        k_tile=int(k_tile),
    )


def invalid_topk_from_tokens_fn(
    x: np.ndarray,  # [b, s, ...]
    *,
    k: int,
) -> np.ndarray:
    """Build an all-invalid top-k tensor using token dimensions from ``x``."""
    b, s = x.shape[:2]
    anchor = x[:, :, :1].astype(np.int32) * np.int32(0)
    return np.full((int(b), int(s), int(k)), -1, dtype=np.int32) + anchor


def kv_rope_quant_fn(
    kv: np.ndarray,  # [..., d]
    *,
    rope_head_dim: int,
    block_size: int,
    fp8_max: float,
) -> np.ndarray:
    """Fuse post-QKV FP8 qdq on the non-RoPE legs and keep the RoPE tail.

    Replaces the host ``kv[..., :-rd] = fp8_act_quant_inplace(...)`` in the
    attention backend path so the qdq lands on device alongside the rest of
    the attention fragments.
    """
    rd = int(rope_head_dim)
    if rd == 0:
        return fp8_act_qdq_fn(kv, block_size=block_size, fp8_max=fp8_max)
    head = kv[..., :-rd]
    tail = kv[..., -rd:]
    head_qdq = fp8_act_qdq_fn(head, block_size=block_size, fp8_max=fp8_max)
    return np.concatenate([head_qdq, tail.astype(head_qdq.dtype)], axis=-1)


def mhc_post_fn(
    out: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
) -> np.ndarray:
    y = post[..., None].astype(np.float32) * out[..., None, :].astype(np.float32) + (
        comb[..., None].astype(np.float32)
        * residual[..., None, :, :].astype(np.float32)
    ).sum(axis=2)
    return y.astype(residual.dtype)


def mhc_post_hc_head_flatten_pad_fn(
    out: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_head_fn_weight: np.ndarray,
    hc_head_scale: np.ndarray,
    hc_head_base: np.ndarray,
    *,
    norm_eps: float,
    hc_eps: float,
    n_tokens: int,
    hidden_size: int,
    rows: int,
) -> np.ndarray:
    """Fuse final mHC post into sampled-head HC and flat token-bucket output."""
    h = mhc_post_fn(out, residual, post, comb)
    return hc_head_flatten_pad_fn(
        h,
        hc_head_fn_weight,
        hc_head_scale,
        hc_head_base,
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
        n_tokens=int(n_tokens),
        hidden_size=int(hidden_size),
        rows=int(rows),
    )


def mhc_post_pre_fn(
    out: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse an mHC post boundary into the next mHC pre boundary."""
    h = mhc_post_fn(out, residual, post, comb)
    y, next_post, next_comb = mhc_pre_fn(
        h,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    return h, y, next_post, next_comb


def mhc_pre_apply_fn(
    x: np.ndarray,
    pre: np.ndarray,
    norm_weight: np.ndarray,
    *,
    norm_eps: float,
) -> np.ndarray:
    """mHC pre-apply fused with the downstream RMS norm.

    The per-layer ``attn_norm``/``ffn_norm`` RMS weight is folded in here so
    the mHC block's output is already normalized. 2 × 43 numpy RMS calls per
    forward collapse into the mHC device fragment.
    """
    yf = (pre[..., None].astype(np.float32) * x.astype(np.float32)).sum(axis=2)
    nw = np.reshape(
        norm_weight.astype(np.float32),
        (1,) * (yf.ndim - 1) + (-1,),
    )
    inv_rms = np.float32(1.0) / np.sqrt(
        np.mean(yf * yf, axis=-1, keepdims=True) + np.float32(norm_eps)
    )
    out = yf * inv_rms * nw
    return out.astype(x.dtype)


def mhc_pre_fn(
    x: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused mHC pre-stage for product execution.

    The graph path keeps this split into gemm/mix/apply fragments so each
    piece can be inspected independently. Product starts fusing at this
    boundary because the whole stage is pure tensor math and has no
    attention/MoE scheduler side effects.
    """
    mixes = mhc_pre_gemm_fn(x, hc_fn, norm_eps=float(norm_eps))
    pre, post, comb = mhc_pre_mix_sinkhorn_fn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        hc_eps=float(hc_eps),
    )
    y = mhc_pre_apply_fn(
        x,
        pre,
        norm_weight,
        norm_eps=float(norm_eps),
    )
    return y, post, comb


def mhc_pre_gemm_fn(
    x: np.ndarray,
    hc_fn: np.ndarray,
    *,
    norm_eps: float,
) -> np.ndarray:
    shape = x.shape
    xf = np.reshape(x, (shape[0], shape[1], -1)).astype(np.float32)
    rsqrt = np.float32(1.0) / np.sqrt(
        np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(norm_eps)
    )
    return (xf @ hc_fn.astype(np.float32).T) * rsqrt


def mhc_pre_mix_sinkhorn_fn(
    mixes: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, s, _ = mixes.shape
    pre, post, comb = hc_split_sinkhorn(
        np.reshape(mixes, (b * s, -1)),
        hc_scale,
        hc_base,
        int(hc_mult),
        int(sinkhorn_iters),
        float(hc_eps),
    )
    return (
        np.reshape(pre, (b, s, int(hc_mult))),
        np.reshape(post, (b, s, int(hc_mult))),
        np.reshape(comb, (b, s, int(hc_mult), int(hc_mult))),
    )


def overlap_transform_fn(
    tensor: np.ndarray,  # [b, s, r, 2*head_dim]  fp32
    *,
    ratio: int,
    head_dim: int,
    fill_value: float,
) -> np.ndarray:
    """Compressor.overlap_transform (c4a prefill path).

    ``[b, s, r, 2*head_dim] -> [b, s, 2*r, head_dim]`` with the bottom
    half copied from the top half of the previous position (sliding
    overlap). Pure slice + concat; HLO-traceable.
    """
    b, s, r, _ = tensor.shape
    d = int(head_dim)
    # new[b, s, R:, :] = tensor[b, s, :, d:]                (top->bottom)
    # new[b, s, :R, :] has tensor[b, s-1, :, :d] for s>=1; fill_value for s=0.
    top_of_curr = tensor[..., d:]  # [b, s, r, d]
    prev_bottom = np.roll(tensor[..., :d], shift=1, axis=1)  # [b, s, r, d]
    # For s=0, roll wrapped in garbage from position s=-1; overwrite with fill.
    first = np.full(prev_bottom[:, :1].shape, fill_value, dtype=prev_bottom.dtype)
    prev_bottom = np.concatenate([first, prev_bottom[:, 1:]], axis=1)
    # Concatenate along the r axis: [b, s, 2*r, d].
    out = np.concatenate([prev_bottom, top_of_curr], axis=2)
    return out


def pad_flat_rows_fn(
    x: np.ndarray,  # [n, d]
    *,
    rows: int,
) -> np.ndarray:
    """Pad a flat row-major tensor to a static row count on device."""
    target = int(rows)
    n, d = x.shape
    if int(n) > target:
        raise ValueError(f"input rows {int(n)} exceed target rows {target}")
    if int(n) == target:
        return x
    pad = np.zeros((target - int(n), int(d)), dtype=x.dtype)
    return np.concatenate((x, pad), axis=0)


def pad_router_rows_fn(
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad router weights and top-k indices in one static-bucket fragment."""
    return (
        pad_flat_rows_fn(weights, rows=int(rows)),
        pad_topk_rows_fn(indices, rows=int(rows)),
    )


def pad_topk_rows_fn(
    topk: np.ndarray,
    *,
    rows: int,
) -> np.ndarray:
    """Flatten leading token dims and pad top-k rows with invalid entries."""
    target = int(rows)
    k = int(topk.shape[-1])
    flat = np.reshape(topk, (-1, k)).astype(np.int32)
    n = int(flat.shape[0])
    if n > target:
        raise ValueError(f"topk rows {n} exceed target rows {target}")
    if n == target:
        return flat
    pad = np.full((target - n, k), -1, dtype=np.int32)
    return np.concatenate((flat, pad), axis=0)


def q_partial_rope_fn(
    q: np.ndarray,  # [..., head_dim]
    cos: np.ndarray,  # [s, rope_head_dim // 2]
    sin: np.ndarray,  # [s, rope_head_dim // 2]
    *,
    rope_head_dim: int,
) -> np.ndarray:
    """Apply interleaved RoPE to the trailing ``rope_head_dim`` slice of ``q``.

    Replaces the host mutation ``q[..., -rd:] = apply_rotary_emb(...)`` in
    the indexer / compressor post-pool paths.
    """
    rd = int(rope_head_dim)
    if rd == 0:
        return q
    head = q[..., :-rd]
    tail = _apply_interleaved_rope(q[..., -rd:], cos, sin)
    return np.concatenate((head, tail.astype(head.dtype)), axis=-1)


def sequence_hidden_pad_fn(
    x: np.ndarray,  # [bsz, seqlen, dim]
    *,
    rows: int,
    dim: int,
) -> np.ndarray:
    """Pad token rows and expose them as one static sequence."""
    hidden = int(dim)
    target = int(rows)
    flat = np.reshape(x, (-1, hidden))
    n = int(flat.shape[0])
    if n > target:
        raise ValueError(f"hidden rows {n} exceed target rows {target}")
    if n < target:
        pad = np.zeros((target - n, hidden), dtype=x.dtype)
        flat = np.concatenate((flat, pad), axis=0)
    return np.reshape(flat, (1, target, hidden))


def topk_concat_fn(
    topk_win: np.ndarray,  # [..., K_win] int32
    topk_comp: np.ndarray,  # [..., K_comp] int32
) -> np.ndarray:
    """Concat SWA and compressed top-k tails along the last axis."""
    return np.concatenate(
        [topk_win.astype(np.int32), topk_comp.astype(np.int32)],
        axis=-1,
    )


def topk_concat_pad_sparse_attention_prep_fn(
    topk_win: np.ndarray,  # [..., K_win] int32
    topk_comp: np.ndarray,  # [..., K_comp] int32
    *,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Product fused top-k tail for sparse attention.

    Combines the serving hot-path sequence:
    ``topk_concat`` -> optional row padding -> sparse-attention prep.
    The output contract is identical to ``topk_sparse_attention_prep_fn``.
    """
    topk = topk_concat_fn(topk_win, topk_comp)
    topk = pad_topk_rows_fn(topk, rows=int(rows))
    return topk_sparse_attention_prep_fn(topk, k_tile=int(k_tile))


def topk_rebase_fn(
    topk: np.ndarray,  # [b, s, k] int32
    seqlen: np.ndarray,  # [s] int32 positions (1..seqlen or row count)
    *,
    ratio: int,
    offset: int,
    prefill: bool,
) -> np.ndarray:
    """Rebase indexer top-k indices onto the global KV axis.

    Prefill: invalidate entries whose compressed-row index is at or past
    the current token's row (``topk >= row``), then add ``offset``.
    Decode: simply add ``offset`` to every entry.
    Fuses ``sampled_forward.py:538-543``.
    """
    t = topk.astype(np.int32)
    if not bool(prefill):
        return t + np.int32(offset)
    # Integer division lowers to ``floor`` on device, which doesn't accept
    # int operands; do the divide in fp32 and cast back.
    ratio_f = np.float32(ratio)
    row = np.floor(seqlen.astype(np.float32) / ratio_f).astype(np.int32)[
        :, None
    ]  # [s, 1]
    invalid = t >= row[None, :, :]  # [b, s, k]
    return np.where(invalid, np.int32(-1), t + np.int32(offset))


def topk_rebase_static_fn(
    topk: np.ndarray,  # [b, s, k] int32
    *,
    seqlen: int,
    ratio: int,
    offset: int,
    prefill: bool,
) -> np.ndarray:
    """Static-seqlen product variant of ``topk_rebase_fn``.

    The generic graph path passes a host ``seqlen`` vector. Product mode uses this
    variant so the rebase tail has only the device top-k tensor as runtime
    input; the row vector is generated inside the traced graph from static
    bucket metadata.
    """
    s = int(seqlen)
    row_ids = np.arange(1, s + 1, dtype=np.int32)
    return topk_rebase_fn(
        topk,
        row_ids,
        ratio=int(ratio),
        offset=int(offset),
        prefill=bool(prefill),
    )


def topk_rebase_dynamic_offset_fn(
    topk: np.ndarray,  # [b, s, k] int32
    seqlen: np.ndarray,  # [s] int32 positions (1..seqlen or row count)
    offset: np.ndarray,  # [1, 1] int32 runtime scalar
    *,
    ratio: int,
    prefill: bool,
) -> np.ndarray:
    """Rebase indexer top-k using a runtime offset scalar."""
    t = topk.astype(np.int32)
    offset_v = offset.astype(np.int32).reshape(-1)[:1].reshape(1, 1)
    if not bool(prefill):
        return t + offset_v
    ratio_f = np.float32(ratio)
    row = np.floor(seqlen.astype(np.float32) / ratio_f).astype(np.int32)[:, None]
    invalid = t >= row[None, :, :]
    return np.where(invalid, np.int32(-1), t + offset_v)


def topk_rebase_static_dynamic_offset_fn(
    topk: np.ndarray,  # [b, s, k] int32
    offset: np.ndarray,  # [1, 1] int32 runtime scalar
    *,
    seqlen: int,
    ratio: int,
    prefill: bool,
) -> np.ndarray:
    """Static-seqlen variant of ``topk_rebase_dynamic_offset_fn``."""
    s = int(seqlen)
    row_ids = np.arange(1, s + 1, dtype=np.int32)
    return topk_rebase_dynamic_offset_fn(
        topk,
        row_ids,
        offset,
        ratio=int(ratio),
        prefill=bool(prefill),
    )


def topk_sparse_attention_prep_fn(
    topk: np.ndarray,  # [..., K_raw] int32 with -1 sentinels
    *,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the backend's host-side topk prep into one device fragment.

    Accepts ``topk`` with any leading dims (e.g. ``[bsz, seqlen, K_raw]``
    from ``topk_concat``), flattens leading dims, pads the last axis to
    a multiple of ``k_tile``, and produces the two inputs the sparse-
    attention kernels expect:

    - ``topk_T [K, N_q]`` int32, with ``K`` rounded up to ``k_tile`` and
      negative / padded entries safe-clamped to 0.
    - ``mask_bf [N_q, K]`` bfloat16, ``1.0`` where the original topk was
      non-negative, ``0.0`` where it was ``-1`` (or pad).

    Lets ``topk_concat`` output flow directly as a DeviceTensor into the
    attention kernels without a host round-trip.
    """
    import ml_dtypes as _ml

    topk_i = topk.astype(np.int32)
    k_raw = int(topk_i.shape[-1])
    topk_flat = topk_i.reshape(-1, k_raw)
    n_q = topk_flat.shape[0]
    kt = int(k_tile)
    k_padded = ((k_raw + kt - 1) // kt) * kt
    if k_padded != k_raw:
        pad = np.full((n_q, k_padded - k_raw), -1, dtype=np.int32)
        topk_flat = np.concatenate([topk_flat, pad], axis=-1)
    valid = topk_flat >= 0
    safe = np.where(valid, topk_flat, 0).astype(np.int32)
    topk_t = safe.T.astype(np.int32)  # [K, N_q]
    mask_bf = valid.astype(np.float32).astype(_ml.bfloat16)  # [N_q, K]
    return topk_t, mask_bf


def topk_tokens_concat_pad_sparse_attention_prep_decode_from_positions_fn(
    x: np.ndarray,  # [b, 1, ...]
    positions: np.ndarray,  # [b] or padded flat positions
    *,
    window_size: int,
    ratio: int,
    offset: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode-only token top-k generation with runtime device positions."""
    topk_win = window_topk_decode_from_positions_fn(
        x,
        positions,
        window_size=int(window_size),
    )
    if int(max_c_len) <= 0:
        topk_comp = invalid_topk_from_tokens_fn(x, k=1)
    else:
        topk_comp = compressed_topk_no_indexer_decode_from_positions_fn(
            x,
            positions,
            ratio=int(ratio),
            offset=int(offset),
            max_c_len=int(max_c_len),
        )
    return topk_concat_pad_sparse_attention_prep_fn(
        topk_win,
        topk_comp,
        rows=int(rows),
        k_tile=int(k_tile),
    )


def topk_tokens_concat_pad_sparse_attention_prep_fn(
    x: np.ndarray,  # [b, s, ...]
    *,
    window_size: int,
    ratio: int,
    offset: int,
    start_pos: int,
    max_c_len: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Product fused token-derived top-k generation plus sparse-attn prep.

    This covers the no-indexer compressed-attention path where both the SWA
    window top-k and compressed top-k are static functions of token shape and
    step position. Indexer-produced top-k still uses the generic concat/prep
    function because its compressed side is data-dependent.
    """
    topk_win = window_topk_from_tokens_fn(
        x,
        window_size=int(window_size),
        start_pos=int(start_pos),
    )
    if (int(start_pos) == 0 and int(x.shape[1]) // int(ratio) == 0) or (
        int(start_pos) > 0 and int(max_c_len) <= 0
    ):
        topk_comp = invalid_topk_from_tokens_fn(x, k=1)
    else:
        topk_comp = compressed_topk_no_indexer_from_tokens_fn(
            x,
            ratio=int(ratio),
            offset=int(offset),
            start_pos=int(start_pos),
            max_c_len=int(max_c_len),
        )
    return topk_concat_pad_sparse_attention_prep_fn(
        topk_win,
        topk_comp,
        rows=int(rows),
        k_tile=int(k_tile),
    )


def window_topk_decode_from_positions_fn(
    x: np.ndarray,  # [b, 1, ...]
    positions: np.ndarray,  # [b] or padded flat positions
    *,
    window_size: int,
) -> np.ndarray:
    """Decode SWA window top-k using runtime device positions.

    The static decode variant bakes ``start_pos`` into the NEFF. Product
    serving already has per-token positions on device, so the decode-only
    fused path derives the uniform step from ``positions[0]`` and reuses one
    kernel across decode steps with identical tensor shapes.
    """
    b = int(x.shape[0])
    win = int(window_size)
    pos = positions.astype(np.int32).reshape(-1)
    sp = pos[:1].reshape(1, 1)
    idx = np.arange(win, dtype=np.int32).reshape(1, win)
    # NK compiler rejects integer floor used by numpy's dynamic mod/div lowering.
    # Positions are non-negative and < max context, so fp32 reciprocal truncation
    # is exact for these small positive integers.
    inv_win = np.float32(1.0) / np.float32(win)
    sp_div = (sp.astype(np.float32) * inv_win).astype(np.int32)
    sp_mod = sp - sp_div * np.int32(win)
    shifted = idx + sp_mod + np.int32(1)
    shifted_div = (shifted.astype(np.float32) * inv_win).astype(np.int32)
    warmed_val = shifted - shifted_div * np.int32(win)
    unwarmed_val = np.where(idx <= sp, idx, np.int32(-1))
    row = np.where(sp >= np.int32(win - 1), warmed_val, unwarmed_val)
    anchor = x[:, :1, :1].astype(np.int32) * np.int32(0)
    return np.broadcast_to(row.reshape(1, 1, win), (b, 1, win)) + anchor


def window_topk_from_tokens_fn(
    x: np.ndarray,  # [b, s, ...]
    *,
    window_size: int,
    start_pos: int,
) -> np.ndarray:
    """Build SWA window top-k on device using token dimensions from ``x``."""
    b, s = x.shape[:2]
    win = int(window_size)
    sp = int(start_pos)
    if sp > 0:
        if sp >= win - 1:
            spw = sp % win
            row = np.concatenate(
                (
                    np.arange(spw + 1, win, dtype=np.int32),
                    np.arange(0, spw + 1, dtype=np.int32),
                ),
                axis=0,
            )
        else:
            idx = np.arange(win, dtype=np.int32)
            row = np.where(idx <= np.int32(sp), idx, np.int32(-1))
        anchor = x[:, :1, :1].astype(np.int32) * np.int32(0)
        return np.broadcast_to(row.reshape(1, 1, win), (int(b), 1, win)) + anchor

    k = min(int(s), win)
    base = np.arange(int(s), dtype=np.int32)[:, None]
    vals = (
        np.maximum(base - np.int32(win - 1), np.int32(0))
        + np.arange(
            k,
            dtype=np.int32,
        )[None, :]
    )
    vals = np.where(vals > base, np.int32(-1), vals)
    anchor = x[:, :, :1].astype(np.int32) * np.int32(0)
    return np.broadcast_to(vals[None, :, :], (int(b), int(s), k)) + anchor


def embedding_hc_fn(
    input_ids: np.ndarray,
    embeddings: np.ndarray,
    *,
    hc_mult: int,
) -> np.ndarray:
    """Embedding lookup plus HC-stack expansion: ``[b, s] -> [b, s, hc, dim]``."""
    h = embeddings[input_ids.astype(np.int32)]
    return np.broadcast_to(
        h[:, :, None, :],
        (h.shape[0], h.shape[1], int(hc_mult), h.shape[-1]),
    ).copy()


def vocab_parallel_embedding_hc_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
    hc_mult: int,
) -> np.ndarray:
    h = vocab_parallel_embedding_no_sp_fn(
        input_ids.astype(np.int32),
        local_embeddings,
        vocab_start_index=int(vocab_start_index),
        vocab_end_index=int(vocab_end_index),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    return np.broadcast_to(
        h[:, :, None, :],
        (h.shape[0], h.shape[1], int(hc_mult), h.shape[-1]),
    ).copy()


def embedding_hc_unpad_fn(
    hidden: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
) -> np.ndarray:
    """Slice canonical padded HC embeddings back to the active rectangle."""
    return hidden[: int(bsz), : int(seqlen), :, :].copy()


def rms_norm_fn(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    return apply_rms_norm(x, weight, eps=eps)


def linear_fn(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _linear_out_in(x, w)


def attention_zero_like_fn(x: np.ndarray) -> np.ndarray:
    """Return a zero attention output matching the float32 projection contract."""
    return x.astype(np.float32) * np.float32(0.0)


def two_linear_fn(
    x: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _linear_out_in(x, w_a), _linear_out_in(x, w_b)


def vocab_parallel_embedding_hc_mhc_pre_from_ids_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vocab-parallel embedding directly into the first layer mHC pre."""
    ids = input_ids[: int(bsz), : int(seqlen)].astype(np.int32)
    h = vocab_parallel_embedding_no_sp_fn(
        ids,
        local_embeddings,
        vocab_start_index=int(vocab_start_index),
        vocab_end_index=int(vocab_end_index),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    active = np.broadcast_to(
        h[:, :, None, :],
        (int(bsz), int(seqlen), int(hc_mult), h.shape[-1]),
    ).copy()
    y, post, comb = mhc_pre_fn(
        active,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    return active, y, post, comb


def vocab_parallel_embedding_hc_mhc_pre_from_ids_dynamic_range_fn(
    input_ids: np.ndarray,
    local_embeddings: np.ndarray,
    vocab_range: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vocab-parallel embedding+mHC-pre with runtime TP vocab range.

    Keeping the local vocab range as a tiny tensor input lets TP ranks share
    one compiled NEFF for the same shape instead of specializing on static
    ``vocab_start_index``/``vocab_end_index`` constants.
    """
    ids = input_ids[: int(bsz), : int(seqlen)].astype(np.int32)
    h = vocab_parallel_embedding_no_sp_dynamic_range_fn(
        ids,
        local_embeddings,
        vocab_range,
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )
    active = np.broadcast_to(
        h[:, :, None, :],
        (int(bsz), int(seqlen), int(hc_mult), h.shape[-1]),
    ).copy()
    y, post, comb = mhc_pre_fn(
        active,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    return active, y, post, comb


__all__ = [
    "_all_reduce_last_dim_preserve_shape",
    "_apply_interleaved_rope",
    "_linear_out_in",
    "_sigmoid",
    "_softmax",
    "attention_zero_like_fn",
    "cast_bf16_fn",
    "compressed_topk_no_indexer_decode_from_positions_fn",
    "compressed_topk_no_indexer_from_tokens_fn",
    "compressor_decode_pool_from_state_plus_current_fn",
    "compressor_decode_pool_post_qdq_from_state_plus_current_freq_table_fn",
    "compressor_kv_score_bf16_fn",
    "compressor_kv_score_token_topk_prep_decode_from_positions_fn",
    "compressor_kv_score_token_topk_prep_fn",
    "compressor_pool_fn",
    "compressor_post_pool_freqs_from_table_fn",
    "compressor_post_qdq_from_freq_table_fn",
    "compressor_prefill_post_qdq_from_token_slabs_fn",
    "compressor_qdq_bf16_fn",
    "dp_attention_all_reduce_fn",
    "dp_attention_unpad_reshape_fn",
    "dp_attention_unpad_reshape_mhc_post_pre_fn",
    "embedding_hc_fn",
    "embedding_hc_unpad_fn",
    "fp8_act_qdq_fn",
    "gate_scores_no_bias_fn",
    "gate_scores_with_bias_fn",
    "hadamard_fn",
    "hc_head_flatten_pad_fn",
    "hc_head_fn",
    "head_hidden_flatten_pad_fn",
    "indexer_all_kv_topk_decode_from_positions_fn",
    "indexer_all_kv_topk_static_fn",
    "indexer_compressor_kv_score_project_qw_prep_from_freq_table_fn",
    "indexer_project_fn",
    "indexer_project_qw_prep_fn",
    "indexer_project_qw_prep_from_freq_table_fn",
    "indexer_q_transform_fn",
    "indexer_score_qw_prep_fn",
    "indexer_sparse_attention_prep_all_kv_decode_from_positions_fn",
    "indexer_sparse_attention_prep_all_kv_static_fn",
    "invalid_topk_from_tokens_fn",
    "kv_rope_quant_fn",
    "linear_fn",
    "mhc_post_fn",
    "mhc_post_hc_head_flatten_pad_fn",
    "mhc_post_pre_fn",
    "mhc_pre_apply_fn",
    "mhc_pre_fn",
    "mhc_pre_gemm_fn",
    "mhc_pre_mix_sinkhorn_fn",
    "overlap_transform_fn",
    "pad_flat_rows_fn",
    "pad_router_rows_fn",
    "pad_topk_rows_fn",
    "q_partial_rope_fn",
    "rms_norm_fn",
    "sequence_hidden_pad_fn",
    "topk_concat_fn",
    "topk_concat_pad_sparse_attention_prep_fn",
    "topk_rebase_dynamic_offset_fn",
    "topk_rebase_fn",
    "topk_rebase_static_dynamic_offset_fn",
    "topk_rebase_static_fn",
    "topk_sparse_attention_prep_fn",
    "topk_tokens_concat_pad_sparse_attention_prep_decode_from_positions_fn",
    "topk_tokens_concat_pad_sparse_attention_prep_fn",
    "two_linear_fn",
    "vocab_parallel_embedding_hc_fn",
    "vocab_parallel_embedding_hc_mhc_pre_from_ids_dynamic_range_fn",
    "vocab_parallel_embedding_hc_mhc_pre_from_ids_fn",
    "window_topk_decode_from_positions_fn",
    "window_topk_from_tokens_fn",
]
