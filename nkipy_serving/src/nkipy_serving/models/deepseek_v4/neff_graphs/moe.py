"""DeepSeek-V4 MoE / compressor / indexer trace functions.

Merged from compressor_topk + router_indexer + moe_head + dp_moe. Pure HLO-traceable;
bodies byte-identical to pre-merge. Imports primitives from graphs.common only."""

from __future__ import annotations

import numpy as np

from nkipy_serving.models.deepseek_v4.eager_ops import swiglu_with_limit
from nkipy_serving.models.deepseek_v4.neff_graphs.common import (
    _all_reduce_last_dim_preserve_shape,
    _apply_interleaved_rope,
    _linear_out_in,
    _softmax,
    compressor_post_qdq_from_freq_table_fn,
    dp_attention_all_reduce_fn,
    dp_attention_unpad_reshape_mhc_post_pre_fn,
    gate_scores_no_bias_fn,
    gate_scores_with_bias_fn,
    hc_head_fn,
    mhc_post_fn,
    mhc_post_hc_head_flatten_pad_fn,
    mhc_post_pre_fn,
    pad_flat_rows_fn,
    pad_router_rows_fn,
    sequence_hidden_pad_fn,
    topk_concat_pad_sparse_attention_prep_fn,
    topk_rebase_static_dynamic_offset_fn,
    topk_rebase_static_fn,
    window_topk_decode_from_positions_fn,
    window_topk_from_tokens_fn,
)
from nkipy_serving.ops.nn import apply_rms_norm


def hash_moe_dispatch_no_bias_fn(
    x: np.ndarray,  # [bsz, seqlen, dim]
    input_ids: np.ndarray,  # [max_bsz, max_seqlen] int32
    weight: np.ndarray,  # [n_experts, dim]
    tid2eid: np.ndarray,  # [vocab, topk] int32
    *,
    bsz: int,
    seqlen: int,
    dim: int,
    rows: int,
    score_func: str,
    route_scale: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse hash-MoE dispatch frontend for product execution.

    This is the first layer-phase boundary for hash-router layers:
    flatten/cast hidden for blockwise MoE, compute hash-layer gate scores,
    gather static token-id expert IDs, and compute routed weights.
    """
    flat_hidden = moe_hidden_flatten_fn(x, dim=int(dim))
    orig, _scores = gate_scores_no_bias_fn(
        flat_hidden,
        weight,
        score_func=str(score_func),
    )
    ids = input_ids.astype(np.int32)
    if ids.ndim == 2:
        active_ids = ids[: int(bsz), : int(seqlen)].reshape(-1)
    else:
        active_ids = ids[: int(bsz) * int(seqlen)]
    indices = hash_route_fn(active_ids, tid2eid)
    weights = router_tail_fn(
        orig,
        indices,
        route_scale=float(route_scale),
        normalize=bool(normalize),
    )
    hidden_for_moe = pad_flat_rows_fn(flat_hidden, rows=int(rows))
    weights_for_moe, indices_for_moe = pad_router_rows_fn(
        weights,
        indices,
        rows=int(rows),
    )
    shared_hidden = sequence_hidden_pad_fn(x, rows=int(rows), dim=int(dim))
    return hidden_for_moe, weights_for_moe, indices_for_moe, shared_hidden


def hash_route_fn(
    input_ids: np.ndarray,  # [N] int32
    tid2eid: np.ndarray,  # [V, topk] int32
) -> np.ndarray:
    """Hash-MoE routing: ``tid2eid[input_ids]`` gather.

    Produces ``indices [N, topk]`` int32. Used for V4 layers 0-2 where
    routing is a static per-token-id table rather than learned gate
    scores. Reproduces ``sampled_forward._run_gate`` hash branch.
    """
    return tid2eid[input_ids.astype(np.int32)].astype(np.int32)


def learned_moe_dispatch_no_bias_fn(
    x: np.ndarray,  # [bsz, seqlen, dim]
    weight: np.ndarray,  # [n_experts, dim]
    *,
    bsz: int,
    seqlen: int,
    dim: int,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse learned no-bias router dispatch into the blockwise-MoE ABI."""
    flat_hidden = moe_hidden_flatten_fn(x, dim=int(dim))
    weights, indices = learned_router_no_bias_fn(
        flat_hidden,
        weight,
        score_func=str(score_func),
        topk=int(topk),
        n_experts=int(n_experts),
        route_scale=float(route_scale),
        normalize=bool(normalize),
    )
    hidden_for_moe = pad_flat_rows_fn(flat_hidden, rows=int(rows))
    weights_for_moe, indices_for_moe = pad_router_rows_fn(
        weights,
        indices,
        rows=int(rows),
    )
    shared_hidden = sequence_hidden_pad_fn(x, rows=int(rows), dim=int(dim))
    return hidden_for_moe, weights_for_moe, indices_for_moe, shared_hidden


def learned_moe_dispatch_with_bias_fn(
    x: np.ndarray,  # [bsz, seqlen, dim]
    weight: np.ndarray,  # [n_experts, dim]
    bias: np.ndarray,  # [n_experts]
    *,
    bsz: int,
    seqlen: int,
    dim: int,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse learned bias router dispatch into the blockwise-MoE ABI."""
    flat_hidden = moe_hidden_flatten_fn(x, dim=int(dim))
    weights, indices = learned_router_with_bias_fn(
        flat_hidden,
        weight,
        bias,
        score_func=str(score_func),
        topk=int(topk),
        n_experts=int(n_experts),
        route_scale=float(route_scale),
        normalize=bool(normalize),
    )
    hidden_for_moe = pad_flat_rows_fn(flat_hidden, rows=int(rows))
    weights_for_moe, indices_for_moe = pad_router_rows_fn(
        weights,
        indices,
        rows=int(rows),
    )
    shared_hidden = sequence_hidden_pad_fn(x, rows=int(rows), dim=int(dim))
    return hidden_for_moe, weights_for_moe, indices_for_moe, shared_hidden


def learned_router_no_bias_fn(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse learned-router score projection, top-k, and weight tail."""
    orig, scores = gate_scores_no_bias_fn(x, weight, score_func=str(score_func))
    indices = topk_idx_fn(scores, k=int(topk), t=int(n_experts))
    weights = router_tail_fn(
        orig,
        indices,
        route_scale=float(route_scale),
        normalize=bool(normalize),
    )
    return weights, indices


def learned_router_with_bias_fn(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse learned-router score+bias projection, top-k, and weight tail."""
    orig, scores = gate_scores_with_bias_fn(
        x,
        weight,
        bias,
        score_func=str(score_func),
    )
    indices = topk_idx_fn(scores, k=int(topk), t=int(n_experts))
    weights = router_tail_fn(
        orig,
        indices,
        route_scale=float(route_scale),
        normalize=bool(normalize),
    )
    return weights, indices


def moe_hidden_flatten_fn(
    x: np.ndarray,  # [bsz, seqlen, dim] any float dtype
    *,
    dim: int,
) -> np.ndarray:
    """Flatten the mHC output to ``[bsz*seqlen, dim]`` bf16 for the MoE kernel.

    The blockwise MoE kernel takes ``hidden_states`` as 2D bf16. Running
    the flatten + cast as a device fragment lets ``_run_moe`` chain
    directly from the mHC output without a host round-trip.
    """
    import ml_dtypes as _ml

    return np.reshape(x, (-1, int(dim))).astype(_ml.bfloat16)


def router_tail_fn(
    orig_scores: np.ndarray,  # [..., n_experts]
    indices: np.ndarray,  # [..., topk] int32
    *,
    route_scale: float,
    normalize: bool,
) -> np.ndarray:
    """Router tail: gather selected expert scores, normalize, scale.

    ``normalize=False`` corresponds to ``score_func == "softmax"`` — the
    top-k of a softmaxed distribution isn't re-normalized.
    Reproduces ``sampled_forward._run_gate`` tail in one fragment.
    """
    idx = indices.astype(np.int32)
    w = np.take_along_axis(orig_scores.astype(np.float32), idx, axis=-1)
    if normalize:
        w = w / w.sum(axis=-1, keepdims=True)
    return (w * np.float32(route_scale)).astype(orig_scores.dtype)


def topk_idx_fn(
    scores: np.ndarray,  # [..., t]   fp32
    *,
    k: int,
    t: int,
    neg_inf: float = -1e9,
) -> np.ndarray:
    """HLO-traceable top-k.

    Returns the indices of the top-``k`` values along the last axis,
    shape ``[..., k]`` int32. Uses ``k`` sequential masked-argmax
    passes (no sort, no partition). Each pass:
        1. idx_i = argmax(scores, axis=-1)
        2. mask out the chosen slot via ``scores - neg_inf * one_hot``
        3. append idx_i to output.

    O(k * t) HLO ops — cheap at V4 shapes (k=512 index_topk, t up to
    ~a few thousand). Avoids the indirect scatter by computing the
    one-hot mask from ``arange(t) == idx_i[..., None]``, which is
    pointwise HLO.
    """
    K = int(k)
    T = int(t)
    arange_t = np.arange(T, dtype=np.int32)  # [t]

    # Collect per-iteration argmax results and concatenate at the end.
    # Using ``out[..., i] = idx_i`` would force a numpy-array coercion,
    # which NKIPy's tracer rejects ("cannot be converted to a numpy
    # array"). Concatenation of per-pass [..., 1] tensors stays in the
    # HLO graph.
    picks = []
    for _ in range(K):
        idx_i = np.argmax(scores, axis=-1).astype(np.int32)  # [...]
        picks.append(idx_i[..., None])  # [..., 1]
        one_hot = (arange_t == idx_i[..., None]).astype(np.float32)  # [..., t]
        scores = scores - np.float32(abs(neg_inf)) * one_hot
    return np.concatenate(picks, axis=-1)  # [..., K]


def compressor_decode_pool_from_state_fn(
    kv_score_state: np.ndarray,  # [owners * ring_size, 2 * state_width]
    owner_ids: np.ndarray,  # [bsz] int32
    end_positions: np.ndarray,  # [bsz] int32
    *,
    ratio: int,
    head_dim: int,
    state_width: int,
    ring_size: int,
    overlap: bool,
) -> np.ndarray:
    """Decode-pool one compressed row per request from ring compressor state."""
    bsz_i = int(owner_ids.shape[0])
    ratio_i = int(ratio)
    d_i = int(head_dim)
    width_i = int(state_width)
    ring_i = int(ring_size)
    owners = owner_ids.astype(np.int32).reshape(bsz_i, 1)
    end_pos = end_positions.astype(np.int32).reshape(bsz_i, 1)
    offs = np.arange(ratio_i, dtype=np.int32).reshape(1, ratio_i)

    def ring_rows(pos: np.ndarray) -> np.ndarray:
        # Neuron verifier rejects integer floor/divide here. Positions are
        # bounded by the token bucket, so fp32 divide + int truncation is exact.
        safe_pos = np.where(pos >= np.int32(0), pos, np.zeros_like(pos))
        quot = (safe_pos.astype(np.float32) / np.float32(ring_i)).astype(np.int32)
        rem = safe_pos - quot * np.int32(ring_i)
        return owners * np.int32(ring_i) + rem

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
        kv_parts = state[:, :, :d_i]
        score_parts = state[:, :, width_i : width_i + d_i]

    score_max = score_parts.max(axis=1, keepdims=True)
    weights = np.exp(score_parts - score_max)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (kv_parts * weights).sum(axis=1).astype(np.float32)


def compressor_decode_pool_post_qdq_from_state_freq_table_fn(
    kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
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
    """Fuse decode compressor state pool with post-pool RMS/RoPE/qDQ."""
    pooled = compressor_decode_pool_from_state_fn(
        kv_score_state,
        owner_ids,
        end_positions,
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


def decode_pool_fn(
    kv_state: np.ndarray,  # [bsz, ratio, d]
    score_state: np.ndarray,  # [bsz, ratio, d]
) -> np.ndarray:
    """Softmax-weighted pool over the ratio axis, decode-path no-overlap.

    Fuses ``_softmax(score_state, axis=1)`` + ``(kv_state * w).sum(axis=1,
    keepdims=True)`` in one traceable fragment. Fires once per ``ratio``
    decode steps on compressed layers when ``compressor.overlap`` is
    false and ``should_compress`` is true.
    """
    w = _softmax(score_state.astype(np.float32), axis=1)
    return (kv_state.astype(np.float32) * w).sum(axis=1, keepdims=True)


def decode_overlap_pool_fn(
    kv_state: np.ndarray,  # [bsz, 2*ratio, d]
    score_state: np.ndarray,  # [bsz, 2*ratio, d]
    *,
    ratio: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concat rolling-window halves, softmax pool, and return the new state
    shift all in one fragment. Decode overlap branch in ``_run_compressor``.

    Rebuilds the V4 overlap pooling shape:

        kv_st = concat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]])
        sc_st = concat([score_state[:, :ratio, :d], score_state[:, ratio:, d:]])
        w = softmax(sc_st, axis=1)
        kv_pool = (kv_st * w).sum(axis=1, keepdims=True)

    Returns ``(kv_pool, new_kv_state_head, new_score_state_head)`` where
    the "head" slices are the updated ``state.kv_state[:, :ratio]`` and
    ``state.score_state[:, :ratio]`` (shifted from the ``[ratio:]`` tail).
    """
    r = int(ratio)
    d = int(head_dim)
    kv_st = np.concatenate(
        [kv_state[:, :r, :d], kv_state[:, r:, d:]],
        axis=1,
    )
    sc_st = np.concatenate(
        [score_state[:, :r, :d], score_state[:, r:, d:]],
        axis=1,
    )
    w = _softmax(sc_st.astype(np.float32), axis=1)
    kv_pool = (kv_st.astype(np.float32) * w).sum(axis=1, keepdims=True)
    return kv_pool


def indexer_score_reshape_fn(
    score: np.ndarray,  # [bsz * seqlen, kv_len]
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
) -> np.ndarray:
    """Reshape indexer score rows for causal masking and top-k."""
    return np.reshape(score, (int(bsz), int(seqlen), int(kv_len)))


def indexer_topk_static_fn(
    score: np.ndarray,  # [bsz * seqlen, kv_len]
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    offset: int,
    prefill: bool,
) -> np.ndarray:
    """Fuse indexer score reshape, optional causal mask, top-k, and rebase."""
    scores = indexer_score_reshape_fn(
        score,
        bsz=int(bsz),
        seqlen=int(seqlen),
        kv_len=int(kv_len),
    )
    if bool(prefill):
        scores = causal_mask_add_fn(
            scores,
            seqlen=int(seqlen),
            ratio=int(ratio),
            kv_len=int(kv_len),
        )
    topk = topk_idx_fn(scores, k=int(k), t=int(kv_len))
    return topk_rebase_static_fn(
        topk,
        seqlen=int(seqlen),
        ratio=int(ratio),
        offset=int(offset),
        prefill=bool(prefill),
    )


def indexer_topk_static_dynamic_offset_fn(
    score: np.ndarray,  # [bsz * seqlen, kv_len]
    offset: np.ndarray,  # [1, 1] int32 runtime scalar
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    prefill: bool,
) -> np.ndarray:
    """Fuse indexer top-k and rebase with a runtime offset scalar."""
    scores = indexer_score_reshape_fn(
        score,
        bsz=int(bsz),
        seqlen=int(seqlen),
        kv_len=int(kv_len),
    )
    if bool(prefill):
        scores = causal_mask_add_fn(
            scores,
            seqlen=int(seqlen),
            ratio=int(ratio),
            kv_len=int(kv_len),
        )
    topk = topk_idx_fn(scores, k=int(k), t=int(kv_len))
    return topk_rebase_static_dynamic_offset_fn(
        topk,
        offset,
        seqlen=int(seqlen),
        ratio=int(ratio),
        prefill=bool(prefill),
    )


def topk_linearize_fn(
    topk: np.ndarray,  # [b, s, k] int32
    *,
    kv_len: int,
) -> np.ndarray:
    """Add per-batch offset to top-k so indices flatten over ``bsz * kv_len``.

    Reproduces ``np.where(topk >= 0, topk + b*kv_len, -1).reshape(b*s, -1)``.
    """
    t = topk.astype(np.int32)
    b, s, k = t.shape
    offsets = np.arange(int(b), dtype=np.int32).reshape(b, 1, 1) * np.int32(kv_len)
    adjusted = t + offsets
    out = np.where(t >= np.int32(0), adjusted, np.int32(-1))
    return out.reshape(b * s, k)


def causal_mask_add_fn(
    scores: np.ndarray,  # [b, s, kv_len] fp32
    *,
    seqlen: int,
    ratio: int,
    kv_len: int,
) -> np.ndarray:
    """Add -inf causal mask on prefill indexer scores.

    Reproduces ``sampled_forward.py:525-529``:
        row = arange(1, seqlen + 1) // ratio
        col = arange(kv_len)
        mask = col[None, :] >= row[:, None]
        scores = scores + where(mask, -inf, 0)
    """
    s = int(seqlen)
    r = int(ratio)
    k = int(kv_len)
    row = np.arange(1, s + 1, dtype=np.int32) // np.int32(r)
    col = np.arange(k, dtype=np.int32)
    mask = (col[None, :] >= row[:, None]).astype(np.float32)
    add = np.where(mask.astype(bool), np.float32(-np.inf), np.float32(0.0))
    return scores + add[None, :, :]


def indexer_sparse_attention_prep_static_fn(
    score: np.ndarray,  # [bsz * seqlen, kv_len]
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
    """Fuse indexer compressed top-k and SWA top-k into sparse-attn prep."""
    topk_win = window_topk_from_tokens_fn(
        x,
        window_size=int(window_size),
        start_pos=int(start_pos),
    )
    topk_comp = indexer_topk_static_fn(
        score,
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


def indexer_sparse_attention_prep_static_dynamic_offset_fn(
    score: np.ndarray,  # [bsz * seqlen, kv_len]
    x: np.ndarray,  # [bsz, seqlen, hidden]
    offset: np.ndarray,  # [1, 1] int32 runtime scalar
    *,
    bsz: int,
    seqlen: int,
    kv_len: int,
    k: int,
    ratio: int,
    prefill: bool,
    window_size: int,
    start_pos: int,
    rows: int,
    k_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse indexer compressed top-k and SWA top-k with runtime offset."""
    topk_win = window_topk_from_tokens_fn(
        x,
        window_size=int(window_size),
        start_pos=int(start_pos),
    )
    topk_comp = indexer_topk_static_dynamic_offset_fn(
        score,
        offset,
        bsz=int(bsz),
        seqlen=int(seqlen),
        kv_len=int(kv_len),
        k=int(k),
        ratio=int(ratio),
        prefill=bool(prefill),
    )
    return topk_concat_pad_sparse_attention_prep_fn(
        topk_win,
        topk_comp,
        rows=int(rows),
        k_tile=int(k_tile),
    )


def indexer_sparse_attention_prep_decode_from_positions_fn(
    score: np.ndarray,  # [bsz, kv_len]
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
    """Decode indexer sparse-prep with runtime-position SWA top-k."""
    del start_pos
    if bool(prefill) or int(seqlen) != 1:
        raise RuntimeError("dynamic-position indexer sparse prep is decode-only")
    topk_win = window_topk_decode_from_positions_fn(
        x,
        positions,
        window_size=int(window_size),
    )
    topk_comp = indexer_topk_static_fn(
        score,
        bsz=int(bsz),
        seqlen=int(seqlen),
        kv_len=int(kv_len),
        k=int(k),
        ratio=int(ratio),
        offset=int(offset),
        prefill=False,
    )
    return topk_concat_pad_sparse_attention_prep_fn(
        topk_win,
        topk_comp,
        rows=int(rows),
        k_tile=int(k_tile),
    )


def inverse_rope_tail_fn(
    o: np.ndarray,  # [b, s, h, d]  attention output
    cos: np.ndarray,  # [s, rd//2]
    sin: np.ndarray,  # [s, rd//2]
    *,
    rope_head_dim: int,
) -> np.ndarray:
    """Inverse RoPE on the trailing ``rope_head_dim`` slice.

    Replaces the host mutation
        o[..., -rd:] = apply_rotary_emb(o[..., -rd:], fc, inverse=True)
    with a traceable fragment. Returns a new tensor.
    """
    rd = int(rope_head_dim)
    head = o[..., :-rd].astype(np.float32)
    tail = _apply_interleaved_rope(
        o[..., -rd:].astype(np.float32),
        cos,
        sin,
        inverse=True,
    )
    return np.concatenate((head, tail), axis=-1).astype(o.dtype)


def attention_out_proj_fn(
    o: np.ndarray,
    wo_a: np.ndarray,
    wo_b: np.ndarray,
    *,
    n_groups: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    bsz, seqlen, n_heads, head_dim = o.shape
    groups = int(n_groups)
    heads_per_group = n_heads // groups
    group_dim = heads_per_group * head_dim
    rank = wo_a.shape[0] // groups
    o_g = o.reshape(bsz * seqlen, groups, group_dim).astype(np.float32)
    wo_a_g = wo_a.reshape(groups, rank, group_dim).astype(np.float32)
    parts = []
    for gi in range(groups):
        parts.append(o_g[:, gi, :] @ wo_a_g[gi].T)
    o_flat = np.concatenate(parts, axis=-1).reshape(bsz, seqlen, groups * rank)
    projected = np.reshape(
        _linear_out_in(o_flat, wo_b),
        (int(bsz), int(seqlen), int(wo_b.shape[0])),
    )
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


def indexer_q_reshape_fn(
    q: np.ndarray,  # [bsz, seqlen, h*d]
    *,
    bsz: int,
    seqlen: int,
    n_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Reshape indexer query projection for the q-transform fragment."""
    return np.reshape(
        q,
        (int(bsz), int(seqlen), int(n_heads), int(head_dim)),
    )


def final_lm_head_fn(
    x: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    *,
    norm_eps: float,
) -> np.ndarray:
    h = apply_rms_norm(x, final_norm, eps=norm_eps)
    return _linear_out_in(h.astype(np.float32), lm_head)


def swiglu_fn(
    gate: np.ndarray,
    up: np.ndarray,
    *,
    limit: float,
) -> np.ndarray:
    return swiglu_with_limit(gate, up, float(limit))


def shared_expert_fn(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    limit: float,
) -> np.ndarray:
    """Dense shared-expert FFN: gate/up linear → SwiGLU → down linear.

    Weights are stored ``[out, in]`` (V4 convention). Input ``x`` is
    ``[n_tokens, dim]``; output matches.
    """
    gate = _linear_out_in(x, w_gate)
    up = _linear_out_in(x, w_up)
    y = swiglu_with_limit(gate, up, float(limit))
    return _linear_out_in(y, w_down)


def prefix_two_token_flats_fn(
    kv: np.ndarray,  # [bsz*seqlen, width]
    score: np.ndarray,  # [bsz*seqlen, width]
    *,
    bsz: int,
    seqlen: int,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``[b, :cutoff]`` rows from paired flat token slabs on device."""
    width = int(kv.shape[-1])
    b = int(bsz)
    s = int(seqlen)
    c = int(cutoff)
    kv_prefix = kv.reshape(b, s, width)[:, :c, :].reshape(b * c, width)
    score_prefix = score.reshape(b, s, width)[:, :c, :].reshape(b * c, width)
    return kv_prefix, score_prefix


def moe_hidden_flatten_pad_fn(
    x: np.ndarray,  # [bsz, seqlen, dim] any float dtype
    *,
    dim: int,
    rows: int,
) -> np.ndarray:
    """Flatten MoE hidden states and pad singleton decode for collectives."""
    import ml_dtypes as _ml

    hidden = int(dim)
    target = int(rows)
    flat = np.reshape(x, (-1, hidden)).astype(_ml.bfloat16)
    n_rows = int(flat.shape[0])
    if n_rows > target:
        raise ValueError(f"input rows {n_rows} exceed target rows {target}")
    if n_rows == target:
        return flat
    pad = np.zeros((target - n_rows, hidden), dtype=flat.dtype)
    return np.concatenate((flat, pad), axis=0)


def moe_routed_unpad_fn(
    x: np.ndarray,
    *,
    n_tokens: int,
    dim: int,
) -> np.ndarray:
    """Drop MoE decode padding before shared-expert add."""
    return np.reshape(x[: int(n_tokens)], (int(n_tokens), int(dim)))


def shared_expert_add_fn(
    acc: np.ndarray,  # [n_tokens, dim]  MoE routed output
    x: np.ndarray,  # [bsz, seqlen, dim] layer input to shared expert
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    limit: float,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """``acc + shared_expert(x)`` fused on device, output shape matches ``x``.

    ``acc`` is the flat ``[n_tokens, dim]`` MoE routed output; ``x`` is
    the original ``[bsz, seqlen, dim]`` layer input. The shared-expert
    FFN runs on the 3D ``x`` (``_linear_out_in`` broadcasts over leading
    dims). If the shared intermediate is TP-sharded, the local shared output
    is all-reduced across the TP row before adding the routed output.
    Returns fp32 ``[bsz, seqlen, dim]``; the downstream ``mhc_post``
    fragment upcasts to fp32 anyway so no explicit dtype cast is needed.
    """
    shared = shared_expert_fn(
        x,
        w_gate,
        w_up,
        w_down,
        limit=float(limit),
    )
    if int(tp_degree) > 1:
        groups = (
            [list(group) for group in tp_replica_groups]
            if tp_replica_groups
            else [list(range(int(tp_degree)))]
        )
        shared = _all_reduce_last_dim_preserve_shape(
            shared,
            replica_groups=groups,
        )
    return np.reshape(acc, x.shape).astype(np.float32) + shared


def shared_expert_add_restore_fn(
    acc: np.ndarray,
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    limit: float,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    moe_replica_groups: tuple = (),
) -> np.ndarray:
    """Shared-expert add on active rows followed by sampled hidden restore."""
    n_tokens = int(bsz) * int(seqlen)
    hidden = int(hidden_size)
    routed = np.reshape(acc, (-1, hidden))
    if moe_replica_groups:
        routed = _all_reduce_last_dim_preserve_shape(
            routed,
            replica_groups=[list(group) for group in moe_replica_groups],
        )
    acc_active = routed[:n_tokens]
    x_active = np.reshape(x, (-1, hidden))[:n_tokens].reshape(
        int(bsz),
        int(seqlen),
        hidden,
    )
    return shared_expert_add_fn(
        acc_active,
        x_active,
        w_gate,
        w_up,
        w_down,
        limit=float(limit),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
    )


def shared_expert_add_restore_mhc_post_pre_fn(
    acc: np.ndarray,
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    limit: float,
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
    """Fuse padded shared-expert restore into the next mHC post/pre boundary."""
    restored = shared_expert_add_restore_fn(
        acc,
        x,
        w_gate,
        w_up,
        w_down,
        limit=float(limit),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
        moe_replica_groups=moe_replica_groups,
    )
    return mhc_post_pre_fn(
        restored,
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


def shared_expert_add_restore_mhc_post_head_flatten_pad_fn(
    acc: np.ndarray,
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_head_fn_weight: np.ndarray,
    hc_head_scale: np.ndarray,
    hc_head_base: np.ndarray,
    *,
    limit: float,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    moe_replica_groups: tuple = (),
    norm_eps: float,
    hc_eps: float,
    n_tokens: int,
    rows: int,
) -> np.ndarray:
    """Fuse padded shared-expert restore into final mHC/head flatten-pad."""
    restored = shared_expert_add_restore_fn(
        acc,
        x,
        w_gate,
        w_up,
        w_down,
        limit=float(limit),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
        moe_replica_groups=moe_replica_groups,
    )
    return mhc_post_hc_head_flatten_pad_fn(
        restored,
        residual,
        post,
        comb,
        hc_head_fn_weight,
        hc_head_scale,
        hc_head_base,
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
        n_tokens=int(n_tokens),
        hidden_size=int(hidden_size),
        rows=int(rows),
    )


def shared_expert_add_restore_mhc_post_head_select_pad_fn(
    acc: np.ndarray,
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_head_fn_weight: np.ndarray,
    hc_head_scale: np.ndarray,
    hc_head_base: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    limit: float,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    moe_replica_groups: tuple = (),
    norm_eps: float,
    hc_eps: float,
    n_tokens: int,
    rows: int,
) -> np.ndarray:
    """Final full-sampler boundary that keeps only per-request sampled rows."""
    restored = shared_expert_add_restore_fn(
        acc,
        x,
        w_gate,
        w_up,
        w_down,
        limit=float(limit),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
        moe_replica_groups=moe_replica_groups,
    )
    h = mhc_post_fn(restored, residual, post, comb)
    hidden = hc_head_fn(
        h,
        hc_head_fn_weight,
        hc_head_scale,
        hc_head_base,
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    flat = np.reshape(hidden, (int(n_tokens), int(hidden_size)))
    selected = flat[last_token_indices.astype(np.int32)[: int(bsz)]]
    return pad_flat_rows_fn(selected, rows=int(rows))


def shared_expert_add_restore_mhc_post_head_top1_fn(
    acc: np.ndarray,
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_head_fn_weight: np.ndarray,
    hc_head_scale: np.ndarray,
    hc_head_base: np.ndarray,
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    last_token_indices: np.ndarray,
    *,
    limit: float,
    bsz: int,
    seqlen: int,
    hidden_size: int,
    tp_degree: int = 1,
    tp_replica_groups: tuple = (),
    moe_replica_groups: tuple = (),
    norm_eps: float,
    hc_eps: float,
    n_tokens: int,
    rows: int,
    lm_norm_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Final product boundary with greedy local LM-head top1 fused in."""
    from nkipy.core import tensor_apis

    from nkipy_serving.ops.nn import apply_rms_norm

    hidden = shared_expert_add_restore_mhc_post_head_flatten_pad_fn(
        acc,
        x,
        w_gate,
        w_up,
        w_down,
        residual,
        post,
        comb,
        hc_head_fn_weight,
        hc_head_scale,
        hc_head_base,
        limit=float(limit),
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        tp_degree=int(tp_degree),
        tp_replica_groups=tp_replica_groups,
        moe_replica_groups=moe_replica_groups,
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
        n_tokens=int(n_tokens),
        rows=int(rows),
    )
    hidden_dtype = hidden.dtype
    selected = hidden[last_token_indices.astype(np.int32)]
    normed = apply_rms_norm(
        selected,
        final_norm.astype(np.float32),
        eps=float(lm_norm_eps),
    ).astype(hidden_dtype)
    logits = (normed @ lm_head.transpose(1, 0)).astype(np.float32)
    top1_vals, top1_idx = tensor_apis.topk(logits, k=1, axis=1)
    return top1_vals.reshape((-1,)), top1_idx.reshape((-1,)).astype(np.int32)


def dp_attention_all_reduce_unpad_reshape_mhc_post_pre_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse DP all-reduce with restore and the following mHC post/pre stage."""
    reduced = dp_attention_all_reduce_fn(x, replica_groups=replica_groups)
    return dp_attention_unpad_reshape_mhc_post_pre_fn(
        reduced,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )


def dp_attention_all_reduce_post_pre_hash_moe_dispatch_no_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    input_ids: np.ndarray,
    weight: np.ndarray,
    tid2eid: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    route_scale: float,
    normalize: bool,
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
    """Fuse DP attention restore/post-pre with hash-router dispatch."""
    h, y, next_post, next_comb = dp_attention_all_reduce_unpad_reshape_mhc_post_pre_fn(
        x,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        replica_groups=replica_groups,
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    hidden_for_moe, weights_for_moe, indices_for_moe, shared_hidden = (
        hash_moe_dispatch_no_bias_fn(
            y,
            input_ids,
            weight,
            tid2eid,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=int(hidden_size),
            rows=int(rows),
            score_func=str(score_func),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    return (
        h,
        y,
        next_post,
        next_comb,
        hidden_for_moe,
        weights_for_moe,
        indices_for_moe,
        shared_hidden,
    )


def dp_attention_all_reduce_post_pre_learned_moe_dispatch_no_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    weight: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
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
    """Fuse DP attention restore/post-pre with learned no-bias dispatch."""
    h, y, next_post, next_comb = dp_attention_all_reduce_unpad_reshape_mhc_post_pre_fn(
        x,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        replica_groups=replica_groups,
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    hidden_for_moe, weights_for_moe, indices_for_moe, shared_hidden = (
        learned_moe_dispatch_no_bias_fn(
            y,
            weight,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=int(hidden_size),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    return (
        h,
        y,
        next_post,
        next_comb,
        hidden_for_moe,
        weights_for_moe,
        indices_for_moe,
        shared_hidden,
    )


def dp_attention_all_reduce_post_pre_learned_moe_dispatch_with_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
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
    """Fuse DP attention restore/post-pre with learned bias dispatch."""
    h, y, next_post, next_comb = dp_attention_all_reduce_unpad_reshape_mhc_post_pre_fn(
        x,
        residual,
        post,
        comb,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        replica_groups=replica_groups,
        bsz=int(bsz),
        seqlen=int(seqlen),
        hidden_size=int(hidden_size),
        hc_mult=int(hc_mult),
        sinkhorn_iters=int(sinkhorn_iters),
        norm_eps=float(norm_eps),
        hc_eps=float(hc_eps),
    )
    hidden_for_moe, weights_for_moe, indices_for_moe, shared_hidden = (
        learned_moe_dispatch_with_bias_fn(
            y,
            weight,
            bias,
            bsz=int(bsz),
            seqlen=int(seqlen),
            dim=int(hidden_size),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    return (
        h,
        y,
        next_post,
        next_comb,
        hidden_for_moe,
        weights_for_moe,
        indices_for_moe,
        shared_hidden,
    )


def _blockwise_moe_prefill_router_output_fn(
    hidden_states: np.ndarray,
    moe_output: np.ndarray,
    router_weights_hbm: np.ndarray,
    router_indices_hbm: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    num_static_blocks: int,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> np.ndarray:
    """Run device-router prefill blockwise MoE inside a product graph."""
    from nkipy_serving.ops.moe.device_schedule import make_prefill_fused_entry

    if not bool(has_down_bias):
        from nkipy_serving.ops.moe.blockwise_index import (
            BLOCK_SIZE as MOE_BLOCK_SIZE,
        )
        from nkipy_serving.ops.moe.blockwise_nki_beta2 import (
            blockwise_nki_prefill_dsv4_beta2,
        )
        from nkipy_serving.ops.moe.device_schedule import wrap_nki_framework_kernel

        del moe_output, down_bias_broadcasted_hbm, num_static_blocks
        schedule_entry = make_prefill_fused_entry(
            token_bucket=int(token_bucket),
            local_num_experts=int(local_num_experts),
            experts_per_token=int(experts_per_token),
            num_blocks=int(num_blocks),
            f_len=int(f_len),
            output_len=int(output_len),
            logical_nc_config=int(logical_nc_config),
            compress_block_to_expert=False,
        )
        (
            expert_affinities_masked_hbm,
            token_position_to_id,
            block_to_expert,
        ) = schedule_entry(router_weights_hbm, router_indices_hbm, ep_start)
        return wrap_nki_framework_kernel(
            blockwise_nki_prefill_dsv4_beta2,
            lnc=int(logical_nc_config),
            args=(
                hidden_states,
                expert_affinities_masked_hbm,
                gate_up_proj_weight,
                gate_up_bias_plus1_T_hbm,
                down_proj_weight,
                token_position_to_id.reshape((int(num_blocks) * int(MOE_BLOCK_SIZE),)),
                block_to_expert,
            ),
            kwargs={
                "gate_clamp_upper": float(gate_clamp_upper),
                "gate_clamp_lower": gate_clamp_lower,
                "up_clamp_upper": float(up_clamp_upper),
                "up_clamp_lower": float(up_clamp_lower),
            },
        )

    import ml_dtypes
    from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType
    from nkipy.core.nki_op import wrap_nki_kernel

    from nkipy_serving.ops.moe.blockwise_nki import blockwise_nki_static

    schedule_entry = make_prefill_fused_entry(
        token_bucket=int(token_bucket),
        local_num_experts=int(local_num_experts),
        experts_per_token=int(experts_per_token),
        num_blocks=int(num_blocks),
        f_len=int(f_len),
        output_len=int(output_len),
        logical_nc_config=int(logical_nc_config),
    )
    (
        expert_affinities_masked_hbm,
        token_position_to_id,
        block_to_expert,
    ) = schedule_entry(router_weights_hbm, router_indices_hbm, ep_start)
    nki_op = wrap_nki_kernel(
        blockwise_nki_static,
        [
            hidden_states,
            moe_output,
            expert_affinities_masked_hbm,
            gate_up_proj_weight,
            gate_up_bias_plus1_T_hbm,
            down_proj_weight,
            down_bias_broadcasted_hbm,
            token_position_to_id,
            block_to_expert,
            int(num_static_blocks),
            ActFnType.SiLU,
            ml_dtypes.bfloat16,
            True,
            1,
            bool(has_down_bias),
            True,
            float(gate_clamp_upper),
            gate_clamp_lower,
            float(up_clamp_upper),
            float(up_clamp_lower),
        ],
    )
    return nki_op(
        hidden_states,
        moe_output,
        expert_affinities_masked_hbm,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
    )


def _blockwise_moe_decode_router_output_fn(
    hidden_states: np.ndarray,
    router_weights_hbm: np.ndarray,
    router_indices_hbm: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> np.ndarray:
    """Run device-router decode blockwise MoE inside a product graph."""
    import ml_dtypes
    from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType
    from nkipy.core import tensor_apis
    from nkipy.core.nki_op import wrap_nki_kernel

    from nkipy_serving.ops.moe.blockwise_nki import (
        TILE_SIZE as MOE_TILE_SIZE,
    )
    from nkipy_serving.ops.moe.blockwise_nki import (
        blockwise_nki_decode,
    )
    from nkipy_serving.ops.moe.device_schedule import (
        local_expert_affinities_dynamic_ep_fn,
    )

    T = int(hidden_states.shape[0])
    E = int(gate_up_proj_weight.shape[0])
    affinities_T = np.transpose(
        local_expert_affinities_dynamic_ep_fn(
            router_weights_hbm,
            router_indices_hbm,
            ep_start,
            local_num_experts=E,
        )
    )
    token_position_to_id = tensor_apis.full(
        (1, int(MOE_TILE_SIZE)),
        -1,
        dtype=np.int32,
    )
    token_position_to_id[0, :T] = np.arange(T, dtype=np.int32) + tensor_apis.zeros(
        (T,), dtype=np.int32
    )
    token_position_to_id = np.broadcast_to(
        token_position_to_id,
        (E, int(MOE_TILE_SIZE)),
    )
    block_to_expert = np.arange(E, dtype=np.int8) + tensor_apis.zeros(
        (E,),
        dtype=np.int8,
    )
    nki_op = wrap_nki_kernel(
        blockwise_nki_decode,
        [
            hidden_states,
            affinities_T,
            gate_up_proj_weight,
            gate_up_bias_plus1_T_hbm,
            down_proj_weight,
            down_bias_broadcasted_hbm,
            token_position_to_id,
            block_to_expert,
            ActFnType.SiLU,
            ml_dtypes.bfloat16,
            True,
            3,
            bool(has_down_bias),
            float(gate_clamp_upper),
            gate_clamp_lower,
            float(up_clamp_upper),
            float(up_clamp_lower),
        ],
    )
    return nki_op(
        hidden_states,
        affinities_T,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
    )


def dp_attention_all_reduce_post_pre_hash_moe_blockwise_prefill_no_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    input_ids: np.ndarray,
    weight: np.ndarray,
    tid2eid: np.ndarray,
    moe_output: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    route_scale: float,
    normalize: bool,
    num_static_blocks: int,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, y, next_post, next_comb, hidden, weights, indices, shared_hidden = (
        dp_attention_all_reduce_post_pre_hash_moe_dispatch_no_bias_fn(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            input_ids,
            weight,
            tid2eid,
            replica_groups=replica_groups,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rows=int(rows),
            score_func=str(score_func),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    routed = _blockwise_moe_prefill_router_output_fn(
        hidden,
        moe_output,
        weights,
        indices,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        num_static_blocks=int(num_static_blocks),
        token_bucket=int(token_bucket),
        local_num_experts=int(local_num_experts),
        experts_per_token=int(experts_per_token),
        num_blocks=int(num_blocks),
        f_len=int(f_len),
        output_len=int(output_len),
        logical_nc_config=int(logical_nc_config),
        has_down_bias=bool(has_down_bias),
        gate_clamp_upper=float(gate_clamp_upper),
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=float(up_clamp_upper),
        up_clamp_lower=float(up_clamp_lower),
    )
    return h, y, next_post, next_comb, routed, shared_hidden


def dp_attention_all_reduce_post_pre_hash_moe_blockwise_decode_no_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    input_ids: np.ndarray,
    weight: np.ndarray,
    tid2eid: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    route_scale: float,
    normalize: bool,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, y, next_post, next_comb, hidden, weights, indices, shared_hidden = (
        dp_attention_all_reduce_post_pre_hash_moe_dispatch_no_bias_fn(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            input_ids,
            weight,
            tid2eid,
            replica_groups=replica_groups,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rows=int(rows),
            score_func=str(score_func),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    routed = _blockwise_moe_decode_router_output_fn(
        hidden,
        weights,
        indices,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        has_down_bias=bool(has_down_bias),
        gate_clamp_upper=float(gate_clamp_upper),
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=float(up_clamp_upper),
        up_clamp_lower=float(up_clamp_lower),
    )
    return h, y, next_post, next_comb, routed, shared_hidden


def dp_attention_all_reduce_post_pre_learned_moe_blockwise_prefill_no_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    weight: np.ndarray,
    moe_output: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
    num_static_blocks: int,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, y, next_post, next_comb, hidden, weights, indices, shared_hidden = (
        dp_attention_all_reduce_post_pre_learned_moe_dispatch_no_bias_fn(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            weight,
            replica_groups=replica_groups,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    routed = _blockwise_moe_prefill_router_output_fn(
        hidden,
        moe_output,
        weights,
        indices,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        num_static_blocks=int(num_static_blocks),
        token_bucket=int(token_bucket),
        local_num_experts=int(local_num_experts),
        experts_per_token=int(experts_per_token),
        num_blocks=int(num_blocks),
        f_len=int(f_len),
        output_len=int(output_len),
        logical_nc_config=int(logical_nc_config),
        has_down_bias=bool(has_down_bias),
        gate_clamp_upper=float(gate_clamp_upper),
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=float(up_clamp_upper),
        up_clamp_lower=float(up_clamp_lower),
    )
    return h, y, next_post, next_comb, routed, shared_hidden


def dp_attention_all_reduce_post_pre_learned_moe_blockwise_decode_no_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    weight: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, y, next_post, next_comb, hidden, weights, indices, shared_hidden = (
        dp_attention_all_reduce_post_pre_learned_moe_dispatch_no_bias_fn(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            weight,
            replica_groups=replica_groups,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    routed = _blockwise_moe_decode_router_output_fn(
        hidden,
        weights,
        indices,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        has_down_bias=bool(has_down_bias),
        gate_clamp_upper=float(gate_clamp_upper),
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=float(up_clamp_upper),
        up_clamp_lower=float(up_clamp_lower),
    )
    return h, y, next_post, next_comb, routed, shared_hidden


def dp_attention_all_reduce_post_pre_learned_moe_blockwise_prefill_with_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    moe_output: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
    num_static_blocks: int,
    token_bucket: int,
    local_num_experts: int,
    experts_per_token: int,
    num_blocks: int,
    f_len: int,
    output_len: int,
    logical_nc_config: int,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, y, next_post, next_comb, hidden, weights, indices, shared_hidden = (
        dp_attention_all_reduce_post_pre_learned_moe_dispatch_with_bias_fn(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            weight,
            bias,
            replica_groups=replica_groups,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    routed = _blockwise_moe_prefill_router_output_fn(
        hidden,
        moe_output,
        weights,
        indices,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        num_static_blocks=int(num_static_blocks),
        token_bucket=int(token_bucket),
        local_num_experts=int(local_num_experts),
        experts_per_token=int(experts_per_token),
        num_blocks=int(num_blocks),
        f_len=int(f_len),
        output_len=int(output_len),
        logical_nc_config=int(logical_nc_config),
        has_down_bias=bool(has_down_bias),
        gate_clamp_upper=float(gate_clamp_upper),
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=float(up_clamp_upper),
        up_clamp_lower=float(up_clamp_lower),
    )
    return h, y, next_post, next_comb, routed, shared_hidden


def dp_attention_all_reduce_post_pre_learned_moe_blockwise_decode_with_bias_fn(
    x: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_weight: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    ep_start: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    *,
    replica_groups: tuple = (),
    bsz: int,
    seqlen: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    rows: int,
    score_func: str,
    topk: int,
    n_experts: int,
    route_scale: float,
    normalize: bool,
    has_down_bias: bool,
    gate_clamp_upper: float,
    gate_clamp_lower: float | None,
    up_clamp_upper: float,
    up_clamp_lower: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, y, next_post, next_comb, hidden, weights, indices, shared_hidden = (
        dp_attention_all_reduce_post_pre_learned_moe_dispatch_with_bias_fn(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            weight,
            bias,
            replica_groups=replica_groups,
            bsz=int(bsz),
            seqlen=int(seqlen),
            hidden_size=int(hidden_size),
            hc_mult=int(hc_mult),
            sinkhorn_iters=int(sinkhorn_iters),
            norm_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            rows=int(rows),
            score_func=str(score_func),
            topk=int(topk),
            n_experts=int(n_experts),
            route_scale=float(route_scale),
            normalize=bool(normalize),
        )
    )
    routed = _blockwise_moe_decode_router_output_fn(
        hidden,
        weights,
        indices,
        ep_start,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        has_down_bias=bool(has_down_bias),
        gate_clamp_upper=float(gate_clamp_upper),
        gate_clamp_lower=gate_clamp_lower,
        up_clamp_upper=float(up_clamp_upper),
        up_clamp_lower=float(up_clamp_lower),
    )
    return h, y, next_post, next_comb, routed, shared_hidden


__all__ = [
    "_blockwise_moe_decode_router_output_fn",
    "_blockwise_moe_prefill_router_output_fn",
    "attention_out_proj_fn",
    "causal_mask_add_fn",
    "compressor_decode_pool_from_state_fn",
    "compressor_decode_pool_post_qdq_from_state_freq_table_fn",
    "decode_overlap_pool_fn",
    "decode_pool_fn",
    "dp_attention_all_reduce_post_pre_hash_moe_blockwise_decode_no_bias_fn",
    "dp_attention_all_reduce_post_pre_hash_moe_blockwise_prefill_no_bias_fn",
    "dp_attention_all_reduce_post_pre_hash_moe_dispatch_no_bias_fn",
    "dp_attention_all_reduce_post_pre_learned_moe_blockwise_decode_no_bias_fn",
    "dp_attention_all_reduce_post_pre_learned_moe_blockwise_decode_with_bias_fn",
    "dp_attention_all_reduce_post_pre_learned_moe_blockwise_prefill_no_bias_fn",
    "dp_attention_all_reduce_post_pre_learned_moe_blockwise_prefill_with_bias_fn",
    "dp_attention_all_reduce_post_pre_learned_moe_dispatch_no_bias_fn",
    "dp_attention_all_reduce_post_pre_learned_moe_dispatch_with_bias_fn",
    "dp_attention_all_reduce_unpad_reshape_mhc_post_pre_fn",
    "final_lm_head_fn",
    "hash_moe_dispatch_no_bias_fn",
    "hash_route_fn",
    "indexer_q_reshape_fn",
    "indexer_score_reshape_fn",
    "indexer_sparse_attention_prep_decode_from_positions_fn",
    "indexer_sparse_attention_prep_static_dynamic_offset_fn",
    "indexer_sparse_attention_prep_static_fn",
    "indexer_topk_static_dynamic_offset_fn",
    "indexer_topk_static_fn",
    "inverse_rope_tail_fn",
    "learned_moe_dispatch_no_bias_fn",
    "learned_moe_dispatch_with_bias_fn",
    "learned_router_no_bias_fn",
    "learned_router_with_bias_fn",
    "moe_hidden_flatten_fn",
    "moe_hidden_flatten_pad_fn",
    "moe_routed_unpad_fn",
    "prefix_two_token_flats_fn",
    "router_tail_fn",
    "shared_expert_add_fn",
    "shared_expert_add_restore_fn",
    "shared_expert_add_restore_mhc_post_head_flatten_pad_fn",
    "shared_expert_add_restore_mhc_post_head_select_pad_fn",
    "shared_expert_add_restore_mhc_post_head_top1_fn",
    "shared_expert_add_restore_mhc_post_pre_fn",
    "shared_expert_fn",
    "swiglu_fn",
    "topk_idx_fn",
    "topk_linearize_fn",
]
