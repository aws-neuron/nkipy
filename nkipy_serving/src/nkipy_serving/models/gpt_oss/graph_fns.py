"""GPT-OSS device-traceable per-layer functions (NKI kernels).

These use nkipy imports and are passed to DeviceKernel.compile_and_load().
"""

from __future__ import annotations

import math

import numpy as np

from nkipy_serving.ops.moe.blockwise_nki import (
    blockwise_add_residual,
    blockwise_decode_all_reduce_add_residual,
    output_init,
)
from nkipy_serving.ops.nn import (
    apply_rms_norm as _apply_rms_norm,
)
from nkipy_serving.ops.nn import (
    apply_rope as _apply_rope,
)
from nkipy_serving.ops.vocab_parallel_embedding import (
    vocab_parallel_embedding_local_fn,
)

# ---------------------------------------------------------------------------
# RoPE helpers (device-traceable) -- GPT-OSS uses YaRN scaling
# ---------------------------------------------------------------------------


def _compute_yarn_concentration_and_inv_freq(
    *,
    head_dim: int,
    base: float,
    initial_context_length: int,
    scaling_factor: float,
    ntk_alpha: float,
    ntk_beta: float,
) -> tuple[np.float32, np.ndarray]:
    """YaRN inverse frequencies + concentration factor (float32)."""
    freq = (
        np.float32(base)
        ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    ).astype(np.float32)
    if scaling_factor > 1.0:
        concentration = np.float32(0.1) * np.log(
            np.float32(scaling_factor)
        ) + np.float32(1.0)
        d_half = np.float32(head_dim) / np.float32(2.0)
        low = (
            d_half
            * np.log(
                np.float32(initial_context_length)
                / (np.float32(ntk_beta) * np.float32(2 * math.pi))
            )
            / np.log(np.float32(base))
        )
        high = (
            d_half
            * np.log(
                np.float32(initial_context_length)
                / (np.float32(ntk_alpha) * np.float32(2 * math.pi))
            )
            / np.log(np.float32(base))
        )
        if not (np.float32(0.0) < low < high < (d_half - np.float32(1.0))):
            raise RuntimeError(
                f"Invalid YaRN ramp range: {low=} {high=} {head_dim=} {base=}"
            )

        interpolation = np.float32(1.0) / (np.float32(scaling_factor) * freq)
        extrapolation = np.float32(1.0) / freq

        ramp = (np.arange(int(d_half), dtype=np.float32) - low) / (high - low)
        mask = np.float32(1.0) - np.clip(ramp, np.float32(0.0), np.float32(1.0))
        inv_freq = interpolation * (np.float32(1.0) - mask) + extrapolation * mask
    else:
        concentration = np.float32(1.0)
        inv_freq = np.float32(1.0) / freq
    return concentration, inv_freq.astype(np.float32)


def _build_rope_cache_for_positions_yarn(
    positions: np.ndarray,
    *,
    head_dim: int,
    theta: float,
    initial_context_length: int,
    scaling_factor: float,
    ntk_alpha: float,
    ntk_beta: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    if head_dim % 2 != 0:
        raise RuntimeError(f"head_dim must be even for RoPE, got {head_dim}")
    concentration, inv_freq = _compute_yarn_concentration_and_inv_freq(
        head_dim=head_dim,
        base=float(theta),
        initial_context_length=int(initial_context_length),
        scaling_factor=float(scaling_factor),
        ntk_alpha=float(ntk_alpha),
        ntk_beta=float(ntk_beta),
    )
    t = positions.astype(np.float32).reshape((-1,))
    freqs = np.outer(t, inv_freq)
    cos = (np.cos(freqs) * concentration).astype(dtype)
    sin = (np.sin(freqs) * concentration).astype(dtype)
    return cos, sin


# ---------------------------------------------------------------------------
# Device-traceable per-layer functions (seq-parallel)
# ---------------------------------------------------------------------------


def embedding_fn(
    input_ids: np.ndarray,
    embeddings: np.ndarray,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    del tp_degree, tp_replica_groups
    return vocab_parallel_embedding_local_fn(
        input_ids,
        embeddings,
        vocab_start_index=int(vocab_start_index),
        vocab_end_index=int(vocab_end_index),
    )


def tp_all_reduce_hidden_fn(
    hidden: np.ndarray,
    *,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    import nkipy.distributed.collectives as cc

    if int(tp_degree) <= 1:
        return hidden
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    return cc.all_reduce(
        hidden,
        replica_groups=_tp_groups,
        reduce_op=np.add,
    )


def tp_reduce_scatter_hidden_fn(
    hidden: np.ndarray,
    *,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    import nkipy.distributed.collectives as cc

    if int(tp_degree) <= 1:
        return hidden
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    return cc.reduce_scatter(
        hidden,
        reduce_scatter_dim=0,
        replica_groups=_tp_groups,
    )


def pre_attn_fn(
    hidden_shard: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    b_q: np.ndarray,
    b_k: np.ndarray,
    b_v: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seq-parallel pre-attn: RMSNorm(shard) -> all_gather -> QKV + RoPE (full)."""
    import nkipy.distributed.collectives as cc

    hidden_dtype = hidden_shard.dtype
    normed_shard = _apply_rms_norm(hidden_shard, input_norm, eps=rms_norm_eps)
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    normed = cc.all_gather(normed_shard, all_gather_dim=0, replica_groups=_tp_groups)

    total_tokens = normed.shape[0]
    q = (
        (normed @ w_q + b_q)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_heads, head_dim)
    )
    k = (
        (normed @ w_k + b_k)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    v = (
        (normed @ w_v + b_v)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    q = _apply_rope(q, cos=cos, sin=sin)
    k = _apply_rope(k, cos=cos, sin=sin)
    return q, k, v


def post_attn_fn(
    residual_shard: np.ndarray,
    context: np.ndarray,
    w_o: np.ndarray,
    b_o_sharded: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Output proj + bias + reduce_scatter + residual add (seq-parallel)."""
    import nkipy.distributed.collectives as cc

    hidden_dtype = residual_shard.dtype
    total_tokens = context.shape[0]
    attn_out_full = (
        context.reshape(total_tokens, num_heads * head_dim).astype(hidden_dtype) @ w_o
        + b_o_sharded
    ).astype(hidden_dtype)
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    attn_out_shard = cc.reduce_scatter(
        attn_out_full,
        reduce_scatter_dim=0,
        replica_groups=_tp_groups,
    )
    return (residual_shard + attn_out_shard).astype(hidden_dtype)


def router_fn(
    hidden_shard: np.ndarray,
    post_attn_norm: np.ndarray,
    router_weight: np.ndarray,
    router_bias: np.ndarray,
    *,
    rms_norm_eps: float,
    top_k: int,
    tp_degree: int,
    ep_rank: int = 0,
    local_num_experts: int = 0,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seq-parallel router: RMSNorm(shard) -> router logits -> topk + masked softmax -> all_gather.

    With EP, affinities are sliced to local experts after all_gather.
    """
    import nkipy.distributed.collectives as cc

    hidden_dtype = hidden_shard.dtype
    normed_shard = _apply_rms_norm(hidden_shard, post_attn_norm, eps=rms_norm_eps)
    logits_shard = (normed_shard @ router_weight + router_bias).astype(hidden_dtype)
    topk_idx, affinities_shard = _router_topk_and_affinities(
        logits_shard,
        top_k=int(top_k),
    )

    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    topk_full = cc.all_gather(topk_idx, all_gather_dim=0, replica_groups=_tp_groups)
    affinities_full = cc.all_gather(
        affinities_shard, all_gather_dim=0, replica_groups=_tp_groups
    )
    normed_full = cc.all_gather(
        normed_shard, all_gather_dim=0, replica_groups=_tp_groups
    )

    # EP: slice affinities to local experts.
    if int(local_num_experts) > 0 and int(local_num_experts) < int(
        logits_shard.shape[1]
    ):
        expert_start = int(ep_rank) * int(local_num_experts)
        affinities_full = affinities_full[
            :, expert_start : expert_start + int(local_num_experts)
        ]

    return topk_full, affinities_full, normed_full


def _router_topk_and_affinities(
    logits_shard: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Traceable top-k routing and masked-softmax over local router logits."""
    from nkipy.core import tensor_apis

    hidden_dtype = logits_shard.dtype
    _topk_logits, topk_idx = tensor_apis.topk(logits_shard, k=int(top_k), axis=1)
    topk_idx = topk_idx.astype(np.int8)

    # Build dense mask for top-k experts.
    n_experts = int(logits_shard.shape[1])
    expert_mask = tensor_apis.zeros(logits_shard.shape, dtype=np.float32)
    expert_range = np.arange(n_experts, dtype=np.float32) + tensor_apis.zeros(
        n_experts, dtype=np.float32
    )
    for k in range(int(top_k)):
        expert_mask = expert_mask + np.equal(
            topk_idx[:, k : k + 1].astype(np.float32), expert_range
        )

    neg_inf = tensor_apis.full(
        logits_shard.shape, np.float32(-100000.0), dtype=np.float32
    )
    masked = (
        expert_mask * logits_shard.astype(np.float32)
        + (np.float32(1.0) - expert_mask) * neg_inf
    )
    masked = masked.astype(np.float32)
    exp_x = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
    affinities_shard = (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(
        hidden_dtype
    )
    return topk_idx, affinities_shard


def prefill_layer_pre_moe_nki_fn(
    hidden_shard: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    slot_mapping: np.ndarray,
    p_tqi: np.ndarray,
    p_tbt: np.ndarray,
    p_tm: np.ndarray,
    p_ndls: np.ndarray,
    p_qup: np.ndarray,
    p_lti: np.ndarray,
    d_tqi: np.ndarray,
    d_tbt: np.ndarray,
    d_tm: np.ndarray,
    d_ndls: np.ndarray,
    d_qup: np.ndarray,
    d_lti: np.ndarray,
    kv_cache: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    b_q: np.ndarray,
    b_k: np.ndarray,
    b_v: np.ndarray,
    sink: np.ndarray,
    w_o: np.ndarray,
    b_o: np.ndarray,
    post_attn_norm: np.ndarray,
    router_w: np.ndarray,
    router_b: np.ndarray,
    *,
    token_bucket: int,
    attn_bucket: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    softmax_scale: float,
    top_k: int,
    tp_degree: int,
    ep_rank: int = 0,
    local_num_experts: int = 0,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Traceable prefill layer graph up to the CPU MoE scheduling boundary."""
    from nkipy.core import tensor_apis

    from nkipy_serving.attention.nki_blocksparse_flash_attention import (
        nki_attention_unified_with_sink,
        nki_update_kv_cache_core,
    )

    q, k, v = pre_attn_fn(
        hidden_shard,
        input_norm=input_norm,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        b_q=b_q,
        b_k=b_k,
        b_v=b_v,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    kv_cache = nki_update_kv_cache_core(k, v, kv_cache, slot_mapping)

    if attn_bucket > token_bucket:
        pad_t = attn_bucket - token_bucket
        q_pad = tensor_apis.zeros((pad_t, num_heads, head_dim), dtype=q.dtype)
        k_pad = tensor_apis.zeros((pad_t, num_kv_heads, head_dim), dtype=k.dtype)
        v_pad = tensor_apis.zeros((pad_t, num_kv_heads, head_dim), dtype=v.dtype)
        q_attn = np.concatenate((q, q_pad), axis=0)
        k_attn = np.concatenate((k, k_pad), axis=0)
        v_attn = np.concatenate((v, v_pad), axis=0)
    else:
        q_attn, k_attn, v_attn = q, k, v

    context_attn = nki_attention_unified_with_sink(
        q_attn,
        k_attn,
        v_attn,
        kv_cache,
        sink,
        p_tqi,
        p_tbt,
        p_tm,
        p_ndls,
        p_qup,
        p_lti,
        d_tqi,
        d_tbt,
        d_tm,
        d_ndls,
        d_qup,
        d_lti,
        softmax_scale=softmax_scale,
    )
    context = (
        context_attn[:token_bucket] if attn_bucket > token_bucket else context_attn
    )
    hidden_attn = post_attn_fn(
        hidden_shard,
        context=context,
        w_o=w_o,
        b_o_sharded=b_o,
        num_heads=num_heads,
        head_dim=head_dim,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    topk, affinities, normed = router_fn(
        hidden_attn,
        post_attn_norm=post_attn_norm,
        router_weight=router_w,
        router_bias=router_b,
        rms_norm_eps=rms_norm_eps,
        top_k=top_k,
        tp_degree=tp_degree,
        ep_rank=ep_rank,
        local_num_experts=local_num_experts,
        tp_replica_groups=tp_replica_groups,
    )
    return kv_cache, hidden_attn, topk, affinities, normed


def prefill_layer_post_moe_fn(
    hidden_states: np.ndarray,
    residual_2d_shard: np.ndarray,
    output: np.ndarray,
    expert_affinities_masked_hbm: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    token_position_to_id: np.ndarray,
    block_to_expert: np.ndarray,
    *,
    num_static_blocks: int,
    tp_degree: int,
    ep_degree: int = 1,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Traceable prefill MoE epilogue after CPU block scheduling."""
    output = output_init(output)
    return blockwise_add_residual(
        hidden_states=hidden_states,
        residual_2d_shard=residual_2d_shard,
        output=output,
        expert_affinities_masked_hbm=expert_affinities_masked_hbm,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm=gate_up_bias_plus1_T_hbm,
        down_proj_weight=down_proj_weight,
        down_bias_broadcasted_hbm=down_bias_broadcasted_hbm,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        num_static_blocks=int(num_static_blocks),
        tp_degree=int(tp_degree),
        ep_degree=int(ep_degree),
        ep_replica_groups=ep_replica_groups,
        tp_replica_groups=tp_replica_groups,
    )


def pre_attn_decode_no_sp_fn(
    hidden: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    b_q: np.ndarray,
    b_k: np.ndarray,
    b_v: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, input_norm, eps=rms_norm_eps)
    total_tokens = normed.shape[0]
    q = (
        (normed @ w_q + b_q)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_heads, head_dim)
    )
    k = (
        (normed @ w_k + b_k)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    v = (
        (normed @ w_v + b_v)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    q = _apply_rope(q, cos=cos, sin=sin)
    k = _apply_rope(k, cos=cos, sin=sin)
    return q, k, v


def nki_attn_with_sink_fn(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    kv_cache: np.ndarray,
    sink: np.ndarray,
    slot_mapping: np.ndarray,
    p_tqi: np.ndarray,
    p_tbt: np.ndarray,
    p_tm: np.ndarray,
    p_ndls: np.ndarray,
    p_qup: np.ndarray,
    p_lti: np.ndarray,
    d_tqi: np.ndarray,
    d_tbt: np.ndarray,
    d_tm: np.ndarray,
    d_ndls: np.ndarray,
    d_qup: np.ndarray,
    d_lti: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """KV cache update + NKI paged attention with attention sinks (GPT-OSS)."""
    from nkipy_serving.attention.nki_blocksparse_flash_attention import (
        NKI_MIN_Q_SEQLEN,
        nki_attention_unified_with_sink,
        nki_update_kv_cache_core,
    )

    token_bucket = int(q.shape[0])
    attn_bucket = max(token_bucket, int(NKI_MIN_Q_SEQLEN))
    softmax_scale = 1.0 / (float(head_dim) ** 0.5)

    kv_cache = nki_update_kv_cache_core(k, v, kv_cache, slot_mapping)

    if attn_bucket > token_bucket:
        pad_t = attn_bucket - token_bucket
        from nkipy.core.tensor_apis import zeros

        q_pad = zeros((pad_t, num_heads, head_dim), dtype=q.dtype)
        k_pad = zeros((pad_t, num_kv_heads, head_dim), dtype=k.dtype)
        v_pad = zeros((pad_t, num_kv_heads, head_dim), dtype=v.dtype)
        q_attn = np.concatenate((q, q_pad), axis=0)
        k_attn = np.concatenate((k, k_pad), axis=0)
        v_attn = np.concatenate((v, v_pad), axis=0)
    else:
        q_attn, k_attn, v_attn = q, k, v

    context_attn = nki_attention_unified_with_sink(
        q_attn,
        k_attn,
        v_attn,
        kv_cache,
        sink,
        p_tqi,
        p_tbt,
        p_tm,
        p_ndls,
        p_qup,
        p_lti,
        d_tqi,
        d_tbt,
        d_tm,
        d_ndls,
        d_qup,
        d_lti,
        softmax_scale=softmax_scale,
    )
    return context_attn[:token_bucket] if attn_bucket > token_bucket else context_attn


def cpu_attn_with_sink_fn(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    kv_cache: np.ndarray,
    sink: np.ndarray,
    attn_metadata,
) -> np.ndarray:
    """CPU reference for GPT-OSS attention with learned sink logit per head.

    Sink is shape [num_heads, 1] (per local Q head). Each head's softmax
    denominator is augmented by exp(sink - scores_max); the sink contributes
    zero to the output (value=0), matching the kernel.
    """
    from nkipy_serving.attention.base import FORWARD_MODE_EXTEND
    from nkipy_serving.attention.vanilla import vanilla_update_kv_cache

    vanilla_update_kv_cache(k, v, kv_cache, attn_metadata.slot_mapping)

    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = attn_metadata.num_kv_heads
    block_size = attn_metadata.block_size
    batch_size = attn_metadata.batch_size
    seq_lens = np.asarray(attn_metadata.seq_lens, dtype=np.int64).reshape(-1)
    block_tables = np.asarray(attn_metadata.block_tables, dtype=np.int64)
    query_start_loc = np.asarray(attn_metadata.query_start_loc, dtype=np.int64).reshape(
        -1
    )
    sink_1d = np.asarray(sink, dtype=np.float32).reshape(-1)
    if sink_1d.shape[0] != num_heads:
        raise ValueError(
            f"sink shape {sink.shape} incompatible with num_heads={num_heads}"
        )

    heads_per_kv = num_heads // num_kv_heads
    scale = np.float32(1.0 / np.sqrt(np.float32(head_dim)))
    output = np.zeros((total_tokens, num_heads, head_dim), dtype=np.float32)

    for seq_idx in range(batch_size):
        seq_len = int(seq_lens[seq_idx])
        q_start = int(query_start_loc[seq_idx])
        q_end = int(query_start_loc[seq_idx + 1])
        q_len = q_end - q_start

        num_blocks_needed = (seq_len + block_size - 1) // block_size
        k_gathered = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        v_gathered = np.zeros((seq_len, num_kv_heads, head_dim), dtype=np.float32)
        for blk_idx in range(num_blocks_needed):
            block_id = int(block_tables[seq_idx, blk_idx])
            start = blk_idx * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            k_gathered[start:end] = (
                kv_cache[0, block_id, :, :length, :]
                .transpose(1, 0, 2)
                .astype(np.float32)
            )
            v_gathered[start:end] = (
                kv_cache[1, block_id, :, :length, :]
                .transpose(1, 0, 2)
                .astype(np.float32)
            )

        q_seq = q[q_start:q_end].astype(np.float32)
        for h in range(num_heads):
            kv_h = h // heads_per_kv
            scores = (q_seq[:, h, :] @ k_gathered[:, kv_h, :].T) * scale
            if attn_metadata.forward_mode == FORWARD_MODE_EXTEND:
                for qi in range(q_len):
                    max_kv_pos = seq_len - q_len + qi
                    if max_kv_pos + 1 < seq_len:
                        scores[qi, max_kv_pos + 1 :] = -np.inf
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            sink_exp = np.exp(np.float32(sink_1d[h]) - scores_max[:, 0])
            denom = np.sum(scores_exp, axis=-1) + sink_exp
            attn_weights = scores_exp / denom[:, None]
            output[q_start:q_end, h, :] = attn_weights @ v_gathered[:, kv_h, :]

    return output.astype(q.dtype)


def post_attn_decode_no_sp_fn(
    hidden: np.ndarray,
    context: np.ndarray,
    w_o: np.ndarray,
    b_o_sharded: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    import nkipy.distributed.collectives as cc

    hidden_dtype = hidden.dtype
    total_tokens = hidden.shape[0]
    attn_out = (
        context.reshape(total_tokens, num_heads * head_dim).astype(hidden_dtype) @ w_o
        + b_o_sharded
    ).astype(hidden_dtype)
    if int(tp_degree) > 1:
        _tp_groups = (
            list(tp_replica_groups)
            if tp_replica_groups
            else [list(range(int(tp_degree)))]
        )
        attn_out = cc.all_reduce(attn_out, replica_groups=_tp_groups, reduce_op=np.add)
    return (hidden + attn_out).astype(hidden_dtype)


def router_moe_decode_no_sp_fn(
    hidden: np.ndarray,
    post_attn_norm: np.ndarray,
    router_weight: np.ndarray,
    router_bias: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    residual_2d: np.ndarray,
    *,
    rms_norm_eps: float,
    top_k: int,
    tp_degree: int,
    ep_degree: int = 1,
    ep_rank: int = 0,
    local_num_experts: int = 0,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, post_attn_norm, eps=rms_norm_eps)
    logits = (normed @ router_weight + router_bias).astype(hidden_dtype)
    _topk_idx, affinities = _router_topk_and_affinities(
        logits,
        top_k=int(top_k),
    )
    if int(local_num_experts) > 0 and int(local_num_experts) < int(logits.shape[1]):
        expert_start = int(ep_rank) * int(local_num_experts)
        affinities = affinities[:, expert_start : expert_start + int(local_num_experts)]
    return blockwise_decode_all_reduce_add_residual(
        hidden_states=normed,
        residual_2d=residual_2d,
        expert_affinities_masked_hbm=affinities,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm=gate_up_bias_plus1_T_hbm,
        down_proj_weight=down_proj_weight,
        down_bias_broadcasted_hbm=down_bias_broadcasted_hbm,
        tp_degree=int(tp_degree),
        num_experts=int(local_num_experts)
        if int(local_num_experts) > 0
        else int(logits.shape[1]),
        ep_degree=int(ep_degree),
        ep_replica_groups=ep_replica_groups,
        tp_replica_groups=tp_replica_groups,
    )


def router_prefill_no_sp_fn(
    hidden: np.ndarray,
    post_attn_norm: np.ndarray,
    router_weight: np.ndarray,
    router_bias: np.ndarray,
    *,
    rms_norm_eps: float,
    top_k: int,
    ep_rank: int = 0,
    local_num_experts: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-SP prefill router with bias. Returns (topk_idx, affinities_local, normed)."""
    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, post_attn_norm, eps=rms_norm_eps)
    logits = (normed @ router_weight + router_bias).astype(hidden_dtype)
    topk_idx, affinities = _router_topk_and_affinities(logits, top_k=int(top_k))
    if int(local_num_experts) > 0 and int(local_num_experts) < int(logits.shape[1]):
        expert_start = int(ep_rank) * int(local_num_experts)
        affinities = affinities[:, expert_start : expert_start + int(local_num_experts)]
    return topk_idx, affinities, normed


def moe_dispatch_prefill_no_sp_fn(
    hidden_states: np.ndarray,
    residual_2d: np.ndarray,
    expert_affinities_masked_hbm: np.ndarray,
    token_position_to_id: np.ndarray,
    block_to_expert: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T_hbm: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted_hbm: np.ndarray,
    moe_out_inout: np.ndarray,
    *,
    num_static_blocks: int,
    tp_degree: int,
    ep_degree: int = 1,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """No-SP prefill MoE dispatch: blockwise MoE + TP all_reduce + residual add."""
    from nkipy_serving.ops.moe.blockwise_nki import (
        blockwise_prefill_all_reduce_add_residual,
    )

    return blockwise_prefill_all_reduce_add_residual(
        hidden_states=hidden_states,
        residual_2d=residual_2d,
        output=moe_out_inout,
        expert_affinities_masked_hbm=expert_affinities_masked_hbm,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm=gate_up_bias_plus1_T_hbm,
        down_proj_weight=down_proj_weight,
        down_bias_broadcasted_hbm=down_bias_broadcasted_hbm,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        num_static_blocks=int(num_static_blocks),
        tp_degree=int(tp_degree),
        ep_degree=int(ep_degree),
        ep_replica_groups=ep_replica_groups,
        tp_replica_groups=tp_replica_groups,
    )


def decode_group4_nki_no_sp_fn(
    hidden: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    slot_mapping: np.ndarray,
    p_tqi: np.ndarray,
    p_tbt: np.ndarray,
    p_tm: np.ndarray,
    p_ndls: np.ndarray,
    p_qup: np.ndarray,
    p_lti: np.ndarray,
    d_tqi: np.ndarray,
    d_tbt: np.ndarray,
    d_tm: np.ndarray,
    d_ndls: np.ndarray,
    d_qup: np.ndarray,
    d_lti: np.ndarray,
    kv_cache_0: np.ndarray,
    input_norm_0: np.ndarray,
    w_q_0: np.ndarray,
    w_k_0: np.ndarray,
    w_v_0: np.ndarray,
    b_q_0: np.ndarray,
    b_k_0: np.ndarray,
    b_v_0: np.ndarray,
    sink_0: np.ndarray,
    w_o_0: np.ndarray,
    b_o_0: np.ndarray,
    post_attn_norm_0: np.ndarray,
    router_w_0: np.ndarray,
    router_b_0: np.ndarray,
    gup_w_0: np.ndarray,
    gup_bias_0: np.ndarray,
    down_w_0: np.ndarray,
    down_bias_bc_0: np.ndarray,
    kv_cache_1: np.ndarray,
    input_norm_1: np.ndarray,
    w_q_1: np.ndarray,
    w_k_1: np.ndarray,
    w_v_1: np.ndarray,
    b_q_1: np.ndarray,
    b_k_1: np.ndarray,
    b_v_1: np.ndarray,
    sink_1: np.ndarray,
    w_o_1: np.ndarray,
    b_o_1: np.ndarray,
    post_attn_norm_1: np.ndarray,
    router_w_1: np.ndarray,
    router_b_1: np.ndarray,
    gup_w_1: np.ndarray,
    gup_bias_1: np.ndarray,
    down_w_1: np.ndarray,
    down_bias_bc_1: np.ndarray,
    kv_cache_2: np.ndarray,
    input_norm_2: np.ndarray,
    w_q_2: np.ndarray,
    w_k_2: np.ndarray,
    w_v_2: np.ndarray,
    b_q_2: np.ndarray,
    b_k_2: np.ndarray,
    b_v_2: np.ndarray,
    sink_2: np.ndarray,
    w_o_2: np.ndarray,
    b_o_2: np.ndarray,
    post_attn_norm_2: np.ndarray,
    router_w_2: np.ndarray,
    router_b_2: np.ndarray,
    gup_w_2: np.ndarray,
    gup_bias_2: np.ndarray,
    down_w_2: np.ndarray,
    down_bias_bc_2: np.ndarray,
    kv_cache_3: np.ndarray,
    input_norm_3: np.ndarray,
    w_q_3: np.ndarray,
    w_k_3: np.ndarray,
    w_v_3: np.ndarray,
    b_q_3: np.ndarray,
    b_k_3: np.ndarray,
    b_v_3: np.ndarray,
    sink_3: np.ndarray,
    w_o_3: np.ndarray,
    b_o_3: np.ndarray,
    post_attn_norm_3: np.ndarray,
    router_w_3: np.ndarray,
    router_b_3: np.ndarray,
    gup_w_3: np.ndarray,
    gup_bias_3: np.ndarray,
    down_w_3: np.ndarray,
    down_bias_bc_3: np.ndarray,
    *,
    token_bucket: int,
    attn_bucket: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    softmax_scale: float,
    top_k: int,
    tp_degree: int,
    ep_degree: int = 1,
    ep_rank: int = 0,
    local_num_experts: int = 0,
    ep_replica_groups: tuple = (),
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    from nkipy.core import tensor_apis

    from nkipy_serving.attention.nki_blocksparse_flash_attention import (
        nki_attention_unified_with_sink,
        nki_update_kv_cache_core,
    )

    layers = [
        (
            kv_cache_0,
            input_norm_0,
            w_q_0,
            w_k_0,
            w_v_0,
            b_q_0,
            b_k_0,
            b_v_0,
            sink_0,
            w_o_0,
            b_o_0,
            post_attn_norm_0,
            router_w_0,
            router_b_0,
            gup_w_0,
            gup_bias_0,
            down_w_0,
            down_bias_bc_0,
        ),
        (
            kv_cache_1,
            input_norm_1,
            w_q_1,
            w_k_1,
            w_v_1,
            b_q_1,
            b_k_1,
            b_v_1,
            sink_1,
            w_o_1,
            b_o_1,
            post_attn_norm_1,
            router_w_1,
            router_b_1,
            gup_w_1,
            gup_bias_1,
            down_w_1,
            down_bias_bc_1,
        ),
        (
            kv_cache_2,
            input_norm_2,
            w_q_2,
            w_k_2,
            w_v_2,
            b_q_2,
            b_k_2,
            b_v_2,
            sink_2,
            w_o_2,
            b_o_2,
            post_attn_norm_2,
            router_w_2,
            router_b_2,
            gup_w_2,
            gup_bias_2,
            down_w_2,
            down_bias_bc_2,
        ),
        (
            kv_cache_3,
            input_norm_3,
            w_q_3,
            w_k_3,
            w_v_3,
            b_q_3,
            b_k_3,
            b_v_3,
            sink_3,
            w_o_3,
            b_o_3,
            post_attn_norm_3,
            router_w_3,
            router_b_3,
            gup_w_3,
            gup_bias_3,
            down_w_3,
            down_bias_bc_3,
        ),
    ]

    for (
        kv_cache,
        input_norm,
        w_q,
        w_k,
        w_v,
        b_q,
        b_k,
        b_v,
        sink,
        w_o,
        b_o,
        post_attn_norm,
        router_w,
        router_b,
        gup_w,
        gup_bias,
        down_w,
        down_bias_bc,
    ) in layers:
        q, k, v = pre_attn_decode_no_sp_fn(
            hidden,
            input_norm=input_norm,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            b_q=b_q,
            b_k=b_k,
            b_v=b_v,
            cos=cos,
            sin=sin,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )
        kv_cache = nki_update_kv_cache_core(k, v, kv_cache, slot_mapping)

        if attn_bucket > token_bucket:
            pad_t = attn_bucket - token_bucket
            q_pad = tensor_apis.zeros((pad_t, num_heads, head_dim), dtype=q.dtype)
            k_pad = tensor_apis.zeros((pad_t, num_kv_heads, head_dim), dtype=k.dtype)
            v_pad = tensor_apis.zeros((pad_t, num_kv_heads, head_dim), dtype=v.dtype)
            q_attn = np.concatenate((q, q_pad), axis=0)
            k_attn = np.concatenate((k, k_pad), axis=0)
            v_attn = np.concatenate((v, v_pad), axis=0)
        else:
            q_attn, k_attn, v_attn = q, k, v

        context_attn = nki_attention_unified_with_sink(
            q_attn,
            k_attn,
            v_attn,
            kv_cache,
            sink,
            p_tqi,
            p_tbt,
            p_tm,
            p_ndls,
            p_qup,
            p_lti,
            d_tqi,
            d_tbt,
            d_tm,
            d_ndls,
            d_qup,
            d_lti,
            softmax_scale=softmax_scale,
        )
        context = (
            context_attn[:token_bucket] if attn_bucket > token_bucket else context_attn
        )

        hidden = post_attn_decode_no_sp_fn(
            hidden,
            context=context,
            w_o=w_o,
            b_o_sharded=b_o,
            num_heads=num_heads,
            head_dim=head_dim,
            tp_degree=tp_degree,
            tp_replica_groups=tp_replica_groups,
        )
        hidden = router_moe_decode_no_sp_fn(
            hidden,
            post_attn_norm=post_attn_norm,
            router_weight=router_w,
            router_bias=router_b,
            gate_up_proj_weight=gup_w,
            gate_up_bias_plus1_T_hbm=gup_bias,
            down_proj_weight=down_w,
            down_bias_broadcasted_hbm=down_bias_bc,
            residual_2d=hidden,
            rms_norm_eps=rms_norm_eps,
            top_k=top_k,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            ep_rank=ep_rank,
            local_num_experts=local_num_experts,
            ep_replica_groups=ep_replica_groups,
            tp_replica_groups=tp_replica_groups,
        )
    return hidden
