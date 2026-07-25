"""Device-traceable per-layer functions for Qwen3 MoE (seq-parallel).

These use nkipy imports for collective operations and tensor APIs.
Each can be passed to DeviceKernel.compile_and_load().
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.ops.nn import (
    apply_head_rms_norm as _apply_head_rms_norm,
)
from nkipy_serving.ops.nn import (
    apply_rms_norm as _apply_rms_norm,
)
from nkipy_serving.ops.nn import (
    apply_rope as _apply_rope,
)


def embedding_fn(input_ids: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return embeddings[input_ids]


def pre_attn_fn(
    hidden_shard: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
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
    """Seq-parallel pre-attn: RMSNorm(shard) -> all_gather -> QKV + head norms + RoPE."""
    import nkipy.distributed.collectives as cc

    hidden_dtype = hidden_shard.dtype
    normed_shard = _apply_rms_norm(hidden_shard, input_norm, eps=rms_norm_eps)
    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    normed = cc.all_gather(normed_shard, all_gather_dim=0, replica_groups=_tp_groups)

    total_tokens = normed.shape[0]
    q = (normed @ w_q).astype(hidden_dtype).reshape(total_tokens, num_heads, head_dim)
    k = (
        (normed @ w_k)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    v = (
        (normed @ w_v)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    q = _apply_head_rms_norm(q, q_norm, eps=rms_norm_eps)
    k = _apply_head_rms_norm(k, k_norm, eps=rms_norm_eps)
    q = _apply_rope(q, cos=cos, sin=sin)
    k = _apply_rope(k, cos=cos, sin=sin)
    return q, k, v


def post_attn_fn(
    residual_shard: np.ndarray,
    context: np.ndarray,
    w_o: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """Output proj + reduce_scatter + residual add (seq-parallel). No bias."""
    import nkipy.distributed.collectives as cc

    hidden_dtype = residual_shard.dtype
    total_tokens = context.shape[0]
    attn_out_full = (
        context.reshape(total_tokens, num_heads * head_dim).astype(hidden_dtype) @ w_o
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
    *,
    rms_norm_eps: float,
    top_k: int,
    tp_degree: int,
    ep_rank: int = 0,
    local_num_experts: int = 0,
    tp_replica_groups: tuple = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seq-parallel router: RMSNorm(shard) -> router logits -> topk + masked softmax -> all_gather.

    No router bias (Qwen3 MoE).
    With EP, affinities are sliced to local experts after all_gather.
    """
    import nkipy.distributed.collectives as cc
    from nkipy.core import tensor_apis

    hidden_dtype = hidden_shard.dtype
    normed_shard = _apply_rms_norm(hidden_shard, post_attn_norm, eps=rms_norm_eps)
    logits_shard = (normed_shard @ router_weight).astype(hidden_dtype)

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

    _tp_groups = (
        list(tp_replica_groups) if tp_replica_groups else [list(range(int(tp_degree)))]
    )
    topk_full = cc.all_gather(topk_idx, all_gather_dim=0, replica_groups=_tp_groups)
    affinities_full = cc.all_gather(
        affinities_shard,
        all_gather_dim=0,
        replica_groups=_tp_groups,
    )
    normed_full = cc.all_gather(
        normed_shard, all_gather_dim=0, replica_groups=_tp_groups
    )

    # EP: slice affinities to local experts.
    if int(local_num_experts) > 0 and int(local_num_experts) < n_experts:
        expert_start = int(ep_rank) * int(local_num_experts)
        affinities_full = affinities_full[
            :, expert_start : expert_start + int(local_num_experts)
        ]

    return topk_full, affinities_full, normed_full


# ---------------------------------------------------------------------------
# No-SP (full-token) decode functions for all-layers-in-one-graph mode
# ---------------------------------------------------------------------------


def pre_attn_decode_no_sp_fn(
    hidden: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-SP pre-attn: RMSNorm -> QKV proj (no bias) -> head RMS norms -> RoPE."""
    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, input_norm, eps=rms_norm_eps)
    total_tokens = normed.shape[0]
    q = (normed @ w_q).astype(hidden_dtype).reshape(total_tokens, num_heads, head_dim)
    k = (
        (normed @ w_k)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    v = (
        (normed @ w_v)
        .astype(hidden_dtype)
        .reshape(total_tokens, num_kv_heads, head_dim)
    )
    q = _apply_head_rms_norm(q, q_norm, eps=rms_norm_eps)
    k = _apply_head_rms_norm(k, k_norm, eps=rms_norm_eps)
    q = _apply_rope(q, cos=cos, sin=sin)
    k = _apply_rope(k, cos=cos, sin=sin)
    return q, k, v


def post_attn_decode_no_sp_fn(
    hidden: np.ndarray,
    context: np.ndarray,
    w_o: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    tp_degree: int,
    tp_replica_groups: tuple = (),
) -> np.ndarray:
    """No-SP post-attn: output proj (no bias) -> all_reduce -> residual add."""
    import nkipy.distributed.collectives as cc

    hidden_dtype = hidden.dtype
    total_tokens = hidden.shape[0]
    attn_out = (
        context.reshape(total_tokens, num_heads * head_dim).astype(hidden_dtype) @ w_o
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
    """No-SP router+MoE: RMSNorm -> router (no bias) -> topk+softmax -> blockwise decode MoE."""
    from nkipy.core import tensor_apis

    from nkipy_serving.ops.moe.blockwise_nki import (
        blockwise_decode_all_reduce_add_residual,
    )

    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, post_attn_norm, eps=rms_norm_eps)
    logits = (normed @ router_weight).astype(hidden_dtype)

    # Top-k routing and masked softmax (same logic as seq-parallel router_fn).
    _topk_logits, topk_idx = tensor_apis.topk(logits, k=int(top_k), axis=1)
    topk_idx = topk_idx.astype(np.int8)

    n_experts = int(logits.shape[1])
    expert_mask = tensor_apis.zeros(logits.shape, dtype=np.float32)
    expert_range = np.arange(n_experts, dtype=np.float32) + tensor_apis.zeros(
        n_experts, dtype=np.float32
    )
    for k in range(int(top_k)):
        expert_mask = expert_mask + np.equal(
            topk_idx[:, k : k + 1].astype(np.float32), expert_range
        )

    neg_inf = tensor_apis.full(logits.shape, np.float32(-100000.0), dtype=np.float32)
    masked = (
        expert_mask * logits.astype(np.float32)
        + (np.float32(1.0) - expert_mask) * neg_inf
    )
    masked = masked.astype(np.float32)
    exp_x = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
    affinities = (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(hidden_dtype)

    # EP: slice affinities to local experts.
    if int(local_num_experts) > 0 and int(local_num_experts) < n_experts:
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
        num_experts=int(local_num_experts) if int(local_num_experts) > 0 else n_experts,
        ep_degree=int(ep_degree),
        ep_replica_groups=ep_replica_groups,
        tp_replica_groups=tp_replica_groups,
    )


def router_prefill_no_sp_fn(
    hidden: np.ndarray,
    post_attn_norm: np.ndarray,
    router_weight: np.ndarray,
    *,
    rms_norm_eps: float,
    top_k: int,
    ep_rank: int = 0,
    local_num_experts: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-SP prefill router: RMSNorm -> router -> topk+softmax.

    Returns (topk_idx [T, top_k] int8, affinities_local [T, local_E], normed [T, H]).
    The affinities are EP-sliced to local experts; no collectives.
    """
    from nkipy.core import tensor_apis

    hidden_dtype = hidden.dtype
    normed = _apply_rms_norm(hidden, post_attn_norm, eps=rms_norm_eps)
    logits = (normed @ router_weight).astype(hidden_dtype)

    _topk_logits, topk_idx = tensor_apis.topk(logits, k=int(top_k), axis=1)
    topk_idx = topk_idx.astype(np.int8)

    n_experts = int(logits.shape[1])
    expert_mask = tensor_apis.zeros(logits.shape, dtype=np.float32)
    expert_range = np.arange(n_experts, dtype=np.float32) + tensor_apis.zeros(
        n_experts, dtype=np.float32
    )
    for k in range(int(top_k)):
        expert_mask = expert_mask + np.equal(
            topk_idx[:, k : k + 1].astype(np.float32), expert_range
        )

    neg_inf = tensor_apis.full(logits.shape, np.float32(-100000.0), dtype=np.float32)
    masked = (
        expert_mask * logits.astype(np.float32)
        + (np.float32(1.0) - expert_mask) * neg_inf
    )
    masked = masked.astype(np.float32)
    exp_x = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
    affinities = (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(hidden_dtype)

    if int(local_num_experts) > 0 and int(local_num_experts) < n_experts:
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


def decode_layer_nki_no_sp_fn(
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
    kv_cache: np.ndarray,
    input_norm: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    q_norm: np.ndarray,
    k_norm: np.ndarray,
    w_o: np.ndarray,
    post_attn_norm: np.ndarray,
    router_w: np.ndarray,
    gup_w: np.ndarray,
    gup_bias: np.ndarray,
    down_w: np.ndarray,
    down_bias_bc: np.ndarray,
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
    """Single-layer no-SP decode: pre_attn -> KV update -> attention -> post_attn -> MoE."""
    from nkipy.core import tensor_apis

    from nkipy_serving.attention.nki_blocksparse_flash_attention import (
        nki_attention_unified,
        nki_update_kv_cache_core,
    )

    q, k, v = pre_attn_decode_no_sp_fn(
        hidden,
        input_norm=input_norm,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        q_norm=q_norm,
        k_norm=k_norm,
        cos=cos,
        sin=sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
    )
    kv_cache = nki_update_kv_cache_core(k, v, kv_cache, slot_mapping)

    # Pad QKV to attn_bucket if NKI attention requires a larger sequence dim.
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

    context_attn = nki_attention_unified(
        q_attn,
        k_attn,
        v_attn,
        kv_cache,
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
        num_heads=num_heads,
        head_dim=head_dim,
        tp_degree=tp_degree,
        tp_replica_groups=tp_replica_groups,
    )
    hidden = router_moe_decode_no_sp_fn(
        hidden,
        post_attn_norm=post_attn_norm,
        router_weight=router_w,
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
