"""Build DSV4 runtime components from device weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.assembly.surface import (
    build_dsv4_runtime_surface_from_weights,
)
from nkipy_serving.models.deepseek_v4.assembly.topology import (
    _default_blockwise_ep_groups,
    _default_blockwise_tp_groups,
    _default_v4_tp_groups,
    _ensure_target_only_sampled_runtime,
    _v4_collective_rank_world,
)
from nkipy_serving.models.deepseek_v4.graph_types import (
    Dsv4SampledForwardOptions,
)
from nkipy_serving.models.deepseek_v4.neff_graphs.registry import (
    build_dsv4_graph_fns,
)
from nkipy_serving.models.deepseek_v4.neff_runtime.components import (
    Dsv4RuntimeComponents,
)
from nkipy_serving.sampling.logits_processor import LogitsProcessor


def build_dsv4_runtime_components_from_weights(
    *,
    model_config: Any,
    v4_weights: Any,
    device_weights: Any,
    max_batch_size: int,
    max_seq_len: int,
    attention_backend: Any,
    device_state: Any | None = None,
    build_dir: str | Path | None = None,
    compiler_args: str = "",
    index_construction_max_c_len: int = 0,
    use_blockwise_moe: bool = True,
    blockwise_moe_ep_degree: int = 1,
    blockwise_moe_ep_rank: int = 0,
    blockwise_moe_ep_replica_groups: tuple[tuple[int, ...], ...] | None = None,
    blockwise_moe_tp_degree: int | None = None,
    blockwise_moe_tp_rank: int | None = None,
    blockwise_moe_tp_replica_groups: tuple[tuple[int, ...], ...] | None = None,
    dense_local_topk: int = 1,
    max_requests_per_step: int | None = None,
    product_prefill_moe_blockwise_fusion_max_rows: int = 0,
    product_prefill_moe_dispatch_fusion_max_rows: int = 0,
    product_prefill_dp_attention_post_pre_fusion_max_rows: int = 0,
) -> Dsv4RuntimeComponents:
    """Build runtime components from model metadata + device weights.

    All static shape/RoPE metadata and tensor handles come from
    ``DeepseekV4Weights`` and ``V4DeviceWeights``.
    """
    if attention_backend is None:
        raise RuntimeError("attention_backend is required")
    if device_state is None:
        device_state = getattr(attention_backend, "_device_state", None)
    if device_state is None:
        raise RuntimeError(
            "device_state is required (pass explicitly or set "
            "attention_backend._device_state)"
        )
    if device_weights is None:
        raise RuntimeError("device_weights is required")
    _ensure_target_only_sampled_runtime(model_config, v4_weights)

    runtime_surface = build_dsv4_runtime_surface_from_weights(
        model_config=model_config,
        v4_weights=v4_weights,
        device_weights=device_weights,
        max_batch_size=int(max_batch_size),
        max_seq_len=int(max_seq_len),
    )

    if int(index_construction_max_c_len) <= 0:
        ratios = tuple(
            r
            for r in getattr(runtime_surface.args, "compress_ratios", ())
            if int(r) > 0
        )
        if ratios:
            min_r = min(int(r) for r in ratios)
            derived = int(runtime_surface.max_seq_len) // min_r
            index_construction_max_c_len = max(derived, 1)

    options = Dsv4SampledForwardOptions(
        index_construction_max_c_len=int(index_construction_max_c_len),
    )
    build_dir_s = str(build_dir) if build_dir is not None else None
    collective_rank, collective_world = _v4_collective_rank_world(v4_weights)
    graph = build_dsv4_graph_fns(
        build_dir=build_dir_s,
        compiler_args=compiler_args,
        cc_enabled=int(getattr(v4_weights, "tp_degree", 1)) > 1,
        rank_id=collective_rank,
        world_size=collective_world,
    )
    tp_degree_i = int(
        blockwise_moe_tp_degree
        if blockwise_moe_tp_degree is not None
        else getattr(v4_weights, "tp_degree", 1)
    )
    tp_rank_i = int(
        blockwise_moe_tp_rank
        if blockwise_moe_tp_rank is not None
        else getattr(v4_weights, "tp_rank", 0)
    )
    tp_replica_groups = (
        tuple(tuple(int(r) for r in group) for group in blockwise_moe_tp_replica_groups)
        if blockwise_moe_tp_replica_groups is not None
        else _default_blockwise_tp_groups(v4_weights, tp_degree=tp_degree_i)
    )
    ep_replica_groups = (
        tuple(tuple(int(r) for r in group) for group in blockwise_moe_ep_replica_groups)
        if blockwise_moe_ep_replica_groups is not None
        else _default_blockwise_ep_groups(v4_weights, tp_rank=tp_rank_i)
    )

    blockwise_state = None
    if use_blockwise_moe:
        from nkipy_serving.models.deepseek_v4.neff_runtime.moe.blockwise import (
            build_blockwise_state_from_device_weights,
        )

        blockwise_state = build_blockwise_state_from_device_weights(
            v4_weights,
            device_weights,
            experts_per_token=int(runtime_surface.args.n_activated_experts),
            ep_degree=int(blockwise_moe_ep_degree),
            ep_rank=int(blockwise_moe_ep_rank),
            ep_replica_groups=ep_replica_groups,
            tp_degree=tp_degree_i,
            tp_rank=tp_rank_i,
            tp_replica_groups=tp_replica_groups,
            collective_rank=collective_rank,
            collective_world_size=collective_world,
            swiglu_limit=float(runtime_surface.args.swiglu_limit),
        )
    final_norm_dev = getattr(device_weights, "final_norm", None)
    lm_head_dev = getattr(device_weights, "lm_head", None)
    if final_norm_dev is None or lm_head_dev is None:
        raise RuntimeError(
            "device_weights must provide final_norm and lm_head for DSV4 sampled head"
        )
    lp_dtype = np.dtype(getattr(final_norm_dev, "dtype", np.float32))
    lm_head_rows = int(getattr(lm_head_dev, "shape", (0,))[0])
    if int(dense_local_topk) > lm_head_rows:
        raise RuntimeError(
            "dense_local_topk exceeds DSV4 local LM-head shard size: "
            f"{int(dense_local_topk)} > {lm_head_rows}"
        )
    max_requests_i = int(
        max_requests_per_step
        if max_requests_per_step is not None
        else int(max_batch_size)
    )
    logits_processor = LogitsProcessor(
        vocab_size=int(v4_weights.vocab_size),
        local_vocab_size=int(v4_weights.local_vocab_size),
        vocab_offset=int(v4_weights.lm_head_vocab_offset),
        hidden_size=int(v4_weights.hidden_size),
        dtype=lp_dtype,
        tp_degree=int(v4_weights.tp_degree),
        tp_rank=int(v4_weights.tp_rank),
        tp_replica_groups=_default_v4_tp_groups(
            v4_weights,
            tp_degree=int(v4_weights.tp_degree),
        ),
        collective_rank=int(collective_rank),
        collective_world_size=int(collective_world),
        rms_norm_eps=float(v4_weights.rms_norm_eps),
        dense_local_topk=int(dense_local_topk),
        gather_hidden=False,
        nkipy_compiler_args=compiler_args,
        build_dir=build_dir_s or "/tmp/build",
        max_requests_per_step=max_requests_i,
    )
    embed_tp_sharded = bool(getattr(device_weights, "embed_tp_sharded", False))
    embed_tp_degree = (
        int(getattr(v4_weights, "tp_degree", 1)) if embed_tp_sharded else 1
    )
    embed_tp_replica_groups = (
        _default_v4_tp_groups(v4_weights, tp_degree=embed_tp_degree)
        if embed_tp_sharded
        else ()
    )

    return Dsv4RuntimeComponents(
        runtime_surface=runtime_surface,
        graph=graph,
        options=options,
        build_dir=build_dir_s,
        attention_backend=attention_backend,
        device_state=device_state,
        blockwise_moe_state=blockwise_state,
        logits_processor=logits_processor,
        final_norm_dev=final_norm_dev,
        lm_head_dev=lm_head_dev,
        embed_tp_sharded=embed_tp_sharded,
        embed_vocab_offset=int(getattr(device_weights, "embed_vocab_offset", 0)),
        embed_vocab_end=int(
            getattr(
                device_weights,
                "embed_vocab_end",
                getattr(v4_weights, "vocab_size", 0),
            )
        ),
        embed_tp_degree=embed_tp_degree,
        embed_tp_replica_groups=embed_tp_replica_groups,
        max_requests_per_step=max_requests_i,
        compiler_args=compiler_args,
        product_prefill_moe_blockwise_fusion_max_rows=int(
            product_prefill_moe_blockwise_fusion_max_rows
        ),
        product_prefill_moe_dispatch_fusion_max_rows=int(
            product_prefill_moe_dispatch_fusion_max_rows
        ),
        product_prefill_dp_attention_post_pre_fusion_max_rows=int(
            product_prefill_dp_attention_post_pre_fusion_max_rows
        ),
    )
