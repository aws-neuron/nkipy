"""Runtime-assembly topology helpers for DeepSeek-V4."""

from __future__ import annotations

from typing import Any

__all__ = [
    "_default_blockwise_ep_groups",
    "_default_blockwise_tp_groups",
    "_default_v4_tp_groups",
    "_ensure_target_only_sampled_runtime",
    "_v4_collective_rank_world",
]


def _ensure_target_only_sampled_runtime(model_config: Any, v4_weights: Any) -> None:
    if bool(getattr(model_config, "dsv4_disable_mtp", True)):
        return
    if int(getattr(v4_weights, "num_nextn_predict_layers", 0)) <= 0:
        return
    raise RuntimeError(
        "DSV4 sampled runtime is target-only today. Set "
        "dsv4_disable_mtp=True until device-resident MTP and DSV4 state "
        "rollback/snapshot support are implemented."
    )


def _default_blockwise_tp_groups(
    v4_weights: Any,
    *,
    tp_degree: int,
) -> tuple[tuple[int, ...], ...]:
    return _default_v4_tp_groups(v4_weights, tp_degree=tp_degree)


def _default_v4_tp_groups(
    v4_weights: Any,
    *,
    tp_degree: int,
) -> tuple[tuple[int, ...], ...]:
    tp_degree_i = int(tp_degree)
    has_layout = any(
        hasattr(v4_weights, name)
        for name in ("ep_degree", "replica_degree", "attention_dp_degree")
    )
    if not has_layout:
        lane = int(getattr(v4_weights, "attention_lane", 0))
        base = lane * tp_degree_i
        return (tuple(range(base, base + tp_degree_i)),)
    ep_degree = int(getattr(v4_weights, "ep_degree", 1))
    replica_degree = int(getattr(v4_weights, "replica_degree", 1))
    rows_total = max(
        1,
        int(getattr(v4_weights, "attention_dp_degree", ep_degree * replica_degree)),
        ep_degree * replica_degree,
    )
    return tuple(
        tuple(range(row * tp_degree_i, (row + 1) * tp_degree_i))
        for row in range(rows_total)
    )


def _default_blockwise_ep_groups(
    v4_weights: Any,
    *,
    tp_rank: int,
) -> tuple[tuple[int, ...], ...]:
    tp_degree = int(getattr(v4_weights, "tp_degree", 1))
    ep_degree = int(getattr(v4_weights, "ep_degree", 1))
    replica = int(getattr(v4_weights, "replica_rank", 0))
    base = replica * ep_degree * tp_degree + int(tp_rank)
    return (tuple(base + row * tp_degree for row in range(ep_degree)),)


def _v4_collective_rank_world(v4_weights: Any) -> tuple[int, int]:
    tp_degree = int(getattr(v4_weights, "tp_degree", 1))
    tp_rank = int(getattr(v4_weights, "tp_rank", 0))
    attention_lane = int(getattr(v4_weights, "attention_lane", 0))
    rows_total = int(getattr(v4_weights, "ep_degree", 1)) * int(
        getattr(v4_weights, "replica_degree", 1)
    )
    return attention_lane * tp_degree + tp_rank, tp_degree * rows_total
