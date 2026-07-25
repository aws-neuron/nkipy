"""Rank to (row, col, replica) layout for DeepSeek-V4-Flash.

Ranks are arranged as `replica_degree * rows_per_replica` rows by `tp_degree`
columns. TP rows are dense/attention tensor-parallel groups. MoE expert
parallel groups are TP-column groups across the EP rows of one replica.

Indexing:

    row            = rank // tp_degree          # absolute EP row
    col            = rank % tp_degree           # TP column
    rows_total     = total_workers // tp_degree
    rows_per_repl  = ep_degree                  # one row per EP member
    replica        = row // rows_per_repl
    row_in_replica = row %  rows_per_repl       # MoE EP rank
    attn_lane      = row                        # 0..(rows_total-1)

Defaults at `tp=8, ep=8, replica=2` yield 128 ranks, 16 rows, 16 lanes,
and 2 replicas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V4RankCoord:
    rank: int
    row: int
    col: int
    replica: int
    row_in_replica: int
    attn_lane: int


@dataclass(frozen=True)
class V4LaneRoute:
    """Request routing view for one attention-DP lane.

    A request assigned to ``lane`` runs dense/attention on ``tp_group`` and owns
    KV/state only in that lane. MoE collectives stay inside the lane's model
    replica, represented by ``replica_group`` and ``moe_ep_groups``.
    """

    lane: int
    replica: int
    row_in_replica: int
    tp_group: tuple[int, ...]
    replica_group: tuple[int, ...]
    moe_ep_groups: tuple[tuple[int, ...], ...]
    kv_replicated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "replica": self.replica,
            "row_in_replica": self.row_in_replica,
            "tp_group": list(self.tp_group),
            "replica_group": list(self.replica_group),
            "moe_ep_groups": [list(group) for group in self.moe_ep_groups],
            "kv_replicated": self.kv_replicated,
        }


def _total_workers(tp_degree: int, ep_degree: int, replica_degree: int) -> int:
    return tp_degree * ep_degree * replica_degree


def validate_v4_rank_layout(
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
    world_size: int,
) -> None:
    if tp_degree <= 0 or ep_degree <= 0 or replica_degree <= 0:
        raise RuntimeError(
            f"V4 rank axes must be positive: tp={tp_degree}, ep={ep_degree}, "
            f"replica={replica_degree}"
        )
    expected = _total_workers(tp_degree, ep_degree, replica_degree)
    if expected != world_size:
        raise RuntimeError(
            "V4 rank layout mismatch: tp*ep*replica != world_size. "
            f"tp={tp_degree}, ep={ep_degree}, replica={replica_degree}, "
            f"product={expected}, world_size={world_size}"
        )


def coord_for_rank(
    rank: int,
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
) -> V4RankCoord:
    world_size = _total_workers(tp_degree, ep_degree, replica_degree)
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"rank {rank} out of range for world_size={world_size}")
    rows_per_replica = ep_degree
    row = rank // tp_degree
    col = rank % tp_degree
    replica = row // rows_per_replica
    return V4RankCoord(
        rank=rank,
        row=row,
        col=col,
        replica=replica,
        row_in_replica=row % rows_per_replica,
        attn_lane=row,
    )


def build_tp_row_groups(
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
) -> list[list[int]]:
    """TP rows: one contiguous group of `tp_degree` ranks per row.

    Example (tp=8, ep=8, replica=2): [[0..7], [8..15], ..., [120..127]].
    """
    rows_total = ep_degree * replica_degree
    return [list(range(r * tp_degree, (r + 1) * tp_degree)) for r in range(rows_total)]


def build_moe_ep_row_groups(
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
) -> list[list[int]]:
    """MoE EP groups: same TP column across EP rows of one replica.

    Example (tp=8, ep=8, replica=1): group 0 is
    ``[0, 8, 16, 24, 32, 40, 48, 56]``. With ``ep=1`` each TP rank gets a
    singleton EP group and therefore owns a replicated full expert set.
    """
    groups: list[list[int]] = []
    rows_per_replica = ep_degree
    ranks_per_replica = rows_per_replica * tp_degree
    for replica in range(replica_degree):
        base = replica * ranks_per_replica
        for col in range(tp_degree):
            groups.append(
                [base + row * tp_degree + col for row in range(rows_per_replica)]
            )
    return groups


def build_replica_groups(
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
) -> list[list[int]]:
    """Replica groups: the rows of one model replica.

    One group per replica. Cross-replica communication is never required
    during a step, so this is here for loader/scheduling bookkeeping only.
    """
    rows_per_replica = ep_degree
    return [
        list(
            range(
                r * rows_per_replica * tp_degree,
                (r + 1) * rows_per_replica * tp_degree,
            )
        )
        for r in range(replica_degree)
    ]


def build_attention_dp_lane_routes(
    tp_degree: int,
    ep_degree: int,
    replica_degree: int,
) -> list[V4LaneRoute]:
    """Build the request-routing table for attention-DP lanes.

    For the primary ``TP8/EP8/R2/ADP16`` shape this returns 16 routes: lanes
    0..7 in replica 0 and lanes 8..15 in replica 1. KV is lane-owned, not
    replicated, even when ``replica_degree > 1`` duplicates model weights.
    """
    validate_v4_rank_layout(
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        replica_degree=replica_degree,
        world_size=_total_workers(tp_degree, ep_degree, replica_degree),
    )
    tp_rows = build_tp_row_groups(tp_degree, ep_degree, replica_degree)
    replica_groups = build_replica_groups(tp_degree, ep_degree, replica_degree)
    moe_ep_groups = build_moe_ep_row_groups(tp_degree, ep_degree, replica_degree)

    routes: list[V4LaneRoute] = []
    for lane, tp_group in enumerate(tp_rows):
        replica = lane // ep_degree
        replica_moe_start = replica * tp_degree
        replica_moe_groups = tuple(
            tuple(group)
            for group in moe_ep_groups[
                replica_moe_start : replica_moe_start + tp_degree
            ]
        )
        routes.append(
            V4LaneRoute(
                lane=lane,
                replica=replica,
                row_in_replica=lane % ep_degree,
                tp_group=tuple(tp_group),
                replica_group=tuple(replica_groups[replica]),
                moe_ep_groups=replica_moe_groups,
                kv_replicated=False,
            )
        )
    return routes


def local_expert_ids(
    num_routed_experts: int,
    ep_degree: int,
    ep_rank: int,
) -> tuple[int, ...]:
    """Contiguous expert slice owned by one MoE EP rank.

    256 experts / ep=8 = 32 experts per EP rank. The TP axis does not change
    expert IDs: with ep=1 every TP rank owns the full replicated expert set.
    """
    if num_routed_experts % ep_degree != 0:
        raise RuntimeError(
            f"num_routed_experts {num_routed_experts} not divisible by ep_degree "
            f"{ep_degree}"
        )
    if ep_rank < 0 or ep_rank >= ep_degree:
        raise RuntimeError(f"ep_rank {ep_rank} out of range for ep_degree={ep_degree}")
    local = num_routed_experts // ep_degree
    start = ep_rank * local
    return tuple(range(start, start + local))
