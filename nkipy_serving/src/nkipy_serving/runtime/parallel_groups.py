"""Collective communication group builders for TP + EP parallelism.

Global rank layout:  rank = ep_rank * tp_degree + tp_rank
  - tp_rank = rank % tp_degree   (0 .. tp_degree-1)
  - ep_rank = rank // tp_degree  (0 .. ep_degree-1)

TP groups contain ranks with the same ep_rank.
EP groups contain ranks with the same tp_rank.
"""

from __future__ import annotations


def build_tp_replica_groups(tp_degree: int, ep_degree: int) -> list[list[int]]:
    """TP groups: consecutive ranks within each EP slice.

    Example (tp=8, ep=16): [[0..7], [8..15], ..., [120..127]]
    """
    return [
        list(range(ep * tp_degree, (ep + 1) * tp_degree)) for ep in range(ep_degree)
    ]


def build_ep_replica_groups(tp_degree: int, ep_degree: int) -> list[list[int]]:
    """EP groups: same tp_rank across all EP slices.

    Example (tp=8, ep=16): [[0,8,16,...,120], [1,9,...,121], ..., [7,15,...,127]]
    """
    return [
        [tp_rank + ep * tp_degree for ep in range(ep_degree)]
        for tp_rank in range(tp_degree)
    ]
