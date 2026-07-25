"""Contract tests for TP/EP replica group builders."""

from nkipy_serving.runtime.parallel_groups import (
    build_ep_replica_groups,
    build_tp_replica_groups,
)


def test_tp_ep_replica_group_contracts():
    tp, ep = 8, 16
    total = tp * ep
    tp_groups = build_tp_replica_groups(tp, ep)
    ep_groups = build_ep_replica_groups(tp, ep)

    assert tp_groups[0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert tp_groups[-1] == [120, 121, 122, 123, 124, 125, 126, 127]
    assert ep_groups[0] == [
        0,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
    ]
    assert ep_groups[-1] == [
        7,
        15,
        23,
        31,
        39,
        47,
        55,
        63,
        71,
        79,
        87,
        95,
        103,
        111,
        119,
        127,
    ]

    tp_flat = sorted(r for g in tp_groups for r in g)
    ep_flat = sorted(r for g in ep_groups for r in g)
    assert tp_flat == list(range(total))
    assert ep_flat == list(range(total))

    for global_rank in range(tp * ep):
        tp_rank = global_rank % tp
        ep_rank = global_rank // tp
        assert global_rank in tp_groups[ep_rank]
        assert global_rank in ep_groups[tp_rank]
