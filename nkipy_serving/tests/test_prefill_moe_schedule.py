"""CPU-only tests for build_prefill_moe_schedule.

Covers the two semantics that are load-bearing for both production executors
and the MoE eager executors: padded rows become SKIP_DMA, and EP remaps global
expert ids to local ids.
"""

from __future__ import annotations

import numpy as np

from nkipy_serving.ops.moe.blockwise_index import (
    ControlType as MoEControlType,
)
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule

_SKIP = int(MoEControlType.SKIP_DMA.value)


def test_padding_rows_become_skip_dma():
    """Rows >= real_total_tokens must not route to any expert."""
    token_bucket = 128
    real = 3
    top_k = 2
    E = 4

    topk = np.zeros((token_bucket, top_k), dtype=np.int8)
    topk[:real] = [[0, 1], [1, 2], [2, 3]]
    # Give padding rows bogus expert ids so we can verify masking happens.
    topk[real:] = [[0, 0]]

    token_pos, _b2e, _nb, _ns = build_prefill_moe_schedule(
        topk,
        token_bucket=token_bucket,
        real_total_tokens=real,
        experts_per_token=top_k,
        local_num_experts=E,
        ep_degree=1,
        ep_rank=0,
    )

    # Every populated slot must reference a real token id (< real) or SKIP.
    mask = token_pos != _SKIP
    referenced = token_pos[mask]
    assert referenced.size > 0, "schedule produced no real references"
    assert (referenced < real).all(), (
        f"token_pos references padded row ids: {np.unique(referenced)}"
    )


def test_ep_partitions_experts_across_ranks():
    """Every real (token, expert) routing lands on exactly one rank's schedule."""
    token_bucket = 128
    real = token_bucket
    top_k = 2
    E_global = 8
    E_local = 4
    ep_degree = 2

    rng = np.random.default_rng(1)
    topk = rng.integers(0, E_global, size=(token_bucket, top_k)).astype(np.int8)

    # For each rank, collect (token_id, global_expert_id) pairs it schedules.
    scheduled_pairs: list[set[tuple[int, int]]] = []
    for ep_rank in (0, 1):
        tp, b2e, _nb, _ns = build_prefill_moe_schedule(
            topk,
            token_bucket=token_bucket,
            real_total_tokens=real,
            experts_per_token=top_k,
            local_num_experts=E_local,
            ep_degree=ep_degree,
            ep_rank=ep_rank,
        )
        # tp[block, slot] = token_id or SKIP; each block maps to one expert.
        pairs: set[tuple[int, int]] = set()
        for block_idx in range(tp.shape[0]):
            local_expert = int(b2e[block_idx])
            if local_expert < 0:
                continue
            global_expert = ep_rank * E_local + local_expert
            for slot in range(tp.shape[1]):
                tok = int(tp[block_idx, slot])
                if tok != _SKIP:
                    pairs.add((tok, global_expert))
        scheduled_pairs.append(pairs)

    # No (token, expert) pair scheduled on both ranks.
    overlap = scheduled_pairs[0] & scheduled_pairs[1]
    assert not overlap, f"pairs scheduled on both ranks: {overlap}"

    # Union covers every (token, top-k expert) pair from the router output.
    expected = {(t, int(topk[t, k])) for t in range(real) for k in range(top_k)}
    got = scheduled_pairs[0] | scheduled_pairs[1]
    assert got == expected, f"missing: {expected - got}; extraneous: {got - expected}"


def test_caller_array_not_mutated():
    """The helper must not mutate the caller's topk array (defensive copy)."""
    topk = np.array([[0, 1], [2, 3]], dtype=np.int8)
    before = topk.copy()
    build_prefill_moe_schedule(
        topk,
        token_bucket=2,
        real_total_tokens=1,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )
    assert np.array_equal(topk, before), "build_prefill_moe_schedule mutated input"
