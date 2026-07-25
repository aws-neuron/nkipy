"""Importable DSV4 bucket-prefill contract experiments.

These experiments are intentionally outside the serving executor. They validate
the shape and masking assumptions needed before product prefill can stop
compiling live-token-shaped NEFFs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from nkipy_serving.ops.deepseek_v4.compressor_state import (
    _bucketed_prefill_swa_owner_pos,
)
from nkipy_serving.ops.moe.blockwise_index import (
    BLOCK_SIZE as MOE_BLOCK_SIZE,
)
from nkipy_serving.ops.moe.blockwise_index import (
    ControlType as MoEControlType,
)
from nkipy_serving.ops.moe.blockwise_index import (
    get_blockwise_expert_and_token_mapping,
)
from nkipy_serving.ops.moe.blockwise_index import (
    get_n_blocks as moe_get_n_blocks,
)
from nkipy_serving.ops.moe.prefill_schedule import build_prefill_moe_schedule
from nkipy_serving.runtime.kernel_compile import kernel_signature_cache_key

_SKIP_DMA = int(MoEControlType.SKIP_DMA.value)


@dataclass(frozen=True)
class BucketPrefillCase:
    batch_size: int
    real_seqlen: int
    bucket_seqlen: int
    hidden_size: int = 8
    window_size: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.real_seqlen <= 0:
            raise ValueError("real_seqlen must be positive")
        if self.bucket_seqlen < self.real_seqlen:
            raise ValueError("bucket_seqlen must cover real_seqlen")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

    @property
    def total_bucket_tokens(self) -> int:
        return int(self.batch_size) * int(self.bucket_seqlen)

    @property
    def total_real_tokens(self) -> int:
        return int(self.batch_size) * int(self.real_seqlen)


def _toy_product_graph(
    x: np.ndarray,
    positions: np.ndarray,
    valid_mask: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Small nkipy-style trace body used by the experiment tests."""

    projected = np.einsum("bsh,hk->bsk", x.astype(np.float32), weight)
    projected = projected + bias.reshape(1, 1, -1)
    projected = projected + positions.astype(np.float32)[..., None] * np.float32(0.01)
    return np.where(valid_mask[..., None], projected, np.float32(0.0)).astype(
        np.float32
    )


def make_request_major_bucket_inputs(case: BucketPrefillCase) -> dict[str, np.ndarray]:
    """Build padded request-major inputs for one bucket-prefill experiment."""

    rng = np.random.default_rng(int(case.seed))
    bsz = int(case.batch_size)
    real = int(case.real_seqlen)
    bucket = int(case.bucket_seqlen)
    hidden = int(case.hidden_size)

    live_x = rng.standard_normal((bsz, real, hidden)).astype(np.float32)
    x = np.zeros((bsz, bucket, hidden), dtype=np.float32)
    x[:, :real, :] = live_x

    positions = np.zeros((bsz, bucket), dtype=np.int32)
    positions[:, :real] = np.arange(real, dtype=np.int32).reshape(1, real)

    valid_mask = np.zeros((bsz, bucket), dtype=bool)
    valid_mask[:, :real] = True

    guard_owner = bsz
    owner_ids = np.full((bsz, bucket), guard_owner, dtype=np.int32)
    owner_ids[:, :real] = np.arange(bsz, dtype=np.int32).reshape(bsz, 1)

    return {
        "live_x": live_x,
        "x": x,
        "positions": positions,
        "valid_mask": valid_mask,
        "owner_ids": owner_ids,
    }


def bucket_graph_matches_live_reference(case: BucketPrefillCase) -> bool:
    """Check that valid bucket rows match the equivalent live-shaped graph."""

    inputs = make_request_major_bucket_inputs(case)
    rng = np.random.default_rng(int(case.seed) + 1)
    hidden = int(case.hidden_size)
    weight = rng.standard_normal((hidden, hidden)).astype(np.float32)
    bias = rng.standard_normal((hidden,)).astype(np.float32)

    bucket_out = _toy_product_graph(
        inputs["x"],
        inputs["positions"],
        inputs["valid_mask"],
        weight,
        bias,
    )
    live_positions = inputs["positions"][:, : int(case.real_seqlen)]
    live_mask = np.ones(
        (int(case.batch_size), int(case.real_seqlen)),
        dtype=bool,
    )
    live_out = _toy_product_graph(
        inputs["live_x"],
        live_positions,
        live_mask,
        weight,
        bias,
    )
    return bool(
        np.allclose(bucket_out[:, : int(case.real_seqlen), :], live_out)
        and np.array_equal(
            bucket_out[:, int(case.real_seqlen) :, :],
            np.zeros_like(bucket_out[:, int(case.real_seqlen) :, :]),
        )
    )


def bucket_signature_key(case: BucketPrefillCase) -> str:
    """Return the shape-based signature key for bucket-shaped samples."""

    inputs = make_request_major_bucket_inputs(case)
    hidden = int(case.hidden_size)
    return kernel_signature_cache_key(
        _toy_product_graph,
        name="dsv4_bucket_prefill_experiment",
        sample_args=(
            inputs["x"],
            inputs["positions"],
            inputs["valid_mask"],
            np.zeros((hidden, hidden), dtype=np.float32),
            np.zeros((hidden,), dtype=np.float32),
        ),
        kwargs={},
        additional_compiler_args="",
        target="experiment-target",
    )


def live_signature_key(case: BucketPrefillCase) -> str:
    """Return the signature key that a live-shaped sample would produce."""

    inputs = make_request_major_bucket_inputs(case)
    hidden = int(case.hidden_size)
    live_positions = inputs["positions"][:, : int(case.real_seqlen)]
    live_mask = np.ones(
        (int(case.batch_size), int(case.real_seqlen)),
        dtype=bool,
    )
    return kernel_signature_cache_key(
        _toy_product_graph,
        name="dsv4_bucket_prefill_experiment",
        sample_args=(
            inputs["live_x"],
            live_positions,
            live_mask,
            np.zeros((hidden, hidden), dtype=np.float32),
            np.zeros((hidden,), dtype=np.float32),
        ),
        kwargs={},
        additional_compiler_args="",
        target="experiment-target",
    )


def guarded_swa_scatter_matches_tail_reference(case: BucketPrefillCase) -> bool:
    """Check the bucketed SWA owner/position recipe against a host oracle."""

    rng = np.random.default_rng(int(case.seed) + 2)
    bsz = int(case.batch_size)
    bucket = int(case.bucket_seqlen)
    real = int(case.real_seqlen)
    hidden = int(case.hidden_size)
    window = int(case.window_size)
    guard_owner = bsz
    n_owners = guard_owner + 1

    rows = rng.standard_normal((bsz * bucket, hidden)).astype(np.float32)
    sentinel = np.float32(-777.0)
    got = np.full((n_owners * window, hidden), sentinel, dtype=np.float32)
    expected = got.copy()

    owners, positions = _bucketed_prefill_swa_owner_pos(
        bsz=bsz,
        bucket_seqlen=bucket,
        real_seqlen=real,
        window_size=window,
        guard_owner=guard_owner,
    )
    for src, (owner, pos) in enumerate(zip(owners, positions, strict=True)):
        dst = int(owner) * window + int(pos) % window
        got[dst] = rows[src]

    for batch in range(bsz):
        for pos in range(max(0, real - window), real):
            expected[batch * window + pos % window] = rows[batch * bucket + pos]

    live_rows = bsz * window
    return bool(np.array_equal(got[:live_rows], expected[:live_rows]))


def make_request_major_valid_mask(
    *,
    batch_size: int,
    real_seqlen: int,
    bucket_seqlen: int,
) -> np.ndarray:
    valid = np.zeros((int(batch_size), int(bucket_seqlen)), dtype=bool)
    valid[:, : int(real_seqlen)] = True
    return valid


def build_prefill_moe_schedule_from_valid_mask(
    topk_idx: np.ndarray,
    *,
    valid_mask: np.ndarray,
    token_bucket: int,
    experts_per_token: int,
    local_num_experts: int,
    ep_degree: int,
    ep_rank: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Experiment-only MoE schedule builder for non-prefix valid rows.

    Production ``build_prefill_moe_schedule`` only receives a total real-token
    count, which is sufficient for packed or single-request layouts. A
    request-major bucket rectangle needs a row-validity mask instead.
    """

    topk = np.asarray(topk_idx, dtype=np.int32).copy()
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if topk.shape[0] != valid.shape[0]:
        raise ValueError(
            f"topk rows {topk.shape[0]} do not match valid rows {valid.shape[0]}"
        )
    topk[~valid] = _SKIP_DMA

    if int(ep_degree) > 1:
        e0 = int(ep_rank) * int(local_num_experts)
        e1 = e0 + int(local_num_experts)
        local = (topk >= e0) & (topk < e1)
        topk = np.where(local, topk - e0, _SKIP_DMA).astype(np.int32)

    num_blocks, num_static_blocks = moe_get_n_blocks(
        int(token_bucket),
        int(experts_per_token),
        int(local_num_experts),
    )
    _num_real, block_to_expert, token_position_to_id = (
        get_blockwise_expert_and_token_mapping(
            top_k_indices=topk,
            num_blocks=int(num_blocks),
            block_size=MOE_BLOCK_SIZE,
            num_experts=int(local_num_experts),
            num_static_blocks=int(num_static_blocks),
        )
    )
    return (
        token_position_to_id,
        block_to_expert,
        int(num_blocks),
        int(num_static_blocks),
    )


def scheduled_token_ids(token_position_to_id: np.ndarray) -> set[int]:
    values = np.asarray(token_position_to_id, dtype=np.int32).reshape(-1)
    return {int(v) for v in values if int(v) != _SKIP_DMA}


def run_summary() -> dict[str, Any]:
    lengths = (10, 37, 129)
    bucket_cases = [
        BucketPrefillCase(batch_size=1, real_seqlen=n, bucket_seqlen=256, seed=n)
        for n in lengths
    ]
    bucket_keys = {bucket_signature_key(case) for case in bucket_cases}
    live_keys = {live_signature_key(case) for case in bucket_cases}

    bsz = 2
    real = 3
    bucket = 16
    topk = np.zeros((bsz * bucket, 2), dtype=np.int32)
    topk[:, 0] = np.arange(bsz * bucket, dtype=np.int32) % 4
    topk[:, 1] = (topk[:, 0] + 1) % 4
    valid_mask = make_request_major_valid_mask(
        batch_size=bsz,
        real_seqlen=real,
        bucket_seqlen=bucket,
    )
    current_tp, _current_b2e, _nb, _ns = build_prefill_moe_schedule(
        topk,
        token_bucket=bsz * bucket,
        real_total_tokens=bsz * real,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )
    mask_tp, _mask_b2e, _nb, _ns = build_prefill_moe_schedule_from_valid_mask(
        topk,
        valid_mask=valid_mask,
        token_bucket=bsz * bucket,
        experts_per_token=2,
        local_num_experts=4,
        ep_degree=1,
        ep_rank=0,
    )
    valid_ids = {int(i) for i in np.flatnonzero(valid_mask.reshape(-1))}

    return {
        "graph_boundary": "nkipy",
        "moe_boundary": "NKI kernel wrapped by nkipy",
        "bucket_signature_count": len(bucket_keys),
        "live_signature_count": len(live_keys),
        "bucket_graph_matches_live": all(
            bucket_graph_matches_live_reference(case) for case in bucket_cases
        ),
        "guarded_swa_matches_tail": all(
            guarded_swa_scatter_matches_tail_reference(case) for case in bucket_cases
        ),
        "current_total_count_schedule_is_request_major_safe": (
            scheduled_token_ids(current_tp) == valid_ids
        ),
        "mask_schedule_is_request_major_safe": scheduled_token_ids(mask_tp)
        == valid_ids,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON summary")
    args = parser.parse_args(argv)

    summary = run_summary()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
