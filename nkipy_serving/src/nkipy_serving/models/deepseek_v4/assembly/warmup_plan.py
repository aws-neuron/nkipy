"""Build DSV4 startup warmup plans from runtime bucket configuration."""

from __future__ import annotations

from dataclasses import dataclass

from nkipy_serving.attention.base import (
    FORWARD_MODE_DECODE,
)
from nkipy_serving.runtime.precompile_paddings import PrecompilePaddings
from nkipy_serving.runtime.warmup import (
    SyntheticWarmupStep,
    build_standard_warmup_steps,
)


@dataclass(frozen=True)
class Dsv4WarmupPlan:
    token_paddings: tuple[int, ...]
    bs_paddings: tuple[int, ...]
    steps: tuple[SyntheticWarmupStep, ...]
    state_write_buckets: tuple[int, ...]
    decode_write_buckets: tuple[int, ...]


def _step_key(step: SyntheticWarmupStep) -> tuple[int, int, int, int, int | None]:
    real_total = (
        int(step.real_total_tokens)
        if step.real_total_tokens is not None
        else int(step.input_token_bucket)
    )
    decode_target_pos = (
        None if step.decode_target_pos is None else int(step.decode_target_pos)
    )
    return (
        int(step.forward_mode),
        int(step.input_token_bucket),
        int(step.batch_size),
        real_total,
        decode_target_pos,
    )


def _append_step_once(
    steps: list[SyntheticWarmupStep],
    seen_steps: set[tuple[int, int, int, int, int | None]],
    step: SyntheticWarmupStep,
) -> None:
    key = _step_key(step)
    if key in seen_steps:
        return
    steps.append(step)
    seen_steps.add(key)


def _state_write_buckets(
    token_paddings: tuple[int, ...],
    *,
    max_batch_size: int,
) -> tuple[int, ...]:
    # Small prompts (< compressor ring) write their full seqlen rows into ring
    # state. The write kernels are bucket-shaped and receive live_rows at
    # runtime, so one bucket covers the old 1..32 exact-length matrix.
    small_prefill_write_bucket = max(
        32,
        max((int(b) for b in token_paddings if 0 < int(b) <= 32), default=0),
    )
    bucket_set = {
        *((int(small_prefill_write_bucket),) if small_prefill_write_bucket > 0 else ()),
    }
    max_batch = max(1, int(max_batch_size))
    for token_bucket in token_paddings:
        token_bucket_i = int(token_bucket)
        if token_bucket_i <= 0:
            continue
        lane_bucket = max(2, (token_bucket_i + max_batch - 1) // max_batch)
        bucket_set.add(int(lane_bucket))
    return tuple(sorted(bucket_set))


def _decode_write_buckets(bs_paddings: tuple[int, ...]) -> tuple[int, ...]:
    # Decode support writes are bucket-shaped and receive live_rows at runtime,
    # so compile the configured request buckets instead of every partial live
    # count inside the largest bucket.
    return tuple(sorted({int(b) for b in bs_paddings if int(b) > 0}))


def build_dsv4_warmup_plan(
    paddings: PrecompilePaddings,
    *,
    product_warmup_enabled: bool,
    has_compressed_layers: bool,
    compressed_boundary_pos: int | None = None,
) -> Dsv4WarmupPlan:
    token_paddings = tuple(int(bucket) for bucket in paddings.token_paddings)
    bs_paddings = tuple(int(bucket) for bucket in paddings.bs_paddings)
    steps = list(build_standard_warmup_steps(paddings))
    seen_steps = {_step_key(step) for step in steps}

    max_batch_size = int(paddings.max_padded_batch_size)

    if product_warmup_enabled:
        product_decode_buckets = tuple(
            sorted({int(bucket) for bucket in bs_paddings if int(bucket) > 0})
        )
        for decode_bucket in product_decode_buckets:
            decode_bucket_i = int(decode_bucket)
            if decode_bucket_i <= 1 or max_batch_size < 1:
                continue
            decode_batch_size = min(decode_bucket_i, max_batch_size)
            _append_step_once(
                steps,
                seen_steps,
                SyntheticWarmupStep(
                    name=f"decode_t{decode_bucket_i}_b1_product_partial",
                    forward_mode=int(FORWARD_MODE_DECODE),
                    input_token_bucket=decode_bucket_i,
                    batch_size=1,
                ),
            )
            if (
                not has_compressed_layers
                or compressed_boundary_pos is None
                or int(compressed_boundary_pos) <= 1
            ):
                continue
            for boundary_batch in sorted({1, int(decode_batch_size)}, reverse=True):
                _append_step_once(
                    steps,
                    seen_steps,
                    SyntheticWarmupStep(
                        name=(
                            f"decode_t{decode_bucket_i}_"
                            f"b{int(boundary_batch)}_"
                            f"product_compress_boundary_p"
                            f"{int(compressed_boundary_pos)}"
                        ),
                        forward_mode=int(FORWARD_MODE_DECODE),
                        input_token_bucket=decode_bucket_i,
                        batch_size=int(boundary_batch),
                        decode_target_pos=int(compressed_boundary_pos),
                    ),
                )

    return Dsv4WarmupPlan(
        token_paddings=token_paddings,
        bs_paddings=bs_paddings,
        steps=tuple(steps),
        state_write_buckets=_state_write_buckets(
            token_paddings,
            max_batch_size=max_batch_size,
        ),
        decode_write_buckets=_decode_write_buckets(bs_paddings),
    )


__all__ = [
    "Dsv4WarmupPlan",
    "build_dsv4_warmup_plan",
]
