from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager

import numpy as np

from nkipy_serving.attention.base import (
    FORWARD_MODE_DECODE,
    FORWARD_MODE_EXTEND,
    AttentionMetadata,
)
from nkipy_serving.runtime.precompile_paddings import PrecompilePaddings
from nkipy_serving.runtime.shape_guard import select_bucket


@dataclass(frozen=True)
class SyntheticWarmupStep:
    name: str
    forward_mode: int
    input_token_bucket: int
    batch_size: int
    real_total_tokens: int | None = None
    decode_target_pos: int | None = None
    execute_forward: bool = True


def build_standard_warmup_steps(
    paddings: PrecompilePaddings,
) -> tuple[SyntheticWarmupStep, ...]:
    token_paddings = tuple(int(bucket) for bucket in paddings.token_paddings)
    max_batch_size = int(paddings.max_padded_batch_size)
    steps: list[SyntheticWarmupStep] = []
    for token_bucket in token_paddings:
        # Serving fills a bucket with any batch in the request ladder; each
        # (bucket, bs) rectangle is its own kernel key, so warm them all (not
        # only max batch).
        for raw_bs in paddings.bs_paddings:
            bs = min(int(token_bucket), int(raw_bs), max_batch_size)
            if bs <= 0:
                continue
            steps.append(
                SyntheticWarmupStep(
                    name=f"extend_t{int(token_bucket)}_b{bs}",
                    forward_mode=int(FORWARD_MODE_EXTEND),
                    input_token_bucket=int(token_bucket),
                    batch_size=bs,
                )
            )
    for raw_bs in paddings.bs_paddings:
        input_token_bucket = int(raw_bs)
        if input_token_bucket <= 0:
            continue
        # bs_paddings may be normalized upward (1 -> 2) to avoid NKI
        # dim-1 codegen issues.  Keep the padded decode token bucket, but
        # only synthesize as many live requests as the scheduler can produce.
        batch_size = min(input_token_bucket, max_batch_size)
        if batch_size <= 0:
            continue
        steps.append(
            SyntheticWarmupStep(
                name=f"decode_t{input_token_bucket}_b{batch_size}",
                forward_mode=int(FORWARD_MODE_DECODE),
                input_token_bucket=input_token_bucket,
                batch_size=batch_size,
            )
        )
    return tuple(steps)


def balanced_seq_lens(total_tokens: int, batch_size: int) -> np.ndarray:
    total_tokens = int(total_tokens)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"batch_size must be positive, got {batch_size}")
    if total_tokens < batch_size:
        raise RuntimeError(
            "total_tokens must be at least batch_size for synthetic warmup. "
            f"Got total_tokens={total_tokens}, batch_size={batch_size}"
        )
    base, rem = divmod(total_tokens, batch_size)
    seq_lens = np.full((batch_size,), base, dtype=np.int64)
    if rem > 0:
        seq_lens[:rem] += 1
    return seq_lens


def build_synthetic_warmup_inputs(
    step: SyntheticWarmupStep,
    *,
    token_paddings: tuple[int, ...],
    bs_paddings: tuple[int, ...] | None = None,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, AttentionMetadata]:
    input_token_bucket = int(step.input_token_bucket)
    batch_size = int(step.batch_size)
    forward_mode = int(step.forward_mode)
    configured_real_total = step.real_total_tokens
    if forward_mode == int(FORWARD_MODE_DECODE):
        real_total_tokens = batch_size
        if step.decode_target_pos is None:
            seq_lens = balanced_seq_lens(input_token_bucket, batch_size)
        else:
            target_pos = max(0, int(step.decode_target_pos))
            seq_lens = np.full(
                (batch_size,),
                target_pos + 1,
                dtype=np.int64,
            )
    else:
        real_total_tokens = (
            int(configured_real_total)
            if configured_real_total is not None
            else input_token_bucket
        )
        if real_total_tokens <= 0:
            raise RuntimeError(
                "Synthetic warmup real_total_tokens must be positive for extend. "
                f"Got real_total_tokens={real_total_tokens}"
            )
        if real_total_tokens > input_token_bucket:
            raise RuntimeError(
                "Synthetic warmup real_total_tokens must fit input token bucket. "
                f"Got real_total_tokens={real_total_tokens}, "
                f"input_token_bucket={input_token_bucket}"
            )
        seq_lens = balanced_seq_lens(real_total_tokens, batch_size)

    block_counts = np.asarray(
        [(int(seq_len) + block_size - 1) // block_size for seq_len in seq_lens],
        dtype=np.int64,
    )
    total_blocks = int(block_counts.sum())
    if total_blocks > int(num_blocks):
        raise RuntimeError(
            "Synthetic warmup requires more KV cache blocks than available. "
            f"required_blocks={total_blocks}, available_blocks={int(num_blocks)}, "
            f"total_tokens={int(real_total_tokens)}, block_size={int(block_size)}"
        )
    start_block = max(0, int(num_blocks) - 1 - total_blocks)
    max_block_count = int(block_counts.max()) if block_counts.size > 0 else 1
    block_tables = np.zeros((batch_size, max(max_block_count, 1)), dtype=np.int64)
    slot_segments: list[np.ndarray] = []
    current_block = start_block
    for req_idx, seq_len in enumerate(seq_lens):
        req_block_count = int(block_counts[req_idx])
        req_block_ids = np.arange(
            current_block, current_block + req_block_count, dtype=np.int64
        )
        current_block += req_block_count
        block_tables[req_idx, :req_block_count] = req_block_ids
        positions = np.arange(int(seq_len), dtype=np.int64)
        req_slots = req_block_ids[positions // block_size] * block_size + (
            positions % block_size
        )
        slot_segments.append(req_slots)

    if forward_mode == int(FORWARD_MODE_DECODE):
        input_ids_raw = np.zeros((batch_size,), dtype=np.int32)
        positions_raw = (seq_lens.astype(np.int32) - 1).astype(np.int32, copy=False)
        slot_mapping_raw = np.asarray(
            [int(slot_segments[i][int(seq_lens[i]) - 1]) for i in range(batch_size)],
            dtype=np.int64,
        )
        query_start_loc = np.arange(batch_size + 1, dtype=np.int64)
    else:
        input_ids_raw = np.zeros((real_total_tokens,), dtype=np.int32)
        positions_raw = np.concatenate(
            [np.arange(int(seq_len), dtype=np.int32) for seq_len in seq_lens],
            axis=0,
        )
        slot_mapping_raw = np.concatenate(slot_segments, axis=0)
        query_start_loc = np.zeros((batch_size + 1,), dtype=np.int64)
        query_start_loc[1:] = np.cumsum(seq_lens, dtype=np.int64)

    if forward_mode == int(FORWARD_MODE_DECODE):
        allowed_decode_buckets = tuple(
            sorted(
                {
                    int(bucket)
                    for bucket in (
                        tuple(token_paddings)
                        + tuple(() if bs_paddings is None else bs_paddings)
                    )
                }
            )
        )
        if input_token_bucket not in allowed_decode_buckets:
            raise RuntimeError(
                "Synthetic warmup decode token bucket is not configured. "
                f"got={input_token_bucket}, allowed={allowed_decode_buckets}"
            )
    else:
        resolved_input_bucket = select_bucket(
            int(real_total_tokens), token_paddings, "token"
        )
        if resolved_input_bucket != input_token_bucket:
            raise RuntimeError(
                "Synthetic warmup input token bucket mismatch. "
                f"expected={input_token_bucket} resolved={resolved_input_bucket} "
                f"forward_mode={forward_mode} real_total_tokens={real_total_tokens}"
            )

    input_ids = np.zeros((input_token_bucket,), dtype=np.int32)
    input_ids[: int(real_total_tokens)] = input_ids_raw
    positions = np.zeros((input_token_bucket,), dtype=np.int32)
    positions[: int(real_total_tokens)] = positions_raw
    # Pad to the pool's last slot (the serve-time pad sink), so padded rows
    # never collide with the warmup blocks at the pool tail's start.
    pad_slot = int(num_blocks) * int(block_size) - 1
    slot_mapping = np.full((input_token_bucket,), pad_slot, dtype=np.int64)
    slot_mapping[: int(real_total_tokens)] = slot_mapping_raw
    # Keep prefill slot_mapping at the FULL token bucket: serve presents
    # bucket-padded device tensors (backend.prepare pads with the sink slot),
    # so a real-sized warmup slot_mapping makes per-length NEFFs the serve
    # path never uses (QKV/MoE capacity gates key on slot rows). Padding rows
    # use the pool's last slot, mirroring serve.
    metadata = AttentionMetadata(
        forward_mode=forward_mode,
        seq_lens=seq_lens.astype(np.int64, copy=False),
        slot_mapping=(
            slot_mapping[: int(real_total_tokens)]
            if forward_mode == int(FORWARD_MODE_DECODE)
            else slot_mapping
        ),
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        total_tokens=int(real_total_tokens),
        batch_size=int(batch_size),
        max_seq_len=int(seq_lens.max()) if seq_lens.size > 0 else 0,
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        block_size=int(block_size),
    )
    return input_ids, positions, metadata


def run_synthetic_warmup_steps(
    steps: tuple[SyntheticWarmupStep, ...],
    *,
    token_paddings: tuple[int, ...],
    bs_paddings: tuple[int, ...] | None = None,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    forward: Callable[..., object],
    profiling_context: Callable[[], ContextManager[None]] | None = None,
) -> None:
    context_factory = (
        profiling_context if profiling_context is not None else nullcontext
    )
    with context_factory():
        for step in steps:
            input_ids, positions, attn_metadata = build_synthetic_warmup_inputs(
                step,
                token_paddings=token_paddings,
                bs_paddings=bs_paddings,
                num_blocks=int(num_blocks),
                block_size=int(block_size),
                num_kv_heads=int(num_kv_heads),
                head_dim=int(head_dim),
            )
            forward(
                input_ids=input_ids,
                positions=positions,
                kv_caches=[],
                attn_metadata=attn_metadata,
                token_bucket=int(step.input_token_bucket),
                real_total_tokens=int(attn_metadata.total_tokens),
            )
