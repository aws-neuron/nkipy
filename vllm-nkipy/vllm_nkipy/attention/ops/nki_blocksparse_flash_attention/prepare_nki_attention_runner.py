# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Utility function to prepare NKIFlashPagedAttentionRunner
from scheduler output and input batch.
"""

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.nki_attention_runner import (  # noqa: E501
    ContextAttnInputs,
    NKIFlashPagedAttentionRunner,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.nki_attention_runner_device import (  # noqa: E501
    NKIFlashPagedAttentionRunnerForDevice,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.tpu_input_batch import InputBatch


def _get_rank():
    """Get the current process rank for distributed training."""
    # Try different ways to get rank
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif (
        hasattr(torch.distributed, "is_initialized")
        and torch.distributed.is_initialized()
    ):
        return torch.distributed.get_rank()
    else:
        return 0  # Default to rank 0 if not in distributed mode


def _should_log():
    """Check if current process should log (only rank 0)."""
    return _get_rank() == 0


def _log_context_attn_inputs(inputs: ContextAttnInputs, name: str):
    """Helper function to log ContextAttnInputs details."""
    if inputs is None:
        print(f"{name}: None")
        return

    print(f"\n=== {name} ===")

    # Log tile_q_indices
    if inputs.tile_q_indices is not None:
        tqi = inputs.tile_q_indices
        print(
            f"tile_q_indices: shape={tqi.shape},"
            f" dtype={tqi.dtype},"
            f" device={tqi.device}"
        )
        print(
            f"  sample values:"
            f" {tqi.flatten()[:10].tolist()}"
        )
    else:
        print("tile_q_indices: None")

    # Log tile_block_tables
    if inputs.tile_block_tables is not None:
        tbt = inputs.tile_block_tables
        print(
            f"tile_block_tables: shape={tbt.shape},"
            f" dtype={tbt.dtype},"
            f" device={tbt.device}"
        )
        print(
            f"  sample values:"
            f" {tbt.flatten()[:10].tolist()}"
        )
    else:
        print("tile_block_tables: None")

    # Log tile_masks
    if inputs.tile_masks is not None:
        tm = inputs.tile_masks
        print(
            f"tile_masks: shape={tm.shape},"
            f" dtype={tm.dtype},"
            f" device={tm.device}"
        )
        print(
            f"  sample values:"
            f" {tm.flatten()[:10].tolist()}"
        )
    else:
        print("tile_masks: None")

    # Log num_dynamic_loop_steps
    if inputs.num_dynamic_loop_steps is not None:
        ndls = inputs.num_dynamic_loop_steps
        print(
            f"num_dynamic_loop_steps:"
            f" shape={ndls.shape},"
            f" dtype={ndls.dtype},"
            f" device={ndls.device}"
        )
        print(
            f"  values: {ndls.flatten().tolist()}"
        )
    else:
        print("num_dynamic_loop_steps: None")

    # Log last_tile_indices
    if inputs.last_tile_indices is not None:
        lti = inputs.last_tile_indices
        print(
            f"last_tile_indices: shape={lti.shape},"
            f" dtype={lti.dtype},"
            f" device={lti.device}"
        )
        print(
            f"  sample values:"
            f" {lti.flatten()[:10].tolist()}"
        )
    else:
        print("last_tile_indices: None")

    # Log q_update_pred
    if inputs.q_update_pred is not None:
        qup = inputs.q_update_pred
        print(
            f"q_update_pred: shape={qup.shape},"
            f" dtype={qup.dtype},"
            f" device={qup.device}"
        )
        print(
            f"  sample values:"
            f" {qup.flatten()[:10].tolist()}"
        )
    else:
        print("q_update_pred: None")

    print(f"=== End {name} ===\n")


def _log_active_mask(active_mask, name: str = "active_mask"):
    """Helper function to log active_mask details."""
    if active_mask is None:
        print(f"{name}: None")
        return

    print(f"\n=== {name} ===")
    print(
        f"shape: {active_mask.shape},"
        f" dtype: {active_mask.dtype},"
        f" device: {active_mask.device}"
    )
    print(
        f"non-zero elements: {active_mask.nonzero().shape[0]} / {active_mask.numel()}"
    )

    # Show a small sample of the mask
    if active_mask.numel() > 0:
        sample_size = min(10, active_mask.shape[0], active_mask.shape[1])
        print(f"sample ({sample_size}x{sample_size} top-left corner):")
        print(active_mask[:sample_size, :sample_size].cpu().numpy())

    print(f"=== End {name} ===\n")


def _prepare_nki_attention_runner(
    req_ids: np.ndarray,
    num_scheduled_tokens: dict,
    num_computed_tokens_cpu: np.ndarray,
    block_table: torch.Tensor,
    num_reqs: int,
    num_blocks: int,
    block_size: int,
    large_q_tile_size: int,
    large_kv_tile_size: int,
    max_model_len: int,
    dynamic_loop_unrolling_size: int = 8,
    enable_separate_prefill_decode: bool = True,
    mixed_precision: bool = True,
    skip_active: bool = False,
    include_prompt_in_ctx: bool = True,
    max_num_prefill_tiles: int = None,
    max_num_decode_tiles: int = None,
    padded_query_length: int = None,
    max_num_reqs: Optional[int] = None,
    sliding_window: Optional[int] = None,
) -> NKIFlashPagedAttentionRunner:
    """
    Internal function to prepare a
    NKIFlashPagedAttentionRunner from individual components.

    This function takes the extracted components from scheduler output and input batch
    and creates the NKI attention runner.

    Args:
        req_ids: Array of request IDs
        num_scheduled_tokens: Dictionary mapping request ID
            to number of scheduled tokens
        num_computed_tokens_cpu: Array of number of computed tokens per request
        block_table: Block table tensor containing KV cache block indices
        num_reqs: Number of requests in the batch
        num_blocks: Total number of blocks in KV cache
        block_size: Size of each KV cache block (e.g., 32, 64)
        large_kv_tile_size: Tile size for KV dimension (e.g., 1024, 2048, 4096)
        max_model_len: Maximum sequence length the model supports
        dynamic_loop_unrolling_size: Loop unrolling factor for compilation (default: 8)
        enable_separate_prefill_decode: Whether to separate prefill and decode phases
        mixed_precision: Whether to use mixed precision (bf16/fp32)
        skip_active: Whether to skip active token computation
        include_prompt_in_ctx: Whether to include prompt tokens in context
        max_num_prefill_tiles: Maximum number of prefill tiles (for padding)
        max_num_decode_tiles: Maximum number of decode tiles (for padding)
        padded_query_length: Padded query length (for compilation)
        max_num_reqs: Maximum number of requests to pad to (for avoiding recompilation)

    Returns:
        NKIFlashPagedAttentionRunner: Initialized runner with tile plans prepared
    """
    # Get query lengths for each request from scheduler output
    # query_lens represents how many new tokens are being processed for each request
    query_lens = []
    for req_id in req_ids[:num_reqs]:
        if req_id is None:
            break
        num_tokens = num_scheduled_tokens.get(req_id, 0)
        query_lens.append(num_tokens)

    query_lens = torch.tensor(query_lens, dtype=torch.long)

    # Get context lengths for each request
    # context_lens represents how many tokens have already been computed (in KV cache)
    context_lens = torch.tensor(
        num_computed_tokens_cpu[: len(query_lens)].tolist(), dtype=torch.long
    )

    # Pad query_lens and context_lens if max_num_reqs is set to avoid recompilation
    if max_num_reqs is not None and max_num_reqs > len(query_lens):
        padding_size = max_num_reqs - len(query_lens)
        query_lens = torch.cat(
            [query_lens, torch.zeros(padding_size, dtype=torch.long)]
        )
        context_lens = torch.cat(
            [context_lens, torch.zeros(padding_size, dtype=torch.long)]
        )

    # Determine if this is mostly decode (most query_lens == 1)
    (query_lens == 1).sum().item()
    # is_mostly_decode = num_decode_reqs > len(query_lens) // 2

    # Set large_q_tile_size based on workload
    # For decode-heavy workloads, use tile size of 1
    # For prefill-heavy workloads, use larger tile sizes
    # if is_mostly_decode:
    #     large_q_tile_size = 1
    # else:
    # For prefill, use a reasonable tile size (32, 64, 128, etc.)
    # Choose based on typical query lengths
    # max_query_len = query_lens.max().item()
    # if max_query_len <= 32:
    #     large_q_tile_size = 32
    # elif max_query_len <= 64:
    #     large_q_tile_size = 64
    # elif max_query_len <= 128:
    #     large_q_tile_size = 128
    # else:
    #     large_q_tile_size = 256
    # large_q_tile_size = 128

    # Get block tables - already extracted as parameter
    # block_table shape: (max_num_reqs, max_num_blocks_per_req)
    block_tables = block_table[:num_reqs]

    # Calculate max_kv_cache_size from the KV cache shape
    # This is the total number of blocks available in the cache
    max_kv_cache_size = num_blocks

    # Initialize NKIFlashPagedAttentionRunner
    nki_runner = NKIFlashPagedAttentionRunner(
        query_lens=query_lens,
        context_lens=context_lens,
        large_q_tile_size=large_q_tile_size,
        large_kv_tile_size=large_kv_tile_size,
        block_size=block_size,
        max_req_seq_len=max_model_len,
        include_prompt_in_ctx=include_prompt_in_ctx,
        dynamic_loop_unrolling_size=dynamic_loop_unrolling_size,
        enable_separate_prefill_decode=enable_separate_prefill_decode,
        padded_query_length=padded_query_length,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        skip_active=skip_active or include_prompt_in_ctx,
        exec_mode="dynamo",
    )

    # Prepare tile plan inputs
    nki_runner.prepare_tile_plan_inputs(
        block_tables=block_tables,
        max_kv_cache_size=max_kv_cache_size,
        sliding_window=sliding_window,
    )

    # Log tile plan and active_mask after prepare_tile_plan_inputs (only on rank 0)
    if _should_log():
        print("\n" + "=" * 80)
        print("TILE PLAN AND ACTIVE MASK LOGGING")
        print("=" * 80)

        # Log prefill context inputs
        _log_context_attn_inputs(nki_runner.prefill_ctx_inputs, "PREFILL_CTX_INPUTS")

        # Log decode context inputs
        _log_context_attn_inputs(nki_runner.decode_ctx_inputs, "DECODE_CTX_INPUTS")

        # Log active mask
        _log_active_mask(nki_runner.active_mask, "ACTIVE_MASK")

        # Log additional tile plan information
        print("\n=== TILE PLAN SUMMARY ===")
        print(f"padded_query_length: {padded_query_length}")
        print(f"query_lens: {query_lens}")
        print(f"context_lens: {context_lens.cpu().tolist()}")
        print(f"prefill_batch_size: {nki_runner.prefill_batch_size}")
        print(f"batch_size: {nki_runner.batch_size}")
        print(f"num_actual_tokens: {nki_runner.num_actual_tokens}")
        print(
            "num_active_tokens_after_padding:"
            f" {nki_runner.num_active_tokens_after_padding}"
        )
        print(f"large_q_tile_size: {nki_runner.large_q_tile_size}")
        print(f"large_kv_tile_size: {nki_runner.large_kv_tile_size}")
        print(f"block_size: {nki_runner.block_size}")

        if nki_runner.prefill_plan is not None:
            print(f"prefill_plan.num_tiles: {nki_runner.prefill_plan.num_tiles}")
            print(
                f"prefill_plan.num_real_tiles: {nki_runner.prefill_plan.num_real_tiles}"
            )
        else:
            print("prefill_plan: None")

        if nki_runner.decode_plan is not None:
            print(f"decode_plan.num_tiles: {nki_runner.decode_plan.num_tiles}")
            print(
                f"decode_plan.num_real_tiles: {nki_runner.decode_plan.num_real_tiles}"
            )
        else:
            print("decode_plan: None")

        print("=== END TILE PLAN SUMMARY ===")
        print("=" * 80)
        print("END TILE PLAN AND ACTIVE MASK LOGGING")
        print("=" * 80 + "\n")

    device_runner = NKIFlashPagedAttentionRunnerForDevice(
        prefill_ctx_inputs=nki_runner.prefill_ctx_inputs,
        decode_ctx_inputs=nki_runner.decode_ctx_inputs,
        active_mask=nki_runner.active_mask,
        num_active_tokens_after_padding=nki_runner.num_active_tokens_after_padding,
        large_q_tile_size=nki_runner.large_q_tile_size,
        large_kv_tile_size=nki_runner.large_kv_tile_size,
        block_size=nki_runner.block_size,
        dynamic_loop_unrolling_size=nki_runner.dynamic_loop_unrolling_size,
        skip_active=nki_runner.skip_active,
        max_num_prefill_tiles=nki_runner.max_num_prefill_tiles,
        max_num_decode_tiles=nki_runner.max_num_decode_tiles,
        prefill_plan_num_tiles=nki_runner.prefill_plan.num_tiles
        if nki_runner.prefill_plan is not None
        else 0,
    )
    return device_runner


def prepare_nki_attention_runner(
    scheduler_output: "SchedulerOutput",
    input_batch: "InputBatch",
    num_blocks: int,
    block_size: int,
    large_q_tile_size: int,
    large_kv_tile_size: int,
    max_model_len: int,
    dynamic_loop_unrolling_size: int = 8,
    enable_separate_prefill_decode: bool = True,
    mixed_precision: bool = True,
    skip_active: bool = False,
    include_prompt_in_ctx: bool = True,
    max_num_prefill_tiles: int = None,
    max_num_decode_tiles: int = None,
    padded_query_length: int = None,
    max_num_reqs: Optional[int] = None,
    sliding_window: Optional[int] = None,
) -> NKIFlashPagedAttentionRunner:
    """
    Prepare a NKIFlashPagedAttentionRunner from scheduler output and input batch.

    This function bridges the vLLM scheduler/input batch format with the NKI attention
    runner format, handling the conversion of query lengths,
    context lengths, and block tables.

    Args:
        scheduler_output: Output from the vLLM scheduler containing scheduled tokens
        input_batch: Input batch containing request states and block tables
        num_blocks: Total number of blocks in KV cache
        block_size: Size of each KV cache block (e.g., 32, 64)
        large_kv_tile_size: Tile size for KV dimension (e.g., 1024, 2048, 4096)
        max_model_len: Maximum sequence length the model supports
        dynamic_loop_unrolling_size: Loop unrolling factor for compilation (default: 8)
        enable_separate_prefill_decode: Whether to separate prefill and decode phases
        mixed_precision: Whether to use mixed precision (bf16/fp32)
        skip_active: Whether to skip active token computation
        include_prompt_in_ctx: Whether to include prompt tokens in context
        max_num_prefill_tiles: Maximum number of prefill tiles (for padding)
        max_num_decode_tiles: Maximum number of decode tiles (for padding)
        padded_query_length: Padded query length (for compilation)
        max_num_reqs: Maximum number of requests to pad to (for avoiding recompilation)

    Returns:
        NKIFlashPagedAttentionRunner: Initialized runner with tile plans prepared
    """
    # Extract components from scheduler_output and input_batch
    num_reqs = input_batch.num_reqs
    req_ids = input_batch.req_ids
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    num_computed_tokens_cpu = input_batch.num_computed_tokens_cpu
    block_table = input_batch.block_table[0].get_cpu_tensor()

    # Call the internal function with extracted components
    return _prepare_nki_attention_runner(
        req_ids=req_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        block_table=block_table,
        num_reqs=num_reqs,
        num_blocks=num_blocks,
        block_size=block_size,
        large_q_tile_size=large_q_tile_size,
        large_kv_tile_size=large_kv_tile_size,
        max_model_len=max_model_len,
        dynamic_loop_unrolling_size=dynamic_loop_unrolling_size,
        enable_separate_prefill_decode=enable_separate_prefill_decode,
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        include_prompt_in_ctx=include_prompt_in_ctx,
        max_num_prefill_tiles=max_num_prefill_tiles,
        max_num_decode_tiles=max_num_decode_tiles,
        padded_query_length=padded_query_length,
        max_num_reqs=max_num_reqs,
        sliding_window=sliding_window,
    )
