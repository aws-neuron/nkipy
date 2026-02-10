# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Wrapper for flash attention NKI kernels to integrate with torch custom ops.
"""

import torch

from torch_to_nkipy.utils.nki import NKIOpRegistry
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_pa_with_schedule import (  # noqa: E501
    flash_paged_attention_blocksparse,
)
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_paged_attn_varlen import (  # noqa: E501
    flash_paged_attention_varlen,
)

# ============================================================================
# Wrapper for flash_paged_attention_blocksparse
# ============================================================================

# @NKIOpRegistry.register(
#     "mylib::flash_paged_attention_blocksparse_custom_op",
#     skip_middle_end=True
# )
# def flash_paged_attention_blocksparse_wrapper(
#     query,
#     key,
#     value,
#     key_cache,
#     value_cache,
#     tile_q_indices,
#     tile_block_tables,
#     tile_masks,
#     active_mask,
#     num_dynamic_loop_steps,
#     last_tile_indices,
#     q_update_pred=None,
#     softmax_scale=None,
#     dynamic_loop_unroll_factor: int = 1,
#     mixed_precision: bool = True,
#     skip_active: bool = False,
#     decode_mode: bool = False,
# ):
#     """
#     Wrapper function for flash_paged_attention_blocksparse NKI kernel.

#     This wrapper handles the conversion between torch tensors and the NKI kernel,
#     and registers the operation for use with torch custom ops.
#     """
#     return flash_paged_attention_blocksparse(
#         query=query,
#         key=key,
#         value=value,
#         key_cache=key_cache,
#         value_cache=value_cache,
#         tile_q_indices=tile_q_indices,
#         tile_block_tables=tile_block_tables,
#         tile_ma.sks=tile_masks,
#         active_mask=active_mask,
#         num_dynamic_loop_steps=num_dynamic_loop_steps,
#         last_tile_indices=last_tile_indices,
#         q_update_pred=q_update_pred,
#         dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
#         softmax_scale=softmax_scale,
#         mixed_precision=mixed_precision,
#         skip_active=skip_active,
#         decode_mode=decode_mode,
#     )


@torch.library.custom_op(
    "mylib::flash_paged_attention_blocksparse_custom_op", mutates_args=()
)
def flash_paged_attention_blocksparse_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tile_q_indices: torch.Tensor,
    tile_block_tables: torch.Tensor,
    tile_masks: torch.Tensor,
    active_mask: torch.Tensor,
    num_dynamic_loop_steps: torch.Tensor,
    last_tile_indices: torch.Tensor,
    q_update_pred: torch.Tensor = None,
    sinks: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    decode_mode: bool = False,
) -> torch.Tensor:
    """
    Custom op for flash_paged_attention_blocksparse.

    Args:
        query: Query tensor (1, n_heads, seq_q, d)
        key: Key tensor (1, n_kv_heads, d, seq_k) or None
        value: Value tensor (1, n_kv_heads, seq_v, d) or None
        key_cache: KV cache for keys (max_num_blocks, n_kv_heads, block_size, d)
        value_cache: KV cache for values (max_num_blocks, n_kv_heads, block_size, d)
        tile_q_indices: Query indices for tiles (num_large_tiles, large_tile_size_q)
        tile_block_tables: Block tables for tiles
            (num_large_tiles, num_block_per_large_tile)
        tile_masks: Attention masks for tiles
        active_mask: Active token mask (seq_q, seq_q)
        num_dynamic_loop_steps: Number of dynamic loop steps (1, 1)
        last_tile_indices: Last tile indices for each query position
        q_update_pred: Query update predicates (optional)
        dynamic_loop_unroll_factor: Loop unrolling factor
        softmax_scale: Softmax scaling factor (default: 1/sqrt(d))
        mixed_precision: Whether to use mixed precision
        skip_active: Whether to skip active token computation
        decode_mode: Whether in decode mode (True) or prefill mode (False)

    Returns:
        Output tensor (1, n_heads, seq_q, d)
    """
    n_kv_head = key_cache.shape[1]
    key_cache.shape[-1]
    output = flash_paged_attention_blocksparse[1, n_kv_head](
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        active_mask=active_mask,
        num_dynamic_loop_steps=num_dynamic_loop_steps,
        last_tile_indices=last_tile_indices,
        q_update_pred=q_update_pred,
        sinks=sinks,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=softmax_scale,
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        decode_mode=decode_mode,
    )
    return output


@flash_paged_attention_blocksparse_custom_op.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tile_q_indices: torch.Tensor,
    tile_block_tables: torch.Tensor,
    tile_masks: torch.Tensor,
    active_mask: torch.Tensor,
    num_dynamic_loop_steps: torch.Tensor,
    last_tile_indices: torch.Tensor,
    q_update_pred: torch.Tensor = None,
    sinks: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    decode_mode: bool = False,
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    return torch.empty_like(query)


from vllm_nkipy.compile import local_compile  # noqa: E402


@local_compile(
    backend="nkipy",
    device="nkipy",
    force=True,
    name="flash_paged_attention_blocksparse_custom_op_compiled",
    fullgraph=True,
    dynamic=False,
)
def flash_paged_attention_blocksparse_custom_op_compiled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tile_q_indices: torch.Tensor,
    tile_block_tables: torch.Tensor,
    tile_masks: torch.Tensor,
    active_mask: torch.Tensor,
    num_dynamic_loop_steps: torch.Tensor,
    last_tile_indices: torch.Tensor,
    q_update_pred: torch.Tensor = None,
    sinks: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    decode_mode: bool = False,
) -> torch.Tensor:
    return flash_paged_attention_blocksparse_custom_op(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        active_mask=active_mask,
        num_dynamic_loop_steps=num_dynamic_loop_steps,
        last_tile_indices=last_tile_indices,
        q_update_pred=q_update_pred,
        sinks=sinks,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=softmax_scale,
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        decode_mode=decode_mode,
    )


def flash_paged_attention_blocksparse_custom_op_nkifunc(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    tile_q_indices: torch.Tensor,
    tile_block_tables: torch.Tensor,
    tile_masks: torch.Tensor,
    active_mask: torch.Tensor,
    num_dynamic_loop_steps: torch.Tensor,
    last_tile_indices: torch.Tensor,
    q_update_pred: torch.Tensor = None,
    sinks: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    n_kv_head: int = None,
    head_size: int = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    decode_mode: bool = False,
):
    if n_kv_head is None:
        n_kv_head = key_cache.shape[1]
    assert key_cache.shape[1] == n_kv_head
    if head_size is None:
        head_size = key_cache.shape[-1]
    return flash_paged_attention_blocksparse_custom_op_compiled(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        active_mask=active_mask,
        num_dynamic_loop_steps=num_dynamic_loop_steps,
        last_tile_indices=last_tile_indices,
        q_update_pred=q_update_pred,
        sinks=sinks,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=1.0 / (head_size**0.5),
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        decode_mode=decode_mode,
    )


# ============================================================================
# Wrapper for flash_paged_attention_varlen
# ============================================================================


@NKIOpRegistry.register(
    "mylib::flash_paged_attention_varlen_custom_op",
)
def flash_paged_attention_varlen_wrapper(
    query,
    key,
    value,
    key_cache,
    value_cache,
    active_mask,
    prefill_tile_q_indices,
    prefill_tile_block_tables,
    prefill_tile_masks,
    prefill_num_dynamic_loop_steps,
    prefill_last_tile_indices,
    decode_tile_q_indices,
    decode_tile_block_tables,
    decode_tile_masks,
    decode_num_dynamic_loop_steps,
    decode_last_tile_indices,
    prefill_q_update_pred=None,
    decode_q_update_pred=None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    return_accum_buffers: bool = False,
):
    """
    Wrapper function for flash_paged_attention_varlen NKI kernel.

    This wrapper handles the conversion between torch tensors and the NKI kernel,
    and registers the operation for use with torch custom ops.
    """
    return flash_paged_attention_varlen(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        active_mask=active_mask,
        prefill_tile_q_indices=prefill_tile_q_indices,
        prefill_tile_block_tables=prefill_tile_block_tables,
        prefill_tile_masks=prefill_tile_masks,
        prefill_num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        prefill_last_tile_indices=prefill_last_tile_indices,
        decode_tile_q_indices=decode_tile_q_indices,
        decode_tile_block_tables=decode_tile_block_tables,
        decode_tile_masks=decode_tile_masks,
        decode_num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
        decode_last_tile_indices=decode_last_tile_indices,
        prefill_q_update_pred=prefill_q_update_pred,
        decode_q_update_pred=decode_q_update_pred,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=softmax_scale,
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        return_accum_buffers=return_accum_buffers,
    )


@torch.library.custom_op(
    "mylib::flash_paged_attention_varlen_custom_op", mutates_args=()
)
def flash_paged_attention_varlen_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    active_mask: torch.Tensor,
    prefill_tile_q_indices: torch.Tensor,
    prefill_tile_block_tables: torch.Tensor,
    prefill_tile_masks: torch.Tensor,
    prefill_num_dynamic_loop_steps: torch.Tensor,
    prefill_last_tile_indices: torch.Tensor,
    decode_tile_q_indices: torch.Tensor,
    decode_tile_block_tables: torch.Tensor,
    decode_tile_masks: torch.Tensor,
    decode_num_dynamic_loop_steps: torch.Tensor,
    decode_last_tile_indices: torch.Tensor,
    prefill_q_update_pred: torch.Tensor = None,
    decode_q_update_pred: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    return_accum_buffers: bool = False,
) -> torch.Tensor:
    """
    Custom op for flash_paged_attention_varlen.

    This kernel handles both prefill and decode phases in a single call.

    Args:
        query: Query tensor (1, n_heads, seq_q, d)
        key: Key tensor (1, n_kv_heads, d, seq_k) or None
        value: Value tensor (1, n_kv_heads, seq_v, d) or None
        key_cache: KV cache for keys (max_num_blocks, n_kv_heads, block_size, d)
        value_cache: KV cache for values (max_num_blocks, n_kv_heads, block_size, d)
        active_mask: Active token mask (seq_q, seq_q)
        prefill_tile_q_indices: Prefill query indices
            (max_num_prefill_tiles, large_tile_size_q)
        prefill_tile_block_tables: Prefill block tables
            (max_num_prefill_tiles,
             num_block_per_large_tile)
        prefill_tile_masks: Prefill attention masks
        prefill_num_dynamic_loop_steps: Prefill loop steps
            (1, 1)
        prefill_last_tile_indices: Prefill last tile indices
        decode_tile_q_indices: Decode query indices
            (max_num_decode_tiles, 1)
        decode_tile_block_tables: Decode block tables
            (max_num_decode_tiles,
             num_block_per_large_tile)
        decode_tile_masks: Decode attention masks
        decode_num_dynamic_loop_steps: Decode loop steps
            (1, 1)
        decode_last_tile_indices: Decode last tile indices
            (max_batch_size, 1)
        prefill_q_update_pred: Prefill query update
            predicates (optional)
        decode_q_update_pred: Decode query update
            predicates (optional)
        dynamic_loop_unroll_factor: Loop unrolling factor
        softmax_scale: Softmax scaling factor
            (default: 1/sqrt(d))
        mixed_precision: Whether to use mixed precision
        skip_active: Whether to skip active token
            computation
        return_accum_buffers: Whether to return
            accumulation buffers instead of final output

    Returns:
        Output tensor (1, n_heads, seq_q, d) or tuple of
        (o_buffer, l_buffer, m_buffer) if
        return_accum_buffers=True
    """
    n_kv_head = key_cache.shape[1]
    output = flash_paged_attention_varlen_wrapper[1, n_kv_head](
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        active_mask=active_mask,
        prefill_tile_q_indices=prefill_tile_q_indices,
        prefill_tile_block_tables=prefill_tile_block_tables,
        prefill_tile_masks=prefill_tile_masks,
        prefill_num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        prefill_last_tile_indices=prefill_last_tile_indices,
        decode_tile_q_indices=decode_tile_q_indices,
        decode_tile_block_tables=decode_tile_block_tables,
        decode_tile_masks=decode_tile_masks,
        decode_num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
        decode_last_tile_indices=decode_last_tile_indices,
        prefill_q_update_pred=prefill_q_update_pred,
        decode_q_update_pred=decode_q_update_pred,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=softmax_scale,
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        return_accum_buffers=return_accum_buffers,
    )
    return output


@flash_paged_attention_varlen_custom_op.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    active_mask: torch.Tensor,
    prefill_tile_q_indices: torch.Tensor,
    prefill_tile_block_tables: torch.Tensor,
    prefill_tile_masks: torch.Tensor,
    prefill_num_dynamic_loop_steps: torch.Tensor,
    prefill_last_tile_indices: torch.Tensor,
    decode_tile_q_indices: torch.Tensor,
    decode_tile_block_tables: torch.Tensor,
    decode_tile_masks: torch.Tensor,
    decode_num_dynamic_loop_steps: torch.Tensor,
    decode_last_tile_indices: torch.Tensor,
    prefill_q_update_pred: torch.Tensor = None,
    decode_q_update_pred: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    return_accum_buffers: bool = False,
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    if return_accum_buffers:
        # Return tuple of (o_buffer, l_buffer, m_buffer)
        b, h, seq_q, d = query.shape
        o_buffer = torch.empty((b, seq_q, h, d), dtype=query.dtype, device=query.device)
        l_buffer = torch.empty((b, seq_q, h, 1), dtype=query.dtype, device=query.device)
        m_buffer = torch.empty((b, seq_q, h, 1), dtype=query.dtype, device=query.device)
        return (o_buffer, l_buffer, m_buffer)
    else:
        return torch.empty_like(query)


def flash_paged_attention_varlen_custom_op_nkifunc(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    active_mask: torch.Tensor,
    prefill_tile_q_indices: torch.Tensor,
    prefill_tile_block_tables: torch.Tensor,
    prefill_tile_masks: torch.Tensor,
    prefill_num_dynamic_loop_steps: torch.Tensor,
    prefill_last_tile_indices: torch.Tensor,
    decode_tile_q_indices: torch.Tensor,
    decode_tile_block_tables: torch.Tensor,
    decode_tile_masks: torch.Tensor,
    decode_num_dynamic_loop_steps: torch.Tensor,
    decode_last_tile_indices: torch.Tensor,
    prefill_q_update_pred: torch.Tensor = None,
    decode_q_update_pred: torch.Tensor = None,
    dynamic_loop_unroll_factor: int = 1,
    softmax_scale: float = None,
    mixed_precision: bool = True,
    skip_active: bool = False,
    return_accum_buffers: bool = False,
) -> torch.Tensor:
    return flash_paged_attention_varlen_custom_op(
        query,
        key,
        value,
        key_cache,
        value_cache,
        active_mask,
        prefill_tile_q_indices,
        prefill_tile_block_tables,
        prefill_tile_masks,
        prefill_num_dynamic_loop_steps,
        prefill_last_tile_indices,
        decode_tile_q_indices,
        decode_tile_block_tables,
        decode_tile_masks,
        decode_num_dynamic_loop_steps,
        decode_last_tile_indices,
        prefill_q_update_pred,
        decode_q_update_pred,
        dynamic_loop_unroll_factor,
        softmax_scale,
        mixed_precision,
        skip_active,
        return_accum_buffers,
    )
