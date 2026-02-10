# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Wrapper for flash attention NKI kernels to integrate with torch custom ops.
"""

import torch

from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_pa_with_schedule_kv_cache import (  # noqa: E501
    flash_paged_attention_blocksparse_kv_cache,
)
from vllm_nkipy.compile import local_compile


@torch.library.custom_op(
    "mylib::flash_paged_attention_blocksparse_kv_cache_custom_op", mutates_args=()
)
def flash_paged_attention_blocksparse_kv_cache_custom_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
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
    Custom op for flash_paged_attention_blocksparse_kv_cache.

    Args:
        query: Query tensor (1, n_heads, seq_q, d)
        key: Key tensor (1, n_kv_heads, d, seq_k) or None
        value: Value tensor (1, n_kv_heads, seq_v, d) or None
        kv_cache: KV cache (2, max_num_blocks, n_kv_heads, block_size, d)
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
    n_kv_head = kv_cache.shape[2]
    kv_cache.shape[-1]
    output = flash_paged_attention_blocksparse_kv_cache[1, n_kv_head](
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
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


@flash_paged_attention_blocksparse_kv_cache_custom_op.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
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


@local_compile(
    backend="nkipy",
    device="nkipy",
    force=True,
    name="flash_paged_attention_blocksparse_kv_cache_custom_op_compiled",
    fullgraph=True,
    dynamic=False,
)
def flash_paged_attention_blocksparse_kv_cache_custom_op_compiled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
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
    return flash_paged_attention_blocksparse_kv_cache_custom_op(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
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


def flash_paged_attention_blocksparse_kv_cache_custom_op_nkifunc(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
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
        n_kv_head = kv_cache.shape[2]
    assert kv_cache.shape[2] == n_kv_head
    if head_size is None:
        head_size = kv_cache.shape[-1]
    return flash_paged_attention_blocksparse_kv_cache_custom_op_compiled(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
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
