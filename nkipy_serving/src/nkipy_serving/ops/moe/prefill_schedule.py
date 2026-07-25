"""CPU scheduling for prefill blockwise MoE dispatch.

Given top-k expert indices from the router, build the `token_position_to_id`
and `block_to_expert` tensors that the blockwise NKI kernel consumes.

Used by the MoE eager executors; production executors inline the same
logic in their `forward()` (qwen3_moe/executor.py and gpt_oss/executor.py)
and can be refactored to call this helper as a follow-up.
"""

from __future__ import annotations

import numpy as np

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


def build_prefill_moe_schedule(
    topk_idx: np.ndarray,
    *,
    token_bucket: int,
    real_total_tokens: int,
    experts_per_token: int,
    local_num_experts: int,
    ep_degree: int,
    ep_rank: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build blockwise MoE schedule tensors for one prefill forward.

    Args:
        topk_idx: [token_bucket, top_k] integer array of global expert ids.
        token_bucket: padded token count (kernel input shape).
        real_total_tokens: non-padded token count; rows >= this get SKIP_DMA.
        experts_per_token: router top-k.
        local_num_experts: experts owned by this EP rank.
        ep_degree, ep_rank: EP parallelism (for global->local id remap).

    Returns:
        (token_position_to_id, block_to_expert, num_blocks, num_static_blocks)
    """
    topk = topk_idx.copy()
    if real_total_tokens < topk.shape[0]:
        topk[real_total_tokens:] = int(MoEControlType.SKIP_DMA.value)

    if int(ep_degree) > 1:
        e0 = int(ep_rank) * int(local_num_experts)
        e1 = e0 + int(local_num_experts)
        mask = (topk >= e0) & (topk < e1)
        topk = np.where(mask, topk - e0, int(MoEControlType.SKIP_DMA.value))

    num_blocks, num_static_blocks = moe_get_n_blocks(
        int(token_bucket),
        int(experts_per_token),
        int(local_num_experts),
    )
    _num_real, block_to_expert, token_pos_to_id = (
        get_blockwise_expert_and_token_mapping(
            top_k_indices=topk,
            num_blocks=num_blocks,
            block_size=MOE_BLOCK_SIZE,
            num_experts=int(local_num_experts),
            num_static_blocks=num_static_blocks,
        )
    )
    return token_pos_to_id, block_to_expert, int(num_blocks), int(num_static_blocks)
