# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Blockwise index computation for MoE.

This module provides functions for computing blockwise expert and token mappings
used in the MoE blockwise processing algorithm.
"""

import math
import os

import torch
from torch.utils.cpp_extension import load

from vllm_nkipy.ops.moe.constants import BLOCK_SIZE

_src_path = os.path.join(os.path.dirname(__file__), "blockwise_index.cpp")

blockwise_ext = load(
    name="blockwise_index_ext",
    sources=[_src_path],
    verbose=True,
)


def get_n_blocks(T, TOPK, E):
    num_static_blocks = math.ceil((T * TOPK - (E - 1)) / BLOCK_SIZE) + E - 1

    # num_static_blocks = math.ceil(T * TOPK / B)
    # # make sure num_static_blocks is even for lnc2
    # num_static_blocks = 2 * math.ceil(num_static_blocks / 2)
    # num_dynamic_blocks = (
    #     math.ceil((T * TOPK - (E - 1)) / B) + E - 1
    # ) - num_static_blocks
    # num_dynamic_blocks = math.ceil(num_dynamic_blocks / 2) * 2
    # #TODO: need to run for lnc2
    # #TODO: round to multiple of n_block_per_iter
    # # for dynamic_blockwise padding. Can we do better?
    # num_dynamic_blocks = Config.num_blocks_per_launch * math.ceil(
    #     num_dynamic_blocks / Config.num_blocks_per_launch
    # )
    # return num_static_blocks + num_dynamic_blocks, num_static_blocks

    return num_static_blocks, num_static_blocks


def get_blockwise_expert_and_token_mapping(
    top_k_indices: torch.Tensor,
    num_blocks: int,
    block_size: int,
    num_experts: int,
    num_static_blocks: int,
):
    """
    top_k_indices: [T, TOP_K], int8 or int32, CPU for now
    returns: (num_real_blocks: int,
              block_to_expert: torch.int8[B],
              token_pos: torch.int32[B, BS])
    """

    num_real_blocks, block_to_expert, token_position_to_id = (
        blockwise_ext.get_blockwise_expert_and_token_mapping(
            top_k_indices,
            int(num_blocks),
            int(block_size),
            int(num_experts),
            int(num_static_blocks),
        )
    )
    return num_real_blocks, block_to_expert, token_position_to_id
