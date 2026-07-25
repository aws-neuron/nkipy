from __future__ import annotations

import numpy as np

from nkipy_serving.ops.moe import blockwise_index


def test_blockwise_index_compiled_matches_python() -> None:
    assert blockwise_index.using_compiled_impl(), (
        blockwise_index._COMPILED_IMPL_LOAD_ERROR
    )

    topk = np.array(
        [
            [0, 1, 3, -1],
            [1, 1, 2, -1],
            [2, 3, -1, -1],
            [3, 0, 1, 2],
            [-1, -1, -1, -1],
            [0, 0, 0, 1],
        ],
        dtype=np.int32,
    )
    num_blocks = 8
    block_size = 4
    num_experts = 4
    num_static_blocks = 8

    ref = blockwise_index._python_get_blockwise_expert_and_token_mapping(
        top_k_indices=topk,
        num_blocks=num_blocks,
        block_size=block_size,
        num_experts=num_experts,
        num_static_blocks=num_static_blocks,
    )
    got = blockwise_index.get_blockwise_expert_and_token_mapping(
        top_k_indices=topk,
        num_blocks=num_blocks,
        block_size=block_size,
        num_experts=num_experts,
        num_static_blocks=num_static_blocks,
    )

    assert int(got[0]) == int(ref[0])
    assert np.array_equal(got[1], ref[1])
    assert np.array_equal(got[2], ref[2])


def test_blockwise_index_python_reference_handles_empty_tokens() -> None:
    topk = np.full((5, 4), blockwise_index.ControlType.SKIP_DMA.value, dtype=np.int32)
    num_real, block_to_expert, token_pos = (
        blockwise_index._python_get_blockwise_expert_and_token_mapping(
            top_k_indices=topk,
            num_blocks=6,
            block_size=4,
            num_experts=4,
            num_static_blocks=6,
        )
    )

    assert num_real == 0
    assert block_to_expert.shape == (6,)
    assert token_pos.shape == (6, 4)
    assert np.all(block_to_expert == blockwise_index.ControlType.SKIP_BLOCK.value)
    assert np.all(token_pos == blockwise_index.ControlType.SKIP_DMA.value)
