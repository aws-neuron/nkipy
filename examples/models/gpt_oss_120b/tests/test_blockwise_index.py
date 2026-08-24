import time
import numpy as np
import pytest
from kernels.blockwise_index import (
    build_blockwise_index_cpp,
    add_blockwise_index_to_path,
    get_n_blocks,
    BLOCK_SIZE,
    ControlType,
)

@pytest.mark.parametrize(
    "sequence_length, num_experts, top_k, ep",
    [
        (3, 4, 2, 2),  # Simple test case
        (1024, 128, 4, 16),
        (16 * 1024, 128, 4, 16),
        (128 * 1024, 128, 4, 16),
    ],
)
def test_blockwise_index(
    sequence_length,
    num_experts,
    top_k,
    ep,
):
    top_k_indices = np.zeros((sequence_length, top_k), dtype=np.int8)
    for i in range(sequence_length):
        top_k_indices[i] = np.random.choice(num_experts, size=top_k, replace=False)
    n_experts_per_ep = num_experts // ep
    # only focus on ep 0
    top_k_indices[top_k_indices >= n_experts_per_ep] = ControlType.SKIP_DMA.value
    n_blocks, n_static_blocks = get_n_blocks(sequence_length, top_k, num_experts)

    # build_blockwise_index_cpp(
    #     n_experts=n_experts_per_ep,
    #     top_k=top_k,
    #     n_blocks=n_blocks,
    #     n_static_blocks=n_static_blocks,
    # )
    # add_blockwise_index_to_path()
    # from blockwise_index_cpp import get_blockwise_expert_and_token_mapping


    start = time.perf_counter()
    # _, block_to_expert, token_position_to_id = get_blockwise_expert_and_token_mapping(
    #     top_k_indices=top_k_indices
    # )
    
    from kernels.blockwise_index import get_blockwise_expert_and_token_mapping
    _, block_to_expert, token_position_to_id = get_blockwise_expert_and_token_mapping(
        top_k_indices=top_k_indices,
        num_blocks=n_blocks,
        block_size=BLOCK_SIZE,
        num_experts=n_experts_per_ep,
        num_static_blocks=n_static_blocks,
    )
    end = time.perf_counter()
    print(
        f"{sequence_length=}: {1e3 * (end - start):.2f} ms"
    )
    top_k_mask = np.zeros((sequence_length, n_experts_per_ep), dtype=np.uint32)
    current_expert = None
    for block_id in range(n_blocks):
        if block_to_expert[block_id] == ControlType.SKIP_BLOCK.value:
            assert (token_position_to_id[block_id, :] == ControlType.SKIP_DMA.value).all()
            continue  # Skip this block entirely
        elif block_to_expert[block_id] == ControlType.SKIP_DMA.value:
            # Keep using the current_expert from previous block
            assert current_expert is not None
        else:
            assert (
                block_id != n_static_blocks
                or current_expert != block_to_expert[block_id]
            ), "should skip dma"
            # Update current_expert for this block
            current_expert = block_to_expert[block_id]
        
        # Process tokens in this block with current_expert
        for token_id in range(BLOCK_SIZE):
            token_pos = token_position_to_id[block_id, token_id]
            if token_pos == ControlType.SKIP_DMA.value or token_pos == ControlType.SKIP_BLOCK.value:
                continue
            assert top_k_mask[token_pos, current_expert] == 0
            top_k_mask[token_pos, current_expert] = 1
    
    # Check that seen matches top_k_indices
    baseline_top_k_mask = np.zeros((sequence_length, n_experts_per_ep), dtype=np.uint32)
    for i in range(sequence_length):
        for j in range(top_k):
            if top_k_indices[i, j] == ControlType.SKIP_DMA.value:
                continue
            baseline_top_k_mask[i, top_k_indices[i, j]] += 1
    
    assert np.array_equal(top_k_mask, baseline_top_k_mask)


if __name__ == "__main__":
    pytest.main(["-s", __file__])