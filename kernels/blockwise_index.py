import math
import os
import subprocess
import sys
from enum import Enum

import numpy as np

BLOCK_SIZE = 128
MAX_N_BLOCKS = (1 << 16) -1 # uint16_t
MAX_N_EXPERTS = 128 # int8_t

MODULE_NAME = "blockwise_index_cpp"

class ControlType(Enum):
    SKIP_DMA = -1
    SKIP_BLOCK = -2

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
    # #TODO: round to multiple of n_block_per_iter for dynamic_blockwise padding. Can we do better?
    # num_dynamic_blocks = Config.num_blocks_per_launch * math.ceil(
    #     num_dynamic_blocks / Config.num_blocks_per_launch
    # )
    # return num_static_blocks + num_dynamic_blocks, num_static_blocks

    return num_static_blocks, num_static_blocks


def build_blockwise_index_cpp(n_experts, top_k, n_blocks, n_static_blocks):
    from nkipy.core.compile import _get_build_dir
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup
    
    
    # Clean build directory
    subprocess.run(f"rm -rf {_get_build_dir()}/{MODULE_NAME}", shell=True)
    os.makedirs(f"{_get_build_dir()}/{MODULE_NAME}", exist_ok=True)

    # Define the extension with compiler defines
    ext_modules = [
        Pybind11Extension(
            MODULE_NAME,
            ["kernels/blockwise_index.cpp"],
            cxx_std=14,
            include_dirs=["/usr/local/include"],
            define_macros=[
                ('N_EXPERTS', str(n_experts)),
                ('TOP_K', str(top_k)),
                ('N_BLOCKS', str(n_blocks)),
                ('N_STATIC_BLOCKS', str(n_static_blocks)),
                ('BLOCK_SIZE', str(BLOCK_SIZE)),
                ('SKIP_DMA', str(ControlType.SKIP_DMA.value)),
                ('SKIP_BLOCK', str(ControlType.SKIP_BLOCK.value)),
            ]
        )
    ]
    
    # Build the extension directly
    try:
        setup(
            name=MODULE_NAME,
            ext_modules=ext_modules,
            cmdclass={"build_ext": build_ext},
            script_args=[
                "build_ext",
                f"--build-temp={_get_build_dir()}/{MODULE_NAME}",
                f"--build-lib={_get_build_dir()}/{MODULE_NAME}",
            ],
            zip_safe=False,
            python_requires=">=3.6",
        )
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError("Build failed")

def add_blockwise_index_to_path():
    from nkipy.core.compile import _get_build_dir

    sys.path.append(f"{_get_build_dir()}/{MODULE_NAME}")


def get_blockwise_expert_and_token_mapping(
    top_k_indices,  # top_k out of interest masked as SKIP_DMA
    num_blocks,
    block_size,
    num_experts,
    num_static_blocks,
):
    tokens_per_expert = np.bincount(top_k_indices[top_k_indices != ControlType.SKIP_DMA.value].flatten(), minlength=num_experts)
    blocks_per_expert = np.ceil(
        np.divide(tokens_per_expert, block_size, dtype=np.float32)
    ).astype(np.uint32)
    cumulative_blocks_per_expert = np.cumsum(blocks_per_expert, axis=0, dtype=np.uint32)
    num_real_blocks = cumulative_blocks_per_expert[-1]
    assert num_real_blocks <= num_blocks, (
        f"num_real_blocks: {num_real_blocks}, num_blocks: {num_blocks}"
    )
    block_to_expert = np.full(
        (num_blocks,), ControlType.SKIP_BLOCK.value, dtype=np.int8
    )
    block_to_expert[:num_real_blocks] = np.repeat(
        np.arange(num_experts), blocks_per_expert
    )
    token_position_to_id = np.full(
        (num_blocks, block_size), ControlType.SKIP_DMA.value, dtype=np.int32
    )
    current_block_idx = np.zeros((num_experts,), dtype=np.int32)
    current_block_idx[1:] = cumulative_blocks_per_expert[:-1]
    current_token_id_in_block = np.zeros((num_experts,), dtype=np.int32)

    for t in range(top_k_indices.shape[0]):
        top_k_indices_ = top_k_indices[t]
        for k in range(top_k_indices_.shape[0]):
            expert_id = top_k_indices_[k]
            if expert_id == ControlType.SKIP_DMA.value:
                continue
            token_position_to_id[
                current_block_idx[expert_id], current_token_id_in_block[expert_id]
            ] = t
            current_token_id_in_block[expert_id] += 1
            if current_token_id_in_block[expert_id] == block_size:
                current_block_idx[expert_id] += 1
                current_token_id_in_block[expert_id] = 0

    # Set consecutive blocks with same expert to skip_weight_dma
    for i in range(num_real_blocks - 1, 0, -1):
        # TODO: not skip on first dynamic block
        if block_to_expert[i] == block_to_expert[i - 1] and i != num_static_blocks:
            block_to_expert[i] = ControlType.SKIP_DMA.value

    return (
        num_real_blocks,
        block_to_expert,
        token_position_to_id,
    )

def partition_array(nums):
    """
    Partition the array into two groups with the smallest difference in sum.
    Uses exact dynamic programming approach (optimal but slower).

    Args:
        nums: Input array of numbers (can be list or numpy array)

    Returns:
        A boolean numpy array indicating which elements belong to the first group
    """
    # Convert input to numpy array if it's not already
    nums = np.asarray(nums, dtype=np.uint16)
    n = len(nums)
    if n == 0:
        return np.array([], dtype=bool)

    # Calculate total sum of the array
    total_sum = int(np.sum(nums))

    # Initialize DP array to track achievable sums
    dp = np.zeros(total_sum + 1, dtype=bool)
    dp[0] = True
    last_index = np.full(total_sum + 1, -1, dtype=np.int16)

    # Fill DP array with achievable sums
    for i in range(n):
        num = int(nums[i])
        # Reverse iteration to avoid using updated values
        for j in range(total_sum, num - 1, -1):
            if dp[j - num] and not dp[j]:
                dp[j] = True
                last_index[j] = i

    # Find the sum closest to total_sum / 2
    target = total_sum // 2
    best_sum = 0
    min_diff = total_sum

    # OPTIMIZATION: Only search up to target to reduce iterations
    for i in range(target + 1):
        if dp[i]:
            diff = abs(2 * i - total_sum)
            if diff < min_diff:
                min_diff = diff
                best_sum = i

    # Reconstruct the first group
    included = np.zeros(n, dtype=bool)
    current_sum = best_sum
    while current_sum > 0:
        i = last_index[current_sum]
        included[i] = True
        current_sum -= int(nums[i])

    return included


def partition_array_greedy(nums):
    """
    Partition the array into two groups using greedy approximation.
    Fast but not optimal - assigns each element to the group with smaller current sum.

    Args:
        nums: Input array of numbers (can be list or numpy array)

    Returns:
        A boolean numpy array indicating which elements belong to the first group
    """
    nums = np.asarray(nums, dtype=np.uint16)
    n = len(nums)
    if n == 0:
        return np.array([], dtype=bool)

    # Create index-value pairs for sorting
    indexed_nums = [(nums[i], i) for i in range(n)]

    # Sort by value in descending order (largest first)
    indexed_nums.sort(key=lambda x: x[0], reverse=True)

    result = np.zeros(n, dtype=bool)
    sum0, sum1 = 0, 0

    # Greedy assignment: always assign to the group with smaller sum
    for value, index in indexed_nums:
        if sum0 <= sum1:
            result[index] = True
            sum0 += value
        else:
            sum1 += value

    return result
