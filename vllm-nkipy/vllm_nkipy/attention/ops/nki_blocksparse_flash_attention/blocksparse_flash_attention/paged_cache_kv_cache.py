# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
Flash Paged Attention kernels with variable-length sequence inputs.
"""

from .constants import B_P_SIZE
from .utils import (
    is_power_of_2,
)


def prepare_kv_block_dim_tiling_kv_cache(kv_cache, LARGE_KV_TILE_SIZE):
    """
    If number of blocks to load for a tile
    (i.e. LARGE_KV_TILE_SIZE // block_size) is smaller than
    B_P_SIZE(128), tiling on block_size dimension is applied
    so that there are 128 loads to fully utilize Vector DGE.

    This function decides the new tiled_block_size. It also
    reshapes KV cache to a 2-D layout to load KV data in
    block granularity
    """
    _, num_blocks, k_h, block_size, d = kv_cache.shape
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    assert is_power_of_2(num_blocks_per_large_tile), (
        f"{num_blocks_per_large_tile=} is expected of be power of 2"
    )

    # load block tables
    if num_blocks_per_large_tile < B_P_SIZE:
        # we checked num_blocks_per_tile is a power of 2
        assert B_P_SIZE % num_blocks_per_large_tile == 0
        block_size_tiling_factor = B_P_SIZE // num_blocks_per_large_tile
        assert block_size % block_size_tiling_factor == 0
        num_blocks_per_large_tile *= block_size_tiling_factor  # i.e. = B_P_SIZE
    else:
        block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    # flatten KV cache to be 2D for loading into SBUF
    new_cache_shape = (
        2,
        num_blocks * k_h * block_size_tiling_factor,
        tiled_block_size * d,
    )
    kv_cache = kv_cache.reshape(new_cache_shape)
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]
    return key_cache, value_cache, block_size_tiling_factor, tiled_block_size
