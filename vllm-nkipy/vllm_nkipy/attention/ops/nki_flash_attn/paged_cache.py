# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
kernels - Builtin high performance attention kernels
"""
# ruff: noqa

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim

from .utils import (
    PF_transpose_with_PE,
    broadcast_partition_with_PE,
    ceil_div,
    transform_to_vector_dge_layout,
)


def transform_block_tables_for_indirect_load(
    block_tables,
    block_size_tiling_factor,
    num_head,
    head_id,
):
    B_P_SIZE = 128
    num_tiles_per_partition, num_partitions, num_blocks_per_tile = block_tables.shape

    num_loads = ceil_div(num_blocks_per_tile, B_P_SIZE)
    block_tables_transposed = nl.ndarray(
        (par_dim(B_P_SIZE), num_loads, num_partitions * num_tiles_per_partition),
        dtype=nl.int32,
    )

    # prepare iota ahead of time to avoid repeatedly using Gpsimd
    if num_head > 1:
        # helper func may not properly broadcast int32, need testing
        head_id_0 = nisa.iota(head_id, dtype=nl.uint32).reshape((1, 1))
        if num_tiles_per_partition > 1:
            head_id = nl.ndarray(
                (par_dim(num_tiles_per_partition), 1),
                dtype=nl.int32,
            )
            broadcast_partition_with_PE(head_id_0, head_id, out_in_psum=False)
        else:
            head_id = head_id_0
        if num_blocks_per_tile > 1:
            head_id = head_id.broadcast_to(
                (num_tiles_per_partition, num_blocks_per_tile)
            )

    if block_size_tiling_factor > 1:
        broadcast_shape = (
            num_tiles_per_partition,
            num_blocks_per_tile,
            block_size_tiling_factor,
        )
        offset = nisa.iota(
            nl.arange(block_size_tiling_factor)[None, None, :], dtype=nl.int32
        )
        if num_tiles_per_partition > 1 or num_blocks_per_tile > 1:
            offset = offset.broadcast_to(broadcast_shape)

    block_tables_transposed_reshaped = block_tables_transposed.reshape(
        (par_dim(B_P_SIZE), num_loads, num_partitions, num_tiles_per_partition)
    )
    identity_for_transpose_hbm = nl.shared_constant(
        np.identity(n=num_tiles_per_partition, dtype=np.uint8),
        dtype=nl.uint8,
    )
    identity_for_transpose = nl.load(identity_for_transpose_hbm)
    for partition_id in nl.affine_range(num_partitions):
        block_tables_partition = block_tables[:, partition_id]
        if num_head > 1:
            # fuse num_block and num_head dimension
            block_tables_partition = block_tables_partition * num_head
            block_tables_partition = nisa.tensor_tensor(
                block_tables_partition,
                head_id,
                nl.add,
                dtype=nl.int32,
            )

        # tile block size dimension
        if block_size_tiling_factor > 1:
            assert num_blocks_per_tile * block_size_tiling_factor == B_P_SIZE
            block_tables_partition = (
                (block_tables_partition * block_size_tiling_factor)
                .reshape((num_tiles_per_partition, num_blocks_per_tile, 1))
                .broadcast_to(broadcast_shape)
            )
            new_block_tables = block_tables_partition + offset
            new_block_tables = new_block_tables.reshape(
                (num_tiles_per_partition, B_P_SIZE)
            )
        else:
            new_block_tables = block_tables_partition

        # transpose the block table so that it can be used by vector DGE
        transform_to_vector_dge_layout(
            indices_in=new_block_tables,
            indices_out=block_tables_transposed_reshaped[:, :, partition_id, :],
            identity_for_transpose=identity_for_transpose,
        )
    return block_tables_transposed


def load_kv_tile_from_cache(
    kv_cache,
    # key_cache,
    # value_cache,
    block_tables,
    large_k_tile_idx,
    num_blocks_per_large_tile,
    block_size,
    B_D_SIZE,
    kernel_dtype,
    k_load_buffer,
    v_load_buffer,
    identity_for_transpose,
):
    B_P_SIZE = 128
    # load key cache
    assert num_blocks_per_large_tile % B_P_SIZE == 0
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    LARGE_KV_TILE_SIZE = num_blocks_per_large_tile * block_size
    transposed_k_tile = nl.ndarray(
        (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transposed_k_tile_reshape = transposed_k_tile.reshape(
        (par_dim(B_D_SIZE), num_loads, block_size, B_P_SIZE)
    )
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        k_load_buffer[i_p, load_idx, i_f] = nl.load(
            kv_cache[0, block_tables[i_p, load_idx, large_k_tile_idx], i_f],
            mode=oob_mode.skip,
        )

        # Transpose SBUF tensor using PE
        for tb_i in nl.affine_range(block_size):
            if kv_cache.dtype != kernel_dtype:
                k_src = nl.copy(
                    k_load_buffer[:, load_idx, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)],
                    dtype=kernel_dtype,
                )
            else:
                k_src = k_load_buffer[:, load_idx, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
            PF_transpose_with_PE(
                src=k_src,
                out=transposed_k_tile_reshape[
                    :,
                    load_idx,
                    tb_i,
                ],
                identity_for_transpose=identity_for_transpose,
            )
    # load value cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        v_load_buffer[i_p, i_f + load_idx * block_size * B_D_SIZE] = nl.load(
            kv_cache[1, block_tables[i_p, load_idx, large_k_tile_idx], i_f],
            mode=oob_mode.skip,
        )
    if kernel_dtype != v_load_buffer.dtype:
        v_tile = nl.ndarray(
            v_load_buffer.shape,
            dtype=kernel_dtype,
        )
        v_tile[...] = nl.copy(v_load_buffer, dtype=kernel_dtype)
    else:
        v_tile = v_load_buffer
    return transposed_k_tile, v_tile
