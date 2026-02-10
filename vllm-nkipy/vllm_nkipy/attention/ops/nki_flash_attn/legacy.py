# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
# Adapted from vLLM (https://github.com/vllm-project/vllm)
# ruff: noqa

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
import torch
from neuronxcc import nki
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim
from vllm.utils import cdiv


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


def pad_to_next_power_of_2(a):
    assert a > 0
    return 2 ** int(a - 1).bit_length()


def _print(*args, **kwargs):
    print(*args)


@nki.jit
def load_active_block_tables(
    block_tables_hbm, num_tiles, num_blocks_per_tile, B_P_SIZE
):
    """
    Load block tables from HBM into SRAM

    `block_tables_hbm` has shape `(num_tiles * num_blocks_per_tile, )`.
    In case `num_tiles > B_P_SIZE`, we need further tile `num_tile` dimension.
    """

    partition_size = min(B_P_SIZE, num_tiles)
    assert partition_size > 0, f"invalid partition size: {partition_size=}"
    _print(
        f"{block_tables_hbm.shape=}, {num_tiles=}, {num_blocks_per_tile=}, {partition_size=}"
    )

    # reshape as `(num_tiles, num_blocks_per_tile)`
    assert len(block_tables_hbm.shape) == 1
    (num_total_blocks,) = block_tables_hbm.shape
    num_tiles_pad = pad_to_next_power_of_2(num_tiles)
    assert num_blocks_per_tile * num_tiles_pad == num_total_blocks
    block_tables_hbm = block_tables_hbm.reshape((num_tiles_pad, num_blocks_per_tile))

    num_partitions = cdiv(num_tiles, partition_size)
    block_tables_sbuf = nl.zeros(
        (par_dim(partition_size), num_partitions, num_blocks_per_tile),
        dtype=nl.int32,
    )
    for i in nl.affine_range(num_partitions):
        i_p = nl.arange(partition_size)[:, None]
        i_f = nl.arange(num_blocks_per_tile)[None, :]
        block_tables_sbuf[i_p, i, i_f] = nl.load(
            block_tables_hbm[i_p + i * partition_size, i_f],
            dtype=nl.int32,
            mask=(i_p + i * partition_size < num_tiles),
        )
    return block_tables_sbuf


@nki.jit
def transform_block_tables_for_indirect_load(
    block_tables,
    block_size_tiling_factor,
    num_head,
    head_id,
    B_P_SIZE,
):
    """
    This function does two things:
    1. calculate new `block_tables` for a `head_id` after flattening
    `num_block`, `num_head`, and `block_size_tiling_factor` dimensions
    2. transpose the result so that `block_table` for each tile is mapped to
    SBUF Partition dimension for vectorized DMA

    Tiling trick to further improve DMA performance:
    Given KV cache shape `(num_block, num_head, block_size, D)`, when loading M
    blocks of a given `head_id` from HBM, the load `cache[block_tables,
    head_id]` has shape `(M, block_size, D)`. If M < B_P_SIZE = 128, DMA may not
    fully utilize hardware parallelization. The solution is to tile `block_size`
    into `(block_size_tiling_factor, tiled_block_size)` s.t. `M *
    block_size_tiling_factor = B_P_SIZE`. After tiling, KV cache has shape
    `(num_block, num_head, block_size_tiling_factor, tiled_block_size, D)`.

    Note:
    We don't further tile D dimension as small DMA size also hurts performance.
    """
    num_tiles_per_partition, num_partitions, num_blocks_per_tile = block_tables.shape
    assert is_power_of_2(num_blocks_per_tile), (
        f"{num_blocks_per_tile=} is not power of 2"
    )

    num_loads = cdiv(num_blocks_per_tile, B_P_SIZE)
    block_tables_transposed = nl.ndarray(
        (
            par_dim(B_P_SIZE),
            num_loads,
            num_partitions * num_tiles_per_partition,
        ),
        dtype=nl.int32,
    )

    # prepare iota ahead of time to avoid repeatedly using Gpsimd
    if num_head > 1:
        head_id = nisa.iota(head_id, dtype=nl.int32).reshape((1, 1))
        if num_tiles_per_partition > 1:
            head_id = head_id.broadcast_to((1, num_tiles_per_partition))
            head_id = nl.transpose(head_id)
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
        ).broadcast_to(broadcast_shape)

    for partition_id in nl.affine_range(num_partitions):
        block_tables_partition = block_tables[:, partition_id]
        if num_head > 1:
            # fuse num_block and num_head dimension
            block_tables_partition = block_tables_partition * num_head + head_id

        if block_size_tiling_factor > 1:
            # need to apply block size tiling trick
            assert num_blocks_per_tile * block_size_tiling_factor == B_P_SIZE
            block_tables_partition = (
                (block_tables_partition * block_size_tiling_factor)
                .reshape(shape=(num_tiles_per_partition, num_blocks_per_tile, 1))
                .broadcast_to(broadcast_shape)
            )
            new_block_tables = block_tables_partition + offset
            new_block_tables = new_block_tables.reshape(
                (num_tiles_per_partition, B_P_SIZE)
            )
        else:
            new_block_tables = block_tables_partition

        # transpose the block table so that it can be used by vector DGE
        for i in nl.affine_range(num_loads):
            block_tables_transposed[
                :,
                i,
                nl.ds(partition_id * num_tiles_per_partition, num_tiles_per_partition),
            ] = nl.transpose(new_block_tables[:, nl.ds(i * B_P_SIZE, B_P_SIZE)])
    return block_tables_transposed


@nki.jit
def load_kv_tile_from_cache(
    kv_cache,
    block_tables,
    large_k_tile_idx,
    num_blocks_per_large_tile,
    tiled_block_size,
    B_P_SIZE,
    B_D_SIZE,
    kernel_dtype,
    k_load_buffer,
    v_load_buffer,
    v_cache_offset,
):
    """
    Load KV cache and transform Key and Value into layout required by Matmul

    Vectorized DMA Load layout:
    Key and Value: (par_dim(B_P_SIZE), seqlen_kv // B_P_SIZE * B_D_SIZE)

    Layout used by attention matmuls:
    Key: (par_dim(B_D_SIZE), seqlen_kv)
    Value: (seqlen_kv // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE)
           equivalent to (par_dim(B_P_SIZE), seqlen_kv // B_P_SIZE * B_D_SIZE)
    """
    i_p = nl.arange(B_P_SIZE)[:, None]
    i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]

    # load key cache
    assert num_blocks_per_large_tile % B_P_SIZE == 0
    num_loads = cdiv(num_blocks_per_large_tile, B_P_SIZE)
    LARGE_KV_TILE_SIZE = num_blocks_per_large_tile * tiled_block_size
    tranposed_k_tile = nl.ndarray(
        (par_dim(B_D_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )

    for load_idx in nl.affine_range(num_loads):
        tiled_block_idx = block_tables[i_p, load_idx, large_k_tile_idx]
        k_load_buffer[i_p, load_idx, i_f] = nl.load(
            kv_cache[tiled_block_idx, i_f], mode=oob_mode.skip
        )
        if kernel_dtype != kv_cache.dtype:
            k_load_buffer[i_p, load_idx, i_f] = nl.copy(
                k_load_buffer[i_p, load_idx, i_f], dtype=kernel_dtype
            )

        # Transpose SBUF tensor using PE
        for tb_i in nl.affine_range(tiled_block_size):
            tranposed_k_tile[
                :,
                nl.ds(
                    load_idx * B_P_SIZE * tiled_block_size + tb_i * B_P_SIZE,
                    B_P_SIZE,
                ),
            ] = nl.transpose(
                k_load_buffer[:, load_idx, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
            )

    # load value cache
    for load_idx in nl.affine_range(num_loads):
        loaded = nl.load(
            kv_cache[
                v_cache_offset + block_tables[i_p, load_idx, large_k_tile_idx], i_f
            ],
            mode=oob_mode.skip,
        )
        if kernel_dtype != kv_cache.dtype:
            loaded = nl.copy(loaded, dtype=kernel_dtype)
        v_load_buffer[
            :,
            nl.ds(
                load_idx * tiled_block_size * B_D_SIZE,
                tiled_block_size * B_D_SIZE,
            ),
        ] = loaded

    return tranposed_k_tile, v_load_buffer


@nki.jit
def transpose_p_local(p_local_transposed, p_local, LARGE_TILE_SZ, B_F_SIZE=512):
    for i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray(
                (par_dim(128), B_F_SIZE), buffer=nl.sbuf, dtype=p_local.dtype
            )
        else:
            p_local_t_tmp = nl.ndarray(
                (par_dim(128), B_F_SIZE), buffer=nl.psum, dtype=np.float32
            )

        for j in nl.affine_range(B_F_SIZE // 128):
            j_128_slice = nl.ds(j * 128, 128)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * 128, 128)

            if nisa.get_nc_version() == nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
                    p_local[:, i_j_128_slice]
                )
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
                    p_local[:, i_j_128_slice]
                )

        p_local_transposed[:, nl.ds(i * B_F_SIZE, B_F_SIZE)] = nl.copy(
            p_local_t_tmp, dtype=p_local_transposed.dtype
        )


@nki.jit
def _flash_attention_core(
    q_local_tile,
    k,
    v,
    o_buffer,
    l_buffer,
    m_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    use_causal_mask,
    q_tile_idx=None,
    initialize=False,
    B_D_SIZE=128,
    qk_res_buffer=None,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    The q_local_tile has (B_P_SIZE, B_D_SIZE)
    The K and V have shape (B_D_SIZE, LARGE_TILE_SZ), whose free dimension will
    be split into size B_F_SIZE tiles

    The results are stored in the following three buffers
    o_buffer: (B_P_SIZE, d)
    l_buffer: (B_P_SIZE, 1)
    m_buffer: (B_P_SIZE, 1)

    All IO buffers are in SBUF.
    """
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE

    qk_res_buf = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type
    )
    max_local = nl.ndarray(
        (par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type
    )
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        if use_causal_mask:
            # mask are used to only apply computation to the lower half of the
            # matrix, which reduce the arithmetic intensity by up to 50%
            multiplication_required_selection = q_tile_idx * B_P_SIZE >= k_i * B_F_SIZE
        else:
            multiplication_required_selection = True

        if multiplication_required_selection:
            qk_psum = nl.ndarray(
                (par_dim(B_P_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum
            )  # (128, 512)
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(128), 512)
            qk_res_buf[:, k_i_b_f_slice] = nl.where(
                tile_mask[:, k_i_b_f_slice],
                qk_psum[:, nl.ds(0, B_F_SIZE)],
                -9984.0,
                dtype=acc_type,
            )
        else:
            qk_res_buf[:, k_i_b_f_slice] = -9984.0

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max,
            qk_res_buf[:, k_i_b_f_slice],
            axis=(1,),
            dtype=acc_type,
            negate=False,
        )

    if qk_res_buffer is not None:
        qk_res_buffer[:, :] = nl.copy(qk_res_buf[:, :])

    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=o_buffer.dtype)

    if initialize:
        m_buffer[:, 0] = nl.copy(max_)
        m_current = max_
    else:
        m_previous = nl.copy(m_buffer[:, 0])
        m_buffer[:, 0] = nl.maximum(m_previous, max_)  # (128,1)

        m_current = m_buffer[:, 0]
        # Compute scaling factor
        alpha = nisa.activation(
            np.exp,
            m_previous,
            bias=-1 * m_current,
            scale=1.0,
        )
        o_previous_scaled[...] = nl.multiply(o_buffer[:, :], alpha)

    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

    p_partial_sum = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE),
        dtype=acc_type,
    )

    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk - max)
        # Compute partial row - tile sum of exp(qk - max))
        # FIXME : Use activation accumulate to accumulate over k_r_i loop ?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype
    )
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        LARGE_TILE_SZ=LARGE_TILE_SZ,
        B_F_SIZE=B_F_SIZE,
    )

    pv_psum = nl.zeros(
        (par_dim(B_P_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            transpose_x=True,
        )  # (128, 128) (p(Br), d)

    if initialize:
        o_buffer[:, :] = nl.copy(pv_psum[:, :])
        l_buffer[:, 0] = nl.add(nl.log(ps), max_)
    else:
        o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum)

        l_prev = l_buffer[:, 0]
        l_exp = nl.add(
            nl.exp(nl.subtract(l_prev, m_current)),
            ps,
        )
        l_buffer[:, 0] = nl.add(m_current, nl.log(l_exp))


@nki.jit
def load_v_tile(v_hbm_tile, cur_v_tile, large_tile_idx, v_i, LARGE_TILE_SZ):
    B_D_SIZE = v_hbm_tile.shape[-1]
    loaded = nl.load(
        v_hbm_tile[
            nl.ds(large_tile_idx * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE),
            :,
        ]
    )
    if cur_v_tile.dtype != loaded.dtype:
        loaded = nl.copy(loaded, dtype=cur_v_tile.dtype)
    cur_v_tile[:, nl.ds(v_i * B_D_SIZE, B_D_SIZE)] = loaded


@nki.jit(experimental_flags="enable-mutable-parameter")
def build_attention_mask(
    prior_mask: nt.mutable_tensor,
    cu_query_lens,
    cu_ctx_lens,
    seq_lens,
    max_num_queries,
    max_num_keys,
    max_num_seqs,
    block_size,
    # tiled_block_size,
    contexted=False,
    B_F_SIZE=512,
):
    """
    Creates block diagonal causal masks for multiple prompts.

    This mask is dynamic and depends on the input prompt lengths.

    Example:
        query_lens = [2, 3, 1, 0], key_lens = [4, 5, 4, 0], max_num_queries = 8, max_num_keys = 16, max_num_seqs = 4

        prior_mask = [
        #   i0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19
        #   c0,c0,c0,n0,c1,c1,c1,n1,n1,p1,p1,p1,c2,c2,c2,c2,p3,p3,p3,p3 - (c - cached token, n - new token, p - padding)
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 0 (4 tokens, aligned to 1 block)
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 0 (4 tokens, aligned to 1 block)
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 1 (5 tokens, aligned to 2 blocks)
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 1 (5 tokens, aligned to 2 blocks)
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 1 (5 tokens, aligned to 2 blocks)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], # sequence 2 (4 tokens, aligned to 1 block)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        ]
        active_mask = None

    Args:
        prompt_lens: The list of prompt lengths.
        n_positions: The total size of the KV cache to consider. This is
            equal to the current bucket size.

    Returns:
        prior_mask: The attention mask to apply to the KV cache.
        active_mask: The attention mask to apply to the active tokens (None).

    For each sequence, we start from building a triangle mask with an offset.
    As shown in the following figure, with

      ri = q_cumsum[seq_id]
      ci = k_cumsum[seq_id] (k_cumsum need to be block-aligned)
      nr = query_lens[seq_id]
      nc = key_lens[seq_id]

    we can trim the triangle mask from different directions, in order to get the desired shape.

                 (0, ci+nc-ri-nr)
    +--------------------+----------------------------------+
    |                      \                                |
    |                        \   (ri, ci+nc-nr)             |
    |                          \   |                        |
    |   (ri, ci)                 \       (ri, ci+nc)        |
    |       +----------------------+ - - - +                |
    |       | xxxxxxxxxxxxxxxxxxxxxx \     |                |
    |       | xxxxxxxxxxxxxxxxxxxxxxxx \   |                |
    |       | xxxxxxxxxxxxxxxxxxxxxxxxxx \ |                |
    |       +------------------------------+                |
    |   (ri+nr, ci)                      (ri+nr, ci+nc)     |
    |                                                       |
    +-------------------------------------------------------+
    """

    _print(f"{cu_query_lens.shape=}")
    i_p = nl.arange(1)[:, None]
    cu_f = nl.arange(max_num_seqs + 1)[None, :]
    i_f = nl.arange(max_num_seqs)[None, :]
    cu_query_lens_sbuf = nl.load(cu_query_lens[cu_f])
    cu_ctx_lens_sbuf = nl.load(cu_ctx_lens[cu_f])
    seq_lens_sbuf = seq_lens
    _print(
        f"{cu_query_lens_sbuf.shape=}, {max_num_queries=}, {max_num_keys=}, {max_num_seqs=}, {block_size=}"
    )
    query_lens = cu_query_lens_sbuf[i_p, i_f + 1] - cu_query_lens_sbuf[i_p, i_f]
    ctx_lens = seq_lens_sbuf

    assert (int(cu_query_lens_sbuf.shape[1]) == max_num_seqs + 1) and (
        int(cu_query_lens_sbuf.ndim) == 2
    ), f"invalid {cu_query_lens=}"
    assert (int(cu_ctx_lens_sbuf.shape[1]) == max_num_seqs + 1) and (
        int(cu_ctx_lens_sbuf.ndim) == 2
    ), f"invalid {cu_ctx_lens=}"
    assert (int(query_lens.shape[1]) == max_num_seqs) and (int(query_lens.ndim) == 2), (
        f"invalid {query_lens=}"
    )
    assert (int(seq_lens_sbuf.shape[1]) == max_num_seqs) and (
        int(seq_lens_sbuf.ndim) == 2
    ), f"invalid {seq_lens=}"

    # Create mask buffers
    # prior_mask = nl.ndarray((1, max_num_queries, max_num_keys), dtype=nl.bool_, buffer=nl.shared_hbm)

    q_tile_size, k_tile_size = max_num_queries, B_F_SIZE  # max_num_queries, 8
    n_q_tile = (max_num_queries + q_tile_size - 1) // q_tile_size
    n_k_tile = (max_num_keys + k_tile_size - 1) // k_tile_size
    _print(f"{n_q_tile=}, {n_k_tile=}")
    i_p = nl.arange(q_tile_size)[:, None]  # this works for dynamic access
    i_f = nl.arange(k_tile_size)[None, :]

    def _br(x, out):
        # Broadcast weight along first axis to match tensor shape
        src_p, dst_p = x.shape[0], out.shape[0]
        n_iters = cdiv(dst_p // src_p, 32)
        br_size = min(32, dst_p // src_p)
        for j in range(n_iters):
            out[j * br_size : (j + 1) * br_size, :] = nl.broadcast_to(
                x, shape=(br_size, k_tile_size)
            )

    def _br2(x):
        return nl.broadcast_to(x, shape=(q_tile_size, k_tile_size))

    # a = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)
    b = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)
    # expr_a = nl.arange(0, k_tile_size)[None, :]
    expr_a = nl.arange(0, q_tile_size)[:, None]
    a_tile = nisa.iota(expr_a, dtype=nl.int32)
    a = _br2(a_tile)
    # expr_b = nl.arange(0, q_tile_size)[:, None]
    expr_b = nl.arange(0, k_tile_size)[None, :]
    b_tile = nisa.iota(expr_b, dtype=nl.int32)
    _br(b_tile, b)
    _print(f"{a.shape=}, {b.shape=}, {a_tile.shape=}, {b_tile.shape=}")

    # iota_hbm = nl.ndarray((2, q_tile_size, k_tile_size), dtype=nl.int32, buffer=nl.shared_hbm)
    # new_mask_hbm = nl.ndarray((max_num_seqs, n_k_tile, q_tile_size, k_tile_size), dtype=nl.int32, buffer=nl.shared_hbm)
    # loop_var = nl.ndarray((max_num_seqs, n_k_tile, 10), dtype=nl.int32, buffer=nl.shared_hbm)

    # initialize with zero
    for k_tile_idx in nl.sequential_range(n_k_tile):
        q_tile_idx = 0
        q_slice = nl.ds(q_tile_idx * q_tile_size, q_tile_size)
        k_slice = nl.ds(k_tile_idx * k_tile_size, k_tile_size)
        q_bound = (q_tile_idx * q_tile_size + i_p) < max_num_queries
        k_bound = (k_tile_idx * k_tile_size + i_f) < max_num_keys
        nl.store(prior_mask[q_slice, k_slice], value=0, mask=q_bound & k_bound)

    ri = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)
    ci = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)
    nr = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)
    nc = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)
    tile_offset = nl.ndarray((q_tile_size, k_tile_size), buffer=nl.sbuf, dtype=nl.int32)

    # Process each sequence
    for seq_idx in nl.sequential_range(max_num_seqs):
        # ri = _br(cu_query_lens_sbuf[0, seq_idx])
        # ci = _br(cu_ctx_lens_sbuf[0, seq_idx])
        # nr = _br(query_lens[0, seq_idx])
        # nc = _br(ctx_lens[0, seq_idx])
        _br(cu_query_lens_sbuf[0, seq_idx], ri)
        _br(cu_ctx_lens_sbuf[0, seq_idx], ci)
        _br(query_lens[0, seq_idx], nr)
        _br(ctx_lens[0, seq_idx], nc)
        _print(f"{ci.shape=}, {nc.shape=}, {ri.shape=}, {nr.shape=}")

        for k_tile_idx in nl.sequential_range(n_k_tile):
            q_tile_idx = 0
            _br(b_tile[0, k_tile_idx] * k_tile_size, tile_offset)

            if contexted:
                a_offset = (ci + nc) - (ri + nr)
                # new_mask = nl.greater_equal(a + a_offset - tile_offset, b)
                new_mask = nl.greater_equal(nl.subtract(a + a_offset, tile_offset), b)
            else:
                new_mask = nl.greater_equal(nl.subtract(ci + nc - 1, tile_offset), b)

            # nl.store(new_mask_hbm[seq_idx, k_tile_idx, :, :], new_mask)

            left_mask = nl.greater_equal(b, ci - tile_offset)
            top_mask = nl.greater_equal(a, ri)
            bottom_mask = nl.less(a, ri + nr)
            new_mask = new_mask & left_mask & top_mask & bottom_mask

            q_slice = nl.ds(q_tile_idx * q_tile_size, q_tile_size)
            k_slice = nl.ds(k_tile_idx * k_tile_size, k_tile_size)
            q_bound = (q_tile_idx * q_tile_size + i_p) < max_num_queries
            k_bound = (k_tile_idx * k_tile_size + i_f) < max_num_keys

            new_mask = (
                nl.load(prior_mask[q_slice, k_slice], mask=q_bound & k_bound)
                | new_mask[i_p, i_f]
            )
            nl.store(
                prior_mask[q_slice, k_slice], new_mask[i_p, i_f], mask=q_bound & k_bound
            )

        # nl.store(loop_var[seq_idx, k_tile_idx, 0], ri[0, 0])
        # nl.store(loop_var[seq_idx, k_tile_idx, 1], ci[0, 0])
        # nl.store(loop_var[seq_idx, k_tile_idx, 2], nr[0, 0])
        # nl.store(loop_var[seq_idx, k_tile_idx, 3], nc[0, 0])
        # nl.store(loop_var[seq_idx, k_tile_idx, 4], tile_offset[0, 0])
        # nl.store(loop_var[seq_idx, k_tile_idx, 5], a_offset[0, 0])

    # only reorder cached token mask
    # if not contexted:
    #     transpose_kv_token_mask(prior_mask, tiled_block_size)

    # nl.store(iota_hbm[0, :, :], a)
    # nl.store(iota_hbm[1, :, :], b)

    # return prior_mask, iota_hbm, new_mask_hbm, loop_var
    return prior_mask


@nki.jit
def transpose_kv_token_mask(mask, tiled_block_size, B_P_SIZE=128):
    """
    from
    (total_query_len, context_kv_len // LARGE_TILE_SZ, num_tiled_blocks // B_P_SIZE, B_P_SIZE, tiled_block_size)
    to
    (total_query_len, context_kv_len // LARGE_TILE_SZ, num_tiled_blocks // B_P_SIZE, tiled_block_size, B_P_SIZE)
    """
    size_q, size_kv = mask.shape
    assert size_kv % tiled_block_size == 0
    num_block = size_kv // tiled_block_size
    assert num_block >= B_P_SIZE, (
        f"invalid {num_block=}, {size_kv=}, {tiled_block_size=}"
    )
    num_batch = num_block // B_P_SIZE
    _print(f"{size_q=}, {num_batch=}, {B_P_SIZE=}, {tiled_block_size=}")
    transpose_range = B_P_SIZE * tiled_block_size
    new_mask_shape1 = (size_q * num_batch, B_P_SIZE, tiled_block_size)
    new_mask_shape2 = (size_q * num_batch, tiled_block_size, B_P_SIZE)
    mask_re = mask.reshape(new_mask_shape1)
    mask_out = mask.reshape(new_mask_shape2)
    i_p_src, i_f_src = nl.mgrid[0:B_P_SIZE, 0:tiled_block_size]
    i_p_dst, i_f_dst = nl.mgrid[0:tiled_block_size, 0:B_P_SIZE]
    for q_idx in nl.sequential_range(size_q):
        tile = nl.load(mask_re[q_idx, i_p_src, i_f_src])
        transposed = nl.transpose(tile)
        _print(f"{transposed.shape=}")
        nl.store(mask_out[q_idx, i_p_dst, i_f_dst], value=transposed)
    import pdb

    pdb.set_trace()
    return mask_out


@nki.jit(debug_kernel=True)
def flash_paged_attention(
    query,
    key,
    value,
    kv_cache,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    num_seqs,
    block_tables,
    softmax_scale,
):
    """
    Flash PagedAttention Forward Kernel.

    IO tensor layouts:
      - query: shape   (1, n_heads, d, seq_q)
      - key:   shape   (1, n_kv_heads, d, seq_k)
      - value: shape   (1, n_kv_heads, seq_v, d)
      - kv_cache: (2, num_blocks, n_kv_heads, block_size, d)
      - block_tables: (num_active_blocks, )
      - mask: (seq_q, num_active_blocks * block_size + seq_q)
      - o: shape (1, n_heads, seq_q, d)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is
        always 1, and different requests are concatenated along sequence
        dimension.
      - We use paged cache blocks (kv_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for
        block_tables (int32) and mask (int32)
      - If mixed_percision is True, then all Tensor Engine operation will be
        performed in bfloat16 and accumulation will be performed in float32.
        Otherwise the intermediates will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, default
        is set to `true`, if false, we use same precision as input types
      - LARGE_TILE_SZ: `default=2048`, size of the kv tile size for attention
        computation reduction

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of
      nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    mixed_precision = True
    return_debug_tensors = False

    b, h, d, seqlen_q = query.shape

    # automatic determination of tunable parameters
    B_P_SIZE = min(128, seqlen_q)
    B_F_SIZE = 128
    LARGE_TILE_SZ = min(512, B_F_SIZE)
    B_D_SIZE = d

    n_tile_q = cdiv(seqlen_q, B_P_SIZE)  # since q will be loaded on tensor engine
    assert n_tile_q >= 1, f"at least 1 query-tile is required: {seqlen_q} vs {B_P_SIZE}"
    _, num_blocks, k_h, block_size, _ = kv_cache.shape
    q_h_per_k_h = h // k_h
    assert b == 1, f"invalid batch size {b=}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d=}"
    cache_shape = (2, num_blocks, k_h, block_size, d)
    assert tuple(kv_cache.shape) == cache_shape, (
        f"{kv_cache.shape=} mismatch, expect {cache_shape}"
    )
    assert key is None or tuple(key.shape) == (
        1,
        k_h,
        d,
        seqlen_q,
    ), f"key shape {key.shape} mismatch!"
    assert value is None or tuple(value.shape) == (
        1,
        k_h,
        seqlen_q,
        d,
    ), f"value shape {value.shape} mismatch!"

    assert nl.program_ndim() == 2, (
        f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    )
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)

    # num_active_blocks, = block_tables.shape
    num_active_blocks = cdiv(max_seqlen_k, block_size)
    context_kv_len = num_active_blocks * block_size
    assert LARGE_TILE_SZ % B_F_SIZE == 0, (
        f"Need {LARGE_TILE_SZ=} to be divisible by {B_F_SIZE=} in transpose_p"
    )
    assert context_kv_len % LARGE_TILE_SZ == 0, (
        f"Need {context_kv_len=} to be divisible by {LARGE_TILE_SZ=}"
    )

    num_blocks_per_large_tile = LARGE_TILE_SZ // block_size
    assert is_power_of_2(num_blocks_per_large_tile), (
        f"{num_blocks_per_large_tile=} is expected of be power of 2"
    )
    if seqlen_q > B_F_SIZE:
        MAX_REDUCTION_TILE = 2048
        if seqlen_q // 2 > MAX_REDUCTION_TILE:
            assert seqlen_q % MAX_REDUCTION_TILE == 0, (
                f"{seqlen_q=} should be divisible by {MAX_REDUCTION_TILE=}"
            )
        else:
            assert seqlen_q % B_F_SIZE == 0, (
                f"{seqlen_q=} should be divisible by {B_F_SIZE=})"
            )

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    num_large_k_tile = context_kv_len // LARGE_TILE_SZ
    assert (context_kv_len % LARGE_TILE_SZ) == 0, (
        f"invalid num_large_k_tile: {num_large_k_tile=}, {context_kv_len=}"
    )

    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    hbm_l_buffer, hbm_m_buffer, hbm_qk_res, qk_res_buffer = (
        None,
        None,
        None,
        None,
    )
    if return_debug_tensors:
        hbm_l_buffer = nl.ndarray(
            (b, h, seqlen_q), dtype=acc_type, buffer=nl.shared_hbm
        )
        hbm_m_buffer = nl.ndarray(
            (b, h, seqlen_q), dtype=acc_type, buffer=nl.shared_hbm
        )
        hbm_qk_res = nl.ndarray(
            (b, h, B_P_SIZE, seqlen_q), dtype=acc_type, buffer=nl.shared_hbm
        )
        qk_res_buffer = nl.zeros(
            (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), seqlen_q),
            dtype=acc_type,
            buffer=nl.sbuf,
            lazy_initialization=True,
        )

    if max_seqlen_k > 0:
        block_tables_sbuf = load_active_block_tables(
            block_tables_hbm=block_tables,
            num_tiles=num_large_k_tile,
            num_blocks_per_tile=num_blocks_per_large_tile,
            B_P_SIZE=B_P_SIZE,
        )

        # On Neuron, we need B_P_SIZE(128) blocks to make DMA efficient
        if num_blocks_per_large_tile < B_P_SIZE:
            # we checked num_blocks_per_tile is a power of 2
            assert B_P_SIZE % num_blocks_per_large_tile == 0
            block_size_tiling_factor = B_P_SIZE // num_blocks_per_large_tile
            assert block_size % block_size_tiling_factor == 0
            num_blocks_per_large_tile *= block_size_tiling_factor  # i.e. = B_P_SIZE
        else:
            block_size_tiling_factor = 1
        tiled_block_size = block_size // block_size_tiling_factor
        _print(f"{tiled_block_size=}, {block_size=}, {block_size_tiling_factor=}")

        # Indirect DMA load must be placed along Partition Dimension
        block_tables_sbuf = transform_block_tables_for_indirect_load(
            block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
            B_P_SIZE=B_P_SIZE,
        )

        # Flatten KV cache to be 2D for loading into SBUF
        new_cache_shape = (
            2 * num_blocks * k_h * block_size_tiling_factor,
            tiled_block_size * d,
        )
        v_cache_offset = (
            num_blocks * k_h * block_size_tiling_factor * tiled_block_size * d
        )
        kv_cache = kv_cache.reshape(new_cache_shape)

    # Global Flash Attention accumulators
    o_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), d),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    l_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    m_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )

    max_num_seqs = num_seqs  # cu_seqlens_q.shape[0] - 1

    active_mask = nl.ndarray(
        (max_seqlen_q, max_seqlen_q), dtype=nl.bool_, buffer=nl.private_hbm
    )
    # mask_output = nl.ndarray((k_h, num_large_k_tile, B_P_SIZE, LARGE_TILE_SZ), buffer=nl.shared_hbm, dtype=nl.bool_)
    _print(f"{num_large_k_tile=}, {n_tile_q=}, {B_P_SIZE=}, {LARGE_TILE_SZ=}")

    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(max_num_seqs)[None, :]
    seqused_k_sbuf = nl.load(seqused_k[i_f])

    if max_seqlen_k > 0:
        prior_mask = nl.ndarray(
            (max_seqlen_q, max_seqlen_k), dtype=nl.bool_, buffer=nl.private_hbm
        )
        prior_mask = build_attention_mask(
            prior_mask,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k_sbuf,
            max_seqlen_q,
            max_seqlen_k,
            max_num_seqs=max_num_seqs,
            block_size=block_size,
            # tiled_block_size=tiled_block_size,
            B_F_SIZE=B_F_SIZE,
        )

        # only reorder cached token mask
        tiled_mask = transpose_kv_token_mask(
            prior_mask, tiled_block_size, B_P_SIZE=B_P_SIZE
        )
        assert (
            tiled_mask.shape[-2]
            * tiled_mask.shape[-1]
            % (num_large_k_tile * LARGE_TILE_SZ)
        ) == 0, (
            f"unexpected tiled_mask shape: {tiled_mask.shape=}, {prior_mask.shape=}, {num_large_k_tile=}, "
            f"{context_kv_len=}, {num_active_blocks=}, {block_tables.shape=}"
        )

    for large_k_tile_idx in nl.sequential_range(0, num_large_k_tile):
        # cur_mask = nl.load(tiled_mask[
        #     nl.ds(0, B_P_SIZE),
        #     nl.ds(large_k_tile_idx * LARGE_TILE_SZ, LARGE_TILE_SZ),
        # ])
        # nl.store(mask_output[head_id, large_k_tile_idx, :, :], cur_mask[:, :])
        # cur_mask = nl.load(tiled_mask[
        #     nl.ds(0, B_P_SIZE),
        #     nl.ds(large_k_tile_idx * LARGE_TILE_SZ, LARGE_TILE_SZ),
        # ])
        pass

    if max_seqlen_k > 0:
        num_loads = num_blocks_per_large_tile // B_P_SIZE
        # XXX: Work around a DMA skipping correctness issue:
        #      If nl.ndarray is used to allocate buffer for DMA skipping,
        #      kernel does not produce correct results.
        k_load_buffer = nl.zeros(
            (par_dim(B_P_SIZE), num_loads, tiled_block_size * B_D_SIZE),
            dtype=kernel_dtype,
        )
        v_load_buffer = nl.zeros(
            (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE),
            dtype=kernel_dtype,
        )

    for large_k_tile_idx in nl.sequential_range(0, num_large_k_tile):
        # cur_k_tile = nl.ndarray(
        #     (par_dim(B_D_SIZE), LARGE_TILE_SZ),
        #     dtype=kernel_dtype,
        # )
        # cur_v_tile = nl.ndarray(
        #     (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE),
        #     dtype=kernel_dtype,
        # )
        cur_k_tile, cur_v_tile = load_kv_tile_from_cache(
            kv_cache=kv_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=large_k_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            B_P_SIZE=B_P_SIZE,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            v_cache_offset=v_cache_offset,
        )

        cur_mask = nl.ndarray(
            (par_dim(B_P_SIZE), n_tile_q, LARGE_TILE_SZ),
            dtype=nl.bool_,
        )
        for small_q_idx in nl.affine_range(n_tile_q):
            # cur_mask = nl.load(prior_mask[
            #     nl.ds(i * B_P_SIZE, B_P_SIZE),
            #     nl.ds(large_k_tile_idx * LARGE_TILE_SZ, LARGE_TILE_SZ),
            # ])
            cur_mask[:, small_q_idx, :] = nl.load(
                tiled_mask[
                    large_k_tile_idx,
                    nl.ds(small_q_idx * tiled_block_size, tiled_block_size),
                    :,
                ]
            )
            # import pdb
            # pdb.set_trace()
            # cur_mask = nl.load(prior_mask[
            #     nl.ds(i * B_P_SIZE, B_P_SIZE),
            #     nl.ds(large_k_tile_idx * LARGE_TILE_SZ, LARGE_TILE_SZ),
            # ])
            # if tiled_block_size > 1:
            #     cur_mask = transpose_kv_token_mask(cur_mask, tiled_block_size)
            # nl.store(mask_output[large_k_tile_idx, small_q_idx, :, :], cur_mask[:, :])
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(
                    q_hbm_tile[:, nl.ds(small_q_idx * B_P_SIZE, B_P_SIZE)]
                )
                if q_sbuf_tile.dtype != kernel_dtype:
                    q_sbuf_tile = nl.copy(q_sbuf_tile, dtype=kernel_dtype)
                q_tile[:, :] = q_sbuf_tile * softmax_scale

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_buffer[small_q_idx, i_q_h],
                    l_buffer=l_buffer[small_q_idx, i_q_h],
                    m_buffer=m_buffer[small_q_idx, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask[:, small_q_idx, :],
                    use_causal_mask=False,
                    q_tile_idx=None,
                    initialize=large_k_tile_idx == 0,
                    B_D_SIZE=B_D_SIZE,
                )

    i_p = nl.arange(1)[:, None]
    cu_f = nl.arange(max_num_seqs + 1)[None, :]
    i_f = nl.arange(max_num_seqs)[None, :]
    cu_seqlens_q_sbuf = nl.load(cu_seqlens_q[cu_f])
    seqused_q = cu_seqlens_q_sbuf[i_p, i_f + 1] - cu_seqlens_q_sbuf[i_p, i_f]
    active_mask = build_attention_mask(
        active_mask,
        cu_seqlens_q,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_seqlen_q,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        # tiled_block_size=tiled_block_size,
        contexted=True,
    )
    # cu_query_lens, cu_query_lens, query_lens, max_query_lens, max_query_lens, max_num_seqs, block_size, contexted=True)

    # compute attention between input query, key and value
    if key is not None and value is not None:
        B_F_SIZE = min(seqlen_q, B_F_SIZE)
        LARGE_TILE_SZ = seqlen_q

        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        cur_v_tile = nl.ndarray(
            (par_dim(B_P_SIZE), LARGE_TILE_SZ // B_P_SIZE * B_D_SIZE),
            dtype=kernel_dtype,
        )

        loaded = nl.load(key[batch_id, head_id, :, :])
        if loaded.dtype != kernel_dtype:
            loaded = nl.copy(loaded, dtype=kernel_dtype)
        cur_k_tile[:, :] = loaded

        v_hbm_tile = value[batch_id, head_id]
        for v_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
            load_v_tile(
                v_hbm_tile=v_hbm_tile,
                cur_v_tile=cur_v_tile,
                large_tile_idx=0,
                v_i=v_i,
                LARGE_TILE_SZ=LARGE_TILE_SZ,
            )

        for i in nl.affine_range(n_tile_q):
            # cur_mask = nl.load(mask[
            #     nl.ds(i * B_P_SIZE, B_P_SIZE),
            #     nl.ds(context_kv_len, LARGE_TILE_SZ),
            # ])
            cur_mask = nl.load(
                active_mask[
                    nl.ds(i * B_P_SIZE, B_P_SIZE),
                    nl.ds(0, LARGE_TILE_SZ),
                ]
            )
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)])
                if q_sbuf_tile.dtype != kernel_dtype:
                    q_sbuf_tile = nl.copy(q_sbuf_tile, dtype=kernel_dtype)
                q_tile[:, :] = q_sbuf_tile * softmax_scale
                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    o_buffer=o_buffer[i, i_q_h],
                    l_buffer=l_buffer[i, i_q_h],
                    m_buffer=m_buffer[i, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=cur_mask,
                    use_causal_mask=True,
                    q_tile_idx=i,
                    initialize=(i == 0),
                    # initialize=False,
                    B_D_SIZE=B_D_SIZE,
                    qk_res_buffer=(
                        qk_res_buffer[i, i_q_h] if qk_res_buffer is not None else None
                    ),
                )

    # -- -- -- -- write output to buffer on HBM -- -- -- -- -- -- #
    for i_q_h in nl.affine_range(q_h_per_k_h):
        for i in nl.affine_range(n_tile_q):
            out = nl.multiply(
                o_buffer[i, i_q_h],
                nl.exp(m_buffer[i, i_q_h] - l_buffer[i, i_q_h]),
                dtype=kernel_dtype,
            )

            nl.store(
                o[
                    batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    nl.ds(i * B_P_SIZE, B_P_SIZE),
                    :,
                ],
                out,
            )
            # maximum and summation statistics
            if return_debug_tensors:
                nl.store(
                    hbm_m_buffer[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                    ],
                    m_buffer[i, i_q_h, :, :],
                )
                nl.store(
                    hbm_l_buffer[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        nl.ds(i * B_P_SIZE, B_P_SIZE),
                    ],
                    l_buffer[i, i_q_h],
                )
                nl.store(
                    hbm_qk_res[batch_id, head_id * q_h_per_k_h + i_q_h, :, :],
                    qk_res_buffer[batch_id, i_q_h, :, :],
                )

    if return_debug_tensors:
        return o, hbm_m_buffer, hbm_l_buffer, hbm_qk_res
    return o, active_mask


def reorder_context_mask(mask, LARGE_TILE_SZ, block_size):
    """
    Reorder the mask to make it compatible with the flash attention kernel.

    We vectorize KV cache read to improve DMA utilization. However, the layout
    that maximizes DMA bandwidth changes the order tokens are consumed.

    The token layout (inner 2 dimensions) after vectorized load is (B_P_SIZE,
    tiled_block_size) in a tile of `B_P_SIZE * tiled_block_size` tokens. And
    each step the engine consumes a column (rather than a row) of B_P_SIZE
    tokens. Therefore, the tokens are visited in a strided way.

    To make sure mask matches the order tokens are consumed, we need to properly
    transpose mask.
    """
    total_query_len, total_seq_len = mask.shape
    context_kv_len = total_seq_len - total_query_len

    assert LARGE_TILE_SZ >= B_P_SIZE, (
        f"{LARGE_TILE_SZ=} must be larger than {B_P_SIZE=}"
    )
    num_tiled_blocks = max(B_P_SIZE, LARGE_TILE_SZ // block_size)
    tiled_block_size = LARGE_TILE_SZ // num_tiled_blocks
    if tiled_block_size > 1:
        # Mask reordering is needed when tiled_block_size > 1
        device = mask.device
        mask = mask.cpu()
        context_mask = mask[:, :context_kv_len]
        context_mask = context_mask.view(
            total_query_len,
            context_kv_len // LARGE_TILE_SZ,
            num_tiled_blocks // B_P_SIZE,
            B_P_SIZE,
            tiled_block_size,
        )
        context_mask = context_mask.transpose(3, 4).reshape(
            total_query_len, context_kv_len
        )
        new_mask = mask[:, context_kv_len:]
        return torch.concat([context_mask, new_mask], dim=1).to(device)
    else:
        return mask


def flash_attn_varlen_func(
    query,
    key,
    value,
    kv_cache,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    num_seqs,
    block_tables,
    softmax_scale,
):
    """
    Compute flash paged attention for variable length sequences.

    This function is a wrapper around the flash attention NKI kernel. It takes
    in the following arguments:
      - query: (1, n_heads, d, seq_q)
      - key:   (1, n_kv_heads, d, seq_k)
      - value: (1, n_kv_heads, seq_v, d)
      - kv_cache:   (2, n_blocks, n_kv_heads, block_size, d)
      - block_tables: (n_active_blocks, )

    Notes:
      - Key/value cache layout must be (n_blocks, n_kv_heads, block_size, d)
        for better DMA throughput
    """
    N, _, n_kv_head, _, _ = kv_cache.shape
    assert N == 2, f"invalid {kv_cache.shape=}"
    # assert block_tables.shape[0] == 128, f"invalid block_tables shape: {block_tables.shape=}"

    o, *debug_tensors = flash_paged_attention[1, n_kv_head](
        query,
        key,
        value,
        kv_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        block_tables,
        softmax_scale,
    )
    return o, *debug_tensors
