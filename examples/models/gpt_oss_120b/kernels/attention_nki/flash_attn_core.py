"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Migrated from the legacy ``neuronxcc.nki`` API to standalone ``nki`` (beta-3):
- ``nl.par_dim(...)`` removed: the partition dim is just the leading plain int in
  ``nl.ndarray(...)``.
- ``nisa`` ops are now ``dst``-first and return nothing (tensor_tensor/tensor_reduce/
  tensor_scalar/activation/activation_reduce/nc_matmul/nc_transpose/select_reduce/
  tensor_copy_predicated).
- ``nl.matmul(x, y, transpose_x=True)`` is retained (it lowers to nc_matmul in PSUM);
  the accumulating ``+=`` matmuls are expressed with ``nc_matmul(accumulate=...)``.
"""

import numpy as np

import nki.isa as nisa
import nki.language as nl

from .attn_utils import (
    B_P_SIZE,
    B_FMAX_SIZE,
    NEG_INF,
)


def transpose_p_local(
    p_local_transposed,
    p_local,
    Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
    q_idx,
    kv_idx,
    B_F_SIZE=B_FMAX_SIZE,
):
    assert p_local.shape == (Q_TILE_SIZE, LARGE_KV_TILE_SIZE)
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    CONTRACTION_TILE_SIZE = min(B_P_SIZE, LARGE_KV_TILE_SIZE)
    for i in nl.affine_range(LARGE_KV_TILE_SIZE // B_F_SIZE):
        if q_idx * Q_TILE_SIZE >= kv_idx * LARGE_KV_TILE_SIZE + i * B_F_SIZE:
            p_local_t_tmp = nl.ndarray(
                (
                    CONTRACTION_TILE_SIZE,
                    B_F_SIZE // CONTRACTION_TILE_SIZE * Q_TILE_SIZE,
                ),
                buffer=nl.psum,
                dtype=nl.float32 if is_nc_gen2 else p_local.dtype,
            )
            for j in nl.affine_range(B_F_SIZE // CONTRACTION_TILE_SIZE):

                j_128_slice = nl.ds(j * Q_TILE_SIZE, Q_TILE_SIZE)
                i_j_128_slice = nl.ds(
                    i * B_F_SIZE + j * CONTRACTION_TILE_SIZE, CONTRACTION_TILE_SIZE
                )
                nisa.nc_transpose(
                    dst=p_local_t_tmp[:, j_128_slice],
                    data=p_local[:, i_j_128_slice],
                    engine=nisa.engine.tensor,
                )
            nisa.tensor_copy(
                dst=p_local_transposed[
                    :,
                    nl.ds(
                        i * (B_F_SIZE // CONTRACTION_TILE_SIZE * Q_TILE_SIZE),
                        (B_F_SIZE // CONTRACTION_TILE_SIZE * Q_TILE_SIZE),
                    ),
                ],
                src=p_local_t_tmp,
            )


def _flash_attention_core(
    q_local_tile,
    k,
    v,
    sink,
    o_buffer,
    l_buffer,
    m_buffer,
    kernel_dtype,
    acc_type,
    tile_mask,
    q_tile_idx=None,
    local_k_tile_idx=None,
    Q_TILE_SIZE=128,
    LARGE_KV_TILE_SIZE=2048,
    B_F_SIZE=512,
    B_D_SIZE=128,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q_local_tile: (B_D_SIZE, Q_TILE_SIZE)
    k: (B_D_SIZE, LARGE_KV_TILE_SIZE)
    v: (B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE)
    The results are stored in the following three buffers
    o_buffer: (Q_TILE_SIZE, num_large_k_tiles + 1, B_D_SIZE)
    l_buffer: (Q_TILE_SIZE, num_large_k_tiles + 1, 1)
    m_buffer: (Q_TILE_SIZE, num_large_k_tiles + 1, 1)
    """
    assert (
        LARGE_KV_TILE_SIZE % B_P_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    assert (
        LARGE_KV_TILE_SIZE % B_F_SIZE == 0
    ), f"{LARGE_KV_TILE_SIZE=} not divisive by {B_F_SIZE=}"
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    qk_res_buf = nl.ndarray(
        (Q_TILE_SIZE, LARGE_KV_TILE_SIZE),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    if sink is not None:
        max_local = nl.full(
            (Q_TILE_SIZE, num_k_tile_per_large_tile + 1),
            NEG_INF,
            dtype=acc_type,
        )
        nisa.tensor_copy(dst=max_local[:, num_k_tile_per_large_tile], src=sink)
    else:
        max_local = nl.full(
            (Q_TILE_SIZE, num_k_tile_per_large_tile),
            NEG_INF,
            dtype=acc_type,
        )
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        if (
            q_tile_idx * Q_TILE_SIZE
            >= local_k_tile_idx * LARGE_KV_TILE_SIZE + k_i * B_F_SIZE
        ):
            qk_psum = nl.ndarray(
                (Q_TILE_SIZE, B_F_SIZE), dtype=nl.float32, buffer=nl.psum
            )  # (128, 512)
            # q_local_tile: (B_D_SIZE, Q_TILE_SIZE); transpose_x=True => q.T @ k
            nisa.nc_matmul(
                dst=qk_psum,
                stationary=q_local_tile,
                moving=k[:, k_i_b_f_slice],
            )  # (p(128), 512)
            nisa.select_reduce(
                dst=qk_res_buf[:, k_i_b_f_slice],
                predicate=tile_mask[:, k_i_b_f_slice],
                on_true=qk_psum[:, nl.ds(0, B_F_SIZE)],
                on_false=NEG_INF,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                reduce_res=max_local[:, k_i],
                reduce_op=nl.maximum,
            )

    # Calculate max of the current tile
    max_ = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type)
    nisa.tensor_reduce(
        dst=max_,
        op=nl.max,
        data=max_local[:, :],
        axis=(1,),
        negate=False,
    )

    o_previous_scaled = nl.ndarray(
        (Q_TILE_SIZE, B_D_SIZE),
        dtype=o_buffer.dtype,
    )

    m_previous = m_buffer[:, local_k_tile_idx]
    m_current_neg = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type)
    nisa.tensor_scalar(
        dst=m_current_neg,
        data=max_,
        op0=nl.maximum,
        operand0=m_previous,
        op1=nl.multiply,
        operand1=-1,
    )

    p_local = nl.ndarray(
        (Q_TILE_SIZE, LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    REDUCTION_TILE = B_F_SIZE

    if sink is not None:
        p_partial_sum = nl.zeros(
            (Q_TILE_SIZE, (LARGE_KV_TILE_SIZE // REDUCTION_TILE) + 1),
            dtype=acc_type,
        )
        nisa.activation(
            dst=p_partial_sum[:, LARGE_KV_TILE_SIZE // REDUCTION_TILE],
            op=nl.exp,
            data=sink,
            bias=m_current_neg,
            scale=1.0,
        )
    else:
        p_partial_sum = nl.zeros(
            (Q_TILE_SIZE, LARGE_KV_TILE_SIZE // REDUCTION_TILE),
            dtype=acc_type,
        )

    for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
        if (
            q_tile_idx * Q_TILE_SIZE
            >= local_k_tile_idx * LARGE_KV_TILE_SIZE + k_r_i * REDUCTION_TILE
        ):
            k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)
            nisa.activation_reduce(
                dst=p_local[:, k_r_i_reduce_slice],
                op=nl.exp,
                data=qk_res_buf[:, k_r_i_reduce_slice],
                bias=m_current_neg,
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=p_partial_sum[:, k_r_i],
            )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, keepdims=True)

    p_local_transposed = nl.ndarray(
        (B_P_SIZE, LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
        dtype=kernel_dtype,
    )
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        Q_TILE_SIZE=Q_TILE_SIZE,
        LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
        B_F_SIZE=B_F_SIZE,
        q_idx=q_tile_idx,
        kv_idx=local_k_tile_idx,
    )

    pv_psum = nl.zeros(
        (Q_TILE_SIZE, B_D_SIZE),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
        if (
            q_tile_idx * Q_TILE_SIZE
            >= local_k_tile_idx * LARGE_KV_TILE_SIZE + k_i * B_P_SIZE
        ):
            # p_local_transposed[:, k_i tile]: (B_P_SIZE, Q_TILE_SIZE),
            # transpose_x=True => (Q_TILE_SIZE, B_P_SIZE) @ v (B_P_SIZE, B_D_SIZE)
            nisa.nc_matmul(
                dst=pv_psum,
                stationary=p_local_transposed[:, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)],
                moving=v[:, k_i, :],
                accumulate=True,
            )  # (128, 128) (p(Br), d)

    # Compute scaling factor
    alpha = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type)
    nisa.activation(
        dst=alpha,
        op=nl.exp,
        data=m_previous,
        bias=m_current_neg,
        scale=1.0,
    )

    nisa.activation(
        dst=m_buffer[:, local_k_tile_idx + 1],
        op=nl.copy,
        data=m_current_neg,
        scale=-1.0,
    )
    nisa.tensor_scalar(
        dst=o_previous_scaled,
        data=o_buffer[:, local_k_tile_idx],
        op0=nl.multiply,
        operand0=alpha,
    )
    nisa.tensor_tensor(
        dst=o_buffer[:, local_k_tile_idx + 1],
        data1=o_previous_scaled,
        data2=pv_psum,
        op=nl.add,
    )

    l_prev = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type)
    nisa.tensor_scalar(
        dst=l_prev,
        data=l_buffer[:, local_k_tile_idx],
        op0=nl.multiply,
        operand0=alpha,
    )
    nisa.tensor_tensor(
        dst=l_buffer[:, local_k_tile_idx + 1],
        data1=l_prev,
        data2=ps,
        op=nl.add,
    )


def partition_broadcast_fp32(src, out_psum):
    assert src.shape[1:] == out_psum.shape[1:]
    assert src.dtype in [nl.bfloat16, nl.float32]
    assert out_psum.dtype == nl.float32
    p_size = out_psum.shape[0]
    ones = nl.ones((1, p_size), dtype=src.dtype)
    nisa.nc_matmul(
        dst=out_psum,
        stationary=ones,
        moving=src,
        is_stationary_onezero=True,
    )


def transpose_with_matmul(src, out_psum):
    # beta-3: transpose via the tensor engine directly (fp32 tensor-engine
    # nc_transpose is exact and lands in PSUM), dropping the old identity-matrix
    # matmul which required an np.identity operand the beta-3 frontend can't
    # build inside a kernel body.
    nisa.nc_transpose(dst=out_psum, data=src, engine=nisa.engine.tensor)


def _load_k_cache(k_load_buffer, cache_k, k_id, kv_head_id, batch_size, MULTI_BUFFER):
    for batch_id in nl.affine_range(batch_size):
        nisa.dma_copy(
            dst=k_load_buffer[:, k_id % MULTI_BUFFER, batch_id, :],
            src=cache_k[batch_id, kv_head_id, :, k_id],
            dge_mode=nisa.dge_mode.swdge,
        )


def _load_v_cache(v_load_buffer, cache_v, v_id, kv_head_id, batch_size, MULTI_BUFFER):
    for batch_id in nl.affine_range(batch_size):
        nisa.dma_copy(
            dst=v_load_buffer[:, v_id % MULTI_BUFFER, batch_id, :, :],
            src=cache_v[batch_id, kv_head_id, v_id],
            dge_mode=nisa.dge_mode.swdge,
        )


def _flash_attention_core_kq_matmul(
    q_local_tile,
    cache_k,
    cache_v,
    k_load_buffer,
    v_load_buffer,
    tile_mask,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    large_kv_tile_id,
    num_large_kv_tiles,
    kv_head_id,
    kernel_dtype,
    acc_type,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    Input:
    q_local_tile: (B_D_SIZE, batch_size, q_h_per_k_h)
    k_load_buffer: (B_D_SIZE, MULTI_BUFFER, batch_size, large_kv_tile_size)
    v_load_buffer: (B_P_SIZE, MULTI_BUFFER, batch_size, B_D_SIZE, num_k_tiles)
                   where num_k_tiles = large_kv_tile_size // B_P_SIZE
    tile_mask: (B_P_SIZE, batch_size, large_kv_tile_size // B_P_SIZE)
    The results are stored in the following three buffers
    o_buffer_sbuf: (B_D_SIZE, num_k_tiles, batch_size * q_h_per_k_h)
    l_buffer_sbuf: (1, num_k_tiles, batch_size * q_h_per_k_h)
    m_buffer_sbuf: (1, num_k_tiles, batch_size * q_h_per_k_h)
    """
    num_k_tiles = v_load_buffer.shape[3]
    assert acc_type == nl.float32
    MULTI_BUFFER = k_load_buffer.shape[1]
    B_D_SIZE, batch_size, q_h_per_k_h = q_local_tile.shape
    assert batch_size * q_h_per_k_h <= B_P_SIZE

    k = k_load_buffer.reshape(
        (B_D_SIZE, MULTI_BUFFER, batch_size, B_P_SIZE, num_k_tiles)
    )[:, large_kv_tile_id % MULTI_BUFFER]

    # load current v
    _load_v_cache(
        v_load_buffer, cache_v, large_kv_tile_id, kv_head_id, batch_size, MULTI_BUFFER
    )

    # load next k
    if large_kv_tile_id + 1 < num_large_kv_tiles:
        _load_k_cache(
            k_load_buffer,
            cache_k,
            large_kv_tile_id + 1,
            kv_head_id,
            batch_size,
            MULTI_BUFFER,
        )

    # Calculate KQ and mask
    kq_res_buf = nl.full(
        (B_P_SIZE, batch_size, num_k_tiles, q_h_per_k_h),
        NEG_INF,
        dtype=acc_type,
    )
    for batch_id in nl.affine_range(batch_size):
        kq_res_psum = nl.ndarray(
            (B_P_SIZE, num_k_tiles, q_h_per_k_h),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for k_i in nl.affine_range(num_k_tiles):
            nisa.nc_matmul(
                dst=kq_res_psum[:, k_i, :],
                stationary=k[:, batch_id, :, k_i],
                moving=q_local_tile[:, batch_id, :],
            )  # (p(128), Q_TILE_SIZE)

        nisa.tensor_copy_predicated(
            src=kq_res_psum,
            dst=kq_res_buf[:, batch_id],
            predicate=tile_mask[:, batch_id],
        )

    # Calculate max of the current tile (cascade reduction). beta-3 tensor_reduce
    # only reduces trailing dims, so reduce over the num_k_tiles axis explicitly.
    max_partial = nl.ndarray(
        (B_P_SIZE, batch_size, q_h_per_k_h),
        dtype=acc_type,
    )
    nisa.tensor_copy(dst=max_partial, src=kq_res_buf[:, :, 0, :])
    for k_i in range(1, num_k_tiles):
        nisa.tensor_tensor(
            dst=max_partial,
            data1=max_partial,
            data2=kq_res_buf[:, :, k_i, :],
            op=nl.maximum,
        )
    max_partial_transposed = nl.ndarray(
        (q_h_per_k_h * batch_size, B_P_SIZE), dtype=acc_type, buffer=nl.psum
    )
    transpose_with_matmul(
        max_partial.reshape((B_P_SIZE, batch_size * q_h_per_k_h)),
        max_partial_transposed,
    )
    max_ = nl.ndarray((batch_size * q_h_per_k_h, 1), dtype=acc_type)
    nisa.tensor_reduce(
        dst=max_,
        op=nl.max,
        data=max_partial_transposed,
        axis=(1,),
        negate=False,
        keepdims=True,
    )  # (batch_size * q_h_per_k_h, 1)

    # Calculate max
    max_transposed_psum = nl.ndarray(
        (1, batch_size * q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    transpose_with_matmul(max_, max_transposed_psum)
    m_previous = m_buffer_sbuf[:, large_kv_tile_id]
    m_current = m_buffer_sbuf[:, large_kv_tile_id + 1]
    nisa.tensor_tensor(
        dst=m_current,
        data1=m_previous,
        data2=max_transposed_psum,
        op=nl.maximum,
    )  # (128,1)

    # Compute scaling factor and broadcast
    bias = nl.zeros((1, 1), dtype=m_previous.dtype)
    m_diff = nl.ndarray((1, batch_size * q_h_per_k_h), dtype=acc_type)
    nisa.tensor_tensor(dst=m_diff, data1=m_previous, data2=m_current, op=nl.subtract)
    alpha = nl.ndarray((1, batch_size * q_h_per_k_h), dtype=acc_type)
    nisa.activation(
        dst=alpha,
        op=nl.exp,
        data=m_diff,
        scale=1.0,
        bias=bias,
    )
    alpha_broadcasted = nl.ndarray(
        (B_D_SIZE, batch_size * q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    partition_broadcast_fp32(alpha, alpha_broadcasted)

    # Rescale previous output
    o_previous_scaled = nl.ndarray(
        (B_D_SIZE, batch_size * q_h_per_k_h),
        dtype=o_buffer_sbuf.dtype,
    )
    nisa.tensor_tensor(
        dst=o_previous_scaled,
        data1=o_buffer_sbuf[:, large_kv_tile_id],
        data2=alpha_broadcasted,
        op=nl.multiply,
    )

    max_broadcasted = nl.ndarray(
        (B_P_SIZE, batch_size, 1, q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    max_broadcasted_2d_view = max_broadcasted.reshape(
        (B_P_SIZE, batch_size * q_h_per_k_h)
    )
    partition_broadcast_fp32(m_current, max_broadcasted_2d_view)

    # Calculate unnormed Softmax(P) = exp(P - max)
    p_local = nl.ndarray(
        (B_P_SIZE, batch_size, num_k_tiles, q_h_per_k_h),
        dtype=kernel_dtype,
    )
    bias = nl.zeros((B_P_SIZE, 1), dtype=max_broadcasted.dtype)
    kq_minus_max = nl.ndarray(
        (B_P_SIZE, batch_size, num_k_tiles, q_h_per_k_h), dtype=acc_type
    )
    # kq_res_buf - max_broadcasted, broadcasting max over the num_k_tiles axis
    # (beta-3 tensor_tensor doesn't broadcast a middle dim -> loop over k tiles).
    for k_i in nl.affine_range(num_k_tiles):
        nisa.tensor_tensor(
            dst=kq_minus_max[:, :, k_i, :],
            data1=kq_res_buf[:, :, k_i, :],
            data2=max_broadcasted[:, :, 0, :],
            op=nl.subtract,
        )
    nisa.activation(
        dst=p_local,
        op=nl.exp,
        data=kq_minus_max,
        scale=1.0,
        bias=bias,
    )

    # Calculate PV
    v = v_load_buffer[:, large_kv_tile_id % MULTI_BUFFER]
    pv_psum = nl.zeros(
        (B_D_SIZE, batch_size, q_h_per_k_h),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    for batch_id in nl.affine_range(batch_size):
        for k_i in nl.affine_range(num_k_tiles):
            nisa.nc_matmul(
                dst=pv_psum[:, batch_id, :],
                stationary=v[:, batch_id, k_i, :],
                moving=p_local[:, batch_id, k_i],
                accumulate=True,
            )

    # Calculate SumExp
    # XXX: batch_size * num_k_tiles * q_h_per_k_h <= 512 (psum_fmax) may not
    # hold for large tile
    assert num_k_tiles * q_h_per_k_h <= 512
    sumexp_ones = nl.ones((B_P_SIZE, 1), dtype=p_local.dtype)
    ps = nl.ndarray(
        (1, batch_size, q_h_per_k_h),
        dtype=acc_type,
    )
    for batch_id in nl.affine_range(batch_size):
        # sum along partition first
        ps_partial = nl.ndarray(
            (1, num_k_tiles, q_h_per_k_h),
            dtype=acc_type,
            buffer=nl.psum,
        )
        ps_partial_reshape = ps_partial.reshape((1, num_k_tiles * q_h_per_k_h))
        nisa.nc_matmul(
            dst=ps_partial_reshape,
            stationary=sumexp_ones,
            moving=p_local.reshape(
                (
                    B_P_SIZE,
                    batch_size,
                    num_k_tiles * q_h_per_k_h,
                )
            )[:, batch_id],
            is_stationary_onezero=True,
        )
        # Sum over the num_k_tiles axis (middle dim); beta-3 tensor_reduce only
        # reduces trailing dims, so accumulate explicitly.
        nisa.tensor_copy(dst=ps[:, batch_id, :], src=ps_partial[:, 0, :])
        for k_i in range(1, num_k_tiles):
            nisa.tensor_tensor(
                dst=ps[:, batch_id, :],
                data1=ps[:, batch_id, :],
                data2=ps_partial[:, k_i, :],
                op=nl.add,
            )

    # Update output buffer
    nisa.tensor_tensor(
        dst=o_buffer_sbuf[:, large_kv_tile_id + 1, :],
        data1=o_previous_scaled,
        data2=pv_psum.reshape((B_D_SIZE, batch_size * q_h_per_k_h)),
        op=nl.add,
    )
    l_prev_scaled = nl.ndarray((1, batch_size * q_h_per_k_h), dtype=acc_type)
    nisa.tensor_tensor(
        dst=l_prev_scaled,
        data1=l_buffer_sbuf[:, large_kv_tile_id],
        data2=alpha,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=l_buffer_sbuf[:, large_kv_tile_id + 1],
        data1=l_prev_scaled,
        data2=ps.reshape((1, batch_size * q_h_per_k_h)),
        op=nl.add,
    )


def _active_attention_core_batched(
    q,
    k,
    v,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    sink,
    kernel_dtype,
    acc_type,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    q: (B_D_SIZE, q_h_per_k_h, batch_size)
    k: (B_D_SIZE, batch_size)
    v: (B_D_SIZE, batch_size)
    The results are stored in the following three buffers
    o_buffer_sbuf: (B_D_SIZE, batch_size * q_h_per_k_h)
    l_buffer_sbuf: (1, batch_size * q_h_per_k_h)
    m_buffer_sbuf: (1, batch_size * q_h_per_k_h)
    """
    B_D_SIZE, batch_size, q_h_per_k_h = q.shape
    # calculate qk
    ones = nl.ones((B_D_SIZE, 1), dtype=acc_type)
    qk_mul = nl.ndarray(
        (B_D_SIZE, batch_size, q_h_per_k_h),
        dtype=acc_type,
    )
    # qk_mul[:, :, qh] = q[:, :, qh] * k  (k broadcast over the q-head dim; loop
    # because beta-3 tensor_tensor does not broadcast a trailing size-1 dim).
    for qh in nl.affine_range(q_h_per_k_h):
        nisa.tensor_tensor(
            dst=qk_mul[:, :, qh],
            data1=q[:, :, qh],
            data2=k,
            op=nl.multiply,
        )
    qk_psum = nl.ndarray(
        (1, batch_size, q_h_per_k_h),
        buffer=nl.psum,
        dtype=nl.float32,
    )
    nisa.nc_matmul(
        dst=qk_psum,
        stationary=ones,
        moving=qk_mul,
        is_stationary_onezero=True,
    )
    MULTI_BUFFER = m_buffer_sbuf.shape[1]
    m_current = m_buffer_sbuf.reshape((1, MULTI_BUFFER, batch_size, q_h_per_k_h))[:, 0]
    l_current = l_buffer_sbuf.reshape((1, MULTI_BUFFER, batch_size, q_h_per_k_h))[:, 0]
    o_current = o_buffer_sbuf.reshape(
        (
            B_D_SIZE,
            MULTI_BUFFER,
            batch_size,
            q_h_per_k_h,
        )
    )[:, 0]
    if sink is not None:
        qk_sbuf = nl.ndarray((1, batch_size, q_h_per_k_h, 2), dtype=acc_type)
        # load sink; broadcast over the batch dim by looping (a single dma_copy
        # cannot broadcast a size-1 source dim across batches in beta-3).
        sink_r = sink.reshape((1, 1, q_h_per_k_h))
        for b_i in nl.affine_range(batch_size):
            nisa.dma_copy(dst=qk_sbuf[:, nl.ds(b_i, 1), :, 1], src=sink_r)
        nisa.tensor_copy(dst=qk_sbuf[:, :, :, 0], src=qk_psum)
        nisa.tensor_reduce(
            dst=m_current, op=nl.max, data=qk_sbuf, axis=(3,)
        )
        bias = nl.zeros((1, 1), dtype=m_buffer_sbuf.dtype)
        p_local_sink = nl.ndarray(
            (1, batch_size, q_h_per_k_h, 2),
            dtype=kernel_dtype,
        )
        # m_current is a strided view; copy to a contiguous tile and subtract it
        # from each of the two last-axis columns (qk and sink) explicitly.
        m_c = nl.ndarray((1, batch_size, q_h_per_k_h), dtype=acc_type)
        nisa.tensor_copy(dst=m_c, src=m_current)
        qk_minus_m = nl.ndarray((1, batch_size, q_h_per_k_h, 2), dtype=acc_type)
        nisa.tensor_tensor(
            dst=qk_minus_m[:, :, :, 0], data1=qk_sbuf[:, :, :, 0], data2=m_c, op=nl.subtract
        )
        nisa.tensor_tensor(
            dst=qk_minus_m[:, :, :, 1], data1=qk_sbuf[:, :, :, 1], data2=m_c, op=nl.subtract
        )
        nisa.activation(
            dst=p_local_sink,
            op=nl.exp,
            data=qk_minus_m,
            scale=1.0,
            bias=bias,
        )
        p_local = p_local_sink[:, :, :, 0]
        p_local_broadcast = nl.ndarray(
            (B_D_SIZE, batch_size, q_h_per_k_h),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        partition_broadcast_fp32(p_local, p_local_broadcast)
        nisa.tensor_reduce(
            dst=l_current,
            op=nl.add,
            data=p_local_sink,
            axis=(3,),
        )
        # o_current[:, :, qh] = p_local_broadcast[:, :, qh] * v (v broadcast over
        # the q-head dim).
        for qh in nl.affine_range(q_h_per_k_h):
            nisa.tensor_tensor(
                dst=o_current[:, :, qh],
                data1=p_local_broadcast[:, :, qh],
                data2=v,
                op=nl.multiply,
            )
    else:
        nisa.tensor_copy(dst=m_current, src=qk_psum)
        p_local = nl.ones((1, batch_size, q_h_per_k_h), dtype=kernel_dtype)
        nisa.tensor_copy(dst=l_current, src=p_local)
        for qh in nl.affine_range(q_h_per_k_h):
            nisa.tensor_copy(dst=o_current[:, :, qh], src=v)
