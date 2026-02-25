"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

"""

import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim

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
                    par_dim(CONTRACTION_TILE_SIZE),
                    B_F_SIZE // CONTRACTION_TILE_SIZE * Q_TILE_SIZE,
                ),
                buffer=nl.psum,
                dtype=np.float32 if is_nc_gen2 else p_local.dtype,
            )
            for j in nl.affine_range(B_F_SIZE // CONTRACTION_TILE_SIZE):

                j_128_slice = nl.ds(j * Q_TILE_SIZE, Q_TILE_SIZE)
                i_j_128_slice = nl.ds(
                    i * B_F_SIZE + j * CONTRACTION_TILE_SIZE, CONTRACTION_TILE_SIZE
                )
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
                    p_local[:, i_j_128_slice],
                    engine=nisa.tensor_engine,
                )
            p_local_transposed[
                :,
                nl.ds(
                    i * (B_F_SIZE // CONTRACTION_TILE_SIZE * Q_TILE_SIZE),
                    (B_F_SIZE // CONTRACTION_TILE_SIZE * Q_TILE_SIZE),
                ),
            ] = nl.copy(p_local_t_tmp, dtype=p_local_transposed.dtype)


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
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    if sink is not None:
        max_local = nl.full(
            (par_dim(Q_TILE_SIZE), num_k_tile_per_large_tile + 1),
            NEG_INF,
            dtype=acc_type,
        )
        max_local[:, num_k_tile_per_large_tile] = nl.copy(sink)
    else:
        max_local = nl.full(
            (par_dim(Q_TILE_SIZE), num_k_tile_per_large_tile),
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
                (par_dim(Q_TILE_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum
            )  # (128, 512)
            qk_psum[:, :] = nl.matmul(
                q_local_tile, k[:, k_i_b_f_slice], transpose_x=True
            )  # (p(128), 512)
            nisa.select_reduce(
                dst=qk_res_buf[:, k_i_b_f_slice],
                predicate=tile_mask[:, k_i_b_f_slice],
                on_true=qk_psum[:, nl.ds(0, B_F_SIZE)],
                on_false=NEG_INF,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                reduce_res=max_local[:, k_i],
                reduce_op=np.max,
            )

    # Calculate max of the current tile
    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1,),
        dtype=acc_type,
        negate=False,
    )

    o_previous_scaled = nl.ndarray(
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=o_buffer.dtype,
    )

    m_previous = m_buffer[:, local_k_tile_idx]
    m_current_neg = nl.ndarray((par_dim(Q_TILE_SIZE), 1), dtype=acc_type)
    m_current_neg[...] = nisa.tensor_scalar(
        max_,
        nl.maximum,
        m_previous,
        op1=nl.multiply,
        operand1=-1,
    )

    p_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    REDUCTION_TILE = B_F_SIZE

    if sink is not None:
        p_partial_sum = nl.zeros(
            (par_dim(Q_TILE_SIZE), (LARGE_KV_TILE_SIZE // REDUCTION_TILE) + 1),
            dtype=acc_type,
        )
        p_partial_sum[:, LARGE_KV_TILE_SIZE // REDUCTION_TILE] = nisa.activation(
            np.exp,
            sink,
            bias=m_current_neg,
            scale=1.0,
            dtype=acc_type,
        )
    else:
        p_partial_sum = nl.zeros(
            (par_dim(Q_TILE_SIZE), LARGE_KV_TILE_SIZE // REDUCTION_TILE),
            dtype=acc_type,
        )

    for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
        if (
            q_tile_idx * Q_TILE_SIZE
            >= local_k_tile_idx * LARGE_KV_TILE_SIZE + k_r_i * REDUCTION_TILE
        ):
            k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)
            p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
                np.exp,
                qk_res_buf[:, k_r_i_reduce_slice],
                bias=m_current_neg,
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=p_partial_sum[:, k_r_i],
                dtype=kernel_dtype,
            )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type)

    p_local_transposed = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
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
        (par_dim(Q_TILE_SIZE), B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)],
            v[:, k_i, :],
            transpose_x=True,
            mask=(
                q_tile_idx * Q_TILE_SIZE
                >= local_k_tile_idx * LARGE_KV_TILE_SIZE + k_i * B_P_SIZE
            ),
        )  # (128, 128) (p(Br), d)

    # Compute scaling factor
    alpha = nisa.activation(
        np.exp,
        m_previous,
        bias=m_current_neg,
        scale=1.0,
    )

    m_buffer[:, local_k_tile_idx + 1] = nisa.activation(
        nl.copy,
        m_current_neg,
        scale=-1.0,
    )
    o_previous_scaled[...] = nl.multiply(o_buffer[:, local_k_tile_idx], alpha)
    o_buffer[:, local_k_tile_idx + 1] = nl.add(o_previous_scaled, pv_psum)

    l_prev = l_buffer[:, local_k_tile_idx] * alpha
    l_buffer[:, local_k_tile_idx + 1] = l_prev + ps


def partition_broadcast_fp32(src, out_psum):
    assert src.shape[1:] == out_psum.shape[1:]
    assert src.dtype in [nl.bfloat16, nl.float32]
    assert out_psum.dtype == nl.float32
    p_size = out_psum.shape[0]
    ones = nl.ones((1, p_size), dtype=src.dtype)
    out_psum[...] = nisa.nc_matmul(
        ones,
        src,
        is_stationary_onezero=True,
    )


def transpose_with_matmul(src, out_psum, identity):
    assert src.dtype == out_psum.dtype == identity.dtype
    out_psum[...] = nisa.nc_matmul(
        src,
        identity,
        is_moving_onezero=True,
        is_transpose=True,
    )


def _flash_attention_core_kq_matmul(
    *,
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
    identity_m1,
    identity_m2,
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

    def load_k_cache(k_id):
        for batch_id in nl.affine_range(batch_size):
            nisa.dma_copy(
                dst=k_load_buffer[:, k_id % MULTI_BUFFER, batch_id, :],
                src=cache_k[batch_id, kv_head_id, :, k_id],
                dge_mode=nisa.dge_mode.swdge,
            )

    def load_v_cache(v_id):
        for batch_id in nl.affine_range(batch_size):
            nisa.dma_copy(
                dst=v_load_buffer[:, v_id % MULTI_BUFFER, batch_id, :, :],
                src=cache_v[batch_id, kv_head_id, v_id],
                dge_mode=nisa.dge_mode.swdge,
            )

    # load current v
    load_v_cache(large_kv_tile_id)

    # load next k
    if large_kv_tile_id + 1 < num_large_kv_tiles:
        load_k_cache(large_kv_tile_id + 1)

    # Calculate KQ and mask
    kq_res_buf = nl.full(
        (par_dim(B_P_SIZE), batch_size, num_k_tiles, q_h_per_k_h),
        NEG_INF,
        dtype=acc_type,
    )
    for batch_id in nl.affine_range(batch_size):
        kq_res_psum = nl.ndarray(
            (par_dim(B_P_SIZE), num_k_tiles, q_h_per_k_h),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for k_i in nl.affine_range(num_k_tiles):
            kq_res_psum[:, k_i, :] = nisa.nc_matmul(
                k[:, batch_id, :, k_i],
                q_local_tile[:, batch_id, :],
            )  # (p(128), Q_TILE_SIZE)

        nisa.tensor_copy_predicated(
            src=kq_res_psum,
            dst=kq_res_buf[:, batch_id],
            predicate=tile_mask[:, batch_id],
        )

    # Calculate max of the current tile (cascade reduction)
    max_partial = nl.ndarray(
        (par_dim(B_P_SIZE), batch_size, q_h_per_k_h),
        dtype=acc_type,
    )
    max_partial[:, :] = nisa.tensor_reduce(
        np.max,
        kq_res_buf,
        axis=(2,),
        dtype=acc_type,
        negate=False,
    )
    max_partial_transposed = nl.ndarray(
        (par_dim(q_h_per_k_h * batch_size), B_P_SIZE), dtype=acc_type, buffer=nl.psum
    )
    transpose_with_matmul(
        max_partial.reshape((B_P_SIZE, batch_size * q_h_per_k_h)),
        max_partial_transposed,
        identity=identity_m1,
    )
    max_ = nisa.tensor_reduce(
        np.max,
        max_partial_transposed,
        axis=(1,),
        dtype=acc_type,
        negate=False,
        keepdims=True,
    )  # (batch_size * q_h_per_k_h, 1)

    # Calculate max
    max_transposed_psum = nl.ndarray(
        (par_dim(1), batch_size * q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    transpose_with_matmul(max_, max_transposed_psum, identity=identity_m2)
    m_previous = m_buffer_sbuf[:, large_kv_tile_id]
    m_current = m_buffer_sbuf[:, large_kv_tile_id + 1]
    m_current[...] = nisa.tensor_tensor(
        m_previous,
        max_transposed_psum,
        np.maximum,
    )  # (128,1)

    # Compute scaling factor and broadcast
    bias = nl.zeros((par_dim(1), 1), dtype=m_previous.dtype)
    alpha = nisa.activation(
        np.exp,
        m_previous - m_current,
        scale=1.0,
        bias=bias,
    )
    alpha_broadcasted = nl.ndarray(
        (par_dim(B_D_SIZE), batch_size * q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    partition_broadcast_fp32(alpha, alpha_broadcasted)

    # Rescale previous output
    o_previous_scaled = nl.ndarray(
        (par_dim(B_D_SIZE), batch_size * q_h_per_k_h),
        dtype=o_buffer_sbuf.dtype,
    )
    o_previous_scaled[...] = nl.multiply(
        o_buffer_sbuf[:, large_kv_tile_id],
        alpha_broadcasted,
    )

    max_broadcasted = nl.ndarray(
        (par_dim(B_P_SIZE), batch_size, 1, q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    max_broadcasted_2d_view = max_broadcasted.reshape(
        (B_P_SIZE, batch_size * q_h_per_k_h)
    )
    partition_broadcast_fp32(m_current, max_broadcasted_2d_view)

    # Calculate unnormed Softmax(P) = exp(P - max)
    p_local = nl.ndarray(
        (par_dim(B_P_SIZE), batch_size, num_k_tiles, q_h_per_k_h),
        dtype=kernel_dtype,
    )
    bias = nl.zeros((par_dim(B_P_SIZE), 1), dtype=max_broadcasted.dtype)
    p_local[...] = nisa.activation(
        np.exp,
        kq_res_buf - max_broadcasted,
        scale=1.0,
        bias=bias,
        dtype=kernel_dtype,
    )

    # Calculate PV
    v = v_load_buffer[:, large_kv_tile_id % MULTI_BUFFER]
    pv_psum = nl.zeros(
        (par_dim(B_D_SIZE), batch_size, q_h_per_k_h),
        dtype=np.float32,
        buffer=nl.psum,
    )
    for batch_id in nl.affine_range(batch_size):
        for k_i in nl.affine_range(num_k_tiles):
            pv_psum[:, batch_id, :] += nisa.nc_matmul(
                v[:, batch_id, k_i, :],
                p_local[:, batch_id, k_i],
            )

    # Calculate SumExp
    # XXX: batch_size * num_k_tiles * q_h_per_k_h <= 512 (psum_fmax) may not
    # hold for large tile
    assert num_k_tiles * q_h_per_k_h <= 512
    sumexp_ones = nl.ones((B_P_SIZE, 1), dtype=p_local.dtype)
    ps = nl.ndarray(
        (par_dim(1), batch_size, q_h_per_k_h),
        dtype=acc_type,
    )
    for batch_id in nl.affine_range(batch_size):
        # sum along partition first
        ps_partial = nl.ndarray(
            (par_dim(1), num_k_tiles, q_h_per_k_h),
            dtype=acc_type,
            buffer=nl.psum,
        )
        ps_partial_reshape = ps_partial.reshape((1, num_k_tiles * q_h_per_k_h))
        ps_partial_reshape[...] = nisa.nc_matmul(
            sumexp_ones,
            p_local.reshape(
                (
                    B_P_SIZE,
                    batch_size,
                    num_k_tiles * q_h_per_k_h,
                )
            )[:, batch_id],
            is_stationary_onezero=True,
        )
        ps[:, batch_id, :] = nisa.tensor_reduce(
            nl.add,
            ps_partial,
            axis=(1,),
            dtype=acc_type,
            negate=False,
            keepdims=False,
        )

    # Update output buffer
    o_buffer_sbuf[:, large_kv_tile_id + 1, :] = nl.add(
        o_previous_scaled,
        pv_psum.reshape((B_D_SIZE, batch_size * q_h_per_k_h)),
    )
    l_prev_scaled = l_buffer_sbuf[:, large_kv_tile_id] * alpha
    l_buffer_sbuf[:, large_kv_tile_id + 1] = l_prev_scaled + ps.reshape(
        (1, batch_size * q_h_per_k_h)
    )


def _active_attention_core_batched(
    *,
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
    q: (par_dim(B_D_SIZE), q_h_per_k_h, batch_size)
    k: (par_dim(B_D_SIZE), batch_size)
    v: (par_dim(B_D_SIZE), batch_size)
    The results are stored in the following three buffers
    o_buffer_sbuf: (B_D_SIZE, batch_size * q_h_per_k_h)
    l_buffer_sbuf: (1, batch_size * q_h_per_k_h)
    m_buffer_sbuf: (1, batch_size * q_h_per_k_h)
    """
    B_D_SIZE, batch_size, q_h_per_k_h = q.shape
    # calculate qk
    ones = nl.ones((par_dim(B_D_SIZE), 1), dtype=acc_type)
    qk_mul = nl.ndarray(
        (par_dim(B_D_SIZE), batch_size, q_h_per_k_h),
        dtype=acc_type,
    )
    qk_mul[...] = nisa.tensor_tensor(
        q,
        k.reshape((B_D_SIZE, batch_size, 1)),
        nl.multiply,
        dtype=qk_mul.dtype,
    )
    qk_psum = nl.ndarray(
        (par_dim(1), batch_size, q_h_per_k_h),
        buffer=nl.psum,
        dtype=np.float32,
    )
    qk_psum[...] = nisa.nc_matmul(
        ones,
        qk_mul,
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
        # load sink
        qk_sbuf[:, :, :, 1] = nl.load(sink.reshape((1, 1, q_h_per_k_h)), dtype=acc_type)
        qk_sbuf[:, :, :, 0] = nl.copy(qk_psum)
        m_current[...] = nisa.tensor_reduce(np.max, qk_sbuf, axis=(3,), dtype=acc_type)
        bias = nl.zeros((par_dim(1), 1), dtype=m_buffer_sbuf.dtype)
        p_local_sink = nl.ndarray(
            (par_dim(1), batch_size, q_h_per_k_h, 2),
            dtype=kernel_dtype,
        )
        p_local_sink[:, :, :, :] = nisa.activation(
            np.exp,
            qk_sbuf - m_current,
            scale=1.0,
            bias=bias,
            dtype=kernel_dtype,
        )
        p_local = p_local_sink[:, :, :, 0]
        p_local_broadcast = nl.ndarray(
            (B_D_SIZE, batch_size, q_h_per_k_h),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        partition_broadcast_fp32(p_local, p_local_broadcast)
        l_current[...] = nisa.tensor_reduce(
            np.add,
            p_local_sink,
            axis=(3,),
            dtype=acc_type,
        )
        o_current[...] = nisa.tensor_tensor(
            p_local_broadcast,
            v.reshape((B_D_SIZE, batch_size, 1)),
            nl.multiply,
            dtype=acc_type,
        )
    else:
        m_current[...] = nl.copy(qk_psum)
        p_local = nl.ones((par_dim(1), batch_size, q_h_per_k_h), dtype=kernel_dtype)
        l_current[...] = nl.copy(p_local)
        o_current[...] = nl.copy(v.reshape((B_D_SIZE, batch_size, 1)), dtype=acc_type)
