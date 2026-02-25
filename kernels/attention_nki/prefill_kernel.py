"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa

from .flash_attn_core import _flash_attention_core
from .attn_utils import (
    NEG_INF,
    B_P_SIZE,
    B_FMAX_SIZE,
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


def load_and_broadcast_sink(
    sink_hbm,
    kv_head_id,
    k_h,
    q_h_per_k_h,
    Q_TILE_SIZE,
    kernel_dtype,
):
    sink_hbm = sink_hbm.reshape((k_h, 1, q_h_per_k_h))
    sink = nl.ndarray((1, q_h_per_k_h), dtype=kernel_dtype)
    nisa.dma_copy(dst=sink, src=sink_hbm[kv_head_id])
    return nl.broadcast_to(sink, shape=(Q_TILE_SIZE, q_h_per_k_h))


@nki.compiler.skip_middle_end_transformations
@nki.jit
def flash_attn_prefill(
    q,
    k,
    v,
    sink,
    mask,
    softmax_scale=None,
    mixed_precision=True,
    Q_TILE_SIZE=B_P_SIZE,
    LARGE_KV_TILE_SIZE=None,
):
    """
    Flash Attention Forward kernel

    IO tensor layouts:
      - q: shape    (bs, d, n_heads, seq_q)
      - k: shape    (bs, nk_heads, d, seq_k)
      - v: shape    (bs, nv_heads, d, seq_v)
      - mask: shape (seq_q, seq_k)
      - This kernel requires seq_k == seq_v

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype
      - If mixed_precision is True, then all Tensor Engine operation will be
      performed in bfloat16 and accumulation will be performed in float32.
      Otherwise the intermediates will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt
      is set to `true`, if false, we use same precision as input types
      - causal_mask: flag to set causal masking

    Performance Notes:
      For better performance, the kernel is tiled to be of size
      `LARGE_KV_TILE_SIZE`, and Flash attention math techniques are applied in
      unit of `LARGE_KV_TILE_SIZE`. Seqlen that is not divisible by
      `LARGE_KV_TILE_SIZE` is not supported at the moment.

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    B_F_SIZE = B_FMAX_SIZE
    b, d, h, seqlen_q = q.shape
    B_D_SIZE = d
    _, k_h, _, seqlen_k = k.shape
    assert seqlen_k == seqlen_q
    assert tuple(mask.shape) == (seqlen_q, seqlen_k)
    assert tuple(k.shape) == (
        b,
        k_h,
        d,
        seqlen_k,
    ), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} but got {k.shape}"
    assert tuple(v.shape) == (
        b,
        k_h,
        d,
        seqlen_k,
    ), f"Expect shape of V to be {(b, k_h, seqlen_k, d)} but got {v.shape}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
    kernel_dtype = q.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    batch_id = nl.program_id(axis=0)
    kv_head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    n_tile_q = seqlen_q // Q_TILE_SIZE

    if LARGE_KV_TILE_SIZE is None:
        LARGE_KV_TILE_SIZE = min(2048, seqlen_k)
    assert LARGE_KV_TILE_SIZE % B_P_SIZE == 0

    assert (
        seqlen_k % LARGE_KV_TILE_SIZE == 0
    ), f"Need seqlen_k to be divisible by {LARGE_KV_TILE_SIZE} but got {seqlen_k}"

    q_h_per_k_h = h // k_h

    # load and broadcast sink
    sink_sbuf = load_and_broadcast_sink(
        sink,
        kv_head_id,
        k_h,
        q_h_per_k_h,
        Q_TILE_SIZE,
        kernel_dtype,
    )
    num_large_k_tile = seqlen_k // LARGE_KV_TILE_SIZE
    k = k.reshape((b, k_h, d, num_large_k_tile, LARGE_KV_TILE_SIZE))
    v = v.reshape(
        (b, k_h, d, num_large_k_tile, LARGE_KV_TILE_SIZE // B_P_SIZE, B_P_SIZE)
    )
    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.zeros(
        (par_dim(Q_TILE_SIZE), q_h_per_k_h, num_large_k_tile + 1, B_D_SIZE),
        dtype=acc_type,
    )
    l_buffer = nl.zeros(
        (par_dim(Q_TILE_SIZE), q_h_per_k_h, num_large_k_tile + 1, 1), dtype=acc_type
    )
    m_buffer = nl.full(
        (par_dim(Q_TILE_SIZE), q_h_per_k_h, num_large_k_tile + 1, 1),
        NEG_INF,
        dtype=acc_type,
    )
    # =============== Global Flash Attention accumulators END ================== #

    for i in nl.sequential_range(n_tile_q):

        load_tile_size = B_P_SIZE
        MULTI_BUFFER = 2
        cur_k_tile = nl.ndarray(
            (par_dim(B_D_SIZE), MULTI_BUFFER, LARGE_KV_TILE_SIZE),
            dtype=kernel_dtype,
        )
        cur_v_tile = nl.ndarray(
            (
                par_dim(load_tile_size),
                MULTI_BUFFER,
                LARGE_KV_TILE_SIZE // load_tile_size,
                B_D_SIZE,
            ),
            dtype=kernel_dtype,
        )
        tile_mask = nl.ndarray(
            (par_dim(Q_TILE_SIZE), MULTI_BUFFER, LARGE_KV_TILE_SIZE),
            dtype=mask.dtype,
        )

        def load_k(k_id):
            assert (
                cur_k_tile.dtype == k.dtype
            ), f"Expecting {cur_k_tile.dtype=} matches {k.dtype=}"
            cur_k_tile[:, k_id % MULTI_BUFFER, :] = nl.load(
                k[batch_id, kv_head_id, :, k_id, :],
                dtype=cur_k_tile.dtype,
            )

        load_k(0)
        cur_q_tile = nl.ndarray(
            (B_D_SIZE, q_h_per_k_h, Q_TILE_SIZE), dtype=kernel_dtype
        )
        nisa.dma_copy(
            dst=cur_q_tile[...],
            src=q[
                batch_id,
                :,
                nl.ds(kv_head_id * q_h_per_k_h, q_h_per_k_h),
                nl.ds(i * Q_TILE_SIZE, Q_TILE_SIZE),
            ],
        )
        q_tile_scaled = nl.multiply(
            cur_q_tile,
            softmax_scale,
            dtype=kernel_dtype,
        )

        def process_large_kv_tile(kv_tile_id, attn_sink):
            tile_mask[:, kv_tile_id % MULTI_BUFFER, :] = nl.load(
                mask[
                    nl.ds(i * Q_TILE_SIZE, Q_TILE_SIZE),
                    nl.ds(kv_tile_id * LARGE_KV_TILE_SIZE, LARGE_KV_TILE_SIZE),
                ]
            )
            assert (
                cur_v_tile.dtype == v.dtype
            ), f"Expecting {cur_v_tile.dtype=} matches {v.dtype=}"
            cur_v_tile[:, kv_tile_id % MULTI_BUFFER, :, :] = nisa.dma_transpose(
                v[batch_id, kv_head_id, :, kv_tile_id, :, :],
                axes=(2, 1, 0),
                dtype=cur_v_tile.dtype,
            )
            # XXX: prefetch k here makes perf worse
            for i_q_h in nl.affine_range(q_h_per_k_h):
                _flash_attention_core(
                    q_local_tile=q_tile_scaled[:, i_q_h, :],
                    k=cur_k_tile[:, kv_tile_id % MULTI_BUFFER],
                    v=cur_v_tile[:, kv_tile_id % MULTI_BUFFER],
                    sink=None if attn_sink is None else attn_sink[:, i_q_h],
                    tile_mask=tile_mask[:, kv_tile_id % MULTI_BUFFER],
                    o_buffer=o_buffer[:, i_q_h],
                    l_buffer=l_buffer[:, i_q_h],
                    m_buffer=m_buffer[:, i_q_h],
                    q_tile_idx=i,
                    local_k_tile_idx=kv_tile_id,
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    Q_TILE_SIZE=Q_TILE_SIZE,
                    LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )
            if (
                kv_tile_id + 1 < num_large_k_tile
                and i * Q_TILE_SIZE >= (kv_tile_id + 1) * LARGE_KV_TILE_SIZE
            ):
                load_k(kv_tile_id + 1)

        # XXX: handle first tile differently to avoid tracing issue, otherwise,
        # sink may never get used
        process_large_kv_tile(0, sink_sbuf)
        for j in nl.sequential_range(1, num_large_k_tile):
            if i * Q_TILE_SIZE >= j * LARGE_KV_TILE_SIZE:
                process_large_kv_tile(j, None)

        # -------- write output to buffer on HBM ------------ #
        out = nl.ndarray(
            (par_dim(Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=kernel_dtype
        )
        last_tile_idx = i * Q_TILE_SIZE // LARGE_KV_TILE_SIZE + 1
        out[...] = nl.multiply(
            o_buffer[:, :, last_tile_idx],
            1.0 / l_buffer[:, :, last_tile_idx],
            dtype=kernel_dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            nl.store(
                o[
                    batch_id,
                    kv_head_id * q_h_per_k_h + i_q_h,
                    nl.ds(i * Q_TILE_SIZE, Q_TILE_SIZE),
                    :,
                ],
                out[:, i_q_h, :],
            )

    return o


@nki.compiler.skip_middle_end_transformations
@nki.jit
def attention_prefill_sw128(
    q,
    k,
    v,
    sink,
    sliding_window=128,
    softmax_scale=None,
    mixed_precision=True,
):
    assert (
        sliding_window == B_P_SIZE
    ), f"Only sliding window size = {B_P_SIZE=} is supported"
    b, d, h, seqlen_q = q.shape
    assert nl.num_programs(0) == b, f"{nl.num_programs(0)=} {b=}"
    B_D_SIZE = d
    _, k_h, _, seqlen_k = k.shape
    assert nl.num_programs(1) == k_h, f"{nl.num_programs(1)=} {k_h=}"
    assert seqlen_k == seqlen_q
    assert tuple(k.shape) == (
        b,
        k_h,
        d,
        seqlen_k,
    ), f"Expect shape of K to be {(b, k_h, d, seqlen_k)} but got {k.shape}"
    assert tuple(v.shape) == (
        b,
        k_h,
        d,
        seqlen_k,
    ), f"Expect shape of V to be {(b, k_h, seqlen_k, d)} but got {v.shape}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
    kernel_dtype = q.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    batch_id = nl.program_id(axis=0)
    kv_head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    Q_TILE_SIZE = B_P_SIZE
    n_tile_q = seqlen_q // Q_TILE_SIZE

    assert seqlen_k % B_P_SIZE == 0, f"Need {seqlen_k=} to be divisible by {B_P_SIZE=}"

    q_h_per_k_h = h // k_h

    # load and broadcast sink
    sink_sbuf = load_and_broadcast_sink(
        sink,
        kv_head_id,
        k_h,
        q_h_per_k_h,
        Q_TILE_SIZE,
        kernel_dtype,
    )
    v = v.reshape((b, k_h, d, seqlen_k // B_P_SIZE, B_P_SIZE))
    assert kernel_dtype == k.dtype, f"Expecting {kernel_dtype=} matches {k.dtype=}"
    assert kernel_dtype == v.dtype, f"Expecting {kernel_dtype=} matches {v.dtype=}"
    i_p, i_f = nl.mgrid[:B_D_SIZE, :seqlen_k]
    k_sbuf = nl.ndarray(
        (par_dim(B_D_SIZE), seqlen_k),
        dtype=kernel_dtype,
    )
    nisa.dma_copy(
        dst=k_sbuf[i_p, i_f],
        src=k[batch_id, kv_head_id, i_p, i_f],
    )
    v_sbuf = nl.ndarray(
        (par_dim(B_P_SIZE), seqlen_k // B_P_SIZE, B_D_SIZE),
        dtype=kernel_dtype,
    )
    v_sbuf[...] = nisa.dma_transpose(
        v[batch_id, kv_head_id, :, :, :],
        axes=(2, 1, 0),
        dtype=v_sbuf.dtype,
    )

    def handle_q_tile(q_tile_id, mask):
        if q_tile_id == 0:
            KV_TILE_SIZE = B_P_SIZE
            KV_START_POS = 0
        else:
            KV_TILE_SIZE = 2 * B_P_SIZE
            KV_START_POS = q_tile_id - 1
        cur_q_tile = nl.ndarray(
            (B_D_SIZE, q_h_per_k_h, Q_TILE_SIZE), dtype=kernel_dtype
        )
        nisa.dma_copy(
            dst=cur_q_tile[...],
            src=q[
                batch_id,
                :,
                nl.ds(kv_head_id * q_h_per_k_h, q_h_per_k_h),
                nl.ds(q_tile_id * Q_TILE_SIZE, Q_TILE_SIZE),
            ],
        )
        q_tile_scaled = nl.multiply(
            cur_q_tile,
            softmax_scale,
            dtype=kernel_dtype,
        )

        qk_res_buf = nl.ndarray(
            (par_dim(Q_TILE_SIZE), q_h_per_k_h, KV_TILE_SIZE + 1),
            buffer=nl.sbuf,
            dtype=acc_type,
        )
        max_local = nl.ndarray(
            (par_dim(Q_TILE_SIZE), q_h_per_k_h, 2),
            dtype=acc_type,
        )
        max_local[:, :, 1] = nl.copy(sink_sbuf)
        for i_q_h in nl.affine_range(q_h_per_k_h):
            qk_psum = nl.ndarray(
                (par_dim(Q_TILE_SIZE), KV_TILE_SIZE), dtype=np.float32, buffer=nl.psum
            )  # (128, 256)
            qk_psum[:, :] = nl.matmul(
                q_tile_scaled[:, i_q_h],
                k_sbuf[:, nl.ds(KV_START_POS * B_P_SIZE, KV_TILE_SIZE)],
                transpose_x=True,
            )  # (p(128), 512)
            nisa.select_reduce(
                dst=qk_res_buf[:, i_q_h, nl.ds(0, KV_TILE_SIZE)],
                predicate=mask,
                on_true=qk_psum,
                on_false=NEG_INF,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                reduce_res=max_local[:, i_q_h, 0],
                reduce_op=np.max,
            )
            # qk_res_buf[:, i_q_h, :] = nisa.range_select(
            #     on_true_tile=qk_psum,
            #     comp_op0=np.greater_equal,
            #     comp_op1=np.less,
            #     bound0=bound0_tile,
            #     bound1=bound1_tile,
            #     reduce_cmd=nisa.reduce_cmd.reset_reduce,
            #     reduce_op=np.max,
            #     reduce_res=max_local[:, i_q_h, 0],
            #     range_start=0,
            #     on_false_value=nl.fp32.min,
            # )
        qk_res_buf[:, :, KV_TILE_SIZE] = nl.copy(sink_sbuf)

        # Calculate max of the current tile
        max_ = nl.ndarray((Q_TILE_SIZE, q_h_per_k_h, 1), dtype=acc_type)
        max_[...] = nisa.tensor_reduce(
            np.max,
            max_local,
            axis=(2,),
            dtype=acc_type,
        )

        p_local = nl.ndarray(
            (par_dim(Q_TILE_SIZE), q_h_per_k_h, KV_TILE_SIZE + 1),
            dtype=kernel_dtype,
        )
        p_local[:, :, :] = nisa.activation(
            np.exp,
            qk_res_buf - max_,
            scale=1.0,
            dtype=kernel_dtype,
        )
        ps = nl.ndarray((Q_TILE_SIZE, q_h_per_k_h, 1), dtype=acc_type)
        ps = nl.sum(p_local, axis=2, dtype=acc_type)

        for i_q_h in nl.affine_range(q_h_per_k_h):
            p_local_transposed = nl.ndarray(
                (par_dim(B_P_SIZE), KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
                dtype=kernel_dtype,
            )
            transpose_p_local(
                p_local_transposed=p_local_transposed,
                p_local=p_local[:, i_q_h, :],
                Q_TILE_SIZE=Q_TILE_SIZE,
                LARGE_KV_TILE_SIZE=KV_TILE_SIZE,
                B_F_SIZE=KV_TILE_SIZE,
                q_idx=0,
                kv_idx=0,
            )

            pv_psum = nl.zeros(
                (par_dim(Q_TILE_SIZE), B_D_SIZE),
                dtype=np.float32,
                buffer=nl.psum,
            )
            for k_i in nl.affine_range(KV_TILE_SIZE // B_P_SIZE):
                pv_psum[:, :] += nl.matmul(
                    p_local_transposed[:, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)],
                    v_sbuf[:, KV_START_POS + k_i, :],
                    transpose_x=True,
                )  # (128, 128) (p(Br), d)
            out = nl.ndarray((par_dim(Q_TILE_SIZE), B_D_SIZE), dtype=kernel_dtype)
            out[...] = nl.multiply(
                pv_psum,
                1.0 / ps[:, i_q_h],
                dtype=kernel_dtype,
            )

            # -------- write output to buffer on HBM ------------ #
            nl.store(
                o[
                    batch_id,
                    kv_head_id * q_h_per_k_h + i_q_h,
                    nl.ds(q_tile_id * Q_TILE_SIZE, Q_TILE_SIZE),
                    :,
                ],
                out,
            )

    tile_0_mask = nl.load(
        nl.shared_constant(
            np.tril(np.ones((Q_TILE_SIZE, Q_TILE_SIZE), dtype=np.uint8)),
            dtype=nl.uint8,
        )
    )
    tile_i_mask = nl.load(
        nl.shared_constant(
            np.tril(
                np.triu(np.ones((Q_TILE_SIZE, 2 * Q_TILE_SIZE), dtype=np.uint8), k=1),
                k=Q_TILE_SIZE,
            ),
            dtype=nl.uint8,
        )
    )
    handle_q_tile(0, mask=tile_0_mask)
    for i in nl.sequential_range(1, n_tile_q):
        handle_q_tile(i, tile_i_mask)

    return o
