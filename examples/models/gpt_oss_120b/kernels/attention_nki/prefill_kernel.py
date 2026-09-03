"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Migrated from the legacy ``neuronxcc.nki`` API to standalone ``nki`` (beta-3):
- ``par_dim`` / ``nl.mgrid`` removed (leading plain int is the partition dim; use
  explicit slices instead of mgrid).
- ``@nki.compiler.skip_middle_end_transformations`` removed; plain ``@nki.jit``.
- ``nisa`` ops are ``dst``-first and return nothing.
"""

import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa

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
    # NOTE: unlike flash_attn_core.transpose_p_local, this variant tolerates a
    # p_local whose free dim carries an extra sink column (KV_TILE_SIZE + 1);
    # only the [0:LARGE_KV_TILE_SIZE] region is transposed.
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


# Sliding-window (sw=128) causal masks, built host-side. They are passed to
# attention_prefill_sw128 as HBM operands: the beta-3 frontend does not evaluate
# numpy inside a kernel body, so the caller constructs them with these helpers
# (Q_TILE_SIZE == B_P_SIZE is a compile-time constant).
SW128_TILE0_MASK = np.tril(np.ones((B_P_SIZE, B_P_SIZE), dtype=np.uint8))
SW128_TILEI_MASK = np.tril(
    np.triu(np.ones((B_P_SIZE, 2 * B_P_SIZE), dtype=np.uint8), k=1),
    k=B_P_SIZE,
)


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


# NOTE: beta-3 forbids inner function definitions inside a kernel; the
# per-tile helpers below are hoisted to module level and take their state
# explicitly.
def _load_k(cur_k_tile, k, batch_id, kv_head_id, k_id, MULTI_BUFFER):
    assert (
        cur_k_tile.dtype == k.dtype
    ), f"Expecting {cur_k_tile.dtype=} matches {k.dtype=}"
    nisa.dma_copy(
        dst=cur_k_tile[:, k_id % MULTI_BUFFER, :],
        src=k[batch_id, kv_head_id, :, k_id, :],
    )


def _process_large_kv_tile(
    kv_tile_id,
    attn_sink,
    i,
    tile_mask,
    mask,
    cur_v_tile,
    cur_k_tile,
    v,
    k,
    q_tile_scaled,
    o_buffer,
    l_buffer,
    m_buffer,
    batch_id,
    kv_head_id,
    q_h_per_k_h,
    num_large_k_tile,
    MULTI_BUFFER,
    kernel_dtype,
    acc_type,
    Q_TILE_SIZE,
    LARGE_KV_TILE_SIZE,
    B_F_SIZE,
    B_D_SIZE,
):
    nisa.dma_copy(
        dst=tile_mask[:, kv_tile_id % MULTI_BUFFER, :],
        src=mask[
            nl.ds(i * Q_TILE_SIZE, Q_TILE_SIZE),
            nl.ds(kv_tile_id * LARGE_KV_TILE_SIZE, LARGE_KV_TILE_SIZE),
        ],
    )
    assert (
        cur_v_tile.dtype == v.dtype
    ), f"Expecting {cur_v_tile.dtype=} matches {v.dtype=}"
    nisa.dma_transpose(
        dst=cur_v_tile[:, kv_tile_id % MULTI_BUFFER, :, :],
        src=v[batch_id, kv_head_id, :, kv_tile_id, :, :],
        axes=(2, 1, 0),
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
        _load_k(cur_k_tile, k, batch_id, kv_head_id, kv_tile_id + 1, MULTI_BUFFER)


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
    acc_type = nl.float32 if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # beta-3 has no user-defined SPMD launch grid (nl.program_id maps to LNC
    # only), so the old external grid=[b, k_h] is now explicit python loops
    # over the compile-time-constant batch and kv-head counts.
    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    n_tile_q = seqlen_q // Q_TILE_SIZE

    if LARGE_KV_TILE_SIZE is None:
        LARGE_KV_TILE_SIZE = min(2048, seqlen_k)
    assert LARGE_KV_TILE_SIZE % B_P_SIZE == 0

    assert (
        seqlen_k % LARGE_KV_TILE_SIZE == 0
    ), f"Need seqlen_k to be divisible by {LARGE_KV_TILE_SIZE} but got {seqlen_k}"

    q_h_per_k_h = h // k_h

    num_large_k_tile = seqlen_k // LARGE_KV_TILE_SIZE
    k = k.reshape((b, k_h, d, num_large_k_tile, LARGE_KV_TILE_SIZE))
    v = v.reshape(
        (b, k_h, d, num_large_k_tile, LARGE_KV_TILE_SIZE // B_P_SIZE, B_P_SIZE)
    )

    for batch_id in range(b):
        for kv_head_id in range(k_h):
            # load and broadcast sink
            sink_sbuf = load_and_broadcast_sink(
                sink,
                kv_head_id,
                k_h,
                q_h_per_k_h,
                Q_TILE_SIZE,
                kernel_dtype,
            )
            # =========== Global Flash Attention accumulators =============== #
            o_buffer = nl.zeros(
                (Q_TILE_SIZE, q_h_per_k_h, num_large_k_tile + 1, B_D_SIZE),
                dtype=acc_type,
            )
            l_buffer = nl.zeros(
                (Q_TILE_SIZE, q_h_per_k_h, num_large_k_tile + 1, 1),
                dtype=acc_type,
            )
            m_buffer = nl.full(
                (Q_TILE_SIZE, q_h_per_k_h, num_large_k_tile + 1, 1),
                NEG_INF,
                dtype=acc_type,
            )
            # ========= Global Flash Attention accumulators END ============= #

            for i in nl.sequential_range(n_tile_q):

                load_tile_size = B_P_SIZE
                MULTI_BUFFER = 2
                cur_k_tile = nl.ndarray(
                    (B_D_SIZE, MULTI_BUFFER, LARGE_KV_TILE_SIZE),
                    dtype=kernel_dtype,
                )
                cur_v_tile = nl.ndarray(
                    (
                        load_tile_size,
                        MULTI_BUFFER,
                        LARGE_KV_TILE_SIZE // load_tile_size,
                        B_D_SIZE,
                    ),
                    dtype=kernel_dtype,
                )
                tile_mask = nl.ndarray(
                    (Q_TILE_SIZE, MULTI_BUFFER, LARGE_KV_TILE_SIZE),
                    dtype=mask.dtype,
                )

                _load_k(cur_k_tile, k, batch_id, kv_head_id, 0, MULTI_BUFFER)
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
                q_tile_scaled = nl.ndarray(
                    (B_D_SIZE, q_h_per_k_h, Q_TILE_SIZE), dtype=kernel_dtype
                )
                nisa.tensor_scalar(
                    dst=q_tile_scaled,
                    data=cur_q_tile,
                    op0=nl.multiply,
                    operand0=softmax_scale,
                )

                # XXX: handle first tile differently to avoid tracing issue,
                # otherwise, sink may never get used
                _process_large_kv_tile(
                    0, sink_sbuf, i, tile_mask, mask, cur_v_tile, cur_k_tile,
                    v, k, q_tile_scaled, o_buffer, l_buffer, m_buffer, batch_id,
                    kv_head_id, q_h_per_k_h, num_large_k_tile, MULTI_BUFFER,
                    kernel_dtype, acc_type, Q_TILE_SIZE, LARGE_KV_TILE_SIZE,
                    B_F_SIZE, B_D_SIZE,
                )
                for j in nl.sequential_range(1, num_large_k_tile):
                    if i * Q_TILE_SIZE >= j * LARGE_KV_TILE_SIZE:
                        _process_large_kv_tile(
                            j, None, i, tile_mask, mask, cur_v_tile, cur_k_tile,
                            v, k, q_tile_scaled, o_buffer, l_buffer, m_buffer,
                            batch_id, kv_head_id, q_h_per_k_h, num_large_k_tile,
                            MULTI_BUFFER, kernel_dtype, acc_type, Q_TILE_SIZE,
                            LARGE_KV_TILE_SIZE, B_F_SIZE, B_D_SIZE,
                        )

                # -------- write output to buffer on HBM ------------ #
                last_tile_idx = i * Q_TILE_SIZE // LARGE_KV_TILE_SIZE + 1
                l_inv = nl.ndarray((Q_TILE_SIZE, q_h_per_k_h, 1), dtype=acc_type)
                nisa.reciprocal(dst=l_inv, data=l_buffer[:, :, last_tile_idx])
                for i_q_h in nl.affine_range(q_h_per_k_h):
                    # l_inv[:, i_q_h] is a (Q, 1) per-partition scalar; broadcast
                    # it over the B_D free dim via tensor_scalar.
                    out = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=kernel_dtype)
                    nisa.tensor_scalar(
                        dst=out,
                        data=o_buffer[:, i_q_h, last_tile_idx],
                        op0=nl.multiply,
                        operand0=l_inv[:, i_q_h],
                    )
                    nl.store(
                        o[
                            batch_id,
                            kv_head_id * q_h_per_k_h + i_q_h,
                            nl.ds(i * Q_TILE_SIZE, Q_TILE_SIZE),
                            :,
                        ],
                        out,
                    )

    return o


@nki.jit
def attention_prefill_sw128(
    q,
    k,
    v,
    sink,
    tile_0_mask,
    tile_i_mask,
    sliding_window=128,
    softmax_scale=None,
    mixed_precision=True,
):
    # beta-3 note: the causal masks are passed as HBM operands (built host-side,
    # see SW128_TILE0_MASK / SW128_TILEI_MASK). The beta-3 frontend does not
    # evaluate numpy (np.tril/np.triu) inside a kernel body, and shared_constant
    # cannot capture an outer numpy array, so an operand is the clean path.
    assert (
        sliding_window == B_P_SIZE
    ), f"Only sliding window size = {B_P_SIZE=} is supported"
    b, d, h, seqlen_q = q.shape
    B_D_SIZE = d
    _, k_h, _, seqlen_k = k.shape
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
    acc_type = nl.float32 if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # beta-3 has no user-defined SPMD launch grid: loop over batch / kv-head.
    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    Q_TILE_SIZE = B_P_SIZE
    n_tile_q = seqlen_q // Q_TILE_SIZE

    assert seqlen_k % B_P_SIZE == 0, f"Need {seqlen_k=} to be divisible by {B_P_SIZE=}"

    q_h_per_k_h = h // k_h

    assert kernel_dtype == k.dtype, f"Expecting {kernel_dtype=} matches {k.dtype=}"
    assert kernel_dtype == v.dtype, f"Expecting {kernel_dtype=} matches {v.dtype=}"
    v = v.reshape((b, k_h, d, seqlen_k // B_P_SIZE, B_P_SIZE))

    tile_0_mask = nl.load(tile_0_mask)
    tile_i_mask = nl.load(tile_i_mask)

    for batch_id in range(b):
        for kv_head_id in range(k_h):
            # load and broadcast sink
            sink_sbuf = load_and_broadcast_sink(
                sink,
                kv_head_id,
                k_h,
                q_h_per_k_h,
                Q_TILE_SIZE,
                kernel_dtype,
            )
            k_sbuf = nl.ndarray(
                (B_D_SIZE, seqlen_k),
                dtype=kernel_dtype,
            )
            nisa.dma_copy(
                dst=k_sbuf[0:B_D_SIZE, 0:seqlen_k],
                src=k[batch_id, kv_head_id, 0:B_D_SIZE, 0:seqlen_k],
            )
            v_sbuf = nl.ndarray(
                (B_P_SIZE, seqlen_k // B_P_SIZE, B_D_SIZE),
                dtype=kernel_dtype,
            )
            nisa.dma_transpose(
                dst=v_sbuf[...],
                src=v[batch_id, kv_head_id, :, :, :],
                axes=(2, 1, 0),
            )

            _handle_q_tile(
                0, tile_0_mask, q, o, k_sbuf, v_sbuf, sink_sbuf, batch_id,
                kv_head_id, q_h_per_k_h, softmax_scale, kernel_dtype, acc_type,
                Q_TILE_SIZE, B_D_SIZE,
            )
            for i in nl.sequential_range(1, n_tile_q):
                _handle_q_tile(
                    i, tile_i_mask, q, o, k_sbuf, v_sbuf, sink_sbuf, batch_id,
                    kv_head_id, q_h_per_k_h, softmax_scale, kernel_dtype,
                    acc_type, Q_TILE_SIZE, B_D_SIZE,
                )

    return o


def _handle_q_tile(
    q_tile_id,
    mask,
    q,
    o,
    k_sbuf,
    v_sbuf,
    sink_sbuf,
    batch_id,
    kv_head_id,
    q_h_per_k_h,
    softmax_scale,
    kernel_dtype,
    acc_type,
    Q_TILE_SIZE,
    B_D_SIZE,
):
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
    q_tile_scaled = nl.ndarray(
        (B_D_SIZE, q_h_per_k_h, Q_TILE_SIZE), dtype=kernel_dtype
    )
    nisa.tensor_scalar(
        dst=q_tile_scaled,
        data=cur_q_tile,
        op0=nl.multiply,
        operand0=softmax_scale,
    )

    qk_res_buf = nl.ndarray(
        (Q_TILE_SIZE, q_h_per_k_h, KV_TILE_SIZE + 1),
        buffer=nl.sbuf,
        dtype=acc_type,
    )
    max_local = nl.ndarray(
        (Q_TILE_SIZE, q_h_per_k_h, 2),
        dtype=acc_type,
    )
    nisa.tensor_copy(dst=max_local[:, :, 1], src=sink_sbuf)
    for i_q_h in nl.affine_range(q_h_per_k_h):
        qk_psum = nl.ndarray(
            (Q_TILE_SIZE, KV_TILE_SIZE), dtype=nl.float32, buffer=nl.psum
        )  # (128, 256)
        # q_tile_scaled[:, i_q_h]: (B_D_SIZE, Q_TILE_SIZE); transpose => q.T @ k
        nisa.nc_matmul(
            dst=qk_psum,
            stationary=q_tile_scaled[:, i_q_h],
            moving=k_sbuf[:, nl.ds(KV_START_POS * B_P_SIZE, KV_TILE_SIZE)],
        )  # (p(128), 512)
        nisa.select_reduce(
            dst=qk_res_buf[:, i_q_h, nl.ds(0, KV_TILE_SIZE)],
            predicate=mask,
            on_true=qk_psum,
            on_false=NEG_INF,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            reduce_res=max_local[:, i_q_h, 0],
            reduce_op=nl.maximum,
        )
    nisa.tensor_copy(dst=qk_res_buf[:, :, KV_TILE_SIZE], src=sink_sbuf)

    # Calculate max of the current tile
    max_ = nl.ndarray((Q_TILE_SIZE, q_h_per_k_h, 1), dtype=acc_type)
    nisa.tensor_reduce(
        dst=max_,
        op=nl.max,
        data=max_local,
        axis=(2,),
    )
    # neg_max is a per-partition scalar used as an activation bias to compute
    # exp(qk - max) without a broadcasting tensor_tensor.
    neg_max = nl.ndarray((Q_TILE_SIZE, q_h_per_k_h, 1), dtype=acc_type)
    nisa.tensor_scalar(dst=neg_max, data=max_, op0=nl.multiply, operand0=-1.0)

    p_local = nl.ndarray(
        (Q_TILE_SIZE, q_h_per_k_h, KV_TILE_SIZE + 1),
        dtype=kernel_dtype,
    )
    for i_q_h in nl.affine_range(q_h_per_k_h):
        nisa.activation(
            dst=p_local[:, i_q_h, :],
            op=nl.exp,
            data=qk_res_buf[:, i_q_h, :],
            bias=neg_max[:, i_q_h],
            scale=1.0,
        )
    ps = nl.sum(p_local, axis=2, dtype=acc_type)

    for i_q_h in nl.affine_range(q_h_per_k_h):
        p_local_transposed = nl.ndarray(
            (B_P_SIZE, KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE),
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
            (Q_TILE_SIZE, B_D_SIZE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for k_i in nl.affine_range(KV_TILE_SIZE // B_P_SIZE):
            nisa.nc_matmul(
                dst=pv_psum,
                stationary=p_local_transposed[
                    :, nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE)
                ],
                moving=v_sbuf[:, KV_START_POS + k_i, :],
                accumulate=True,
            )  # (128, 128) (p(Br), d)
        out = nl.ndarray((Q_TILE_SIZE, B_D_SIZE), dtype=kernel_dtype)
        ps_inv = nl.ndarray((Q_TILE_SIZE, 1), dtype=acc_type)
        nisa.reciprocal(dst=ps_inv, data=ps[:, i_q_h])
        nisa.tensor_scalar(
            dst=out,
            data=pv_psum,
            op0=nl.multiply,
            operand0=ps_inv,
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
