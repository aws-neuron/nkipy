# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
kernels - Builtin high performance attention kernels
"""
# ruff: noqa

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.typing import scalar
from torch_to_nkipy.utils.nki import NKIOpRegistry
from vllm.utils import cdiv

from .arith import _cumsum, _floor_divide, _remainder
from .flash_attn_core import (
    _active_attention_core_batched,
    _flash_attention_core,
    _flash_attention_core_kq_matmul,
)
from .paged_cache import (
    load_kv_tile_from_cache,
    transform_block_tables_for_indirect_load,
)
from .utils import (
    PF_transpose_with_PE,
    broadcast_partition_with_PE,
    ceil_div,
    create_identity_for_transpose,
    is_multiple_of,
    is_power_of_2,
    load_indices_for_loop_step,
    pad_to_multiple,
    transform_to_vector_dge_layout,
)

NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
EPSILON = 0.7 * (1.0 / abs(NEG_INF))


def _br(x, out):
    # Broadcast weight along first axis to match tensor shape
    assert x.shape[-1] == out.shape[-1], (
        f"invalid input shapes: {x.shape=}, {out.shape=}"
    )
    src_p, dst_p = x.shape[0], out.shape[0]
    step_size = out.shape[-1]
    n_iters = cdiv(dst_p // src_p, 32)
    br_size = min(32, dst_p // src_p)
    x_re = x.reshape(shape=(src_p, step_size))
    for j in nl.affine_range(n_iters):
        i_p = nl.arange(br_size)[:, None, None] + (j * br_size)
        i_f = nl.arange(step_size)[None, None, :]
        out[i_p, i_f] = nl.broadcast_to(x_re, shape=(br_size, step_size))


def build_prefill_last_tile_indices(
    prefill_last_tile_indices_hbm,
    iota_a,
    is_prefill_seq,
    total_tiles_per_seq,
    num_prefill_seqs,
    num_q_tiles,
    active_q_tile_size,
):
    # iota_a[i_p, i_f]: [1 2 3 4 5 6 7 8]
    #   is_prefill_seq: [1 0 1 0 0 1 1 0]
    #   copy_predicate: [7 -1 5 -1 -1 2 1 -1]
    #      num_q_tiles: [2 0 1 0 0 2 1 0]
    #        num_tiles: [2 0 1 0 0 2 1 0]
    # indices:          [0 2 5 6 1 1 1 1]
    # expected:         [2 1 2 1 0 0 0 0]
    # assert is_multiple_of(max_num_seqs, 8), f"invalid {max_num_seqs=}, expected to be multiple of 8"
    _, max_num_seqs = is_prefill_seq.shape
    total_prefill_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    cu_prefill_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    max_vals = nl.ndarray((1, max_num_seqs), dtype=nl.int32)
    addr = nl.ndarray((max_num_seqs, 1), dtype=nl.int32)
    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(max_num_seqs)[None, :]
    i_t = nl.arange(active_q_tile_size)[None, :]
    num_prefill_seqs_imm = nl.copy(num_prefill_seqs, dtype=nl.float32)

    cu_prefill_seq = nl.ndarray((1, max_num_seqs), dtype=nl.int32)
    cu_prefill_seq[...] = _cumsum(is_prefill_seq)
    # nl.device_print("flash_pa_with_schedule.py: cu_prefill_seq=", cu_prefill_seq)
    nisa.tensor_copy_predicated(
        src=int(NEG_INF),
        dst=cu_prefill_seq,
        predicate=is_prefill_seq,
        reverse_pred=True,
    )
    addr[...] = nl.copy(
        nisa.nc_transpose(
            nl.subtract(cu_prefill_seq, 1, dtype=nl.float32), dtype=nl.float32
        ),
        dtype=nl.int32,
    )
    total_tiles_per_seq_t = nl.copy(
        nisa.nc_transpose(nl.copy(total_tiles_per_seq, dtype=nl.float32)),
        dtype=nl.int32,
    )
    ipp, iff = nl.mgrid[0:max_num_seqs, 0:1]
    nisa.dma_copy(
        dst=prefill_last_tile_indices_hbm[addr[ipp, iff]],
        src=total_tiles_per_seq_t,
        oob_mode=oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
    )
    prefill_last_tile_indices_sbuf = nl.ndarray(
        prefill_last_tile_indices_hbm[ipp, iff].shape, dtype=nl.float32
    )
    nisa.dma_copy(
        dst=prefill_last_tile_indices_sbuf,
        src=prefill_last_tile_indices_hbm[ipp, iff],
        dge_mode=nisa.dge_mode.swdge,
    )
    num_tiles = nisa.nc_transpose(prefill_last_tile_indices_sbuf)
    cu_prefill_tiles_per_seq[i_p, i_f + 1] = _cumsum(num_tiles)
    prefill_valid_mask = nisa.tensor_scalar(
        iota_a[i_p, i_f], nl.less, num_prefill_seqs_imm, dtype=nl.bool_
    )
    valid_last_tile_indices_sbuf = nl.where(
        prefill_valid_mask,
        nl.subtract(cu_prefill_tiles_per_seq[i_p, i_f + 1], 1.0),
        cu_prefill_tiles_per_seq[0, max_num_seqs],
        dtype=nl.int32,
    )
    # nl.device_print("cu_prefill_seq=", cu_prefill_seq)
    # nl.device_print("addr=", addr)
    # nl.device_print("prefill_last_tile_indices_sbuf=", prefill_last_tile_indices_sbuf)
    # nl.device_print("num_q_tiles=", num_q_tiles)
    # nl.device_print("num_tiles=", num_tiles)
    # nl.device_print("cu_prefill_tiles_per_seq=", cu_prefill_tiles_per_seq)
    # nl.device_print("prefill_valid_mask=", prefill_valid_mask)
    # nl.device_print("valid_last_tile_indices_sbuf=", valid_last_tile_indices_sbuf)
    nisa.dma_copy(
        dst=prefill_last_tile_indices_hbm.reshape(
            [
                active_q_tile_size,
            ]
        )[i_t],
        src=valid_last_tile_indices_sbuf[0, i_t],
        dge_mode=nisa.dge_mode.swdge,
    )


def build_decode_last_tile_indices(
    decode_last_tile_indices_hbm,
    iota_a,
    is_decode_seq,
    total_tiles_per_seq,
    num_decode_seqs,
):
    # iota_a[i_p, i_f]: [1 2 3 4 5 6 7 8]
    #    is_decode_seq: [1 0 1 0 0 1 1 0]
    #         multiply: [1 0 3 0 0 6 7 0] -> [7 -1 5 -1 -1 2 1 -1]
    #        num_tiles: [2 0 1 0 0 2 1 0]
    # indices:          [0 2 5 6 1 1 1 1]
    # expected:         [2 1 2 1 0 0 0 0]
    # assert is_multiple_of(max_num_seqs, 8), f"invalid {max_num_seqs=}, expected to be multiple of 8"
    _, max_num_seqs = is_decode_seq.shape
    total_decode_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    cu_decode_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    max_vals = nl.ndarray((1, max_num_seqs), dtype=nl.int32)
    addr = nl.ndarray((max_num_seqs, 1), dtype=nl.int32)
    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(max_num_seqs)[None, :]
    num_decode_seqs_imm = nl.copy(num_decode_seqs, dtype=nl.float32)

    cu_decode_seq = nl.ndarray((1, max_num_seqs), dtype=nl.int32)
    cu_decode_seq[...] = _cumsum(is_decode_seq)
    # nl.device_print("flash_pa_with_schedule.py: cu_decode_seq=", cu_decode_seq)
    nisa.tensor_copy_predicated(
        src=int(NEG_INF), dst=cu_decode_seq, predicate=is_decode_seq, reverse_pred=True
    )
    addr[...] = nl.copy(
        nisa.nc_transpose(
            nl.subtract(cu_decode_seq, 1, dtype=nl.float32), dtype=nl.float32
        ),
        dtype=nl.int32,
    )
    total_tiles_per_seq_t = nl.copy(
        nisa.nc_transpose(nl.copy(total_tiles_per_seq, dtype=nl.float32)),
        dtype=nl.int32,
    )
    ipp, iff = nl.mgrid[0:max_num_seqs, 0:1]
    nisa.dma_copy(
        dst=decode_last_tile_indices_hbm[addr[ipp, iff]],
        src=total_tiles_per_seq_t,
        oob_mode=oob_mode.skip,
        dge_mode=nisa.dge_mode.swdge,
    )
    decode_last_tile_indices_sbuf = nl.ndarray(
        decode_last_tile_indices_hbm[ipp, iff].shape, dtype=nl.float32
    )
    nisa.dma_copy(
        dst=decode_last_tile_indices_sbuf,
        src=decode_last_tile_indices_hbm[ipp, iff],
        dge_mode=nisa.dge_mode.swdge,
    )
    num_tiles = nisa.nc_transpose(decode_last_tile_indices_sbuf)
    cu_decode_tiles_per_seq[i_p, i_f + 1] = _cumsum(num_tiles)
    decode_valid_mask = nisa.tensor_scalar(
        iota_a[i_p, i_f], nl.less, num_decode_seqs_imm, dtype=nl.bool_
    )
    valid_last_tile_indices_sbuf = nl.where(
        decode_valid_mask,
        nl.subtract(cu_decode_tiles_per_seq[i_p, i_f + 1], 1.0),
        cu_decode_tiles_per_seq[0, max_num_seqs],
        dtype=nl.int32,
    )
    nisa.dma_copy(
        dst=decode_last_tile_indices_hbm.reshape(
            [
                max_num_seqs,
            ]
        )[i_f],
        src=valid_last_tile_indices_sbuf,
        dge_mode=nisa.dge_mode.swdge,
    )


def allocate_prefill_accum_buffers(
    seqlen_q,
    INNER_Q_TILE_SIZE,
    q_h_per_k_h,
    B_D_SIZE,
    acc_type,
):
    # =============== Global Flash Attention accumulators ====================== #
    olm_buffer = nl.ndarray(
        (seqlen_q, q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
        buffer=nl.hbm,
    )

    for i in nl.affine_range(ceil_div(seqlen_q, INNER_Q_TILE_SIZE)):
        i_x, i_y, i_z = nl.mgrid[:INNER_Q_TILE_SIZE, :q_h_per_k_h, : B_D_SIZE + 1]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(dst=olm_buffer[i_x, i_y, i_z], value=0.0, mask=(i_x < seqlen_q))
        i_x, i_y, i_z = nl.mgrid[
            :INNER_Q_TILE_SIZE, :q_h_per_k_h, B_D_SIZE + 1 : B_D_SIZE + 2
        ]
        i_x = i_x + i * INNER_Q_TILE_SIZE
        nl.store(dst=olm_buffer[i_x, i_y, i_z], value=NEG_INF, mask=(i_x < seqlen_q))

    # =============== Global Flash Attention accumulators END ================== #
    return (olm_buffer,)


def allocate_decode_accum_buffers(
    MAX_NUM_TILE,
    q_h_per_k_h,
    B_D_SIZE,
    acc_type,
):
    # =============== Global Flash Attention accumulators ====================== #
    olm_buffer = nl.ndarray(
        (par_dim(MAX_NUM_TILE), q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
        buffer=nl.hbm,
    )
    # =============== Global Flash Attention accumulators END ================== #
    return (olm_buffer,)


def transpose_broadcast_q_pred(q_update_pred_sbuf, partition_size):
    B_P_SIZE = nl.tile_size.pmax
    assert partition_size <= B_P_SIZE
    num_tiles_unrolled = q_update_pred_sbuf.shape[0]
    q_update_pred_broadcast = nl.ndarray(
        (num_tiles_unrolled, partition_size), dtype=nl.bfloat16
    )
    q_update_pred_broadcast[:, :] = nl.copy(
        q_update_pred_sbuf.broadcast_to((num_tiles_unrolled, partition_size)),
        dtype=nl.bfloat16,
    )
    if nisa.get_nc_version() == nisa.nc_version.gen2:
        q_update_pred_broadcast_transposed = nl.ndarray(
            (partition_size, num_tiles_unrolled),
            dtype=nl.uint8,
        )
        PF_transpose_with_PE(
            src=q_update_pred_broadcast,
            out=q_update_pred_broadcast_transposed,
            out_in_psum=False,
        )
    else:
        q_update_pred_transposed = nl.ndarray(
            (partition_size, num_tiles_unrolled),
            dtype=nl.bfloat16,
            buffer=nl.psum,
        )
        q_update_pred_transposed[...] = nisa.nc_transpose(
            q_update_pred_broadcast,
            engine=nisa.tensor_engine,
        )
        q_update_pred_broadcast_transposed = nl.copy(
            q_update_pred_transposed, dtype=nl.uint8
        )
    return q_update_pred_broadcast_transposed


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    enable_out_of_bound_check=False,
    experimental_flags="experimental-native-scalar-support",
)  # , experimental-local-tensor-parent")
def prepare_sequence_offsets(
    seqused_q,
    seqused_k,
    num_seqs,
    identity_for_transpose_p_hbm,
    max_num_tiles,
    num_queries,
    q_tile_size,
    large_k_tile_size,
):
    """
    Prepares sequence IDs and offsets for each block tile in paged attention.

    Args:
        seqused_q (ndarray[int32]): Length of each query sequence [num_seqs]
        seqused_k (ndarray[int32]): Length of each key/value sequence [num_seqs]
        num_seqs (ndarray[int32]): Number of sequences in the batch [1]
        q_tile_size (int): Size of each query tile partition
        large_k_tile_size (int): Size of each key tile partition

    Returns:
        prefill_seq_ids (ndarray[int32]): Sequence IDs per block tile [max_num_tiles]
        q_offsets (ndarray[int32]): Query offsets for each block tile [max_num_tiles]
        k_offsets (ndarray[int32]): Key offsets for each block tile [max_num_tiles]
        dynamic_loop_trip_count (ndarray[int32]): Number of tiles needed [1]
        decode_last_tile_indices (ndarray[int32]): Last tile index per sequence [max_num_seqs, 1]
    """
    pmax = nl.tile_size.pmax
    max_num_seqs = seqused_k.shape[-1]
    active_q_tile_size = (
        max_num_seqs  # TODO: experiment with =ceil_div(num_queries, q_tile_size)
    )
    # assert max_num_seqs <= active_q_tile_size, f"invalid {active_q_tile_size=}"
    # FIXME: check invalid shapes
    # assert seqused_q.shape[-1] == seqused_k.shape[-1], f"invalid shapes: {seqused_q.shape=}, {seqused_k.shape=}"

    prefill_seq_ids = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    q_offsets = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    k_offsets = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    decode_seq_ids = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    decode_q_offsets = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    decode_k_offsets = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    # decode_last_tile_indices = nl.ndarray((max_num_seqs, 1), dtype=nl.int32, buffer=nl.hbm)
    prefill_last_tile_indices = nl.ndarray(
        (active_q_tile_size, 1), dtype=nl.int32, buffer=nl.hbm
    )
    decode_last_tile_indices = nl.ndarray(
        (max_num_seqs, 1), dtype=nl.int32, buffer=nl.hbm
    )
    prefill_trip_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)
    decode_trip_count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)

    ix = nl.arange(1)[:, None]
    iy = nl.arange(0, max_num_tiles)[None, :]
    prefill_seq_ids_zero = nisa.memset(
        (1, max_num_tiles), value=0, dtype=prefill_seq_ids.dtype
    )
    q_offsets_zero = nisa.memset((1, max_num_tiles), value=0, dtype=q_offsets.dtype)
    k_offsets_zero = nisa.memset((1, max_num_tiles), value=0, dtype=k_offsets.dtype)
    decode_seq_ids_zero = nisa.memset(
        (1, max_num_tiles), value=0, dtype=decode_seq_ids.dtype
    )
    decode_q_offsets_zero = nisa.memset(
        (1, max_num_tiles), value=0, dtype=decode_q_offsets.dtype
    )
    decode_k_offsets_zero = nisa.memset(
        (1, max_num_tiles), value=0, dtype=decode_k_offsets.dtype
    )
    nisa.dma_copy(
        dst=prefill_seq_ids[iy], src=prefill_seq_ids_zero, dge_mode=nisa.dge_mode.swdge
    )
    nisa.dma_copy(dst=q_offsets[iy], src=q_offsets_zero, dge_mode=nisa.dge_mode.swdge)
    nisa.dma_copy(dst=k_offsets[iy], src=k_offsets_zero, dge_mode=nisa.dge_mode.swdge)
    nisa.dma_copy(
        dst=decode_seq_ids[iy], src=decode_seq_ids_zero, dge_mode=nisa.dge_mode.swdge
    )
    nisa.dma_copy(
        dst=decode_q_offsets[iy],
        src=decode_q_offsets_zero,
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=decode_k_offsets[iy],
        src=decode_k_offsets_zero,
        dge_mode=nisa.dge_mode.swdge,
    )
    prefill_last_tile_indices_zero = nisa.memset(
        (active_q_tile_size, 1), value=0, dtype=prefill_last_tile_indices.dtype
    )
    decode_last_tile_indices_zero = nisa.memset(
        (max_num_seqs, 1), value=0, dtype=decode_last_tile_indices.dtype
    )
    nisa.dma_copy(
        dst=prefill_last_tile_indices[nl.arange(0, active_q_tile_size)[:, None], 0],
        src=prefill_last_tile_indices_zero,
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=decode_last_tile_indices[nl.arange(0, max_num_seqs)[:, None], 0],
        src=decode_last_tile_indices_zero,
        dge_mode=nisa.dge_mode.swdge,
    )

    seqused_q_sbuf = nl.ndarray(
        (par_dim(1), max_num_seqs), dtype=nl.int32, buffer=nl.sbuf
    )
    seqused_k_sbuf = nl.ndarray(
        (par_dim(1), max_num_seqs), dtype=nl.int32, buffer=nl.sbuf
    )
    is_prefill_seq = nl.zeros(
        (par_dim(1), max_num_seqs), dtype=nl.int32, buffer=nl.sbuf
    )
    is_decode_seq = nl.zeros((par_dim(1), max_num_seqs), dtype=nl.int32, buffer=nl.sbuf)
    total_tiles_per_seq = nl.zeros(
        (par_dim(1), max_num_seqs), dtype=nl.int32, buffer=nl.sbuf
    )
    total_prefill_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    total_decode_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    total_seq_q_len = nl.zeros((1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf)
    cu_prefill_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    cu_decode_tiles_per_seq = nl.zeros(
        (1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf
    )
    cu_seq_q_len = nl.ndarray((1, max_num_seqs + 1), dtype=nl.int32, buffer=nl.sbuf)
    # num_seqs_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    # num_prefill_tiles = nl.zeros((1, 1), dtype=nl.int32, buffer=nl.sbuf)

    # load sequences into sbuf
    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(max_num_seqs)[None, :]
    # i_cu = nl.arange(max_num_seqs+1)[None, :]
    print(
        f"flash_pa_with_schedule.py: {max_num_seqs=}, {max_num_tiles=}, {seqused_q.shape=}, {seqused_k.shape=}"
    )
    nisa.dma_copy(
        dst=seqused_q_sbuf[i_p, i_f], src=seqused_q[i_f], dge_mode=nisa.dge_mode.swdge
    )
    nisa.dma_copy(
        dst=seqused_k_sbuf[i_p, i_f], src=seqused_k[i_f], dge_mode=nisa.dge_mode.swdge
    )
    identity_for_transpose_p = nl.load(identity_for_transpose_p_hbm)

    # num_seqs_sbuf[0, 0] = nl.load(num_seqs[0])

    # Allocate and initialize counters
    # cur_q_pos = nl.ndarray((1,), dtype=nl.int32, buffer=nl.hbm)
    # cur_pos = nl.ndarray((1,), dtype=nl.int32, buffer=nl.hbm)
    # nl.store(cur_q_pos[0], 0)
    # nl.store(cur_pos[0], 0)

    expr_a = nl.arange(0, max_num_tiles)[None, :]
    iota_a = nisa.iota(expr_a, dtype=nl.int32)
    padded_max_num_tiles = pad_to_multiple(max_num_tiles, pmax)
    num_partitions = cdiv(padded_max_num_tiles, pmax)
    i_pmax = nl.arange(pmax)[:, None, None]
    i_batch = nl.arange(max_num_seqs)[None, :, None]
    i_fnp = nl.arange(num_partitions)[None, None, :]
    # padded_constant = np.transpose(np.arange(padded_max_num_tiles).reshape([num_partitions, pmax])).reshape([pmax, 1, num_partitions])
    # batched_iota_a_hbm = nl.shared_constant(padded_constant, dtype=nl.float32)
    # batched_iota_a = nl.load(batched_iota_a_hbm[i_pmax, 0, i_fnp]).reshape([pmax, num_partitions])
    iota_pmax = nisa.iota(nl.arange(pmax)[:, None], dtype=nl.float32)
    iota_fnp = nl.multiply(
        nisa.iota(nl.arange(num_partitions)[None, :], dtype=nl.int32), pmax
    ).broadcast_to([pmax, num_partitions])
    batched_iota_a = nisa.tensor_scalar(iota_fnp, nl.add, iota_pmax, dtype=nl.int32)
    # import pdb
    # pdb.set_trace()

    # seq_ids_values = nl.full((1, max_num_tiles), 0, dtype=nl.int32, buffer=nl.sbuf)
    zeros = nl.full((pmax, num_partitions), 0, dtype=nl.uint8, buffer=nl.sbuf)
    ones = nl.full((pmax, num_partitions), 1, dtype=nl.uint8, buffer=nl.sbuf)
    q_tile_idx = nl.ndarray((1, max_num_tiles), dtype=nl.int32, buffer=nl.sbuf)
    k_tile_idx = nl.ndarray((1, max_num_tiles), dtype=nl.int32, buffer=nl.sbuf)

    def _cdiv(x, y):
        return nl.ceil(nl.divide(x, y), dtype=nl.int32)

    # for seq_id in nl.affine_range(max_num_seqs):
    # Get sequence lengths
    i_f = nl.arange(max_num_seqs)[None, :]
    seq_q_len = nl.copy(seqused_q_sbuf[0, i_f], dtype=nl.float32)
    seq_k_len = nl.copy(seqused_k_sbuf[0, i_f], dtype=nl.float32)
    contexted_seq_k_len = nisa.tensor_tensor(
        seq_q_len, seq_k_len, nl.add, dtype=nl.float32
    )

    # is prefill/decode
    is_prefill_seq[...] = nisa.tensor_scalar(seq_q_len, nl.greater, 1.0, dtype=nl.int32)
    is_decode_seq[...] = nisa.tensor_scalar(seq_q_len, nl.equal, 1.0, dtype=nl.int32)

    # Calculate number of tiles needed
    num_q_tiles = _cdiv(seq_q_len, q_tile_size)
    num_k_tiles = _cdiv(contexted_seq_k_len, large_k_tile_size)
    total_tiles_per_seq[...] = nisa.tensor_tensor(
        num_q_tiles, nl.copy(num_k_tiles, dtype=nl.float32), nl.multiply
    )
    # num_k_tiles_per_q_tile = nl.ndarray((active_q_tile_size, 1), dtype=nl.int32)
    # build_num_k_tiles_per_q_tile(num_k_tiles_per_q_tile, num_q_tiles, num_k_tiles, total_tiles_per_seq)

    # total_tiles_hbm = nl.ndarray((max_num_seqs, ), dtype=nl.int32, buffer=nl.hbm)
    # nl.store(total_tiles_hbm[i_f], total_tiles_per_seq[i_p, i_f])

    num_prefill_seqs = nisa.tensor_reduce(nl.add, is_prefill_seq, axis=1)  # sum
    num_decode_seqs = nisa.tensor_reduce(nl.add, is_decode_seq, axis=1)  # sum
    num_prefill_tiles = nisa.tensor_reduce(
        nl.add,
        nl.multiply(is_prefill_seq, total_tiles_per_seq),
        axis=1,
        dtype=nl.float32,
    )  # sum
    num_decode_tiles = nisa.tensor_reduce(
        nl.add, nl.multiply(is_decode_seq, total_tiles_per_seq), axis=1
    )  # sum
    total_num_tiles = nisa.tensor_scalar(
        num_prefill_tiles,
        nl.add,
        nl.copy(num_decode_tiles, dtype=nl.float32),
        dtype=nl.float32,
    )  # imm

    num_decode_seqs_imm = nl.copy(num_decode_seqs, dtype=nl.float32)

    total_prefill_tiles_per_seq[0, i_f + 1] = nl.multiply(
        total_tiles_per_seq, is_prefill_seq
    )
    total_decode_tiles_per_seq[0, i_f + 1] = nl.multiply(
        total_tiles_per_seq, is_decode_seq
    )
    total_seq_q_len[0, i_f + 1] = nl.copy(seq_q_len[0, i_f])
    cu_seq_q_len[...] = _cumsum(total_seq_q_len)
    cu_prefill_tiles_per_seq[...] = _cumsum(total_prefill_tiles_per_seq)
    cu_decode_tiles_per_seq[...] = _cumsum(total_decode_tiles_per_seq)

    batched_prefill_seq_ids = nl.zeros(
        (pmax, max_num_seqs, num_partitions), dtype=nl.int32
    )
    batched_prefill_q_offsets = nl.zeros(
        (pmax, max_num_seqs, num_partitions), dtype=nl.int32
    )
    batched_prefill_k_offsets = nl.zeros(
        (pmax, max_num_seqs, num_partitions), dtype=nl.int32
    )

    cu_prefill_tiles_imm_br = nl.copy(
        cu_prefill_tiles_per_seq, dtype=nl.float32
    ).broadcast_to([pmax, max_num_seqs + 1])
    cu_seq_q_len_imm_br = nl.copy(cu_seq_q_len, dtype=nl.float32).broadcast_to(
        [pmax, max_num_seqs + 1]
    )
    seq_ids_imm_br = nisa.iota(
        nl.arange(max_num_seqs)[None, :], dtype=nl.float32
    ).broadcast_to([pmax, max_num_seqs])
    num_q_tiles_imm_br = nl.copy(num_q_tiles, dtype=nl.float32).broadcast_to(
        [pmax, max_num_seqs]
    )
    num_k_tiles_imm_br = nl.copy(num_k_tiles, dtype=nl.float32).broadcast_to(
        [pmax, max_num_seqs]
    )

    # nl.device_print("flash_pa_with_schedule.py: batched_iota_a=", batched_iota_a)

    # decode_valid_mask_hbm = nl.ndarray((pmax, max_num_seqs, 4), dtype=nl.int32, buffer=nl.hbm)
    # batched_iota_a_debug = nl.ndarray((pmax, max_num_seqs, 4), dtype=nl.int32, buffer=nl.hbm)
    # decode_valid_mask_hbm = nl.ndarray((pmax, max_num_seqs, num_partitions), dtype=nl.int32, buffer=nl.hbm)

    # for each prefill sequence
    for seq_id in nl.affine_range(max_num_seqs):
        q_index_tensor = nl.ndarray(
            (pmax, num_partitions), dtype=nl.int32, buffer=nl.sbuf
        )

        cur_pos_ptr_imm = cu_prefill_tiles_imm_br[i_pmax, seq_id]
        upper_bound_imm = cu_prefill_tiles_imm_br[i_pmax, seq_id + 1]
        cur_q_pos_imm = cu_seq_q_len_imm_br[i_pmax, seq_id]
        num_q_tiles_imm = num_q_tiles_imm_br[i_pmax, seq_id]
        num_k_tiles_imm = num_k_tiles_imm_br[i_pmax, seq_id]

        # nl.device_print("flash_pa_with_schedule.py: (prefill) cur_pos_ptr_imm=", cu_prefill_tiles_imm_br[0, seq_id])
        # nl.device_print("flash_pa_with_schedule.py: (prefill) upper_bound_imm=", cu_prefill_tiles_imm_br[0, seq_id+1])
        # nl.device_print("flash_pa_with_schedule.py: (prefill) cur_q_pos_imm=", cu_seq_q_len_imm_br[0, seq_id])
        # nl.device_print("flash_pa_with_schedule.py: (prefill) num_q_tiles_imm=", num_q_tiles_imm_br[0, seq_id])

        # Create threshold tensor and valid position mask
        # upper_bound = nl.add(cur_pos_ptr, total_tiles_imm, dtype=nl.float32)
        valid_mask_lb = nisa.tensor_scalar(
            batched_iota_a, nl.greater_equal, cur_pos_ptr_imm, dtype=nl.bool_
        )
        valid_mask_ub = nisa.tensor_scalar(
            batched_iota_a, nl.less, upper_bound_imm, dtype=nl.bool_
        )
        valid_mask = nisa.tensor_tensor(
            valid_mask_lb, valid_mask_ub, nl.logical_and, dtype=nl.bool_
        ).reshape([pmax, num_partitions])
        # nl.store(batched_iota_a_debug[i_pmax, seq_id, 0], cur_pos_ptr_imm)
        # nl.store(batched_iota_a_debug[i_pmax, seq_id, 1], upper_bound_imm)
        # nl.store(batched_iota_a_debug[i_pmax, seq_id, 2], cur_q_pos_imm)
        # nl.store(batched_iota_a_debug[i_pmax, seq_id, 3], num_q_tiles_imm)
        # nl.store(decode_valid_mask_hbm[i_pmax, seq_id, i_fnp], valid_mask.reshape([pmax, 1, num_partitions])[i_pmax, 0, i_fnp])

        # Create sequence ID tensor and update using mask
        seq_id_tensor = nisa.tensor_scalar(
            ones, nl.multiply, seq_ids_imm_br[:, seq_id], dtype=nl.int32
        )
        batched_prefill_seq_ids[:, seq_id, :] = nisa.tensor_tensor(
            seq_id_tensor, valid_mask, nl.multiply
        )
        # nl.store(decode_valid_mask_hbm[:, seq_id, :], batched_prefill_seq_ids[:, seq_id, :])

        # For each k-tile, we repeat the q-tile pattern
        relative_position = nl.subtract(batched_iota_a, cur_pos_ptr_imm).reshape(
            [pmax, num_partitions]
        )

        # Calculate q_index as position % num_q_tiles (cycles through q-tiles)
        # Calculate k_index as position // num_q_tiles (increments after all q-tiles)
        _floor_divide(
            relative_position,
            nl.copy(num_k_tiles_imm).reshape([pmax, 1]),
            q_index_tensor,
        )
        product = nisa.tensor_scalar(q_index_tensor, nl.multiply, num_k_tiles_imm)
        k_index_tensor = nisa.tensor_tensor(relative_position, product, nl.subtract)
        # nl.device_print("relative_position=", relative_position)
        # nl.device_print("k_index_tensor=", k_index_tensor)
        # nl.device_print("q_index_tensor=", q_index_tensor)
        # nl.device_print("num_q_tiles_imm=", num_q_tiles_imm)
        # nl.device_print("num_k_tiles_imm=", num_k_tiles_imm)

        # nl.device_print("flash_pa_with_schedule.py: (prefill) k_index_tensor=", k_index_tensor)
        # nl.device_print("flash_pa_with_schedule.py: (prefill) q_index_tensor=", q_index_tensor)

        # Calculate offsets - q_offset repeats pattern for each k-tile
        # k_offset starts at 0 for each sequence and increments by large_k_tile_size
        q_tile_offsets = nisa.tensor_scalar(
            q_index_tensor,
            nl.multiply,
            float(q_tile_size),
            op1=nl.add,
            operand1=cur_q_pos_imm,
        )
        k_tile_offsets = nisa.tensor_scalar(
            k_index_tensor, nl.multiply, float(large_k_tile_size)
        )

        # Apply offsets using mask
        batched_prefill_q_offsets[:, seq_id, :] = nisa.tensor_tensor(
            q_tile_offsets, valid_mask, nl.multiply
        )
        batched_prefill_k_offsets[:, seq_id, :] = nisa.tensor_tensor(
            k_tile_offsets, valid_mask, nl.multiply
        )

        # nl.device_print("flash_pa_with_schedule.py: (prefill) batched_prefill_seq_ids[:, seq_id, :]=", batched_prefill_seq_ids[:, seq_id, :])
        # nl.device_print("flash_pa_with_schedule.py: (prefill) batched_prefill_q_offsets[:, seq_id, :]=", batched_prefill_q_offsets[:, seq_id, :])
        # nl.device_print("flash_pa_with_schedule.py: (prefill) batched_prefill_k_offsets[:, seq_id, :]=", batched_prefill_k_offsets[:, seq_id, :])

    # Perform BP-transpose on batched output buffers
    packed_prefill_seq_ids = nl.ndarray((1, padded_max_num_tiles), dtype=nl.int32)
    packed_prefill_q_offsets = nl.ndarray((1, padded_max_num_tiles), dtype=nl.int32)
    packed_prefill_k_offsets = nl.ndarray((1, padded_max_num_tiles), dtype=nl.int32)
    for fidx in nl.affine_range(num_partitions):
        tmp_prefill_seq_ids = nl.ndarray((1, pmax), dtype=nl.int32)
        tmp_prefill_q_offsets = nl.ndarray((1, pmax), dtype=nl.int32)
        tmp_prefill_k_offsets = nl.ndarray((1, pmax), dtype=nl.int32)
        ipp, iff = nl.mgrid[0:pmax, 0:max_num_seqs]
        local_seq_ids_tile = nisa.tensor_reduce(
            nl.add,
            batched_prefill_seq_ids.reshape([pmax, max_num_seqs * num_partitions])[
                ipp, iff * num_partitions + fidx
            ],
            axis=[1],
        )
        local_q_offsets_tile = nisa.tensor_reduce(
            nl.add,
            batched_prefill_q_offsets.reshape([pmax, max_num_seqs * num_partitions])[
                ipp, iff * num_partitions + fidx
            ],
            axis=[1],
        )
        local_k_offsets_tile = nisa.tensor_reduce(
            nl.add,
            batched_prefill_k_offsets.reshape([pmax, max_num_seqs * num_partitions])[
                ipp, iff * num_partitions + fidx
            ],
            axis=[1],
        )
        PF_transpose_with_PE(
            local_seq_ids_tile,  # batched_prefill_seq_ids.reshape([pmax, max_num_seqs*num_partitions])[ipp, iff*num_partitions+fidx],
            tmp_prefill_seq_ids,
            identity_for_transpose=identity_for_transpose_p,
            out_in_psum=False,
        )
        PF_transpose_with_PE(
            local_q_offsets_tile,  # batched_prefill_q_offsets.reshape([pmax, max_num_seqs*num_partitions])[ipp, iff*num_partitions+fidx],
            tmp_prefill_q_offsets,
            identity_for_transpose=identity_for_transpose_p,
            out_in_psum=False,
        )
        PF_transpose_with_PE(
            local_k_offsets_tile,  # batched_prefill_k_offsets.reshape([pmax, max_num_seqs*num_partitions])[ipp, iff*num_partitions+fidx],
            tmp_prefill_k_offsets,
            identity_for_transpose=identity_for_transpose_p,
            out_in_psum=False,
        )
        ipp, iff = nl.mgrid[0:1, 0:pmax]
        packed_prefill_seq_ids[ipp, iff + fidx * pmax] = nl.copy(tmp_prefill_seq_ids)
        packed_prefill_q_offsets[ipp, iff + fidx * pmax] = nl.copy(
            tmp_prefill_q_offsets
        )
        packed_prefill_k_offsets[ipp, iff + fidx * pmax] = nl.copy(
            tmp_prefill_k_offsets
        )

    # nl.device_print("flash_pa_with_schedule.py: batched_prefill_seq_ids=", batched_prefill_seq_ids)
    # nl.device_print("flash_pa_with_schedule.py: batched_prefill_q_offsets=", batched_prefill_q_offsets)
    # nl.device_print("flash_pa_with_schedule.py: batched_prefill_k_offsets=", batched_prefill_k_offsets)

    prefill_seq_ids_sbuf = nl.copy(
        packed_prefill_seq_ids
    )  # nisa.tensor_partition_reduce(nl.add, packed_prefill_seq_ids)
    prefill_q_offsets_sbuf = nl.copy(
        packed_prefill_q_offsets
    )  # nisa.tensor_partition_reduce(nl.add, packed_prefill_q_offsets)
    prefill_k_offsets_sbuf = nl.copy(
        packed_prefill_k_offsets
    )  # nisa.tensor_partition_reduce(nl.add, packed_prefill_k_offsets)
    nisa.dma_copy(
        dst=prefill_seq_ids[iy],
        src=prefill_seq_ids_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=q_offsets[iy],
        src=prefill_q_offsets_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=k_offsets[iy],
        src=prefill_k_offsets_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        dge_mode=nisa.dge_mode.swdge,
    )

    ## =======================================================

    batched_decode_seq_ids = nl.zeros(
        (pmax, max_num_seqs, num_partitions), dtype=nl.int32
    )
    batched_decode_q_offsets = nl.zeros(
        (pmax, max_num_seqs, num_partitions), dtype=nl.int32
    )
    batched_decode_k_offsets = nl.zeros(
        (pmax, max_num_seqs, num_partitions), dtype=nl.int32
    )

    cu_decode_tiles_imm_br = nl.copy(
        cu_decode_tiles_per_seq, dtype=nl.float32
    ).broadcast_to([pmax, max_num_seqs + 1])
    cu_seq_q_len_imm_br = nl.copy(cu_seq_q_len, dtype=nl.float32).broadcast_to(
        [pmax, max_num_seqs + 1]
    )
    seq_ids_imm_br = nisa.iota(
        nl.arange(max_num_seqs)[None, :], dtype=nl.float32
    ).broadcast_to([pmax, max_num_seqs])
    num_q_tiles_imm_br = nl.copy(num_q_tiles, dtype=nl.float32).broadcast_to(
        [pmax, max_num_seqs]
    )
    # nl.device_print("flash_pa_with_schedule.py: cu_seq_q_len=", cu_seq_q_len)

    # For each decode sequence
    for loop_index in nl.affine_range(max_num_seqs):
        # seq_id = nl.add(num_prefill_seqs, loop_index, dtype=nl.uint32)
        seq_id = nisa.iota(loop_index, dtype=nl.uint32)
        seq_id_imm_br = nl.copy(seq_id.broadcast_to([pmax, 1]), dtype=nl.float32)
        # q_offset_val = nl.load(decode_q_offsets[iy])
        # k_offset_val = nl.load(decode_k_offsets[iy])
        # seq_ids_values = nl.load(decode_seq_ids[iy])
        cur_pos_ptr_imm = cu_decode_tiles_imm_br[i_pmax, loop_index]
        upper_bound_imm = cu_decode_tiles_imm_br[i_pmax, loop_index + 1]
        cur_q_pos_imm = nisa.tensor_copy_dynamic_src(
            cu_seq_q_len_imm_br[i_pmax, seq_id], dtype=nl.float32
        )

        # Create threshold tensor and valid position mask
        valid_mask_lb = nisa.tensor_scalar(
            batched_iota_a, nl.greater_equal, cur_pos_ptr_imm, dtype=nl.bool_
        )
        valid_mask_ub = nisa.tensor_scalar(
            batched_iota_a, nl.less, upper_bound_imm, dtype=nl.bool_
        )
        valid_mask = nisa.tensor_tensor(
            valid_mask_lb, valid_mask_ub, nl.logical_and, dtype=nl.bool_
        ).reshape([pmax, num_partitions])

        # Create sequence ID tensor and update using mask
        seq_id_tensor = nisa.tensor_scalar(
            ones, nl.multiply, seq_id_imm_br, dtype=nl.int32
        )
        batched_decode_seq_ids[:, loop_index, :] = nisa.tensor_tensor(
            seq_id_tensor, valid_mask, nl.multiply
        )
        # nl.device_print("flash_pa_with_schedule.py: batched_decode_seq_ids[:, loop_index, :]=", batched_decode_seq_ids[:16, loop_index, :])

        # For each k-tile, we repeat the q-tile pattern
        k_index_tensor = nl.subtract(batched_iota_a, cur_pos_ptr_imm).reshape(
            [pmax, num_partitions]
        )

        # Calculate offsets - q_offset repeats pattern for each k-tile
        # k_offset starts at 0 for each sequence and increments by large_k_tile_size
        q_tile_offsets = nisa.tensor_scalar(zeros, nl.add, cur_q_pos_imm)
        # nl.device_print("flash_pa_with_schedule.py: cur_q_pos_imm=", cur_q_pos_imm)
        # nl.device_print("flash_pa_with_schedule.py: q_tile_offsets=", q_tile_offsets)
        k_tile_offsets = nisa.tensor_scalar(
            k_index_tensor, nl.multiply, float(large_k_tile_size)
        )

        # Apply offsets using mask
        batched_decode_q_offsets[:, loop_index, :] = nisa.tensor_tensor(
            q_tile_offsets, valid_mask, nl.multiply
        )
        batched_decode_k_offsets[:, loop_index, :] = nisa.tensor_tensor(
            k_tile_offsets, valid_mask, nl.multiply
        )

    # nl.device_print("flash_pa_with_schedule.py: batched_decode_seq_ids=", batched_decode_seq_ids)
    # nl.device_print("flash_pa_with_schedule.py: batched_decode_q_offsets=", batched_decode_q_offsets)
    # nl.device_print("flash_pa_with_schedule.py: batched_decode_k_offsets=", batched_decode_k_offsets)

    # Perform BP-transpose on batched output buffers
    packed_decode_seq_ids = nl.ndarray((1, padded_max_num_tiles), dtype=nl.int32)
    packed_decode_q_offsets = nl.ndarray((1, padded_max_num_tiles), dtype=nl.int32)
    packed_decode_k_offsets = nl.ndarray((1, padded_max_num_tiles), dtype=nl.int32)
    for fidx in nl.affine_range(num_partitions):
        tmp_decode_seq_ids = nl.ndarray((1, pmax), dtype=nl.int32)
        tmp_decode_q_offsets = nl.ndarray((1, pmax), dtype=nl.int32)
        tmp_decode_k_offsets = nl.ndarray((1, pmax), dtype=nl.int32)
        ipp, iff = nl.mgrid[0:pmax, 0:max_num_seqs]
        local_seq_ids_tile = nisa.tensor_reduce(
            nl.add,
            batched_decode_seq_ids.reshape([pmax, max_num_seqs * num_partitions])[
                ipp, iff * num_partitions + fidx
            ],
            axis=[1],
        )
        local_q_offsets_tile = nisa.tensor_reduce(
            nl.add,
            batched_decode_q_offsets.reshape([pmax, max_num_seqs * num_partitions])[
                ipp, iff * num_partitions + fidx
            ],
            axis=[1],
        )
        local_k_offsets_tile = nisa.tensor_reduce(
            nl.add,
            batched_decode_k_offsets.reshape([pmax, max_num_seqs * num_partitions])[
                ipp, iff * num_partitions + fidx
            ],
            axis=[1],
        )
        PF_transpose_with_PE(
            local_seq_ids_tile,  # batched_decode_seq_ids.reshape([pmax, max_num_seqs*num_partitions])[ipp, iff*num_partitions+fidx],
            tmp_decode_seq_ids,
            identity_for_transpose=identity_for_transpose_p,
            out_in_psum=False,
        )
        PF_transpose_with_PE(
            local_q_offsets_tile,  # batched_decode_q_offsets.reshape([pmax, max_num_seqs*num_partitions])[ipp, iff*num_partitions+fidx],
            tmp_decode_q_offsets,
            identity_for_transpose=identity_for_transpose_p,
            out_in_psum=False,
        )
        PF_transpose_with_PE(
            local_k_offsets_tile,  # batched_decode_k_offsets.reshape([pmax, max_num_seqs*num_partitions])[ipp, iff*num_partitions+fidx],
            tmp_decode_k_offsets,
            identity_for_transpose=identity_for_transpose_p,
            out_in_psum=False,
        )
        ipp, iff = nl.mgrid[0:1, 0:pmax]
        packed_decode_seq_ids[ipp, iff + fidx * pmax] = nl.copy(tmp_decode_seq_ids)
        packed_decode_q_offsets[ipp, iff + fidx * pmax] = nl.copy(tmp_decode_q_offsets)
        packed_decode_k_offsets[ipp, iff + fidx * pmax] = nl.copy(tmp_decode_k_offsets)

    decode_seq_ids_sbuf = nl.copy(
        packed_decode_seq_ids
    )  # nisa.tensor_partition_reduce(nl.add, packed_decode_seq_ids)
    decode_q_offsets_sbuf = nl.copy(packed_decode_q_offsets)
    decode_k_offsets_sbuf = nl.copy(packed_decode_k_offsets)
    nisa.dma_copy(
        dst=decode_q_offsets[iy],
        src=decode_q_offsets_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=decode_k_offsets[iy],
        src=decode_k_offsets_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        dge_mode=nisa.dge_mode.swdge,
    )

    nisa.dma_copy(
        dst=prefill_trip_count, src=num_prefill_tiles, dge_mode=nisa.dge_mode.swdge
    )
    nisa.dma_copy(
        dst=decode_trip_count, src=num_decode_tiles, dge_mode=nisa.dge_mode.swdge
    )

    num_prefill_tiles_imm = nl.copy(num_prefill_tiles, dtype=nl.float32)
    num_decode_tiles_imm = nl.copy(num_decode_tiles, dtype=nl.float32)
    # num_seqs_minus_one_imm = nl.subtract(num_seqs_sbuf, 1, dtype=nl.float32)
    max_num_seqs_minus_one_imm = max_num_seqs - 1
    max_num_seqs_minus_one = nl.full(
        (1, max_num_tiles), max_num_seqs_minus_one_imm, dtype=nl.int32
    )
    minus_one = nl.full((1, max_num_tiles), -1.0, dtype=nl.int32)
    # nl.device_print("flash_pa_with_schedule.py: num_prefill_tiles_imm=", num_prefill_tiles_imm)

    valid_mask = nisa.tensor_scalar(iota_a, nl.less, num_prefill_tiles, dtype=nl.bool_)

    k_offset_val = nl.ndarray((1, max_num_tiles), dtype=nl.int32)
    q_offset_val = nl.ndarray((1, max_num_tiles), dtype=nl.int32)
    nisa.dma_copy(dst=k_offset_val, src=k_offsets[iy], dge_mode=nisa.dge_mode.swdge)
    # valid_mask = nisa.tensor_scalar(iota_a, nl.greater_equal, num_prefill_tiles_imm, op1=nl.multiply, operand1=-1.0, dtype=nl.int32)
    nisa.dma_copy(
        dst=k_offsets[iy],
        src=nl.where(valid_mask, k_offset_val, minus_one),
        dge_mode=nisa.dge_mode.swdge,
    )

    nisa.dma_copy(dst=q_offset_val, src=q_offsets[iy], dge_mode=nisa.dge_mode.swdge)
    cur_q_pos_imm = nl.copy(cu_seq_q_len[0, max_num_seqs], dtype=nl.int32)
    # valid_mask = nisa.tensor_scalar(iota_a, nl.greater_equal, num_prefill_tiles_imm, op1=nl.multiply, operand1=cur_q_pos_imm, dtype=nl.int32)
    nisa.dma_copy(
        dst=q_offsets[iy],
        src=nl.where(valid_mask, q_offset_val, cur_q_pos_imm),
        dge_mode=nisa.dge_mode.swdge,
    )

    seq_ids_values = nl.ndarray(
        prefill_seq_ids[iy].shape, dtype=prefill_seq_ids[iy].dtype
    )
    nisa.dma_copy(
        dst=seq_ids_values, src=prefill_seq_ids[iy], dge_mode=nisa.dge_mode.swdge
    )
    # valid_mask = nisa.tensor_scalar(iota_a, nl.greater_equal, num_prefill_tiles, dtype=nl.bool_)
    nisa.dma_copy(
        dst=prefill_seq_ids[iy],
        src=nl.where(
            valid_mask, seq_ids_values, max_num_seqs_minus_one_imm, dtype=nl.int32
        ),
        dge_mode=nisa.dge_mode.swdge,
    )

    build_prefill_last_tile_indices(
        prefill_last_tile_indices,
        iota_a,
        is_prefill_seq,
        total_tiles_per_seq,
        num_prefill_seqs,
        num_q_tiles,
        active_q_tile_size,
    )
    build_decode_last_tile_indices(
        decode_last_tile_indices,
        iota_a,
        is_decode_seq,
        total_tiles_per_seq,
        num_decode_seqs,
    )

    # seq_ids_values = nl.load(decode_seq_ids[iy])
    # valid_mask = nisa.tensor_scalar(iota_a, nl.greater_equal, num_decode_tiles_imm, op1=nl.multiply, operand1=max_num_seqs_minus_one_imm, dtype=nl.int32)
    decode_valid_mask = nisa.tensor_scalar(
        iota_a, nl.less, num_decode_tiles_imm, dtype=nl.bool_
    )
    decode_seq_ids_values = nl.where(
        decode_valid_mask,
        decode_seq_ids_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        max_num_seqs_minus_one_imm,
    )
    nisa.dma_copy(
        dst=decode_seq_ids[iy], src=decode_seq_ids_values, dge_mode=nisa.dge_mode.swdge
    )

    # q_offset_val = nl.load(decode_q_offsets[iy])
    cur_q_pos_imm = nl.copy(cu_seq_q_len[0, max_num_seqs], dtype=nl.float32)
    # valid_mask = nisa.tensor_scalar(iota_a, nl.greater_equal, num_decode_tiles_imm, op1=nl.multiply, operand1=cur_q_pos_imm, dtype=nl.int32)
    decode_q_offsets_values = nl.where(
        decode_valid_mask,
        decode_q_offsets_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        cur_q_pos_imm,
        dtype=nl.int32,
    )
    nisa.dma_copy(
        dst=decode_q_offsets[iy],
        src=decode_q_offsets_values,
        dge_mode=nisa.dge_mode.swdge,
    )

    # k_offset_val = nl.load(decode_k_offsets[iy])
    # valid_mask = nisa.tensor_scalar(iota_a, nl.greater_equal, num_decode_tiles_imm, op1=nl.multiply, operand1=-1.0, dtype=nl.int32)
    decode_k_offsets_values = nl.where(
        decode_valid_mask,
        decode_k_offsets_sbuf.reshape([1, padded_max_num_tiles])[ix, iy],
        -1.0,
        dtype=nl.int32,
    )
    nisa.dma_copy(
        dst=decode_k_offsets[iy],
        src=decode_k_offsets_values,
        dge_mode=nisa.dge_mode.swdge,
    )

    num_prefill_seqs_hbm = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.hbm)
    nisa.dma_copy(
        dst=num_prefill_seqs_hbm, src=num_prefill_seqs, dge_mode=nisa.dge_mode.swdge
    )

    return (
        prefill_seq_ids,
        q_offsets,
        k_offsets,
        prefill_trip_count,
        prefill_last_tile_indices,
        decode_seq_ids,
        decode_q_offsets,
        decode_k_offsets,
        decode_trip_count,
        decode_last_tile_indices,
    )


def diff(cu_seqlens_q, seqused_q):
    # Get the maximum number of queries from the shape of cu_seqlens_q
    max_num_seqs = seqused_q.shape[-1]
    assert cu_seqlens_q.shape[-1] == (seqused_q.shape[-1] + 1), (
        f"invalid input shapes for diff function {cu_seqlens_q.shape[-1]=} vs {seqused_q.shape[-1]=}"
    )

    # Create indices for loading the entire cu_seqlens_q array
    i_p, i_f = nl.mgrid[0:1, 0 : max_num_seqs + 1]
    # Load the cu_seqlens_q array into buffer
    # cu_seqlens_q_sbuf = nl.load(cu_seqlens_q[i_f])

    cu_seqlens_q_sbuf0 = nl.ndarray((1, max_num_seqs), buffer=nl.sbuf, dtype=nl.int32)
    cu_seqlens_q_sbuf1 = nl.ndarray((1, max_num_seqs), buffer=nl.sbuf, dtype=nl.int32)

    # Create indices for accessing all elements except the last one
    i_p, i_f0 = nl.mgrid[0:1, 0:max_num_seqs]
    # Create indices for accessing all elements except the first one
    i_p, i_f1 = nl.mgrid[0:1, 1 : max_num_seqs + 1]
    nisa.dma_copy(
        dst=cu_seqlens_q_sbuf0, src=cu_seqlens_q[i_f0], dge_mode=nisa.dge_mode.swdge
    )
    nisa.dma_copy(
        dst=cu_seqlens_q_sbuf1, src=cu_seqlens_q[i_f1], dge_mode=nisa.dge_mode.swdge
    )

    # Calculate differences between adjacent elements in cu_seqlens_q
    seqused_q_sbuf = nisa.tensor_tensor(
        cu_seqlens_q_sbuf1[i_p, i_f0], cu_seqlens_q_sbuf0[i_p, i_f0], nl.subtract
    )

    # Store the first element from cu_seqlens_q to seqused_q
    # nl.store(seqused_q[i_p+i_f0], cu_seqlens_q_sbuf[i_p, i_f0])
    # Store the calculated differences to the rest of seqused_q
    # nl.store(seqused_q[i_p+i_f1], seqused_q_sbuf)
    nisa.dma_copy(dst=seqused_q[i_f0], src=seqused_q_sbuf, dge_mode=nisa.dge_mode.swdge)


def transpose_kv_token_mask(mask, new_mask, tiled_block_size, B_P_SIZE=128):
    """
    from
    (total_query_len, context_kv_len // LARGE_TILE_SZ, num_tiled_blocks // B_P_SIZE, B_P_SIZE, tiled_block_size)
    to
    (total_query_len, context_kv_len // LARGE_TILE_SZ, num_tiled_blocks // B_P_SIZE, tiled_block_size, B_P_SIZE)
    """
    size_q, n_small_in_large_q_tile, size_kv = mask.shape
    # Ensure the KV size is divisible by the block size
    assert size_kv % tiled_block_size == 0
    num_block = size_kv // tiled_block_size
    # Ensure there are enough blocks to form at least one batch
    assert num_block >= B_P_SIZE, (
        f"invalid {num_block=}, {size_kv=}, {tiled_block_size=}"
    )
    num_batch = num_block // B_P_SIZE

    # Create indices for each dimension
    i_f1 = nl.arange(num_batch)[None, :, None, None]  # Batch dimension
    i_f2 = nl.arange(B_P_SIZE)[None, None, :, None]  # Block index within batch
    i_f3 = nl.arange(tiled_block_size)[None, None, None, :]  # Position within block

    # Calculate total range covered by the transpose operation
    transpose_range = B_P_SIZE * tiled_block_size

    # Calculate source indices (original layout)
    i_f_src = i_f1 * transpose_range + i_f2 * tiled_block_size + i_f3

    # Calculate destination indices (transposed layout)
    i_f_dst = i_f1 * transpose_range + i_f3 * B_P_SIZE + i_f2

    # Create query indices
    i_p = nl.arange(size_q)[:, None, None, None]

    # Perform the transposition by indexing
    for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
        new_mask[i_p, small_q_idx, i_f_dst] = nl.copy(mask[i_p, small_q_idx, i_f_src])
    return new_mask


def build_tiled_block_tables(
    block_tables,
    k_offsets,
    sequence_ids,
    loop_index,
    broadcasted_iota_a_vec,
    broadcasted_iota_b_vec,
    num_tiles_unrolled,
    num_loads,
    num_blocks_per_large_tile,
    tiled_block_size,
    block_size_tiling_factor,
    B_P_SIZE,
):
    """
    Build block_tables_sbuf_v2 from block_tables and k_offsets

    Args:
        block_tables: 2D array with shape (max_num_seqs, max_blocks_per_seq)
        k_offsets: Array containing offsets
        sequence_ids: Array containing sequence IDs
        loop_index: Current loop index
        num_tiles_unrolled: Number of tiles being processed in parallel
        num_loads: Number of loads
        num_blocks_per_large_tile: Number of blocks per large tile
        tiled_block_size: Size of a tiled block
        block_size_tiling_factor: Tiling factor for blocks
        B_P_SIZE: Block processing size

    Returns:
        block_tables_sbuf_v2: Processed block tables array
    """
    assert B_P_SIZE * num_loads == num_blocks_per_large_tile, (
        f"invalid shape: {B_P_SIZE=}, {num_loads=}, {num_blocks_per_large_tile=}"
    )
    (max_num_tiles,) = k_offsets.shape
    max_num_seqs, max_blocks_per_seq = block_tables.shape
    block_tables_sbuf_v2 = nl.ndarray(
        (B_P_SIZE, num_loads, num_tiles_unrolled), dtype=nl.int32
    )
    # Initialize with zeros, since we use DMA skipping when loading block_table elements
    block_tables_loaded = nl.zeros(
        (par_dim(B_P_SIZE), num_loads, num_tiles_unrolled), dtype=block_tables.dtype
    )
    block_tables_scaled = nl.zeros(
        (par_dim(B_P_SIZE), num_loads, num_tiles_unrolled), dtype=block_tables.dtype
    )
    base_idx_br = nl.ndarray(
        (par_dim(B_P_SIZE), num_loads, num_tiles_unrolled), dtype=nl.float32
    )
    block_indices = nl.ndarray((1, num_tiles_unrolled), dtype=nl.int32, buffer=nl.sbuf)
    block_size_sbuf = nl.full(
        (1, 1),
        fill_value=(tiled_block_size * block_size_tiling_factor),
        dtype=nl.int32,
        buffer=nl.sbuf,
    )
    offsets_base = nisa.nc_transpose(
        nisa.tensor_tensor(
            nisa.iota(nl.arange(num_tiles_unrolled)[None, :], dtype=nl.int32),
            nl.multiply(loop_index, num_tiles_unrolled),
            nl.add,
        ),
        engine=nki.isa.constants.engine.vector,
    )
    ix, iy = nl.mgrid[0:num_tiles_unrolled, 0:1]
    k_offsets_local = nisa.nc_transpose(
        nl.load(
            k_offsets.reshape([max_num_tiles, 1])[offsets_base[ix, 0], 0],
            dtype=nl.int32,
        ),
        engine=nki.isa.constants.engine.vector,
    )
    _floor_divide(k_offsets_local, block_size_sbuf, block_indices)
    seq_ids = nisa.nc_transpose(
        nl.load(sequence_ids.reshape([max_num_tiles, 1])[offsets_base[ix, 0], 0]),
        engine=nki.isa.constants.engine.vector,
    )
    block_tables_flat = block_tables.reshape(
        [
            max_num_seqs * max_blocks_per_seq,
        ]
    )
    i_p = nl.arange(B_P_SIZE)[:, None]
    for load_idx in nl.affine_range(num_loads):
        block_offset = nl.multiply(
            load_idx, num_blocks_per_large_tile, dtype=nl.float32
        )
        i_sequence_offset = nl.multiply(seq_ids, max_blocks_per_seq)
        block_table_offset = nl.add(block_indices, block_offset)
        flattened_base_index = nl.add(
            i_sequence_offset, block_table_offset, dtype=nl.int32
        ).broadcast_to([B_P_SIZE, num_tiles_unrolled])
        # For each tile, we'll load num_blocks_per_large_tile consecutive values
        for t_idx in nl.affine_range(num_tiles_unrolled):
            base_idx_br[:, load_idx, t_idx] = nl.copy(
                flattened_base_index[i_p, t_idx], dtype=nl.float32
            )  # Get the base index for this tile
            load_indices = nisa.tensor_scalar(
                broadcasted_iota_a_vec,
                nl.add,
                base_idx_br[:, load_idx, t_idx],
                dtype=nl.int32,
            )
            block_tables_loaded[:, load_idx, t_idx] = nl.load(
                block_tables_flat[load_indices],
                dtype=nl.int32,
                mode=oob_mode.skip,
            )
            block_tables_scaled[:, load_idx, t_idx] = nisa.tensor_scalar(
                block_tables_loaded[:, load_idx, t_idx],
                nl.multiply,
                np.float32(block_size_tiling_factor),
            )
            block_tables_sbuf_v2[i_p, load_idx, t_idx] = nisa.tensor_tensor(
                block_tables_scaled[:, load_idx, t_idx], broadcasted_iota_b_vec, nl.add
            )

    return block_tables_sbuf_v2


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    enable_out_of_bound_check=False,
    experimental_flags="experimental-native-scalar-support, experimental-local-tensor-parent",  #
    debug_kernel=True,
    show_compiler_tb=True,
)
def flash_paged_attention_blockspase(
    query,
    key,
    value,
    kv_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    last_tile_indices,
    q_update_pred=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_k=None,
    num_seqs=None,
    block_tables=None,
    dynamic_loop_trip_count=None,
    softmax_scale=None,
    loop_unroll_factor=1,
    skip_active=False,
    decode_mode=False,
    enable_prefill=True,
    large_q_tile_size=128,
    large_kv_tile_size=1024,  ## TODO: change back to 1024
):
    """
    Flash PagedAttention Forward Kernel.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape (1, n_heads, seq_q, d)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - key_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - value_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - tile_q_indices: (num_large_tiles, large_tile_size_q)
      - tile_block_tables: (num_large_tiles, num_block_per_large_tile)
      - tile_masks: (num_large_tiles, large_tile_size_q, large_tile_size_k) if not decode_mode
          else (num_large_tiles, large_tile_size_q, large_tile_size_k)
      - active_mask: (seq_q, seq_q)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is always 1, and different
        requests are concatenated along sequence dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for block_tables (uint32) and mask (uint8)
      - If mixed_percision is True, then all Tensor Engine operation will be performed in
        bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
        will be in the same type as the inputs.

    Compile-time Constants:
      - sequence_parallel_group: sequence parallel group to shard the cache blocks, List[int].
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, default is set to `true`,
          if false, we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    mixed_precision = True

    # Tiling factors that should be ideally tunable
    B_P_SIZE = 128
    if decode_mode:
        LARGE_Q_TILE_SIZE = 1
        if tile_masks is not None:
            INNER_KV_TILE_SIZE, _, N_INNER_KV_TILE = tile_masks.shape
            assert INNER_KV_TILE_SIZE == B_P_SIZE
            LARGE_KV_TILE_SIZE = INNER_KV_TILE_SIZE * N_INNER_KV_TILE
        else:
            LARGE_KV_TILE_SIZE = large_kv_tile_size
    else:
        if tile_masks is not None:
            _, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
        else:
            LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = (
                large_q_tile_size,
                large_kv_tile_size,
            )
    INNER_Q_TILE_SIZE = min(B_P_SIZE, LARGE_Q_TILE_SIZE)
    assert LARGE_Q_TILE_SIZE % INNER_Q_TILE_SIZE == 0
    MAX_NUM_TILES = 256  # 1024 # 512
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // INNER_Q_TILE_SIZE

    b, h, seqlen_q, d = query.shape
    print(f"flash_pa_with_schedule.py: {b=}, {h=}, {seqlen_q=}, {d=}")
    assert is_power_of_2(seqlen_q), f"{seqlen_q=} is expected to be power of 2"
    assert seqlen_q <= 8192, f"Large {seqlen_q=} consumes too much sbuf space"
    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    # assert (
    #     d >= 16 and d <= 128 and is_power_of_2(d)
    # ), f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    B_D_SIZE = d
    _, num_blocks, k_h, block_size, _ = kv_cache.shape
    q_h_per_k_h = h // k_h
    assert tuple(kv_cache.shape) == (
        2,
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{kv_cache.shape=} mismatch!"
    # assert tuple(value_cache.shape) == (
    #     num_blocks,
    #     k_h,
    #     block_size,
    #     d,
    # ), f"{value_cache.shape=} mismatch!"
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

    if tile_masks is not None:
        assert tile_masks.dtype == nl.uint8, (
            f"{tile_masks.dtype=} is expected to be uint8"
        )
    if active_mask is not None:
        assert active_mask.dtype == nl.uint8, (
            f"{active_mask.dtype=} is expected to be uint8"
        )

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    print(f"flash_pa_with_schedule.py: {kernel_dtype=}, {acc_type=}")
    o = nl.ndarray((h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    # nl.store(o, 0)
    # tile_mask_hbm = nl.ndarray((INNER_Q_TILE_SIZE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE), # (128, 32, 8),
    #     dtype=nl.bool_, buffer=nl.shared_hbm)
    # nl.store(tile_mask_hbm, 0)
    tile_mask_hbm = None
    # iota_b_hbm = nl.ndarray((INNER_Q_TILE_SIZE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE), dtype=nl.int32, buffer=nl.shared_hbm)
    # nl.store(iota_b_hbm, 0)
    iota_b_hbm = None
    # tile_q_indices_hbm = nl.ndarray((seqlen_q, INNER_Q_TILE_SIZE, n_small_in_large_q_tile, loop_unroll_factor),
    #                                 dtype=nl.int32, buffer=nl.shared_hbm)
    # for i in nl.affine_range(seqlen_q // INNER_Q_TILE_SIZE):
    #     nl.store(tile_q_indices_hbm[nl.ds(i*INNER_Q_TILE_SIZE, INNER_Q_TILE_SIZE)], 0)
    tile_q_indices_hbm = None
    # tile_block_tables_hbm = nl.ndarray((seqlen_q, INNER_Q_TILE_SIZE, n_small_in_large_q_tile, loop_unroll_factor),
    #                                 dtype=nl.int32, buffer=nl.shared_hbm)
    # for i in nl.affine_range(seqlen_q // INNER_Q_TILE_SIZE):
    #     nl.store(tile_block_tables_hbm[nl.ds(i*INNER_Q_TILE_SIZE, INNER_Q_TILE_SIZE)], 0)
    # l_buffer_hbm = nl.ndarray((seqlen_q, h, 1), dtype=acc_type, buffer=nl.shared_hbm)
    # for i in nl.affine_range(seqlen_q // INNER_Q_TILE_SIZE):
    #     nl.store(l_buffer_hbm[nl.ds(i*INNER_Q_TILE_SIZE, INNER_Q_TILE_SIZE)], 0)
    l_buffer_hbm = None

    assert nl.program_ndim() == 2, (
        f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    )

    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    assert is_power_of_2(num_blocks_per_large_tile), (
        f"{num_blocks_per_large_tile=} is expected of be power of 2"
    )

    # max_num_queries = cu_seqlens_q.shape[-1]
    max_num_seqs = seqused_k.shape[-1]
    seqused_q = nl.ndarray((max_num_seqs,), dtype=nl.int32, buffer=nl.hbm)
    assert cu_seqlens_q.shape[-1] == (max_num_seqs + 1), (
        f"invalid {cu_seqlens_q.shape=}, expect {(max_num_seqs+1)=}"
    )
    print(f"flash_pa_with_schedule.py: {max_num_seqs=}")
    # prefill_seq_ids = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    # nl.store(prefill_seq_ids[nl.arange(max_num_tiles)[None, :]], value=0.0)

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

    # Compute length differences between cumulative sequence lengths and used sequences
    diff(cu_seqlens_q, seqused_q)

    # Prepare sequence IDs for tiled attention computation
    ACTIVE_Q_TILE_SIZE = min(seqlen_q, B_P_SIZE)
    # sequence_ids, q_offsets, k_offsets, dynamic_loop_trip_count, last_tile_indices_v2 = \
    #     prepare_sequence_offsets(
    #         seqused_q, seqused_k, num_seqs, max_num_tiles=MAX_NUM_TILES,
    #         q_tile_size=LARGE_Q_TILE_SIZE, large_k_tile_size=LARGE_KV_TILE_SIZE)
    identity_for_transpose_p_hbm = nl.shared_constant(
        np.identity(n=nl.tile_size.pmax, dtype=np.uint8), dtype=nl.uint8
    )
    (
        sequence_ids,
        q_offsets,
        k_offsets,
        dynamic_loop_trip_count,
        prefill_last_tile_indices,
        decode_seq_ids,
        decode_q_offsets,
        decode_k_offsets,
        decode_trip_count,
        decode_last_tile_indices,
    ) = prepare_sequence_offsets(
        seqused_q,
        seqused_k,
        num_seqs,
        identity_for_transpose_p_hbm,
        max_num_tiles=MAX_NUM_TILES,
        num_queries=seqlen_q,
        q_tile_size=LARGE_Q_TILE_SIZE,
        large_k_tile_size=LARGE_KV_TILE_SIZE,
    )  # , active_q_tile_size=max_num_seqs) # ACTIVE_Q_TILE_SIZE)
    # nl.device_print("flash_pa_with_schedule.py: sequence_ids=", sequence_ids.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: q_offsets=", q_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: k_offsets=", k_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: dynamic_loop_trip_count=", dynamic_loop_trip_count)
    # nl.device_print("flash_pa_with_schedule.py: last_tile_indices_v2=", last_tile_indices_v2)
    # nl.device_print("flash_pa_with_schedule.py: decode_seq_ids=", decode_seq_ids.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: decode_q_offsets=", decode_q_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: decode_k_offsets=", decode_k_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: decode_trip_count=", decode_trip_count)

    # enable_prefill = False
    enable_decode = True

    if enable_prefill:
        print(f"running prefill_kernel")
        assert seqlen_q % LARGE_Q_TILE_SIZE == 0
        prefill_subkernel(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            sequence_ids=sequence_ids,
            q_offsets=q_offsets,
            k_offsets=k_offsets,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            active_mask=active_mask,
            q_update_pred=None,  # q_update_pred,
            last_tile_indices=prefill_last_tile_indices,  # last_tile_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_seqs=num_seqs,
            block_tables=block_tables,
            o=o,
            dynamic_loop_trip_count=dynamic_loop_trip_count,
            loop_unroll_factor=loop_unroll_factor,
            batch_id=batch_id,
            head_id=head_id,
            seqlen_q=seqlen_q,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            block_size_tiling_factor=block_size_tiling_factor,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            skip_active=skip_active,
            tile_mask_hbm=tile_mask_hbm,
            iota_b_hbm=iota_b_hbm,
            tile_q_indices_hbm=tile_q_indices_hbm,
            l_buffer_hbm=l_buffer_hbm,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
        )
    else:
        print(f"prefill_kernel disabled")

    if enable_decode:
        print(f"running decode_kernel")
        decode_subkernel(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            sequence_ids=decode_seq_ids,
            q_offsets=decode_q_offsets,
            k_offsets=decode_k_offsets,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            q_update_pred=None,  # q_update_pred,
            last_tile_indices=decode_last_tile_indices,  # last_tile_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_seqs=num_seqs,
            block_tables=block_tables,
            o=o,
            dynamic_loop_trip_count=decode_trip_count,
            loop_unroll_factor=loop_unroll_factor,
            batch_id=batch_id,
            head_id=head_id,
            seqlen_q=max_num_seqs,  # seqlen_q,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            block_size_tiling_factor=block_size_tiling_factor,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            skip_active=skip_active,
            tile_mask_hbm=tile_mask_hbm,
        )

    return o
    # return o, seqused_q, sequence_ids, dynamic_loop_trip_count, q_offsets, k_offsets, prefill_last_tile_indices, \
    #     decode_seq_ids, decode_q_offsets, decode_k_offsets, decode_trip_count, decode_last_tile_indices


def prefill_subkernel(
    query,
    key,
    value,
    kv_cache,
    sequence_ids,
    q_offsets,
    k_offsets,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    q_update_pred,
    last_tile_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_seqs,
    block_tables,
    o,
    dynamic_loop_trip_count,
    loop_unroll_factor,
    batch_id,
    head_id,
    seqlen_q,
    num_blocks_per_large_tile,
    block_size_tiling_factor,
    tiled_block_size,
    kernel_dtype,
    acc_type,
    k_h,
    q_h_per_k_h,
    INNER_Q_TILE_SIZE,
    B_D_SIZE,
    softmax_scale,
    skip_active,
    tile_mask_hbm,
    iota_b_hbm,
    tile_q_indices_hbm,
    l_buffer_hbm,
    large_q_tile_size,
    large_kv_tile_size,
):
    # assert len(dynamic_loop_conds.shape) == 1 and dynamic_loop_conds.dtype == nl.int32
    # assert (dynamic_loop_conds.shape[0] - 1) % loop_unroll_factor == 0
    B_P_SIZE = 128
    B_F_SIZE = 512
    (max_num_tiles,) = q_offsets.shape
    if tile_masks is not None:
        _, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    else:
        LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = large_q_tile_size, large_kv_tile_size
    assert LARGE_KV_TILE_SIZE % B_F_SIZE == 0, (
        f"Need LARGE_KV_TILE_SIZE ({LARGE_KV_TILE_SIZE=}) to be divisible by ({B_F_SIZE=})"
    )
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // INNER_Q_TILE_SIZE

    (olm_buffer,) = allocate_prefill_accum_buffers(
        seqlen_q=seqlen_q,
        INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        acc_type=acc_type,
    )
    olm_next_tile_sbuf = nl.zeros(
        (
            par_dim(INNER_Q_TILE_SIZE),
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE + 2,
        ),
        dtype=acc_type,
    )
    olm_unrolled_sbuf = nl.ndarray(
        (
            par_dim(INNER_Q_TILE_SIZE),
            loop_unroll_factor + 1,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE + 2,
        ),
        dtype=acc_type,
    )
    last_tile_indices_sbuf = nl.load(last_tile_indices)
    q_update_pred_hbm = init_q_update_pred(max_num_tiles, last_tile_indices_sbuf)
    q_update_pred_hbm = prepare_q_update_pred(q_update_pred_hbm, last_tile_indices_sbuf)
    q_update_pred = q_update_pred_hbm.reshape(
        (max_num_tiles // loop_unroll_factor, loop_unroll_factor, 1)
    )

    # transpose identity matrix on hbm
    identity_for_transpose_k_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )
    identity_for_transpose_p_hbm = nl.shared_constant(
        np.identity(n=INNER_Q_TILE_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )

    stride_size = B_P_SIZE // block_size_tiling_factor
    broadcasted_iota_a_hbm = nl.shared_constant(
        np.arange(stride_size).repeat(block_size_tiling_factor), dtype=nl.int32
    )
    broadcasted_iota_a_vec = nl.load(
        broadcasted_iota_a_hbm[nl.arange(B_P_SIZE)[:, None]]
    )
    broadcasted_iota_b_hbm = nl.shared_constant(
        np.tile(np.arange(block_size_tiling_factor), stride_size), dtype=nl.int32
    )
    broadcasted_iota_b_vec = nl.load(
        broadcasted_iota_b_hbm[nl.arange(B_P_SIZE)[:, None]]
    )

    q_update_pred_sbuf = nl.ndarray((loop_unroll_factor, 1), dtype=nl.uint8)
    nisa.dma_copy(
        dst=q_update_pred_sbuf[...], src=q_update_pred[0], dge_mode=nisa.dge_mode.swdge
    )

    def _cdiv(x, y):
        return nl.ceil(nl.divide(x, y), dtype=nl.int32)

    trip_count_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    dynamic_for_ub_hbm = nl.ndarray((1, 1), buffer=nl.hbm, dtype=nl.int32)
    nisa.dma_copy(
        dst=trip_count_sbuf[0, 0],
        src=dynamic_loop_trip_count[0],
        dge_mode=nisa.dge_mode.swdge,
    )
    dynamic_for_ub_sbuf = nl.copy(
        _cdiv(
            nl.copy(trip_count_sbuf, dtype=nl.float32),
            nisa.memset(shape=(1, 1), value=loop_unroll_factor, dtype=nl.float32),
        ),
        dtype=nl.int32,
    )
    nisa.dma_copy(
        dst=dynamic_for_ub_hbm, src=dynamic_for_ub_sbuf, dge_mode=nisa.dge_mode.swdge
    )
    dynamic_for_ub = nl.ndarray(dynamic_for_ub_hbm.shape, dtype=nl.int32)
    nisa.dma_copy(
        dst=dynamic_for_ub, src=dynamic_for_ub_hbm, dge_mode=nisa.dge_mode.swdge
    )
    loop_index = nisa.memset(shape=(1, 1), value=0, dtype=nl.int32)

    # num_max_step = (dynamic_loop_conds.shape[0] - 1) // loop_unroll_factor
    # for loop_step in nl.sequential_range(2):
    #     loop_index = nisa.iota(loop_step, dtype=nl.int32)
    # for loop_step in nl.sequential_range(4):
    #     loop_index = nisa.iota(loop_step, dtype=nl.int32)
    # for _ in range(scalar(dynamic_for_ub)):
    #     loop_index = nl.load(index[0], dtype=np.int32)
    # while scalar(cond_var):
    # for _ in range(scalar(dynamic_for_ub)):
    #     loop_index = nl.load(index[0], dtype=np.int32)
    # for loop_step in nl.sequential_range(ceil_div(2, loop_unroll_factor)):
    #     loop_index[...] = nisa.iota(loop_step, dtype=nl.int32)
    for _ in range(scalar(dynamic_for_ub)):
        olm_unrolled_sbuf[:, nl.ds(1, loop_unroll_factor)] = 0
        olm_unrolled_sbuf[:, :, :, :, B_D_SIZE + 1] = NEG_INF
        olm_unrolled_sbuf[:, 0] = nl.copy(olm_next_tile_sbuf)
        q_update_pred_broadcast = transpose_broadcast_q_pred(
            q_update_pred_sbuf, INNER_Q_TILE_SIZE
        )
        prefill_prior(
            loop_index=nisa.tensor_copy(loop_index[...]),
            num_tiles_unrolled=loop_unroll_factor,
            query=query,
            kv_cache=kv_cache,
            olm_buffer_hbm=olm_buffer,
            olm_unrolled_sbuf=olm_unrolled_sbuf,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_seqs=num_seqs,
            block_tables=block_tables,
            sequence_ids=sequence_ids,
            q_offsets=q_offsets,
            k_offsets=k_offsets,
            tile_q_indices=tile_q_indices,
            tile_masks=tile_masks,
            q_update_pred_broadcast=q_update_pred_broadcast,
            tile_block_tables=tile_block_tables,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            block_size_tiling_factor=block_size_tiling_factor,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            identity_for_transpose_k_hbm=identity_for_transpose_k_hbm,
            identity_for_transpose_p_hbm=identity_for_transpose_p_hbm,
            broadcasted_iota_a_vec=broadcasted_iota_a_vec,
            broadcasted_iota_b_vec=broadcasted_iota_b_vec,
            batch_id=batch_id,
            head_id=head_id,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            softmax_scale=softmax_scale,
            n_small_in_large_q_tile=n_small_in_large_q_tile,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            B_F_SIZE=B_F_SIZE,
            B_D_SIZE=B_D_SIZE,
            tile_mask_hbm=tile_mask_hbm,
            iota_b_hbm=iota_b_hbm,
            tile_q_indices_hbm=tile_q_indices_hbm,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
        )
        olm_next_tile_sbuf[...] = nl.copy(olm_unrolled_sbuf[:, loop_unroll_factor])

        # # update index
        loop_index[...] = nl.add(loop_index[...], 1)
        # nl.store(index[0], loop_index_next[0, 0])
        # update conditions
        # cond_next = nl.load(dynamic_loop_conds[loop_index_next * loop_unroll_factor])
        # cond_next = nisa.tensor_tensor(loop_index_next * loop_unroll_factor, trip_count_sbuf, nl.less)
        # nl.store(dst=cond[0, 0], value=cond_next)
        # cond_var = cond

        q_update_pred_sbuf[...] = nl.load(
            q_update_pred[loop_index[0, 0]],
            mode=oob_mode.skip,
        )

    if l_buffer_hbm is not None:
        l_buffer_sbuf = nl.load(l_buffer)
        nl.store(
            l_buffer_hbm[:, nl.ds(head_id * q_h_per_k_h, q_h_per_k_h), 0],
            l_buffer_sbuf[:, :, 0],
        )

    epilogue_prefill(
        o=o,
        query=query,
        key=key,
        value=value,
        active_mask=active_mask,
        softmax_scale=softmax_scale,
        olm_buffer=olm_buffer,
        ACTIVE_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
        skip_active=True if active_mask is None else skip_active,
    )


def prepare_q_indices_range(
    q_offsets, tile_q_indices, loop_index, num_tiles_per_step, INNER_Q_TILE_SIZE
):
    assert len(tile_q_indices.shape) == 2
    q_indices_sbuf = load_indices_for_loop_step(
        tile_q_indices, loop_index, num_tiles_per_step
    )
    tile_partition_size, num_tile_partitions, num_indices_per_tile = (
        q_indices_sbuf.shape
    )
    B_P_SIZE = 128
    assert INNER_Q_TILE_SIZE <= B_P_SIZE
    index_partition_size = min(num_indices_per_tile, INNER_Q_TILE_SIZE)
    num_index_partitions = ceil_div(num_indices_per_tile, index_partition_size)
    transposed_q_indices = nl.ndarray(
        (
            par_dim(index_partition_size),
            num_index_partitions,
            num_tile_partitions * tile_partition_size,
        ),
        dtype=tile_q_indices.dtype,
    )
    transposed_q_indices_reshape = transposed_q_indices.reshape(
        (
            index_partition_size,
            num_index_partitions,
            num_tile_partitions,
            tile_partition_size,
        )
    )
    for i in range(num_tile_partitions):
        transform_to_vector_dge_layout(
            q_indices_sbuf[:, i],
            transposed_q_indices_reshape[:, :, i, :],
            index_partition_size,
        )
    return transposed_q_indices


def prefill_prior(
    loop_index,
    num_tiles_unrolled,
    query,
    kv_cache,
    olm_buffer_hbm,
    olm_unrolled_sbuf,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_seqs,
    block_tables,
    sequence_ids,
    q_offsets,
    k_offsets,
    tile_q_indices,
    tile_masks,
    q_update_pred_broadcast,
    tile_block_tables,
    num_blocks_per_large_tile,
    tiled_block_size,
    block_size_tiling_factor,
    kernel_dtype,
    acc_type,
    identity_for_transpose_k_hbm,
    identity_for_transpose_p_hbm,
    broadcasted_iota_a_vec,
    broadcasted_iota_b_vec,
    batch_id,
    head_id,
    k_h,
    q_h_per_k_h,
    softmax_scale,
    n_small_in_large_q_tile,
    INNER_Q_TILE_SIZE,
    B_F_SIZE,
    B_D_SIZE,
    tile_mask_hbm,
    iota_b_hbm,
    tile_q_indices_hbm,
    large_q_tile_size,
    large_kv_tile_size,
):
    B_P_SIZE = 128
    (MAX_NUM_TILE,) = q_offsets.shape
    identity_for_transpose_k = nl.load(identity_for_transpose_k_hbm)
    identity_for_transpose_p = nl.load(identity_for_transpose_p_hbm)

    if tile_masks is not None:
        MAX_NUM_TILE, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
    else:
        LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = large_q_tile_size, large_kv_tile_size

    num_loads = num_blocks_per_large_tile // B_P_SIZE
    # XXX: Work around a DMA skipping correctness issue:
    #      If nl.ndarray is used to allocate buffer for DMA skipping,
    #      kernel does not produce correct results.
    k_load_buffer = nl.zeros(
        (par_dim(B_P_SIZE), num_loads, tiled_block_size * B_D_SIZE),
        dtype=kv_cache.dtype,
    )
    v_load_buffer = nl.zeros(
        (par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE),
        dtype=kv_cache.dtype,
    )

    tile_q_indices_sbuf = (
        prepare_q_indices_range(
            q_offsets,
            tile_q_indices,
            loop_index,
            num_tiles_unrolled,
            INNER_Q_TILE_SIZE,
        )
        if tile_q_indices is not None
        else None
    )

    block_tables_sbuf = (
        load_indices_for_loop_step(
            tile_block_tables,
            loop_index,
            num_tiles_unrolled,
        )
        if tile_block_tables is not None
        else None
    )
    block_tables_sbuf = (
        transform_block_tables_for_indirect_load(
            block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
        )
        if tile_block_tables is not None
        else None
    )

    def _br_p(x, out):
        # Broadcast weight along first axis to match tensor shape
        assert x.shape[-1] == out.shape[-1], (
            f"invalid input shapes: {x.shape=}, {out.shape=}"
        )
        src_p, dst_p = x.shape[0], out.shape[0]
        batch_size = out.shape[1]
        assert batch_size == x.shape[1], (
            f"invalid input shape: {batch_size=} (expected {x.shape[1]=})"
        )
        step_size = out.shape[-1]
        n_iters = cdiv(dst_p // src_p, 32)
        br_size = min(32, dst_p // src_p)
        x_re = x.reshape(shape=(src_p, batch_size, step_size))
        for j in nl.sequential_range(n_iters):
            # for batch_idx in nl.sequential_range(batch_size):
            i_p = nl.arange(br_size)[:, None, None] + (j * br_size)
            i_b = nl.arange(batch_size)[None, :, None]
            i_f = nl.arange(step_size)[None, None, :]
            out[i_p, i_b, i_f] = nl.broadcast_to(
                x_re, shape=(br_size, batch_size, step_size)
            )

    def _br_f(x, out):
        src_p, dst_p = x.shape[0], out.shape[0]
        step_size = out.shape[-1]
        batch_size = out.shape[1]
        assert x.shape[1] == out.shape[1], (
            f"invalid batch dimension {x.shape[1]=} vs {out.shape[1]=}"
        )
        assert src_p == dst_p, f"invalid input shapes: {x.shape=}, {out.shape=}"
        x_re = x.reshape(shape=(src_p, batch_size, 1))
        i_p = nl.arange(src_p)[:, None, None]
        i_b = nl.arange(batch_size)[None, :, None]
        i_f = nl.arange(step_size)[None, None, :]
        out[i_p, i_b, i_f] = nl.broadcast_to(x_re, shape=(src_p, batch_size, step_size))

    # iota_a_vec = nisa.iota(nl.arange(0, LARGE_Q_TILE_SIZE)[:, None], dtype=nl.int32)
    batched_iota_a_hbm = nl.shared_constant(
        np.arange(LARGE_Q_TILE_SIZE)
        .reshape([n_small_in_large_q_tile, INNER_Q_TILE_SIZE])
        .T,
        dtype=nl.int32,
    )
    batched_iota_a_vec = nl.load(batched_iota_a_hbm)
    # iota_b_vec = nisa.iota(nl.arange(0, LARGE_KV_TILE_SIZE)[None, :], dtype=nl.int32)
    batched_iota_b_hbm = nl.shared_constant(
        np.broadcast_to(
            np.arange(LARGE_KV_TILE_SIZE), [n_small_in_large_q_tile, LARGE_KV_TILE_SIZE]
        ),
        dtype=nl.int32,
    )
    ix, iy, iz = nl.mgrid[0:1, 0:n_small_in_large_q_tile, 0:LARGE_KV_TILE_SIZE]
    batched_iota_b_vec = nl.load(batched_iota_b_hbm[ix + iy, iz])
    batched_iota_a = nl.ndarray(
        (par_dim(INNER_Q_TILE_SIZE), n_small_in_large_q_tile, LARGE_KV_TILE_SIZE),
        dtype=nl.int32,
        buffer=nl.sbuf,
    )
    batched_iota_b = nl.ndarray(
        (par_dim(INNER_Q_TILE_SIZE), n_small_in_large_q_tile, LARGE_KV_TILE_SIZE),
        dtype=nl.int32,
        buffer=nl.sbuf,
    )
    _br_f(batched_iota_a_vec, batched_iota_a)
    _br_p(batched_iota_b_vec, batched_iota_b)

    ## DEBUGGING ONLY
    # nl.store(iota_b_hbm[:, nl.ds(0, n_small_in_large_q_tile), :], batched_iota_b)

    # build block_tables_sbuf_v2 from block_tables and k_offsets
    max_num_seqs, max_blocks_per_seq = block_tables.shape
    assert seqused_k.shape[-1] == max_num_seqs, (
        f"Sequence dimension mismatch: expected {max_num_seqs} but got {seqused_k.shape[-1]}"
    )

    block_tables_sbuf_v2 = build_tiled_block_tables(
        block_tables,
        k_offsets,
        sequence_ids,
        loop_index,
        broadcasted_iota_a_vec,
        broadcasted_iota_b_vec,
        num_tiles_unrolled,
        num_loads,
        num_blocks_per_large_tile,
        tiled_block_size,
        block_size_tiling_factor,
        B_P_SIZE,
    )

    tile_q_indices_sbuf_v2 = nl.ndarray(
        (INNER_Q_TILE_SIZE, n_small_in_large_q_tile, num_tiles_unrolled),
        dtype=nl.int32,
        buffer=nl.sbuf,
    )

    for local_tile_idx in nl.sequential_range(num_tiles_unrolled):
        local_tile_idx_val = batched_iota_b_vec[0, 0, local_tile_idx]

        cur_k_tile, cur_v_tile = load_kv_tile_from_cache(
            kv_cache=kv_cache,
            block_tables=block_tables_sbuf_v2[:, :, :],  # block_tables_sbuf,
            large_k_tile_idx=local_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=tiled_block_size,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            identity_for_transpose=identity_for_transpose_k,
        )
        # load aggregation buffer and q from HBM
        q_sbuf_tile_transposed = nl.ndarray(
            (
                par_dim(B_D_SIZE),
                q_h_per_k_h,
                n_small_in_large_q_tile,
                INNER_Q_TILE_SIZE,
            ),
            dtype=query.dtype,
            buffer=nl.sbuf,
        )
        # XXX: nl.zeros due to DMA skipping, otherwise, will get NaNs
        q_sbuf_tmp = nl.zeros(
            (
                par_dim(INNER_Q_TILE_SIZE),
                n_small_in_large_q_tile,
                q_h_per_k_h,
                B_D_SIZE,
            ),
            dtype=kernel_dtype,
            buffer=nl.sbuf,
        )

        # Calculate global tile index based on loop index and unroll factor
        # g_tile_idx = loop_index * num_tiles_unrolled + local_tile_idx_val
        global_tile_idx = nisa.tensor_scalar(
            loop_index,
            nl.multiply,
            float(num_tiles_unrolled),
            op1=nl.add,
            operand1=nl.copy(local_tile_idx_val, dtype=nl.float32),
            dtype=nl.int32,
        )
        seq_idx = nl.copy(nl.load(sequence_ids[global_tile_idx]), dtype=nl.int32)
        next_seq_idx = nisa.tensor_scalar(
            seq_idx,
            nl.add,
            1.0,
            op1=nl.minimum,
            operand1=float(max_num_seqs),
            dtype=nl.int32,
        )

        cu_nr_begin = (
            nl.copy(nl.load(cu_seqlens_q[seq_idx]), dtype=nl.float32)
            .reshape(shape=(1, 1))
            .broadcast_to([INNER_Q_TILE_SIZE, 1])
        )
        cu_nr_br = (
            nl.copy(nl.load(cu_seqlens_q[next_seq_idx]), dtype=nl.float32)
            .reshape(shape=(1, 1))
            .broadcast_to([INNER_Q_TILE_SIZE, 1])
        )
        nr_br = (
            nl.copy(nl.load(seqused_q[seq_idx]), dtype=nl.float32)
            .reshape(shape=(1, 1))
            .broadcast_to([INNER_Q_TILE_SIZE, 1])
        )
        nc_br = (
            nl.copy(nl.load(seqused_k[seq_idx]), dtype=nl.float32)
            .reshape(shape=(1, 1))
            .broadcast_to([INNER_Q_TILE_SIZE, 1])
        )
        q_offset = nl.copy(nl.load(q_offsets[global_tile_idx]), dtype=nl.float32)
        k_offset = nl.copy(nl.load(k_offsets[global_tile_idx]), dtype=nl.float32)
        q_tile_offset = nl.subtract(q_offset, cu_nr_begin)

        k_length = nl.subtract(nc_br, k_offset)
        q_length = nl.subtract(cu_nr_br, q_offset)

        ## solution 2: mask both cached and new tokens
        col_mask = nisa.tensor_scalar(batched_iota_b, nl.less, k_length, dtype=nl.bool_)
        row_mask = nisa.tensor_scalar(
            batched_iota_a, nl.less, q_length, dtype=nl.bool_
        )  # nr_br, dtype=nl.bool_)

        # sliding_window_limit = batched_iota_a + (nc_br + q_offset + 1 - k_offset)
        # nc_br: num of cached tokens per sequence
        # nr_br: num of new tokens per sequence
        # q_offset: tiling offset for query-dimension
        # k_offset: tiling offset for kv dimension
        ##### TODO:
        # 1. cu_nr_br should start with 0, and cumulative thereafter
        # 2. load sequence-level query_start and query_end index from cu_nr_br
        # 3. calculate tile-level query-offset to replace q_offset value in the above equation
        base_offset = nl.add(
            q_tile_offset, nc_br
        )  #### UPDATE HERE <----------------<<<--------------
        base_offset = nisa.tensor_scalar(base_offset, nl.add, 1.0)
        base_offset = nl.subtract(base_offset, k_offset)

        # Add this offset to the row indices
        sliding_window_limit = nl.add(batched_iota_a, base_offset)

        # Create the sliding window mask
        sliding_window_mask = nisa.tensor_tensor(
            batched_iota_b, sliding_window_limit, nl.less, dtype=nl.bool_
        )

        # Combine all mask conditions
        tile_mask = nisa.tensor_tensor(
            row_mask, sliding_window_mask, nl.logical_and, dtype=nl.bool_
        )
        # tile_mask = nisa.tensor_tensor(tile_mask, col_mask, nl.logical_and, dtype=nl.bool_)
        tile_mask_tr = nl.copy(tile_mask)

        ## DEBUGGING ONLY
        if tile_mask_hbm is not None:
            nl.store(
                tile_mask_hbm[:, nl.ds(0, n_small_in_large_q_tile), :], tile_mask[...]
            )

        transpose_kv_token_mask(
            tile_mask.reshape(
                [INNER_Q_TILE_SIZE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE]
            ),
            tile_mask_tr.reshape(
                [INNER_Q_TILE_SIZE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE]
            ),
            tiled_block_size=tiled_block_size,
            B_P_SIZE=128,
        )

        ## compute tile_q_indices
        # iota_a_vec = [0,1], q_offset=[0,128,154], nr_br=[154,154,1], cu_nr_br=[154,154,155]
        # expected: [[0,1,....,127],[128,129,....153,2560,....2560],[154,2560,2560,...]]
        _, _, max_num_queries, _ = query.shape
        tile_q_indices_sbuf_val = nl.add(batched_iota_a_vec, q_offset)
        tile_q_indices_sbuf_cond = nisa.tensor_scalar(
            tile_q_indices_sbuf_val, nl.less, cu_nr_br, dtype=nl.bool_
        )
        tile_q_indices_sbuf_mask = nisa.tensor_scalar(
            tile_q_indices_sbuf_val,
            nl.greater_equal,
            cu_nr_br,
            op1=nl.multiply,
            operand1=float(max_num_queries - 1),
            dtype=nl.float32,
        )
        tile_q_indices_sbuf_raw = nl.where(
            tile_q_indices_sbuf_cond, tile_q_indices_sbuf_val, tile_q_indices_sbuf_mask
        )
        # tile_q_indices_sbuf_v2 = nl.copy(tile_q_indices_sbuf_raw, dtype=nl.int32).reshape([INNER_Q_TILE_SIZE, n_small_in_large_q_tile])
        tile_q_indices_sbuf_v2[:, :, local_tile_idx] = nl.copy(
            tile_q_indices_sbuf_raw, dtype=nl.int32
        ).reshape([INNER_Q_TILE_SIZE, n_small_in_large_q_tile, 1])

        # nl.store(tile_q_indices_hbm[:, :n_small_in_large_q_tile, :], tile_q_indices_sbuf)
        if tile_q_indices_hbm is not None:
            nl.store(
                tile_q_indices_hbm[global_tile_idx, :, :n_small_in_large_q_tile, :],
                tile_q_indices_sbuf_v2,
            )

        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None]
                i_f = nl.arange(B_D_SIZE)[None, :]
                q_sbuf_tmp[i_p, small_q_idx, i_q_h, i_f] = nl.load(
                    query[
                        batch_id,
                        head_id * q_h_per_k_h + i_q_h,
                        tile_q_indices_sbuf_v2[i_p, small_q_idx, local_tile_idx],
                        i_f,
                    ],
                    mode=oob_mode.skip,
                )
                q_tile_scaled = nl.multiply(
                    q_sbuf_tmp[:, small_q_idx, i_q_h, :],
                    softmax_scale,
                    dtype=kernel_dtype,
                )
                PF_transpose_with_PE(
                    q_tile_scaled,
                    q_sbuf_tile_transposed[:, i_q_h, small_q_idx, :],
                    identity_for_transpose=identity_for_transpose_p,
                    out_in_psum=False,
                )

        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            for i_q_h in nl.affine_range(q_h_per_k_h):
                q_tile = q_sbuf_tile_transposed[:, i_q_h, small_q_idx]

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    olm_buffer=olm_unrolled_sbuf[:, local_tile_idx, small_q_idx, i_q_h],
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    tile_mask=tile_mask_tr[:, small_q_idx, :],
                    identity_for_transpose=identity_for_transpose_p,
                    use_causal_mask=False,
                    q_tile_idx=None,
                    Q_TILE_SIZE=INNER_Q_TILE_SIZE,
                    LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                )

        nisa.tensor_copy_predicated(
            src=olm_unrolled_sbuf[:, local_tile_idx],
            dst=olm_unrolled_sbuf[:, local_tile_idx + 1],
            predicate=q_update_pred_broadcast[:, local_tile_idx],
        )

    for local_tile_idx in nl.affine_range(num_tiles_unrolled):
        # write out aggregation buffer
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            i_p = nl.arange(INNER_Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(B_D_SIZE + 2)[None, None, :]
            nl.store(
                olm_buffer_hbm[
                    tile_q_indices_sbuf_v2[i_p, small_q_idx, local_tile_idx],
                    i_f_h,
                    i_f_d,
                ],
                olm_unrolled_sbuf[i_p, local_tile_idx, small_q_idx, i_f_h, i_f_d],
                mode=oob_mode.skip,
            )


def epilogue_prefill(
    o,
    query,
    key,
    value,
    active_mask,
    softmax_scale,
    olm_buffer,
    ACTIVE_Q_TILE_SIZE,
    seqlen_q,
    batch_id,
    head_id,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    B_F_SIZE,
    B_D_SIZE,
    skip_active,
):
    B_P_SIZE = 128
    # -------- Load l, m, o back to SBUF from HBM ------------ #
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0

    olm_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
    )
    for i in nl.affine_range(num_active_tiles):
        olm_buffer_sbuf[:, i] = nl.load(
            olm_buffer[nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE)]
        )

    # -------- write output to buffer on HBM ------------ #
    for i in nl.affine_range(num_active_tiles):
        out = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=kernel_dtype
        )
        # WARNING: lse is 0 in padded tokens
        # lse_epsilon = nisa.tensor_scalar(l_buffer_sbuf[:, i], nl.add, 1e-5, dtype=acc_type)
        lse_epsilon = nisa.tensor_scalar(
            olm_buffer_sbuf[:, i, :, nl.ds(B_D_SIZE, 1)], nl.add, 1e-5, dtype=acc_type
        )
        lse_reciprocal = nisa.reciprocal(lse_epsilon, dtype=acc_type)
        out[...] = nl.multiply(
            olm_buffer_sbuf[:, i, :, nl.ds(0, B_D_SIZE)],  # o_buffer_sbuf[:, i],
            lse_reciprocal,
            dtype=kernel_dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            nl.store(
                o[
                    # batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    nl.ds(i * ACTIVE_Q_TILE_SIZE, ACTIVE_Q_TILE_SIZE),
                    :,
                ],
                out[:, i_q_h, :],
            )


def init_q_update_pred(MAX_NUM_TILE, last_tile_indices_sbuf):
    B_P_SIZE = nl.tile_size.pmax
    memset_tile_size = min(B_P_SIZE, MAX_NUM_TILE)
    num_memset_tiles = ceil_div(MAX_NUM_TILE, memset_tile_size)
    update_pred = nl.ndarray((MAX_NUM_TILE, 1), dtype=nl.uint8, buffer=nl.hbm)
    one = nl.ones((memset_tile_size, num_memset_tiles), dtype=nl.uint8)
    i_p = nl.arange(memset_tile_size)[:, None]
    i_f = nl.arange(num_memset_tiles)[None, :]
    nisa.dma_copy(
        dst=update_pred.reshape([memset_tile_size, num_memset_tiles])[i_p, i_f],
        src=one,
        mask=((i_p * num_memset_tiles + i_f) < MAX_NUM_TILE),
        dge_mode=nisa.dge_mode.swdge,
    )
    return update_pred


def prepare_q_update_pred(update_pred, last_tile_indices_sbuf):
    q_tile_size, num_tile = last_tile_indices_sbuf.shape
    i_p = nl.arange(q_tile_size)[:, None]
    i_f = nl.arange(1)[None, :]
    zero = nl.zeros((q_tile_size, 1), dtype=nl.uint8)
    for i in nl.affine_range(num_tile):
        nisa.dma_copy(
            dst=update_pred[last_tile_indices_sbuf[i_p, i], i_f],
            src=zero,
            dge_mode=nisa.dge_mode.swdge,
        )
    return update_pred


def load_and_broadcast_q_update_preds(q_update_pred, loop_index, B_D_SIZE):
    B_F_SIZE = 512
    _, free_dim_size = q_update_pred.shape
    padded_free_dim_size = (
        free_dim_size
        if free_dim_size <= B_F_SIZE
        else pad_to_multiple(free_dim_size, B_F_SIZE)
    )
    out = nl.zeros((par_dim(1), padded_free_dim_size), dtype=nl.uint8)
    i_p = nl.arange(1)[:, None]
    i_f = nl.arange(free_dim_size)[None, :]
    out[i_p, i_f] = nl.load(  # TODO: avoid DMA_DIRECT2D
        q_update_pred[loop_index[i_p, 0], i_f],
        dtype=nl.uint8,
    )
    out_broadcast = nl.ndarray(
        (par_dim(B_D_SIZE), padded_free_dim_size), dtype=nl.uint8
    )
    tile_size = min(B_F_SIZE, free_dim_size)
    num_tiles = padded_free_dim_size // tile_size
    out_broadcast_reshape = out_broadcast.reshape((B_D_SIZE, num_tiles, tile_size))
    if nisa.get_nc_version() == nisa.nc_version.gen3:
        out_broadcast = nl.broadcast_to(out, shape=(B_D_SIZE, padded_free_dim_size))
    else:
        out_broadcast = nl.ndarray(
            (par_dim(B_D_SIZE), padded_free_dim_size), dtype=nl.uint8
        )
        tile_size = min(B_F_SIZE, free_dim_size)
        num_tiles = padded_free_dim_size // tile_size
        out_broadcast_reshape = out_broadcast.reshape((B_D_SIZE, num_tiles, tile_size))
        for i in nl.affine_range(num_tiles):
            broadcast_partition_with_PE(
                src=out[:, nl.ds(i * tile_size, tile_size)],
                out=out_broadcast_reshape[:, i, :],
                src_one_zero=True,
                out_in_psum=False,
            )
    return out, out_broadcast


def decode_subkernel(
    query,
    key,
    value,
    kv_cache,
    sequence_ids,
    q_offsets,
    k_offsets,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    q_update_pred,
    last_tile_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_seqs,
    block_tables,
    o,
    dynamic_loop_trip_count,
    loop_unroll_factor,
    batch_id,
    head_id,
    seqlen_q,
    num_blocks_per_large_tile,
    tiled_block_size,
    block_size_tiling_factor,
    kernel_dtype,
    acc_type,
    k_h,
    q_h_per_k_h,
    B_D_SIZE,
    softmax_scale,
    skip_active,
    tile_mask_hbm,
):
    B_P_SIZE = 128
    NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses
    # max_num_tiles = tile_masks.shape[1]
    (max_num_tiles,) = q_offsets.shape
    LARGE_Q_TILE_SIZE = 1
    INNER_Q_TILE_SIZE = 1
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // INNER_Q_TILE_SIZE
    if tile_masks is not None:
        assert tile_masks.shape[1] == max_num_tiles, f"invalid {tile_masks.shape=}"
    assert max_num_tiles % loop_unroll_factor == 0

    last_tile_indices_sbuf = nl.load(last_tile_indices)
    q_update_pred_hbm = init_q_update_pred(max_num_tiles, last_tile_indices_sbuf)
    (olm_buffer,) = allocate_decode_accum_buffers(
        MAX_NUM_TILE=max_num_tiles,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        acc_type=acc_type,
    )
    olm_next_tile_sbuf = nl.zeros(
        (par_dim(B_D_SIZE), 3, q_h_per_k_h),
        dtype=acc_type,
    )
    olm_next_tile_sbuf[:, nl.ds(2, 1), :] = NEG_INF

    identity_for_transpose_o = create_identity_for_transpose(
        olm_next_tile_sbuf, B_D_SIZE
    )  # TODO: avoid DMA_DIRECT2D

    # transpose identity matrix for m_buffer on hbm
    identity_for_transpose_m_step1_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=acc_type,
    )
    identity_for_transpose_m_step2_hbm = nl.shared_constant(
        np.identity(n=q_h_per_k_h, dtype=np.uint8),
        dtype=acc_type,
    )
    identity_for_transpose_k_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=kernel_dtype,
    )

    identity_for_transpose_query = create_identity_for_transpose(
        query,  # kernel_dtype
        loop_unroll_factor,  # local_q_addr.shape[0]
    )
    identity_for_transpose_k = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=kernel_dtype)
    identity_for_transpose_m_step1 = nl.ndarray((B_P_SIZE, B_P_SIZE), dtype=acc_type)
    identity_for_transpose_m_step2 = nl.ndarray(
        (q_h_per_k_h, q_h_per_k_h), dtype=acc_type
    )
    nisa.dma_copy(
        dst=identity_for_transpose_k,
        src=identity_for_transpose_k_hbm,
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=identity_for_transpose_m_step1,
        src=identity_for_transpose_m_step1_hbm,
        dge_mode=nisa.dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=identity_for_transpose_m_step2,
        src=identity_for_transpose_m_step2_hbm,
        dge_mode=nisa.dge_mode.swdge,
    )

    stride_size = B_P_SIZE // block_size_tiling_factor
    broadcasted_iota_a_hbm = nl.shared_constant(
        np.arange(stride_size).repeat(block_size_tiling_factor), dtype=nl.int32
    )
    broadcasted_iota_a_vec = nl.load(
        broadcasted_iota_a_hbm[nl.arange(B_P_SIZE)[:, None]]
    )
    broadcasted_iota_b_hbm = nl.shared_constant(
        np.tile(np.arange(block_size_tiling_factor), stride_size), dtype=nl.int32
    )
    broadcasted_iota_b_vec = nl.load(
        broadcasted_iota_b_hbm[nl.arange(B_P_SIZE)[:, None]]
    )

    num_loads = num_blocks_per_large_tile // B_P_SIZE
    num_k_tiles = num_loads * tiled_block_size
    tile_mask_const = np.arange(B_P_SIZE * num_k_tiles).reshape(
        num_k_tiles, B_P_SIZE
    )  # (num_k_tiles, B_P_SIZE)
    tile_mask_const = (
        tile_mask_const.reshape(
            num_k_tiles // tiled_block_size, B_P_SIZE, tiled_block_size
        )
        .transpose(1, 0, 2)
        .reshape(B_P_SIZE, num_k_tiles)
    )
    tile_mask_iota_hbm = nl.shared_constant(tile_mask_const, dtype=nl.int32)
    tile_mask_iota = nl.load(tile_mask_iota_hbm[:, :])

    q_update_pred_hbm = prepare_q_update_pred(q_update_pred_hbm, last_tile_indices_sbuf)
    q_update_pred = q_update_pred_hbm.reshape(
        (max_num_tiles // loop_unroll_factor, loop_unroll_factor)
    )

    # nl.device_print("flash_pa_with_schedule.py: decode_subkernel(): q_update_pred=", q_update_pred.reshape([1, max_num_tiles]))
    # nl.device_print("flash_pa_with_schedule.py: decode_subkernel(): last_tile_indices_sbuf=", last_tile_indices_sbuf)

    # tile_mask_loaded = nl.load(tile_masks)
    # nl.store(tile_mask_hbm, tile_mask_loaded)

    def _cdiv(x, y):
        return nl.ceil(nl.divide(x, y), dtype=nl.int32)

    cond_dtype = np.int32
    cond = nl.ndarray((1, 1), buffer=nl.hbm, dtype=cond_dtype)
    # index = nl.ndarray((1, 1), buffer=nl.hbm, dtype=np.int32)
    # nl.store(dst=index[0, 0], value=0)
    cond_var = True
    # dynamic_loop_conds = dynamic_loop_conds.reshape((dynamic_loop_conds.shape[0], 1))
    # dynamic_loop_conds = dynamic_loop_trip_count.reshape((dynamic_loop_trip_count.shape[0], 1))

    trip_count_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
    dynamic_for_ub_hbm = nl.ndarray((1, 1), buffer=nl.hbm, dtype=nl.int32)
    nisa.dma_copy(
        dst=trip_count_sbuf[0, 0],
        src=dynamic_loop_trip_count[0],
        dge_mode=nisa.dge_mode.swdge,
    )
    nl.store(
        dynamic_for_ub_hbm,
        nl.copy(
            _cdiv(
                nl.copy(trip_count_sbuf, dtype=nl.float32),
                nisa.memset(shape=(1, 1), value=loop_unroll_factor, dtype=nl.float32),
            ),
            dtype=nl.int32,
        ),
    )
    dynamic_for_ub = nl.load(dynamic_for_ub_hbm, dtype=nl.int32)
    # loop_index = nl.load(index[0], dtype=np.int32)
    loop_index = nisa.memset(shape=(1, 1), value=0, dtype=nl.int32)

    # num_max_step = (dynamic_loop_conds.shape[0] - 1) // loop_unroll_factor
    # for loop_step in nl.sequential_range(num_max_step):
    #     loop_index = nisa.iota(loop_step, dtype=nl.int32)
    # for loop_step in nl.sequential_range(1):
    #     loop_index = nisa.iota(loop_step, dtype=nl.int32)
    # for _ in range(scalar(dynamic_for_ub)):
    #     loop_index = nl.load(index[0], dtype=np.int32)
    # while scalar(cond_var):
    # for _ in range(scalar(dynamic_for_ub)):
    #     loop_index = nl.load(index[0], dtype=np.int32)
    # for loop_step in nl.sequential_range(ceil_div(5, loop_unroll_factor)):
    #     loop_index[...] = nisa.iota(loop_step, dtype=nl.int32)
    for _ in range(scalar(dynamic_for_ub)):
        o_buffer_sbuf = nl.zeros(
            (par_dim(B_D_SIZE), loop_unroll_factor + 1, q_h_per_k_h),
            dtype=acc_type,
        )
        m_buffer_sbuf = nl.full(
            (par_dim(1), loop_unroll_factor + 1, q_h_per_k_h),
            NEG_INF,
            dtype=acc_type,
        )
        l_buffer_sbuf = nl.zeros(
            (par_dim(1), loop_unroll_factor + 1, q_h_per_k_h),
            dtype=acc_type,
        )

        # trip_count_sbuf = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)
        # trip_count_sbuf[0, 0] = nl.load(dynamic_loop_trip_count[0])

        decode_prior(
            loop_index=nisa.tensor_copy(loop_index[...]),
            num_tiles_unrolled=loop_unroll_factor,
            query=query,
            kv_cache=kv_cache,
            # lmo_buffer=lmo_buffer,
            # m_next_tile = m_next_tile,
            # l_next_tile = l_next_tile,
            # o_next_tile = o_next_tile,
            olm_buffer=olm_buffer,
            olm_next_tile_sbuf=olm_next_tile_sbuf,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_seqs=num_seqs,
            block_tables=block_tables,
            sequence_ids=sequence_ids,
            q_offsets=q_offsets,
            k_offsets=k_offsets,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            q_update_pred=q_update_pred,
            batch_id=batch_id,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            block_size_tiling_factor=block_size_tiling_factor,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            head_id=head_id,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            identity_for_transpose_o=identity_for_transpose_o,
            identity_for_transpose_m_step1=identity_for_transpose_m_step1,
            identity_for_transpose_m_step2=identity_for_transpose_m_step2,
            identity_for_transpose_query=identity_for_transpose_query,
            identity_for_transpose_k=identity_for_transpose_k,
            broadcasted_iota_a_vec=broadcasted_iota_a_vec,
            broadcasted_iota_b_vec=broadcasted_iota_b_vec,
            tile_mask_iota=tile_mask_iota,
            tile_mask_hbm=tile_mask_hbm,
        )
        # nl.device_print("flash_pa_with_schedule.py: lmo_buffer[:8,0,0]=", lmo_buffer[:8,0,0])
        # nl.device_print("flash_pa_with_schedule.py: o_next_tile[0,0]=", o_next_tile[0,0])

        # # update index
        loop_index[...] = nl.add(loop_index[...], 1)
        # nl.store(index[0], loop_index_next[0, 0])
        # update conditions
        # cond_next = nl.load(dynamic_loop_conds[loop_index_next * loop_unroll_factor])
        # cond_next = nisa.tensor_tensor(loop_index_next * loop_unroll_factor, trip_count_sbuf, nl.less)
        # nl.store(dst=cond, value=cond_next)
        # cond_var = cond

    assert last_tile_indices_sbuf.dtype == nl.int32, (
        f"{last_tile_indices_sbuf.dtype=} expected to be nl.int32"
    )
    # query location of the first decode sequence
    query_indices = nl.load(q_offsets[last_tile_indices_sbuf])
    epilogue_decode(
        o=o,
        query=query,
        key=key,
        value=value,
        olm_buffer=olm_buffer,
        # q_offset=q_offset,
        query_indices=query_indices,
        softmax_scale=softmax_scale,
        last_tile_indices_sbuf=last_tile_indices_sbuf,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_D_SIZE=B_D_SIZE,
        skip_active=True if key is None else skip_active,
    )


def load_decode_query(
    q_offsets,
    offsets,
    query,
    tile_q_indices,
    identity_for_transpose,
    softmax_scale,
    batch_id,
    head_id,
    q_h_per_k_h,
    B_D_SIZE,
    kernel_dtype,
):
    (max_num_tiles,) = q_offsets.shape
    load_size = offsets.shape[0]
    i_p = nl.arange(load_size)[:, None]
    i_f = nl.arange(1)[None, :]
    q_indices = nl.ndarray((load_size, 1), dtype=nl.int32)
    # q_indices[i_p, i_f] = nl.load(tile_q_indices[offsets[i_p, 0], i_f], dtype=nl.uint32)
    q_indices[i_p, i_f] = nl.load(
        q_offsets.reshape([max_num_tiles, 1])[offsets[i_p, 0], i_f], dtype=nl.uint32
    )
    query_scaled_transposed = nl.ndarray(
        (B_D_SIZE, load_size, q_h_per_k_h),
        dtype=kernel_dtype,
    )
    q_tile = nl.ndarray((load_size, q_h_per_k_h, B_D_SIZE), dtype=query.dtype)
    i_p = nl.arange(load_size)[:, None]
    i_f = nl.arange(B_D_SIZE)[None, :]
    for i_q_h in nl.affine_range(q_h_per_k_h):
        q_tile[i_p, i_q_h, i_f] = nl.load(
            query[
                batch_id,
                head_id * q_h_per_k_h + i_q_h,
                q_indices[i_p, 0],
                i_f,
            ],
            mode=oob_mode.skip,
        )
        q_tile_scaled = nisa.tensor_scalar(
            q_tile[i_p, i_q_h, i_f],
            nl.multiply,
            softmax_scale,
            dtype=kernel_dtype,
            engine=nisa.vector_engine,
        )
        PF_transpose_with_PE(
            q_tile_scaled,
            query_scaled_transposed[:, :, i_q_h],
            identity_for_transpose=identity_for_transpose,
            out_in_psum=False,
        )
    return query_scaled_transposed


def decode_prior(
    loop_index,
    num_tiles_unrolled,
    query,
    kv_cache,
    # lmo_buffer,
    # m_next_tile,
    # l_next_tile,
    # o_next_tile,
    olm_buffer,
    olm_next_tile_sbuf,
    o_buffer_sbuf,
    l_buffer_sbuf,
    m_buffer_sbuf,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    num_seqs,
    block_tables,
    sequence_ids,
    q_offsets,
    k_offsets,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    q_update_pred,
    batch_id,
    num_blocks_per_large_tile,
    tiled_block_size,
    block_size_tiling_factor,
    kernel_dtype,
    acc_type,
    head_id,
    k_h,
    q_h_per_k_h,
    B_D_SIZE,
    softmax_scale,
    identity_for_transpose_o,
    identity_for_transpose_m_step1,
    identity_for_transpose_m_step2,
    identity_for_transpose_query,
    identity_for_transpose_k,
    broadcasted_iota_a_vec,
    broadcasted_iota_b_vec,
    tile_mask_iota,
    tile_mask_hbm,
):
    MULTI_BUFFER_SIZE = 2
    MULTI_BUFFER_SIZE = min(MULTI_BUFFER_SIZE, num_tiles_unrolled)
    # B_P_SIZE, MAX_NUM_TILE, num_k_tiles = tile_masks.shape
    B_P_SIZE = 128
    (MAX_NUM_TILE,) = q_offsets.shape
    # assert B_P_SIZE == 128
    assert q_h_per_k_h <= B_P_SIZE

    num_loads = num_blocks_per_large_tile // B_P_SIZE
    num_k_tiles = num_loads * tiled_block_size
    # assert num_k_tiles == num_loads * block_size

    block_tables_sbuf = (
        load_indices_for_loop_step(
            tile_block_tables,
            loop_index,
            num_tiles_unrolled,
        )
        if tile_block_tables is not None
        else None
    )
    block_tables_sbuf = (
        transform_block_tables_for_indirect_load(
            block_tables_sbuf,
            block_size_tiling_factor=block_size_tiling_factor,
            num_head=k_h,
            head_id=head_id,
        )
        if tile_block_tables is not None
        else None
    )

    # offsets = nisa.iota(nl.arange(num_tiles_unrolled)[:, None], dtype=nl.uint32)
    base_addr = nisa.iota(nl.arange(num_tiles_unrolled)[:, None], dtype=nl.uint32)
    local_q_addr = nl.add(
        base_addr,
        nl.multiply(loop_index, num_tiles_unrolled, dtype=nl.float32),
        dtype=nl.uint32,
    )
    # local_q_offsets = nl.load(q_offsets[local_q_addr])

    # load all q and multiply scale
    query_sbuf = load_decode_query(
        q_offsets,
        offsets=local_q_addr,  # offsets,
        query=query,
        tile_q_indices=tile_q_indices,
        identity_for_transpose=identity_for_transpose_query,
        softmax_scale=softmax_scale,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        kernel_dtype=kernel_dtype,
    )

    block_tables_sbuf = build_tiled_block_tables(
        block_tables,
        k_offsets,
        sequence_ids,
        loop_index,
        broadcasted_iota_a_vec,
        broadcasted_iota_b_vec,
        num_tiles_unrolled,
        num_loads,
        num_blocks_per_large_tile,
        tiled_block_size,
        block_size_tiling_factor,
        B_P_SIZE,
    )

    # XXX: no dma skipping for decode kernel, no need to zero
    k_load_buffer = nl.ndarray(
        (par_dim(B_D_SIZE), MULTI_BUFFER_SIZE, num_loads * tiled_block_size * B_P_SIZE),
        dtype=kernel_dtype,
    )
    v_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER_SIZE, num_loads * tiled_block_size * B_D_SIZE),
        dtype=kernel_dtype,
    )

    # load first iteration of key cache
    k_load_buffer_reshaped = k_load_buffer.reshape(
        (B_D_SIZE, MULTI_BUFFER_SIZE, num_loads, tiled_block_size, B_P_SIZE),
    )
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]
        loaded = nl.ndarray(
            (B_P_SIZE, tiled_block_size * B_D_SIZE), dtype=kv_cache.dtype
        )
        loaded[i_p, i_f] = nl.load(
            kv_cache[0, block_tables_sbuf[i_p, load_idx, 0], i_f]
        )
        for tb_i in nl.affine_range(tiled_block_size):
            if loaded.dtype != kernel_dtype:
                k_src = nl.copy(
                    loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)],
                    dtype=kernel_dtype,
                )
            else:
                k_src = loaded[:, nl.ds(tb_i * B_D_SIZE, B_D_SIZE)]
            PF_transpose_with_PE(
                src=k_src,
                out=k_load_buffer_reshaped[:, 0, load_idx, tb_i],
                identity_for_transpose=identity_for_transpose_k,
            )

    # load value cache
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(tiled_block_size * B_D_SIZE)[None, :]
        if kernel_dtype == kv_cache.dtype:
            v_load_buffer[i_p, 0, i_f + load_idx * tiled_block_size * B_D_SIZE] = (
                nl.load(
                    kv_cache[1, block_tables_sbuf[i_p, load_idx, 0], i_f],
                )
            )
        else:
            loaded = nl.ndarray(
                (B_P_SIZE, tiled_block_size * B_D_SIZE), dtype=kv_cache.dtype
            )
            loaded[...] = nl.load(
                kv_cache[1, block_tables_sbuf[i_p, load_idx, 0], i_f],
            )
            v_load_buffer[i_p, 0, i_f + load_idx * tiled_block_size * B_D_SIZE] = (
                nl.copy(
                    loaded,
                    dtype=kernel_dtype,
                )
            )

    NEG_INF = -9984.0  # Magic number to replace -inf similar to what Tensorizer uses

    # TODO: optimize away the static DMA with
    o_buffer_sbuf[:, 0, :] = nl.copy(olm_next_tile_sbuf[:, 0, :])
    l_buffer_sbuf[:, 0, :] = nl.copy(olm_next_tile_sbuf[nl.ds(0, 1), 1, :])
    m_buffer_sbuf[:, 0, :] = nl.copy(olm_next_tile_sbuf[nl.ds(0, 1), 2, :])
    # o_buffer_sbuf = nl.zeros(
    #     (par_dim(B_D_SIZE), num_tiles_unrolled + 1, q_h_per_k_h),
    #     dtype=acc_type,
    # )
    # m_buffer_sbuf = nl.full(
    #     (par_dim(1), num_tiles_unrolled + 1, q_h_per_k_h),
    #     NEG_INF,
    #     dtype=acc_type,
    # )
    # l_buffer_sbuf = nl.zeros(
    #     (par_dim(1), num_tiles_unrolled + 1, q_h_per_k_h),
    #     dtype=acc_type,
    # )
    # o_buffer_sbuf[:, 0, :] = nl.load(o_next_tile)
    # l_buffer_sbuf[:, 0, :] = nl.load(l_next_tile)
    # m_buffer_sbuf[:, 0, :] = nl.load(m_next_tile)

    # prepare tile_mask
    if tile_masks is not None:
        tile_mask_sbuf = nl.ndarray(
            (par_dim(B_P_SIZE), num_tiles_unrolled, num_k_tiles),
            dtype=tile_masks.dtype,
        )
        tile_masks = tile_masks.reshape(
            (
                B_P_SIZE,
                MAX_NUM_TILE // num_tiles_unrolled,
                num_tiles_unrolled,
                num_k_tiles,
            )
        )
        i_p = nl.arange(B_P_SIZE)[:, None, None]
        i_q = nl.arange(num_tiles_unrolled)[None, :, None]
        i_k = nl.arange(num_k_tiles)[None, None, :]
        tile_mask_sbuf[i_p, i_q, i_k] = nl.load(tile_masks[i_p, loop_index, i_q, i_k])

    # dynamic (in-tile) mask generation
    tile_mask_sbuf_v2 = nl.ndarray(
        (par_dim(B_P_SIZE), num_tiles_unrolled, num_k_tiles), dtype=nl.bool_
    )

    # nl.device_print("flash_pa_with_schedule.py: decode_prior(): q_update_pred=", q_update_pred.reshape([1, MAX_NUM_TILE]))
    q_update_pred_sbuf, q_update_pred_broadcast = load_and_broadcast_q_update_preds(
        q_update_pred,
        loop_index,
        B_D_SIZE,
    )

    kq_res_ones = nl.ones(
        (par_dim(B_P_SIZE), num_k_tiles, q_h_per_k_h),
        dtype=acc_type,
    )
    sumexp_ones = nl.ones((par_dim(B_P_SIZE), 1), dtype=kernel_dtype)

    # Calculate global tile index based on loop index and unroll factor
    # g_tile_idx = loop_index * num_tiles_unrolled + local_tile_idx_val
    global_tile_idx_base = nisa.tensor_scalar(
        loop_index, nl.multiply, float(num_tiles_unrolled), dtype=nl.float32
    )
    iota_b_vec = nisa.iota(nl.arange(0, num_tiles_unrolled)[:, None], dtype=nl.int32)
    global_tile_idx = nl.add(iota_b_vec, global_tile_idx_base, dtype=nl.int32)
    seq_id = nl.load(sequence_ids[global_tile_idx])
    k_offset = nl.copy(nl.load(k_offsets[global_tile_idx]), dtype=nl.float32)
    nc_br = nl.load(seqused_k[seq_id], dtype=nl.float32)
    tile_mask_threshold = nisa.nc_transpose(
        nl.subtract(nc_br, k_offset), engine=nisa.vector_engine
    ).broadcast_to([B_P_SIZE, num_tiles_unrolled])

    for local_tile_idx in nl.affine_range(num_tiles_unrolled):
        # Generate attention mask with iota
        # Use nl.less_equal to include active-token, use nl.less otherwise
        # tile_mask_sbuf_v2[:, local_tile_idx, :] = nisa.tensor_scalar(tile_mask_iota, nl.less_equal, nl.subtract(nc_br, k_offset), dtype=nl.bool_)
        tile_mask_sbuf_v2[:, local_tile_idx, :] = nisa.tensor_scalar(
            tile_mask_iota,
            nl.less_equal,
            tile_mask_threshold[:, local_tile_idx],
            dtype=nl.bool_,
        )

    for local_tile_idx in nl.sequential_range(num_tiles_unrolled):
        _flash_attention_core_kq_matmul(
            q_local_tile=query_sbuf,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            q_update_pred_sbuf=q_update_pred_sbuf,
            q_update_pred_broadcast=q_update_pred_broadcast,
            tile_mask=tile_mask_sbuf_v2[:, :, :],
            kv_cache=kv_cache,
            block_tables_sbuf=block_tables_sbuf,
            large_tile_idx=local_tile_idx,
            MULTI_BUFFER_SIZE=MULTI_BUFFER_SIZE,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=tiled_block_size,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            kq_res_ones=kq_res_ones,
            sumexp_ones=sumexp_ones,
            identity_for_transpose_k=identity_for_transpose_k,
            identity_for_transpose_m_step1=identity_for_transpose_m_step1,
            identity_for_transpose_m_step2=identity_for_transpose_m_step2,
        )

    # nl.store(o_next_tile, o_buffer_sbuf[:, num_tiles_unrolled, :])
    # nl.store(l_next_tile, l_buffer_sbuf[:, num_tiles_unrolled, :])
    # nl.store(m_next_tile, m_buffer_sbuf[:, num_tiles_unrolled, :])
    olm_next_tile_sbuf[:, 0, :] = nl.copy(o_buffer_sbuf[:, num_tiles_unrolled, :])
    olm_next_tile_sbuf[nl.ds(0, 1), 1, :] = nl.copy(
        l_buffer_sbuf[:, num_tiles_unrolled, :]
    )
    olm_next_tile_sbuf[nl.ds(0, 1), 2, :] = nl.copy(
        m_buffer_sbuf[:, num_tiles_unrolled, :]
    )

    # store L, M, O in (seqlen, h, d) format
    lmo_sbuf = nl.ndarray(
        (num_tiles_unrolled, q_h_per_k_h, B_D_SIZE + 2),
        dtype=o_buffer_sbuf.dtype,
    )
    identity_for_transpose_lm = nl.ones((1, 1), dtype=acc_type)
    for i_q_h in nl.affine_range(q_h_per_k_h):
        o_tmp = nl.ndarray(
            (num_tiles_unrolled, B_D_SIZE), dtype=acc_type, buffer=nl.psum
        )
        PF_transpose_with_PE(
            src=o_buffer_sbuf[:, nl.ds(0, num_tiles_unrolled), i_q_h],
            out=o_tmp,
            identity_for_transpose=identity_for_transpose_o,
            out_in_psum=True,
        )
        lmo_sbuf[:, i_q_h, nl.ds(0, B_D_SIZE)] = nl.copy(o_tmp)
        l_tmp = nl.ndarray((num_tiles_unrolled, 1), dtype=acc_type, buffer=nl.psum)
        PF_transpose_with_PE(
            src=l_buffer_sbuf[:, nl.ds(0, num_tiles_unrolled), i_q_h],
            out=l_tmp,
            identity_for_transpose=identity_for_transpose_lm,
            out_in_psum=True,
        )
        lmo_sbuf[:, i_q_h, nl.ds(B_D_SIZE, 1)] = nl.copy(l_tmp)
        m_tmp = nl.ndarray((num_tiles_unrolled, 1), dtype=acc_type, buffer=nl.psum)
        PF_transpose_with_PE(
            src=m_buffer_sbuf[:, nl.ds(0, num_tiles_unrolled), i_q_h],
            out=m_tmp,
            identity_for_transpose=identity_for_transpose_lm,
            out_in_psum=True,
        )
        lmo_sbuf[:, i_q_h, nl.ds(B_D_SIZE + 1, 1)] = nl.copy(m_tmp)
    i_q = nl.arange(num_tiles_unrolled)[:, None, None]
    i_h = nl.arange(q_h_per_k_h)[None, :, None]
    i_d = nl.arange(B_D_SIZE + 2)[None, None, :]
    # nl.store(lmo_buffer[offsets[i_q, 0], i_h, i_d], lmo_sbuf[i_q, i_h, i_d]) ### WARNING: offsets
    nl.store(
        olm_buffer[local_q_addr[i_q, 0], i_h, i_d], lmo_sbuf[i_q, i_h, i_d]
    )  ### WARNING: offsets


def epilogue_decode(
    o,
    query,
    key,
    value,
    olm_buffer,
    # q_offset,
    query_indices,
    softmax_scale,
    last_tile_indices_sbuf,
    seqlen_q,
    batch_id,
    head_id,
    q_h_per_k_h,
    kernel_dtype,
    acc_type,
    B_D_SIZE,
    skip_active,
):
    B_P_SIZE = 128
    ACTIVE_Q_TILE_SIZE = min(seqlen_q, B_P_SIZE)
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE
    # if seqlen_q > 32 and num_active_tiles == 1:
    #     # XXX: work around a weird accuracy bug
    #     num_active_tiles = 2
    #     ACTIVE_Q_TILE_SIZE = seqlen_q // num_active_tiles
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0
    assert last_tile_indices_sbuf.shape == (ACTIVE_Q_TILE_SIZE, num_active_tiles)

    o_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, B_D_SIZE),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    for i in nl.affine_range(num_active_tiles):
        lmo_tmp = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE + 2),
            dtype=acc_type,
        )
        i_q = nl.arange(ACTIVE_Q_TILE_SIZE)[:, None, None]
        i_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        # nl.device_print("flash_pa_with_schedule.py: last_tile_indices_sbuf[i_q, i]", last_tile_indices_sbuf[i_q, i])
        lmo_tmp[i_q, i_h, i_d] = nl.load(
            olm_buffer[last_tile_indices_sbuf[i_q, i], i_h, i_d]
        )
        o_buffer_sbuf[:, i, :, :] = nl.copy(lmo_tmp[:, :, nl.ds(0, B_D_SIZE)])
        l_buffer_sbuf[:, i, :, :] = nl.copy(lmo_tmp[:, :, nl.ds(B_D_SIZE, 1)])
        m_buffer_sbuf[:, i, :, :] = nl.copy(lmo_tmp[:, :, nl.ds(B_D_SIZE + 1, 1)])
        # nl.device_print("flash_pa_with_schedule.py: o_buffer_sbuf[:, i, 0, 0]", o_buffer_sbuf[:, i, 0, 0])
        # nl.device_print("flash_pa_with_schedule.py: l_buffer_sbuf[:, i, 0, 0]", l_buffer_sbuf[:, i, 0, 0])
        # nl.device_print("flash_pa_with_schedule.py: m_buffer_sbuf[:, i, 0, 0]", m_buffer_sbuf[:, i, 0, 0])

    # -------- write output to buffer on HBM ------------ #
    # base_addr_imm = nisa.iota(nl.arange(ACTIVE_Q_TILE_SIZE)[None, :], dtype=nl.float32)
    # base_addr_imm = nisa.iota(nl.arange(ACTIVE_Q_TILE_SIZE)[:, None], dtype=nl.float32)
    # query_indices = nisa.tensor_scalar(base_addr_imm, nl.add, nl.copy(q_offset.broadcast_to([ACTIVE_Q_TILE_SIZE, 1]), dtype=nl.float32), dtype=nl.int32)
    # nl.device_print("flash_pa_with_schedule.py: query_indices=", query_indices)
    for i in nl.affine_range(num_active_tiles):
        out = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), q_h_per_k_h, B_D_SIZE), dtype=kernel_dtype
        )
        # WARNING: lse is 0 in padded tokens
        lse_epsilon = nisa.tensor_scalar(
            l_buffer_sbuf[:, i], nl.add, 1e-5, dtype=acc_type
        )
        lse_reciprocal = nisa.reciprocal(lse_epsilon, dtype=acc_type)
        out[...] = nl.multiply(
            o_buffer_sbuf[:, i],
            # 1.0 / l_buffer_sbuf[:, i],  # XXX: l is 0 in padded tokens, warning in simulation
            lse_reciprocal,
            dtype=kernel_dtype,
        )
        ipp = nl.arange(ACTIVE_Q_TILE_SIZE)[:, None]
        # nl.device_print("flash_pa_with_schedule.py: out[ipp, i_q_h, iy]=", out[ipp, 0, 0])
        for i_q_h in nl.affine_range(q_h_per_k_h):
            ix, iy = nl.mgrid[0:1, 0:B_D_SIZE]
            nl.store(
                o[
                    # batch_id,
                    head_id * q_h_per_k_h + i_q_h,
                    query_indices,
                    iy,
                ],
                out[ipp, i_q_h, iy],
                mode=oob_mode.skip,  # DMA skipping
            )


def flash_attn_varlen_blocksparse_nkifunc(
    query,
    key,
    value,
    kv_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    last_tile_indices,
    q_update_pred,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    num_seqs,
    block_tables,
    dynamic_loop_trip_count=None,
    loop_unroll_factor=1,
    softmax_scale=None,
    skip_active=False,
    decode_mode=False,
    enable_prefill=True,
):
    N, n_blocks, n_kv_head, block_size, head_size = kv_cache.shape
    assert N == 2, f"invalid {kv_cache.shape=}"

    args = (
        query,
        key,
        value,
        kv_cache,
        tile_q_indices,
        tile_block_tables,
        tile_masks,
        active_mask,
        last_tile_indices,
        q_update_pred,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        num_seqs,
        block_tables,
        dynamic_loop_trip_count,
    )
    kwargs = dict(
        softmax_scale=1.0 / (head_size**0.5)
        if softmax_scale is None
        else softmax_scale,
        loop_unroll_factor=loop_unroll_factor,
        skip_active=skip_active,
        decode_mode=decode_mode,
        enable_prefill=enable_prefill,
    )

    # print(f"flash_pa_with_schedule.py: {n_kv_head=}")
    return flash_paged_attention_blockspase[1, n_kv_head](*args, **kwargs)


# @nki.compiler.skip_middle_end_transformations
# @nki.jit(enable_out_of_bound_check=False,
#          experimental_flags="experimental-native-scalar-support, experimental-local-tensor-parent", #
#          debug_kernel=True,
#          show_compiler_tb=True)
@NKIOpRegistry.register("mylib::flash_attn_varlen")
def flash_attn_varlen_nki(
    query,
    kv_cache,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    num_seqs,
    block_tables,
    enable_prefill=True,
):
    key = None
    value = None

    tile_q_indices = None
    tile_block_tables = None
    tile_masks = None
    active_mask = None
    last_tile_indices = None
    q_update_pred = None

    dynamic_loop_trip_count = None
    softmax_scale = None

    loop_unroll_factor = 8
    skip_active = True
    decode_mode = False

    large_q_tile_size = 128
    large_kv_tile_size = 1024

    mixed_precision = True

    # Tiling factors that should be ideally tunable
    B_P_SIZE = 128
    if decode_mode:
        LARGE_Q_TILE_SIZE = 1
        if tile_masks is not None:
            INNER_KV_TILE_SIZE, _, N_INNER_KV_TILE = tile_masks.shape
            assert INNER_KV_TILE_SIZE == B_P_SIZE
            LARGE_KV_TILE_SIZE = INNER_KV_TILE_SIZE * N_INNER_KV_TILE
        else:
            LARGE_KV_TILE_SIZE = large_kv_tile_size
    else:
        if tile_masks is not None:
            _, LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = tile_masks.shape
        else:
            LARGE_Q_TILE_SIZE, LARGE_KV_TILE_SIZE = (
                large_q_tile_size,
                large_kv_tile_size,
            )
    INNER_Q_TILE_SIZE = min(B_P_SIZE, LARGE_Q_TILE_SIZE)
    assert LARGE_Q_TILE_SIZE % INNER_Q_TILE_SIZE == 0
    MAX_NUM_TILES = 256  # 1024 # 512
    n_small_in_large_q_tile = LARGE_Q_TILE_SIZE // INNER_Q_TILE_SIZE

    b, h, seqlen_q, d = query.shape
    print(f"flash_pa_with_schedule.py: {b=}, {h=}, {seqlen_q=}, {d=}")
    assert is_power_of_2(seqlen_q), f"{seqlen_q=} is expected to be power of 2"
    assert seqlen_q <= 8192, f"Large {seqlen_q=} consumes too much sbuf space"
    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    # assert (
    #     d >= 16 and d <= 128 and is_power_of_2(d)
    # ), f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    B_D_SIZE = d
    _, num_blocks, k_h, block_size, _ = kv_cache.shape
    q_h_per_k_h = h // k_h
    assert tuple(kv_cache.shape) == (
        2,
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{kv_cache.shape=} mismatch!"
    # assert tuple(value_cache.shape) == (
    #     num_blocks,
    #     k_h,
    #     block_size,
    #     d,
    # ), f"{value_cache.shape=} mismatch!"
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

    if tile_masks is not None:
        assert tile_masks.dtype == nl.uint8, (
            f"{tile_masks.dtype=} is expected to be uint8"
        )
    if active_mask is not None:
        assert active_mask.dtype == nl.uint8, (
            f"{active_mask.dtype=} is expected to be uint8"
        )

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    print(f"flash_pa_with_schedule.py: {kernel_dtype=}, {acc_type=}")
    o = nl.ndarray((h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    # nl.store(o, 0)
    # tile_mask_hbm = nl.ndarray((INNER_Q_TILE_SIZE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE), # (128, 32, 8),
    #     dtype=nl.bool_, buffer=nl.shared_hbm)
    # nl.store(tile_mask_hbm, 0)
    tile_mask_hbm = None
    # iota_b_hbm = nl.ndarray((INNER_Q_TILE_SIZE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE), dtype=nl.int32, buffer=nl.shared_hbm)
    # nl.store(iota_b_hbm, 0)
    iota_b_hbm = None
    # tile_q_indices_hbm = nl.ndarray((seqlen_q, INNER_Q_TILE_SIZE, n_small_in_large_q_tile, loop_unroll_factor),
    #                                 dtype=nl.int32, buffer=nl.shared_hbm)
    # for i in nl.affine_range(seqlen_q // INNER_Q_TILE_SIZE):
    #     nl.store(tile_q_indices_hbm[nl.ds(i*INNER_Q_TILE_SIZE, INNER_Q_TILE_SIZE)], 0)
    tile_q_indices_hbm = None
    # tile_block_tables_hbm = nl.ndarray((seqlen_q, INNER_Q_TILE_SIZE, n_small_in_large_q_tile, loop_unroll_factor),
    #                                 dtype=nl.int32, buffer=nl.shared_hbm)
    # for i in nl.affine_range(seqlen_q // INNER_Q_TILE_SIZE):
    #     nl.store(tile_block_tables_hbm[nl.ds(i*INNER_Q_TILE_SIZE, INNER_Q_TILE_SIZE)], 0)
    # l_buffer_hbm = nl.ndarray((seqlen_q, h, 1), dtype=acc_type, buffer=nl.shared_hbm)
    # for i in nl.affine_range(seqlen_q // INNER_Q_TILE_SIZE):
    #     nl.store(l_buffer_hbm[nl.ds(i*INNER_Q_TILE_SIZE, INNER_Q_TILE_SIZE)], 0)
    l_buffer_hbm = None

    assert nl.program_ndim() == 2, (
        f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    )

    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    assert is_power_of_2(num_blocks_per_large_tile), (
        f"{num_blocks_per_large_tile=} is expected of be power of 2"
    )

    # max_num_queries = cu_seqlens_q.shape[-1]
    max_num_seqs = seqused_k.shape[-1]
    seqused_q = nl.ndarray((max_num_seqs,), dtype=nl.int32, buffer=nl.hbm)
    assert cu_seqlens_q.shape[-1] == (max_num_seqs + 1), (
        f"invalid {cu_seqlens_q.shape=}, expect {(max_num_seqs+1)=}"
    )
    print(f"flash_pa_with_schedule.py: {max_num_seqs=}")
    # prefill_seq_ids = nl.ndarray((max_num_tiles,), dtype=nl.int32, buffer=nl.hbm)
    # nl.store(prefill_seq_ids[nl.arange(max_num_tiles)[None, :]], value=0.0)

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

    # Compute length differences between cumulative sequence lengths and used sequences
    diff(cu_seqlens_q, seqused_q)

    # Prepare sequence IDs for tiled attention computation
    ACTIVE_Q_TILE_SIZE = min(seqlen_q, B_P_SIZE)
    # sequence_ids, q_offsets, k_offsets, dynamic_loop_trip_count, last_tile_indices_v2 = \
    #     prepare_sequence_offsets(
    #         seqused_q, seqused_k, num_seqs, max_num_tiles=MAX_NUM_TILES,
    #         q_tile_size=LARGE_Q_TILE_SIZE, large_k_tile_size=LARGE_KV_TILE_SIZE)
    identity_for_transpose_p_hbm = nl.shared_constant(
        np.identity(n=nl.tile_size.pmax, dtype=np.uint8), dtype=nl.uint8
    )
    (
        sequence_ids,
        q_offsets,
        k_offsets,
        dynamic_loop_trip_count,
        prefill_last_tile_indices,
        decode_seq_ids,
        decode_q_offsets,
        decode_k_offsets,
        decode_trip_count,
        decode_last_tile_indices,
    ) = prepare_sequence_offsets(
        seqused_q,
        seqused_k,
        num_seqs,
        identity_for_transpose_p_hbm,
        max_num_tiles=MAX_NUM_TILES,
        num_queries=seqlen_q,
        q_tile_size=LARGE_Q_TILE_SIZE,
        large_k_tile_size=LARGE_KV_TILE_SIZE,
    )  # , active_q_tile_size=max_num_seqs) # ACTIVE_Q_TILE_SIZE)
    # nl.device_print("flash_pa_with_schedule.py: sequence_ids=", sequence_ids.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: q_offsets=", q_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: k_offsets=", k_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: dynamic_loop_trip_count=", dynamic_loop_trip_count)
    # nl.device_print("flash_pa_with_schedule.py: last_tile_indices_v2=", last_tile_indices_v2)
    # nl.device_print("flash_pa_with_schedule.py: decode_seq_ids=", decode_seq_ids.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: decode_q_offsets=", decode_q_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: decode_k_offsets=", decode_k_offsets.reshape([1, MAX_NUM_TILES]))
    # nl.device_print("flash_pa_with_schedule.py: decode_trip_count=", decode_trip_count)

    # enable_prefill = False
    enable_decode = True

    if enable_prefill:
        print(f"running prefill_kernel")
        assert seqlen_q % LARGE_Q_TILE_SIZE == 0
        prefill_subkernel(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            sequence_ids=sequence_ids,
            q_offsets=q_offsets,
            k_offsets=k_offsets,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            active_mask=active_mask,
            q_update_pred=None,  # q_update_pred,
            last_tile_indices=prefill_last_tile_indices,  # last_tile_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_seqs=num_seqs,
            block_tables=block_tables,
            o=o,
            dynamic_loop_trip_count=dynamic_loop_trip_count,
            loop_unroll_factor=loop_unroll_factor,
            batch_id=batch_id,
            head_id=head_id,
            seqlen_q=seqlen_q,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            block_size_tiling_factor=block_size_tiling_factor,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            skip_active=skip_active,
            tile_mask_hbm=tile_mask_hbm,
            iota_b_hbm=iota_b_hbm,
            tile_q_indices_hbm=tile_q_indices_hbm,
            l_buffer_hbm=l_buffer_hbm,
            large_q_tile_size=large_q_tile_size,
            large_kv_tile_size=large_kv_tile_size,
        )
    else:
        print(f"prefill_kernel disabled")

    if enable_decode:
        print(f"running decode_kernel")
        decode_subkernel(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            sequence_ids=decode_seq_ids,
            q_offsets=decode_q_offsets,
            k_offsets=decode_k_offsets,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            q_update_pred=None,  # q_update_pred,
            last_tile_indices=decode_last_tile_indices,  # last_tile_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            num_seqs=num_seqs,
            block_tables=block_tables,
            o=o,
            dynamic_loop_trip_count=decode_trip_count,
            loop_unroll_factor=loop_unroll_factor,
            batch_id=batch_id,
            head_id=head_id,
            seqlen_q=max_num_seqs,  # seqlen_q,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            tiled_block_size=tiled_block_size,
            block_size_tiling_factor=block_size_tiling_factor,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            skip_active=skip_active,
            tile_mask_hbm=tile_mask_hbm,
        )

    return o
    # return o, seqused_q, sequence_ids, dynamic_loop_trip_count, q_offsets, k_offsets, prefill_last_tile_indices, \
    #     decode_seq_ids, decode_q_offsets, decode_k_offsets, decode_trip_count, decode_last_tile_indices
