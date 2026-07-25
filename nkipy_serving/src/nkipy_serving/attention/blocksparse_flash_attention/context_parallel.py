import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl
import numpy as np
from neuronxcc.nki.language import par_dim


def softmax_correction_allgather(
    olm_buffer,
    olm_buffer_ag,
    ACTIVE_Q_TILE_SIZE,
    acc_type,
    cp_replica_group,
    kv_head_id,
):
    num_kv_heads, seqlen_q, q_h_per_k_h, total_feat_size = olm_buffer.shape
    B_D_SIZE = total_feat_size - 2
    cp_group_size = len(cp_replica_group[0])
    nccl.all_gather(
        op=np.add,
        srcs=[olm_buffer],
        dsts=[olm_buffer_ag],
        all_gather_dim=0,
        replica_groups=cp_replica_group,
    )
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE

    # XXX: token processing order does not matter
    olm_buffer_ag_reshaped = olm_buffer_ag.reshape(
        (
            cp_group_size * num_kv_heads,
            ACTIVE_Q_TILE_SIZE,
            num_active_tiles,
            q_h_per_k_h,
            total_feat_size,
        )
    )
    olm_ag_sbuf = nl.ndarray(
        (
            par_dim(ACTIVE_Q_TILE_SIZE),
            cp_group_size,
            num_active_tiles,
            q_h_per_k_h,
            total_feat_size,
        ),
        dtype=acc_type,
    )
    for cp_rank in nl.affine_range(cp_group_size):
        nisa.dma_copy(
            dst=olm_ag_sbuf[:, cp_rank],
            src=olm_buffer_ag_reshaped[cp_rank * num_kv_heads + kv_head_id,],
            # dge_mode=nisa.dge_mode.swdge,
        )
    olm_buffer_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, B_D_SIZE + 2),
        dtype=acc_type,
    )
    for i in nl.affine_range(num_active_tiles):
        global_max = nisa.tensor_reduce(
            nl.max,
            olm_ag_sbuf[:, :, i, :, nl.ds(B_D_SIZE + 1, 1)],
            axis=(1,),
            keepdims=True,
        )
        assert global_max.shape == (ACTIVE_Q_TILE_SIZE, 1, q_h_per_k_h, 1)
        bias = nl.zeros((par_dim(ACTIVE_Q_TILE_SIZE), 1), dtype=acc_type)
        alpha = nl.ndarray(
            (par_dim(ACTIVE_Q_TILE_SIZE), cp_group_size, q_h_per_k_h, 1),
            dtype=acc_type,
        )
        alpha[...] = nisa.activation(
            np.exp,
            olm_ag_sbuf[:, :, i, :, nl.ds(B_D_SIZE + 1, 1)] - global_max,
            scale=1.0,
            bias=bias,
        )
        olm_buffer_sbuf[:, i, :, nl.ds(0, B_D_SIZE + 1)] = nisa.tensor_reduce(
            nl.add,
            olm_ag_sbuf[:, :, i, :, nl.ds(0, B_D_SIZE + 1)] * alpha,
            axis=(1,),
        )
        olm_buffer_sbuf[:, i, :, nl.ds(B_D_SIZE + 1, 1)] = global_max[:, 0]
    olm_buffer_reshaped = olm_buffer.reshape(
        (
            num_kv_heads,
            ACTIVE_Q_TILE_SIZE,
            num_active_tiles,
            q_h_per_k_h,
            total_feat_size,
        )
    )
    nl.store(
        olm_buffer_reshaped[kv_head_id],
        olm_buffer_sbuf,
    )


def softmax_correction_allreduce(
    olm_buffer,
    max_allreduce_buf,
    ol_allreduce_buf,
    ACTIVE_Q_TILE_SIZE,
    acc_type,
    cp_replica_group,
    kv_head_id,
):
    num_kv_heads, seqlen_q, q_h_per_k_h, total_feat_size = olm_buffer.shape
    B_D_SIZE = total_feat_size - 2
    assert seqlen_q % ACTIVE_Q_TILE_SIZE == 0
    nisa.dma_copy(
        dst=max_allreduce_buf[kv_head_id],
        src=olm_buffer[kv_head_id, :, :, nl.ds(B_D_SIZE + 1, 1)],
        # dge_mode=nisa.dge_mode.swdge,
    )
    nccl.all_reduce(
        op=np.maximum,
        srcs=[max_allreduce_buf],
        dsts=[max_allreduce_buf],
        replica_groups=cp_replica_group,
    )
    num_active_tiles = seqlen_q // ACTIVE_Q_TILE_SIZE
    # XXX: token processing order does not matter
    olm_buffer_reshaped = olm_buffer.reshape(
        (
            num_kv_heads,
            ACTIVE_Q_TILE_SIZE,
            num_active_tiles,
            q_h_per_k_h,
            total_feat_size,
        )
    )
    olm_sbuf = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, total_feat_size),
        dtype=olm_buffer.dtype,
    )
    nisa.dma_copy(
        dst=olm_sbuf,
        src=olm_buffer_reshaped[kv_head_id],
        # dge_mode=nisa.dge_mode.swdge,
    )
    # copy max
    max_allreduce_buf_reshaped = max_allreduce_buf.reshape(
        (
            num_kv_heads,
            ACTIVE_Q_TILE_SIZE,
            num_active_tiles,
            q_h_per_k_h,
            1,
        )
    )
    global_max_sbuf = nl.ndarray(
        (ACTIVE_Q_TILE_SIZE, num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    nisa.dma_copy(
        dst=global_max_sbuf,
        src=max_allreduce_buf_reshaped[kv_head_id],
    )
    bias = nl.zeros((par_dim(ACTIVE_Q_TILE_SIZE), 1, 1, 1), dtype=acc_type)
    alpha = nl.ndarray(
        (par_dim(ACTIVE_Q_TILE_SIZE), num_active_tiles, q_h_per_k_h, 1),
        dtype=acc_type,
    )
    alpha[...] = nisa.activation(
        np.exp,
        olm_sbuf[:, :, :, nl.ds(B_D_SIZE + 1, 1)] - global_max_sbuf,
        scale=1.0,
        bias=bias,
    )
    ol_rescaled = olm_sbuf[:, :, :, nl.ds(0, B_D_SIZE + 1)] * alpha
    ol_allreduce_buf_reshaped = ol_allreduce_buf.reshape(
        (
            num_kv_heads,
            ACTIVE_Q_TILE_SIZE,
            num_active_tiles,
            q_h_per_k_h,
            B_D_SIZE + 1,
        )
    )
    nisa.dma_copy(dst=ol_allreduce_buf_reshaped[kv_head_id], src=ol_rescaled)
    nccl.all_reduce(
        op=np.add,
        srcs=[ol_allreduce_buf],
        dsts=[ol_allreduce_buf],
        replica_groups=cp_replica_group,
    )
