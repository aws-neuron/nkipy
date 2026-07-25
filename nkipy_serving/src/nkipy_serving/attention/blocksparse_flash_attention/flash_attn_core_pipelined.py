import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim

from .constants import B_P_SIZE, NEG_INF
from .flash_attn_core import transpose_p_local
from .paged_cache import load_k_tile_from_cache, load_v_tile_from_cache
from .utils import PF_transpose_with_PE


def _flash_attention_core_prefill_prior_pipelined(
    *,
    loop_index,
    query,
    key_cache,
    value_cache,
    tile_q_indices_sbuf,
    block_tables_sbuf,
    olm_buffer_hbm,
    olm_buffer_sbuf,
    olm_reset_buf,
    tile_masks,
    q_update_pred_broadcast,
    num_blocks_per_large_tile,
    block_size,
    softmax_scale,
    seqlen_q,
    batch_id,
    kv_head_id,
    kernel_dtype,
    acc_type,
    q_load_buffer,
    k_load_buffer,
    v_load_buffer,
    mask_buffer,
    identity_p,
    B_F_SIZE=512,
    q_tile_sbuf=None,
):
    CACHE_Q_IN_SBUF = q_tile_sbuf is not None
    is_nc_gen2 = nisa.get_nc_version() == nisa.nc_version.gen2
    Q_TILE_SIZE, _, n_small_in_large_q_tile, q_h_per_k_h, _ = olm_buffer_sbuf.shape
    num_kv_tiles = tile_q_indices_sbuf.shape[-1]
    num_q_tiles = n_small_in_large_q_tile * q_h_per_k_h
    B_D_SIZE = olm_buffer_sbuf.shape[-1] - 2
    assert num_q_tiles >= 1 and num_kv_tiles >= 1
    LARGE_KV_TILE_SIZE = tile_masks.shape[-1]
    assert LARGE_KV_TILE_SIZE % B_P_SIZE == 0, (
        f"{LARGE_KV_TILE_SIZE=} not divisive by {B_P_SIZE=}"
    )
    assert LARGE_KV_TILE_SIZE % B_F_SIZE == 0, (
        f"{LARGE_KV_TILE_SIZE=} not divisive by {B_F_SIZE=}"
    )
    num_k_tile_per_large_tile = LARGE_KV_TILE_SIZE // B_F_SIZE

    NUM_COMPUTE_BUF = k_load_buffer.shape[1]
    NUM_WRITE_BUF = olm_buffer_sbuf.shape[1]
    assert v_load_buffer.shape[1] == mask_buffer.shape[1] == NUM_COMPUTE_BUF
    assert CACHE_Q_IN_SBUF or q_load_buffer.shape[1] == NUM_COMPUTE_BUF

    if not CACHE_Q_IN_SBUF:
        q_sbuf_tile_transposed = nl.ndarray(
            (
                par_dim(B_D_SIZE),
                NUM_COMPUTE_BUF,
                Q_TILE_SIZE,
            ),
            dtype=query.dtype,
        )
    olm_buffer_reshaped = olm_buffer_sbuf.reshape(
        (
            Q_TILE_SIZE,
            NUM_WRITE_BUF,
            num_q_tiles,
            B_D_SIZE + 2,
        )
    )
    qk_res_buf = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF, LARGE_KV_TILE_SIZE),
        dtype=acc_type,
    )
    max_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF, num_k_tile_per_large_tile),
        dtype=acc_type,
    )
    max_ = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF),
        dtype=acc_type,
    )
    m_current_neg = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF),
        dtype=acc_type,
    )
    o_previous_scaled = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF, B_D_SIZE),
        dtype=acc_type,
    )
    l_previous_scaled = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF),
        dtype=acc_type,
    )
    k_load_buffer_reshaped = k_load_buffer.reshape(
        (B_D_SIZE, NUM_COMPUTE_BUF, LARGE_KV_TILE_SIZE)
    )
    alpha = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF),
        dtype=acc_type,
    )
    REDUCTION_TILE = min(2048, LARGE_KV_TILE_SIZE // 2)
    p_local = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF, LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    p_partial_sum = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF, LARGE_KV_TILE_SIZE // REDUCTION_TILE),
        dtype=acc_type,
    )
    ps = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF),
        dtype=acc_type,
    )
    p_local_transposed = nl.ndarray(
        (
            par_dim(B_P_SIZE),
            NUM_COMPUTE_BUF,
            LARGE_KV_TILE_SIZE // B_P_SIZE * Q_TILE_SIZE,
        ),
        dtype=kernel_dtype,
    )
    pv_psum = nl.ndarray(
        (par_dim(Q_TILE_SIZE), NUM_COMPUTE_BUF, B_D_SIZE),
        dtype=np.float32,
        buffer=nl.psum,
    )
    tile_q_offsets = nl.ndarray((1, num_kv_tiles), dtype=nl.uint32)

    def qk_mask_max(q_idx, large_tile_idx):
        local_tile_idx = large_tile_idx * num_q_tiles + q_idx
        for k_i in nl.affine_range(num_k_tile_per_large_tile):
            k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)
            if CACHE_Q_IN_SBUF:
                q_tile_sbuf_reshaped = q_tile_sbuf.reshape(
                    (B_D_SIZE, q_h_per_k_h, n_small_in_large_q_tile, Q_TILE_SIZE)
                )
                q_small = q_idx // q_h_per_k_h
                i_q_h = q_idx % q_h_per_k_h
                q_tile = q_tile_sbuf_reshaped[:, i_q_h, q_small]
            else:
                q_tile = q_sbuf_tile_transposed[:, local_tile_idx % NUM_COMPUTE_BUF]
            k_tile = k_load_buffer_reshaped[
                :,
                large_tile_idx % NUM_COMPUTE_BUF,
                k_i_b_f_slice,
            ]
            qk_psum = nl.ndarray(
                (par_dim(Q_TILE_SIZE), B_F_SIZE), dtype=np.float32, buffer=nl.psum
            )  # (128, 512)
            qk_psum[:, :] = nisa.nc_matmul(q_tile, k_tile)  # (p(128), 512)
            if is_nc_gen2:
                # XXX: nisa.select_reduce produces wrong results on Trn1
                qk_res_buf[:, local_tile_idx % NUM_COMPUTE_BUF, k_i_b_f_slice] = (
                    nl.where(
                        mask_buffer[
                            :,
                            large_tile_idx % NUM_COMPUTE_BUF,
                            q_idx // q_h_per_k_h,
                            k_i_b_f_slice,
                        ],
                        qk_psum,
                        NEG_INF,
                        dtype=acc_type,
                    )
                )
                # Calculate max of the current tile
                max_local[:, local_tile_idx % NUM_COMPUTE_BUF, k_i] = (
                    nisa.tensor_reduce(
                        np.max,
                        qk_res_buf[:, local_tile_idx % NUM_COMPUTE_BUF, k_i_b_f_slice],
                        axis=(1,),
                        dtype=acc_type,
                        negate=False,
                    )
                )
            else:
                nisa.select_reduce(
                    dst=qk_res_buf[:, local_tile_idx % NUM_COMPUTE_BUF, k_i_b_f_slice],
                    predicate=mask_buffer[
                        :,
                        large_tile_idx % NUM_COMPUTE_BUF,
                        q_idx // q_h_per_k_h,
                        k_i_b_f_slice,
                    ],
                    on_true=qk_psum,
                    on_false=NEG_INF,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                    reduce_res=max_local[:, local_tile_idx % NUM_COMPUTE_BUF, k_i],
                    reduce_op=np.max,
                )
        # Calculate max of the current tile
        max_[:, local_tile_idx % NUM_COMPUTE_BUF] = nisa.tensor_reduce(
            np.max,
            max_local[:, local_tile_idx % NUM_COMPUTE_BUF, :],
            axis=(1,),
            dtype=acc_type,
            negate=False,
        )

    def calc_alpha_and_update_max(q_idx, large_tile_idx):
        local_tile_idx = large_tile_idx * num_q_tiles + q_idx
        m_previous = olm_buffer_reshaped[
            :, large_tile_idx % NUM_WRITE_BUF, q_idx, B_D_SIZE + 1
        ]
        m_current_neg[:, local_tile_idx % NUM_COMPUTE_BUF] = nisa.tensor_scalar(
            max_[:, local_tile_idx % NUM_COMPUTE_BUF],
            nl.maximum,
            m_previous,
            op1=nl.multiply,
            operand1=-1,
        )
        # Compute scaling factor
        alpha[:, local_tile_idx % NUM_COMPUTE_BUF] = nisa.activation(
            np.exp,
            m_previous,
            bias=m_current_neg[:, local_tile_idx % NUM_COMPUTE_BUF],
            scale=1.0,
        )
        # update max buffer
        olm_buffer_reshaped[
            :,
            large_tile_idx % NUM_WRITE_BUF,
            q_idx,
            B_D_SIZE + 1,
        ] = nisa.activation(
            nl.copy,
            m_current_neg[:, local_tile_idx % NUM_COMPUTE_BUF],
            scale=-1.0,
        )

    def exp_sum(q_idx, large_tile_idx):
        local_tile_idx = large_tile_idx * num_q_tiles + q_idx
        for k_r_i in nl.affine_range(LARGE_KV_TILE_SIZE // REDUCTION_TILE):
            k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)
            # Compute partial row - tile sum of exp(qk - max))
            p_local[
                :,
                local_tile_idx % NUM_COMPUTE_BUF,
                k_r_i_reduce_slice,
            ] = nisa.activation_reduce(
                np.exp,
                qk_res_buf[:, local_tile_idx % NUM_COMPUTE_BUF, k_r_i_reduce_slice],
                bias=m_current_neg[:, local_tile_idx % NUM_COMPUTE_BUF],
                scale=1.0,
                reduce_op=nl.add,
                reduce_res=p_partial_sum[:, local_tile_idx % NUM_COMPUTE_BUF, k_r_i],
                dtype=kernel_dtype,
            )
        ps[:, local_tile_idx % NUM_COMPUTE_BUF] = nl.sum(
            p_partial_sum[:, local_tile_idx % NUM_COMPUTE_BUF],
            axis=1,
            dtype=acc_type,
        )

    def trans_p_pv(q_idx, large_tile_idx):
        local_tile_idx = large_tile_idx * num_q_tiles + q_idx
        v = v_load_buffer[:, large_tile_idx % NUM_COMPUTE_BUF]
        transpose_p_local(
            p_local_transposed=p_local_transposed[:, local_tile_idx % NUM_COMPUTE_BUF],
            p_local=p_local[:, local_tile_idx % NUM_COMPUTE_BUF],
            Q_TILE_SIZE=Q_TILE_SIZE,
            LARGE_KV_TILE_SIZE=LARGE_KV_TILE_SIZE,
            B_F_SIZE=B_F_SIZE,
        )
        pv_psum[:, local_tile_idx % NUM_COMPUTE_BUF, :] = 0
        for k_i in nl.affine_range(LARGE_KV_TILE_SIZE // B_P_SIZE):
            pv_psum[:, local_tile_idx % NUM_COMPUTE_BUF, :] += nisa.nc_matmul(
                p_local_transposed[
                    :,
                    local_tile_idx % NUM_COMPUTE_BUF,
                    nl.ds(k_i * Q_TILE_SIZE, Q_TILE_SIZE),
                ],
                v[:, nl.ds(k_i * B_D_SIZE, B_D_SIZE)],
            )  # (128, 128) (p(Br), d)

    def update_ol(q_idx, large_tile_idx):
        local_tile_idx = large_tile_idx * num_q_tiles + q_idx
        olm_tile = olm_buffer_reshaped[:, large_tile_idx % NUM_WRITE_BUF, q_idx]
        o_previous_scaled[:, local_tile_idx % NUM_COMPUTE_BUF] = nl.multiply(
            olm_tile[:, nl.ds(0, B_D_SIZE)],
            alpha[:, local_tile_idx % NUM_COMPUTE_BUF],
        )
        olm_tile[:, nl.ds(0, B_D_SIZE)] = nl.add(
            o_previous_scaled[:, local_tile_idx % NUM_COMPUTE_BUF],
            pv_psum[:, local_tile_idx % NUM_COMPUTE_BUF],
        )
        l_previous_scaled[:, local_tile_idx % NUM_COMPUTE_BUF] = (
            olm_tile[:, B_D_SIZE] * alpha[:, local_tile_idx % NUM_COMPUTE_BUF]
        )
        olm_tile[:, B_D_SIZE] = (
            l_previous_scaled[:, local_tile_idx % NUM_COMPUTE_BUF]
            + ps[:, local_tile_idx % NUM_COMPUTE_BUF]
        )

    def first_half_tile():
        # handle top-left tile
        qk_mask_max(0, 0)
        calc_alpha_and_update_max(0, 0)
        exp_sum(0, 0)

    def fused_current_pv_next_qk(curr_q_idx, curr_kv_idx, next_q_idx, next_kv_idx):
        # compute pv for current tile, and qk for next tile
        assert (
            next_kv_idx == curr_kv_idx + 1
            or next_kv_idx == curr_kv_idx
            and next_q_idx == curr_q_idx + 1
        )
        qk_mask_max(next_q_idx, next_kv_idx)
        trans_p_pv(curr_q_idx, curr_kv_idx)
        calc_alpha_and_update_max(next_q_idx, next_kv_idx)
        exp_sum(next_q_idx, next_kv_idx)
        update_ol(curr_q_idx, curr_kv_idx)

    def last_half_tile():
        trans_p_pv(num_q_tiles - 1, num_kv_tiles - 1)
        update_ol(num_q_tiles - 1, num_kv_tiles - 1)

    if CACHE_Q_IN_SBUF:

        def slice_q_range(kv_idx):
            LARGE_Q_TILE_SIZE = q_tile_sbuf.shape[-1]
            i_d, i_h, i_s = nl.mgrid[:B_D_SIZE, :q_h_per_k_h, :LARGE_Q_TILE_SIZE]
            q_tile_sbuf[i_d, i_h, i_s] = nisa.tensor_copy_dynamic_src(
                q_load_buffer[i_d, i_h, i_s + tile_q_offsets[0, kv_idx]]
            )

    else:

        def load_q(q_idx, large_tile_idx):
            # load q
            local_tile_idx = large_tile_idx * num_q_tiles + q_idx
            small_q_idx = q_idx // q_h_per_k_h
            i_q_h = q_idx % q_h_per_k_h
            i_p = nl.arange(Q_TILE_SIZE)[:, None]
            i_f = nl.arange(B_D_SIZE)[None, :]
            q_load_buffer[i_p, local_tile_idx % NUM_COMPUTE_BUF, i_f] = nl.load(
                query[
                    batch_id,
                    kv_head_id * q_h_per_k_h + i_q_h,
                    tile_q_indices_sbuf[i_p, small_q_idx, large_tile_idx],
                    i_f,
                ],
                mode=oob_mode.skip,
            )
            q_t_psum = nl.ndarray(
                (B_D_SIZE, Q_TILE_SIZE),
                dtype=nl.float32 if is_nc_gen2 else query.dtype,
                buffer=nl.psum,
            )
            PF_transpose_with_PE(
                q_load_buffer[:, local_tile_idx % NUM_COMPUTE_BUF, :],
                q_t_psum,
                identity_for_transpose=identity_p,
                out_in_psum=True,
            )
            q_sbuf_tile_transposed[:, local_tile_idx % NUM_COMPUTE_BUF, :] = (
                nl.multiply(
                    q_t_psum,
                    softmax_scale,
                    dtype=kernel_dtype,
                )
            )

    def load_v(large_tile_idx):
        load_v_tile_from_cache(
            value_cache=value_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=large_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            v_load_buffer=v_load_buffer,
        )

    def load_k_and_mask(large_tile_idx):
        load_k_tile_from_cache(
            key_cache=key_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=large_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            B_D_SIZE=B_D_SIZE,
            k_load_buffer=k_load_buffer,
        )
        mask_buffer[:, large_tile_idx % NUM_COMPUTE_BUF, :, :] = nl.load(
            tile_masks[:, loop_index[0, 0], large_tile_idx, :, :]
        )

    is_q_out_of_range = nl.ndarray((1, num_kv_tiles), dtype=nl.uint8)
    is_q_out_of_range[...] = nisa.tensor_scalar(
        tile_q_indices_sbuf[nl.ds(0, 1), 0, :],
        nl.greater_equal,
        float(seqlen_q),
        dtype=nl.uint8,
    )
    tile_q_offsets[...] = nl.copy(
        tile_q_indices_sbuf[nl.ds(0, 1), 0, :], dtype=nl.uint32
    )
    nisa.tensor_copy_predicated(
        src=nl.zeros((1, 1), dtype=nl.uint32),
        dst=tile_q_offsets,
        predicate=is_q_out_of_range,
    )
    # load_k_and_mask(0)

    if CACHE_Q_IN_SBUF:
        slice_q_range(0)
    else:
        load_q(0, 0)
    first_half_tile()

    # column 0 to column n-2
    for kv_idx in nl.sequential_range(num_kv_tiles - 1):
        load_v(kv_idx)
        load_k_and_mask(kv_idx + 1)
        # perform compute
        for q_idx in nl.sequential_range(0, num_q_tiles - 1):
            if not CACHE_Q_IN_SBUF:
                load_q(q_idx + 1, kv_idx)
            fused_current_pv_next_qk(q_idx, kv_idx, q_idx + 1, kv_idx)
        # last tile
        if CACHE_Q_IN_SBUF:
            slice_q_range(kv_idx + 1)
        else:
            load_q(0, kv_idx + 1)
        nisa.tensor_copy_predicated(
            src=olm_buffer_reshaped[:, kv_idx % NUM_WRITE_BUF, :, B_D_SIZE + 1],
            dst=olm_buffer_reshaped[:, (kv_idx + 1) % NUM_WRITE_BUF, :, B_D_SIZE + 1],
            predicate=q_update_pred_broadcast[:, kv_idx],
        )
        fused_current_pv_next_qk(num_q_tiles - 1, kv_idx, 0, kv_idx + 1)
        nisa.tensor_copy_predicated(
            src=olm_buffer_reshaped[
                :, kv_idx % NUM_WRITE_BUF, :, nl.ds(0, B_D_SIZE + 1)
            ],
            dst=olm_buffer_reshaped[
                :, (kv_idx + 1) % NUM_WRITE_BUF, :, nl.ds(0, B_D_SIZE + 1)
            ],
            predicate=q_update_pred_broadcast[:, kv_idx],
        )
        # write out aggregation buffer
        for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
            i_p = nl.arange(Q_TILE_SIZE)[:, None, None]
            i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
            i_f_d = nl.arange(B_D_SIZE + 2)[None, None, :]
            nl.store(
                olm_buffer_hbm[
                    kv_head_id,
                    tile_q_indices_sbuf[i_p, small_q_idx, kv_idx],
                    i_f_h,
                    i_f_d,
                ],
                olm_buffer_sbuf[i_p, kv_idx % NUM_WRITE_BUF, small_q_idx, i_f_h, i_f_d],
                mode=oob_mode.skip,
            )
        # reset current tile
        olm_buffer_sbuf[:, kv_idx % NUM_WRITE_BUF] = olm_reset_buf

    load_v(num_kv_tiles - 1)
    for q_idx in nl.sequential_range(num_q_tiles - 1):
        if not CACHE_Q_IN_SBUF:
            load_q(q_idx + 1, num_kv_tiles - 1)
        fused_current_pv_next_qk(q_idx, num_kv_tiles - 1, q_idx + 1, num_kv_tiles - 1)

    last_half_tile()
    nisa.tensor_copy_predicated(
        src=olm_buffer_reshaped[:, (num_kv_tiles - 1) % NUM_WRITE_BUF],
        dst=olm_buffer_reshaped[:, num_kv_tiles % NUM_WRITE_BUF],
        predicate=q_update_pred_broadcast[:, num_kv_tiles - 1],
    )
    # write out aggregation buffer
    for small_q_idx in nl.affine_range(n_small_in_large_q_tile):
        i_p = nl.arange(Q_TILE_SIZE)[:, None, None]
        i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_f_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        nl.store(
            olm_buffer_hbm[
                kv_head_id,
                tile_q_indices_sbuf[i_p, small_q_idx, num_kv_tiles - 1],
                i_f_h,
                i_f_d,
            ],
            olm_buffer_sbuf[
                i_p, (num_kv_tiles - 1) % NUM_WRITE_BUF, small_q_idx, i_f_h, i_f_d
            ],
            mode=oob_mode.skip,
        )
    if num_kv_tiles % NUM_WRITE_BUF != 0:
        olm_buffer_sbuf[:, 0] = olm_buffer_sbuf[:, num_kv_tiles % NUM_WRITE_BUF]
        olm_buffer_sbuf[:, num_kv_tiles % NUM_WRITE_BUF] = olm_reset_buf
    if num_kv_tiles > 1:
        olm_buffer_sbuf[:, (num_kv_tiles - 1) % NUM_WRITE_BUF] = olm_reset_buf
