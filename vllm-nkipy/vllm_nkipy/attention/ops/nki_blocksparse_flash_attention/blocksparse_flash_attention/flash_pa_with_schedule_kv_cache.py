# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.isa.constants import oob_mode
from neuronxcc.nki.language import par_dim
from neuronxcc.nki.typing import scalar

from torch_to_nkipy.utils.nki import NKIOpRegistry
from vllm_nkipy.attention.ops.nki_blocksparse_flash_attention.blocksparse_flash_attention.flash_pa_with_schedule import (  # noqa: E501
    allocate_decode_accum_buffers,
    allocate_prefill_accum_buffers,
    decode_active_and_epilogue,
    decode_prior,
    load_decode_query,
    prefill_active_and_epilogue,
    prefill_prior,
    prepare_q_indices_range,
    prepare_q_update_pred,
    transpose_broadcast_q_pred,
)

from .constants import B_FMAX_SIZE, B_P_SIZE, NEG_INF
from .paged_cache import (
    load_k_tile_from_cache,
    transform_block_tables_for_indirect_load,
)
from .paged_cache_kv_cache import prepare_kv_block_dim_tiling_kv_cache
from .utils import (
    IdentityStore,
    PF_transpose_with_PE,
    is_power_of_2,
    load_indices,
)


def check_input_shapes_kv_cache(
    *,
    query,
    key,
    value,
    kv_cache,
    tile_masks,
    active_mask,
    decode_mode,
    skip_active,
):
    if decode_mode:
        INNER_KV_TILE_SIZE, _, _ = tile_masks.shape
        LARGE_Q_TILE_SIZE = 1
        assert INNER_KV_TILE_SIZE == B_P_SIZE
    else:
        INNER_Q_TILE_SIZE, _, n_small_in_large_q_tile, _ = tile_masks.shape
        LARGE_Q_TILE_SIZE = INNER_Q_TILE_SIZE * n_small_in_large_q_tile
    b, h, seqlen_q, d = query.shape
    assert seqlen_q <= 8192, f"Large {seqlen_q=} consumes too much sbuf space"
    if seqlen_q <= B_P_SIZE:
        assert is_power_of_2(seqlen_q), f"{seqlen_q=} is expected to be power of 2"
    elif seqlen_q <= B_FMAX_SIZE:
        assert seqlen_q % B_P_SIZE == 0, f"{seqlen_q=} must be mulitple of {B_P_SIZE=}"
    else:
        assert seqlen_q % B_FMAX_SIZE == 0, (
            f"{seqlen_q=} must be multiple of {B_FMAX_SIZE=}"
        )
    assert seqlen_q % LARGE_Q_TILE_SIZE == 0, (
        f"{seqlen_q=} must be multiple of {LARGE_Q_TILE_SIZE=}"
    )
    assert b == 1, f"Batch size must be 1 for Ragged Tensor, got {b}"
    assert d >= 16 and d <= 128 and is_power_of_2(d), (
        f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    )
    _, num_blocks, k_h, block_size, _ = kv_cache.shape
    assert tuple(kv_cache.shape) == (
        2,
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{kv_cache.shape=} mismatch!"
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
    assert tile_masks.dtype == nl.uint8, f"{tile_masks.dtype=} is expected to be uint8"
    if not skip_active and not decode_mode:
        assert active_mask.dtype == nl.uint8, (
            f"{active_mask.dtype=} is expected to be uint8"
        )

    return (
        b,
        h,
        k_h,
        seqlen_q,
        d,
    )


def decode_context_tokens_kv_cache(
    query,
    kv_cache,
    tile_q_indices,
    tile_masks,
    tile_block_tables,
    num_dynamic_loop_steps,
    olm_buffer,
    q_update_pred,
    kernel_dtype,
    acc_type,
    loop_unroll_factor,
    batch_id,
    head_id,
    k_h,
    q_h_per_k_h,
    softmax_scale,
    B_D_SIZE,
):
    INNER_KV_TILE_SIZE, _, N_INNER_KV_TILE = tile_masks.shape
    LARGE_KV_TILE_SIZE = INNER_KV_TILE_SIZE * N_INNER_KV_TILE
    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling_kv_cache(kv_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    # prepare block tables
    tile_block_tables_sbuf = load_indices(tile_block_tables)
    tile_block_tables_transformed_sbuf = transform_block_tables_for_indirect_load(
        tile_block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
        identity_for_transpose=None,
    )
    MAX_NUM_TILE = tile_block_tables.shape[0]
    num_loads = tile_block_tables_transformed_sbuf.shape[1]
    MAX_NUM_LOOP = MAX_NUM_TILE // loop_unroll_factor
    tile_block_tables_transformed_hbm = nl.ndarray(
        (MAX_NUM_LOOP, B_P_SIZE, num_loads, loop_unroll_factor),
        dtype=tile_block_tables.dtype,
        buffer=nl.hbm,
    )
    print(f"{MAX_NUM_LOOP=}, {type(MAX_NUM_LOOP)=}")
    for i in nl.affine_range(MAX_NUM_LOOP):
        nl.store(
            tile_block_tables_transformed_hbm[i],
            tile_block_tables_transformed_sbuf[
                :, :, nl.ds(i * loop_unroll_factor, loop_unroll_factor)
            ],
        )

    olm_next_tile_sbuf = nl.zeros(
        (par_dim(B_D_SIZE), 3, q_h_per_k_h),
        dtype=acc_type,
    )
    olm_next_tile_sbuf[:, nl.ds(2, 1), :] = NEG_INF

    identity_store = IdentityStore(
        (acc_type, B_P_SIZE),  # transpose max cascaded step 1
        (acc_type, q_h_per_k_h),  # transpose max casesded step 2
        (kernel_dtype, B_P_SIZE),  # transpose k
        (kernel_dtype, loop_unroll_factor),  # transpose q
        (acc_type, B_D_SIZE),  # transpose o
    )

    tile_q_indices = tile_q_indices.reshape(
        (MAX_NUM_TILE // loop_unroll_factor, loop_unroll_factor, 1)
    )
    q_indices = nl.ndarray((loop_unroll_factor, 1), dtype=tile_q_indices.dtype)
    q_indices[...] = nl.load(tile_q_indices[0])
    block_tables_sbuf = nl.ndarray(
        tile_block_tables_transformed_hbm.shape[1:],
        dtype=tile_block_tables_transformed_hbm.dtype,
    )
    block_tables_sbuf[...] = nl.load(tile_block_tables_transformed_hbm[0])
    q_update_pred_sbuf = nl.ndarray((loop_unroll_factor, 1), dtype=nl.uint8)
    q_update_pred_sbuf[...] = nl.load(q_update_pred[0])
    num_k_tiles = tile_masks.shape[2]
    assert num_k_tiles == num_loads * block_size
    assert tile_masks.shape[0] == B_P_SIZE
    tile_masks = tile_masks.reshape(
        (
            B_P_SIZE,
            MAX_NUM_TILE // loop_unroll_factor,
            loop_unroll_factor,
            num_k_tiles,
        )
    )
    tile_mask_sbuf = nl.ndarray(
        (par_dim(B_P_SIZE), loop_unroll_factor, num_k_tiles),
        dtype=tile_masks.dtype,
    )
    tile_mask_sbuf[...] = nl.load(tile_masks[:, 0, :, :])

    num_dynamic_loop_steps_sbuf = nl.load(num_dynamic_loop_steps)
    assert num_dynamic_loop_steps_sbuf.shape == (1, 1)
    olm_buffer_reshaped = olm_buffer.reshape(
        (MAX_NUM_TILE // loop_unroll_factor, loop_unroll_factor) + olm_buffer.shape[1:]
    )

    (identity_o, identity_q) = identity_store.get(
        (acc_type, B_D_SIZE),
        (kernel_dtype, loop_unroll_factor),
    )
    query_sbuf = nl.ndarray(
        (B_D_SIZE, q_indices.shape[0], q_h_per_k_h),
        dtype=kernel_dtype,
    )
    load_decode_query(
        query_sbuf=query_sbuf,
        q_indices=q_indices,
        query=query,
        softmax_scale=softmax_scale,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        kernel_dtype=kernel_dtype,
        identity_for_transpose=identity_q,
        is_first_tile_prefetch=True,
    )
    MAX_NUM_BUFFER_ALLOWED = 2
    MULTI_BUFFER = min(MAX_NUM_BUFFER_ALLOWED, loop_unroll_factor)
    assert loop_unroll_factor % MULTI_BUFFER == 0

    # load first iteration of key cache
    k_load_buffer = nl.ndarray(
        (B_P_SIZE, MULTI_BUFFER, num_loads, block_size * B_D_SIZE),
        dtype=key_cache.dtype,
    )
    for load_idx in nl.affine_range(num_loads):
        i_p = nl.arange(B_P_SIZE)[:, None]
        i_f = nl.arange(block_size * B_D_SIZE)[None, :]
        k_load_buffer[i_p, 0, load_idx, i_f] = nl.load(
            key_cache[block_tables_sbuf[i_p, load_idx, 0], i_f]
        )

    identity_for_transpose_lm = nl.ones((1, 1), dtype=acc_type)
    # load all q and multiply scale

    loop_index = nl.zeros((1, 1), dtype=np.int32)
    for _ in range(scalar(num_dynamic_loop_steps_sbuf)):
        q_update_pred_broadcast = transpose_broadcast_q_pred(
            q_update_pred_sbuf, B_D_SIZE
        )
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

        current_index = nl.copy(loop_index)

        # update loop_index
        loop_index[...] = nl.add(loop_index[...], 1)

        decode_prior(
            num_tiles_unrolled=loop_unroll_factor,
            query_sbuf=query_sbuf,
            key_cache=key_cache,
            value_cache=value_cache,
            olm_next_tile_sbuf=olm_next_tile_sbuf,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            k_load_buffer=k_load_buffer,
            block_tables_sbuf=block_tables_sbuf,
            tile_mask_sbuf=tile_mask_sbuf,
            q_update_pred_broadcast=q_update_pred_broadcast,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            identity_store=identity_store,
        )

        q_indices[...] = nl.load(tile_q_indices[loop_index[0, 0]], mode=oob_mode.skip)
        q_update_pred_sbuf[...] = nl.load(
            q_update_pred[loop_index[0, 0]], mode=oob_mode.skip
        )
        block_tables_sbuf[...] = nl.load(
            tile_block_tables_transformed_hbm[loop_index[0, 0]], mode=oob_mode.skip
        )
        i_p = nl.arange(B_P_SIZE)[:, None, None]
        i_q = nl.arange(loop_unroll_factor)[None, :, None]
        i_k = nl.arange(num_k_tiles)[None, None, :]
        tile_mask_sbuf[i_p, i_q, i_k] = nl.load(
            tile_masks[i_p, loop_index[0, 0], i_q, i_k], mode=oob_mode.skip
        )

        # store L, M, O in (seqlen, h, d) format
        olm_sbuf = nl.ndarray(
            (loop_unroll_factor, q_h_per_k_h, B_D_SIZE + 2),
            dtype=o_buffer_sbuf.dtype,
        )
        for i_q_h in nl.affine_range(q_h_per_k_h):
            o_tmp = nl.ndarray(
                (loop_unroll_factor, B_D_SIZE), dtype=acc_type, buffer=nl.psum
            )
            PF_transpose_with_PE(
                src=o_buffer_sbuf[:, nl.ds(0, loop_unroll_factor), i_q_h],
                out=o_tmp,
                identity_for_transpose=identity_o,
                out_in_psum=True,
            )
            olm_sbuf[:, i_q_h, nl.ds(0, B_D_SIZE)] = nl.copy(o_tmp)
            l_tmp = nl.ndarray((loop_unroll_factor, 1), dtype=acc_type, buffer=nl.psum)
            PF_transpose_with_PE(
                src=l_buffer_sbuf[:, nl.ds(0, loop_unroll_factor), i_q_h],
                out=l_tmp,
                identity_for_transpose=identity_for_transpose_lm,
                out_in_psum=True,
            )
            olm_sbuf[:, i_q_h, nl.ds(B_D_SIZE, 1)] = nl.copy(l_tmp)
            m_tmp = nl.ndarray((loop_unroll_factor, 1), dtype=acc_type, buffer=nl.psum)
            PF_transpose_with_PE(
                src=m_buffer_sbuf[:, nl.ds(0, loop_unroll_factor), i_q_h],
                out=m_tmp,
                identity_for_transpose=identity_for_transpose_lm,
                out_in_psum=True,
            )
            olm_sbuf[:, i_q_h, nl.ds(B_D_SIZE + 1, 1)] = nl.copy(m_tmp)

        nl.store(olm_buffer_reshaped[current_index[0, 0]], olm_sbuf)
        load_decode_query(
            query_sbuf=query_sbuf,
            q_indices=q_indices,
            query=query,
            softmax_scale=softmax_scale,
            batch_id=batch_id,
            head_id=head_id,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            identity_for_transpose=identity_q,
        )
        for load_idx in nl.affine_range(num_loads):
            i_p = nl.arange(B_P_SIZE)[:, None]
            i_f = nl.arange(block_size * B_D_SIZE)[None, :]
            k_load_buffer[i_p, 0, load_idx, i_f] = nl.load(
                key_cache[block_tables_sbuf[i_p, load_idx, 0], i_f]
            )


def decode_subkernel_kv_cache(
    query,
    key,
    value,
    kv_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    q_update_pred,
    last_tile_indices,
    o,
    sinks,
    num_dynamic_loop_steps,
    dynamic_loop_unroll_factor,
    batch_id,
    head_id,
    seqlen_q,
    kernel_dtype,
    acc_type,
    k_h,
    q_h_per_k_h,
    B_D_SIZE,
    softmax_scale,
    skip_active,
):
    MAX_NUM_TILE = tile_masks.shape[1]
    assert MAX_NUM_TILE > 0 and MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
    (olm_buffer,) = allocate_decode_accum_buffers(
        MAX_NUM_TILE=MAX_NUM_TILE,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        acc_type=acc_type,
    )
    last_tile_indices_sbuf = nl.load(last_tile_indices)
    if q_update_pred is None:
        q_update_pred_hbm = prepare_q_update_pred(last_tile_indices_sbuf, MAX_NUM_TILE)
    else:
        q_update_pred_hbm = q_update_pred
    q_update_pred = q_update_pred_hbm.reshape(
        (MAX_NUM_TILE // dynamic_loop_unroll_factor, dynamic_loop_unroll_factor, 1)
    )

    decode_context_tokens_kv_cache(
        query=query,
        kv_cache=kv_cache,
        tile_q_indices=tile_q_indices,
        tile_masks=tile_masks,
        tile_block_tables=tile_block_tables,
        num_dynamic_loop_steps=num_dynamic_loop_steps,
        olm_buffer=olm_buffer,
        q_update_pred=q_update_pred,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        loop_unroll_factor=dynamic_loop_unroll_factor,
        batch_id=batch_id,
        head_id=head_id,
        k_h=k_h,
        q_h_per_k_h=q_h_per_k_h,
        softmax_scale=softmax_scale,
        B_D_SIZE=B_D_SIZE,
    )
    decode_active_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        olm_buffer=olm_buffer,
        sinks=sinks,
        softmax_scale=softmax_scale,
        last_tile_indices_sbuf=last_tile_indices_sbuf,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_D_SIZE=B_D_SIZE,
        skip_active=skip_active,
    )


def prefill_context_tokens_kv_cache(
    query,
    kv_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    num_dynamic_loop_steps,
    olm_buffer,
    q_update_pred,
    kernel_dtype,
    acc_type,
    loop_unroll_factor,
    batch_id,
    head_id,
    k_h,
    q_h_per_k_h,
    softmax_scale,
    B_F_SIZE,
    B_D_SIZE,
):
    INNER_Q_TILE_SIZE, MAX_NUM_TILE, n_small_in_large_q_tile, LARGE_KV_TILE_SIZE = (
        tile_masks.shape
    )
    key_cache, value_cache, block_size_tiling_factor, block_size = (
        prepare_kv_block_dim_tiling_kv_cache(kv_cache, LARGE_KV_TILE_SIZE)
    )
    num_blocks_per_large_tile = LARGE_KV_TILE_SIZE // block_size
    olm_next_tile_sbuf = nl.zeros(
        (
            par_dim(INNER_Q_TILE_SIZE),
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE + 2,
        ),
        dtype=acc_type,
    )
    olm_next_tile_sbuf[:, :, :, B_D_SIZE + 1] = NEG_INF
    identity_store = IdentityStore(
        (kernel_dtype, B_P_SIZE),  # transpose k
        (kernel_dtype, INNER_Q_TILE_SIZE),  # transpose p
    )
    # prepare block tables
    tile_block_tables_sbuf = load_indices(tile_block_tables)
    tile_block_tables_transformed_sbuf = transform_block_tables_for_indirect_load(
        tile_block_tables_sbuf,
        block_size_tiling_factor=block_size_tiling_factor,
        num_head=k_h,
        head_id=head_id,
        identity_for_transpose=None,
    )
    num_loads = tile_block_tables_transformed_sbuf.shape[1]
    MAX_NUM_LOOP = MAX_NUM_TILE // loop_unroll_factor
    tile_block_tables_transformed_hbm = nl.ndarray(
        (MAX_NUM_LOOP, B_P_SIZE, num_loads, loop_unroll_factor),
        dtype=tile_block_tables.dtype,
        buffer=nl.hbm,
    )
    for i in nl.affine_range(MAX_NUM_LOOP):
        nl.store(
            tile_block_tables_transformed_hbm[i],
            tile_block_tables_transformed_sbuf[
                :, :, nl.ds(i * loop_unroll_factor, loop_unroll_factor)
            ],
        )

    block_tables_sbuf = nl.ndarray(
        tile_block_tables_transformed_hbm.shape[1:],
        dtype=tile_block_tables_transformed_hbm.dtype,
    )
    block_tables_sbuf[...] = nl.load(tile_block_tables_transformed_hbm[0])

    num_dynamic_loop_steps_sbuf = nl.load(num_dynamic_loop_steps)
    assert num_dynamic_loop_steps_sbuf.shape == (1, 1)
    if loop_unroll_factor > 1:
        num_unrolled_iota = nisa.iota(
            nl.arange(loop_unroll_factor)[None, :], dtype=nl.uint32
        )
    else:
        num_unrolled_iota = None

    MAX_NUM_BUFFER_ALLOWED = 2
    MULTI_BUFFER = min(MAX_NUM_BUFFER_ALLOWED, loop_unroll_factor)
    assert loop_unroll_factor % MULTI_BUFFER == 0
    # XXX: Work around a DMA skipping correctness issue:
    #      If nl.ndarray is used to allocate buffer for DMA skipping,
    #      kernel does not produce correct results.
    q_load_buffer = nl.zeros(
        (
            par_dim(INNER_Q_TILE_SIZE),
            MULTI_BUFFER,
            n_small_in_large_q_tile,
            q_h_per_k_h,
            B_D_SIZE,
        ),
        dtype=kernel_dtype,
        buffer=nl.sbuf,
    )
    num_loads = num_blocks_per_large_tile // B_P_SIZE
    k_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER, num_loads, block_size * B_D_SIZE),
        dtype=key_cache.dtype,
    )
    v_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER, num_loads * block_size * B_D_SIZE),
        dtype=value_cache.dtype,
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
    tile_masks = tile_masks.reshape(
        (
            INNER_Q_TILE_SIZE,
            MAX_NUM_TILE // loop_unroll_factor,
            loop_unroll_factor,
            n_small_in_large_q_tile,
            LARGE_KV_TILE_SIZE,
        )
    )
    mask_buffer = nl.ndarray(
        (
            par_dim(INNER_Q_TILE_SIZE),
            MULTI_BUFFER,
            n_small_in_large_q_tile,
            LARGE_KV_TILE_SIZE,
        ),
        dtype=tile_masks.dtype,
    )
    tile_q_indices_sbuf = nl.ndarray(
        (
            par_dim(INNER_Q_TILE_SIZE),
            n_small_in_large_q_tile,
            loop_unroll_factor,
        ),
        dtype=tile_q_indices.dtype,
    )
    loop_index = nl.zeros((1, 1), dtype=np.int32)
    prepare_q_indices_range(
        tile_q_indices,
        loop_index,
        loop_unroll_factor,
        INNER_Q_TILE_SIZE,
        tile_q_indices_sbuf,
        identity_for_transpose=None,
        num_tiles_iota=num_unrolled_iota,
    )
    mask_buffer[:, 0, :, :] = nl.load(tile_masks[:, 0, 0, :, :])
    q_update_pred_sbuf = nl.ndarray((loop_unroll_factor, 1), dtype=nl.uint8)
    q_update_pred_sbuf[...] = nl.load(q_update_pred[0])
    load_k_tile_from_cache(
        key_cache=key_cache,
        block_tables=block_tables_sbuf,
        large_k_tile_idx=0,
        num_blocks_per_large_tile=num_blocks_per_large_tile,
        block_size=block_size,
        B_D_SIZE=B_D_SIZE,
        k_load_buffer=k_load_buffer,
    )
    for _ in range(scalar(num_dynamic_loop_steps_sbuf)):
        olm_unrolled_sbuf[:, nl.ds(1, loop_unroll_factor)] = 0
        olm_unrolled_sbuf[:, :, :, :, B_D_SIZE + 1] = NEG_INF
        olm_unrolled_sbuf[:, 0] = nl.copy(olm_next_tile_sbuf)
        q_update_pred_broadcast = transpose_broadcast_q_pred(
            q_update_pred_sbuf, INNER_Q_TILE_SIZE
        )
        prefill_prior(
            loop_index=loop_index,
            num_tiles_unrolled=loop_unroll_factor,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            q_load_buffer=q_load_buffer,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            mask_buffer=mask_buffer,
            olm_buffer_hbm=olm_buffer,
            olm_unrolled_sbuf=olm_unrolled_sbuf,
            tile_q_indices_sbuf=tile_q_indices_sbuf,
            block_tables_sbuf=block_tables_sbuf,
            tile_masks=tile_masks,
            q_update_pred_broadcast=q_update_pred_broadcast,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            identity_store=identity_store,
            batch_id=batch_id,
            head_id=head_id,
            q_h_per_k_h=q_h_per_k_h,
            softmax_scale=softmax_scale,
            n_small_in_large_q_tile=n_small_in_large_q_tile,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            B_F_SIZE=B_F_SIZE,
            B_D_SIZE=B_D_SIZE,
        )
        # update loop_index
        loop_index[...] = nl.add(loop_index[...], 1)
        block_tables_sbuf[...] = nl.load(
            tile_block_tables_transformed_hbm[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        prepare_q_indices_range(
            tile_q_indices,
            loop_index,
            loop_unroll_factor,
            INNER_Q_TILE_SIZE,
            tile_q_indices_sbuf,
            identity_for_transpose=None,
            num_tiles_iota=num_unrolled_iota,
        )
        olm_next_tile_sbuf[...] = nl.copy(olm_unrolled_sbuf[:, loop_unroll_factor])

        mask_buffer[:, 0, :, :] = nl.load(
            tile_masks[:, loop_index[0, 0], 0, :, :],
            mode=oob_mode.skip,
        )
        q_update_pred_sbuf[...] = nl.load(
            q_update_pred[loop_index[0, 0]],
            mode=oob_mode.skip,
        )
        load_k_tile_from_cache(
            key_cache=key_cache,
            block_tables=block_tables_sbuf,
            large_k_tile_idx=0,
            num_blocks_per_large_tile=num_blocks_per_large_tile,
            block_size=block_size,
            B_D_SIZE=B_D_SIZE,
            k_load_buffer=k_load_buffer,
        )


def prefill_subkernel_kv_cache(
    query,
    key,
    value,
    kv_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    o,
    q_update_pred,
    last_tile_indices,
    sinks,
    num_dynamic_loop_steps,
    dynamic_loop_unroll_factor,
    batch_id,
    head_id,
    seqlen_q,
    kernel_dtype,
    acc_type,
    k_h,
    q_h_per_k_h,
    B_D_SIZE,
    softmax_scale,
    skip_active,
):
    assert num_dynamic_loop_steps.dtype == nl.int32
    B_F_SIZE = B_FMAX_SIZE
    INNER_Q_TILE_SIZE, MAX_NUM_TILE, _, LARGE_KV_TILE_SIZE = tile_masks.shape
    assert LARGE_KV_TILE_SIZE % B_F_SIZE == 0, (
        f"Need LARGE_KV_TILE_SIZE ({LARGE_KV_TILE_SIZE=})"
        f" to be divisible by ({B_F_SIZE=})"
    )
    assert MAX_NUM_TILE > 0 and MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
    (olm_buffer,) = allocate_prefill_accum_buffers(
        seqlen_q=seqlen_q,
        INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
        acc_type=acc_type,
    )
    if q_update_pred is None:
        last_tile_indices_sbuf = nl.load(last_tile_indices)
        q_update_pred_hbm = prepare_q_update_pred(last_tile_indices_sbuf, MAX_NUM_TILE)
    else:
        q_update_pred_hbm = q_update_pred
    q_update_pred = q_update_pred_hbm.reshape(
        (MAX_NUM_TILE // dynamic_loop_unroll_factor, dynamic_loop_unroll_factor, 1)
    )
    prefill_context_tokens_kv_cache(
        query=query,
        kv_cache=kv_cache,
        tile_q_indices=tile_q_indices,
        tile_block_tables=tile_block_tables,
        tile_masks=tile_masks,
        num_dynamic_loop_steps=num_dynamic_loop_steps,
        olm_buffer=olm_buffer,
        q_update_pred=q_update_pred,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        loop_unroll_factor=dynamic_loop_unroll_factor,
        batch_id=batch_id,
        head_id=head_id,
        k_h=k_h,
        q_h_per_k_h=q_h_per_k_h,
        softmax_scale=softmax_scale,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
    )
    prefill_active_and_epilogue(
        o=o,
        query=query,
        key=key,
        value=value,
        active_mask=active_mask,
        softmax_scale=softmax_scale,
        olm_buffer=olm_buffer,
        sinks=sinks,
        ACTIVE_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
        seqlen_q=seqlen_q,
        batch_id=batch_id,
        head_id=head_id,
        q_h_per_k_h=q_h_per_k_h,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
        B_F_SIZE=B_F_SIZE,
        B_D_SIZE=B_D_SIZE,
        skip_active=skip_active,
    )


@NKIOpRegistry.register(
    "mylib::flash_paged_attention_blocksparse_kv_cache_custom_op",
)
def flash_paged_attention_blocksparse_kv_cache(
    query,
    key,
    value,
    kv_cache,
    tile_q_indices,
    tile_block_tables,
    tile_masks,
    active_mask,
    num_dynamic_loop_steps,
    last_tile_indices,
    q_update_pred=None,
    sinks=None,
    dynamic_loop_unroll_factor=1,
    softmax_scale=None,
    mixed_precision=True,
    skip_active=False,
    decode_mode=False,
):
    """
    Flash PagedAttention Forward Kernel.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape (1, n_heads, seq_q, d)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - kv_cache: (2, max_num_blocks, n_kv_heads, block_size, d)
      - tile_q_indices: (num_large_tiles, large_tile_size_q)
      - tile_block_tables:
          (num_large_tiles, num_block_per_large_tile)
      - tile_masks:
          (B_P_SIZE, num_large_tiles,
           large_tile_size_k // B_P_SIZE) if decode_mode
          else (B_P_SIZE, num_large_tiles,
           large_tile_size_q // B_P_SIZE,
           large_tile_size_k)
      - active_mask: (seq_q, seq_q)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch
        dimension is always 1, and different requests are
        concatenated along sequence dimension.
      - We use paged cache blocks (key_cache, value_cache)
        to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same
        dtype except for block_tables (uint32) and mask
        (uint8)
      - If mixed_percision is True, then all Tensor Engine
        operation will be performed in bfloat16 and
        accumulation will be performed in float32. Otherwise
        the intermediates will be in the same type as the
        inputs.

    Compile-time Constants:
      - sequence_parallel_group: sequence parallel group to
        shard the cache blocks, List[int].
      - softmax_scale: scaling for softmax, is None,
        default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32
        precision, defualt is set to `true`, if false,
        we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    b, h, k_h, seqlen_q, d = check_input_shapes_kv_cache(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        tile_masks=tile_masks,
        active_mask=active_mask,
        decode_mode=decode_mode,
        skip_active=skip_active,
    )
    B_D_SIZE = d
    q_h_per_k_h = h // k_h

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)

    assert nl.program_ndim() == 2, (
        f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    )

    batch_id = nl.program_id(axis=0)  # equals 0
    head_id = nl.program_id(axis=1)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    if decode_mode:
        decode_subkernel_kv_cache(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            q_update_pred=q_update_pred,
            last_tile_indices=last_tile_indices,
            o=o,
            sinks=sinks,
            num_dynamic_loop_steps=num_dynamic_loop_steps,
            dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
            batch_id=batch_id,
            head_id=head_id,
            seqlen_q=seqlen_q,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            skip_active=skip_active,
        )
    else:
        prefill_subkernel_kv_cache(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            tile_q_indices=tile_q_indices,
            tile_block_tables=tile_block_tables,
            tile_masks=tile_masks,
            active_mask=active_mask,
            q_update_pred=q_update_pred,
            last_tile_indices=last_tile_indices,
            o=o,
            sinks=sinks,
            num_dynamic_loop_steps=num_dynamic_loop_steps,
            dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
            batch_id=batch_id,
            head_id=head_id,
            seqlen_q=seqlen_q,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            softmax_scale=softmax_scale,
            skip_active=skip_active,
        )
    return o
