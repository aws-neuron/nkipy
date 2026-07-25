"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Flash Paged Attention kernels with variable-length sequence inputs.

"""

from enum import IntFlag

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.isa.constants import oob_mode

from .constants import B_FMAX_SIZE, B_P_SIZE
from .context_parallel import softmax_correction_allgather, softmax_correction_allreduce
from .flash_pa_with_schedule import (
    allocate_decode_accum_buffers,
    allocate_prefill_accum_buffers,
    decode_active_and_epilogue,
    decode_context_tokens,
    decode_gather_token_last_accum_tile,
    prefill_active_and_epilogue,
    prefill_context_tokens,
    prefill_epilogue,
    prepare_q_update_pred,
)
from .utils import (
    get_program_sharding_info,
    is_power_of_2,
)


class BatchContextInfo(IntFlag):
    NoContext = 0
    PrefillContext = 1
    DecodeContext = 2
    MixedContext = PrefillContext | DecodeContext

    @property
    def has_prefill_ctx(self) -> bool:
        return bool(self & BatchContextInfo.PrefillContext)

    @property
    def has_decode_ctx(self) -> bool:
        return bool(self & BatchContextInfo.DecodeContext)

    @property
    def mixed_ctx(self) -> bool:
        return self == BatchContextInfo.MixedContext

    @property
    def prefill_ctx_only(self) -> bool:
        return self == BatchContextInfo.PrefillContext

    @property
    def decode_ctx_only(self) -> bool:
        return self == BatchContextInfo.DecodeContext

    @property
    def no_context(self) -> bool:
        return self == BatchContextInfo.NoContext


def check_batch_mode(
    prefill_num_dynamic_loop_steps,
    decode_num_dynamic_loop_steps,
):
    batch_ctx_info = BatchContextInfo.NoContext
    if prefill_num_dynamic_loop_steps is not None:
        batch_ctx_info |= BatchContextInfo.PrefillContext
        assert prefill_num_dynamic_loop_steps.dtype == nl.int32
    if decode_num_dynamic_loop_steps is not None:
        batch_ctx_info |= BatchContextInfo.DecodeContext
        assert decode_num_dynamic_loop_steps.dtype == nl.int32
    return batch_ctx_info


def check_input(
    *,
    query,
    key,
    value,
    key_cache,
    value_cache,
    prefill_num_dynamic_loop_steps,
    decode_num_dynamic_loop_steps,
    prefill_tile_masks,
    decode_tile_masks,
    active_mask,
    skip_active,
    decode_last_tile_indices,
):
    batch_ctx_info = check_batch_mode(
        prefill_num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        decode_num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
    )
    b, h, seqlen_q, d = query.shape
    assert b == 1, (
        f"Varlen kernel requires batch to be flattened, i.e. Ragged Tensor, got {b=}"
    )
    assert d >= 16 and d <= 128 and is_power_of_2(d), (
        f" we head_dim must be power of 2 in range [16, 128], got head dim {d}"
    )
    num_blocks, k_h, block_size, _ = key_cache.shape
    assert tuple(key_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{key_cache.shape=} mismatch!"
    assert tuple(value_cache.shape) == (
        num_blocks,
        k_h,
        block_size,
        d,
    ), f"{value_cache.shape=} mismatch!"
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

    if batch_ctx_info.has_prefill_ctx:
        # tile size from prefill
        INNER_Q_TILE_SIZE = prefill_tile_masks.shape[0]
        assert prefill_tile_masks.dtype == nl.uint8, (
            f"{prefill_tile_masks.dtype=} is expected to be uint8"
        )
        PREFILL_LARGE_KV_TILE_SIZE = prefill_tile_masks.shape[-1]
        assert PREFILL_LARGE_KV_TILE_SIZE % B_FMAX_SIZE == 0, (
            f"{PREFILL_LARGE_KV_TILE_SIZE=} not divisible by ({B_FMAX_SIZE=})"
        )
    else:
        INNER_Q_TILE_SIZE = min(seqlen_q, B_P_SIZE)
    if batch_ctx_info.has_decode_ctx:
        DECODE_K_TILE_SIZE = decode_tile_masks.shape[0]
        assert DECODE_K_TILE_SIZE == B_P_SIZE
        assert decode_tile_masks.dtype == nl.uint8, (
            f"{decode_tile_masks.dtype=} is expected to be uint8"
        )

    assert seqlen_q <= 8192, (
        f"Large {seqlen_q=} may consume too much sbuf space, not tested"
    )
    if seqlen_q <= B_P_SIZE:
        assert is_power_of_2(seqlen_q), f"{seqlen_q=} is expected to be power of 2"
    elif seqlen_q <= B_FMAX_SIZE:
        assert seqlen_q % B_P_SIZE == 0, f"{seqlen_q=} must be mulitple of {B_P_SIZE=}"
    else:
        assert seqlen_q % B_FMAX_SIZE == 0, (
            f"{seqlen_q=} must be multiple of {B_FMAX_SIZE=}"
        )
    assert seqlen_q % INNER_Q_TILE_SIZE == 0, (
        f"{seqlen_q=} must be multiple of {INNER_Q_TILE_SIZE=}"
    )
    assert decode_last_tile_indices is None or decode_last_tile_indices.shape[1] == 2, (
        f"{decode_last_tile_indices.shape=}"
    )

    if batch_ctx_info.no_context:
        assert active_mask is not None and not skip_active, (
            f"{active_mask is None=} {skip_active=}"
        )

    if active_mask is not None:
        assert active_mask.dtype == nl.uint8, (
            f"{active_mask.dtype=} is expected to be uint8"
        )

    return (
        batch_ctx_info,
        b,
        h,
        k_h,
        seqlen_q,
        d,
        INNER_Q_TILE_SIZE,
    )


def merge_decode_buffer(
    olm_buffer,
    decode_olm_buffer,
    decode_last_tile_indices_sbuf,
    kv_head_id,
):
    _, _, q_h_per_k_h, total_feat = olm_buffer.shape
    B_D_SIZE = total_feat - 2
    decode_olm_sbuf = decode_gather_token_last_accum_tile(
        olm_buffer=decode_olm_buffer,
        last_tile_indices_sbuf=decode_last_tile_indices_sbuf,
        q_h_per_k_h=q_h_per_k_h,
        B_D_SIZE=B_D_SIZE,
    )
    TILE_SIZE, NUM_TILES, _, _ = decode_olm_sbuf.shape
    for i in nl.affine_range(NUM_TILES):
        i_p = nl.arange(TILE_SIZE)[:, None, None]
        i_f_h = nl.arange(q_h_per_k_h)[None, :, None]
        i_f_d = nl.arange(B_D_SIZE + 2)[None, None, :]
        nl.store(
            olm_buffer[
                kv_head_id, decode_last_tile_indices_sbuf[i_p, i, 1], i_f_h, i_f_d
            ],
            decode_olm_sbuf[i_p, i, i_f_h, i_f_d],
            mode=oob_mode.skip,
        )


@nki.compiler.skip_middle_end_transformations
@nki.jit(
    experimental_flags="experimental-native-scalar-support, experimental-local-tensor-parent",
    enable_out_of_bound_check=False,
)
def flash_paged_attention_varlen(
    query,
    key,
    value,
    key_cache,
    value_cache,
    active_mask,
    sink,
    prefill_tile_q_indices,
    prefill_tile_block_tables,
    prefill_tile_masks,
    prefill_num_dynamic_loop_steps,
    prefill_q_update_pred,
    prefill_last_tile_indices,
    decode_tile_q_indices,
    decode_tile_block_tables,
    decode_tile_masks,
    decode_num_dynamic_loop_steps,
    decode_q_update_pred,
    decode_last_tile_indices,
    dynamic_loop_unroll_factor=1,
    softmax_scale=None,
    mixed_precision=True,
    skip_active=False,
    cp_replica_group=None,
):
    """
    Flash PagedAttention Forward Kernel with Both Prefill and Decode Requests.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape (1, n_q_heads, seq_q, d)
      - key:   shape (1, n_kv_heads, d, seq_k)
      - value: shape (1, n_kv_heads, seq_v, d)
      - key_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - value_cache: (max_num_blocks, n_kv_heads, block_size, d)
      - active_mask: (seq_q, seq_q) or None
      - sink: (n_q_heads, 1)
      - prefill_tile_q_indices: (max_num_prefill_tiles, large_tile_size_q)
      - prefill_tile_block_tables: (max_num_prefill_tiles, num_block_per_large_tile)
      - prefill_tile_masks: (B_P_SIZE, max_num_prefill_tiles, large_tile_size_q // B_P_SIZE, large_tile_size_k)
      - prefill_num_dynamic_loop_steps: (1, 1)
      - prefill_last_tile_indices: (max_num_prefill_q_tiles, 2)
      - prefill_q_update_pred: None or (max_num_prefill_tiles, 1)
      - decode_tile_q_indices: (max_num_decode_tiles, 1)
      - decode_tile_block_tables: (max_num_decode_tiles, num_block_per_large_tile)
      - decode_tile_masks: (B_P_SIZE, max_num_decode_tiles, large_tile_size_k // B_P_SIZE)
      - decode_num_dynamic_loop_steps: (1, 1)
      - decode_last_tile_indices: (max_num_decode_tokens, 2)
      - decode_q_update_pred: (max_num_decode_tiles, 1)

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
      - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`,
          if false, we use same precision as input types

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    batch_ctx_info, b, h, k_h, seqlen_q, d, INNER_Q_TILE_SIZE = check_input(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        prefill_num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        decode_num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
        prefill_tile_masks=prefill_tile_masks,
        decode_tile_masks=decode_tile_masks,
        active_mask=active_mask,
        skip_active=skip_active,
        decode_last_tile_indices=decode_last_tile_indices,
    )
    assert sink is None or sink.shape == (h, 1)
    use_cp = cp_replica_group is not None and len(cp_replica_group[0]) > 1

    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

    batch_id = 0
    n_program, head_id = get_program_sharding_info()
    assert n_program == k_h, f"{n_program=} {k_h=}"

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    B_D_SIZE = d
    q_h_per_k_h = h // k_h

    B_F_SIZE = B_FMAX_SIZE

    # Two types of prefill requests:
    # 1. prefill with prior context
    # 2. prefill from scratch (no context)
    has_prefill_reqs = batch_ctx_info.has_prefill_ctx or active_mask is not None

    # Decode requests must have context
    has_decode_reqs = batch_ctx_info.has_decode_ctx

    if has_prefill_reqs:
        (olm_buffer,) = allocate_prefill_accum_buffers(
            seqlen_q=seqlen_q,
            INNER_Q_TILE_SIZE=INNER_Q_TILE_SIZE,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            acc_type=acc_type,
            n_program=n_program,
            head_id=head_id,
        )
    if batch_ctx_info.has_prefill_ctx:
        PREFILL_MAX_NUM_TILE = prefill_tile_masks.shape[1]
        assert (
            PREFILL_MAX_NUM_TILE > 0
            and PREFILL_MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
        ), f"{PREFILL_MAX_NUM_TILE=} {dynamic_loop_unroll_factor=}"
        if prefill_q_update_pred is None:
            prefill_last_tile_indices_sbuf = nl.load(prefill_last_tile_indices)
            prefill_q_update_pred_hbm = prepare_q_update_pred(
                prefill_last_tile_indices_sbuf,
                PREFILL_MAX_NUM_TILE,
            )
        else:
            prefill_q_update_pred_hbm = prefill_q_update_pred
        prefill_q_update_pred = prefill_q_update_pred_hbm.reshape(
            (
                PREFILL_MAX_NUM_TILE // dynamic_loop_unroll_factor,
                dynamic_loop_unroll_factor,
                1,
            )
        )
        prefill_context_tokens(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            tile_q_indices=prefill_tile_q_indices,
            tile_block_tables=prefill_tile_block_tables,
            tile_masks=prefill_tile_masks,
            num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
            olm_buffer=olm_buffer,
            q_update_pred=prefill_q_update_pred,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            loop_unroll_factor=dynamic_loop_unroll_factor,
            seqlen_q=seqlen_q,
            batch_id=batch_id,
            head_id=head_id,
            k_h=k_h,
            q_h_per_k_h=q_h_per_k_h,
            softmax_scale=softmax_scale,
            B_F_SIZE=B_F_SIZE,
            B_D_SIZE=B_D_SIZE,
        )
    ACTIVE_Q_TILE_SIZE = min(seqlen_q, B_P_SIZE)
    if batch_ctx_info.has_decode_ctx:
        DECODE_MAX_NUM_TILE = decode_tile_masks.shape[1]
        assert (
            DECODE_MAX_NUM_TILE > 0
            and DECODE_MAX_NUM_TILE % dynamic_loop_unroll_factor == 0
        )
        (decode_olm_buffer,) = allocate_decode_accum_buffers(
            MAX_NUM_TILE=DECODE_MAX_NUM_TILE,
            q_h_per_k_h=q_h_per_k_h,
            B_D_SIZE=B_D_SIZE,
            acc_type=acc_type,
        )
        if seqlen_q < decode_last_tile_indices.shape[0]:
            # decode indices might be padded to B_P_SIZE=128 when seqlen_q < 128
            decode_last_tile_indices = decode_last_tile_indices.reshape(
                (decode_last_tile_indices.shape[0], 1, 2)
            )
        else:
            assert decode_last_tile_indices.shape[0] % ACTIVE_Q_TILE_SIZE == 0
            decode_last_tile_indices = decode_last_tile_indices.reshape(
                (
                    ACTIVE_Q_TILE_SIZE,
                    decode_last_tile_indices.shape[0] // ACTIVE_Q_TILE_SIZE,
                    2,
                )
            )
        decode_last_tile_indices_sbuf = nl.load(
            decode_last_tile_indices[nl.ds(0, ACTIVE_Q_TILE_SIZE)]
        )
        decode_q_update_pred = decode_q_update_pred.reshape(
            (
                DECODE_MAX_NUM_TILE // dynamic_loop_unroll_factor,
                dynamic_loop_unroll_factor,
            )
        )

        decode_context_tokens(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            tile_q_indices=decode_tile_q_indices,
            tile_masks=decode_tile_masks,
            tile_block_tables=decode_tile_block_tables,
            num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
            olm_buffer=decode_olm_buffer,
            q_update_pred=decode_q_update_pred,
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

    if has_decode_reqs:
        if has_prefill_reqs:
            # mixed prefill and decode, merge decode olm buffer into global accum buffers
            merge_decode_buffer(
                olm_buffer,
                decode_olm_buffer=decode_olm_buffer,
                decode_last_tile_indices_sbuf=decode_last_tile_indices_sbuf,
                kv_head_id=head_id,
            )
        else:
            # slice decode final results from unrolled tiles
            decode_olm_buffer_sbuf = decode_gather_token_last_accum_tile(
                olm_buffer=decode_olm_buffer,
                last_tile_indices_sbuf=decode_last_tile_indices_sbuf,
                q_h_per_k_h=q_h_per_k_h,
                B_D_SIZE=B_D_SIZE,
            )
    # handle context parallel
    if use_cp and not batch_ctx_info.no_context:
        if not has_prefill_reqs:
            # must be decode-only
            # write buffer to shared hbm for context parallel
            olm_buffer = nl.ndarray(
                (k_h, seqlen_q, q_h_per_k_h, B_D_SIZE + 2),
                dtype=acc_type,
                buffer=nl.shared_hbm,
            )
            olm_buffer_reshaped = olm_buffer.reshape(
                (k_h,) + decode_olm_buffer_sbuf.shape
            )
            nl.store(dst=olm_buffer_reshaped[head_id], value=decode_olm_buffer_sbuf)

        cp_group_size = len(cp_replica_group[0])
        use_allgather = False
        if use_allgather:
            olm_buffer_ag = nl.ndarray(
                (cp_group_size * k_h, seqlen_q, q_h_per_k_h, olm_buffer.shape[-1]),
                dtype=acc_type,
                buffer=nl.shared_hbm,
            )
            softmax_correction_allgather(
                olm_buffer=olm_buffer,
                olm_buffer_ag=olm_buffer_ag,
                ACTIVE_Q_TILE_SIZE=ACTIVE_Q_TILE_SIZE,
                acc_type=acc_type,
                cp_replica_group=cp_replica_group,
                kv_head_id=head_id,
            )
        else:
            max_allreduce_buf = nl.ndarray(
                (k_h, seqlen_q, q_h_per_k_h, 1),
                dtype=acc_type,
                buffer=nl.shared_hbm,
            )
            ol_allreduce_buf = nl.ndarray(
                (k_h, seqlen_q, q_h_per_k_h, B_D_SIZE + 1),
                dtype=acc_type,
                buffer=nl.shared_hbm,
            )
            softmax_correction_allreduce(
                olm_buffer=olm_buffer,
                max_allreduce_buf=max_allreduce_buf,
                ol_allreduce_buf=ol_allreduce_buf,
                ACTIVE_Q_TILE_SIZE=ACTIVE_Q_TILE_SIZE,
                acc_type=acc_type,
                cp_replica_group=cp_replica_group,
                kv_head_id=head_id,
            )
            olm_buffer = (ol_allreduce_buf, max_allreduce_buf)
        if not has_prefill_reqs:
            if use_allgather:
                decode_olm_buffer_sbuf[...] = nl.load(olm_buffer_reshaped[head_id])
            else:
                decode_olm_layout = decode_olm_buffer_sbuf.shape[:-1]
                max_allreduce_buf_reshaped = max_allreduce_buf.reshape(
                    (k_h,) + decode_olm_layout + (1,)
                )
                ol_allreduce_buf_reshaped = ol_allreduce_buf.reshape(
                    (k_h,) + decode_olm_layout + (B_D_SIZE + 1,)
                )
                decode_olm_buffer_sbuf[:, :, :, nl.ds(B_D_SIZE + 1, 1)] = nl.load(
                    max_allreduce_buf_reshaped[head_id]
                )
                decode_olm_buffer_sbuf[:, :, :, nl.ds(0, B_D_SIZE + 1)] = nl.load(
                    ol_allreduce_buf_reshaped[head_id]
                )

    o = nl.ndarray((b, h, seqlen_q, d), dtype=query.dtype, buffer=nl.shared_hbm)
    if not has_prefill_reqs:
        decode_active_and_epilogue(
            o=o,
            query=query,
            key=key,
            value=value,
            olm_buffer_sbuf=decode_olm_buffer_sbuf,
            sink=sink,
            softmax_scale=softmax_scale,
            batch_id=batch_id,
            head_id=head_id,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            skip_active=skip_active,
        )
    elif skip_active:
        prefill_epilogue(
            o=o,
            olm_buffer=olm_buffer,
            sink=sink,
            ACTIVE_Q_TILE_SIZE=ACTIVE_Q_TILE_SIZE,
            q_h_per_k_h=q_h_per_k_h,
            batch_id=batch_id,
            head_id=head_id,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
        )
    else:
        prefill_active_and_epilogue(
            o=o,
            query=query,
            key=key,
            value=value,
            active_mask=active_mask,
            olm_buffer=olm_buffer,
            sink=sink,
            softmax_scale=softmax_scale,
            ACTIVE_Q_TILE_SIZE=ACTIVE_Q_TILE_SIZE,
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
    return o


def flash_attn_varlen_nkifunc(
    *,
    query,
    key,
    value,
    key_cache,
    value_cache,
    active_mask,
    sink,
    prefill_tile_q_indices,
    prefill_tile_block_tables,
    prefill_tile_masks,
    prefill_num_dynamic_loop_steps,
    prefill_last_tile_indices,
    prefill_q_update_pred,
    decode_tile_q_indices,
    decode_tile_block_tables,
    decode_tile_masks,
    decode_num_dynamic_loop_steps,
    decode_last_tile_indices,
    decode_q_update_pred,
    dynamic_loop_unroll_factor=1,
    n_kv_head=None,
    head_size=None,
    mixed_precision=True,
    skip_active=False,
    save_artifact_dir=None,
    cp_replica_group=None,
):
    if n_kv_head is None:
        n_kv_head = key_cache.shape[1]
    assert key_cache.shape[1] == n_kv_head
    if head_size is None:
        head_size = key_cache.shape[-1]
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        active_mask=active_mask,
        sink=sink,
        prefill_tile_q_indices=prefill_tile_q_indices,
        prefill_tile_block_tables=prefill_tile_block_tables,
        prefill_tile_masks=prefill_tile_masks,
        prefill_num_dynamic_loop_steps=prefill_num_dynamic_loop_steps,
        prefill_last_tile_indices=prefill_last_tile_indices,
        prefill_q_update_pred=prefill_q_update_pred,
        decode_tile_q_indices=decode_tile_q_indices,
        decode_tile_block_tables=decode_tile_block_tables,
        decode_tile_masks=decode_tile_masks,
        decode_num_dynamic_loop_steps=decode_num_dynamic_loop_steps,
        decode_last_tile_indices=decode_last_tile_indices,
        decode_q_update_pred=decode_q_update_pred,
        dynamic_loop_unroll_factor=dynamic_loop_unroll_factor,
        softmax_scale=1.0 / (head_size**0.5),
        mixed_precision=mixed_precision,
        skip_active=skip_active,
        cp_replica_group=cp_replica_group,
    )
    if n_kv_head > 1:
        assert n_kv_head == 2
        launch_grid = (nl.nc(2),)
    else:
        assert n_kv_head == 1
        launch_grid = n_kv_head
    if save_artifact_dir:
        assert isinstance(query, np.ndarray), (
            "Only Numpy Kernel supports saving artifact"
        )
        return nki.baremetal(
            flash_paged_attention_varlen,
            debug_kernel=True,
            artifacts_dir=save_artifact_dir,
        )[launch_grid](**kwargs)
    else:
        return flash_paged_attention_varlen[launch_grid](**kwargs)
