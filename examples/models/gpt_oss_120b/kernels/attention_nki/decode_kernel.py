"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Migrated from the legacy ``neuronxcc.nki`` API to standalone ``nki`` (beta-3):
- ``par_dim`` removed; ``@nki.compiler.skip_middle_end_transformations`` and the
  ``debug_kernel`` / ``experimental_flags`` @nki.jit kwargs dropped.
- ``nl.program_id`` grid removed: decode always has a single kv head (k_h == 1),
  so ``kv_head_id`` is just 0.
- ``nl.arange`` + old ``nisa.iota(expr, dtype=...)`` replaced by beta-3
  ``nisa.iota(dst, pattern=[[step, size]], channel_multiplier=...)``.
- identity-matrix matmul transposes replaced by tensor-engine ``nc_transpose``
  (see flash_attn_core.transpose_with_matmul) — beta-3 can't build np.identity
  inside a kernel body.
- ``nisa`` ops are dst-first and return nothing.
- mutable cache operands use ``nkipy.core.typing.mutable_tensor``.
"""

import nkipy.core.typing as nt
import nki
import nki.language as nl
import nki.isa as nisa

from .flash_attn_core import (
    _flash_attention_core_kq_matmul,
    _active_attention_core_batched,
    partition_broadcast_fp32,
    transpose_with_matmul,
)
from .attn_utils import B_P_SIZE


def apply_rope(x, sin, cos):
    assert len(x.shape) == 3 and len(sin.shape) == 3 and sin.shape == cos.shape
    batch, n_head, dd = x.shape
    half_d = sin.shape[-1]
    assert dd == half_d * 2 and sin.shape[1] == 1
    out = nl.ndarray(x.shape, dtype=x.dtype)
    # out0 = x0*cos - x1*sin ; out1 = x0*sin + x1*cos. sin/cos are per-token
    # (n_head==1) so broadcast over the head dim by looping (beta-3 tensor_tensor
    # does not broadcast a middle dim).
    t0 = nl.ndarray((batch, 1, half_d), dtype=x.dtype)
    t1 = nl.ndarray((batch, 1, half_d), dtype=x.dtype)
    sin2 = sin[:, 0, :]
    cos2 = cos[:, 0, :]
    for hh in nl.affine_range(n_head):
        x0 = x[:, hh, nl.ds(0, half_d)]
        x1 = x[:, hh, nl.ds(half_d, half_d)]
        nisa.tensor_tensor(dst=t0[:, 0, :], data1=x0, data2=cos2, op=nl.multiply)
        nisa.tensor_tensor(dst=t1[:, 0, :], data1=x1, data2=sin2, op=nl.multiply)
        nisa.tensor_tensor(
            dst=out[:, hh, nl.ds(0, half_d)], data1=t0[:, 0, :], data2=t1[:, 0, :], op=nl.subtract
        )
        nisa.tensor_tensor(dst=t0[:, 0, :], data1=x0, data2=sin2, op=nl.multiply)
        nisa.tensor_tensor(dst=t1[:, 0, :], data1=x1, data2=cos2, op=nl.multiply)
        nisa.tensor_tensor(
            dst=out[:, hh, nl.ds(half_d, half_d)], data1=t0[:, 0, :], data2=t1[:, 0, :], op=nl.add
        )
    return out


def load_qkv_and_apply_rope(qkv, sin, cos, position_ids, h, k_h, kv_head_id):
    batch_size, _, d = qkv.shape
    qkv_sbuf = nl.load(qkv)
    q_h_per_k_h = h // k_h
    q_sbuf = nl.ndarray((batch_size, q_h_per_k_h, d), dtype=qkv.dtype)
    nisa.tensor_copy(
        dst=q_sbuf, src=qkv_sbuf[:, nl.ds(kv_head_id * q_h_per_k_h, q_h_per_k_h), :]
    )
    k_sbuf = nl.ndarray((batch_size, 1, d), dtype=qkv.dtype)
    nisa.tensor_copy(dst=k_sbuf, src=qkv_sbuf[:, nl.ds(h + kv_head_id, 1), :])
    v_sbuf = nl.ndarray((batch_size, 1, d), dtype=qkv.dtype)
    nisa.tensor_copy(dst=v_sbuf, src=qkv_sbuf[:, nl.ds(h + k_h + kv_head_id, 1), :])

    position_ids_sbuf = nl.load(position_ids.reshape((batch_size, 1)))
    # Gather the sin/cos rows for each batch's position in a single indirect DMA
    # each: vector_select(0, pos) on the 2-D (S, d//2) tensor picks row pos[p]
    # for partition p and copies the whole d//2-wide row. (A per-column loop here
    # floods the gpsimd/DMA engine with d//2 tiny gathers and dominated decode.)
    half_d = d // 2
    sin_g = sin.reshape((sin.shape[0], half_d))
    cos_g = cos.reshape((cos.shape[0], half_d))
    sin_row = nl.ndarray((batch_size, half_d), dtype=sin.dtype)
    cos_row = nl.ndarray((batch_size, half_d), dtype=cos.dtype)
    nisa.dma_copy(dst=sin_row, src=sin_g.vector_select(0, position_ids_sbuf))
    nisa.dma_copy(dst=cos_row, src=cos_g.vector_select(0, position_ids_sbuf))
    sin_sbuf = sin_row.reshape((batch_size, 1, half_d))
    cos_sbuf = cos_row.reshape((batch_size, 1, half_d))
    q_rope = apply_rope(q_sbuf, sin_sbuf, cos_sbuf)
    k_rope = apply_rope(k_sbuf, sin_sbuf, cos_sbuf)
    return q_rope, k_rope, v_sbuf


def update_kv_cache(kv_head_id, k, v, cache_k, cache_v, position_ids):
    batch_size, num_heads, head_size, max_model_len = cache_k.shape
    assert position_ids.shape == (1, batch_size)
    assert cache_v.shape == (batch_size, max_model_len, num_heads, head_size)
    assert k.shape == (head_size, batch_size)
    assert v.shape == (batch_size, head_size)
    assert kv_head_id == 0

    # ---- cache_v[b, pos[b], 0, :] = v[b, :]  (per-token row scatter) ----
    # write_offset[b] = b*max_model_len + pos[b]  into the (B*S, H) view.
    batch_starts = nl.ndarray((1, batch_size), dtype=nl.int32)
    nisa.iota(dst=batch_starts, pattern=[[max_model_len, batch_size]])
    write_offsets = nl.ndarray((1, batch_size), dtype=nl.int32)
    nisa.tensor_tensor(dst=write_offsets, data1=batch_starts, data2=position_ids, op=nl.add)
    # transpose to (batch_size, 1) so it is a per-partition scalar offset
    write_offsets_t = nl.ndarray((batch_size, 1), dtype=nl.int32)
    wo_psum = nl.ndarray((batch_size, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=wo_psum, data=write_offsets.view(nl.float32), engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=write_offsets_t.view(nl.float32), src=wo_psum)
    cache_v_reshaped = cache_v.reshape((batch_size * max_model_len, num_heads * head_size))
    nisa.dma_copy(
        dst=cache_v_reshaped.vector_select(0, write_offsets_t),
        src=v[0:batch_size, 0:head_size],
    )

    # ---- cache_k[b, 0, :, pos[b]] = k[:, b]  (per-token column scatter) ----
    # cache_k is (B, 1, D, S); flatten to (B*D*S, 1). For batch b and feature
    # f the destination flat index is b*(D*S) + f*S + pos[b].
    # Build offsets[f, b] on partition=f (head_size) via broadcast + iota.
    # pos broadcast to (head_size, batch_size):
    pos_br_psum = nl.ndarray((head_size, batch_size), dtype=nl.float32, buffer=nl.psum)
    partition_broadcast_fp32(position_ids.view(nl.float32), pos_br_psum)
    offsets = nl.ndarray((head_size, batch_size), dtype=nl.int32)
    nisa.tensor_copy(dst=offsets.view(nl.float32), src=pos_br_psum)
    # add f*S (per-partition, channel_multiplier=S). The b*(D*S) batch term is
    # added as a python-constant scalar inside the loop: an iota step of
    # num_heads*head_size*max_model_len would exceed the int16 iota-step limit.
    feat_term = nl.ndarray((head_size, batch_size), dtype=nl.int32)
    nisa.iota(
        dst=feat_term,
        pattern=[[0, batch_size]],
        channel_multiplier=max_model_len,
    )
    nisa.tensor_tensor(dst=offsets, data1=offsets, data2=feat_term, op=nl.add)
    cache_k_flat = cache_k.reshape((batch_size * num_heads * head_size * max_model_len, 1))
    stride_b = num_heads * head_size * max_model_len
    for b_i in range(batch_size):
        off_b = nl.ndarray((head_size, 1), dtype=nl.int32)
        nisa.tensor_scalar(
            dst=off_b,
            data=offsets[0:head_size, nl.ds(b_i, 1)],
            op0=nl.add,
            operand0=b_i * stride_b,
        )
        nisa.dma_copy(
            dst=cache_k_flat.vector_select(0, off_b),
            src=k[0:head_size, nl.ds(b_i, 1)],
        )


def transpose_qkv_with_pe(src, out_sbuf, scale):
    # Transpose src (P, F) -> (F, P) via the tensor engine, optionally scaling.
    # nc_transpose (gen3+) requires the PSUM dst dtype to match the input dtype.
    out_psum = nl.ndarray(out_sbuf.shape, dtype=src.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=out_psum, data=src, engine=nisa.engine.tensor)
    if scale is not None and scale != 1:
        nisa.activation(dst=out_sbuf, op=nl.copy, data=out_psum, scale=scale)
    else:
        nisa.tensor_copy(dst=out_sbuf, src=out_psum)


@nki.jit
def flash_attn_decode(
    qkv,
    cos,
    sin,
    cache_k: nt.mutable_tensor,
    cache_v: nt.mutable_tensor,
    sink,
    position_ids,
    sliding_window,
    tile_masks=None,
    softmax_scale=None,
    mixed_precision=True,
    LARGE_KV_TILE_SIZE=None,
):
    """
    Flash Attention Forward kernel (token generation / decode).

    IO tensor layouts:
      - qkv: shape (bs, h + 2*k_h, d)
      - cos/sin: shape (max_model_len, d//2)
      - cache_k: shape (bs, k_h, d, max_model_len)  (mutable, updated in place)
      - cache_v: shape (bs, max_model_len, k_h, d)  (mutable, updated in place)
      - sink: shape (n_heads, 1)
      - position_ids: shape (bs,)
      - tile_masks: optional precomputed masks

    beta-3 note: decode always runs with a single kv head (k_h == 1) at LNC1,
    so there is no SPMD grid; kv_head_id is 0.
    """
    b, total_heads, d = qkv.shape
    assert b <= B_P_SIZE, f"batch size {b} not supported"
    B_D_SIZE = d
    assert d <= B_P_SIZE, f"We do not support head_dim {d} > {B_P_SIZE=}"
    _, k_h, _, max_model_len = cache_k.shape
    h = total_heads - k_h * 2
    q_h_per_k_h = h // k_h
    assert k_h == 1, f"Expecting single KV head but got {k_h=}"
    assert b * q_h_per_k_h <= B_P_SIZE, f"{b * q_h_per_k_h=} > {B_P_SIZE=}"
    assert tuple(cache_k.shape) == (
        b,
        k_h,
        d,
        max_model_len,
    ), f"Expect shape of cache_k to be {(b, k_h, d, max_model_len)=} but got {cache_k.shape=}"
    assert tuple(cache_v.shape) == (
        b,
        max_model_len,
        k_h,
        d,
    ), f"Expect shape of cache_v to be {(b, max_model_len, k_h, d)=} but got {cache_v.shape=}"
    kernel_dtype = qkv.dtype
    assert cache_k.dtype == cache_v.dtype == kernel_dtype
    acc_type = nl.float32 if mixed_precision else kernel_dtype

    # beta-3 has no user-defined SPMD launch grid: single kv head.
    kv_head_id = 0

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    if LARGE_KV_TILE_SIZE is None:
        LARGE_KV_TILE_SIZE = min(2048, max_model_len)
    assert LARGE_KV_TILE_SIZE % B_P_SIZE == 0

    assert (
        max_model_len % LARGE_KV_TILE_SIZE == 0
    ), f"{max_model_len=} to be divisible by {LARGE_KV_TILE_SIZE=}"

    num_large_k_tile = max_model_len // LARGE_KV_TILE_SIZE

    # =============== Global Flash Attention accumulators ====================== #
    o_buffer_sbuf = nl.ndarray(
        (B_D_SIZE, num_large_k_tile + 1, b * q_h_per_k_h),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (1, num_large_k_tile + 1, b * q_h_per_k_h),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (1, num_large_k_tile + 1, b * q_h_per_k_h),
        dtype=acc_type,
    )
    # =============== Global Flash Attention accumulators END ================== #
    q_rope, k_rope, v_sbuf = load_qkv_and_apply_rope(
        qkv,
        sin,
        cos,
        position_ids,
        h,
        k_h,
        kv_head_id,
    )

    q_sbuf = nl.ndarray((d, b, q_h_per_k_h), dtype=kernel_dtype)
    for q_h in nl.affine_range(q_h_per_k_h):
        transpose_qkv_with_pe(
            q_rope[:, q_h, :],
            q_sbuf[:, :, q_h],
            softmax_scale,
        )
    k_transposed_sbuf = nl.ndarray((d, b), dtype=kernel_dtype)
    v_transposed_sbuf = nl.ndarray((d, b), dtype=kernel_dtype)
    transpose_qkv_with_pe(k_rope[:, 0, :], k_transposed_sbuf, None)
    transpose_qkv_with_pe(v_sbuf[:, 0, :], v_transposed_sbuf, None)
    _active_attention_core_batched(
        q_sbuf,
        k_transposed_sbuf,
        v_transposed_sbuf,
        o_buffer_sbuf,
        l_buffer_sbuf,
        m_buffer_sbuf,
        sink,
        kernel_dtype,
        acc_type,
    )
    position_ids_sbuf = nl.load(position_ids.reshape((1, b)), dtype=nl.int32)

    if tile_masks is None:
        k_stride = LARGE_KV_TILE_SIZE // B_P_SIZE
        # token_pos[p, ki] = p * k_stride + ki  (absolute KV token index within a
        # large tile). Kept in fp32 (positions are small ints) so masks can be
        # built with the proven fp32 partition-broadcast + tensor_scalar compare.
        token_pos_f = nl.ndarray((B_P_SIZE, k_stride), dtype=nl.float32, buffer=nl.psum)
        iota_row = nl.ndarray((1, k_stride), dtype=nl.bfloat16)
        nisa.iota(dst=iota_row, pattern=[[1, k_stride]])
        # XXX: if k_stride can be very large (>128), use nl.float32
        partition_broadcast_fp32(iota_row, token_pos_f)
        token_pos_p = nl.ndarray((B_P_SIZE, k_stride), dtype=nl.float32)
        nisa.iota(
            dst=token_pos_p,
            pattern=[[0, k_stride]],
            channel_multiplier=k_stride,
        )
        token_pos = nl.ndarray((B_P_SIZE, k_stride), dtype=nl.float32)
        nisa.tensor_tensor(
            dst=token_pos,
            data1=token_pos_f,
            data2=token_pos_p,
            op=nl.add,
        )
        # Per-partition-broadcast position bounds: pos_ub[p, b] = position_id[b].
        pos_upper_bound = nl.ndarray((B_P_SIZE, b), dtype=nl.float32, buffer=nl.psum)
        pos_f = nl.ndarray((1, b), dtype=nl.float32)
        nisa.tensor_copy(dst=pos_f, src=position_ids_sbuf)
        partition_broadcast_fp32(pos_f, pos_upper_bound)
        pos_upper_bound_sb = nl.ndarray((B_P_SIZE, b), dtype=nl.float32)
        nisa.tensor_copy(dst=pos_upper_bound_sb, src=pos_upper_bound)
        pos_lower_bound_sb = nl.ndarray((B_P_SIZE, b), dtype=nl.float32)
        if sliding_window > 0:
            nisa.tensor_scalar(
                dst=pos_lower_bound_sb,
                data=pos_upper_bound_sb,
                op0=nl.subtract,
                operand0=float(sliding_window),
            )
        tile_masks_sbuf = None
    else:
        tile_masks_sbuf = nl.load(tile_masks, dtype=nl.uint8)

    MULTI_BUFFER = 2
    cache_k_reshaped = cache_k.reshape(
        (
            b,
            k_h,
            d,
            num_large_k_tile,
            LARGE_KV_TILE_SIZE,
        )
    )
    cache_v_reshaped = cache_v.reshape(
        (
            b,
            k_h,
            num_large_k_tile,
            B_P_SIZE,
            LARGE_KV_TILE_SIZE // B_P_SIZE,
            d,
        )
    )
    k_load_buffer = nl.ndarray(
        (B_D_SIZE, MULTI_BUFFER, b, LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    v_load_buffer = nl.ndarray(
        (B_P_SIZE, MULTI_BUFFER, b, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE),
        dtype=kernel_dtype,
    )

    if num_large_k_tile > 0:
        # fetch cache_k for step 0
        for b_i in nl.affine_range(b):
            nisa.dma_copy(
                dst=k_load_buffer[:, 0, b_i, :],
                src=cache_k_reshaped[b_i, kv_head_id, :, 0],
                dge_mode=nisa.dge_mode.swdge,
            )
    num_k_tiles = LARGE_KV_TILE_SIZE // B_P_SIZE
    for kv_tile_id in nl.sequential_range(num_large_k_tile):
        if tile_masks is None:
            # Build mask (B_P, b, num_k_tiles, q_h) matching kq_res_psum. For each
            # batch, compare token_pos[p, ki] against that batch's position bound
            # (a per-partition scalar), then broadcast across the q-head dim.
            mask = nl.ndarray(
                (B_P_SIZE, b, num_k_tiles, q_h_per_k_h), dtype=nl.uint8
            )
            for b_i in nl.affine_range(b):
                m_bk = nl.ndarray((B_P_SIZE, num_k_tiles), dtype=nl.uint8)
                nisa.tensor_scalar(
                    dst=m_bk,
                    data=token_pos,
                    op0=nl.less,
                    operand0=pos_upper_bound_sb[:, nl.ds(b_i, 1)],
                )
                if sliding_window > 0:
                    m_lo = nl.ndarray((B_P_SIZE, num_k_tiles), dtype=nl.uint8)
                    nisa.tensor_scalar(
                        dst=m_lo,
                        data=token_pos,
                        # greater (not >=) due to decomposed attention
                        op0=nl.greater,
                        operand0=pos_lower_bound_sb[:, nl.ds(b_i, 1)],
                    )
                    nisa.tensor_tensor(
                        dst=m_bk, data1=m_bk, data2=m_lo, op=nl.logical_and
                    )
                for q_h in nl.affine_range(q_h_per_k_h):
                    nisa.tensor_copy(dst=mask[:, b_i, :, q_h], src=m_bk)
        else:
            # Fed mask is (B_P, b, num_k_tiles); expand to (B_P, b, num_k_tiles,
            # q_h) to match kq_res_psum for the predicated copy.
            fed = tile_masks_sbuf[:, kv_tile_id]
            mask = nl.ndarray(
                (B_P_SIZE, b, num_k_tiles, q_h_per_k_h), dtype=nl.uint8
            )
            for q_h in nl.affine_range(q_h_per_k_h):
                nisa.tensor_copy(dst=mask[:, :, :, q_h], src=fed)
        if tile_masks is None and kv_tile_id < num_large_k_tile:
            nisa.tensor_scalar(
                dst=token_pos,
                data=token_pos,
                op0=nl.add,
                operand0=float(LARGE_KV_TILE_SIZE),
            )
        _flash_attention_core_kq_matmul(
            q_sbuf,
            cache_k_reshaped,
            cache_v_reshaped,
            k_load_buffer,
            v_load_buffer,
            mask,
            o_buffer_sbuf,
            l_buffer_sbuf,
            m_buffer_sbuf,
            kv_tile_id,
            num_large_k_tile,
            kv_head_id,
            kernel_dtype,
            acc_type,
        )

    # -------- write output to buffer on HBM ------------ #
    l_broadcasted_psum = nl.ndarray(
        (B_D_SIZE, b * q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.psum,
    )
    last_large_tile = num_large_k_tile
    partition_broadcast_fp32(l_buffer_sbuf[:, last_large_tile, :], l_broadcasted_psum)
    # out buffer layout (B_D_SIZE, batch_size * q_h_per_k_h)
    l_inv = nl.ndarray((B_D_SIZE, b * q_h_per_k_h), dtype=acc_type)
    nisa.reciprocal(dst=l_inv, data=l_broadcasted_psum)
    out = nl.ndarray((B_D_SIZE, b * q_h_per_k_h), dtype=acc_type)
    nisa.tensor_tensor(
        dst=out,
        data1=o_buffer_sbuf[:, last_large_tile, :],
        data2=l_inv,
        op=nl.multiply,
    )
    o = nl.ndarray((b, h, d), dtype=qkv.dtype, buffer=nl.shared_hbm)
    # k_h == 1: transpose the whole (B_D_SIZE, b*q_h) block at once.
    out_transpose_psum = nl.ndarray(
        (b * q_h_per_k_h, d), dtype=nl.float32, buffer=nl.psum
    )
    transpose_with_matmul(out, out_transpose_psum)
    out_transpose_sbuf = nl.ndarray((b * q_h_per_k_h, d), dtype=o.dtype)
    nisa.tensor_copy(dst=out_transpose_sbuf, src=out_transpose_psum)
    o_reshaped = o.reshape((b * h, d))
    nl.store(o_reshaped[:, :], out_transpose_sbuf)

    update_kv_cache(
        kv_head_id,
        k_transposed_sbuf,
        v_sbuf.reshape((b, d)),
        cache_k,
        cache_v,
        position_ids_sbuf,
    )
    return o, cache_k, cache_v
