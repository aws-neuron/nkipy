"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

"""

import numpy as np
import neuronxcc.nki.typing as nt
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa

from .flash_attn_core import (
    _flash_attention_core_kq_matmul,
    _active_attention_core_batched,
    partition_broadcast_fp32,
    transpose_with_matmul,
)
from .attn_utils import B_P_SIZE


def create_identity_for_transpose(dtype, *sizes):
    ret = []
    for size in sizes:
        assert size > 0
        if size == 1:
            identity = nl.ones((1, 1), dtype=dtype)
        else:
            identity_hbm = nl.shared_constant(
                np.identity(n=size, dtype=np.uint8),
                dtype=dtype,
            )
            identity = nl.load(identity_hbm)
        ret.append(identity)
    return tuple(ret)


def apply_rope(x, sin, cos):
    assert len(x.shape) == 3 and len(sin.shape) == 3 and sin.shape == cos.shape
    half_d = sin.shape[-1]
    assert x.shape[-1] == half_d * 2 and sin.shape[1] == 1
    x0 = x[:, :, nl.ds(0, half_d)]
    x1 = x[:, :, nl.ds(half_d, half_d)]
    out = nl.ndarray(x.shape, dtype=x.dtype)
    out[:, :, nl.ds(0, half_d)] = x0 * cos - x1 * sin
    out[:, :, nl.ds(half_d, half_d)] = x0 * sin + x1 * cos
    return out


def load_qkv_and_apply_rope(qkv, sin, cos, position_ids, h, k_h, kv_head_id):
    batch_size, _, d = qkv.shape
    qkv_sbuf = nl.load(qkv)
    q_h_per_k_h = h // k_h
    q_sbuf = nl.copy(qkv_sbuf[:, nl.ds(kv_head_id * q_h_per_k_h, q_h_per_k_h), :])
    k_sbuf = nl.copy(qkv_sbuf[:, nl.ds(h + kv_head_id, 1), :])
    v_sbuf = nl.copy(qkv_sbuf[:, nl.ds(h + k_h + kv_head_id, 1), :])
    position_ids_sbuf = nl.load(position_ids.reshape((batch_size, 1)))
    i_p = nl.arange(batch_size)[:, None]
    i_f = nl.arange(d // 2)[None, :]
    sin_sbuf = nl.ndarray((batch_size, 1, d // 2), dtype=sin.dtype)
    cos_sbuf = nl.ndarray((batch_size, 1, d // 2), dtype=cos.dtype)
    sin_sbuf[i_p, 0, i_f] = nl.load(sin[position_ids_sbuf[i_p, 0], i_f])
    cos_sbuf[i_p, 0, i_f] = nl.load(cos[position_ids_sbuf[i_p, 0], i_f])
    q_rope = apply_rope(q_sbuf, sin_sbuf, cos_sbuf)
    k_rope = apply_rope(k_sbuf, sin_sbuf, cos_sbuf)
    return q_rope, k_rope, v_sbuf


def update_kv_cache(kv_head_id, k, v, cache_k, cache_v, position_ids):
    batch_size, num_heads, head_size, max_model_len = cache_k.shape
    assert position_ids.shape == (1, batch_size)
    assert cache_v.shape == (batch_size, max_model_len, num_heads, head_size)
    assert k.shape == (head_size, batch_size)
    assert v.shape == (batch_size, head_size)

    batch_starts = nisa.iota(
        nl.arange(batch_size)[None, :] * max_model_len, dtype=nl.int32
    )
    write_offsets = nisa.tensor_tensor(batch_starts, position_ids, nl.add)

    cache_v_reshaped = cache_v.reshape(
        (batch_size * max_model_len, num_heads, head_size)
    )
    write_offsets_fp32 = write_offsets.view(nl.float32)
    indices_sbuf = nl.ndarray((batch_size, 1), dtype=nl.int32)
    indices_sbuf_reinterpreted = indices_sbuf.view(nl.float32)
    indices_psum = nisa.nc_transpose(write_offsets_fp32)
    indices_sbuf_reinterpreted[...] = nl.copy(indices_psum)
    i_p = nl.arange(batch_size)[:, None]
    i_f = nl.arange(head_size)[None, :]
    nl.store(
        dst=cache_v_reshaped[indices_sbuf[i_p, 0], kv_head_id, i_f],
        value=v[i_p, i_f],
    )

    """
    #=========================================================================
    # Scalar DGE
    # 
    # Cleaner but slightly slower
    #=========================================================================
    for b in nl.affine_range(batch_size):
        # Use scalar DGE.
        nl.store(
            cache_k[b][kv_head_id][
                nl.arange(head_size)[:, None],
                nl.arange(1)[None, :] + position_ids[0, b],
            ],
            k[
                nl.arange(head_size)[:, None],
                nl.arange(1)[None, :] + b,
            ],
        )
    """

    # =========================================================================
    # Vector DGE (i.e. manual vectorize)
    #
    # Better looking trace, but no obvious e2e throughput improvement
    # =========================================================================
    if isinstance(kv_head_id, int):
        assert kv_head_id == 0
        head_offsets = position_ids
    else:
        head_offsets = nisa.tensor_tensor(
            nisa.iota(kv_head_id, dtype=nl.int32) * (head_size * max_model_len),
            position_ids,
            nl.add,
        )
    batch_offsets = nisa.tensor_tensor(
        nisa.iota(
            nl.arange(batch_size)[None, :],
            dtype=nl.int32,
        )
        * (num_heads * head_size * max_model_len),
        head_offsets,
        nl.add,
    )
    batch_offsets_br_psum = nl.ndarray(
        (head_size, batch_size), dtype=nl.float32, buffer=nl.psum
    )
    partition_broadcast_fp32(batch_offsets.view(nl.float32), batch_offsets_br_psum)
    batch_offsets_br_sbuf = nl.ndarray((head_size, batch_size), dtype=nl.int32)
    batch_offsets_br_sbuf.view(nl.float32)[...] = nl.copy(batch_offsets_br_psum)
    feat_offsets = nisa.iota(
        nl.arange(head_size)[:, None] * max_model_len,
        dtype=nl.int32,
    )
    offsets = nisa.tensor_tensor(batch_offsets_br_sbuf, feat_offsets, nl.add)
    cache_k = cache_k.reshape((batch_size * num_heads * head_size * max_model_len, 1))
    i_p = nl.arange(head_size)[:, None]
    i_f = nl.arange(1)[None, :]
    k_reshaped = k.reshape((head_size, batch_size, 1))
    for b_i in nl.affine_range(batch_size):
        nl.store(cache_k[offsets[i_p, b_i], i_f], k_reshaped[i_p, b_i, i_f])


@nki.compiler.skip_middle_end_transformations
@nki.jit(debug_kernel=True, experimental_flags="enable-mutable-parameter")
def flash_attn_decode(
    qkv,
    cos,
    sin,
    cache_k: nt.tensor[nt.mutable],
    cache_v: nt.tensor[nt.mutable],
    sink,
    position_ids,
    sliding_window,
    tile_masks=None,
    softmax_scale=None,
    mixed_precision=True,
    LARGE_KV_TILE_SIZE=None,
):
    """
    Flash Attention Forward kernel

    IO tensor layouts:
      - q: shape (bs, h, d)
      - k: shape (bs, k_h, d)
      - v: shape (bs, k_h, d)
      - cache_k: shape (bs, k_h, d, max_model_len)
      - cache_v: shape (bs, max_model_len, k_h, d)
      - sink: shape (n_heads,)
      - position_ids: shape (bs,)
      - tile_mask: shape (max_model_len, max_model_len), optional

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
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype
    spmd_ndim = nl.program_ndim()
    assert spmd_ndim <= 1, f"{nl.program_ndim()=} > 1"
    if spmd_ndim == 0:
        assert k_h == 1
        kv_head_id = 0
    else:
        assert nl.num_programs(axes=0) == k_h
        kv_head_id = nl.program_id(axis=0)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    if LARGE_KV_TILE_SIZE is None:
        LARGE_KV_TILE_SIZE = min(2048, max_model_len)
    assert LARGE_KV_TILE_SIZE % B_P_SIZE == 0

    assert (
        max_model_len % LARGE_KV_TILE_SIZE == 0
    ), f"{max_model_len=} to be divisible by {LARGE_KV_TILE_SIZE=}"

    num_large_k_tile = max_model_len // LARGE_KV_TILE_SIZE

    identity_m1, identity_m2, identity_o = create_identity_for_transpose(
        acc_type, B_P_SIZE, b * q_h_per_k_h, B_D_SIZE
    )
    identity_qkv = create_identity_for_transpose(kernel_dtype, b)[0]

    # =============== Global Flash Attention accumulators ====================== #
    o_buffer_sbuf = nl.ndarray(
        (par_dim(B_D_SIZE), num_large_k_tile + 1, b * q_h_per_k_h),
        dtype=acc_type,
    )
    l_buffer_sbuf = nl.ndarray(
        (par_dim(1), num_large_k_tile + 1, b * q_h_per_k_h),
        dtype=acc_type,
    )
    m_buffer_sbuf = nl.ndarray(
        (par_dim(1), num_large_k_tile + 1, b * q_h_per_k_h),
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

    def transpose_qkv_with_pe(src, out_sbuf, identity, scale=None):
        assert src.dtype == out_sbuf.dtype == identity.dtype
        if src.dtype == nl.bfloat16 and src.shape[0] == 1:
            out_psum = nl.ndarray(out_sbuf.shape, dtype=nl.float32, buffer=nl.psum)
            out_psum[...] = nisa.nc_matmul(
                src,
                identity,
                is_moving_onezero=True,
            )
        else:
            out_psum = nl.ndarray(out_sbuf.shape, dtype=src.dtype, buffer=nl.psum)
            out_psum[...] = nisa.nc_matmul(
                src,
                identity,
                is_moving_onezero=True,
                is_transpose=True,
            )
        if scale is not None and scale != 1:
            out_sbuf[...] = nisa.activation(
                nl.copy,
                out_psum,
                scale=scale,
                dtype=out_sbuf.dtype,
            )
        else:
            out_sbuf[...] = nl.copy(out_psum, dtype=out_sbuf.dtype)

    q_sbuf = nl.ndarray((d, b, q_h_per_k_h), dtype=kernel_dtype)
    for q_h in nl.affine_range(q_h_per_k_h):
        transpose_qkv_with_pe(
            q_rope[:, q_h, :],
            q_sbuf[:, :, q_h],
            identity=identity_qkv,
            scale=softmax_scale,
        )
    k_transposed_sbuf = nl.ndarray((d, b), dtype=kernel_dtype)
    v_transposed_sbuf = nl.ndarray((d, b), dtype=kernel_dtype)
    transpose_qkv_with_pe(k_rope[:, 0, :], k_transposed_sbuf, identity_qkv)
    transpose_qkv_with_pe(v_sbuf[:, 0, :], v_transposed_sbuf, identity_qkv)
    _active_attention_core_batched(
        q=q_sbuf,
        k=k_transposed_sbuf,
        v=v_transposed_sbuf,
        o_buffer_sbuf=o_buffer_sbuf,
        l_buffer_sbuf=l_buffer_sbuf,
        m_buffer_sbuf=m_buffer_sbuf,
        sink=sink,
        kernel_dtype=kernel_dtype,
        acc_type=acc_type,
    )
    position_ids_sbuf = nl.load(position_ids.reshape((1, b)), dtype=nl.int32)

    if tile_masks is None:
        k_stride = LARGE_KV_TILE_SIZE // B_P_SIZE
        token_pos_f = nl.ndarray((B_P_SIZE, k_stride), dtype=nl.float32, buffer=nl.psum)
        partition_broadcast_fp32(
            # XXX: if k_stride can be very large (>128), use nl.float32
            nisa.iota(nl.arange(k_stride)[None, :], dtype=nl.bfloat16),
            token_pos_f,
        )
        token_pos_p = nisa.iota(
            nl.arange(B_P_SIZE)[:, None] * k_stride,
            dtype=np.float32,
        )
        token_pos = nl.ndarray((B_P_SIZE, 1, k_stride), dtype=nl.int32)
        token_pos[:, 0, :] = nisa.tensor_scalar(
            token_pos_f,
            nl.add,
            token_pos_p,
            dtype=nl.int32,
        )
        pos_upper_bound = nl.broadcast_to(
            position_ids_sbuf.reshape((1, b, 1)),
            shape=(B_P_SIZE, b, 1),
        )
        if sliding_window > 0:
            pos_lower_bound = pos_upper_bound - sliding_window
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
        (par_dim(B_D_SIZE), MULTI_BUFFER, b, LARGE_KV_TILE_SIZE),
        dtype=kernel_dtype,
    )
    v_load_buffer = nl.ndarray(
        (par_dim(B_P_SIZE), MULTI_BUFFER, b, LARGE_KV_TILE_SIZE // B_P_SIZE, B_D_SIZE),
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
    for kv_tile_id in nl.sequential_range(num_large_k_tile):
        if tile_masks is None:
            mask = nisa.tensor_tensor(
                token_pos,
                pos_upper_bound,
                nl.less,
                dtype=nl.uint8,
            )
            if sliding_window > 0:
                mask_lower = nisa.tensor_tensor(
                    token_pos,
                    pos_lower_bound,
                    nl.greater,  # XXX: greater but not equal due to decomposed attn
                    dtype=nl.uint8,
                )
                mask = nisa.tensor_tensor(
                    mask,
                    mask_lower,
                    nl.logical_and,
                    dtype=nl.uint8,
                )
        else:
            mask = tile_masks_sbuf[:, kv_tile_id]
        if tile_masks is None and kv_tile_id < num_large_k_tile:
            token_pos[...] = token_pos + LARGE_KV_TILE_SIZE
        _flash_attention_core_kq_matmul(
            q_local_tile=q_sbuf,
            cache_k=cache_k_reshaped,
            cache_v=cache_v_reshaped,
            k_load_buffer=k_load_buffer,
            v_load_buffer=v_load_buffer,
            tile_mask=mask,
            o_buffer_sbuf=o_buffer_sbuf,
            l_buffer_sbuf=l_buffer_sbuf,
            m_buffer_sbuf=m_buffer_sbuf,
            large_kv_tile_id=kv_tile_id,
            num_large_kv_tiles=num_large_k_tile,
            kv_head_id=kv_head_id,
            kernel_dtype=kernel_dtype,
            acc_type=acc_type,
            identity_m1=identity_m1,
            identity_m2=identity_m2,
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
    out = nl.multiply(o_buffer_sbuf[:, last_large_tile, :], 1.0 / l_broadcasted_psum)
    o = nl.ndarray((b, h, d), dtype=qkv.dtype, buffer=nl.shared_hbm)
    if k_h == 1:
        out_transpose_psum = nl.ndarray(
            (b * q_h_per_k_h, d), dtype=nl.float32, buffer=nl.psum
        )
        transpose_with_matmul(
            out,
            out_transpose_psum,
            identity=identity_o,
        )
        out_transpose_sbuf = nl.copy(out_transpose_psum, dtype=o.dtype)
        o_reshaped = o.reshape((b * h, d))
        nl.store(o_reshaped[:, :], out_transpose_sbuf)
    else:
        for batch_id in nl.affine_range(b):
            out_transpose_psum = nl.ndarray(
                (q_h_per_k_h, d), dtype=nl.float32, buffer=nl.psum
            )
            transpose_with_matmul(
                out[:, nl.ds(batch_id * q_h_per_k_h, q_h_per_k_h)],
                out_transpose_psum,
                identity=identity_o,
            )
            out_transpose_sbuf = nl.copy(out_transpose_psum, dtype=o.dtype)
            nl.store(
                o[
                    batch_id,
                    nl.ds(kv_head_id * q_h_per_k_h, q_h_per_k_h),
                    :,
                ],
                out_transpose_sbuf,
            )
    update_kv_cache(
        kv_head_id=kv_head_id,
        k=k_transposed_sbuf,
        v=v_sbuf.reshape((b, d)),
        cache_k=cache_k,
        cache_v=cache_v,
        position_ids=position_ids_sbuf,
    )
    return o, cache_k, cache_v
