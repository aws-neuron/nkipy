"""Sliding-window sparse-attention kernels for DSV4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.ops.attention.sparse_mla import D_BLOCK, K_TILE, P_MAX
from nkipy_serving.runtime.device_tensor import dtype_like as _dtype_like
from nkipy_serving.runtime.device_tensor import sample_like as _sample_like
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

try:
    import neuronxcc.nki as _nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    from neuronxcc.nki.language import par_dim

    _NKI_AVAILABLE = True
except ImportError:
    _nki = None
    nisa = None
    nl = None
    nt = None
    par_dim = None
    _NKI_AVAILABLE = False

_SWA_PAGED_ATTENTION_KERNEL_CACHE: dict[tuple, Any] = {}
_SWA_TOPK_KERNEL_CACHE: dict[tuple, Any] = {}


def _compile_and_load_with_lock(*args: Any, **kwargs: Any) -> Any:
    from nkipy_serving.attention.deepseek_v4 import kernels as dsv4_kernels

    return dsv4_kernels.compile_and_load_with_lock(*args, **kwargs)


if _NKI_AVAILABLE:

    @_nki.jit
    def _sparse_attn_batched_paged_swa_multiK_kernel(
        q_T,
        kv_hbm,
        positions,
        block_tables_per_token,
        sink,
        block_size: int,
        window_size: int,
        max_k: int,
    ):
        """Paged sparse attention that derives SWA global slots in-kernel."""
        B = q_T.shape[0]
        d = q_T.shape[1]
        h = q_T.shape[2]
        max_blocks = block_tables_per_token.shape[1]
        K = K_TILE
        n_k = max_k // K
        n_d = d // D_BLOCK
        effective_window = min(window_size, max_k)

        sink_sb_1h = nl.load(sink)
        sink_on_part_psum = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        sink_on_part_psum[...] = nisa.nc_transpose(
            sink_sb_1h,
            engine=nisa.tensor_engine,
        )
        sink_on_part = nl.ndarray(
            (par_dim(h), 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        sink_on_part[...] = nl.copy(sink_on_part_psum)

        out = nl.ndarray((B, h, d), dtype=nl.float32, buffer=nl.shared_hbm)

        i_p = nl.arange(K)[:, None]
        i_f = nl.arange(d)[None, :]

        for bi in nl.affine_range(B):
            pos = nl.load(positions[bi : bi + 1, :])
            pos_plus_one = nisa.tensor_scalar(
                data=pos,
                op0=nl.add,
                operand0=np.int32(1),
                dtype=np.int32,
            )
            lens = nisa.tensor_scalar(
                data=pos_plus_one,
                op0=nl.minimum,
                operand0=np.int32(effective_window),
                dtype=np.int32,
            )
            start_pos = pos_plus_one - lens

            m_state = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            m_state[...] = nl.full((h, 1), np.float32(-1e30), dtype=nl.float32)
            l_state = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            l_state[...] = nl.full((h, 1), np.float32(0.0), dtype=nl.float32)
            acc = nl.zeros((par_dim(h), d), dtype=nl.float32, buffer=nl.sbuf)

            for kt in nl.sequential_range(n_k):
                topk_sb = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=np.int32,
                    buffer=nl.sbuf,
                )
                global_k = nl.arange(K)[:, None] + kt * K
                lens_b = nl.broadcast_to(lens, shape=(K, 1))
                valid = nl.less(global_k, lens_b)
                logical_pos = nl.add(
                    nl.broadcast_to(start_pos, shape=(K, 1)),
                    global_k,
                )

                # Match the standalone SWA prep kernel's integer division
                # workaround: NKI ``//`` can round near boundaries when
                # lowered through fp32.
                block_offset = nl.mod(logical_pos, nl.int32(block_size))
                block_mul_int = nl.subtract(logical_pos, block_offset)
                block_mul_f = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                block_mul_f[...] = nisa.tensor_scalar(
                    data=block_mul_int,
                    op0=nl.multiply,
                    operand0=np.float32(1.0 / float(block_size)),
                    dtype=np.float32,
                )
                block_idx = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=np.int32,
                    buffer=nl.sbuf,
                )
                block_idx[...] = nl.copy(block_mul_f, dtype=nl.int32)
                safe_block_idx = nisa.tensor_scalar(
                    data=block_idx,
                    op0=nl.minimum,
                    operand0=np.int32(max_blocks - 1),
                    dtype=np.int32,
                )
                block_id = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=np.int32,
                    buffer=nl.sbuf,
                )
                block_id[...] = nl.load(
                    block_tables_per_token[bi : bi + 1, safe_block_idx[i_p, 0]]
                )
                block_base = nisa.tensor_scalar(
                    data=block_id,
                    op0=nl.multiply,
                    operand0=np.int32(block_size),
                    dtype=np.int32,
                )
                slot = nl.add(block_base, block_offset)
                zero_slot = nl.full(
                    (par_dim(K), 1),
                    np.int32(0),
                    dtype=np.int32,
                )
                topk_sb[...] = nl.where(valid, slot, zero_slot, dtype=np.int32)

                one_mask = nl.full(
                    (par_dim(K), 1),
                    1.0,
                    dtype=nl.bfloat16,
                )
                zero_mask = nl.full(
                    (par_dim(K), 1),
                    0.0,
                    dtype=nl.bfloat16,
                )
                mask_col = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                mask_col[...] = nl.where(
                    valid,
                    one_mask,
                    zero_mask,
                    dtype=nl.bfloat16,
                )
                mask_psum = nl.ndarray(
                    (par_dim(1), K),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                mask_psum[...] = nisa.nc_transpose(
                    mask_col,
                    engine=nisa.tensor_engine,
                )
                mask_sb = nl.ndarray(
                    (par_dim(1), K),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                mask_sb[...] = nl.copy(mask_psum, dtype=nl.bfloat16)

                kv_gathered = nl.ndarray(
                    (par_dim(K), d),
                    dtype=kv_hbm.dtype,
                    buffer=nl.sbuf,
                )
                kv_gathered[i_p, i_f] = nl.load(
                    kv_hbm[topk_sb[i_p, 0], i_f],
                )

                qk_psum = nl.zeros(
                    (par_dim(h), K),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for db in nl.affine_range(n_d):
                    q_sb = nl.load(q_T[bi, nl.ds(db * D_BLOCK, D_BLOCK), :])
                    kv_block = kv_gathered[:, nl.ds(db * D_BLOCK, D_BLOCK)]
                    kv_block_T_psum = nl.ndarray(
                        (par_dim(D_BLOCK), K),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    kv_block_T_psum[...] = nisa.nc_transpose(
                        kv_block,
                        engine=nisa.tensor_engine,
                    )
                    k_sb = nl.ndarray(
                        (par_dim(D_BLOCK), K),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    k_sb[...] = nl.copy(kv_block_T_psum, dtype=nl.bfloat16)
                    qk_psum[...] += nisa.nc_matmul(q_sb, k_sb)

                bias_row = nl.ndarray(
                    (par_dim(1), K),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                bias_row[...] = nisa.tensor_scalar(
                    data=mask_sb,
                    op0=nl.subtract,
                    operand0=nl.float32(1.0),
                    op1=nl.multiply,
                    operand1=nl.float32(1e9),
                    dtype=nl.float32,
                )
                bias_bcast = nl.broadcast_to(bias_row, shape=(h, K))
                qk_masked = nl.ndarray(
                    (par_dim(h), K),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                qk_masked[...] = nl.add(qk_psum, bias_bcast)

                m_tile = nisa.tensor_reduce(
                    np.max,
                    qk_masked,
                    axis=(1,),
                    dtype=nl.float32,
                    negate=False,
                )
                m_new = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                m_new[...] = nl.maximum(m_state, m_tile)
                alpha = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                alpha[...] = nisa.activation(
                    np.exp,
                    m_state,
                    bias=nisa.activation(nl.copy, m_new, scale=-1.0),
                    scale=1.0,
                )
                neg_m_new = nisa.activation(nl.copy, m_new, scale=-1.0)
                p_fp32 = nl.ndarray(
                    (par_dim(h), K),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                l_tile = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                p_fp32[...] = nisa.activation_reduce(
                    np.exp,
                    qk_masked,
                    bias=neg_m_new,
                    scale=1.0,
                    reduce_op=nl.add,
                    reduce_res=l_tile,
                    dtype=nl.float32,
                )

                l_prev_scaled = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                l_prev_scaled[...] = nl.multiply(alpha, l_state)
                l_state[...] = nl.add(l_prev_scaled, l_tile)

                alpha_bcast_d = nl.broadcast_to(alpha, shape=(h, d))
                acc_scaled = nl.ndarray(
                    (par_dim(h), d),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                acc_scaled[...] = nl.multiply(acc, alpha_bcast_d)
                acc[...] = nl.copy(acc_scaled)

                p_bf = nl.ndarray(
                    (par_dim(h), K),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                p_bf[...] = nl.copy(p_fp32, dtype=nl.bfloat16)
                p_T_psum = nl.ndarray(
                    (par_dim(K), h),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                p_T_psum[...] = nisa.nc_transpose(
                    p_bf,
                    engine=nisa.tensor_engine,
                )
                p_T = nl.ndarray(
                    (par_dim(K), h),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                p_T[...] = nl.copy(p_T_psum, dtype=nl.bfloat16)

                for db in nl.affine_range(n_d):
                    pv_psum = nl.zeros(
                        (par_dim(h), D_BLOCK),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    pv_psum[...] = nisa.nc_matmul(
                        p_T,
                        kv_gathered[:, nl.ds(db * D_BLOCK, D_BLOCK)],
                    )
                    acc_slice = nl.ndarray(
                        (par_dim(h), D_BLOCK),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    acc_slice[...] = nl.add(
                        acc[:, nl.ds(db * D_BLOCK, D_BLOCK)],
                        pv_psum,
                    )
                    acc[:, nl.ds(db * D_BLOCK, D_BLOCK)] = nl.copy(acc_slice)

                m_state[...] = nl.copy(m_new)

            sink_term = nisa.activation(
                np.exp,
                sink_on_part,
                bias=nisa.activation(nl.copy, m_state, scale=-1.0),
                scale=1.0,
            )
            denom = nl.add(l_state, sink_term)
            inv_denom = nl.ndarray(
                (par_dim(h), 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            inv_denom[...] = nl.divide(nl.float32(1.0), denom)
            inv_bcast_d = nl.broadcast_to(inv_denom, shape=(h, d))
            nl.store(out[bi, :, :], nl.multiply(acc, inv_bcast_d))

        return out


def _sparse_attn_batched_paged_swa_multiK_entry(
    q_T,
    kv_hbm,
    positions,
    block_tables_per_token,
    sink,
    *,
    block_size: int,
    window_size: int,
    max_k: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _sparse_attn_batched_paged_swa_multiK_kernel(
        q_T,
        kv_hbm,
        positions,
        block_tables_per_token,
        sink,
        int(block_size),
        int(window_size),
        int(max_k),
    )


def run_sparse_attention_paged_swa_device(
    *,
    q_scaled_t: Any,
    kv_hbm: Any,
    positions: Any,
    block_tables_per_token: Any,
    sink: Any,
    output: Any,
    block_size: int,
    window_size: int,
    max_k: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Run SWA sparse attention while deriving global slots in the kernel."""
    q_shape = tuple(int(dim) for dim in getattr(q_scaled_t, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_hbm, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    bt_shape = tuple(int(dim) for dim in getattr(block_tables_per_token, "shape"))
    sink_shape = tuple(int(dim) for dim in getattr(sink, "shape"))
    out_shape = tuple(int(dim) for dim in getattr(output, "shape"))

    if len(q_shape) != 3:
        raise ValueError(f"q_scaled_t must be [tokens, head_dim, heads], got {q_shape}")
    if len(kv_shape) != 2:
        raise ValueError(f"kv_hbm must be [num_slots, head_dim], got {kv_shape}")
    if len(pos_shape) != 2 or pos_shape[1] != 1:
        raise ValueError(f"positions must be [tokens, 1], got {pos_shape}")
    if len(bt_shape) != 2:
        raise ValueError(
            f"block_tables_per_token must be [tokens, max_blocks], got {bt_shape}"
        )
    if len(sink_shape) != 2:
        raise ValueError(f"sink must be [1, heads], got {sink_shape}")

    tokens, head_dim, num_heads = q_shape
    max_k = int(max_k)
    block_size = int(block_size)
    window_size = int(window_size)
    if pos_shape[0] != tokens:
        raise ValueError(
            f"positions tokens={pos_shape[0]} must match q tokens={tokens}"
        )
    if bt_shape[0] != tokens:
        raise ValueError(
            f"block_tables_per_token tokens={bt_shape[0]} must match q tokens={tokens}"
        )
    if sink_shape != (1, num_heads):
        raise ValueError(f"sink must be [1, {num_heads}], got {sink_shape}")
    if kv_shape[1] != head_dim:
        raise ValueError(f"kv head_dim={kv_shape[1]} must match q head_dim={head_dim}")
    if out_shape != (tokens, num_heads, head_dim):
        raise ValueError(
            f"output must be [{tokens}, {num_heads}, {head_dim}], got {out_shape}"
        )
    if max_k <= 0 or max_k % K_TILE:
        raise NotImplementedError(
            f"max_k={max_k} must be a positive multiple of K_TILE={K_TILE}"
        )
    if block_size <= 0 or window_size <= 0:
        raise ValueError("block_size and window_size must be positive")
    if head_dim % D_BLOCK:
        raise NotImplementedError(f"head_dim={head_dim} not a multiple of {D_BLOCK}")
    if num_heads > P_MAX:
        raise ValueError(f"num_heads={num_heads} must be <= {P_MAX}")

    cache = (
        _SWA_PAGED_ATTENTION_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    cache_key = (
        "sparse_attention_paged_swa_device",
        q_shape,
        kv_shape,
        pos_shape,
        bt_shape,
        sink_shape,
        out_shape,
        str(_dtype_like(q_scaled_t)),
        str(_dtype_like(kv_hbm)),
        str(_dtype_like(positions)),
        str(_dtype_like(block_tables_per_token)),
        str(_dtype_like(sink)),
        block_size,
        window_size,
        max_k,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            _sparse_attn_batched_paged_swa_multiK_entry,
            _sample_like(q_scaled_t),
            _sample_like(kv_hbm),
            _sample_like(positions),
            _sample_like(block_tables_per_token),
            _sample_like(sink),
            block_size=block_size,
            window_size=window_size,
            max_k=max_k,
            name=(
                "dsv4_sparse_attention_paged_swa_"
                f"t{tokens}_d{head_dim}_k{max_k}_bs{block_size}_w{window_size}"
            ),
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "q_T": q_scaled_t,
            "kv_hbm": kv_hbm,
            "positions": positions,
            "block_tables_per_token": block_tables_per_token,
            "sink": sink,
        },
        outputs={"output0": output},
    )
    return output


# ---------------------------------------------------------------------------
# Unified SWA top-k (works for prefill + decode)
# ---------------------------------------------------------------------------


def swa_global_slots_oracle(
    *,
    positions: np.ndarray,  # [total_tokens] int, absolute token pos
    req_id_per_token: np.ndarray,  # [total_tokens] int, request index
    block_tables: np.ndarray,  # [batch, max_blocks] int
    block_size: int,
    window_size: int,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window top-k as global KV-cache slots for any q_len.

    For each query token at absolute ``pos``, selects the ``min(pos+1,
    window_size, max_k)`` most recent KV positions ``[start_pos .. pos]``
    and resolves each through the request's block table to a flat cache
    slot.

    Returns K-major ``topk_t [max_k, total_tokens]`` int32 (zero-clamped
    where invalid), numeric ``mask [total_tokens, max_k]`` bf16, and
    ``lens [total_tokens]`` int32.
    """
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    req_id_per_token = np.asarray(req_id_per_token, dtype=np.int64).reshape(-1)
    block_tables = np.asarray(block_tables, dtype=np.int64)
    total_tokens = int(positions.shape[0])
    if req_id_per_token.shape != (total_tokens,):
        raise ValueError(
            "req_id_per_token must match positions shape "
            f"{positions.shape}, got {req_id_per_token.shape}"
        )
    if block_tables.ndim != 2:
        raise ValueError(
            f"block_tables must be [batch, max_blocks], got {block_tables.shape}"
        )
    block_size = int(block_size)
    window_size = int(window_size)
    max_k = int(max_k)
    if block_size <= 0 or window_size <= 0 or max_k <= 0:
        raise ValueError("block_size, window_size, and max_k must be positive")

    # Invalid slots sentinel as -1 so ``sparse_attention_oracle`` (which
    # consumes ``topk >= 0``) correctly masks them out. Device-side callers
    # still need to safe-clamp for indirect HBM load; they do that via the
    # kernel's mask path.
    topk_t = np.full((max_k, total_tokens), -1, dtype=np.int32)
    mask = np.zeros((total_tokens, max_k), dtype=ml_dtypes.bfloat16)
    lens = np.zeros((total_tokens,), dtype=np.int32)

    for q in range(total_tokens):
        pos = int(positions[q])
        req = int(req_id_per_token[q])
        cur_len = min(pos + 1, window_size, max_k)
        lens[q] = cur_len
        start_pos = pos - cur_len + 1
        for k_idx in range(cur_len):
            logical_pos = start_pos + k_idx
            block_idx = logical_pos // block_size
            block_offset = logical_pos % block_size
            if block_idx >= block_tables.shape[1]:
                raise ValueError(
                    f"token {q} req={req} pos={pos} needs block {block_idx}, "
                    f"but block_tables has {block_tables.shape[1]} columns"
                )
            block_id = int(block_tables[req, block_idx])
            topk_t[k_idx, q] = block_id * block_size + block_offset
            mask[q, k_idx] = ml_dtypes.bfloat16(1.0)

    return topk_t, mask, lens


if _NKI_AVAILABLE:

    @_nki.jit
    def _swa_global_slots_kernel(
        positions: "nt.tensor",
        block_tables_per_token: "nt.tensor",
        topk_t: "nt.tensor[nt.mutable]",
        topk_lens: "nt.tensor[nt.mutable]",
        topk_mask: "nt.tensor[nt.mutable]",
        block_size: int,
        window_size: int,
    ):
        """Unified SWA top-k kernel: works for any q_len per request.

        Caller supplies ``block_tables_per_token`` — a host-expanded
        per-token view of the per-request block table. This lets the
        kernel use the Python-int ``tok`` (from ``affine_range``) as a
        static slice start, avoiding 2-D indirect SBUF indexing which
        NKI does not support.
        """
        total_tokens = positions.shape[0]
        k_total = topk_t.shape[0]
        max_blocks = block_tables_per_token.shape[1]
        effective_window = min(window_size, k_total)

        for tok in nl.affine_range(total_tokens):
            pos = nl.load(positions[tok : tok + 1, :])

            pos_plus_one = nisa.tensor_scalar(
                data=pos,
                op0=nl.add,
                operand0=np.int32(1),
                dtype=np.int32,
            )
            lens = nisa.tensor_scalar(
                data=pos_plus_one,
                op0=nl.minimum,
                operand0=np.int32(effective_window),
                dtype=np.int32,
            )
            start_pos = pos_plus_one - lens
            nl.store(topk_lens[tok : tok + 1, :], lens)

            for k_idx in nl.affine_range(k_total):
                valid = nisa.tensor_scalar(
                    data=lens,
                    op0=nl.greater,
                    operand0=k_idx,
                    dtype=np.uint8,
                )
                logical_pos = nisa.tensor_scalar(
                    data=start_pos,
                    op0=nl.add,
                    operand0=k_idx,
                    dtype=np.int32,
                )
                # ``logical_pos // block_size`` via NKI ``//`` lowers to
                # fp32 division, which can round ``7 // 8`` to ``1``. Use
                # the subtract-mod + fp32-reciprocal workaround instead so
                # the result matches Python integer truncation exactly.
                block_offset = nl.mod(logical_pos, nl.int32(block_size))
                block_mul_int = nl.subtract(logical_pos, block_offset)
                block_mul_f = nl.ndarray(
                    block_mul_int.shape,
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                block_mul_f[...] = nisa.tensor_scalar(
                    data=block_mul_int,
                    op0=nl.multiply,
                    operand0=np.float32(1.0 / float(block_size)),
                    dtype=np.float32,
                )
                block_idx = nl.ndarray(
                    block_mul_int.shape,
                    dtype=np.int32,
                    buffer=nl.sbuf,
                )
                block_idx[...] = nl.copy(block_mul_f, dtype=nl.int32)
                # Defensive: clamp to ``[0, max_blocks-1]`` so the indirect
                # HBM load always lands in range even if ``valid=False``
                # entries compute a garbage ``logical_pos``.
                safe_block_idx = nisa.tensor_scalar(
                    data=block_idx,
                    op0=nl.minimum,
                    operand0=np.int32(max_blocks - 1),
                    dtype=np.int32,
                )
                block_id = nl.load(
                    block_tables_per_token[tok : tok + 1, safe_block_idx[0, 0]]
                )
                block_base = nisa.tensor_scalar(
                    data=block_id,
                    op0=nl.multiply,
                    operand0=np.int32(block_size),
                    dtype=np.int32,
                )
                slot = block_base + block_offset
                zero_slot = nl.full(slot.shape, np.int32(0), dtype=np.int32)
                safe_slot = nl.where(valid, slot, zero_slot, dtype=np.int32)
                one_mask = nl.full(valid.shape, 1.0, dtype=nl.bfloat16)
                zero_mask = nl.full(valid.shape, 0.0, dtype=nl.bfloat16)
                mask_val = nl.where(
                    valid,
                    one_mask,
                    zero_mask,
                    dtype=nl.bfloat16,
                )

                nl.store(
                    topk_t[k_idx : k_idx + 1, tok : tok + 1],
                    safe_slot,
                )
                nl.store(
                    topk_mask[tok : tok + 1, k_idx : k_idx + 1],
                    mask_val,
                )
        return topk_t, topk_lens, topk_mask


def _swa_global_slots_entry(
    positions,
    block_tables_per_token,
    topk_t,
    topk_lens,
    topk_mask,
    *,
    block_size: int,
    window_size: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _swa_global_slots_kernel(
        positions,
        block_tables_per_token,
        topk_t,
        topk_lens,
        topk_mask,
        int(block_size),
        int(window_size),
    )


def run_swa_global_slots_device(
    *,
    positions: Any,
    block_tables_per_token: Any,
    topk_t: Any,
    topk_lens: Any,
    topk_mask: Any,
    block_size: int,
    window_size: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Unified SWA top-k on device (prefill + decode).

    ``block_tables_per_token`` is the host-expanded per-token view of the
    block table (shape ``[total_tokens, max_blocks]``). NKI can't do
    2-D indirect SBUF indexing, so the per-request → per-token rebase
    must happen host-side before the upload.
    """
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    bt_shape = tuple(int(dim) for dim in getattr(block_tables_per_token, "shape"))
    topk_shape = tuple(int(dim) for dim in getattr(topk_t, "shape"))
    lens_shape = tuple(int(dim) for dim in getattr(topk_lens, "shape"))
    mask_shape = tuple(int(dim) for dim in getattr(topk_mask, "shape"))

    if len(pos_shape) != 2 or pos_shape[1] != 1:
        raise ValueError(f"positions must be [total_tokens, 1], got {pos_shape}")
    total_tokens = pos_shape[0]
    if len(bt_shape) != 2 or bt_shape[0] != total_tokens:
        raise ValueError(
            f"block_tables_per_token must be [{total_tokens}, max_blocks], "
            f"got {bt_shape}"
        )
    if len(topk_shape) != 2 or topk_shape[1] != total_tokens:
        raise ValueError(f"topk_t must be [K, {total_tokens}], got {topk_shape}")
    k_total = topk_shape[0]
    if lens_shape != (total_tokens, 1):
        raise ValueError(f"topk_lens must be [{total_tokens}, 1], got {lens_shape}")
    if mask_shape != (total_tokens, k_total):
        raise ValueError(
            f"topk_mask must be [{total_tokens}, {k_total}], got {mask_shape}"
        )
    block_size = int(block_size)
    window_size = int(window_size)
    if block_size <= 0 or window_size <= 0:
        raise ValueError("block_size and window_size must be positive")

    cache = _SWA_TOPK_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    cache_key = (
        "swa_global_slots",
        pos_shape,
        bt_shape,
        topk_shape,
        lens_shape,
        mask_shape,
        str(_dtype_like(positions)),
        str(_dtype_like(block_tables_per_token)),
        str(_dtype_like(topk_t)),
        str(_dtype_like(topk_lens)),
        str(_dtype_like(topk_mask)),
        block_size,
        window_size,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            _swa_global_slots_entry,
            _sample_like(positions),
            _sample_like(block_tables_per_token),
            _sample_like(topk_t),
            _sample_like(topk_lens),
            _sample_like(topk_mask),
            block_size=block_size,
            window_size=window_size,
            name=(
                "dsv4_swa_global_slots_"
                f"t{total_tokens}_k{k_total}_bs{block_size}_w{window_size}"
            ),
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "positions": positions,
            "block_tables_per_token": block_tables_per_token,
            "topk_t": topk_t,
            "topk_lens": topk_lens,
            "topk_mask": topk_mask,
        },
        outputs={
            "topk_t": topk_t,
            "topk_lens": topk_lens,
            "topk_mask": topk_mask,
        },
    )
    return topk_t, topk_lens, topk_mask
