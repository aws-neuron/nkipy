"""Two-source sparse-attention kernels for DSV4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

_TWO_SOURCE_PAGED_ATTENTION_KERNEL_CACHE: dict[tuple, Any] = {}
_TWO_SOURCE_KERNEL_VERSION = 4


def _compile_and_load_with_lock(*args: Any, **kwargs: Any) -> Any:
    from nkipy_serving.attention.deepseek_v4 import kernels as dsv4_kernels

    return dsv4_kernels.compile_and_load_with_lock(*args, **kwargs)


def gather_two_source_kv_and_mask(
    *,
    kv_primary: np.ndarray,  # [B * primary_len, d]
    kv_secondary: np.ndarray,  # [B * secondary_stride, d]
    topk_idxs: np.ndarray,  # [N_q, K] local, -1 = invalid
    owner_ids: np.ndarray,  # [N_q]
    primary_owner_ids: np.ndarray | None = None,  # [N_q]
    primary_len: int,
    secondary_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU gather for local DSV4 two-source sparse-attention indices."""

    primary = np.asarray(kv_primary)
    secondary = np.asarray(kv_secondary)
    topk = np.asarray(topk_idxs, dtype=np.int64)
    owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
    primary_owners = (
        owners
        if primary_owner_ids is None
        else np.asarray(primary_owner_ids, dtype=np.int64).reshape(-1)
    )
    primary_len = int(primary_len)
    secondary_stride = int(secondary_stride)
    if primary.ndim != 2 or secondary.ndim != 2:
        raise ValueError(
            "kv_primary and kv_secondary must be 2-D, got "
            f"{primary.shape}/{secondary.shape}"
        )
    if primary.shape[1] != secondary.shape[1]:
        raise ValueError(
            f"head_dim mismatch: {primary.shape[1]} vs {secondary.shape[1]}"
        )
    if topk.ndim != 2:
        raise ValueError(f"topk_idxs must be [N_q, K], got {topk.shape}")
    if owners.shape != (topk.shape[0],):
        raise ValueError(f"owner_ids must be [{topk.shape[0]}], got {owners.shape}")
    if primary_owners.shape != (topk.shape[0],):
        raise ValueError(
            f"primary_owner_ids must be [{topk.shape[0]}], got {primary_owners.shape}"
        )
    if primary_len <= 0 or secondary_stride <= 0:
        raise ValueError("primary_len and secondary_stride must be positive")

    n_q, k_max = topk.shape
    d = primary.shape[1]
    gathered = np.zeros((n_q, k_max, d), dtype=primary.dtype)
    valid = topk >= 0
    for q in range(n_q):
        owner = int(owners[q])
        primary_owner = int(primary_owners[q])
        if owner < 0 or primary_owner < 0:
            raise ValueError(
                "owner ids must be non-negative, got "
                f"primary={primary_owner}, secondary={owner}"
            )
        primary_base = primary_owner * primary_len
        secondary_base = owner * secondary_stride
        for k in range(k_max):
            idx = int(topk[q, k])
            if idx < 0:
                continue
            if idx < primary_len:
                row = primary_base + idx
                if row >= primary.shape[0]:
                    raise ValueError(
                        f"primary index out of range: owner={owner}, idx={idx}"
                    )
                gathered[q, k] = primary[row]
            else:
                secondary_idx = idx - primary_len
                if secondary_idx >= secondary_stride:
                    raise ValueError(
                        "secondary index outside owner stride: "
                        f"owner={owner}, idx={idx}, stride={secondary_stride}"
                    )
                row = secondary_base + secondary_idx
                if row >= secondary.shape[0]:
                    raise ValueError(
                        f"secondary index out of range: owner={owner}, idx={idx}"
                    )
                gathered[q, k] = secondary[row]
    return gathered, valid


if _NKI_AVAILABLE:

    @_nki.jit
    def _sparse_attn_batched_paged_two_source_multiK_kernel(
        q_T,
        kv_primary,
        kv_secondary,
        topk_T,
        mask,
        owner_ids,
        primary_owner_ids,
        sink,
        primary_len: int,
        secondary_stride: int,
        primary_prefix_len: int,
    ):
        """Batched sparse attention with local indices over two KV sources.

        ``topk_T`` entries are request-local. Indices below ``primary_len``
        gather from
        ``kv_primary[primary_owner * primary_len + idx]``; larger indices
        gather from ``kv_secondary[owner * secondary_stride + idx-primary_len]``.
        When ``primary_prefix_len`` is positive, top-k is assumed to be
        ordered as primary-window entries first followed by secondary
        compressed entries. Full primary/secondary K tiles avoid the redundant
        gather from the unused source while still clamping local offsets so
        masked/padded top-k entries cannot produce out-of-range HBM loads.
        """
        B = q_T.shape[0]
        d = q_T.shape[1]
        h = q_T.shape[2]
        K_total = topk_T.shape[0]
        K = K_TILE
        n_k = K_total // K
        n_d = d // D_BLOCK

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

        owner_2d = owner_ids.reshape((B, 1))
        primary_owner_2d = primary_owner_ids.reshape((B, 1))
        i_p = nl.arange(K)[:, None]
        i_f = nl.arange(d)[None, :]

        for bi in nl.affine_range(B):
            owner_sb = nl.load(owner_2d[bi : bi + 1, 0:1])
            primary_owner_sb = nl.load(primary_owner_2d[bi : bi + 1, 0:1])
            primary_base = nl.multiply(primary_owner_sb, nl.int32(primary_len))
            secondary_base = nl.multiply(owner_sb, nl.int32(secondary_stride))

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

            for kt in nl.static_range(n_k):
                topk_sb = nl.ndarray(
                    (par_dim(K), 1),
                    dtype=topk_T.dtype,
                    buffer=nl.sbuf,
                )
                topk_sb[...] = nl.load(
                    topk_T[nl.ds(kt * K, K), bi : bi + 1],
                )

                kv_gathered = nl.ndarray(
                    (par_dim(K), d),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                tile_start = int(kt) * K
                tile_end = int(tile_start) + K
                if int(primary_prefix_len) > 0 and int(tile_end) <= int(
                    primary_prefix_len
                ):
                    primary_nonneg = nl.ndarray(
                        (par_dim(K), 1),
                        dtype=nl.int32,
                        buffer=nl.sbuf,
                    )
                    primary_nonneg[...] = nl.maximum(topk_sb, nl.int32(0))
                    primary_local = nisa.tensor_scalar(
                        data=primary_nonneg,
                        op0=nl.minimum,
                        operand0=np.int32(primary_len - 1),
                        dtype=np.int32,
                    )
                    primary_rows = nl.add(
                        nl.broadcast_to(primary_base, shape=(K, 1)),
                        primary_local,
                    )
                    kv_gathered[...] = nl.load(
                        kv_primary[primary_rows[i_p, 0], i_f],
                    )
                elif int(primary_prefix_len) > 0 and int(tile_start) >= int(
                    primary_prefix_len
                ):
                    secondary_raw = nl.subtract(topk_sb, nl.int32(primary_len))
                    secondary_nonneg = nl.ndarray(
                        (par_dim(K), 1),
                        dtype=nl.int32,
                        buffer=nl.sbuf,
                    )
                    secondary_nonneg[...] = nl.maximum(
                        secondary_raw,
                        nl.int32(0),
                    )
                    secondary_local = nisa.tensor_scalar(
                        data=secondary_nonneg,
                        op0=nl.minimum,
                        operand0=np.int32(secondary_stride - 1),
                        dtype=np.int32,
                    )
                    secondary_rows = nl.add(
                        nl.broadcast_to(secondary_base, shape=(K, 1)),
                        secondary_local,
                    )
                    kv_gathered[...] = nl.load(
                        kv_secondary[secondary_rows[i_p, 0], i_f],
                    )
                else:
                    is_primary = nl.less(topk_sb, nl.int32(primary_len))
                    zero_idx = nl.full(
                        (par_dim(K), 1),
                        np.int32(0),
                        dtype=nl.int32,
                        buffer=nl.sbuf,
                    )
                    primary_nonneg = nl.ndarray(
                        (par_dim(K), 1),
                        dtype=nl.int32,
                        buffer=nl.sbuf,
                    )
                    primary_nonneg[...] = nl.maximum(topk_sb, nl.int32(0))
                    primary_clamped = nisa.tensor_scalar(
                        data=primary_nonneg,
                        op0=nl.minimum,
                        operand0=np.int32(primary_len - 1),
                        dtype=np.int32,
                    )
                    primary_local = nl.ndarray(
                        (par_dim(K), 1),
                        dtype=nl.int32,
                        buffer=nl.sbuf,
                    )
                    primary_local[...] = nl.where(
                        is_primary,
                        primary_clamped,
                        zero_idx,
                        dtype=nl.int32,
                    )

                    secondary_raw = nl.subtract(topk_sb, nl.int32(primary_len))
                    secondary_nonneg = nl.ndarray(
                        (par_dim(K), 1),
                        dtype=nl.int32,
                        buffer=nl.sbuf,
                    )
                    secondary_nonneg[...] = nl.maximum(
                        secondary_raw,
                        nl.int32(0),
                    )
                    secondary_local = nisa.tensor_scalar(
                        data=secondary_nonneg,
                        op0=nl.minimum,
                        operand0=np.int32(secondary_stride - 1),
                        dtype=np.int32,
                    )
                    primary_rows = nl.add(
                        nl.broadcast_to(primary_base, shape=(K, 1)),
                        primary_local,
                    )
                    secondary_rows = nl.add(
                        nl.broadcast_to(secondary_base, shape=(K, 1)),
                        secondary_local,
                    )

                    primary_g = nl.load(kv_primary[primary_rows[i_p, 0], i_f])
                    secondary_g = nl.load(
                        kv_secondary[secondary_rows[i_p, 0], i_f],
                    )
                    is_primary_b = nl.broadcast_to(is_primary, shape=(K, d))
                    kv_gathered[...] = nl.where(
                        is_primary_b,
                        primary_g,
                        secondary_g,
                        dtype=nl.bfloat16,
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

                mask_sb = nl.load(mask[bi : bi + 1, kt * K : (kt + 1) * K])
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


def _sparse_attn_batched_paged_two_source_multiK_entry(
    q_T,
    kv_primary,
    kv_secondary,
    topk_T,
    mask,
    owner_ids,
    primary_owner_ids,
    sink,
    *,
    primary_len: int,
    secondary_stride: int,
    primary_prefix_len: int = 0,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _sparse_attn_batched_paged_two_source_multiK_kernel(
        q_T,
        kv_primary,
        kv_secondary,
        topk_T,
        mask,
        owner_ids,
        primary_owner_ids,
        sink,
        int(primary_len),
        int(secondary_stride),
        int(primary_prefix_len),
    )


def _sparse_attn_batched_paged_two_source_shared_owner_multiK_entry(
    q_T,
    kv_primary,
    kv_secondary,
    topk_T,
    mask,
    owner_ids,
    sink,
    *,
    primary_len: int,
    secondary_stride: int,
    primary_prefix_len: int = 0,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _sparse_attn_batched_paged_two_source_multiK_kernel(
        q_T,
        kv_primary,
        kv_secondary,
        topk_T,
        mask,
        owner_ids,
        owner_ids,
        sink,
        int(primary_len),
        int(secondary_stride),
        int(primary_prefix_len),
    )


def run_sparse_attention_paged_two_source_device(
    *,
    q_scaled_t: Any,
    kv_primary: Any,
    kv_secondary: Any,
    topk_t: Any,
    mask: Any,
    owner_ids: Any,
    primary_owner_ids: Any | None = None,
    sink: Any,
    output: Any,
    primary_len: int,
    secondary_stride: int,
    primary_prefix_len: int | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Run sparse attention over local indices split across two KV sources."""

    q_shape = tuple(int(dim) for dim in getattr(q_scaled_t, "shape"))
    primary_shape = tuple(int(dim) for dim in getattr(kv_primary, "shape"))
    secondary_shape = tuple(int(dim) for dim in getattr(kv_secondary, "shape"))
    topk_shape = tuple(int(dim) for dim in getattr(topk_t, "shape"))
    mask_shape = tuple(int(dim) for dim in getattr(mask, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    shared_primary_owner = primary_owner_ids is None or primary_owner_ids is owner_ids
    primary_owner_shape = (
        owner_shape
        if shared_primary_owner
        else tuple(int(dim) for dim in getattr(primary_owner_ids, "shape"))
    )
    sink_shape = tuple(int(dim) for dim in getattr(sink, "shape"))
    out_shape = tuple(int(dim) for dim in getattr(output, "shape"))

    if len(q_shape) != 3:
        raise ValueError(f"q_scaled_t must be [tokens, head_dim, heads], got {q_shape}")
    if len(primary_shape) != 2 or len(secondary_shape) != 2:
        raise ValueError(
            "kv_primary and kv_secondary must be 2-D, got "
            f"{primary_shape}/{secondary_shape}"
        )
    if len(topk_shape) != 2:
        raise ValueError(f"topk_t must be [K, tokens], got {topk_shape}")
    if len(mask_shape) != 2:
        raise ValueError(f"mask must be [tokens, K], got {mask_shape}")
    if len(owner_shape) != 1:
        raise ValueError(f"owner_ids must be [tokens], got {owner_shape}")
    if len(primary_owner_shape) != 1:
        raise ValueError(
            f"primary_owner_ids must be [tokens], got {primary_owner_shape}"
        )
    if len(sink_shape) != 2:
        raise ValueError(f"sink must be [1, heads], got {sink_shape}")

    tokens, head_dim, num_heads = q_shape
    k, topk_tokens = topk_shape
    primary_len = int(primary_len)
    secondary_stride = int(secondary_stride)
    primary_prefix_len = 0 if primary_prefix_len is None else int(primary_prefix_len)
    if 0 < primary_prefix_len < K_TILE:
        primary_prefix_len = 0
    if topk_tokens != tokens:
        raise ValueError(f"topk_t tokens={topk_tokens} must match q tokens={tokens}")
    if mask_shape != (tokens, k):
        raise ValueError(f"mask must be [{tokens}, {k}], got {mask_shape}")
    if owner_shape != (tokens,):
        raise ValueError(f"owner_ids must be [{tokens}], got {owner_shape}")
    if primary_owner_shape != (tokens,):
        raise ValueError(
            f"primary_owner_ids must be [{tokens}], got {primary_owner_shape}"
        )
    if sink_shape != (1, num_heads):
        raise ValueError(f"sink must be [1, {num_heads}], got {sink_shape}")
    if primary_shape[1] != head_dim or secondary_shape[1] != head_dim:
        raise ValueError(
            "KV head_dim must match q head_dim: "
            f"primary={primary_shape[1]}, secondary={secondary_shape[1]}, q={head_dim}"
        )
    if out_shape != (tokens, num_heads, head_dim):
        raise ValueError(
            f"output must be [{tokens}, {num_heads}, {head_dim}], got {out_shape}"
        )
    if primary_len <= 0 or secondary_stride <= 0:
        raise ValueError("primary_len and secondary_stride must be positive")
    if primary_prefix_len < 0 or primary_prefix_len > k:
        raise ValueError(
            f"primary_prefix_len must be in [0, {k}], got {primary_prefix_len}"
        )
    if k % K_TILE:
        raise NotImplementedError(f"K={k} must be a multiple of K_TILE={K_TILE}")
    if head_dim % D_BLOCK:
        raise NotImplementedError(f"head_dim={head_dim} not a multiple of {D_BLOCK}")
    if num_heads > P_MAX:
        raise ValueError(f"num_heads={num_heads} must be <= {P_MAX}")

    cache = (
        _TWO_SOURCE_PAGED_ATTENTION_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    entry = (
        _sparse_attn_batched_paged_two_source_shared_owner_multiK_entry
        if shared_primary_owner
        else _sparse_attn_batched_paged_two_source_multiK_entry
    )
    cache_key = (
        "sparse_attention_paged_two_source_shared_owner_device"
        if shared_primary_owner
        else "sparse_attention_paged_two_source_device",
        _TWO_SOURCE_KERNEL_VERSION,
        q_shape,
        primary_shape,
        secondary_shape,
        topk_shape,
        mask_shape,
        owner_shape,
        primary_owner_shape,
        sink_shape,
        out_shape,
        str(_dtype_like(q_scaled_t)),
        str(_dtype_like(kv_primary)),
        str(_dtype_like(kv_secondary)),
        str(_dtype_like(topk_t)),
        str(_dtype_like(mask)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(owner_ids if shared_primary_owner else primary_owner_ids)),
        str(_dtype_like(sink)),
        primary_len,
        secondary_stride,
        primary_prefix_len,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        sample_args = [
            _sample_like(q_scaled_t),
            _sample_like(kv_primary),
            _sample_like(kv_secondary),
            _sample_like(topk_t),
            _sample_like(mask),
            _sample_like(owner_ids),
        ]
        if not shared_primary_owner:
            sample_args.append(_sample_like(primary_owner_ids))
        sample_args.append(_sample_like(sink))
        owner_mode = "shared_owner" if shared_primary_owner else "split_owner"
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            entry,
            *sample_args,
            primary_len=primary_len,
            secondary_stride=secondary_stride,
            primary_prefix_len=primary_prefix_len,
            name=(
                f"dsv4_sparse_attention_two_source_{owner_mode}_"
                f"v{_TWO_SOURCE_KERNEL_VERSION}_"
                f"t{tokens}_d{head_dim}_k{k}_p{primary_len}_s{secondary_stride}"
                f"_pp{primary_prefix_len}"
            ),
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    inputs = {
        "q_T": q_scaled_t,
        "kv_primary": kv_primary,
        "kv_secondary": kv_secondary,
        "topk_T": topk_t,
        "mask": mask,
        "owner_ids": owner_ids,
        "sink": sink,
    }
    if not shared_primary_owner:
        inputs["primary_owner_ids"] = primary_owner_ids
    kernel(
        inputs=inputs,
        outputs={"output0": output},
    )
    return output
