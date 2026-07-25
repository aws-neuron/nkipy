"""Device mutation helpers for DeepSeek-V4 compressor state.

The key invariant is ring-addressed state.  Decode never physically shifts the
overlap state; kernels write each token to ``position % ring_size`` and later
compression gathers the logical previous/current groups.

The file is intentionally grouped by this shared state layout and kernel-cache
surface. Keep new kernels here only when they mutate the same compressor/SWA
state slabs; move pure graph math to the model trace functions instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.attention.deepseek_v4.state import (
    Dsv4CompressorStateSpec,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_first_dim_slice as _alias_device_value_first_dim_slice,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)
from nkipy_serving.runtime.device_tensor import dtype_like as _dtype_like
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.device_tensor import (
    is_device_array_like as _is_device_tensor_like,
)
from nkipy_serving.runtime.device_tensor import sample_like as _sample_like
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock
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


_WRITE_KV_SCORE_STATE_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_KV_SCORE_STATE_OWNER_POS_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_KV_SCORE_STATE_OWNER_CLEN_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_SWA_KV_SCORE_STATE_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_SWA_DUAL_KV_SCORE_STATE_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_SWA_DUAL_KV_SCORE_STATE_OWNER_POS_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_SWA_DUAL_KV_SCORE_STATE_OWNER_CLEN_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_SWA_KV_SCORE_STATE_OWNER_POS_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_SWA_KV_SCORE_STATE_OWNER_CLEN_KERNEL_CACHE: dict[tuple, Any] = {}
_DECODE_POOL_FROM_STATE_KERNEL_CACHE: dict[tuple, Any] = {}
_PREFILL_POOL_FROM_SLAB_KERNEL_CACHE: dict[tuple, Any] = {}


def _device_or_upload(tensor: Any, *, name: str, dtype: Any | None = None) -> Any:
    if _is_device_tensor_like(tensor):
        return tensor

    arr = np.asarray(tensor)
    if dtype is not None:
        arr = arr.astype(dtype)
    return get_device_tensor_cls().from_numpy(np.ascontiguousarray(arr), name=name)


def _get_or_compile_kernel(
    cache: dict[tuple, Any],
    cache_key: tuple,
    fn: Any,
    *sample_args: Any,
    name: str,
    build_dir: Any,
    namespace: str,
    device_kernel_cls: Any | None = None,
    **compile_kwargs: Any,
) -> Any:
    """Memoized compile-and-load for the state-write/scatter kernels.

    Centralizes the get -> (bridge-resolve DeviceKernel) -> compile_and_load ->
    cache-store idiom repeated by every ``run_*_device`` kernel here. The
    caller still builds ``cache_key``/``name``/sample args and passes them
    through verbatim, so the produced NEFF name and cache key are byte-identical
    to the previous inline code.
    """
    kernel = cache.get(cache_key)
    if kernel is not None:
        return kernel
    if device_kernel_cls is None:
        ensure_nki_bridge()
        from nkipy.runtime import DeviceKernel

        device_kernel_cls = DeviceKernel
    try:
        kernel = compile_and_load_with_lock(
            device_kernel_cls,
            fn,
            *sample_args,
            name=name,
            build_dir=build_dir,
            namespace=namespace,
            **compile_kwargs,
        )
    except RuntimeError as exc:
        if "DeviceKernel late compile blocked after namespace seal" in str(exc):
            raise RuntimeError(f"{exc}; logical_cache_key={cache_key!r}") from exc
        raise
    cache[cache_key] = kernel
    return kernel


_BKT_OWNER_ARRAY_CACHE: dict[tuple, Any] = {}
_BKT_OWNER_ARRAY_CACHE_CAP = 512


def _bkt_owner_dev(arr: np.ndarray, *, key: tuple, name: str) -> Any:
    """Memoized device upload for bucketed owner/pos arrays.

    The arrays are pure functions of their explicit cache key and repeat across
    all layers of a step; cache uploads, bounded LRU-lite.
    """
    cached = _BKT_OWNER_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_BKT_OWNER_ARRAY_CACHE) >= _BKT_OWNER_ARRAY_CACHE_CAP:
        _BKT_OWNER_ARRAY_CACHE.clear()
    dev = get_device_tensor_cls().from_numpy(np.ascontiguousarray(arr), name=name)
    _BKT_OWNER_ARRAY_CACHE[key] = dev
    return dev


_DEVICE_SCALAR_I32_CACHE: dict[tuple[int, str], Any] = {}


def _pad_i32_vector(values: Any, *, rows: int, fill: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32).reshape(-1)
    rows_i = int(rows)
    if arr.shape[0] >= rows_i:
        return np.ascontiguousarray(arr[:rows_i])
    fill_i = int(arr[-1]) if fill is None and arr.size else int(fill or 0)
    out = np.full((rows_i,), np.int32(fill_i), dtype=np.int32)
    out[: arr.shape[0]] = arr
    return out


def _device_scalar_i32(value: int, *, name: str) -> Any:
    """Device ``[1, 1]`` int32 runtime scalar, memoized per (value, name).

    Used for runtime length scalars (e.g. bucketed-prefill ``cache_real_clen``)
    that the NKI kernel loads but are deliberately kept out of the kernel cache
    key, so one NEFF serves every real length within a bucket. Values repeat
    across all layers of a step, so cache the upload."""
    key = (int(value), str(name))
    cached = _DEVICE_SCALAR_I32_CACHE.get(key)
    if cached is not None:
        return cached
    arr = np.asarray([[int(value)]], dtype=np.int32)
    dev = get_device_tensor_cls().from_numpy(np.ascontiguousarray(arr), name=name)
    _DEVICE_SCALAR_I32_CACHE[key] = dev
    return dev


def _device_vector_or_upload(
    dev_value: Any | None,
    arr: np.ndarray,
    *,
    name: str,
) -> Any:
    arr_i32 = np.ascontiguousarray(arr.astype(np.int32, copy=False).reshape(-1))
    if dev_value is not None:
        dev_shape = tuple(int(dim) for dim in getattr(dev_value, "shape", ()))
        if dev_shape == tuple(int(dim) for dim in arr_i32.shape):
            return dev_value
        if dev_shape == (int(arr_i32.shape[0]), 1):
            alias = _alias_device_value_shape(dev_value, tuple(arr_i32.shape))
            if alias is not None:
                return alias
    return get_device_tensor_cls().from_numpy(arr_i32, name=name)


def _prefill_state_tail_len(*, spec: Dsv4CompressorStateSpec, seqlen: int) -> int:
    """Number of prefill token projections that can affect future decode."""

    seqlen_i = int(seqlen)
    if seqlen_i <= 0:
        return 0
    ratio = int(spec.compress_ratio)
    if bool(spec.overlap):
        remainder = seqlen_i % ratio
        return min(seqlen_i, ratio + remainder)
    return seqlen_i % ratio


def write_kv_score_state_oracle(
    kv_score_state: np.ndarray,
    kv_new: np.ndarray,
    score_new: np.ndarray,
    owner_ids: np.ndarray,
    positions: np.ndarray,
    ape: np.ndarray,
    *,
    spec: Dsv4CompressorStateSpec,
) -> np.ndarray:
    """CPU reference for state scatter.

    ``kv_score_state`` is mutated in place.  The last dimension is packed as
    ``[kv, score]`` with each half of width ``spec.state_width``.
    """

    state = np.asarray(kv_score_state)
    kv = np.asarray(kv_new, dtype=np.float32)
    score = np.asarray(score_new, dtype=np.float32)
    ape_arr = np.asarray(ape, dtype=np.float32)
    owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    width = int(spec.state_width)
    if state.shape != spec.state_shape:
        raise ValueError(f"kv_score_state shape {state.shape} != {spec.state_shape}")
    if kv.shape != score.shape:
        raise ValueError(f"kv/score shapes differ: {kv.shape}/{score.shape}")
    if kv.ndim != 2 or kv.shape[1] != width:
        raise ValueError(f"kv_new must be [N, {width}], got {kv.shape}")
    if owners.shape != (kv.shape[0],) or pos.shape != (kv.shape[0],):
        raise ValueError("owner_ids and positions must be [N] matching kv_new rows")
    if ape_arr.shape != (int(spec.compress_ratio), width):
        raise ValueError(
            f"ape must be [{spec.compress_ratio}, {width}], got {ape_arr.shape}"
        )

    rows = spec.state_row(owners, pos)
    ape_rows = ape_arr[pos % int(spec.compress_ratio)]
    state[rows, :width] = kv.astype(state.dtype, copy=False)
    state[rows, width : 2 * width] = (score + ape_rows).astype(
        state.dtype,
        copy=False,
    )
    return state


def decode_pool_from_state_oracle(
    kv_score_state: np.ndarray,
    owner_ids: np.ndarray,
    end_positions: np.ndarray,
    *,
    spec: Dsv4CompressorStateSpec,
) -> np.ndarray:
    """Pool one compressed row per owner from ring-addressed state.

    Returns the pre-RMS/RoPE pooled KV ``[N, head_dim]``.  This is an oracle for
    the later fused device kernel and documents the no-shift overlap semantics.
    """

    state = np.asarray(kv_score_state, dtype=np.float32)
    owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
    end_pos = np.asarray(end_positions, dtype=np.int64).reshape(-1)
    if owners.shape != end_pos.shape:
        raise ValueError(
            f"owner_ids and end_positions must match, got {owners.shape}/{end_pos.shape}"
        )
    if state.shape != spec.state_shape:
        raise ValueError(f"kv_score_state shape {state.shape} != {spec.state_shape}")

    ratio = int(spec.compress_ratio)
    d = int(spec.head_dim)
    width = int(spec.state_width)
    out = np.zeros((owners.shape[0], d), dtype=np.float32)
    for i, (owner, pos) in enumerate(zip(owners, end_pos)):
        if bool(spec.overlap):
            kv_parts = np.zeros((2 * ratio, d), dtype=np.float32)
            score_parts = np.full((2 * ratio, d), -np.inf, dtype=np.float32)
            prev_start = int(pos) - 2 * ratio + 1
            cur_start = int(pos) - ratio + 1
            for j in range(ratio):
                prev_pos = prev_start + j
                if prev_pos >= 0:
                    row = int(
                        spec.state_row(np.asarray([owner]), np.asarray([prev_pos]))[0]
                    )
                    kv_parts[j] = state[row, :d]
                    score_parts[j] = state[row, width : width + d]
                cur_pos = cur_start + j
                row = int(spec.state_row(np.asarray([owner]), np.asarray([cur_pos]))[0])
                kv_parts[ratio + j] = state[row, d : 2 * d]
                score_parts[ratio + j] = state[row, width + d : width + 2 * d]
        else:
            kv_parts = np.zeros((ratio, d), dtype=np.float32)
            score_parts = np.full((ratio, d), -np.inf, dtype=np.float32)
            cur_start = int(pos) - ratio + 1
            for j in range(ratio):
                row = int(
                    spec.state_row(
                        np.asarray([owner]),
                        np.asarray([cur_start + j]),
                    )[0]
                )
                kv_parts[j] = state[row, :d]
                score_parts[j] = state[row, width : width + d]

        score_max = np.max(score_parts, axis=0, keepdims=True)
        weights = np.exp(score_parts - score_max)
        denom = np.sum(weights, axis=0, keepdims=True)
        out[i] = np.sum(kv_parts * (weights / denom), axis=0)
    return out


def prefill_pool_from_slab_oracle(
    kv_new: np.ndarray,
    score_new: np.ndarray,
    ape: np.ndarray,
    *,
    bsz: int,
    seqlen: int,
    ratio: int,
    head_dim: int,
    overlap: bool,
) -> np.ndarray:
    """CPU reference for prefill pooling from direct projection slabs."""

    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    ratio_i = int(ratio)
    d_i = int(head_dim)
    if seqlen_i % ratio_i:
        raise ValueError("seqlen must be a multiple of ratio")
    width = 2 * d_i if bool(overlap) else d_i
    kv = np.asarray(kv_new, dtype=np.float32).reshape(bsz_i, seqlen_i, width)
    score = np.asarray(score_new, dtype=np.float32).reshape(
        bsz_i,
        seqlen_i,
        width,
    )
    ape_arr = np.asarray(ape, dtype=np.float32)
    if ape_arr.shape != (ratio_i, width):
        raise ValueError(f"ape must be [{ratio_i}, {width}], got {ape_arr.shape}")

    groups = seqlen_i // ratio_i
    kv_r = kv.reshape(bsz_i, groups, ratio_i, width)
    score_r = score.reshape(bsz_i, groups, ratio_i, width) + ape_arr
    if bool(overlap):
        kv_parts = np.zeros((bsz_i, groups, 2 * ratio_i, d_i), dtype=np.float32)
        score_parts = np.full(
            (bsz_i, groups, 2 * ratio_i, d_i),
            -np.inf,
            dtype=np.float32,
        )
        kv_parts[:, :, ratio_i:] = kv_r[:, :, :, d_i:]
        score_parts[:, :, ratio_i:] = score_r[:, :, :, d_i:]
        if groups > 1:
            kv_parts[:, 1:, :ratio_i] = kv_r[:, :-1, :, :d_i]
            score_parts[:, 1:, :ratio_i] = score_r[:, :-1, :, :d_i]
        kv_r = kv_parts
        score_r = score_parts

    score_max = np.max(score_r, axis=2, keepdims=True)
    weights = np.exp(score_r - score_max)
    weights = weights / np.sum(weights, axis=2, keepdims=True)
    return np.sum(kv_r * weights, axis=2).astype(np.float32)


if _NKI_AVAILABLE:

    @_nki.jit
    def _write_kv_score_state_kernel(
        kv_score_state: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ape: "nt.tensor",
        live_rows: "nt.tensor",
        ring_size: int,
        guard_owner: int,
    ):
        """In-place ring scatter of ``[kv, score + ape]`` rows."""

        n_new, width = kv_new.shape
        ratio = ape.shape[0]
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        pos_2d = positions.reshape((n_new, 1))
        ring_size_i = nl.int32(ring_size)
        live_rows_sb = nl.load(live_rows[0:1, 0:1])
        live_rows_i = live_rows_sb[0, 0]

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(width)[None, :]
            row_ids = nl.add(i_p, nl.int32(t0))

            owners_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(pos_2d[t0 : t0 + cur])
            row_live = nl.less(row_ids, live_rows_i)
            guard_owner_sb = nl.full(owners_sb.shape, guard_owner, dtype=nl.int32)
            safe_owner = nl.where(row_live, owners_sb, guard_owner_sb)
            ring_offsets = nl.mod(pos_sb, ring_size_i)
            rows = nl.add(
                nl.multiply(safe_owner, ring_size_i),
                ring_offsets,
            )
            ape_offsets = nl.mod(pos_sb, nl.int32(ratio))

            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_f])
            score_ape = nl.add(score_sb, ape_sb)

            nl.store(
                dst=kv_score_state[rows[i_p, 0], i_f],
                value=kv_sb[i_p, i_f],
            )
            nl.store(
                dst=kv_score_state[rows[i_p, 0], width + i_f],
                value=score_ape[i_p, i_f],
            )
        return kv_score_state

    @_nki.jit
    def _write_kv_score_state_owner_pos_kernel(
        kv_score_state: "nt.tensor[nt.mutable]",
        compressed_kv_cache: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        compressed_rows: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ape: "nt.tensor",
        ring_size: int,
        ratio: int,
        max_clen: int,
    ):
        """In one launch, update decode ring state and compressed-KV cache."""

        n_new, width = kv_new.shape
        d = compressed_rows.shape[1]
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        ring_size_i = nl.int32(ring_size)
        ratio_i = nl.int32(ratio)
        max_clen_i = nl.int32(max_clen)
        inv_ratio = np.float32(1.0) / np.float32(ratio)

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_state = nl.arange(width)[None, :]
            i_cache = nl.arange(d)[None, :]

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])

            ring_offsets = nl.mod(pos_sb, ring_size_i)
            state_rows = nl.add(
                nl.multiply(owner_sb, ring_size_i),
                ring_offsets,
            )
            ape_offsets = nl.mod(pos_sb, ratio_i)

            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_state])
            score_ape = nl.add(score_sb, ape_sb)

            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], i_state],
                value=kv_sb[i_p, i_state],
            )
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], width + i_state],
                value=score_ape[i_p, i_state],
            )

            pos_base = nl.subtract(pos_sb, nl.mod(pos_sb, ratio_i))
            cpos_f = nl.ndarray(pos_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            cpos_f[...] = nisa.tensor_scalar(
                data=pos_base,
                op0=nl.multiply,
                operand0=inv_ratio,
                dtype=nl.float32,
            )
            cpos = nl.ndarray(pos_base.shape, dtype=np.int32, buffer=nl.sbuf)
            cpos[...] = nl.copy(cpos_f, dtype=nl.int32)
            cache_rows = nl.add(nl.multiply(owner_sb, max_clen_i), cpos)
            compressed_sb = nl.load(compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=compressed_kv_cache[cache_rows[i_p, 0], i_cache],
                value=compressed_sb[i_p, i_cache],
            )
        return kv_score_state, compressed_kv_cache

    @_nki.jit
    def _write_kv_score_state_owner_clen_kernel(
        kv_score_state: "nt.tensor[nt.mutable]",
        compressed_kv_cache: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        compressed_rows: "nt.tensor",
        state_owner_ids: "nt.tensor",
        state_positions: "nt.tensor",
        cache_owner_ids: "nt.tensor",
        ape: "nt.tensor",
        ring_size: int,
        clen: int,
        owner_id_stride: int,
        max_clen: int,
    ):
        """In one prefill launch, update ring state and compressed-KV cache."""

        n_state, width = kv_new.shape
        n_cache, d = compressed_rows.shape
        ratio = ape.shape[0]
        MAX_T = 128

        state_tiles = (n_state + MAX_T - 1) // MAX_T
        state_last = n_state - (state_tiles - 1) * MAX_T
        cache_tiles = (n_cache + MAX_T - 1) // MAX_T
        cache_last = n_cache - (cache_tiles - 1) * MAX_T

        state_owner_2d = state_owner_ids.reshape((n_state, 1))
        state_pos_2d = state_positions.reshape((n_state, 1))
        cache_owner_2d = cache_owner_ids.reshape((cache_owner_ids.shape[0], 1))
        ring_size_i = nl.int32(ring_size)
        clen_i = nl.int32(clen)
        stride_i = nl.int32(owner_id_stride)
        max_clen_i = nl.int32(max_clen)
        inv_clen = np.float32(1.0) / np.float32(clen)

        for ti in nl.static_range(state_tiles):
            if ti < state_tiles - 1:
                cur = MAX_T
            else:
                cur = state_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(width)[None, :]

            owner_sb = nl.load(state_owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(state_pos_2d[t0 : t0 + cur])
            ring_offsets = nl.mod(pos_sb, ring_size_i)
            rows = nl.add(
                nl.multiply(owner_sb, ring_size_i),
                ring_offsets,
            )
            ape_offsets = nl.mod(pos_sb, nl.int32(ratio))

            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_f])
            score_ape = nl.add(score_sb, ape_sb)

            nl.store(
                dst=kv_score_state[rows[i_p, 0], i_f],
                value=kv_sb[i_p, i_f],
            )
            nl.store(
                dst=kv_score_state[rows[i_p, 0], width + i_f],
                value=score_ape[i_p, i_f],
            )

        for ti in nl.static_range(cache_tiles):
            if ti < cache_tiles - 1:
                cur = MAX_T
            else:
                cur = cache_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d)[None, :]
            row_ids = nl.add(i_p, nl.int32(t0))
            cpos = nl.mod(row_ids, clen_i)
            req_base = nl.subtract(row_ids, cpos)
            req_f = nl.ndarray(req_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            req_f[...] = nisa.tensor_scalar(
                data=req_base,
                op0=nl.multiply,
                operand0=inv_clen,
                dtype=nl.float32,
            )
            req_ids = nl.ndarray(req_base.shape, dtype=nl.int32, buffer=nl.sbuf)
            req_ids[...] = nl.copy(req_f, dtype=nl.int32)
            owner_offsets = nl.multiply(req_ids, stride_i)
            owner_sb = nl.load(cache_owner_2d[owner_offsets[i_p, 0], 0:1])
            rows = nl.add(nl.multiply(owner_sb, max_clen_i), cpos)
            row_sb = nl.load(compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=compressed_kv_cache[rows[i_p, 0], i_f],
                value=row_sb[i_p, i_f],
            )
        return kv_score_state, compressed_kv_cache

    @_nki.jit
    def _write_swa_kv_score_state_kernel(
        swa_kv_cache: "nt.tensor[nt.mutable]",
        kv_score_state: "nt.tensor[nt.mutable]",
        swa_rows: "nt.tensor",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ape: "nt.tensor",
        window_size: int,
        ring_size: int,
    ):
        """In one decode launch, update SWA cache and compressor ring state."""

        n_new, d = swa_rows.shape
        _, width = kv_new.shape
        ratio = ape.shape[0]
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        window_i = nl.int32(window_size)
        ring_size_i = nl.int32(ring_size)

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_swa = nl.arange(d)[None, :]
            i_state = nl.arange(width)[None, :]

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])

            pos_in_window = nl.mod(pos_sb, window_i)
            swa_cache_rows = nl.add(nl.multiply(owner_sb, window_i), pos_in_window)
            swa_sb = nl.load(swa_rows[t0 : t0 + cur])
            nl.store(
                dst=swa_kv_cache[swa_cache_rows[i_p, 0], i_swa],
                value=swa_sb[i_p, i_swa],
            )

            ring_offsets = nl.mod(pos_sb, ring_size_i)
            state_rows = nl.add(
                nl.multiply(owner_sb, ring_size_i),
                ring_offsets,
            )
            ape_offsets = nl.mod(pos_sb, nl.int32(ratio))
            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_state])
            score_ape = nl.add(score_sb, ape_sb)

            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], i_state],
                value=kv_sb[i_p, i_state],
            )
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], width + i_state],
                value=score_ape[i_p, i_state],
            )
        return swa_kv_cache, kv_score_state

    @_nki.jit
    def _write_swa_dual_kv_score_state_kernel(
        swa_kv_cache: "nt.tensor[nt.mutable]",
        kv_score_state: "nt.tensor[nt.mutable]",
        indexer_kv_score_state: "nt.tensor[nt.mutable]",
        swa_rows: "nt.tensor",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        indexer_kv_new: "nt.tensor",
        indexer_score_new: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ape: "nt.tensor",
        indexer_ape: "nt.tensor",
        live_rows: "nt.tensor",
        window_size: int,
        ring_size: int,
        indexer_ring_size: int,
        guard_owner: int,
    ):
        """In one decode launch, update SWA plus main and indexer ring states."""

        n_new, d = swa_rows.shape
        _, width = kv_new.shape
        _, indexer_width = indexer_kv_new.shape
        ratio = ape.shape[0]
        indexer_ratio = indexer_ape.shape[0]
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        window_i = nl.int32(window_size)
        ring_size_i = nl.int32(ring_size)
        indexer_ring_size_i = nl.int32(indexer_ring_size)
        live_rows_sb = nl.load(live_rows[0:1, 0:1])
        live_rows_i = live_rows_sb[0, 0]

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_swa = nl.arange(d)[None, :]
            i_state = nl.arange(width)[None, :]
            i_indexer_state = nl.arange(indexer_width)[None, :]
            row_ids = nl.add(i_p, nl.int32(t0))

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])
            row_live = nl.less(row_ids, live_rows_i)
            guard_owner_sb = nl.full(owner_sb.shape, guard_owner, dtype=nl.int32)
            safe_owner = nl.where(row_live, owner_sb, guard_owner_sb)

            pos_in_window = nl.mod(pos_sb, window_i)
            swa_cache_rows = nl.add(
                nl.multiply(safe_owner, window_i),
                pos_in_window,
            )
            swa_sb = nl.load(swa_rows[t0 : t0 + cur])
            nl.store(
                dst=swa_kv_cache[swa_cache_rows[i_p, 0], i_swa],
                value=swa_sb[i_p, i_swa],
            )

            ring_offsets = nl.mod(pos_sb, ring_size_i)
            state_rows = nl.add(nl.multiply(safe_owner, ring_size_i), ring_offsets)
            ape_offsets = nl.mod(pos_sb, nl.int32(ratio))
            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_state])
            score_ape = nl.add(score_sb, ape_sb)
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], i_state],
                value=kv_sb[i_p, i_state],
            )
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], width + i_state],
                value=score_ape[i_p, i_state],
            )

            indexer_ring_offsets = nl.mod(pos_sb, indexer_ring_size_i)
            indexer_state_rows = nl.add(
                nl.multiply(safe_owner, indexer_ring_size_i),
                indexer_ring_offsets,
            )
            indexer_ape_offsets = nl.mod(pos_sb, nl.int32(indexer_ratio))
            indexer_kv_sb = nl.load(indexer_kv_new[t0 : t0 + cur])
            indexer_score_sb = nl.load(indexer_score_new[t0 : t0 + cur])
            indexer_ape_sb = nl.load(
                indexer_ape[indexer_ape_offsets[i_p, 0], i_indexer_state],
            )
            indexer_score_ape = nl.add(indexer_score_sb, indexer_ape_sb)
            nl.store(
                dst=indexer_kv_score_state[
                    indexer_state_rows[i_p, 0],
                    i_indexer_state,
                ],
                value=indexer_kv_sb[i_p, i_indexer_state],
            )
            nl.store(
                dst=indexer_kv_score_state[
                    indexer_state_rows[i_p, 0],
                    indexer_width + i_indexer_state,
                ],
                value=indexer_score_ape[i_p, i_indexer_state],
            )
        return swa_kv_cache, kv_score_state, indexer_kv_score_state

    @_nki.jit
    def _write_swa_dual_kv_score_state_owner_pos_kernel(
        swa_kv_cache: "nt.tensor[nt.mutable]",
        kv_score_state: "nt.tensor[nt.mutable]",
        compressed_kv_cache: "nt.tensor[nt.mutable]",
        indexer_kv_score_state: "nt.tensor[nt.mutable]",
        indexer_compressed_kv_cache: "nt.tensor[nt.mutable]",
        swa_rows: "nt.tensor",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        compressed_rows: "nt.tensor",
        indexer_kv_new: "nt.tensor",
        indexer_score_new: "nt.tensor",
        indexer_compressed_rows: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ape: "nt.tensor",
        indexer_ape: "nt.tensor",
        window_size: int,
        ring_size: int,
        indexer_ring_size: int,
        ratio: int,
        indexer_ratio: int,
        max_clen: int,
        indexer_max_clen: int,
    ):
        """Boundary decode update: SWA, main cache/state, and indexer cache/state."""

        n_new, d = swa_rows.shape
        _, width = kv_new.shape
        _, d_comp = compressed_rows.shape
        _, indexer_width = indexer_kv_new.shape
        _, indexer_d_comp = indexer_compressed_rows.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        window_i = nl.int32(window_size)
        ring_size_i = nl.int32(ring_size)
        indexer_ring_size_i = nl.int32(indexer_ring_size)
        ratio_i = nl.int32(ratio)
        indexer_ratio_i = nl.int32(indexer_ratio)
        max_clen_i = nl.int32(max_clen)
        indexer_max_clen_i = nl.int32(indexer_max_clen)
        inv_ratio = np.float32(1.0) / np.float32(ratio)
        indexer_inv_ratio = np.float32(1.0) / np.float32(indexer_ratio)

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_swa = nl.arange(d)[None, :]
            i_comp = nl.arange(d_comp)[None, :]
            i_state = nl.arange(width)[None, :]
            i_indexer_comp = nl.arange(indexer_d_comp)[None, :]
            i_indexer_state = nl.arange(indexer_width)[None, :]

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])

            pos_in_window = nl.mod(pos_sb, window_i)
            swa_cache_rows = nl.add(nl.multiply(owner_sb, window_i), pos_in_window)
            swa_sb = nl.load(swa_rows[t0 : t0 + cur])
            nl.store(
                dst=swa_kv_cache[swa_cache_rows[i_p, 0], i_swa],
                value=swa_sb[i_p, i_swa],
            )

            ring_offsets = nl.mod(pos_sb, ring_size_i)
            state_rows = nl.add(nl.multiply(owner_sb, ring_size_i), ring_offsets)
            ape_offsets = nl.mod(pos_sb, ratio_i)
            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_state])
            score_ape = nl.add(score_sb, ape_sb)
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], i_state],
                value=kv_sb[i_p, i_state],
            )
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], width + i_state],
                value=score_ape[i_p, i_state],
            )

            pos_base = nl.subtract(pos_sb, nl.mod(pos_sb, ratio_i))
            cpos_f = nl.ndarray(pos_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            cpos_f[...] = nisa.tensor_scalar(
                data=pos_base,
                op0=nl.multiply,
                operand0=inv_ratio,
                dtype=nl.float32,
            )
            cpos = nl.ndarray(pos_base.shape, dtype=np.int32, buffer=nl.sbuf)
            cpos[...] = nl.copy(cpos_f, dtype=nl.int32)
            cache_rows = nl.add(nl.multiply(owner_sb, max_clen_i), cpos)
            compressed_sb = nl.load(compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=compressed_kv_cache[cache_rows[i_p, 0], i_comp],
                value=compressed_sb[i_p, i_comp],
            )

            indexer_ring_offsets = nl.mod(pos_sb, indexer_ring_size_i)
            indexer_state_rows = nl.add(
                nl.multiply(owner_sb, indexer_ring_size_i),
                indexer_ring_offsets,
            )
            indexer_ape_offsets = nl.mod(pos_sb, indexer_ratio_i)
            indexer_kv_sb = nl.load(indexer_kv_new[t0 : t0 + cur])
            indexer_score_sb = nl.load(indexer_score_new[t0 : t0 + cur])
            indexer_ape_sb = nl.load(
                indexer_ape[indexer_ape_offsets[i_p, 0], i_indexer_state],
            )
            indexer_score_ape = nl.add(indexer_score_sb, indexer_ape_sb)
            nl.store(
                dst=indexer_kv_score_state[
                    indexer_state_rows[i_p, 0],
                    i_indexer_state,
                ],
                value=indexer_kv_sb[i_p, i_indexer_state],
            )
            nl.store(
                dst=indexer_kv_score_state[
                    indexer_state_rows[i_p, 0],
                    indexer_width + i_indexer_state,
                ],
                value=indexer_score_ape[i_p, i_indexer_state],
            )

            indexer_pos_base = nl.subtract(
                pos_sb,
                nl.mod(pos_sb, indexer_ratio_i),
            )
            indexer_cpos_f = nl.ndarray(
                indexer_pos_base.shape,
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            indexer_cpos_f[...] = nisa.tensor_scalar(
                data=indexer_pos_base,
                op0=nl.multiply,
                operand0=indexer_inv_ratio,
                dtype=nl.float32,
            )
            indexer_cpos = nl.ndarray(
                indexer_pos_base.shape,
                dtype=np.int32,
                buffer=nl.sbuf,
            )
            indexer_cpos[...] = nl.copy(indexer_cpos_f, dtype=nl.int32)
            indexer_cache_rows = nl.add(
                nl.multiply(owner_sb, indexer_max_clen_i),
                indexer_cpos,
            )
            indexer_compressed_sb = nl.load(indexer_compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=indexer_compressed_kv_cache[
                    indexer_cache_rows[i_p, 0],
                    i_indexer_comp,
                ],
                value=indexer_compressed_sb[i_p, i_indexer_comp],
            )
        return (
            swa_kv_cache,
            kv_score_state,
            compressed_kv_cache,
            indexer_kv_score_state,
            indexer_compressed_kv_cache,
        )

    @_nki.jit
    def _write_swa_kv_score_state_owner_pos_kernel(
        swa_kv_cache: "nt.tensor[nt.mutable]",
        kv_score_state: "nt.tensor[nt.mutable]",
        compressed_kv_cache: "nt.tensor[nt.mutable]",
        swa_rows: "nt.tensor",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        compressed_rows: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ape: "nt.tensor",
        window_size: int,
        ring_size: int,
        ratio: int,
        max_clen: int,
    ):
        """In one boundary decode launch, update SWA, state, and cache."""

        n_new, d = swa_rows.shape
        _, width = kv_new.shape
        _, d_comp = compressed_rows.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        window_i = nl.int32(window_size)
        ring_size_i = nl.int32(ring_size)
        ratio_i = nl.int32(ratio)
        max_clen_i = nl.int32(max_clen)
        inv_ratio = np.float32(1.0) / np.float32(ratio)

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_swa = nl.arange(d)[None, :]
            i_comp = nl.arange(d_comp)[None, :]
            i_state = nl.arange(width)[None, :]

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])

            pos_in_window = nl.mod(pos_sb, window_i)
            swa_cache_rows = nl.add(nl.multiply(owner_sb, window_i), pos_in_window)
            swa_sb = nl.load(swa_rows[t0 : t0 + cur])
            nl.store(
                dst=swa_kv_cache[swa_cache_rows[i_p, 0], i_swa],
                value=swa_sb[i_p, i_swa],
            )

            ring_offsets = nl.mod(pos_sb, ring_size_i)
            state_rows = nl.add(
                nl.multiply(owner_sb, ring_size_i),
                ring_offsets,
            )
            ape_offsets = nl.mod(pos_sb, ratio_i)
            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_state])
            score_ape = nl.add(score_sb, ape_sb)
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], i_state],
                value=kv_sb[i_p, i_state],
            )
            nl.store(
                dst=kv_score_state[state_rows[i_p, 0], width + i_state],
                value=score_ape[i_p, i_state],
            )

            pos_base = nl.subtract(pos_sb, nl.mod(pos_sb, ratio_i))
            cpos_f = nl.ndarray(pos_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            cpos_f[...] = nisa.tensor_scalar(
                data=pos_base,
                op0=nl.multiply,
                operand0=inv_ratio,
                dtype=nl.float32,
            )
            cpos = nl.ndarray(pos_base.shape, dtype=np.int32, buffer=nl.sbuf)
            cpos[...] = nl.copy(cpos_f, dtype=nl.int32)
            cache_rows = nl.add(nl.multiply(owner_sb, max_clen_i), cpos)
            compressed_sb = nl.load(compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=compressed_kv_cache[cache_rows[i_p, 0], i_comp],
                value=compressed_sb[i_p, i_comp],
            )
        return swa_kv_cache, kv_score_state, compressed_kv_cache

    @_nki.jit
    def _write_swa_kv_score_state_owner_clen_kernel(
        swa_kv_cache: "nt.tensor[nt.mutable]",
        kv_score_state: "nt.tensor[nt.mutable]",
        compressed_kv_cache: "nt.tensor[nt.mutable]",
        swa_rows: "nt.tensor",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        compressed_rows: "nt.tensor",
        swa_owner_ids: "nt.tensor",
        swa_positions: "nt.tensor",
        state_owner_ids: "nt.tensor",
        state_positions: "nt.tensor",
        cache_owner_ids: "nt.tensor",
        cache_real_clen: "nt.tensor",
        ape: "nt.tensor",
        window_size: int,
        ring_size: int,
        clen: int,
        owner_id_stride: int,
        max_clen: int,
        guard_owner: int,
    ):
        """In one prefill launch, update SWA, ring state, and compressed cache."""

        n_swa, d_swa = swa_rows.shape
        n_state, width = kv_new.shape
        n_cache, d_cache = compressed_rows.shape
        ratio = ape.shape[0]
        MAX_T = 128

        swa_tiles = (n_swa + MAX_T - 1) // MAX_T
        swa_last = n_swa - (swa_tiles - 1) * MAX_T
        state_tiles = (n_state + MAX_T - 1) // MAX_T
        state_last = n_state - (state_tiles - 1) * MAX_T
        cache_tiles = (n_cache + MAX_T - 1) // MAX_T
        cache_last = n_cache - (cache_tiles - 1) * MAX_T

        swa_owner_2d = swa_owner_ids.reshape((n_swa, 1))
        swa_pos_2d = swa_positions.reshape((n_swa, 1))
        state_owner_2d = state_owner_ids.reshape((n_state, 1))
        state_pos_2d = state_positions.reshape((n_state, 1))
        cache_owner_2d = cache_owner_ids.reshape((cache_owner_ids.shape[0], 1))

        window_i = nl.int32(window_size)
        ring_size_i = nl.int32(ring_size)
        clen_i = nl.int32(clen)
        stride_i = nl.int32(owner_id_stride)
        max_clen_i = nl.int32(max_clen)
        inv_clen = np.float32(1.0) / np.float32(clen)

        for ti in nl.static_range(swa_tiles):
            if ti < swa_tiles - 1:
                cur = MAX_T
            else:
                cur = swa_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d_swa)[None, :]
            owner_sb = nl.load(swa_owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(swa_pos_2d[t0 : t0 + cur])
            pos_in_window = nl.mod(pos_sb, window_i)
            rows = nl.add(nl.multiply(owner_sb, window_i), pos_in_window)
            row_sb = nl.load(swa_rows[t0 : t0 + cur])
            nl.store(
                dst=swa_kv_cache[rows[i_p, 0], i_f],
                value=row_sb[i_p, i_f],
            )

        for ti in nl.static_range(state_tiles):
            if ti < state_tiles - 1:
                cur = MAX_T
            else:
                cur = state_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(width)[None, :]

            owner_sb = nl.load(state_owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(state_pos_2d[t0 : t0 + cur])
            ring_offsets = nl.mod(pos_sb, ring_size_i)
            rows = nl.add(nl.multiply(owner_sb, ring_size_i), ring_offsets)
            ape_offsets = nl.mod(pos_sb, nl.int32(ratio))

            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_f])
            score_ape = nl.add(score_sb, ape_sb)

            nl.store(
                dst=kv_score_state[rows[i_p, 0], i_f],
                value=kv_sb[i_p, i_f],
            )
            nl.store(
                dst=kv_score_state[rows[i_p, 0], width + i_f],
                value=score_ape[i_p, i_f],
            )

        for ti in nl.static_range(cache_tiles):
            if ti < cache_tiles - 1:
                cur = MAX_T
            else:
                cur = cache_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d_cache)[None, :]
            row_ids = nl.add(i_p, nl.int32(t0))
            cpos = nl.mod(row_ids, clen_i)
            req_base = nl.subtract(row_ids, cpos)
            req_f = nl.ndarray(req_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            req_f[...] = nisa.tensor_scalar(
                data=req_base,
                op0=nl.multiply,
                operand0=inv_clen,
                dtype=nl.float32,
            )
            req_ids = nl.ndarray(req_base.shape, dtype=nl.int32, buffer=nl.sbuf)
            req_ids[...] = nl.copy(req_f, dtype=nl.int32)
            owner_offsets = nl.multiply(req_ids, stride_i)
            owner_sb = nl.load(cache_owner_2d[owner_offsets[i_p, 0], 0:1])
            # Bucketed-prefill column mask: padded compressed columns
            # (cpos >= cache_real_clen) carry a real owner but garbage data, so
            # redirect them to the guard owner block (never read). cache_real_clen
            # is a runtime scalar so one NEFF serves every real length in a bucket.
            real_clen_sb = nl.load(cache_real_clen[0:1, 0:1])
            col_valid = nl.less(cpos, real_clen_sb)
            guard_owner_full = nl.full(owner_sb.shape, guard_owner, dtype=nl.int32)
            safe_owner = nl.where(col_valid, owner_sb, guard_owner_full)
            rows = nl.add(nl.multiply(safe_owner, max_clen_i), cpos)
            row_sb = nl.load(compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=compressed_kv_cache[rows[i_p, 0], i_f],
                value=row_sb[i_p, i_f],
            )
        return swa_kv_cache, kv_score_state, compressed_kv_cache

    @_nki.jit
    def _write_swa_dual_kv_score_state_owner_clen_kernel(
        swa_kv_cache: "nt.tensor[nt.mutable]",
        kv_score_state: "nt.tensor[nt.mutable]",
        compressed_kv_cache: "nt.tensor[nt.mutable]",
        indexer_kv_score_state: "nt.tensor[nt.mutable]",
        indexer_compressed_kv_cache: "nt.tensor[nt.mutable]",
        swa_rows: "nt.tensor",
        kv_new: "nt.tensor",
        score_new: "nt.tensor",
        compressed_rows: "nt.tensor",
        indexer_kv_new: "nt.tensor",
        indexer_score_new: "nt.tensor",
        indexer_compressed_rows: "nt.tensor",
        swa_owner_ids: "nt.tensor",
        swa_positions: "nt.tensor",
        state_owner_ids: "nt.tensor",
        state_positions: "nt.tensor",
        cache_owner_ids: "nt.tensor",
        ape: "nt.tensor",
        indexer_ape: "nt.tensor",
        cache_real_clen: "nt.tensor",
        window_size: int,
        ring_size: int,
        indexer_ring_size: int,
        clen: int,
        owner_id_stride: int,
        max_clen: int,
        indexer_max_clen: int,
        guard_owner: int,
    ):
        """Prefill update: SWA plus main/indexer compressor state and cache.

        When compiled at a token bucket whose ``clen`` (compressed columns per
        request) exceeds the real prompt's compressed length, the cache loop is
        padded with garbage columns ``cpos >= cache_real_clen``. ``cache_real_clen``
        is a runtime ``[1, 1]`` int32 scalar (NOT in the kernel cache key, so one
        NEFF serves every real length in a bucket); padded columns are redirected
        to the ``guard_owner`` block, which is never read. SWA/state padding is
        handled host-side by setting padded rows' owner to ``guard_owner``.
        """

        n_swa, d_swa = swa_rows.shape
        n_state, width = kv_new.shape
        n_cache, d_cache = compressed_rows.shape
        _, indexer_width = indexer_kv_new.shape
        _, indexer_d_cache = indexer_compressed_rows.shape
        ratio = ape.shape[0]
        indexer_ratio = indexer_ape.shape[0]
        MAX_T = 128

        swa_tiles = (n_swa + MAX_T - 1) // MAX_T
        swa_last = n_swa - (swa_tiles - 1) * MAX_T
        state_tiles = (n_state + MAX_T - 1) // MAX_T
        state_last = n_state - (state_tiles - 1) * MAX_T
        cache_tiles = (n_cache + MAX_T - 1) // MAX_T
        cache_last = n_cache - (cache_tiles - 1) * MAX_T

        swa_owner_2d = swa_owner_ids.reshape((n_swa, 1))
        swa_pos_2d = swa_positions.reshape((n_swa, 1))
        state_owner_2d = state_owner_ids.reshape((n_state, 1))
        state_pos_2d = state_positions.reshape((n_state, 1))
        cache_owner_2d = cache_owner_ids.reshape((cache_owner_ids.shape[0], 1))

        window_i = nl.int32(window_size)
        ring_size_i = nl.int32(ring_size)
        indexer_ring_size_i = nl.int32(indexer_ring_size)
        clen_i = nl.int32(clen)
        stride_i = nl.int32(owner_id_stride)
        max_clen_i = nl.int32(max_clen)
        indexer_max_clen_i = nl.int32(indexer_max_clen)
        inv_clen = np.float32(1.0) / np.float32(clen)

        for ti in nl.static_range(swa_tiles):
            if ti < swa_tiles - 1:
                cur = MAX_T
            else:
                cur = swa_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d_swa)[None, :]
            owner_sb = nl.load(swa_owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(swa_pos_2d[t0 : t0 + cur])
            pos_in_window = nl.mod(pos_sb, window_i)
            rows = nl.add(nl.multiply(owner_sb, window_i), pos_in_window)
            row_sb = nl.load(swa_rows[t0 : t0 + cur])
            nl.store(
                dst=swa_kv_cache[rows[i_p, 0], i_f],
                value=row_sb[i_p, i_f],
            )

        for ti in nl.static_range(state_tiles):
            if ti < state_tiles - 1:
                cur = MAX_T
            else:
                cur = state_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(width)[None, :]
            i_idx = nl.arange(indexer_width)[None, :]

            owner_sb = nl.load(state_owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(state_pos_2d[t0 : t0 + cur])

            ring_offsets = nl.mod(pos_sb, ring_size_i)
            rows = nl.add(nl.multiply(owner_sb, ring_size_i), ring_offsets)
            ape_offsets = nl.mod(pos_sb, nl.int32(ratio))
            kv_sb = nl.load(kv_new[t0 : t0 + cur])
            score_sb = nl.load(score_new[t0 : t0 + cur])
            ape_sb = nl.load(ape[ape_offsets[i_p, 0], i_f])
            score_ape = nl.add(score_sb, ape_sb)
            nl.store(
                dst=kv_score_state[rows[i_p, 0], i_f],
                value=kv_sb[i_p, i_f],
            )
            nl.store(
                dst=kv_score_state[rows[i_p, 0], width + i_f],
                value=score_ape[i_p, i_f],
            )

            indexer_ring_offsets = nl.mod(pos_sb, indexer_ring_size_i)
            indexer_rows = nl.add(
                nl.multiply(owner_sb, indexer_ring_size_i),
                indexer_ring_offsets,
            )
            indexer_ape_offsets = nl.mod(pos_sb, nl.int32(indexer_ratio))
            indexer_kv_sb = nl.load(indexer_kv_new[t0 : t0 + cur])
            indexer_score_sb = nl.load(indexer_score_new[t0 : t0 + cur])
            indexer_ape_sb = nl.load(
                indexer_ape[indexer_ape_offsets[i_p, 0], i_idx],
            )
            indexer_score_ape = nl.add(indexer_score_sb, indexer_ape_sb)
            nl.store(
                dst=indexer_kv_score_state[indexer_rows[i_p, 0], i_idx],
                value=indexer_kv_sb[i_p, i_idx],
            )
            nl.store(
                dst=indexer_kv_score_state[
                    indexer_rows[i_p, 0],
                    indexer_width + i_idx,
                ],
                value=indexer_score_ape[i_p, i_idx],
            )

        for ti in nl.static_range(cache_tiles):
            if ti < cache_tiles - 1:
                cur = MAX_T
            else:
                cur = cache_last
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d_cache)[None, :]
            i_idx = nl.arange(indexer_d_cache)[None, :]
            row_ids = nl.add(i_p, nl.int32(t0))
            cpos = nl.mod(row_ids, clen_i)
            req_base = nl.subtract(row_ids, cpos)
            req_f = nl.ndarray(req_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            req_f[...] = nisa.tensor_scalar(
                data=req_base,
                op0=nl.multiply,
                operand0=inv_clen,
                dtype=nl.float32,
            )
            req_ids = nl.ndarray(req_base.shape, dtype=nl.int32, buffer=nl.sbuf)
            req_ids[...] = nl.copy(req_f, dtype=nl.int32)
            owner_offsets = nl.multiply(req_ids, stride_i)
            owner_sb = nl.load(cache_owner_2d[owner_offsets[i_p, 0], 0:1])

            # Bucketed-prefill column mask: padded compressed columns
            # (cpos >= cache_real_clen) carry a real owner but garbage data, so
            # redirect them to the guard owner block (never read). cache_real_clen
            # is a runtime scalar so one NEFF serves every real length in a bucket.
            real_clen_sb = nl.load(cache_real_clen[0:1, 0:1])
            col_valid = nl.less(cpos, real_clen_sb)
            guard_owner_full = nl.full(owner_sb.shape, guard_owner, dtype=nl.int32)
            safe_owner = nl.where(col_valid, owner_sb, guard_owner_full)

            rows = nl.add(nl.multiply(safe_owner, max_clen_i), cpos)
            row_sb = nl.load(compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=compressed_kv_cache[rows[i_p, 0], i_f],
                value=row_sb[i_p, i_f],
            )

            indexer_rows = nl.add(
                nl.multiply(safe_owner, indexer_max_clen_i),
                cpos,
            )
            indexer_row_sb = nl.load(indexer_compressed_rows[t0 : t0 + cur])
            nl.store(
                dst=indexer_compressed_kv_cache[indexer_rows[i_p, 0], i_idx],
                value=indexer_row_sb[i_p, i_idx],
            )
        return (
            swa_kv_cache,
            kv_score_state,
            compressed_kv_cache,
            indexer_kv_score_state,
            indexer_compressed_kv_cache,
        )

    def _make_decode_pool_from_state_kernel(
        *,
        ratio: int,
        head_dim: int,
        state_width: int,
        ring_size: int,
        overlap: bool,
    ):
        ratio_i = int(ratio)
        d_i = int(head_dim)
        width_i = int(state_width)
        ring_i = int(ring_size)
        overlap_b = bool(overlap)
        d_tile = 128
        d_tiles = (d_i + d_tile - 1) // d_tile
        last_d = d_i - (d_tiles - 1) * d_tile

        @_nki.jit
        def decode_pool_from_state_kernel(
            kv_score_state: "nt.tensor",
            owner_ids: "nt.tensor",
            end_positions: "nt.tensor",
        ):
            """Pool compressed KV from flat ring-addressed state."""

            bsz = owner_ids.shape[0]
            owner_2d = owner_ids.reshape((bsz, 1))
            pos_2d = end_positions.reshape((bsz, 1))
            out = nl.ndarray((bsz, d_i), dtype=nl.float32, buffer=nl.shared_hbm)

            for b in nl.affine_range(bsz):
                owner_sb = nl.load(owner_2d[b : b + 1, 0:1])
                end_sb = nl.load(pos_2d[b : b + 1, 0:1])
                base_row = nl.multiply(owner_sb, nl.int32(ring_i))
                for d_tile_i in nl.static_range(d_tiles):
                    if d_tile_i < d_tiles - 1:
                        cur_d = d_tile
                    else:
                        cur_d = last_d
                    d0 = d_tile_i * d_tile
                    i_f = nl.arange(cur_d)[None, :]

                    if overlap_b:
                        i_k = nl.arange(ratio_i)[:, None]
                        pos_prev = nl.add(
                            nl.broadcast_to(end_sb, shape=(ratio_i, 1)),
                            nl.add(i_k, nl.int32(1 - 2 * ratio_i)),
                        )
                        pos_cur = nl.add(
                            nl.broadcast_to(end_sb, shape=(ratio_i, 1)),
                            nl.add(i_k, nl.int32(1 - ratio_i)),
                        )
                        valid_prev = nl.greater_equal(pos_prev, nl.int32(0))
                        safe_prev = nl.maximum(pos_prev, nl.int32(0))
                        rows_prev = nl.add(
                            base_row,
                            nl.mod(safe_prev, nl.int32(ring_i)),
                        )
                        rows_cur = nl.add(base_row, nl.mod(pos_cur, nl.int32(ring_i)))

                        kv_prev_g = nl.load(kv_score_state[rows_prev[i_k, 0], d0 + i_f])
                        score_prev_g = nl.load(
                            kv_score_state[rows_prev[i_k, 0], width_i + d0 + i_f]
                        )
                        valid_prev_b = nl.broadcast_to(
                            valid_prev,
                            shape=(ratio_i, cur_d),
                        )
                        kv_prev_f = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_prev_f = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev_f[...] = nl.copy(kv_prev_g, dtype=nl.float32)
                        score_prev_f[...] = nl.copy(score_prev_g, dtype=nl.float32)
                        zero_prev = nl.zeros(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        neg_prev = nl.full(
                            (par_dim(ratio_i), cur_d),
                            -1e9,
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_prev = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev[...] = nl.where(
                            valid_prev_b,
                            kv_prev_f,
                            zero_prev,
                            dtype=nl.float32,
                        )
                        score_prev[...] = nl.where(
                            valid_prev_b,
                            score_prev_f,
                            neg_prev,
                            dtype=nl.float32,
                        )

                        kv_cur_g = nl.load(
                            kv_score_state[rows_cur[i_k, 0], d_i + d0 + i_f]
                        )
                        score_cur_g = nl.load(
                            kv_score_state[rows_cur[i_k, 0], width_i + d_i + d0 + i_f]
                        )

                        kv_prev_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        kv_prev_t_psum[...] = nisa.nc_transpose(
                            kv_prev,
                            engine=nisa.tensor_engine,
                        )
                        kv_prev_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev_t[...] = nl.copy(kv_prev_t_psum)

                        score_prev_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        score_prev_t_psum[...] = nisa.nc_transpose(
                            score_prev,
                            engine=nisa.tensor_engine,
                        )
                        score_prev_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_prev_t[...] = nl.copy(score_prev_t_psum)

                        kv_cur_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        kv_cur_t_psum[...] = nisa.nc_transpose(
                            kv_cur_g,
                            engine=nisa.tensor_engine,
                        )
                        kv_cur_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_cur_t[...] = nl.copy(kv_cur_t_psum)

                        score_cur_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        score_cur_t_psum[...] = nisa.nc_transpose(
                            score_cur_g,
                            engine=nisa.tensor_engine,
                        )
                        score_cur_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_cur_t[...] = nl.copy(score_cur_t_psum)

                        m_prev = nisa.tensor_reduce(
                            np.max,
                            score_prev_t,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        m_cur = nisa.tensor_reduce(
                            np.max,
                            score_cur_t,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        m = nl.maximum(m_prev, m_cur)
                        neg_m = nisa.activation(nl.copy, m, scale=-1.0)

                        p_prev = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        denom_prev = nl.ndarray(
                            (par_dim(cur_d), 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        p_prev[...] = nisa.activation_reduce(
                            np.exp,
                            score_prev_t,
                            bias=neg_m,
                            scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=denom_prev,
                            dtype=nl.float32,
                        )
                        p_cur = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        denom_cur = nl.ndarray(
                            (par_dim(cur_d), 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        p_cur[...] = nisa.activation_reduce(
                            np.exp,
                            score_cur_t,
                            bias=neg_m,
                            scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=denom_cur,
                            dtype=nl.float32,
                        )

                        weighted_prev = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        weighted_prev[...] = nl.multiply(kv_prev_t, p_prev)
                        weighted_cur = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        weighted_cur[...] = nl.multiply(kv_cur_t, p_cur)
                        numer_prev = nisa.tensor_reduce(
                            nl.add,
                            weighted_prev,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        numer_cur = nisa.tensor_reduce(
                            nl.add,
                            weighted_cur,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        numer = nl.add(numer_prev, numer_cur)
                        denom = nl.add(denom_prev, denom_cur)
                        pooled = nl.divide(numer, denom)
                    else:
                        i_k = nl.arange(ratio_i)[:, None]
                        pos = nl.add(
                            nl.broadcast_to(end_sb, shape=(ratio_i, 1)),
                            nl.add(i_k, nl.int32(1 - ratio_i)),
                        )
                        rows = nl.add(base_row, nl.mod(pos, nl.int32(ring_i)))
                        kv_g = nl.load(kv_score_state[rows[i_k, 0], d0 + i_f])
                        score_g = nl.load(
                            kv_score_state[rows[i_k, 0], width_i + d0 + i_f]
                        )

                        kv_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        kv_t_psum[...] = nisa.nc_transpose(
                            kv_g,
                            engine=nisa.tensor_engine,
                        )
                        kv_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_t[...] = nl.copy(kv_t_psum)

                        score_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        score_t_psum[...] = nisa.nc_transpose(
                            score_g,
                            engine=nisa.tensor_engine,
                        )
                        score_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_t[...] = nl.copy(score_t_psum)

                        m = nisa.tensor_reduce(
                            np.max,
                            score_t,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        neg_m = nisa.activation(nl.copy, m, scale=-1.0)
                        p = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        denom = nl.ndarray(
                            (par_dim(cur_d), 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        p[...] = nisa.activation_reduce(
                            np.exp,
                            score_t,
                            bias=neg_m,
                            scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=denom,
                            dtype=nl.float32,
                        )
                        weighted = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        weighted[...] = nl.multiply(kv_t, p)
                        numer = nisa.tensor_reduce(
                            nl.add,
                            weighted,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        pooled = nl.divide(numer, denom)

                    pooled_t_psum = nl.ndarray(
                        (1, cur_d),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    pooled_t_psum[...] = nisa.nc_transpose(
                        pooled,
                        engine=nisa.tensor_engine,
                    )
                    pooled_t = nl.ndarray(
                        (1, cur_d),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    pooled_t[...] = nl.copy(pooled_t_psum)
                    nl.store(out[b : b + 1, d0 : d0 + cur_d], pooled_t)
            return out

        return decode_pool_from_state_kernel

    def _make_prefill_pool_from_slab_kernel(
        *,
        ratio: int,
        head_dim: int,
        state_width: int,
        overlap: bool,
    ):
        ratio_i = int(ratio)
        d_i = int(head_dim)
        overlap_b = bool(overlap)
        d_tile = 128
        d_tiles = (d_i + d_tile - 1) // d_tile
        last_d = d_i - (d_tiles - 1) * d_tile

        @_nki.jit
        def prefill_pool_from_slab_kernel(
            kv_new: "nt.tensor",
            score_new: "nt.tensor",
            ape: "nt.tensor",
            base_rows: "nt.tensor",
            prev_rows: "nt.tensor",
            prev_valid: "nt.tensor",
        ):
            """Pool prefill compressed KV from flat projection slabs."""

            out_rows = base_rows.shape[0]
            base_2d = base_rows.reshape((out_rows, 1))
            prev_2d = prev_rows.reshape((out_rows, 1))
            valid_2d = prev_valid.reshape((out_rows, 1))
            out = nl.ndarray(
                (out_rows, d_i),
                dtype=nl.float32,
                buffer=nl.shared_hbm,
            )

            for row_i in nl.sequential_range(out_rows):
                base_sb = nl.load(base_2d[row_i : row_i + 1, 0:1])
                for d_tile_i in nl.static_range(d_tiles):
                    if d_tile_i < d_tiles - 1:
                        cur_d = d_tile
                    else:
                        cur_d = last_d
                    d0 = d_tile_i * d_tile
                    i_f = nl.arange(cur_d)[None, :]

                    if overlap_b:
                        prev_sb = nl.load(prev_2d[row_i : row_i + 1, 0:1])
                        valid_sb = nl.load(valid_2d[row_i : row_i + 1, 0:1])
                        i_k = nl.arange(ratio_i)[:, None]
                        prev_idx = nl.add(
                            nl.broadcast_to(prev_sb, shape=(ratio_i, 1)),
                            i_k,
                        )
                        cur_idx = nl.add(
                            nl.broadcast_to(base_sb, shape=(ratio_i, 1)),
                            i_k,
                        )

                        kv_prev_g = nl.load(kv_new[prev_idx[i_k, 0], d0 + i_f])
                        score_prev_g = nl.add(
                            nl.load(score_new[prev_idx[i_k, 0], d0 + i_f]),
                            nl.load(ape[i_k, d0 + i_f]),
                        )
                        valid_b = nl.broadcast_to(valid_sb, shape=(ratio_i, cur_d))
                        kv_prev_f = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_prev_f = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev_f[...] = nl.copy(kv_prev_g, dtype=nl.float32)
                        score_prev_f[...] = nl.copy(score_prev_g, dtype=nl.float32)
                        zero_prev = nl.zeros(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        neg_prev = nl.full(
                            (par_dim(ratio_i), cur_d),
                            -1e9,
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_prev = nl.ndarray(
                            (par_dim(ratio_i), cur_d),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev[...] = nl.where(
                            valid_b,
                            kv_prev_f,
                            zero_prev,
                            dtype=nl.float32,
                        )
                        score_prev[...] = nl.where(
                            valid_b,
                            score_prev_f,
                            neg_prev,
                            dtype=nl.float32,
                        )

                        kv_cur_g = nl.load(kv_new[cur_idx[i_k, 0], d_i + d0 + i_f])
                        score_cur_g = nl.add(
                            nl.load(score_new[cur_idx[i_k, 0], d_i + d0 + i_f]),
                            nl.load(ape[i_k, d_i + d0 + i_f]),
                        )

                        kv_prev_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        kv_prev_t_psum[...] = nisa.nc_transpose(
                            kv_prev,
                            engine=nisa.tensor_engine,
                        )
                        kv_prev_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_prev_t[...] = nl.copy(kv_prev_t_psum)

                        score_prev_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        score_prev_t_psum[...] = nisa.nc_transpose(
                            score_prev,
                            engine=nisa.tensor_engine,
                        )
                        score_prev_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_prev_t[...] = nl.copy(score_prev_t_psum)

                        kv_cur_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        kv_cur_t_psum[...] = nisa.nc_transpose(
                            kv_cur_g,
                            engine=nisa.tensor_engine,
                        )
                        kv_cur_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_cur_t[...] = nl.copy(kv_cur_t_psum)

                        score_cur_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        score_cur_t_psum[...] = nisa.nc_transpose(
                            score_cur_g,
                            engine=nisa.tensor_engine,
                        )
                        score_cur_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_cur_t[...] = nl.copy(score_cur_t_psum)

                        m_prev = nisa.tensor_reduce(
                            np.max,
                            score_prev_t,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        m_cur = nisa.tensor_reduce(
                            np.max,
                            score_cur_t,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        m = nl.maximum(m_prev, m_cur)
                        neg_m = nisa.activation(nl.copy, m, scale=-1.0)

                        p_prev = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        denom_prev = nl.ndarray(
                            (par_dim(cur_d), 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        p_prev[...] = nisa.activation_reduce(
                            np.exp,
                            score_prev_t,
                            bias=neg_m,
                            scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=denom_prev,
                            dtype=nl.float32,
                        )
                        p_cur = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        denom_cur = nl.ndarray(
                            (par_dim(cur_d), 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        p_cur[...] = nisa.activation_reduce(
                            np.exp,
                            score_cur_t,
                            bias=neg_m,
                            scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=denom_cur,
                            dtype=nl.float32,
                        )
                        weighted_prev = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        weighted_prev[...] = nl.multiply(kv_prev_t, p_prev)
                        weighted_cur = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        weighted_cur[...] = nl.multiply(kv_cur_t, p_cur)
                        numer_prev = nisa.tensor_reduce(
                            nl.add,
                            weighted_prev,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        numer_cur = nisa.tensor_reduce(
                            nl.add,
                            weighted_cur,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        numer = nl.add(numer_prev, numer_cur)
                        denom = nl.add(denom_prev, denom_cur)
                        pooled = nl.divide(numer, denom)
                    else:
                        i_k = nl.arange(ratio_i)[:, None]
                        rows = nl.add(
                            nl.broadcast_to(base_sb, shape=(ratio_i, 1)),
                            i_k,
                        )
                        kv_g = nl.load(kv_new[rows[i_k, 0], d0 + i_f])
                        score_g = nl.add(
                            nl.load(score_new[rows[i_k, 0], d0 + i_f]),
                            nl.load(ape[i_k, d0 + i_f]),
                        )

                        kv_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        kv_t_psum[...] = nisa.nc_transpose(
                            kv_g,
                            engine=nisa.tensor_engine,
                        )
                        kv_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        kv_t[...] = nl.copy(kv_t_psum)

                        score_t_psum = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        score_t_psum[...] = nisa.nc_transpose(
                            score_g,
                            engine=nisa.tensor_engine,
                        )
                        score_t = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        score_t[...] = nl.copy(score_t_psum)

                        m = nisa.tensor_reduce(
                            np.max,
                            score_t,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        neg_m = nisa.activation(nl.copy, m, scale=-1.0)
                        p = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        denom = nl.ndarray(
                            (par_dim(cur_d), 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        p[...] = nisa.activation_reduce(
                            np.exp,
                            score_t,
                            bias=neg_m,
                            scale=1.0,
                            reduce_op=nl.add,
                            reduce_res=denom,
                            dtype=nl.float32,
                        )
                        weighted = nl.ndarray(
                            (par_dim(cur_d), ratio_i),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        weighted[...] = nl.multiply(kv_t, p)
                        numer = nisa.tensor_reduce(
                            nl.add,
                            weighted,
                            axis=(1,),
                            dtype=nl.float32,
                            negate=False,
                        )
                        pooled = nl.divide(numer, denom)

                    pooled_t_psum = nl.ndarray(
                        (1, cur_d),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    pooled_t_psum[...] = nisa.nc_transpose(
                        pooled,
                        engine=nisa.tensor_engine,
                    )
                    pooled_t = nl.ndarray(
                        (1, cur_d),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    pooled_t[...] = nl.copy(pooled_t_psum)
                    nl.store(out[row_i : row_i + 1, d0 : d0 + cur_d], pooled_t)
            return out

        return prefill_pool_from_slab_kernel


def _write_kv_score_state_entry(
    kv_score_state,
    kv_new,
    score_new,
    owner_ids,
    positions,
    ape,
    live_rows,
    *,
    ring_size: int,
    guard_owner: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_score_state_kernel(
        kv_score_state,
        kv_new,
        score_new,
        owner_ids,
        positions,
        ape,
        live_rows,
        int(ring_size),
        int(guard_owner),
    )


def _write_kv_score_state_owner_pos_entry(
    kv_score_state,
    compressed_kv_cache,
    kv_new,
    score_new,
    compressed_rows,
    owner_ids,
    positions,
    ape,
    *,
    ring_size: int,
    ratio: int,
    max_clen: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_score_state_owner_pos_kernel(
        kv_score_state,
        compressed_kv_cache,
        kv_new,
        score_new,
        compressed_rows,
        owner_ids,
        positions,
        ape,
        int(ring_size),
        int(ratio),
        int(max_clen),
    )


def _write_kv_score_state_owner_clen_entry(
    kv_score_state,
    compressed_kv_cache,
    kv_new,
    score_new,
    compressed_rows,
    state_owner_ids,
    state_positions,
    cache_owner_ids,
    ape,
    *,
    ring_size: int,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_score_state_owner_clen_kernel(
        kv_score_state,
        compressed_kv_cache,
        kv_new,
        score_new,
        compressed_rows,
        state_owner_ids,
        state_positions,
        cache_owner_ids,
        ape,
        int(ring_size),
        int(clen),
        int(owner_id_stride),
        int(max_clen),
    )


def _write_swa_kv_score_state_entry(
    swa_kv_cache,
    kv_score_state,
    swa_rows,
    kv_new,
    score_new,
    owner_ids,
    positions,
    ape,
    *,
    window_size: int,
    ring_size: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_swa_kv_score_state_kernel(
        swa_kv_cache,
        kv_score_state,
        swa_rows,
        kv_new,
        score_new,
        owner_ids,
        positions,
        ape,
        int(window_size),
        int(ring_size),
    )


def _write_swa_dual_kv_score_state_entry(
    swa_kv_cache,
    kv_score_state,
    indexer_kv_score_state,
    swa_rows,
    kv_new,
    score_new,
    indexer_kv_new,
    indexer_score_new,
    owner_ids,
    positions,
    ape,
    indexer_ape,
    live_rows,
    *,
    window_size: int,
    ring_size: int,
    indexer_ring_size: int,
    guard_owner: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_swa_dual_kv_score_state_kernel(
        swa_kv_cache,
        kv_score_state,
        indexer_kv_score_state,
        swa_rows,
        kv_new,
        score_new,
        indexer_kv_new,
        indexer_score_new,
        owner_ids,
        positions,
        ape,
        indexer_ape,
        live_rows,
        int(window_size),
        int(ring_size),
        int(indexer_ring_size),
        int(guard_owner),
    )


def _write_swa_dual_kv_score_state_owner_pos_entry(
    swa_kv_cache,
    kv_score_state,
    compressed_kv_cache,
    indexer_kv_score_state,
    indexer_compressed_kv_cache,
    swa_rows,
    kv_new,
    score_new,
    compressed_rows,
    indexer_kv_new,
    indexer_score_new,
    indexer_compressed_rows,
    owner_ids,
    positions,
    ape,
    indexer_ape,
    *,
    window_size: int,
    ring_size: int,
    indexer_ring_size: int,
    ratio: int,
    indexer_ratio: int,
    max_clen: int,
    indexer_max_clen: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_swa_dual_kv_score_state_owner_pos_kernel(
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        indexer_kv_score_state,
        indexer_compressed_kv_cache,
        swa_rows,
        kv_new,
        score_new,
        compressed_rows,
        indexer_kv_new,
        indexer_score_new,
        indexer_compressed_rows,
        owner_ids,
        positions,
        ape,
        indexer_ape,
        int(window_size),
        int(ring_size),
        int(indexer_ring_size),
        int(ratio),
        int(indexer_ratio),
        int(max_clen),
        int(indexer_max_clen),
    )


def _write_swa_dual_kv_score_state_owner_clen_entry(
    swa_kv_cache,
    kv_score_state,
    compressed_kv_cache,
    indexer_kv_score_state,
    indexer_compressed_kv_cache,
    swa_rows,
    kv_new,
    score_new,
    compressed_rows,
    indexer_kv_new,
    indexer_score_new,
    indexer_compressed_rows,
    swa_owner_ids,
    swa_positions,
    state_owner_ids,
    state_positions,
    cache_owner_ids,
    ape,
    indexer_ape,
    cache_real_clen,
    *,
    window_size: int,
    ring_size: int,
    indexer_ring_size: int,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
    indexer_max_clen: int,
    guard_owner: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_swa_dual_kv_score_state_owner_clen_kernel(
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        indexer_kv_score_state,
        indexer_compressed_kv_cache,
        swa_rows,
        kv_new,
        score_new,
        compressed_rows,
        indexer_kv_new,
        indexer_score_new,
        indexer_compressed_rows,
        swa_owner_ids,
        swa_positions,
        state_owner_ids,
        state_positions,
        cache_owner_ids,
        ape,
        indexer_ape,
        cache_real_clen,
        int(window_size),
        int(ring_size),
        int(indexer_ring_size),
        int(clen),
        int(owner_id_stride),
        int(max_clen),
        int(indexer_max_clen),
        int(guard_owner),
    )


def _write_swa_kv_score_state_owner_pos_entry(
    swa_kv_cache,
    kv_score_state,
    compressed_kv_cache,
    swa_rows,
    kv_new,
    score_new,
    compressed_rows,
    owner_ids,
    positions,
    ape,
    *,
    window_size: int,
    ring_size: int,
    ratio: int,
    max_clen: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_swa_kv_score_state_owner_pos_kernel(
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        swa_rows,
        kv_new,
        score_new,
        compressed_rows,
        owner_ids,
        positions,
        ape,
        int(window_size),
        int(ring_size),
        int(ratio),
        int(max_clen),
    )


def _write_swa_kv_score_state_owner_clen_entry(
    swa_kv_cache,
    kv_score_state,
    compressed_kv_cache,
    swa_rows,
    kv_new,
    score_new,
    compressed_rows,
    swa_owner_ids,
    swa_positions,
    state_owner_ids,
    state_positions,
    cache_owner_ids,
    ape,
    cache_real_clen,
    *,
    window_size: int,
    ring_size: int,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
    guard_owner: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_swa_kv_score_state_owner_clen_kernel(
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        swa_rows,
        kv_new,
        score_new,
        compressed_rows,
        swa_owner_ids,
        swa_positions,
        state_owner_ids,
        state_positions,
        cache_owner_ids,
        cache_real_clen,
        ape,
        int(window_size),
        int(ring_size),
        int(clen),
        int(owner_id_stride),
        int(max_clen),
        int(guard_owner),
    )


def _make_decode_pool_from_state_entry(
    *,
    ratio: int,
    head_dim: int,
    state_width: int,
    ring_size: int,
    overlap: bool,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    kernel = _make_decode_pool_from_state_kernel(
        ratio=ratio,
        head_dim=head_dim,
        state_width=state_width,
        ring_size=ring_size,
        overlap=overlap,
    )

    def _entry(kv_score_state, owner_ids, end_positions):
        return kernel(kv_score_state, owner_ids, end_positions)

    return _entry


def _make_prefill_pool_from_slab_entry(
    *,
    ratio: int,
    head_dim: int,
    state_width: int,
    overlap: bool,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    kernel = _make_prefill_pool_from_slab_kernel(
        ratio=ratio,
        head_dim=head_dim,
        state_width=state_width,
        overlap=overlap,
    )

    def _entry(kv_new, score_new, ape, base_rows, prev_rows, prev_valid):
        return kernel(kv_new, score_new, ape, base_rows, prev_rows, prev_valid)

    return _entry


def run_write_kv_score_state_device(
    *,
    kv_score_state: Any,
    kv_new: Any,
    score_new: Any,
    owner_ids: Any,
    positions: Any,
    ape: Any,
    live_rows: Any | None = None,
    ring_size: int,
    guard_owner: int | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Device ring scatter for compressor ``kv_score_state``.

    This kernel is deliberately narrow: it only mutates state.  Pool/post/cache
    write fusion is the next kernel layer and can reuse the same flat ring
    layout.
    """

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    if live_rows is None:
        live_rows = _device_scalar_i32(
            int(kv_shape[0]),
            name="dsv4_write_kv_score_state_live_rows",
        )
    live_shape = tuple(int(dim) for dim in getattr(live_rows, "shape"))
    if len(state_shape) != 2 or len(kv_shape) != 2:
        raise ValueError(f"bad state/kv shapes: {state_shape}/{kv_shape}")
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    n_new, width = kv_shape
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if owner_shape != (n_new,) or pos_shape != (n_new,):
        raise ValueError(
            f"owner_ids/positions must be [{n_new}], got {owner_shape}/{pos_shape}"
        )
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    if live_shape != (1, 1):
        raise ValueError(f"live_rows must be [1, 1], got {live_shape}")
    if int(ring_size) <= 0 or state_shape[0] % int(ring_size) != 0:
        raise ValueError(
            f"ring_size={ring_size} incompatible with state rows {state_shape[0]}"
        )
    ring_i = int(ring_size)
    if guard_owner is None:
        guard_owner_i = int(state_shape[0]) // ring_i - 1
    else:
        guard_owner_i = int(guard_owner)
    if guard_owner_i < 0 or (guard_owner_i + 1) * ring_i > int(state_shape[0]):
        raise ValueError(
            f"guard_owner={guard_owner_i} outside compressor state "
            f"(rows={state_shape[0]}, ring={ring_i})"
        )
    if n_new == 0:
        return kv_score_state

    cache = (
        _WRITE_KV_SCORE_STATE_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    cache_key = (
        "write_kv_score_state",
        state_shape,
        kv_shape,
        owner_shape,
        pos_shape,
        ape_shape,
        live_shape,
        int(ring_size),
        guard_owner_i,
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(ape)),
        str(_dtype_like(live_rows)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_kv_score_state_entry,
        _sample_like(kv_score_state),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(owner_ids),
        _sample_like(positions),
        _sample_like(ape),
        _sample_like(live_rows),
        ring_size=int(ring_size),
        guard_owner=guard_owner_i,
        name=(
            f"dsv4_write_kv_score_state_nbucket{n_new}_w{width}"
            f"_r{int(ring_size)}_g{guard_owner_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "kv_score_state.must_alias_input": kv_score_state,
            "kv_new": kv_new,
            "score_new": score_new,
            "owner_ids": owner_ids,
            "positions": positions,
            "ape": ape,
            "live_rows": live_rows,
        },
        outputs={"kv_score_state": kv_score_state},
    )
    return kv_score_state


def run_write_kv_score_state_owner_pos_device(
    *,
    kv_score_state: Any,
    compressed_kv_cache: Any,
    kv_new: Any,
    score_new: Any,
    compressed_rows: Any,
    owner_ids: Any,
    positions: Any,
    ape: Any,
    ring_size: int,
    ratio: int,
    max_clen: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any]:
    """Fused decode scatter into compressor ring state and compressed cache."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    rows_shape = tuple(int(dim) for dim in getattr(compressed_rows, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    if len(state_shape) != 2 or len(cache_shape) != 2 or len(kv_shape) != 2:
        raise ValueError(
            f"bad state/cache/kv shapes: {state_shape}/{cache_shape}/{kv_shape}"
        )
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    n_new, width = kv_shape
    if len(rows_shape) != 2 or rows_shape[0] != n_new:
        raise ValueError(f"compressed_rows must be [{n_new}, d], got {rows_shape}")
    if cache_shape[1] != rows_shape[1]:
        raise ValueError(
            "compressed cache/head dim mismatch: "
            f"cache={cache_shape[1]}, rows={rows_shape[1]}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    ring_i = int(ring_size)
    ratio_i = int(ratio)
    max_clen_i = int(max_clen)
    if ring_i <= 0 or state_shape[0] % ring_i != 0:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if ratio_i <= 0 or max_clen_i <= 0:
        raise ValueError(
            f"ratio and max_clen must be positive, got {ratio_i}/{max_clen_i}"
        )
    if n_new == 0:
        return kv_score_state, compressed_kv_cache

    cache = (
        _WRITE_KV_SCORE_STATE_OWNER_POS_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_kv_score_state_owner_pos",
        state_shape,
        cache_shape,
        kv_shape,
        rows_shape,
        owner_shape,
        pos_shape,
        ape_shape,
        ring_i,
        ratio_i,
        max_clen_i,
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(compressed_kv_cache)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(compressed_rows)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_kv_score_state_owner_pos_entry,
        _sample_like(kv_score_state),
        _sample_like(compressed_kv_cache),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(compressed_rows),
        _sample_like(owner_ids),
        _sample_like(positions),
        _sample_like(ape),
        ring_size=ring_i,
        ratio=ratio_i,
        max_clen=max_clen_i,
        name=(
            "dsv4_write_kv_score_owner_pos_"
            f"n{n_new}_w{width}_d{cache_shape[1]}_"
            f"r{ring_i}_cr{ratio_i}_c{max_clen_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "kv_score_state.must_alias_input": kv_score_state,
            "compressed_kv_cache.must_alias_input": compressed_kv_cache,
            "kv_new": kv_new,
            "score_new": score_new,
            "compressed_rows": compressed_rows,
            "owner_ids": owner_ids,
            "positions": positions,
            "ape": ape,
        },
        outputs={
            "kv_score_state": kv_score_state,
            "compressed_kv_cache": compressed_kv_cache,
        },
    )
    return kv_score_state, compressed_kv_cache


def run_write_kv_score_state_owner_clen_device(
    *,
    kv_score_state: Any,
    compressed_kv_cache: Any,
    kv_new: Any,
    score_new: Any,
    compressed_rows: Any,
    state_owner_ids: Any,
    state_positions: Any,
    cache_owner_ids: Any,
    ape: Any,
    ring_size: int,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any]:
    """Fused prefill scatter into compressor ring state and compressed cache."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    rows_shape = tuple(int(dim) for dim in getattr(compressed_rows, "shape"))
    state_owner_shape = tuple(int(dim) for dim in getattr(state_owner_ids, "shape"))
    state_pos_shape = tuple(int(dim) for dim in getattr(state_positions, "shape"))
    cache_owner_shape = tuple(int(dim) for dim in getattr(cache_owner_ids, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    if len(state_shape) != 2 or len(cache_shape) != 2 or len(kv_shape) != 2:
        raise ValueError(
            f"bad state/cache/kv shapes: {state_shape}/{cache_shape}/{kv_shape}"
        )
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    n_state, width = kv_shape
    if len(rows_shape) != 2:
        raise ValueError(f"compressed_rows must be 2-D, got {rows_shape}")
    if cache_shape[1] != rows_shape[1]:
        raise ValueError(
            "compressed cache/head dim mismatch: "
            f"cache={cache_shape[1]}, rows={rows_shape[1]}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if state_owner_shape != (n_state,) or state_pos_shape != (n_state,):
        raise ValueError(
            "state owner/position shapes must match kv rows: "
            f"{state_owner_shape}/{state_pos_shape} vs {n_state}"
        )
    if len(cache_owner_shape) != 1:
        raise ValueError(f"cache_owner_ids must be 1D, got {cache_owner_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    ring_i = int(ring_size)
    clen_i = int(clen)
    stride_i = int(owner_id_stride)
    max_clen_i = int(max_clen)
    if ring_i <= 0 or state_shape[0] % ring_i != 0:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if clen_i <= 0 or stride_i <= 0 or max_clen_i <= 0:
        raise ValueError(
            "clen, owner_id_stride, and max_clen must be positive, got "
            f"{clen_i}/{stride_i}/{max_clen_i}"
        )
    n_cache = int(rows_shape[0])
    if n_cache == 0 or n_state == 0:
        return kv_score_state, compressed_kv_cache
    if n_cache % clen_i:
        raise ValueError(f"compressed rows {n_cache} must be divisible by clen")
    bsz = n_cache // clen_i
    required_owners = (bsz - 1) * stride_i + 1
    if int(cache_owner_shape[0]) < required_owners:
        raise ValueError(
            "cache_owner_ids first dim too small for request-major scatter: "
            f"got {cache_owner_shape[0]}, need >= {required_owners}"
        )

    cache = (
        _WRITE_KV_SCORE_STATE_OWNER_CLEN_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_kv_score_state_owner_clen",
        state_shape,
        cache_shape,
        kv_shape,
        rows_shape,
        state_owner_shape,
        state_pos_shape,
        cache_owner_shape,
        ape_shape,
        ring_i,
        clen_i,
        stride_i,
        max_clen_i,
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(compressed_kv_cache)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(compressed_rows)),
        str(_dtype_like(state_owner_ids)),
        str(_dtype_like(state_positions)),
        str(_dtype_like(cache_owner_ids)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_kv_score_state_owner_clen_entry,
        _sample_like(kv_score_state),
        _sample_like(compressed_kv_cache),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(compressed_rows),
        _sample_like(state_owner_ids),
        _sample_like(state_positions),
        _sample_like(cache_owner_ids),
        _sample_like(ape),
        ring_size=ring_i,
        clen=clen_i,
        owner_id_stride=stride_i,
        max_clen=max_clen_i,
        name=(
            "dsv4_write_kv_score_owner_clen_"
            f"ns{n_state}_nc{n_cache}_w{width}_d{cache_shape[1]}_"
            f"r{ring_i}_clen{clen_i}_stride{stride_i}_c{max_clen_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "kv_score_state.must_alias_input": kv_score_state,
            "compressed_kv_cache.must_alias_input": compressed_kv_cache,
            "kv_new": kv_new,
            "score_new": score_new,
            "compressed_rows": compressed_rows,
            "state_owner_ids": state_owner_ids,
            "state_positions": state_positions,
            "cache_owner_ids": cache_owner_ids,
            "ape": ape,
        },
        outputs={
            "kv_score_state": kv_score_state,
            "compressed_kv_cache": compressed_kv_cache,
        },
    )
    return kv_score_state, compressed_kv_cache


def run_write_swa_kv_score_state_device(
    *,
    swa_kv_cache: Any,
    kv_score_state: Any,
    swa_rows: Any,
    kv_new: Any,
    score_new: Any,
    owner_ids: Any,
    positions: Any,
    ape: Any,
    window_size: int,
    ring_size: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any]:
    """Fused decode scatter into SWA cache and compressor ring state."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    if len(swa_shape) != 2 or len(state_shape) != 2 or len(kv_shape) != 2:
        raise ValueError(
            f"bad swa/state/kv shapes: {swa_shape}/{state_shape}/{kv_shape}"
        )
    if len(rows_shape) != 2:
        raise ValueError(f"swa_rows must be 2-D, got {rows_shape}")
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    n_new, width = kv_shape
    if rows_shape[0] != n_new:
        raise ValueError(f"swa_rows rows {rows_shape[0]} != kv rows {n_new}")
    if swa_shape[1] != rows_shape[1]:
        raise ValueError(
            f"SWA head_dim mismatch: cache={swa_shape[1]}, rows={rows_shape[1]}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    window_i = int(window_size)
    ring_i = int(ring_size)
    if window_i <= 0 or ring_i <= 0:
        raise ValueError(f"window/ring must be positive, got {window_i}/{ring_i}")
    if state_shape[0] % ring_i:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if n_new == 0:
        return swa_kv_cache, kv_score_state

    cache = (
        _WRITE_SWA_KV_SCORE_STATE_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_swa_kv_score_state",
        swa_shape,
        state_shape,
        rows_shape,
        kv_shape,
        owner_shape,
        pos_shape,
        ape_shape,
        window_i,
        ring_i,
        str(_dtype_like(swa_kv_cache)),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(swa_rows)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_swa_kv_score_state_entry,
        _sample_like(swa_kv_cache),
        _sample_like(kv_score_state),
        _sample_like(swa_rows),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(owner_ids),
        _sample_like(positions),
        _sample_like(ape),
        window_size=window_i,
        ring_size=ring_i,
        name=(
            "dsv4_write_swa_score_state_"
            f"n{n_new}_sd{rows_shape[1]}_w{width}_"
            f"win{window_i}_r{ring_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "swa_kv_cache.must_alias_input": swa_kv_cache,
            "kv_score_state.must_alias_input": kv_score_state,
            "swa_rows": swa_rows,
            "kv_new": kv_new,
            "score_new": score_new,
            "owner_ids": owner_ids,
            "positions": positions,
            "ape": ape,
        },
        outputs={
            "swa_kv_cache": swa_kv_cache,
            "kv_score_state": kv_score_state,
        },
    )
    return swa_kv_cache, kv_score_state


def run_write_swa_dual_kv_score_state_device(
    *,
    swa_kv_cache: Any,
    kv_score_state: Any,
    indexer_kv_score_state: Any,
    swa_rows: Any,
    kv_new: Any,
    score_new: Any,
    indexer_kv_new: Any,
    indexer_score_new: Any,
    owner_ids: Any,
    positions: Any,
    ape: Any,
    indexer_ape: Any,
    live_rows: Any | None = None,
    window_size: int,
    ring_size: int,
    indexer_ring_size: int,
    guard_owner: int | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Fused decode scatter into SWA plus main and indexer ring states."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    indexer_state_shape = tuple(
        int(dim) for dim in getattr(indexer_kv_score_state, "shape")
    )
    rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    indexer_kv_shape = tuple(int(dim) for dim in getattr(indexer_kv_new, "shape"))
    indexer_score_shape = tuple(int(dim) for dim in getattr(indexer_score_new, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    indexer_ape_shape = tuple(int(dim) for dim in getattr(indexer_ape, "shape"))
    if live_rows is None:
        live_rows = _device_scalar_i32(
            int(kv_shape[0]),
            name="dsv4_write_swa_dual_state_live_rows",
        )
    live_shape = tuple(int(dim) for dim in getattr(live_rows, "shape"))
    if (
        len(swa_shape) != 2
        or len(state_shape) != 2
        or len(indexer_state_shape) != 2
        or len(kv_shape) != 2
        or len(indexer_kv_shape) != 2
    ):
        raise ValueError(
            "bad swa/state/indexer/kv shapes: "
            f"{swa_shape}/{state_shape}/{indexer_state_shape}/"
            f"{kv_shape}/{indexer_kv_shape}"
        )
    if len(rows_shape) != 2:
        raise ValueError(f"swa_rows must be 2-D, got {rows_shape}")
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    if indexer_score_shape != indexer_kv_shape:
        raise ValueError(
            "indexer_score_new shape "
            f"{indexer_score_shape} != indexer_kv_new {indexer_kv_shape}"
        )
    n_new, width = kv_shape
    indexer_n_new, indexer_width = indexer_kv_shape
    if rows_shape[0] != n_new or indexer_n_new != n_new:
        raise ValueError(
            "row counts must match main kv rows: "
            f"swa={rows_shape[0]}, indexer={indexer_n_new}, main={n_new}"
        )
    if swa_shape[1] != rows_shape[1]:
        raise ValueError(
            f"SWA head_dim mismatch: cache={swa_shape[1]}, rows={rows_shape[1]}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if indexer_state_shape[1] != 2 * indexer_width:
        raise ValueError(
            "indexer state packed width "
            f"{indexer_state_shape[1]} != 2 * kv width {indexer_width}"
        )
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    if len(indexer_ape_shape) != 2 or indexer_ape_shape[1] != indexer_width:
        raise ValueError(
            f"indexer_ape must be [ratio, {indexer_width}], got {indexer_ape_shape}"
        )
    if live_shape != (1, 1):
        raise ValueError(f"live_rows must be [1, 1], got {live_shape}")
    window_i = int(window_size)
    ring_i = int(ring_size)
    indexer_ring_i = int(indexer_ring_size)
    if window_i <= 0 or ring_i <= 0 or indexer_ring_i <= 0:
        raise ValueError(
            "window/ring/indexer_ring must be positive, got "
            f"{window_i}/{ring_i}/{indexer_ring_i}"
        )
    if state_shape[0] % ring_i:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if indexer_state_shape[0] % indexer_ring_i:
        raise ValueError(
            "indexer_ring_size="
            f"{indexer_ring_i} incompatible with state rows {indexer_state_shape[0]}"
        )
    # SWA may include one final generic padding slot after the guard-owner
    # window, so only require the selected guard owner to fit below.
    if guard_owner is None:
        guard_owner_i = (
            min(
                int(swa_shape[0]) // window_i,
                int(state_shape[0]) // ring_i,
                int(indexer_state_shape[0]) // indexer_ring_i,
            )
            - 1
        )
    else:
        guard_owner_i = int(guard_owner)
    if (
        guard_owner_i < 0
        or (guard_owner_i + 1) * window_i > int(swa_shape[0])
        or (guard_owner_i + 1) * ring_i > int(state_shape[0])
        or (guard_owner_i + 1) * indexer_ring_i > int(indexer_state_shape[0])
    ):
        raise ValueError(
            f"guard_owner={guard_owner_i} outside SWA/state caches "
            f"(swa={swa_shape}, state={state_shape}, "
            f"indexer_state={indexer_state_shape})"
        )
    if n_new == 0:
        return swa_kv_cache, kv_score_state, indexer_kv_score_state

    cache = (
        _WRITE_SWA_DUAL_KV_SCORE_STATE_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_swa_dual_kv_score_state",
        swa_shape,
        state_shape,
        indexer_state_shape,
        rows_shape,
        kv_shape,
        indexer_kv_shape,
        owner_shape,
        pos_shape,
        ape_shape,
        indexer_ape_shape,
        live_shape,
        window_i,
        ring_i,
        indexer_ring_i,
        guard_owner_i,
        str(_dtype_like(swa_kv_cache)),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(indexer_kv_score_state)),
        str(_dtype_like(swa_rows)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(indexer_kv_new)),
        str(_dtype_like(indexer_score_new)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
        str(_dtype_like(live_rows)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_swa_dual_kv_score_state_entry,
        _sample_like(swa_kv_cache),
        _sample_like(kv_score_state),
        _sample_like(indexer_kv_score_state),
        _sample_like(swa_rows),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(indexer_kv_new),
        _sample_like(indexer_score_new),
        _sample_like(owner_ids),
        _sample_like(positions),
        _sample_like(ape),
        _sample_like(indexer_ape),
        _sample_like(live_rows),
        window_size=window_i,
        ring_size=ring_i,
        indexer_ring_size=indexer_ring_i,
        guard_owner=guard_owner_i,
        name=(
            "dsv4_write_swa_dual_score_state_"
            f"nbucket{n_new}_sd{rows_shape[1]}_w{width}_iw{indexer_width}_"
            f"win{window_i}_r{ring_i}_ir{indexer_ring_i}_g{guard_owner_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "swa_kv_cache.must_alias_input": swa_kv_cache,
            "kv_score_state.must_alias_input": kv_score_state,
            "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
            "swa_rows": swa_rows,
            "kv_new": kv_new,
            "score_new": score_new,
            "indexer_kv_new": indexer_kv_new,
            "indexer_score_new": indexer_score_new,
            "owner_ids": owner_ids,
            "positions": positions,
            "ape": ape,
            "indexer_ape": indexer_ape,
            "live_rows": live_rows,
        },
        outputs={
            "swa_kv_cache": swa_kv_cache,
            "kv_score_state": kv_score_state,
            "indexer_kv_score_state": indexer_kv_score_state,
        },
    )
    return swa_kv_cache, kv_score_state, indexer_kv_score_state


def run_write_swa_dual_kv_score_state_owner_pos_device(
    *,
    swa_kv_cache: Any,
    kv_score_state: Any,
    compressed_kv_cache: Any,
    indexer_kv_score_state: Any,
    indexer_compressed_kv_cache: Any,
    swa_rows: Any,
    kv_new: Any,
    score_new: Any,
    compressed_rows: Any,
    indexer_kv_new: Any,
    indexer_score_new: Any,
    indexer_compressed_rows: Any,
    owner_ids: Any,
    positions: Any,
    ape: Any,
    indexer_ape: Any,
    window_size: int,
    ring_size: int,
    indexer_ring_size: int,
    ratio: int,
    indexer_ratio: int,
    max_clen: int,
    indexer_max_clen: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Fused boundary decode update for SWA plus main/indexer state and cache."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape"))
    indexer_state_shape = tuple(
        int(dim) for dim in getattr(indexer_kv_score_state, "shape")
    )
    indexer_cache_shape = tuple(
        int(dim) for dim in getattr(indexer_compressed_kv_cache, "shape")
    )
    rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    comp_rows_shape = tuple(int(dim) for dim in getattr(compressed_rows, "shape"))
    indexer_kv_shape = tuple(int(dim) for dim in getattr(indexer_kv_new, "shape"))
    indexer_score_shape = tuple(int(dim) for dim in getattr(indexer_score_new, "shape"))
    indexer_comp_rows_shape = tuple(
        int(dim) for dim in getattr(indexer_compressed_rows, "shape")
    )
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    indexer_ape_shape = tuple(int(dim) for dim in getattr(indexer_ape, "shape"))
    if (
        len(swa_shape) != 2
        or len(state_shape) != 2
        or len(cache_shape) != 2
        or len(indexer_state_shape) != 2
        or len(indexer_cache_shape) != 2
        or len(kv_shape) != 2
        or len(indexer_kv_shape) != 2
    ):
        raise ValueError(
            "bad cache/state/kv shapes: "
            f"{swa_shape}/{state_shape}/{cache_shape}/"
            f"{indexer_state_shape}/{indexer_cache_shape}/"
            f"{kv_shape}/{indexer_kv_shape}"
        )
    if len(rows_shape) != 2 or len(comp_rows_shape) != 2:
        raise ValueError(
            f"swa/main compressed rows must be 2-D, got {rows_shape}/{comp_rows_shape}"
        )
    if len(indexer_comp_rows_shape) != 2:
        raise ValueError(
            f"indexer compressed rows must be 2-D, got {indexer_comp_rows_shape}"
        )
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    if indexer_score_shape != indexer_kv_shape:
        raise ValueError(
            "indexer_score_new shape "
            f"{indexer_score_shape} != indexer_kv_new {indexer_kv_shape}"
        )
    n_new, width = kv_shape
    indexer_n_new, indexer_width = indexer_kv_shape
    if (
        rows_shape[0] != n_new
        or comp_rows_shape[0] != n_new
        or indexer_n_new != n_new
        or indexer_comp_rows_shape[0] != n_new
    ):
        raise ValueError(
            "row counts must match main kv rows: "
            f"swa={rows_shape[0]}, comp={comp_rows_shape[0]}, "
            f"indexer={indexer_n_new}, indexer_comp={indexer_comp_rows_shape[0]}, "
            f"main={n_new}"
        )
    if (
        swa_shape[1] != rows_shape[1]
        or cache_shape[1] != comp_rows_shape[1]
        or indexer_cache_shape[1] != indexer_comp_rows_shape[1]
    ):
        raise ValueError(
            "cache/head dim mismatch: "
            f"swa={swa_shape}/{rows_shape}, comp={cache_shape}/{comp_rows_shape}, "
            f"indexer={indexer_cache_shape}/{indexer_comp_rows_shape}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if indexer_state_shape[1] != 2 * indexer_width:
        raise ValueError(
            "indexer state packed width "
            f"{indexer_state_shape[1]} != 2 * kv width {indexer_width}"
        )
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    if len(indexer_ape_shape) != 2 or indexer_ape_shape[1] != indexer_width:
        raise ValueError(
            f"indexer_ape must be [ratio, {indexer_width}], got {indexer_ape_shape}"
        )
    window_i = int(window_size)
    ring_i = int(ring_size)
    indexer_ring_i = int(indexer_ring_size)
    ratio_i = int(ratio)
    indexer_ratio_i = int(indexer_ratio)
    max_clen_i = int(max_clen)
    indexer_max_clen_i = int(indexer_max_clen)
    if (
        window_i <= 0
        or ring_i <= 0
        or indexer_ring_i <= 0
        or ratio_i <= 0
        or indexer_ratio_i <= 0
        or max_clen_i <= 0
        or indexer_max_clen_i <= 0
    ):
        raise ValueError(
            "window/ring/ratio/max_clen values must be positive, got "
            f"{window_i}/{ring_i}/{indexer_ring_i}/"
            f"{ratio_i}/{indexer_ratio_i}/{max_clen_i}/{indexer_max_clen_i}"
        )
    if state_shape[0] % ring_i:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if indexer_state_shape[0] % indexer_ring_i:
        raise ValueError(
            "indexer_ring_size="
            f"{indexer_ring_i} incompatible with state rows {indexer_state_shape[0]}"
        )
    if n_new == 0:
        return (
            swa_kv_cache,
            kv_score_state,
            compressed_kv_cache,
            indexer_kv_score_state,
            indexer_compressed_kv_cache,
        )

    cache = (
        _WRITE_SWA_DUAL_KV_SCORE_STATE_OWNER_POS_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_swa_dual_kv_score_state_owner_pos",
        swa_shape,
        state_shape,
        cache_shape,
        indexer_state_shape,
        indexer_cache_shape,
        rows_shape,
        kv_shape,
        comp_rows_shape,
        indexer_kv_shape,
        indexer_comp_rows_shape,
        owner_shape,
        pos_shape,
        ape_shape,
        indexer_ape_shape,
        window_i,
        ring_i,
        indexer_ring_i,
        ratio_i,
        indexer_ratio_i,
        max_clen_i,
        indexer_max_clen_i,
        str(_dtype_like(swa_kv_cache)),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(compressed_kv_cache)),
        str(_dtype_like(indexer_kv_score_state)),
        str(_dtype_like(indexer_compressed_kv_cache)),
        str(_dtype_like(swa_rows)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(compressed_rows)),
        str(_dtype_like(indexer_kv_new)),
        str(_dtype_like(indexer_score_new)),
        str(_dtype_like(indexer_compressed_rows)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_swa_dual_kv_score_state_owner_pos_entry,
        _sample_like(swa_kv_cache),
        _sample_like(kv_score_state),
        _sample_like(compressed_kv_cache),
        _sample_like(indexer_kv_score_state),
        _sample_like(indexer_compressed_kv_cache),
        _sample_like(swa_rows),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(compressed_rows),
        _sample_like(indexer_kv_new),
        _sample_like(indexer_score_new),
        _sample_like(indexer_compressed_rows),
        _sample_like(owner_ids),
        _sample_like(positions),
        _sample_like(ape),
        _sample_like(indexer_ape),
        window_size=window_i,
        ring_size=ring_i,
        indexer_ring_size=indexer_ring_i,
        ratio=ratio_i,
        indexer_ratio=indexer_ratio_i,
        max_clen=max_clen_i,
        indexer_max_clen=indexer_max_clen_i,
        name=(
            "dsv4_write_swa_dual_score_owner_pos_"
            f"n{n_new}_sd{rows_shape[1]}_w{width}_iw{indexer_width}_"
            f"d{cache_shape[1]}_id{indexer_cache_shape[1]}_"
            f"win{window_i}_r{ring_i}_ir{indexer_ring_i}_"
            f"cr{ratio_i}_icr{indexer_ratio_i}_"
            f"c{max_clen_i}_ic{indexer_max_clen_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "swa_kv_cache.must_alias_input": swa_kv_cache,
            "kv_score_state.must_alias_input": kv_score_state,
            "compressed_kv_cache.must_alias_input": compressed_kv_cache,
            "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
            "indexer_compressed_kv_cache.must_alias_input": indexer_compressed_kv_cache,
            "swa_rows": swa_rows,
            "kv_new": kv_new,
            "score_new": score_new,
            "compressed_rows": compressed_rows,
            "indexer_kv_new": indexer_kv_new,
            "indexer_score_new": indexer_score_new,
            "indexer_compressed_rows": indexer_compressed_rows,
            "owner_ids": owner_ids,
            "positions": positions,
            "ape": ape,
            "indexer_ape": indexer_ape,
        },
        outputs={
            "swa_kv_cache": swa_kv_cache,
            "kv_score_state": kv_score_state,
            "compressed_kv_cache": compressed_kv_cache,
            "indexer_kv_score_state": indexer_kv_score_state,
            "indexer_compressed_kv_cache": indexer_compressed_kv_cache,
        },
    )
    return (
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        indexer_kv_score_state,
        indexer_compressed_kv_cache,
    )


def run_write_swa_dual_kv_score_state_owner_clen_device(
    *,
    swa_kv_cache: Any,
    kv_score_state: Any,
    compressed_kv_cache: Any,
    indexer_kv_score_state: Any,
    indexer_compressed_kv_cache: Any,
    swa_rows: Any,
    kv_new: Any,
    score_new: Any,
    compressed_rows: Any,
    indexer_kv_new: Any,
    indexer_score_new: Any,
    indexer_compressed_rows: Any,
    swa_owner_ids: Any,
    swa_positions: Any,
    state_owner_ids: Any,
    state_positions: Any,
    cache_owner_ids: Any,
    ape: Any,
    indexer_ape: Any,
    window_size: int,
    ring_size: int,
    indexer_ring_size: int,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
    indexer_max_clen: int,
    cache_real_clen: int | None = None,
    guard_owner: int | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Fused prefill update for SWA plus main/indexer state and cache.

    ``cache_real_clen`` is the real prompt's compressed columns per request when
    the kernel is compiled at a token bucket whose ``clen`` is larger (bucketed
    prefill). Padded columns ``cpos >= cache_real_clen`` are redirected to the
    ``guard_owner`` block (never read). When ``cache_real_clen is None`` it
    defaults to ``clen`` (no masking) and ``guard_owner`` is unused — preserving
    the legacy per-length callers. ``cache_real_clen`` is uploaded as a runtime
    ``[1, 1]`` int32 scalar and is deliberately EXCLUDED from the kernel cache
    key so one NEFF serves every real length within a bucket."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape"))
    indexer_state_shape = tuple(
        int(dim) for dim in getattr(indexer_kv_score_state, "shape")
    )
    indexer_cache_shape = tuple(
        int(dim) for dim in getattr(indexer_compressed_kv_cache, "shape")
    )
    swa_rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    comp_rows_shape = tuple(int(dim) for dim in getattr(compressed_rows, "shape"))
    indexer_kv_shape = tuple(int(dim) for dim in getattr(indexer_kv_new, "shape"))
    indexer_score_shape = tuple(int(dim) for dim in getattr(indexer_score_new, "shape"))
    indexer_comp_rows_shape = tuple(
        int(dim) for dim in getattr(indexer_compressed_rows, "shape")
    )
    swa_owner_shape = tuple(int(dim) for dim in getattr(swa_owner_ids, "shape"))
    swa_pos_shape = tuple(int(dim) for dim in getattr(swa_positions, "shape"))
    state_owner_shape = tuple(int(dim) for dim in getattr(state_owner_ids, "shape"))
    state_pos_shape = tuple(int(dim) for dim in getattr(state_positions, "shape"))
    cache_owner_shape = tuple(int(dim) for dim in getattr(cache_owner_ids, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    indexer_ape_shape = tuple(int(dim) for dim in getattr(indexer_ape, "shape"))
    if (
        len(swa_shape) != 2
        or len(state_shape) != 2
        or len(cache_shape) != 2
        or len(indexer_state_shape) != 2
        or len(indexer_cache_shape) != 2
        or len(kv_shape) != 2
        or len(indexer_kv_shape) != 2
    ):
        raise ValueError(
            "bad cache/state/kv shapes: "
            f"{swa_shape}/{state_shape}/{cache_shape}/"
            f"{indexer_state_shape}/{indexer_cache_shape}/"
            f"{kv_shape}/{indexer_kv_shape}"
        )
    if len(swa_rows_shape) != 2 or len(comp_rows_shape) != 2:
        raise ValueError(
            "swa/main compressed rows must be 2-D, got "
            f"{swa_rows_shape}/{comp_rows_shape}"
        )
    if len(indexer_comp_rows_shape) != 2:
        raise ValueError(
            f"indexer compressed rows must be 2-D, got {indexer_comp_rows_shape}"
        )
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    if indexer_score_shape != indexer_kv_shape:
        raise ValueError(
            "indexer_score_new shape "
            f"{indexer_score_shape} != indexer_kv_new {indexer_kv_shape}"
        )
    n_swa = int(swa_rows_shape[0])
    n_state, width = kv_shape
    n_cache = int(comp_rows_shape[0])
    indexer_n_state, indexer_width = indexer_kv_shape
    if indexer_n_state != n_state or indexer_comp_rows_shape[0] != n_cache:
        raise ValueError(
            "main/indexer prefill row counts must match: "
            f"state={n_state}/{indexer_n_state}, "
            f"cache={n_cache}/{indexer_comp_rows_shape[0]}"
        )
    if (
        swa_shape[1] != swa_rows_shape[1]
        or cache_shape[1] != comp_rows_shape[1]
        or indexer_cache_shape[1] != indexer_comp_rows_shape[1]
    ):
        raise ValueError(
            "cache/head dim mismatch: "
            f"swa={swa_shape}/{swa_rows_shape}, "
            f"main={cache_shape}/{comp_rows_shape}, "
            f"indexer={indexer_cache_shape}/{indexer_comp_rows_shape}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if indexer_state_shape[1] != 2 * indexer_width:
        raise ValueError(
            "indexer state packed width "
            f"{indexer_state_shape[1]} != 2 * kv width {indexer_width}"
        )
    if swa_owner_shape != (n_swa,) or swa_pos_shape != (n_swa,):
        raise ValueError(
            "SWA owner/position shapes must match swa rows: "
            f"{swa_owner_shape}/{swa_pos_shape} vs {n_swa}"
        )
    if state_owner_shape != (n_state,) or state_pos_shape != (n_state,):
        raise ValueError(
            "state owner/position shapes must match state rows: "
            f"{state_owner_shape}/{state_pos_shape} vs {n_state}"
        )
    if len(cache_owner_shape) != 1:
        raise ValueError(f"cache_owner_ids must be 1D, got {cache_owner_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    if len(indexer_ape_shape) != 2 or indexer_ape_shape[1] != indexer_width:
        raise ValueError(
            f"indexer_ape must be [ratio, {indexer_width}], got {indexer_ape_shape}"
        )
    window_i = int(window_size)
    ring_i = int(ring_size)
    indexer_ring_i = int(indexer_ring_size)
    clen_i = int(clen)
    stride_i = int(owner_id_stride)
    max_clen_i = int(max_clen)
    indexer_max_clen_i = int(indexer_max_clen)
    if (
        window_i <= 0
        or ring_i <= 0
        or indexer_ring_i <= 0
        or clen_i <= 0
        or stride_i <= 0
        or max_clen_i <= 0
        or indexer_max_clen_i <= 0
    ):
        raise ValueError(
            "window/ring/clen/stride/max_clen values must be positive, got "
            f"{window_i}/{ring_i}/{indexer_ring_i}/"
            f"{clen_i}/{stride_i}/{max_clen_i}/{indexer_max_clen_i}"
        )
    if state_shape[0] % ring_i:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if indexer_state_shape[0] % indexer_ring_i:
        raise ValueError(
            "indexer_ring_size="
            f"{indexer_ring_i} incompatible with state rows {indexer_state_shape[0]}"
        )
    if n_swa == 0 or n_state == 0 or n_cache == 0:
        return (
            swa_kv_cache,
            kv_score_state,
            compressed_kv_cache,
            indexer_kv_score_state,
            indexer_compressed_kv_cache,
        )
    if n_cache % clen_i:
        raise ValueError(f"compressed rows {n_cache} must be divisible by clen")
    bsz = n_cache // clen_i
    required_owners = (bsz - 1) * stride_i + 1
    if int(cache_owner_shape[0]) < required_owners:
        raise ValueError(
            "cache_owner_ids first dim too small for request-major scatter: "
            f"got {cache_owner_shape[0]}, need >= {required_owners}"
        )
    # Bucketed-prefill compressed-column mask. Default (None) = no masking: write
    # all clen columns (legacy per-length callers). When bucketed, real_clen_i is
    # the real prompt's compressed columns and padded columns redirect to the
    # guard owner. guard_owner defaults to the last owner block of the compressed
    # cache, which is reserved as the padding sink (state.py guard owner).
    real_clen_i = int(clen_i if cache_real_clen is None else cache_real_clen)
    if real_clen_i <= 0 or real_clen_i > clen_i:
        raise ValueError(f"cache_real_clen={real_clen_i} must be in (0, clen={clen_i}]")
    if guard_owner is None:
        guard_owner_i = int(cache_shape[0]) // max_clen_i - 1
    else:
        guard_owner_i = int(guard_owner)
    if guard_owner_i < 0 or (guard_owner_i + 1) * max_clen_i > int(cache_shape[0]):
        raise ValueError(
            f"guard_owner={guard_owner_i} outside compressed cache "
            f"(rows={cache_shape[0]}, max_clen={max_clen_i})"
        )
    cache_real_clen_dev = _device_scalar_i32(
        real_clen_i,
        name="dsv4_comp_prefill_cache_real_clen",
    )

    cache = (
        _WRITE_SWA_DUAL_KV_SCORE_STATE_OWNER_CLEN_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_swa_dual_kv_score_state_owner_clen",
        int(guard_owner_i),
        swa_shape,
        state_shape,
        cache_shape,
        indexer_state_shape,
        indexer_cache_shape,
        swa_rows_shape,
        kv_shape,
        comp_rows_shape,
        indexer_kv_shape,
        indexer_comp_rows_shape,
        swa_owner_shape,
        swa_pos_shape,
        state_owner_shape,
        state_pos_shape,
        cache_owner_shape,
        ape_shape,
        indexer_ape_shape,
        window_i,
        ring_i,
        indexer_ring_i,
        clen_i,
        stride_i,
        max_clen_i,
        indexer_max_clen_i,
        str(_dtype_like(swa_kv_cache)),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(compressed_kv_cache)),
        str(_dtype_like(indexer_kv_score_state)),
        str(_dtype_like(indexer_compressed_kv_cache)),
        str(_dtype_like(swa_rows)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(compressed_rows)),
        str(_dtype_like(indexer_kv_new)),
        str(_dtype_like(indexer_score_new)),
        str(_dtype_like(indexer_compressed_rows)),
        str(_dtype_like(swa_owner_ids)),
        str(_dtype_like(swa_positions)),
        str(_dtype_like(state_owner_ids)),
        str(_dtype_like(state_positions)),
        str(_dtype_like(cache_owner_ids)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_swa_dual_kv_score_state_owner_clen_entry,
        _sample_like(swa_kv_cache),
        _sample_like(kv_score_state),
        _sample_like(compressed_kv_cache),
        _sample_like(indexer_kv_score_state),
        _sample_like(indexer_compressed_kv_cache),
        _sample_like(swa_rows),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(compressed_rows),
        _sample_like(indexer_kv_new),
        _sample_like(indexer_score_new),
        _sample_like(indexer_compressed_rows),
        _sample_like(swa_owner_ids),
        _sample_like(swa_positions),
        _sample_like(state_owner_ids),
        _sample_like(state_positions),
        _sample_like(cache_owner_ids),
        _sample_like(ape),
        _sample_like(indexer_ape),
        _sample_like(cache_real_clen_dev),
        window_size=window_i,
        ring_size=ring_i,
        indexer_ring_size=indexer_ring_i,
        clen=clen_i,
        owner_id_stride=stride_i,
        max_clen=max_clen_i,
        indexer_max_clen=indexer_max_clen_i,
        guard_owner=guard_owner_i,
        name=(
            "dsv4_write_swa_dual_score_owner_clen_"
            f"nswa{n_swa}_sd{swa_rows_shape[1]}_ns{n_state}_nc{n_cache}_"
            f"w{width}_iw{indexer_width}_d{cache_shape[1]}_"
            f"id{indexer_cache_shape[1]}_win{window_i}_r{ring_i}_"
            f"ir{indexer_ring_i}_clen{clen_i}_stride{stride_i}_"
            f"c{max_clen_i}_ic{indexer_max_clen_i}_g{guard_owner_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "swa_kv_cache.must_alias_input": swa_kv_cache,
            "kv_score_state.must_alias_input": kv_score_state,
            "compressed_kv_cache.must_alias_input": compressed_kv_cache,
            "indexer_kv_score_state.must_alias_input": indexer_kv_score_state,
            "indexer_compressed_kv_cache.must_alias_input": indexer_compressed_kv_cache,
            "swa_rows": swa_rows,
            "kv_new": kv_new,
            "score_new": score_new,
            "compressed_rows": compressed_rows,
            "indexer_kv_new": indexer_kv_new,
            "indexer_score_new": indexer_score_new,
            "indexer_compressed_rows": indexer_compressed_rows,
            "swa_owner_ids": swa_owner_ids,
            "swa_positions": swa_positions,
            "state_owner_ids": state_owner_ids,
            "state_positions": state_positions,
            "cache_owner_ids": cache_owner_ids,
            "ape": ape,
            "indexer_ape": indexer_ape,
            "cache_real_clen": cache_real_clen_dev,
        },
        outputs={
            "swa_kv_cache": swa_kv_cache,
            "kv_score_state": kv_score_state,
            "compressed_kv_cache": compressed_kv_cache,
            "indexer_kv_score_state": indexer_kv_score_state,
            "indexer_compressed_kv_cache": indexer_compressed_kv_cache,
        },
    )
    return (
        swa_kv_cache,
        kv_score_state,
        compressed_kv_cache,
        indexer_kv_score_state,
        indexer_compressed_kv_cache,
    )


def run_write_swa_kv_score_state_owner_pos_device(
    *,
    swa_kv_cache: Any,
    kv_score_state: Any,
    compressed_kv_cache: Any,
    swa_rows: Any,
    kv_new: Any,
    score_new: Any,
    compressed_rows: Any,
    owner_ids: Any,
    positions: Any,
    ape: Any,
    window_size: int,
    ring_size: int,
    ratio: int,
    max_clen: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Fused decode scatter into SWA cache, compressor state, and cache."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape"))
    rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    comp_rows_shape = tuple(int(dim) for dim in getattr(compressed_rows, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    if (
        len(swa_shape) != 2
        or len(state_shape) != 2
        or len(cache_shape) != 2
        or len(kv_shape) != 2
    ):
        raise ValueError(
            "bad swa/state/cache/kv shapes: "
            f"{swa_shape}/{state_shape}/{cache_shape}/{kv_shape}"
        )
    if len(rows_shape) != 2 or len(comp_rows_shape) != 2:
        raise ValueError(
            f"swa/compressed rows must be 2-D, got {rows_shape}/{comp_rows_shape}"
        )
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    n_new, width = kv_shape
    if rows_shape[0] != n_new or comp_rows_shape[0] != n_new:
        raise ValueError(
            "row counts must match kv rows: "
            f"swa={rows_shape[0]}, comp={comp_rows_shape[0]}, kv={n_new}"
        )
    if swa_shape[1] != rows_shape[1] or cache_shape[1] != comp_rows_shape[1]:
        raise ValueError(
            "cache/head dim mismatch: "
            f"swa={swa_shape}/{rows_shape}, comp={cache_shape}/{comp_rows_shape}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    window_i = int(window_size)
    ring_i = int(ring_size)
    ratio_i = int(ratio)
    max_clen_i = int(max_clen)
    if window_i <= 0 or ring_i <= 0 or ratio_i <= 0 or max_clen_i <= 0:
        raise ValueError(
            "window/ring/ratio/max_clen must be positive, got "
            f"{window_i}/{ring_i}/{ratio_i}/{max_clen_i}"
        )
    if state_shape[0] % ring_i:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if n_new == 0:
        return swa_kv_cache, kv_score_state, compressed_kv_cache

    cache = (
        _WRITE_SWA_KV_SCORE_STATE_OWNER_POS_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_swa_kv_score_state_owner_pos",
        swa_shape,
        state_shape,
        cache_shape,
        rows_shape,
        kv_shape,
        comp_rows_shape,
        owner_shape,
        pos_shape,
        ape_shape,
        window_i,
        ring_i,
        ratio_i,
        max_clen_i,
        str(_dtype_like(swa_kv_cache)),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(compressed_kv_cache)),
        str(_dtype_like(swa_rows)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(compressed_rows)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_swa_kv_score_state_owner_pos_entry,
        _sample_like(swa_kv_cache),
        _sample_like(kv_score_state),
        _sample_like(compressed_kv_cache),
        _sample_like(swa_rows),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(compressed_rows),
        _sample_like(owner_ids),
        _sample_like(positions),
        _sample_like(ape),
        window_size=window_i,
        ring_size=ring_i,
        ratio=ratio_i,
        max_clen=max_clen_i,
        name=(
            "dsv4_write_swa_score_owner_pos_"
            f"n{n_new}_sd{rows_shape[1]}_w{width}_d{cache_shape[1]}_"
            f"win{window_i}_r{ring_i}_cr{ratio_i}_c{max_clen_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "swa_kv_cache.must_alias_input": swa_kv_cache,
            "kv_score_state.must_alias_input": kv_score_state,
            "compressed_kv_cache.must_alias_input": compressed_kv_cache,
            "swa_rows": swa_rows,
            "kv_new": kv_new,
            "score_new": score_new,
            "compressed_rows": compressed_rows,
            "owner_ids": owner_ids,
            "positions": positions,
            "ape": ape,
        },
        outputs={
            "swa_kv_cache": swa_kv_cache,
            "kv_score_state": kv_score_state,
            "compressed_kv_cache": compressed_kv_cache,
        },
    )
    return swa_kv_cache, kv_score_state, compressed_kv_cache


def run_write_swa_kv_score_state_owner_clen_device(
    *,
    swa_kv_cache: Any,
    kv_score_state: Any,
    compressed_kv_cache: Any,
    swa_rows: Any,
    kv_new: Any,
    score_new: Any,
    compressed_rows: Any,
    swa_owner_ids: Any,
    swa_positions: Any,
    state_owner_ids: Any,
    state_positions: Any,
    cache_owner_ids: Any,
    ape: Any,
    window_size: int,
    ring_size: int,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
    cache_real_clen: int | None = None,
    guard_owner: int | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Fused prefill scatter into SWA cache, compressor state, and cache.

    ``cache_real_clen``/``guard_owner`` (bucketed-prefill mode): when given,
    compressed columns at/after ``cache_real_clen`` are redirected to the guard
    owner block. Defaults preserve the legacy unmasked behavior (real == clen,
    guard = first owner past stride coverage — never selected since all columns
    stay valid).
    """

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    swa_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(compressed_kv_cache, "shape"))
    swa_rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape"))
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    comp_rows_shape = tuple(int(dim) for dim in getattr(compressed_rows, "shape"))
    swa_owner_shape = tuple(int(dim) for dim in getattr(swa_owner_ids, "shape"))
    swa_pos_shape = tuple(int(dim) for dim in getattr(swa_positions, "shape"))
    state_owner_shape = tuple(int(dim) for dim in getattr(state_owner_ids, "shape"))
    state_pos_shape = tuple(int(dim) for dim in getattr(state_positions, "shape"))
    cache_owner_shape = tuple(int(dim) for dim in getattr(cache_owner_ids, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    if (
        len(swa_shape) != 2
        or len(state_shape) != 2
        or len(cache_shape) != 2
        or len(kv_shape) != 2
    ):
        raise ValueError(
            "bad swa/state/cache/kv shapes: "
            f"{swa_shape}/{state_shape}/{cache_shape}/{kv_shape}"
        )
    if len(swa_rows_shape) != 2 or len(comp_rows_shape) != 2:
        raise ValueError(
            f"swa/compressed rows must be 2-D, got {swa_rows_shape}/{comp_rows_shape}"
        )
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    n_swa = int(swa_rows_shape[0])
    n_state, width = kv_shape
    n_cache = int(comp_rows_shape[0])
    if swa_shape[1] != swa_rows_shape[1] or cache_shape[1] != comp_rows_shape[1]:
        raise ValueError(
            "cache/head dim mismatch: "
            f"swa={swa_shape}/{swa_rows_shape}, "
            f"comp={cache_shape}/{comp_rows_shape}"
        )
    if state_shape[1] != 2 * width:
        raise ValueError(f"state packed width {state_shape[1]} != 2 * kv width {width}")
    if swa_owner_shape != (n_swa,) or swa_pos_shape != (n_swa,):
        raise ValueError(
            "SWA owner/position shapes must match swa rows: "
            f"{swa_owner_shape}/{swa_pos_shape} vs {n_swa}"
        )
    if state_owner_shape != (n_state,) or state_pos_shape != (n_state,):
        raise ValueError(
            "state owner/position shapes must match kv rows: "
            f"{state_owner_shape}/{state_pos_shape} vs {n_state}"
        )
    if len(cache_owner_shape) != 1:
        raise ValueError(f"cache_owner_ids must be 1D, got {cache_owner_shape}")
    if len(ape_shape) != 2 or ape_shape[1] != width:
        raise ValueError(f"ape must be [ratio, {width}], got {ape_shape}")
    window_i = int(window_size)
    ring_i = int(ring_size)
    clen_i = int(clen)
    stride_i = int(owner_id_stride)
    max_clen_i = int(max_clen)
    if window_i <= 0 or ring_i <= 0 or clen_i <= 0 or stride_i <= 0 or max_clen_i <= 0:
        raise ValueError(
            "window, ring, clen, owner_id_stride, and max_clen must be positive, "
            f"got {window_i}/{ring_i}/{clen_i}/{stride_i}/{max_clen_i}"
        )
    if state_shape[0] % ring_i:
        raise ValueError(
            f"ring_size={ring_i} incompatible with state rows {state_shape[0]}"
        )
    if n_swa == 0 or n_state == 0 or n_cache == 0:
        return swa_kv_cache, kv_score_state, compressed_kv_cache
    if n_cache % clen_i:
        raise ValueError(f"compressed rows {n_cache} must be divisible by clen")
    bsz = n_cache // clen_i
    required_owners = (bsz - 1) * stride_i + 1
    if int(cache_owner_shape[0]) < required_owners:
        raise ValueError(
            "cache_owner_ids first dim too small for request-major scatter: "
            f"got {cache_owner_shape[0]}, need >= {required_owners}"
        )

    real_clen_i = int(cache_real_clen) if cache_real_clen is not None else clen_i
    if real_clen_i <= 0 or real_clen_i > clen_i:
        raise ValueError(f"cache_real_clen {real_clen_i} not in (0, {clen_i}]")
    # guard default: row past every real owner block (legacy mask never fires)
    guard_i = int(guard_owner) if guard_owner is not None else required_owners
    real_clen_dev = _device_scalar_i32(real_clen_i, name="dsv4_swa_clen_real")
    cache = (
        _WRITE_SWA_KV_SCORE_STATE_OWNER_CLEN_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "write_swa_kv_score_state_owner_clen",
        int(guard_i),
        swa_shape,
        state_shape,
        cache_shape,
        swa_rows_shape,
        kv_shape,
        comp_rows_shape,
        swa_owner_shape,
        swa_pos_shape,
        state_owner_shape,
        state_pos_shape,
        cache_owner_shape,
        ape_shape,
        window_i,
        ring_i,
        clen_i,
        stride_i,
        max_clen_i,
        str(_dtype_like(swa_kv_cache)),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(compressed_kv_cache)),
        str(_dtype_like(swa_rows)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(compressed_rows)),
        str(_dtype_like(swa_owner_ids)),
        str(_dtype_like(swa_positions)),
        str(_dtype_like(state_owner_ids)),
        str(_dtype_like(state_positions)),
        str(_dtype_like(cache_owner_ids)),
    )
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _write_swa_kv_score_state_owner_clen_entry,
        _sample_like(swa_kv_cache),
        _sample_like(kv_score_state),
        _sample_like(compressed_kv_cache),
        _sample_like(swa_rows),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(compressed_rows),
        _sample_like(swa_owner_ids),
        _sample_like(swa_positions),
        _sample_like(state_owner_ids),
        _sample_like(state_positions),
        _sample_like(cache_owner_ids),
        _sample_like(ape),
        _sample_like(real_clen_dev),
        window_size=window_i,
        ring_size=ring_i,
        clen=clen_i,
        owner_id_stride=stride_i,
        max_clen=max_clen_i,
        guard_owner=guard_i,
        name=(
            "dsv4_write_swa_score_owner_clen_"
            f"nswa{n_swa}_sd{swa_rows_shape[1]}_ns{n_state}_nc{n_cache}_"
            f"w{width}_d{cache_shape[1]}_win{window_i}_r{ring_i}_"
            f"clen{clen_i}_stride{stride_i}_c{max_clen_i}_g{guard_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    kernel(
        inputs={
            "swa_kv_cache.must_alias_input": swa_kv_cache,
            "kv_score_state.must_alias_input": kv_score_state,
            "compressed_kv_cache.must_alias_input": compressed_kv_cache,
            "swa_rows": swa_rows,
            "kv_new": kv_new,
            "score_new": score_new,
            "compressed_rows": compressed_rows,
            "swa_owner_ids": swa_owner_ids,
            "swa_positions": swa_positions,
            "state_owner_ids": state_owner_ids,
            "state_positions": state_positions,
            "cache_owner_ids": cache_owner_ids,
            "cache_real_clen": real_clen_dev,
            "ape": ape,
        },
        outputs={
            "swa_kv_cache": swa_kv_cache,
            "kv_score_state": kv_score_state,
            "compressed_kv_cache": compressed_kv_cache,
        },
    )
    return swa_kv_cache, kv_score_state, compressed_kv_cache


def run_prefill_pool_from_slab_device(
    *,
    kv_new: Any,
    score_new: Any,
    ape: Any,
    bsz: int,
    seqlen: int,
    ratio: int,
    head_dim: int,
    overlap: bool,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
    output: Any | None = None,
) -> Any:
    """Device prefill softmax-pool from flat compressor projection slabs."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    kv_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    score_shape = tuple(int(dim) for dim in getattr(score_new, "shape"))
    ape_shape = tuple(int(dim) for dim in getattr(ape, "shape"))
    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    ratio_i = int(ratio)
    d_i = int(head_dim)
    width_i = 2 * d_i if bool(overlap) else d_i
    if len(kv_shape) != 2:
        raise ValueError(f"kv_new must be [B*S,width], got {kv_shape}")
    if score_shape != kv_shape:
        raise ValueError(f"score_new shape {score_shape} != kv_new {kv_shape}")
    if bsz_i <= 0 or seqlen_i <= 0 or ratio_i <= 0 or d_i <= 0:
        raise ValueError("bsz, seqlen, ratio, and head_dim must be positive")
    if seqlen_i % ratio_i:
        raise ValueError("seqlen must be a multiple of ratio")
    if kv_shape != (bsz_i * seqlen_i, width_i):
        raise ValueError(f"kv_new shape {kv_shape} != [{bsz_i * seqlen_i}, {width_i}]")
    if ape_shape != (ratio_i, width_i):
        raise ValueError(f"ape shape {ape_shape} != [{ratio_i}, {width_i}]")
    if bool(overlap) and width_i != 2 * d_i:
        raise ValueError("overlap prefill requires state_width=2*head_dim")

    groups = seqlen_i // ratio_i
    out_rows = bsz_i * groups
    base_rows = np.empty((out_rows,), dtype=np.int32)
    prev_rows = np.empty((out_rows,), dtype=np.int32)
    prev_valid = np.empty((out_rows,), dtype=np.uint8)
    idx = 0
    for b in range(bsz_i):
        owner_base = b * seqlen_i
        for g in range(groups):
            base = owner_base + g * ratio_i
            base_rows[idx] = base
            prev_rows[idx] = base - ratio_i if g > 0 else base
            prev_valid[idx] = 1 if g > 0 else 0
            idx += 1

    DeviceTensor = get_device_tensor_cls()

    base_dev = DeviceTensor.from_numpy(base_rows, name="dsv4_prefill_base_rows")
    prev_dev = DeviceTensor.from_numpy(prev_rows, name="dsv4_prefill_prev_rows")
    valid_dev = DeviceTensor.from_numpy(
        prev_valid,
        name="dsv4_prefill_prev_valid",
    )

    cache = (
        _PREFILL_POOL_FROM_SLAB_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    cache_key = (
        "prefill_pool_from_slab",
        kv_shape,
        ape_shape,
        tuple(int(dim) for dim in getattr(base_dev, "shape")),
        ratio_i,
        d_i,
        width_i,
        bool(overlap),
        str(_dtype_like(kv_new)),
        str(_dtype_like(score_new)),
        str(_dtype_like(ape)),
    )
    kind = "overlap" if bool(overlap) else "plain"
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _make_prefill_pool_from_slab_entry(
            ratio=ratio_i,
            head_dim=d_i,
            state_width=width_i,
            overlap=bool(overlap),
        ),
        _sample_like(kv_new),
        _sample_like(score_new),
        _sample_like(ape),
        _sample_like(base_dev),
        _sample_like(prev_dev),
        _sample_like(valid_dev),
        name=(
            f"dsv4_prefill_pool_from_slab_{kind}_b{bsz_i}_s{seqlen_i}_r{ratio_i}_d{d_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    out_dev = output
    if out_dev is None:
        out_dev = DeviceTensor.from_numpy(
            np.zeros((out_rows, d_i), dtype=np.float32),
            name="dsv4_prefill_pool_from_slab_out",
        )
    kernel(
        inputs={
            "kv_new": kv_new,
            "score_new": score_new,
            "ape": ape,
            "base_rows": base_dev,
            "prev_rows": prev_dev,
            "prev_valid": valid_dev,
        },
        outputs={"output0": out_dev},
    )
    return out_dev


def run_decode_pool_from_state_device(
    *,
    kv_score_state: Any,
    owner_ids: Any,
    end_positions: Any,
    ratio: int,
    head_dim: int,
    state_width: int,
    ring_size: int,
    overlap: bool,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
    output: Any | None = None,
) -> Any:
    """Device decode softmax-pool from flat ring compressor state."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    state_shape = tuple(int(dim) for dim in getattr(kv_score_state, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(end_positions, "shape"))
    ratio_i = int(ratio)
    d_i = int(head_dim)
    width_i = int(state_width)
    ring_i = int(ring_size)
    if len(state_shape) != 2:
        raise ValueError(f"kv_score_state must be 2D, got {state_shape}")
    if owner_shape != pos_shape or len(owner_shape) != 1:
        raise ValueError(
            f"owner_ids/end_positions must both be [B], got {owner_shape}/{pos_shape}"
        )
    if d_i <= 0:
        raise ValueError(f"head_dim must be positive, got {d_i}")
    if width_i < d_i or state_shape[1] != 2 * width_i:
        raise ValueError(
            f"state packed width {state_shape[1]} incompatible with state_width={width_i}"
        )
    if ratio_i <= 0 or ring_i <= 0:
        raise ValueError("ratio and ring_size must be positive")
    if bool(overlap):
        if width_i != 2 * d_i or ring_i != 2 * ratio_i:
            raise ValueError(
                "overlap decode pool requires state_width=2*head_dim and "
                "ring_size=2*ratio"
            )

    cache = (
        _DECODE_POOL_FROM_STATE_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    cache_key = (
        "decode_pool_from_state",
        state_shape,
        owner_shape,
        ratio_i,
        d_i,
        width_i,
        ring_i,
        bool(overlap),
        str(_dtype_like(kv_score_state)),
        str(_dtype_like(owner_ids)),
    )
    kind = "overlap" if bool(overlap) else "plain"
    kernel = _get_or_compile_kernel(
        cache,
        cache_key,
        _make_decode_pool_from_state_entry(
            ratio=ratio_i,
            head_dim=d_i,
            state_width=width_i,
            ring_size=ring_i,
            overlap=bool(overlap),
        ),
        _sample_like(kv_score_state),
        _sample_like(owner_ids),
        _sample_like(end_positions),
        name=(
            f"dsv4_decode_pool_from_state_{kind}_b{owner_shape[0]}_r{ratio_i}_d{d_i}"
        ),
        build_dir=artifacts_dir,
        namespace="dsv4_compressor_kernels",
        device_kernel_cls=_device_kernel_cls,
    )

    DeviceTensor = get_device_tensor_cls()

    out_dev = output
    if out_dev is None:
        out_dev = DeviceTensor.from_numpy(
            np.zeros((owner_shape[0], d_i), dtype=np.float32),
            name="dsv4_decode_pool_from_state_out",
        )
    kernel(
        inputs={
            "kv_score_state": kv_score_state,
            "owner_ids": owner_ids,
            "end_positions": end_positions,
        },
        outputs={"output0": out_dev},
    )
    return out_dev


# ---------------------------------------------------------------------------
# Stage 3 state-bridging helpers (moved from models/deepseek_v4/sampled_forward.py)
# ---------------------------------------------------------------------------


def mirror_compressor_input_to_device_state(
    compressor: Any,
    kv: Any,
    score: Any,
    start_pos: int,
    *,
    bsz: int,
    seqlen: int,
    device_state: Any,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
) -> None:
    """Scatter compressor KV/score projections into device ring state.

    ``kv`` and ``score`` may be either numpy ``[bsz, seqlen, width]`` float
    or already-resident DeviceTensors of shape ``[bsz*seqlen, width] bf16``
    (the typical production input, produced by the
    ``compressor_kv_score_bf16`` trace function). When ``device_state``
    lacks a device-resident kv_score_state, falls back to the host oracle
    (requires numpy inputs).
    """
    import ml_dtypes as _ml

    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    n = bsz_i * seqlen_i
    if owner_ids is None:
        owner_ids_arr = np.repeat(np.arange(bsz_i, dtype=np.int32), seqlen_i)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (n,):
            raise ValueError(f"owner_ids must be [{n}], got {owner_ids_arr.shape}")
    positions = np.tile(
        np.arange(int(start_pos), int(start_pos) + seqlen_i, dtype=np.int32),
        bsz_i,
    )
    is_dev_kv = _is_device_tensor_like(kv)
    row_indices: np.ndarray | None = None
    if int(start_pos) == 0:
        keep_tokens = _prefill_state_tail_len(
            spec=device_state.spec,
            seqlen=seqlen_i,
        )
        if keep_tokens == 0:
            return
        if keep_tokens < seqlen_i:
            if is_dev_kv and bsz_i != 1:
                # Multi-request tails are not contiguous in flattened
                # [B*S, W] layout. Keep the full mirror until the product path
                # has a batched strided state-write kernel.
                keep_tokens = seqlen_i
            else:
                start = seqlen_i - int(keep_tokens)
                row_indices = np.concatenate(
                    [
                        np.arange(
                            b * seqlen_i + start,
                            (b + 1) * seqlen_i,
                            dtype=np.int64,
                        )
                        for b in range(bsz_i)
                    ]
                )
    if row_indices is not None and is_dev_kv:
        kv_alias = _alias_device_value_first_dim_slice(
            kv,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        score_alias = _alias_device_value_first_dim_slice(
            score,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        if kv_alias is not None and score_alias is not None:
            kv = kv_alias
            score = score_alias
        else:
            row_indices = None
    if row_indices is not None:
        owner_ids_arr = np.ascontiguousarray(owner_ids_arr[row_indices])
        positions = np.ascontiguousarray(positions[row_indices])

    if not hasattr(device_state.kv_score_state, "tensor_ref"):
        if is_dev_kv:
            if not hasattr(kv, "numpy") or not hasattr(score, "numpy"):
                raise RuntimeError(
                    "host compressor-state fallback cannot consume "
                    "non-downloadable device tensors"
                )
            kv_flat = kv.numpy().astype(np.float32)
            score_flat = score.numpy().astype(np.float32)
        else:
            width = int(kv.shape[-1])
            kv_flat = kv.reshape(n, width)
            score_flat = score.reshape(n, width)
            if row_indices is not None:
                kv_flat = kv_flat[row_indices]
                score_flat = score_flat[row_indices]
            kv_flat = np.ascontiguousarray(kv_flat)
            score_flat = np.ascontiguousarray(score_flat)
        write_kv_score_state_oracle(
            device_state.kv_score_state,
            kv_flat,
            score_flat,
            owner_ids_arr,
            positions,
            compressor.ape,
            spec=device_state.spec,
        )
        return

    ape_dev = _device_or_upload(
        compressor.ape,
        name="dsv4_comp_ape",
        dtype=_ml.bfloat16,
    )

    # Fast path: DeviceTensor inputs already shaped [n, width] bf16 from
    # the ``compressor_kv_score_bf16`` fragment. The scatter kernel handles
    # arbitrary final tail tiles, so this path does not need host slicing.
    if is_dev_kv:
        live_rows = None
        kv_shape = tuple(int(dim) for dim in getattr(kv, "shape", ()))
        if len(kv_shape) == 2 and int(kv_shape[0]) > int(owner_ids_arr.shape[0]):
            live_rows = _device_scalar_i32(
                int(owner_ids_arr.shape[0]),
                name="dsv4_comp_live_rows",
            )
            owner_ids_arr = _pad_i32_vector(owner_ids_arr, rows=int(kv_shape[0]))
            positions = _pad_i32_vector(positions, rows=int(kv_shape[0]), fill=0)
        run_write_kv_score_state_device(
            kv_score_state=device_state.kv_score_state,
            kv_new=kv,
            score_new=score,
            owner_ids=_device_vector_or_upload(
                owner_ids_dev,
                owner_ids_arr,
                name="dsv4_comp_owner_ids",
            ),
            positions=_device_vector_or_upload(
                positions_dev,
                positions,
                name="dsv4_comp_positions",
            ),
            ape=ape_dev,
            live_rows=live_rows,
            ring_size=int(device_state.ring_size),
            artifacts_dir=build_dir,
        )
        return

    width = int(kv.shape[-1])
    kv_flat = kv.reshape(n, width)
    score_flat = score.reshape(n, width)
    if row_indices is not None:
        kv_flat = kv_flat[row_indices]
        score_flat = score_flat[row_indices]
    kv_flat = np.ascontiguousarray(kv_flat.astype(_ml.bfloat16))
    score_flat = np.ascontiguousarray(score_flat.astype(_ml.bfloat16))

    DeviceTensor = get_device_tensor_cls()

    for offset in range(0, kv_flat.shape[0], 128):
        end = min(offset + 128, kv_flat.shape[0])
        run_write_kv_score_state_device(
            kv_score_state=device_state.kv_score_state,
            kv_new=DeviceTensor.from_numpy(
                np.ascontiguousarray(kv_flat[offset:end]),
                name="dsv4_comp_kv",
            ),
            score_new=DeviceTensor.from_numpy(
                np.ascontiguousarray(score_flat[offset:end]),
                name="dsv4_comp_score",
            ),
            owner_ids=DeviceTensor.from_numpy(
                np.ascontiguousarray(owner_ids_arr[offset:end]),
                name="dsv4_comp_owner_ids",
            ),
            positions=DeviceTensor.from_numpy(
                np.ascontiguousarray(positions[offset:end]),
                name="dsv4_comp_positions",
            ),
            ape=ape_dev,
            ring_size=int(device_state.ring_size),
            artifacts_dir=build_dir,
        )


def prefill_pool_from_device_slab(
    compressor: Any,
    kv: Any,
    score: Any,
    *,
    bsz: int,
    seqlen: int,
    build_dir: str | Path | None = None,
    device_state: Any | None = None,
    return_device: bool = False,
    output: Any | None = None,
) -> Any:
    """Prefill-pool compressor projections through the DSV4 slab kernel.

    ``kv`` / ``score`` may be numpy ``[bsz, seqlen, width]`` float or
    already-resident DeviceTensors of shape ``[bsz*seqlen, width] bf16``
    (the typical production input, produced by
    ``compressor_kv_score_bf16``). Returns
    ``[bsz, seqlen // ratio, head_dim]`` fp32 numpy by default, or the
    kernel's raw ``[bsz*(seqlen//ratio), head_dim]`` fp32 DeviceTensor
    when ``return_device=True`` so the post-qdq chain can consume it
    without a host round-trip. Falls back to the numpy oracle when
    ``device_state.kv_score_state`` is a host array.
    """
    import ml_dtypes as _ml

    ratio = int(compressor.compress_ratio)
    d = int(compressor.head_dim)
    n = int(bsz) * int(seqlen)
    is_dev_kv = _is_device_tensor_like(kv)

    if device_state is not None and not hasattr(
        device_state.kv_score_state,
        "tensor_ref",
    ):
        if is_dev_kv:
            if not hasattr(kv, "numpy") or not hasattr(score, "numpy"):
                raise RuntimeError(
                    "host prefill-pool fallback cannot consume "
                    "non-downloadable device tensors"
                )
            kv_flat = kv.numpy().astype(np.float32)
            score_flat = score.numpy().astype(np.float32)
        else:
            width = int(kv.shape[-1])
            kv_flat = np.ascontiguousarray(kv.reshape(n, width))
            score_flat = np.ascontiguousarray(score.reshape(n, width))
        return prefill_pool_from_slab_oracle(
            kv_flat,
            score_flat,
            compressor.ape,
            bsz=int(bsz),
            seqlen=int(seqlen),
            ratio=ratio,
            head_dim=d,
            overlap=bool(compressor.overlap),
        )

    DeviceTensor = get_device_tensor_cls()

    if is_dev_kv:
        kv_new_dev = kv
        score_new_dev = score
    else:
        width = int(kv.shape[-1])
        kv_new_dev = DeviceTensor.from_numpy(
            np.ascontiguousarray(kv.reshape(n, width).astype(_ml.bfloat16)),
            name="dsv4_prefill_pool_kv",
        )
        score_new_dev = DeviceTensor.from_numpy(
            np.ascontiguousarray(score.reshape(n, width).astype(_ml.bfloat16)),
            name="dsv4_prefill_pool_score",
        )

    pooled_dev = run_prefill_pool_from_slab_device(
        kv_new=kv_new_dev,
        score_new=score_new_dev,
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_prefill_pool_ape",
            dtype=_ml.bfloat16,
        ),
        bsz=int(bsz),
        seqlen=int(seqlen),
        ratio=ratio,
        head_dim=d,
        overlap=bool(compressor.overlap),
        artifacts_dir=build_dir,
        output=output,
    )
    if bool(return_device):
        return pooled_dev
    return np.asarray(pooled_dev.numpy(), dtype=np.float32).reshape(
        int(bsz),
        int(seqlen) // ratio,
        d,
    )


def decode_pool_from_device_state(
    device_state: Any,
    *,
    bsz: int,
    end_pos: int,
    build_dir: str | Path | None = None,
    return_device: bool = False,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    end_positions_dev: Any | None = None,
    output: Any | None = None,
) -> Any:
    """Decode-pool compressor state from device ring buffers.

    Handles both no-overlap and overlap (c4a) variants based on
    ``device_state.spec.overlap``. Returns ``[bsz, 1, head_dim]`` fp32
    numpy by default, or the kernel's raw ``[bsz, head_dim]`` fp32
    DeviceTensor when ``return_device=True``.
    """
    spec = device_state.spec
    if owner_ids is None:
        owner_ids_arr = np.arange(int(bsz), dtype=np.int32)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (int(bsz),):
            raise ValueError(
                f"owner_ids must be [bsz={int(bsz)}], got {owner_ids_arr.shape}"
            )
    end_positions = np.full((int(bsz),), int(end_pos), dtype=np.int32)
    if hasattr(device_state.kv_score_state, "tensor_ref"):
        pooled_dev = run_decode_pool_from_state_device(
            kv_score_state=device_state.kv_score_state,
            owner_ids=_device_vector_or_upload(
                owner_ids_dev,
                owner_ids_arr,
                name="dsv4_decode_pool_owner_ids",
            ),
            end_positions=_device_vector_or_upload(
                end_positions_dev,
                end_positions,
                name="dsv4_decode_pool_end_positions",
            ),
            ratio=int(spec.compress_ratio),
            head_dim=int(spec.head_dim),
            state_width=int(spec.state_width),
            ring_size=int(spec.ring_size),
            overlap=bool(spec.overlap),
            artifacts_dir=build_dir,
            output=output,
        )
        if bool(return_device):
            return pooled_dev
        pooled = pooled_dev.numpy()
    else:
        if bool(return_device):
            raise RuntimeError(
                "return_device=True requires a device-resident "
                "device_state.kv_score_state"
            )
        pooled = decode_pool_from_state_oracle(
            device_state.kv_score_state,
            owner_ids_arr,
            end_positions,
            spec=spec,
        )
    return np.asarray(pooled, dtype=np.float32).reshape(
        int(bsz),
        1,
        int(spec.head_dim),
    )


def run_compressor_decode_state_cache_scatter_device(
    compressor: Any,
    start_pos: int,
    *,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    bsz: int,
    device_state: Any,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
) -> None:
    """Fuse decode ring-state and compressed-cache writes for boundary tokens."""

    if int(start_pos) == 0:
        raise ValueError("decode fused state/cache scatter requires start_pos != 0")
    if not hasattr(device_state.kv_score_state, "tensor_ref") or not hasattr(
        device_state.compressed_kv_cache,
        "tensor_ref",
    ):
        raise RuntimeError("fused state/cache scatter requires device-resident state")
    if not _is_device_tensor_like(kv) or not _is_device_tensor_like(score):
        raise RuntimeError("fused state/cache scatter requires device kv/score")
    if not _is_device_tensor_like(scatter_rows):
        raise RuntimeError("fused state/cache scatter requires device scatter rows")

    bsz_i = int(bsz)
    if owner_ids is None:
        owner_ids_arr = np.arange(bsz_i, dtype=np.int32)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (bsz_i,):
            raise ValueError(
                f"owner_ids must be [bsz={bsz_i}], got {owner_ids_arr.shape}"
            )
    positions = np.full((bsz_i,), int(start_pos), dtype=np.int32)
    owner_ids_arg = _device_vector_or_upload(
        owner_ids_dev,
        owner_ids_arr,
        name="dsv4_comp_decode_owner_ids",
    )
    positions_arg = _device_vector_or_upload(
        positions_dev,
        positions,
        name="dsv4_comp_decode_positions",
    )
    run_write_kv_score_state_owner_pos_device(
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        kv_new=kv,
        score_new=score,
        compressed_rows=scatter_rows,
        owner_ids=owner_ids_arg,
        positions=positions_arg,
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        ring_size=int(device_state.ring_size),
        ratio=int(compressor.compress_ratio),
        max_clen=int(device_state.spec.max_compressed_len),
        artifacts_dir=build_dir,
    )


def run_compressor_decode_state_swa_scatter_device(
    compressor: Any,
    start_pos: int,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    kv: Any,
    score: Any,
    bsz: int,
    device_state: Any,
    window_size: int,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
) -> None:
    """Fuse decode SWA-cache and compressor ring-state writes."""

    if int(start_pos) == 0:
        raise ValueError("decode fused SWA/state scatter requires start_pos != 0")
    if not hasattr(swa_kv_cache, "tensor_ref") or not hasattr(
        device_state.kv_score_state,
        "tensor_ref",
    ):
        raise RuntimeError("fused SWA/state scatter requires device-resident state")
    if (
        not _is_device_tensor_like(swa_rows)
        or not _is_device_tensor_like(kv)
        or not _is_device_tensor_like(score)
    ):
        raise RuntimeError("fused SWA/state scatter requires device rows")

    bsz_i = int(bsz)
    if owner_ids is None:
        owner_ids_arr = np.arange(bsz_i, dtype=np.int32)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (bsz_i,):
            raise ValueError(
                f"owner_ids must be [bsz={bsz_i}], got {owner_ids_arr.shape}"
            )
    positions = np.full((bsz_i,), int(start_pos), dtype=np.int32)
    owner_ids_arg = _device_vector_or_upload(
        owner_ids_dev,
        owner_ids_arr,
        name="dsv4_comp_decode_swa_owner_ids",
    )
    positions_arg = _device_vector_or_upload(
        positions_dev,
        positions,
        name="dsv4_comp_decode_swa_positions",
    )
    run_write_swa_kv_score_state_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        swa_rows=swa_rows,
        kv_new=kv,
        score_new=score,
        owner_ids=owner_ids_arg,
        positions=positions_arg,
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        window_size=int(window_size),
        ring_size=int(device_state.ring_size),
        artifacts_dir=build_dir,
    )


def run_compressor_decode_dual_state_swa_scatter_device(
    compressor: Any,
    start_pos: int,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    kv: Any,
    score: Any,
    indexer_compressor: Any,
    indexer_kv: Any,
    indexer_score: Any,
    bsz: int,
    device_state: Any,
    indexer_device_state: Any,
    window_size: int,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
) -> None:
    """Fuse decode SWA plus main and indexer compressor ring-state writes."""

    if int(start_pos) == 0:
        raise ValueError("decode fused SWA/dual-state scatter requires start_pos != 0")
    if not hasattr(swa_kv_cache, "tensor_ref"):
        raise RuntimeError("fused SWA/dual-state scatter requires device SWA cache")
    if not hasattr(device_state.kv_score_state, "tensor_ref") or not hasattr(
        indexer_device_state.kv_score_state,
        "tensor_ref",
    ):
        raise RuntimeError("fused SWA/dual-state scatter requires device state")
    if (
        not _is_device_tensor_like(swa_rows)
        or not _is_device_tensor_like(kv)
        or not _is_device_tensor_like(score)
        or not _is_device_tensor_like(indexer_kv)
        or not _is_device_tensor_like(indexer_score)
    ):
        raise RuntimeError("fused SWA/dual-state scatter requires device rows")

    bsz_i = int(bsz)
    if owner_ids is None:
        owner_ids_arr = np.arange(bsz_i, dtype=np.int32)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (bsz_i,):
            raise ValueError(
                f"owner_ids must be [bsz={bsz_i}], got {owner_ids_arr.shape}"
            )
    positions = np.full((bsz_i,), int(start_pos), dtype=np.int32)
    owner_ids_arg = _device_vector_or_upload(
        owner_ids_dev,
        owner_ids_arr,
        name="dsv4_comp_decode_swa_dual_owner_ids",
    )
    positions_arg = _device_vector_or_upload(
        positions_dev,
        positions,
        name="dsv4_comp_decode_swa_dual_positions",
    )
    run_write_swa_dual_kv_score_state_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        indexer_kv_score_state=indexer_device_state.kv_score_state,
        swa_rows=swa_rows,
        kv_new=kv,
        score_new=score,
        indexer_kv_new=indexer_kv,
        indexer_score_new=indexer_score,
        owner_ids=owner_ids_arg,
        positions=positions_arg,
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        indexer_ape=_device_or_upload(
            indexer_compressor.ape,
            name="dsv4_indexer_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        window_size=int(window_size),
        ring_size=int(device_state.ring_size),
        indexer_ring_size=int(indexer_device_state.ring_size),
        artifacts_dir=build_dir,
    )


def run_compressor_decode_dual_state_cache_swa_scatter_device(
    compressor: Any,
    start_pos: int,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    indexer_compressor: Any,
    indexer_kv: Any,
    indexer_score: Any,
    indexer_scatter_rows: Any,
    bsz: int,
    device_state: Any,
    indexer_device_state: Any,
    window_size: int,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
) -> None:
    """Fuse boundary decode SWA plus main/indexer compressor state/cache writes."""

    if int(start_pos) == 0:
        raise ValueError(
            "decode fused SWA/dual-state/cache scatter requires start_pos != 0"
        )
    if (
        not hasattr(swa_kv_cache, "tensor_ref")
        or not hasattr(device_state.kv_score_state, "tensor_ref")
        or not hasattr(device_state.compressed_kv_cache, "tensor_ref")
        or not hasattr(indexer_device_state.kv_score_state, "tensor_ref")
        or not hasattr(indexer_device_state.compressed_kv_cache, "tensor_ref")
    ):
        raise RuntimeError(
            "fused SWA/dual-state/cache scatter requires device-resident caches"
        )
    if (
        not _is_device_tensor_like(swa_rows)
        or not _is_device_tensor_like(kv)
        or not _is_device_tensor_like(score)
        or not _is_device_tensor_like(scatter_rows)
        or not _is_device_tensor_like(indexer_kv)
        or not _is_device_tensor_like(indexer_score)
        or not _is_device_tensor_like(indexer_scatter_rows)
    ):
        raise RuntimeError("fused SWA/dual-state/cache scatter requires device rows")

    bsz_i = int(bsz)
    if owner_ids is None:
        owner_ids_arr = np.arange(bsz_i, dtype=np.int32)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (bsz_i,):
            raise ValueError(
                f"owner_ids must be [bsz={bsz_i}], got {owner_ids_arr.shape}"
            )
    positions = np.full((bsz_i,), int(start_pos), dtype=np.int32)
    owner_ids_arg = _device_vector_or_upload(
        owner_ids_dev,
        owner_ids_arr,
        name="dsv4_comp_decode_swa_dual_cache_owner_ids",
    )
    positions_arg = _device_vector_or_upload(
        positions_dev,
        positions,
        name="dsv4_comp_decode_swa_dual_cache_positions",
    )
    run_write_swa_dual_kv_score_state_owner_pos_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        indexer_kv_score_state=indexer_device_state.kv_score_state,
        indexer_compressed_kv_cache=indexer_device_state.compressed_kv_cache,
        swa_rows=swa_rows,
        kv_new=kv,
        score_new=score,
        compressed_rows=scatter_rows,
        indexer_kv_new=indexer_kv,
        indexer_score_new=indexer_score,
        indexer_compressed_rows=indexer_scatter_rows,
        owner_ids=owner_ids_arg,
        positions=positions_arg,
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        indexer_ape=_device_or_upload(
            indexer_compressor.ape,
            name="dsv4_indexer_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        window_size=int(window_size),
        ring_size=int(device_state.ring_size),
        indexer_ring_size=int(indexer_device_state.ring_size),
        ratio=int(compressor.compress_ratio),
        indexer_ratio=int(indexer_compressor.compress_ratio),
        max_clen=int(device_state.spec.max_compressed_len),
        indexer_max_clen=int(indexer_device_state.spec.max_compressed_len),
        artifacts_dir=build_dir,
    )


def run_compressor_decode_state_cache_swa_scatter_device(
    compressor: Any,
    start_pos: int,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    bsz: int,
    device_state: Any,
    window_size: int,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
) -> None:
    """Fuse boundary decode SWA, ring-state, and compressed-cache writes."""

    if int(start_pos) == 0:
        raise ValueError("decode fused SWA/state/cache scatter requires start_pos != 0")
    if (
        not hasattr(swa_kv_cache, "tensor_ref")
        or not hasattr(device_state.kv_score_state, "tensor_ref")
        or not hasattr(device_state.compressed_kv_cache, "tensor_ref")
    ):
        raise RuntimeError(
            "fused SWA/state/cache scatter requires device-resident state"
        )
    if (
        not _is_device_tensor_like(swa_rows)
        or not _is_device_tensor_like(kv)
        or not _is_device_tensor_like(score)
        or not _is_device_tensor_like(scatter_rows)
    ):
        raise RuntimeError("fused SWA/state/cache scatter requires device rows")

    bsz_i = int(bsz)
    if owner_ids is None:
        owner_ids_arr = np.arange(bsz_i, dtype=np.int32)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (bsz_i,):
            raise ValueError(
                f"owner_ids must be [bsz={bsz_i}], got {owner_ids_arr.shape}"
            )
    positions = np.full((bsz_i,), int(start_pos), dtype=np.int32)
    owner_ids_arg = _device_vector_or_upload(
        owner_ids_dev,
        owner_ids_arr,
        name="dsv4_comp_decode_swa_owner_ids",
    )
    positions_arg = _device_vector_or_upload(
        positions_dev,
        positions,
        name="dsv4_comp_decode_swa_positions",
    )
    run_write_swa_kv_score_state_owner_pos_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        swa_rows=swa_rows,
        kv_new=kv,
        score_new=score,
        compressed_rows=scatter_rows,
        owner_ids=owner_ids_arg,
        positions=positions_arg,
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        window_size=int(window_size),
        ring_size=int(device_state.ring_size),
        ratio=int(compressor.compress_ratio),
        max_clen=int(device_state.spec.max_compressed_len),
        artifacts_dir=build_dir,
    )


def run_compressor_prefill_state_cache_scatter_device(
    compressor: Any,
    *,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    bsz: int,
    seqlen: int,
    clen: int,
    device_state: Any,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
    owner_id_stride: int | None = None,
) -> None:
    """Fuse prefill ring-state tail and compressed-cache writes."""

    if not hasattr(device_state.kv_score_state, "tensor_ref") or not hasattr(
        device_state.compressed_kv_cache,
        "tensor_ref",
    ):
        raise RuntimeError("fused prefill scatter requires device-resident state")
    if not _is_device_tensor_like(kv) or not _is_device_tensor_like(score):
        raise RuntimeError("fused prefill scatter requires device kv/score")
    if not _is_device_tensor_like(scatter_rows):
        raise RuntimeError("fused prefill scatter requires device scatter rows")

    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    n = bsz_i * seqlen_i
    if owner_ids is None:
        owner_ids_arr = np.repeat(np.arange(bsz_i, dtype=np.int32), seqlen_i)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (n,):
            raise ValueError(f"owner_ids must be [{n}], got {owner_ids_arr.shape}")
    positions = np.tile(np.arange(seqlen_i, dtype=np.int32), bsz_i)
    state_owner_ids_dev = owner_ids_dev
    state_positions_dev = positions_dev
    keep_tokens = _prefill_state_tail_len(spec=device_state.spec, seqlen=seqlen_i)
    if keep_tokens == 0:
        raise RuntimeError("fused prefill scatter requires non-empty state tail")
    row_indices: np.ndarray | None = None
    if keep_tokens < seqlen_i and bsz_i == 1:
        start = seqlen_i - int(keep_tokens)
        row_indices = np.arange(start, seqlen_i, dtype=np.int64)
        kv_alias = _alias_device_value_first_dim_slice(
            kv,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        score_alias = _alias_device_value_first_dim_slice(
            score,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        if kv_alias is not None and score_alias is not None:
            kv = kv_alias
            score = score_alias
            if owner_ids_dev is not None:
                state_owner_ids_dev = _alias_device_value_first_dim_slice(
                    owner_ids_dev,
                    start=int(row_indices[0]),
                    size=int(row_indices.shape[0]),
                )
            if positions_dev is not None:
                state_positions_dev = _alias_device_value_first_dim_slice(
                    positions_dev,
                    start=int(row_indices[0]),
                    size=int(row_indices.shape[0]),
                )
        else:
            row_indices = None
    if row_indices is not None:
        state_owner_ids_arr = np.ascontiguousarray(owner_ids_arr[row_indices])
        state_positions_arr = np.ascontiguousarray(positions[row_indices])
    else:
        state_owner_ids_arr = owner_ids_arr
        state_positions_arr = positions

    stride_i = int(owner_id_stride) if owner_id_stride is not None else seqlen_i
    run_write_kv_score_state_owner_clen_device(
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        kv_new=kv,
        score_new=score,
        compressed_rows=scatter_rows,
        state_owner_ids=_device_vector_or_upload(
            state_owner_ids_dev,
            state_owner_ids_arr,
            name="dsv4_comp_prefill_state_owner_ids",
        ),
        state_positions=_device_vector_or_upload(
            state_positions_dev,
            state_positions_arr,
            name="dsv4_comp_prefill_state_positions",
        ),
        cache_owner_ids=_device_vector_or_upload(
            owner_ids_dev,
            owner_ids_arr,
            name="dsv4_comp_prefill_cache_owner_ids",
        ),
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        ring_size=int(device_state.ring_size),
        clen=int(clen),
        owner_id_stride=stride_i,
        max_clen=int(device_state.spec.max_compressed_len),
        artifacts_dir=build_dir,
    )


_BKT1_APE_CACHE: dict[tuple[int, int, int], Any] = {}


def _bkt1_full_width_ape(ape: Any, *, ratio: int, width: int) -> Any:
    """Device ``[ratio, width]`` broadcast of ``ape`` for the masked NKI write.

    The ape table is a per-layer constant; memoize per (table, ratio, width)
    so the broadcast + upload happens once instead of every prefill step.
    """
    key = (id(ape), int(ratio), int(width))
    cached = _BKT1_APE_CACHE.get(key)
    if cached is not None:
        return cached
    arr = ape.numpy() if hasattr(ape, "numpy") else np.asarray(ape)
    arr = np.asarray(arr, dtype=ml_dtypes.bfloat16).reshape(int(ratio), -1)
    if arr.shape[1] != int(width):
        arr = np.broadcast_to(arr, (int(ratio), int(width)))
    dev = get_device_tensor_cls().from_numpy(
        np.ascontiguousarray(arr), name="dsv4_bkt1_ape"
    )
    _BKT1_APE_CACHE[key] = dev
    return dev


def _run_bucketed_prefill_single_scatter(
    compressor: Any,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    bsz: int,
    bucket_seqlen: int,
    real_seqlen: int,
    clen: int,
    device_state: Any,
    window_size: int,
    build_dir: str | Path | None,
    request_owners: np.ndarray | None = None,
) -> None:
    """Bucketed single-compressor prefill SWA + state + cache scatter.

    Three-cache sibling of ``_run_bucketed_prefill_dual_scatter`` for layers
    without an indexer (the token-topk family). Owner/pos arrays are built at
    the COMPILE bucket with guard-owner padding; only the tail-window tokens
    write SWA; the compressed write masks columns past real//ratio.
    """
    bsz_i = int(bsz)
    bucket_i = int(bucket_seqlen)
    real_i = int(real_seqlen)
    spec = device_state.spec
    ratio = int(spec.compress_ratio)
    win = int(window_size)
    guard_owner = int(spec.guard_owner)

    owners = (
        np.arange(bsz_i, dtype=np.int32)
        if request_owners is None
        else np.asarray(request_owners, dtype=np.int32).reshape(bsz_i)
    )
    # The flat KV (swa_rows) is bucket rows when the prologue re-aliased, real
    # rows when it could not; size the SWA owner map to its actual rows (mirror
    # the dual scatter) so owner/pos shapes always match swa_rows.
    swa_total = int(getattr(swa_rows, "shape", (bsz_i * bucket_i,))[0])
    swa_seq = max(1, swa_total // max(1, bsz_i))
    swa_owner, swa_pos = _bucketed_prefill_swa_owner_pos(
        bsz=bsz_i,
        bucket_seqlen=swa_seq,
        real_seqlen=min(real_i, swa_seq),
        window_size=win,
        guard_owner=guard_owner,
        request_owners=owners,
    )

    keep = _prefill_state_tail_len(spec=spec, seqlen=real_i)
    max_keep = (2 * ratio - 1) if bool(spec.overlap) else (ratio - 1 or 1)
    if keep > max_keep:
        raise RuntimeError(f"keep {keep} > max_keep {max_keep} (ratio {ratio})")
    if bucket_i < max_keep:
        raise RuntimeError(
            f"bucketed prefill needs token_bucket>={max_keep}, got {bucket_i}"
        )
    tail_start = max(0, real_i - max_keep)
    if bsz_i != 1:
        raise RuntimeError(
            "bucketed prefill single scatter supports bsz==1 state tail "
            f"aliasing, got bsz={bsz_i}"
        )
    state_kv = _alias_device_value_first_dim_slice(
        kv, start=int(tail_start), size=int(max_keep)
    )
    state_score = _alias_device_value_first_dim_slice(
        score, start=int(tail_start), size=int(max_keep)
    )
    if state_kv is None or state_score is None:
        raise RuntimeError(
            "bucketed prefill single scatter requires state tail aliases"
        )
    tok = tail_start + np.arange(max_keep, dtype=np.int32)
    live = (tok >= real_i - keep) & (tok < real_i)
    state_owner = np.where(live, owners[0], np.int32(guard_owner)).astype(np.int32)
    state_pos = np.where(live, tok, np.int32(0)).astype(np.int32)

    cache_owner = np.full(int(spec.num_state_owners), guard_owner, dtype=np.int32)
    cache_owner[:bsz_i] = owners
    run_write_swa_kv_score_state_owner_clen_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        swa_rows=swa_rows,
        kv_new=state_kv,
        score_new=state_score,
        compressed_rows=scatter_rows,
        swa_owner_ids=_bkt_owner_dev(
            swa_owner,
            key=(
                "bkt1",
                "swa_o",
                (
                    bsz_i,
                    bucket_i,
                    swa_seq,
                    real_i,
                    win,
                    guard_owner,
                    tuple(owners.tolist()),
                ),
            ),
            name="dsv4_bkt1_swa_owner",
        ),
        swa_positions=_bkt_owner_dev(
            swa_pos,
            key=(
                "bkt1",
                "swa_p",
                (
                    bsz_i,
                    bucket_i,
                    swa_seq,
                    real_i,
                    win,
                    guard_owner,
                    tuple(owners.tolist()),
                ),
            ),
            name="dsv4_bkt1_swa_pos",
        ),
        state_owner_ids=_bkt_owner_dev(
            state_owner,
            key=(
                "bkt1",
                "st_o",
                (bsz_i, bucket_i, real_i, win, guard_owner, tuple(owners.tolist())),
            ),
            name="dsv4_bkt1_state_owner",
        ),
        state_positions=_bkt_owner_dev(
            state_pos,
            key=(
                "bkt1",
                "st_p",
                (bsz_i, bucket_i, real_i, win, guard_owner, tuple(owners.tolist())),
            ),
            name="dsv4_bkt1_state_pos",
        ),
        cache_owner_ids=_bkt_owner_dev(
            cache_owner,
            key=(
                "bkt1",
                "c_o",
                (bsz_i, bucket_i, real_i, win, guard_owner, tuple(owners.tolist())),
            ),
            name="dsv4_bkt1_cache_owner",
        ),
        # this family's ape can be [ratio, 1]; the NKI write reads full width
        ape=_bkt1_full_width_ape(
            compressor.ape,
            ratio=ratio,
            width=int(getattr(state_kv, "shape")[1]),
        ),
        window_size=win,
        ring_size=int(spec.ring_size),
        clen=int(clen),
        owner_id_stride=1,
        max_clen=int(spec.max_compressed_len),
        cache_real_clen=int(real_i // ratio),
        guard_owner=guard_owner,
        artifacts_dir=build_dir,
    )


def run_compressor_prefill_state_cache_swa_scatter_device(
    compressor: Any,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    swa_start_pos: int,
    swa_bsz: int,
    swa_seqlen: int,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    bsz: int,
    seqlen: int,
    clen: int,
    device_state: Any,
    window_size: int,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
    swa_owner_ids: np.ndarray | None = None,
    swa_owner_ids_dev: Any | None = None,
    swa_positions_dev: Any | None = None,
    owner_id_stride: int | None = None,
    real_seqlen: int | None = None,
) -> None:
    """Fuse prefill SWA mirror, ring-state tail, and compressed-cache writes.

    ``real_seqlen`` (bucketed mode): when given, ``seqlen`` is the compile
    bucket and the masked scatter writes only the real rows (one NEFF/bucket).
    """
    if real_seqlen is not None:
        request_owners = None
        if owner_ids is not None:
            request_owners = _request_owners_from_flat(owner_ids, bsz=int(bsz))
        _run_bucketed_prefill_single_scatter(
            compressor,
            swa_kv_cache=swa_kv_cache,
            swa_rows=swa_rows,
            kv=kv,
            score=score,
            scatter_rows=scatter_rows,
            bsz=int(bsz),
            bucket_seqlen=int(seqlen),
            real_seqlen=int(real_seqlen),
            clen=int(clen),
            device_state=device_state,
            window_size=int(window_size),
            build_dir=build_dir,
            request_owners=request_owners,
        )
        return

    if (
        not hasattr(swa_kv_cache, "tensor_ref")
        or not hasattr(device_state.kv_score_state, "tensor_ref")
        or not hasattr(device_state.compressed_kv_cache, "tensor_ref")
    ):
        raise RuntimeError("fused prefill SWA scatter requires device-resident caches")
    if (
        not _is_device_tensor_like(swa_rows)
        or not _is_device_tensor_like(kv)
        or not _is_device_tensor_like(score)
        or not _is_device_tensor_like(scatter_rows)
    ):
        raise RuntimeError("fused prefill SWA scatter requires device rows")

    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    n = bsz_i * seqlen_i
    if owner_ids is None:
        owner_ids_arr = np.repeat(np.arange(bsz_i, dtype=np.int32), seqlen_i)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (n,):
            raise ValueError(f"owner_ids must be [{n}], got {owner_ids_arr.shape}")
    positions = np.tile(np.arange(seqlen_i, dtype=np.int32), bsz_i)

    swa_bsz_i = int(swa_bsz)
    swa_seqlen_i = int(swa_seqlen)
    swa_n = swa_bsz_i * swa_seqlen_i
    swa_rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape", ()))
    actual_swa_n = int(swa_rows_shape[0]) if len(swa_rows_shape) == 2 else int(swa_n)
    if swa_owner_ids is None:
        if swa_bsz_i != bsz_i or swa_seqlen_i > seqlen_i:
            raise ValueError(
                "swa_owner_ids is required when SWA mirror shape differs from "
                "the compressor token rectangle"
            )
        source_start = int(swa_start_pos)
        if source_start < 0 or source_start + swa_seqlen_i > seqlen_i:
            source_start = seqlen_i - swa_seqlen_i
        owners_rect = owner_ids_arr.reshape(bsz_i, seqlen_i)
        swa_owner_ids_arr = np.ascontiguousarray(
            owners_rect[:, source_start : source_start + swa_seqlen_i].reshape(-1)
        )
    else:
        swa_owner_ids_arr = np.asarray(swa_owner_ids, dtype=np.int32).reshape(-1)
        if swa_owner_ids_arr.shape != (swa_n,):
            raise ValueError(
                f"swa_owner_ids must be [{swa_n}], got {swa_owner_ids_arr.shape}"
            )
    swa_positions = np.tile(
        np.arange(
            int(swa_start_pos),
            int(swa_start_pos) + swa_seqlen_i,
            dtype=np.int32,
        ),
        swa_bsz_i,
    )
    if actual_swa_n != swa_n:
        if swa_bsz_i <= 0 or actual_swa_n % swa_bsz_i != 0:
            raise ValueError(
                "bucketed SWA rows must divide by SWA batch: "
                f"rows={actual_swa_n}, bsz={swa_bsz_i}"
            )
        request_owners = _request_owners_from_flat(owner_ids_arr, bsz=bsz_i)
        if request_owners is None:
            request_owners = _request_owners_from_flat(
                swa_owner_ids_arr,
                bsz=swa_bsz_i,
            )
        swa_owner_ids_arr, swa_positions = _bucketed_prefill_swa_owner_pos(
            bsz=swa_bsz_i,
            bucket_seqlen=actual_swa_n // swa_bsz_i,
            real_seqlen=min(
                int(swa_start_pos) + int(swa_seqlen_i),
                actual_swa_n // swa_bsz_i,
            ),
            window_size=int(window_size),
            guard_owner=int(device_state.spec.guard_owner),
            request_owners=request_owners,
        )

    state_owner_ids_dev = owner_ids_dev
    state_positions_dev = positions_dev
    keep_tokens = _prefill_state_tail_len(spec=device_state.spec, seqlen=seqlen_i)
    if keep_tokens == 0:
        raise RuntimeError("fused prefill SWA scatter requires non-empty state tail")
    row_indices: np.ndarray | None = None
    if keep_tokens < seqlen_i and bsz_i == 1:
        start = seqlen_i - int(keep_tokens)
        row_indices = np.arange(start, seqlen_i, dtype=np.int64)
        kv_alias = _alias_device_value_first_dim_slice(
            kv,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        score_alias = _alias_device_value_first_dim_slice(
            score,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        if kv_alias is not None and score_alias is not None:
            kv = kv_alias
            score = score_alias
            if owner_ids_dev is not None:
                state_owner_ids_dev = _alias_device_value_first_dim_slice(
                    owner_ids_dev,
                    start=int(row_indices[0]),
                    size=int(row_indices.shape[0]),
                )
            if positions_dev is not None:
                state_positions_dev = _alias_device_value_first_dim_slice(
                    positions_dev,
                    start=int(row_indices[0]),
                    size=int(row_indices.shape[0]),
                )
        else:
            row_indices = None
    if row_indices is not None:
        state_owner_ids_arr = np.ascontiguousarray(owner_ids_arr[row_indices])
        state_positions_arr = np.ascontiguousarray(positions[row_indices])
    else:
        state_owner_ids_arr = owner_ids_arr
        state_positions_arr = positions

    stride_i = int(owner_id_stride) if owner_id_stride is not None else seqlen_i
    run_write_swa_kv_score_state_owner_clen_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        swa_rows=swa_rows,
        kv_new=kv,
        score_new=score,
        compressed_rows=scatter_rows,
        swa_owner_ids=_device_vector_or_upload(
            swa_owner_ids_dev,
            swa_owner_ids_arr,
            name="dsv4_comp_prefill_swa_owner_ids",
        ),
        swa_positions=_device_vector_or_upload(
            swa_positions_dev,
            swa_positions,
            name="dsv4_comp_prefill_swa_positions",
        ),
        state_owner_ids=_device_vector_or_upload(
            state_owner_ids_dev,
            state_owner_ids_arr,
            name="dsv4_comp_prefill_state_owner_ids",
        ),
        state_positions=_device_vector_or_upload(
            state_positions_dev,
            state_positions_arr,
            name="dsv4_comp_prefill_state_positions",
        ),
        cache_owner_ids=_device_vector_or_upload(
            owner_ids_dev,
            owner_ids_arr,
            name="dsv4_comp_prefill_cache_owner_ids",
        ),
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        window_size=int(window_size),
        ring_size=int(device_state.ring_size),
        clen=int(clen),
        owner_id_stride=stride_i,
        max_clen=int(device_state.spec.max_compressed_len),
        artifacts_dir=build_dir,
    )


def _bucketed_prefill_swa_owner_pos(
    *,
    bsz: int,
    bucket_seqlen: int,
    real_seqlen: int,
    window_size: int,
    guard_owner: int,
    request_owners: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SWA owner/pos arrays for the bucketed prefill scatter.

    The SWA slot map is owner*win + pos%win, so only the LAST `win` real tokens
    may write: positions p and p+win collide on a slot with no scatter-order
    guarantee. The last min(real, win) tokens are pairwise distinct mod win
    (collision-free) and exactly reproduce the legacy fused tail-slice write.
    Older real tokens and bucket padding both go to the guard owner.
    """
    bsz_i = int(bsz)
    bucket_i = int(bucket_seqlen)
    real_i = int(real_seqlen)
    win = int(window_size)
    owners = (
        np.arange(bsz_i, dtype=np.int32)
        if request_owners is None
        else np.asarray(request_owners, dtype=np.int32).reshape(bsz_i)
    )
    tail_start = max(0, real_i - win)
    pos = np.tile(np.arange(bucket_i, dtype=np.int32), bsz_i)
    live = (pos >= tail_start) & (pos < real_i)
    swa_owner = np.where(
        live, np.repeat(owners, bucket_i), np.int32(guard_owner)
    ).astype(np.int32)
    swa_pos = np.where(live, pos, np.int32(0)).astype(np.int32)
    return swa_owner, swa_pos


def _request_owners_from_flat(
    owner_ids: np.ndarray | None,
    *,
    bsz: int,
) -> np.ndarray | None:
    if owner_ids is None:
        return None
    arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
    bsz_i = int(bsz)
    if bsz_i <= 0 or arr.size < bsz_i:
        return None
    stride = max(1, arr.size // bsz_i)
    return np.ascontiguousarray(arr[::stride][:bsz_i])


def _run_bucketed_prefill_dual_scatter(
    compressor: Any,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    indexer_compressor: Any,
    indexer_kv: Any,
    indexer_score: Any,
    indexer_scatter_rows: Any,
    bsz: int,
    bucket_seqlen: int,
    real_seqlen: int,
    clen: int,
    device_state: Any,
    indexer_device_state: Any,
    window_size: int,
    build_dir: str | Path | None,
    request_owners: np.ndarray | None = None,
) -> None:
    """Bucketed prefill dual-state/cache/SWA scatter (start_pos==0).

    Builds the owner/position arrays at the COMPILE bucket dims with padding
    redirected to the guard owner, then calls the masked fused NKI write. The
    write kernel is compiled once per bucket; ``real_seqlen`` enters only as the
    ``cache_real_clen`` runtime scalar. Recipe matches
    tests/test_dsv4_writeswa_bucket_device.py (host+device validated).
    """
    bsz_i = int(bsz)
    bucket_i = int(bucket_seqlen)
    real_i = int(real_seqlen)
    spec = device_state.spec
    idx_spec = indexer_device_state.spec
    ratio = int(spec.compress_ratio)
    ring = int(spec.ring_size)
    indexer_ring = int(idx_spec.ring_size)
    win = int(window_size)
    guard_owner = int(spec.guard_owner)
    if int(idx_spec.guard_owner) != guard_owner:
        raise RuntimeError(
            "main/indexer guard owners differ: "
            f"{guard_owner} vs {int(idx_spec.guard_owner)}"
        )

    owners = (
        np.arange(bsz_i, dtype=np.int32)
        if request_owners is None
        else np.asarray(request_owners, dtype=np.int32).reshape(bsz_i)
    )
    # The flat KV can be bucket rows (re-aliased prologue) or real rows
    # (lane without a bucket-sized backing buffer); size the owner map to it.
    swa_total = int(getattr(swa_rows, "shape", (bsz_i * bucket_i,))[0])
    swa_seq = max(1, swa_total // max(1, bsz_i))
    swa_owner, swa_pos = _bucketed_prefill_swa_owner_pos(
        bsz=bsz_i,
        bucket_seqlen=swa_seq,
        real_seqlen=min(real_i, swa_seq),
        window_size=win,
        guard_owner=guard_owner,
        request_owners=owners,
    )

    # ---- state tail: bsz*MAXKEEP; first `keep` real, rest -> guard ----
    keep = _prefill_state_tail_len(spec=spec, seqlen=real_i)
    max_keep = (2 * ratio - 1) if bool(spec.overlap) else (ratio - 1 or 1)
    if keep > max_keep:
        raise RuntimeError(f"keep {keep} > max_keep {max_keep} (ratio {ratio})")
    if bucket_i < max_keep:
        raise RuntimeError(
            f"bucketed prefill needs token_bucket>={max_keep}, got {bucket_i}"
        )
    # The state value slab covers tokens [tail_start, tail_start+max_keep) where
    # tail_start = max(0, real - max_keep). When real >= max_keep this is the true
    # tail; when real < max_keep it starts at 0 and the high rows are real tokens
    # too (all <= real-1), so the position guard below (tok in [real-keep, real))
    # keeps exactly the live tail and redirects the rest to the guard owner.
    tail_start = max(0, real_i - max_keep)
    n_state = bsz_i * max_keep
    state_owner = np.empty(n_state, dtype=np.int32)
    state_pos = np.empty(n_state, dtype=np.int32)

    # State value rows are the `max_keep` token projections per request starting
    # at tail_start. The prologue emits comp_kv/comp_score at bucket rows
    # [bsz*bucket]; slice each request's tail window (always in-range since
    # tail_start + max_keep <= bucket).
    def _state_tail_alias(value: Any) -> Any:
        rows = []
        for b in range(bsz_i):
            start = b * bucket_i + tail_start
            rows.append(
                _alias_device_value_first_dim_slice(
                    value, start=int(start), size=int(max_keep)
                )
            )
        if any(r is None for r in rows):
            return None
        if bsz_i == 1:
            return rows[0]
        return None  # multi-request non-contiguous tail: handled below

    # For bsz==1 the tail is a single contiguous slice; for bsz>1 we need a
    # gathered copy. Keep bsz==1 on the fast alias path (the shipped config).
    state_kv = _state_tail_alias(kv)
    state_score = _state_tail_alias(score)
    state_idx_kv = _state_tail_alias(indexer_kv)
    state_idx_score = _state_tail_alias(indexer_score)
    if (
        state_kv is None
        or state_score is None
        or state_idx_kv is None
        or state_idx_score is None
    ):
        raise RuntimeError(
            "bucketed prefill dual scatter currently supports bsz==1 state tail "
            f"aliasing (got bsz={bsz_i}); multi-request tail gather not wired"
        )

    # state positions/owners for the (max_keep) tail layout built above are
    # already MAXKEEP-major; align owner_id padding: the value tail covers the
    # last max_keep tokens, but only the last `keep` are real (positions
    # real-keep..real-1). Rows [0, max_keep-keep) of the tail slice are tokens
    # (real-max_keep .. real-keep) that must NOT be written -> guard them.
    tok = np.tile(tail_start + np.arange(max_keep, dtype=np.int32), bsz_i)
    live = (tok >= real_i - keep) & (tok < real_i)
    state_owner[:] = np.where(live, np.repeat(owners, max_keep), np.int32(guard_owner))
    state_pos[:] = np.where(live, tok, np.int32(0))

    cache_owner = np.full(int(spec.num_state_owners), guard_owner, dtype=np.int32)
    cache_owner[:bsz_i] = owners
    real_clen = real_i // ratio
    max_clen = int(spec.max_compressed_len)
    indexer_max_clen = int(idx_spec.max_compressed_len)

    run_write_swa_dual_kv_score_state_owner_clen_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        indexer_kv_score_state=indexer_device_state.kv_score_state,
        indexer_compressed_kv_cache=indexer_device_state.compressed_kv_cache,
        swa_rows=swa_rows,
        kv_new=state_kv,
        score_new=state_score,
        compressed_rows=scatter_rows,
        indexer_kv_new=state_idx_kv,
        indexer_score_new=state_idx_score,
        indexer_compressed_rows=indexer_scatter_rows,
        swa_owner_ids=_bkt_owner_dev(
            swa_owner,
            key=(
                "bkt",
                "swa_o",
                (
                    bsz_i,
                    bucket_i,
                    swa_seq,
                    real_i,
                    win,
                    guard_owner,
                    tuple(owners.tolist()),
                ),
            ),
            name="dsv4_bkt_swa_owner",
        ),
        swa_positions=_bkt_owner_dev(
            swa_pos,
            key=(
                "bkt",
                "swa_p",
                (
                    bsz_i,
                    bucket_i,
                    swa_seq,
                    real_i,
                    win,
                    guard_owner,
                    tuple(owners.tolist()),
                ),
            ),
            name="dsv4_bkt_swa_pos",
        ),
        state_owner_ids=_bkt_owner_dev(
            state_owner,
            key=(
                "bkt",
                "st_o",
                (bsz_i, bucket_i, real_i, win, guard_owner, tuple(owners.tolist())),
            ),
            name="dsv4_bkt_state_owner",
        ),
        state_positions=_bkt_owner_dev(
            state_pos,
            key=(
                "bkt",
                "st_p",
                (bsz_i, bucket_i, real_i, win, guard_owner, tuple(owners.tolist())),
            ),
            name="dsv4_bkt_state_pos",
        ),
        cache_owner_ids=_bkt_owner_dev(
            cache_owner,
            key=(
                "bkt",
                "c_o",
                (bsz_i, bucket_i, real_i, win, guard_owner, tuple(owners.tolist())),
            ),
            name="dsv4_bkt_cache_owner",
        ),
        ape=_device_or_upload(
            compressor.ape, name="dsv4_bkt_ape", dtype=ml_dtypes.bfloat16
        ),
        indexer_ape=_device_or_upload(
            indexer_compressor.ape, name="dsv4_bkt_idx_ape", dtype=ml_dtypes.bfloat16
        ),
        window_size=win,
        ring_size=ring,
        indexer_ring_size=indexer_ring,
        clen=int(clen),
        owner_id_stride=1,
        max_clen=max_clen,
        indexer_max_clen=indexer_max_clen,
        cache_real_clen=int(real_clen),
        guard_owner=guard_owner,
        artifacts_dir=build_dir,
    )


def run_compressor_prefill_dual_state_cache_swa_scatter_device(
    compressor: Any,
    *,
    swa_kv_cache: Any,
    swa_rows: Any,
    swa_start_pos: int,
    swa_bsz: int,
    swa_seqlen: int,
    kv: Any,
    score: Any,
    scatter_rows: Any,
    indexer_compressor: Any,
    indexer_kv: Any,
    indexer_score: Any,
    indexer_scatter_rows: Any,
    bsz: int,
    seqlen: int,
    clen: int,
    device_state: Any,
    indexer_device_state: Any,
    window_size: int,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
    swa_owner_ids: np.ndarray | None = None,
    swa_owner_ids_dev: Any | None = None,
    swa_positions_dev: Any | None = None,
    owner_id_stride: int | None = None,
    real_seqlen: int | None = None,
) -> None:
    """Fuse prefill SWA plus main/indexer compressor state/cache writes.

    ``real_seqlen`` (bucketed-prefill mode): when provided, ``seqlen`` is the
    COMPILE token bucket and ``real_seqlen <= seqlen`` is the true prompt length.
    Owner/position arrays are built at the bucket / MAXKEEP=2*ratio-1 dims with
    padding rows redirected to the guard owner, and the compressed write masks
    columns beyond ``real_seqlen//ratio`` (cache_real_clen). This validated
    recipe (tests/test_dsv4_writeswa_bucket_device.py) keeps one NEFF per bucket
    and writes only real rows. When None, the legacy per-length path runs."""

    if (
        not hasattr(swa_kv_cache, "tensor_ref")
        or not hasattr(device_state.kv_score_state, "tensor_ref")
        or not hasattr(device_state.compressed_kv_cache, "tensor_ref")
        or not hasattr(indexer_device_state.kv_score_state, "tensor_ref")
        or not hasattr(indexer_device_state.compressed_kv_cache, "tensor_ref")
    ):
        raise RuntimeError(
            "fused prefill SWA/dual-state/cache scatter requires device caches"
        )
    if (
        not _is_device_tensor_like(swa_rows)
        or not _is_device_tensor_like(kv)
        or not _is_device_tensor_like(score)
        or not _is_device_tensor_like(scatter_rows)
        or not _is_device_tensor_like(indexer_kv)
        or not _is_device_tensor_like(indexer_score)
        or not _is_device_tensor_like(indexer_scatter_rows)
    ):
        raise RuntimeError(
            "fused prefill SWA/dual-state/cache scatter requires device rows"
        )

    if real_seqlen is not None:
        request_owners = None
        if owner_ids is not None:
            request_owners = _request_owners_from_flat(owner_ids, bsz=int(bsz))
        _run_bucketed_prefill_dual_scatter(
            compressor,
            swa_kv_cache=swa_kv_cache,
            swa_rows=swa_rows,
            kv=kv,
            score=score,
            scatter_rows=scatter_rows,
            indexer_compressor=indexer_compressor,
            indexer_kv=indexer_kv,
            indexer_score=indexer_score,
            indexer_scatter_rows=indexer_scatter_rows,
            bsz=int(bsz),
            bucket_seqlen=int(seqlen),
            real_seqlen=int(real_seqlen),
            clen=int(clen),
            device_state=device_state,
            indexer_device_state=indexer_device_state,
            window_size=int(window_size),
            build_dir=build_dir,
            request_owners=request_owners,
        )
        return

    bsz_i = int(bsz)
    seqlen_i = int(seqlen)
    n = bsz_i * seqlen_i
    if owner_ids is None:
        owner_ids_arr = np.repeat(np.arange(bsz_i, dtype=np.int32), seqlen_i)
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (n,):
            raise ValueError(f"owner_ids must be [{n}], got {owner_ids_arr.shape}")
    positions = np.tile(np.arange(seqlen_i, dtype=np.int32), bsz_i)

    swa_bsz_i = int(swa_bsz)
    swa_seqlen_i = int(swa_seqlen)
    swa_n = swa_bsz_i * swa_seqlen_i
    swa_rows_shape = tuple(int(dim) for dim in getattr(swa_rows, "shape", ()))
    actual_swa_n = int(swa_rows_shape[0]) if len(swa_rows_shape) == 2 else int(swa_n)
    if swa_owner_ids is None:
        if swa_bsz_i != bsz_i or swa_seqlen_i > seqlen_i:
            raise ValueError(
                "swa_owner_ids is required when SWA mirror shape differs from "
                "the compressor token rectangle"
            )
        source_start = int(swa_start_pos)
        if source_start < 0 or source_start + swa_seqlen_i > seqlen_i:
            source_start = seqlen_i - swa_seqlen_i
        owners_rect = owner_ids_arr.reshape(bsz_i, seqlen_i)
        swa_owner_ids_arr = np.ascontiguousarray(
            owners_rect[:, source_start : source_start + swa_seqlen_i].reshape(-1)
        )
    else:
        swa_owner_ids_arr = np.asarray(swa_owner_ids, dtype=np.int32).reshape(-1)
        if swa_owner_ids_arr.shape != (swa_n,):
            raise ValueError(
                f"swa_owner_ids must be [{swa_n}], got {swa_owner_ids_arr.shape}"
            )
    swa_positions = np.tile(
        np.arange(
            int(swa_start_pos),
            int(swa_start_pos) + swa_seqlen_i,
            dtype=np.int32,
        ),
        swa_bsz_i,
    )
    if actual_swa_n != swa_n:
        if swa_bsz_i <= 0 or actual_swa_n % swa_bsz_i != 0:
            raise ValueError(
                "bucketed SWA rows must divide by SWA batch: "
                f"rows={actual_swa_n}, bsz={swa_bsz_i}"
            )
        request_owners = _request_owners_from_flat(owner_ids_arr, bsz=bsz_i)
        if request_owners is None:
            request_owners = _request_owners_from_flat(
                swa_owner_ids_arr,
                bsz=swa_bsz_i,
            )
        swa_owner_ids_arr, swa_positions = _bucketed_prefill_swa_owner_pos(
            bsz=swa_bsz_i,
            bucket_seqlen=actual_swa_n // swa_bsz_i,
            real_seqlen=min(
                int(swa_start_pos) + int(swa_seqlen_i),
                actual_swa_n // swa_bsz_i,
            ),
            window_size=int(window_size),
            guard_owner=int(device_state.spec.guard_owner),
            request_owners=request_owners,
        )

    keep_tokens = _prefill_state_tail_len(spec=device_state.spec, seqlen=seqlen_i)
    indexer_keep_tokens = _prefill_state_tail_len(
        spec=indexer_device_state.spec,
        seqlen=seqlen_i,
    )
    if keep_tokens == 0 or indexer_keep_tokens == 0:
        raise RuntimeError("fused prefill dual scatter requires non-empty state tail")
    if int(keep_tokens) != int(indexer_keep_tokens):
        raise RuntimeError(
            "fused prefill dual scatter requires matching state tails: "
            f"{int(keep_tokens)} vs {int(indexer_keep_tokens)}"
        )
    main_rows = tuple(int(dim) for dim in getattr(scatter_rows, "shape", ()))
    indexer_rows = tuple(int(dim) for dim in getattr(indexer_scatter_rows, "shape", ()))
    if len(main_rows) != 2 or len(indexer_rows) != 2 or main_rows[0] != indexer_rows[0]:
        raise ValueError(
            "main/indexer prefill compressed row counts must match, got "
            f"{main_rows}/{indexer_rows}"
        )

    state_owner_ids_dev = owner_ids_dev
    state_positions_dev = positions_dev
    row_indices: np.ndarray | None = None
    if keep_tokens < seqlen_i and bsz_i == 1:
        start = seqlen_i - int(keep_tokens)
        row_indices = np.arange(start, seqlen_i, dtype=np.int64)
        kv_alias = _alias_device_value_first_dim_slice(
            kv,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        score_alias = _alias_device_value_first_dim_slice(
            score,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        indexer_kv_alias = _alias_device_value_first_dim_slice(
            indexer_kv,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        indexer_score_alias = _alias_device_value_first_dim_slice(
            indexer_score,
            start=int(row_indices[0]),
            size=int(row_indices.shape[0]),
        )
        if (
            kv_alias is not None
            and score_alias is not None
            and indexer_kv_alias is not None
            and indexer_score_alias is not None
        ):
            kv = kv_alias
            score = score_alias
            indexer_kv = indexer_kv_alias
            indexer_score = indexer_score_alias
            if owner_ids_dev is not None:
                state_owner_ids_dev = _alias_device_value_first_dim_slice(
                    owner_ids_dev,
                    start=int(row_indices[0]),
                    size=int(row_indices.shape[0]),
                )
            if positions_dev is not None:
                state_positions_dev = _alias_device_value_first_dim_slice(
                    positions_dev,
                    start=int(row_indices[0]),
                    size=int(row_indices.shape[0]),
                )
        else:
            row_indices = None
    if row_indices is not None:
        state_owner_ids_arr = np.ascontiguousarray(owner_ids_arr[row_indices])
        state_positions_arr = np.ascontiguousarray(positions[row_indices])
    else:
        state_owner_ids_arr = owner_ids_arr
        state_positions_arr = positions

    stride_i = int(owner_id_stride) if owner_id_stride is not None else seqlen_i
    run_write_swa_dual_kv_score_state_owner_clen_device(
        swa_kv_cache=swa_kv_cache,
        kv_score_state=device_state.kv_score_state,
        compressed_kv_cache=device_state.compressed_kv_cache,
        indexer_kv_score_state=indexer_device_state.kv_score_state,
        indexer_compressed_kv_cache=indexer_device_state.compressed_kv_cache,
        swa_rows=swa_rows,
        kv_new=kv,
        score_new=score,
        compressed_rows=scatter_rows,
        indexer_kv_new=indexer_kv,
        indexer_score_new=indexer_score,
        indexer_compressed_rows=indexer_scatter_rows,
        swa_owner_ids=_device_vector_or_upload(
            swa_owner_ids_dev,
            swa_owner_ids_arr,
            name="dsv4_comp_prefill_swa_dual_owner_ids",
        ),
        swa_positions=_device_vector_or_upload(
            swa_positions_dev,
            swa_positions,
            name="dsv4_comp_prefill_swa_dual_positions",
        ),
        state_owner_ids=_device_vector_or_upload(
            state_owner_ids_dev,
            state_owner_ids_arr,
            name="dsv4_comp_prefill_swa_dual_state_owner_ids",
        ),
        state_positions=_device_vector_or_upload(
            state_positions_dev,
            state_positions_arr,
            name="dsv4_comp_prefill_swa_dual_state_positions",
        ),
        cache_owner_ids=_device_vector_or_upload(
            owner_ids_dev,
            owner_ids_arr,
            name="dsv4_comp_prefill_swa_dual_cache_owner_ids",
        ),
        ape=_device_or_upload(
            compressor.ape,
            name="dsv4_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        indexer_ape=_device_or_upload(
            indexer_compressor.ape,
            name="dsv4_indexer_comp_ape",
            dtype=ml_dtypes.bfloat16,
        ),
        window_size=int(window_size),
        ring_size=int(device_state.ring_size),
        indexer_ring_size=int(indexer_device_state.ring_size),
        clen=int(clen),
        owner_id_stride=stride_i,
        max_clen=int(device_state.spec.max_compressed_len),
        indexer_max_clen=int(indexer_device_state.spec.max_compressed_len),
        artifacts_dir=build_dir,
    )


def run_compressor_scatter_rows_device(
    compressor: Any,
    start_pos: int,
    *,
    scatter_rows: Any,
    bsz: int,
    clen: int,
    device_state: Any,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
    token_owner_ids_dev: Any | None = None,
    owner_id_stride: int | None = None,
) -> None:
    """Scatter post-qDQ compressed rows into the persistent compressed KV cache."""
    from nkipy_serving.attention.deepseek_v4.kernels import (
        run_write_kv_owner_clen_device,
        run_write_kv_owner_pos_device,
        run_write_kv_slots_device,
    )

    DeviceTensor = get_device_tensor_cls()

    max_clen = int(device_state.spec.max_compressed_len)
    has_device_cache = hasattr(device_state.compressed_kv_cache, "tensor_ref")
    req_owner_ids = None
    if owner_ids is not None:
        req_owner_ids = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if req_owner_ids.shape != (int(bsz),):
            raise ValueError(
                f"owner_ids must be [bsz={int(bsz)}], got {req_owner_ids.shape}"
            )

    if (
        has_device_cache
        and owner_ids_dev is not None
        and positions_dev is not None
        and int(start_pos) != 0
        and int(clen) == 1
    ):
        n_total = int(scatter_rows.shape[0])
        if n_total != int(bsz):
            raise ValueError(f"scatter rows {n_total} != bsz {int(bsz)}")
        owner_shape = tuple(int(dim) for dim in getattr(owner_ids_dev, "shape", ()))
        if owner_shape != (int(bsz),):
            raise ValueError(
                f"owner_ids_dev must be [bsz={int(bsz)}], got {owner_shape}"
            )
        pos_shape = tuple(int(dim) for dim in getattr(positions_dev, "shape", ()))
        if not pos_shape or int(pos_shape[0]) < int(bsz):
            raise ValueError(
                f"positions_dev must have first dim >= bsz={int(bsz)}, got {pos_shape}"
            )
        if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
            raise ValueError(f"positions_dev must be [N] or [N, 1], got {pos_shape}")
        run_write_kv_owner_pos_device(
            kv_cache=device_state.compressed_kv_cache,
            kv_new=scatter_rows,
            owner_ids=owner_ids_dev,
            positions=positions_dev,
            ratio=int(compressor.compress_ratio),
            max_clen=max_clen,
            artifacts_dir=build_dir,
        )
        return

    if (
        has_device_cache
        and token_owner_ids_dev is not None
        and int(start_pos) == 0
        and int(clen) > 0
    ):
        n_total = int(scatter_rows.shape[0])
        if n_total != int(bsz) * int(clen):
            raise ValueError(
                f"scatter rows {n_total} != bsz * clen {int(bsz) * int(clen)}"
            )
        stride_i = int(owner_id_stride) if owner_id_stride is not None else int(clen)
        owner_shape = tuple(
            int(dim) for dim in getattr(token_owner_ids_dev, "shape", ())
        )
        required = (int(bsz) - 1) * stride_i + 1
        if len(owner_shape) != 1 or int(owner_shape[0]) < required:
            raise ValueError(
                "token_owner_ids_dev must be 1D with enough source rows for "
                f"bsz={int(bsz)}, stride={stride_i}: got {owner_shape}"
            )
        run_write_kv_owner_clen_device(
            kv_cache=device_state.compressed_kv_cache,
            kv_new=scatter_rows,
            owner_ids=token_owner_ids_dev,
            clen=int(clen),
            owner_id_stride=stride_i,
            max_clen=max_clen,
            artifacts_dir=build_dir,
        )
        return

    if req_owner_ids is None:
        req_owner_ids = np.arange(int(bsz), dtype=np.int32)
    if int(start_pos) == 0:
        owners = np.repeat(req_owner_ids, int(clen))
        cpos = np.tile(np.arange(int(clen), dtype=np.int32), int(bsz))
    else:
        owners = req_owner_ids
        cpos = np.full(
            (int(bsz),),
            int(start_pos) // int(compressor.compress_ratio),
            dtype=np.int32,
        )
    slots = owners * np.int32(max_clen) + cpos

    if not has_device_cache:
        from nkipy_serving.attention.deepseek_v4.kernels import (
            write_kv_to_flat_cache_oracle,
        )

        if _is_device_tensor_like(scatter_rows):
            if not hasattr(scatter_rows, "numpy"):
                raise RuntimeError(
                    "host compressed-cache fallback cannot consume "
                    "non-downloadable device tensors"
                )
            scatter_rows_host = scatter_rows.numpy()
        else:
            scatter_rows_host = scatter_rows
        write_kv_to_flat_cache_oracle(
            kv_new=scatter_rows_host,
            kv_cache=device_state.compressed_kv_cache,
            slot_mapping=slots,
        )
    else:
        n_total = int(scatter_rows.shape[0])
        if n_total != int(slots.shape[0]):
            raise ValueError(
                f"scatter rows {n_total} != slot count {int(slots.shape[0])}"
            )
        run_write_kv_slots_device(
            kv_cache=device_state.compressed_kv_cache,
            kv_new=scatter_rows,
            slot_mapping=DeviceTensor.from_numpy(
                np.ascontiguousarray(slots.astype(np.int32)),
                name="dsv4_post_pool_rows_slots",
            ),
            artifacts_dir=build_dir,
        )


__all__ = [
    "decode_pool_from_device_state",
    "decode_pool_from_state_oracle",
    "mirror_compressor_input_to_device_state",
    "prefill_pool_from_device_slab",
    "prefill_pool_from_slab_oracle",
    "run_compressor_decode_dual_state_cache_swa_scatter_device",
    "run_compressor_decode_dual_state_swa_scatter_device",
    "run_compressor_decode_state_cache_swa_scatter_device",
    "run_compressor_decode_state_cache_scatter_device",
    "run_compressor_decode_state_swa_scatter_device",
    "run_compressor_prefill_dual_state_cache_swa_scatter_device",
    "run_compressor_prefill_state_cache_scatter_device",
    "run_compressor_prefill_state_cache_swa_scatter_device",
    "run_compressor_scatter_rows_device",
    "run_decode_pool_from_state_device",
    "run_prefill_pool_from_slab_device",
    "run_write_kv_score_state_device",
    "write_kv_score_state_oracle",
]
