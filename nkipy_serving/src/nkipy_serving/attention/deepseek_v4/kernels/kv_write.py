"""KV-cache write kernels for DSV4 attention state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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

_WRITE_KV_SLOTS_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_KV_OWNER_POS_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_KV_OWNER_CLEN_KERNEL_CACHE: dict[tuple, Any] = {}
_WRITE_KV_OWNER_WINDOW_KERNEL_CACHE: dict[tuple, Any] = {}
_DEVICE_SCALAR_I32_CACHE: dict[tuple[int, str], Any] = {}


def _compile_and_load_with_lock(*args: Any, **kwargs: Any) -> Any:
    from nkipy_serving.attention.deepseek_v4 import kernels as dsv4_kernels

    return dsv4_kernels.compile_and_load_with_lock(*args, **kwargs)


def _device_scalar_i32(value: int, *, name: str) -> Any:
    key = (int(value), str(name))
    cached = _DEVICE_SCALAR_I32_CACHE.get(key)
    if cached is not None:
        return cached
    ensure_nki_bridge()
    from nkipy.runtime import DeviceTensor

    arr = np.asarray([[int(value)]], dtype=np.int32)
    dev = DeviceTensor.from_numpy(np.ascontiguousarray(arr), name=name)
    _DEVICE_SCALAR_I32_CACHE[key] = dev
    return dev


# ---------------------------------------------------------------------------
# Device KV scatter by slot_mapping
# ---------------------------------------------------------------------------


def write_kv_to_flat_cache_oracle(
    *,
    kv_new: np.ndarray,  # [total_tokens, head_dim]
    kv_cache: np.ndarray,  # [num_slots, head_dim]
    slot_mapping: np.ndarray,  # [total_tokens] int
) -> np.ndarray:
    """Reference scatter: ``kv_cache[slot_mapping] = kv_new`` (in place).

    Returns the updated cache for convenience.
    """
    if kv_new.ndim != 2 or kv_cache.ndim != 2:
        raise ValueError(
            f"kv_new and kv_cache must be 2D, got {kv_new.shape} / {kv_cache.shape}"
        )
    if kv_new.shape[1] != kv_cache.shape[1]:
        raise ValueError(
            f"head_dim mismatch: kv_new={kv_new.shape[1]}, kv_cache={kv_cache.shape[1]}"
        )
    slots = np.asarray(slot_mapping, dtype=np.int64).reshape(-1)
    if slots.shape != (kv_new.shape[0],):
        raise ValueError(f"slot_mapping must be [{kv_new.shape[0]}], got {slots.shape}")
    if np.any(slots < 0) or np.any(slots >= kv_cache.shape[0]):
        raise ValueError("slot_mapping values outside flat cache range")
    kv_cache[slots] = kv_new.astype(kv_cache.dtype, copy=False)
    return kv_cache


if _NKI_AVAILABLE:

    @_nki.jit
    def _write_kv_slots_kernel(
        kv_cache: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        slot_mapping: "nt.tensor",
    ):
        """In-place scatter ``kv_cache[slot_mapping[i]] = kv_new[i]``."""
        n_new, d = kv_new.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        slot_2d = slot_mapping.reshape((n_new, 1))

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d)[None, :]

            slot_sb = nl.load(slot_2d[t0 : t0 + cur])
            rows_sb = nl.load(kv_new[t0 : t0 + cur])

            nl.store(
                dst=kv_cache[slot_sb[i_p, 0], i_f],
                value=rows_sb[i_p, i_f],
            )
        return kv_cache

    @_nki.jit
    def _write_kv_owner_pos_kernel(
        kv_cache: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        ratio: int,
        max_clen: int,
    ):
        """Scatter ``kv_new`` using device owner ids and token positions."""
        n_new, d = kv_new.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((n_new, 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        inv_ratio = np.float32(1.0) / np.float32(ratio)

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d)[None, :]

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])
            # Avoid NKI integer ``//`` near boundary values: subtract-mod makes
            # the dividend exactly divisible before reciprocal multiply/cast.
            pos_base = nl.subtract(pos_sb, nl.mod(pos_sb, nl.int32(ratio)))
            cpos_f = nl.ndarray(pos_base.shape, dtype=nl.float32, buffer=nl.sbuf)
            cpos_f[...] = nisa.tensor_scalar(
                data=pos_base,
                op0=nl.multiply,
                operand0=inv_ratio,
                dtype=nl.float32,
            )
            cpos = nl.ndarray(pos_base.shape, dtype=np.int32, buffer=nl.sbuf)
            cpos[...] = nl.copy(cpos_f, dtype=nl.int32)
            rows = nl.add(nl.multiply(owner_sb, nl.int32(max_clen)), cpos)
            rows_sb = nl.load(kv_new[t0 : t0 + cur])

            nl.store(
                dst=kv_cache[rows[i_p, 0], i_f],
                value=rows_sb[i_p, i_f],
            )
        return kv_cache

    @_nki.jit
    def _write_kv_owner_clen_kernel(
        kv_cache: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        owner_ids: "nt.tensor",
        clen: int,
        owner_id_stride: int,
        max_clen: int,
    ):
        """Scatter prefill compressed rows using device owner ids.

        ``kv_new`` is request-major ``[bsz * clen, d]``. ``owner_ids`` is the
        token-owner vector for the source token rows; the request owner lives at
        ``request * owner_id_stride``.
        """
        n_new, d = kv_new.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((owner_ids.shape[0], 1))
        inv_clen = np.float32(1.0) / np.float32(clen)

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d)[None, :]

            row_ids = nl.add(i_p, nl.int32(t0))
            cpos = nl.mod(row_ids, nl.int32(clen))
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
            owner_offsets = nl.multiply(req_ids, nl.int32(owner_id_stride))
            owner_sb = nl.load(owner_2d[owner_offsets[i_p, 0], 0:1])
            rows = nl.add(nl.multiply(owner_sb, nl.int32(max_clen)), cpos)
            rows_sb = nl.load(kv_new[t0 : t0 + cur])

            nl.store(
                dst=kv_cache[rows[i_p, 0], i_f],
                value=rows_sb[i_p, i_f],
            )
        return kv_cache

    @_nki.jit
    def _write_kv_owner_window_kernel(
        kv_cache: "nt.tensor[nt.mutable]",
        kv_new: "nt.tensor",
        owner_ids: "nt.tensor",
        positions: "nt.tensor",
        live_rows: "nt.tensor",
        window_size: int,
        guard_owner: int,
    ):
        """Scatter SWA rows using device owner ids and token positions."""
        n_new, d = kv_new.shape
        MAX_T = 128
        n_tiles = (n_new + MAX_T - 1) // MAX_T
        last_tile = n_new - (n_tiles - 1) * MAX_T

        owner_2d = owner_ids.reshape((owner_ids.shape[0], 1))
        positions_2d = positions.reshape((positions.shape[0], 1))
        live_rows_sb = nl.load(live_rows[0:1, 0:1])
        live_rows_i = live_rows_sb[0, 0]

        for ti in nl.static_range(n_tiles):
            if ti < n_tiles - 1:
                cur = MAX_T
            else:
                cur = last_tile
            t0 = ti * MAX_T

            i_p = nl.arange(cur)[:, None]
            i_f = nl.arange(d)[None, :]
            row_ids = nl.add(i_p, nl.int32(t0))

            owner_sb = nl.load(owner_2d[t0 : t0 + cur])
            pos_sb = nl.load(positions_2d[t0 : t0 + cur])
            row_live = nl.less(row_ids, live_rows_i)
            guard_owner_sb = nl.full(owner_sb.shape, guard_owner, dtype=nl.int32)
            safe_owner = nl.where(row_live, owner_sb, guard_owner_sb)
            pos_in_window = nl.mod(pos_sb, nl.int32(window_size))
            rows = nl.add(
                nl.multiply(safe_owner, nl.int32(window_size)),
                pos_in_window,
            )
            rows_sb = nl.load(kv_new[t0 : t0 + cur])

            nl.store(
                dst=kv_cache[rows[i_p, 0], i_f],
                value=rows_sb[i_p, i_f],
            )
        return kv_cache


def _write_kv_slots_entry(kv_cache, kv_new, slot_mapping):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_slots_kernel(kv_cache, kv_new, slot_mapping)


def _write_kv_owner_pos_entry(
    kv_cache,
    kv_new,
    owner_ids,
    positions,
    *,
    ratio: int,
    max_clen: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_owner_pos_kernel(
        kv_cache,
        kv_new,
        owner_ids,
        positions,
        int(ratio),
        int(max_clen),
    )


def _write_kv_owner_clen_entry(
    kv_cache,
    kv_new,
    owner_ids,
    *,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_owner_clen_kernel(
        kv_cache,
        kv_new,
        owner_ids,
        int(clen),
        int(owner_id_stride),
        int(max_clen),
    )


def _write_kv_owner_window_entry(
    kv_cache,
    kv_new,
    owner_ids,
    positions,
    live_rows,
    *,
    window_size: int,
    guard_owner: int,
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _write_kv_owner_window_kernel(
        kv_cache,
        kv_new,
        owner_ids,
        positions,
        live_rows,
        int(window_size),
        int(guard_owner),
    )


def run_write_kv_slots_device(
    *,
    kv_cache: Any,
    kv_new: Any,
    slot_mapping: Any,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Device scatter of ``kv_new`` rows into ``kv_cache`` at ``slot_mapping``."""
    cache_shape = tuple(int(dim) for dim in getattr(kv_cache, "shape"))
    new_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    slot_shape = tuple(int(dim) for dim in getattr(slot_mapping, "shape"))

    if len(cache_shape) != 2:
        raise ValueError(f"kv_cache must be [num_slots, d], got {cache_shape}")
    if len(new_shape) != 2:
        raise ValueError(f"kv_new must be [n_new, d], got {new_shape}")
    if cache_shape[1] != new_shape[1]:
        raise ValueError(
            f"head_dim mismatch: kv_cache={cache_shape[1]}, kv_new={new_shape[1]}"
        )
    n_new = new_shape[0]
    if slot_shape != (n_new,):
        raise ValueError(f"slot_mapping must be [{n_new}], got {slot_shape}")
    if n_new == 0:
        return kv_cache

    cache = _WRITE_KV_SLOTS_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    cache_key = (
        "write_kv_slots",
        cache_shape,
        new_shape,
        slot_shape,
        str(_dtype_like(kv_cache)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(slot_mapping)),
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            _write_kv_slots_entry,
            _sample_like(kv_cache),
            _sample_like(kv_new),
            _sample_like(slot_mapping),
            name=f"dsv4_write_kv_slots_nnew{n_new}_d{cache_shape[1]}",
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "kv_cache.must_alias_input": kv_cache,
            "kv_new": kv_new,
            "slot_mapping": slot_mapping,
        },
        outputs={"kv_cache": kv_cache},
    )
    return kv_cache


def run_write_kv_owner_clen_device(
    *,
    kv_cache: Any,
    kv_new: Any,
    owner_ids: Any,
    clen: int,
    owner_id_stride: int,
    max_clen: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Device prefill scatter into ``kv_cache[owner * max_clen + cpos]``."""
    cache_shape = tuple(int(dim) for dim in getattr(kv_cache, "shape"))
    new_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))

    if len(cache_shape) != 2:
        raise ValueError(f"kv_cache must be [num_slots, d], got {cache_shape}")
    if len(new_shape) != 2:
        raise ValueError(f"kv_new must be [n_new, d], got {new_shape}")
    if len(owner_shape) != 1:
        raise ValueError(f"owner_ids must be 1D, got {owner_shape}")
    if cache_shape[1] != new_shape[1]:
        raise ValueError(
            f"head_dim mismatch: kv_cache={cache_shape[1]}, kv_new={new_shape[1]}"
        )
    clen_i = int(clen)
    stride_i = int(owner_id_stride)
    max_clen_i = int(max_clen)
    if clen_i <= 0 or stride_i <= 0 or max_clen_i <= 0:
        raise ValueError(
            "clen, owner_id_stride, and max_clen must be positive, got "
            f"{clen_i}/{stride_i}/{max_clen_i}"
        )
    n_new = int(new_shape[0])
    if n_new == 0:
        return kv_cache
    if n_new % clen_i:
        raise ValueError(f"kv_new rows {n_new} must be divisible by clen {clen_i}")
    bsz = n_new // clen_i
    required_owners = (bsz - 1) * stride_i + 1
    if int(owner_shape[0]) < required_owners:
        raise ValueError(
            "owner_ids first dim too small for request-major prefill scatter: "
            f"got {owner_shape[0]}, need >= {required_owners}"
        )

    cache = (
        _WRITE_KV_OWNER_CLEN_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    cache_key = (
        "write_kv_owner_clen",
        cache_shape,
        new_shape,
        owner_shape,
        str(_dtype_like(kv_cache)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(owner_ids)),
        clen_i,
        stride_i,
        max_clen_i,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            _write_kv_owner_clen_entry,
            _sample_like(kv_cache),
            _sample_like(kv_new),
            _sample_like(owner_ids),
            clen=clen_i,
            owner_id_stride=stride_i,
            max_clen=max_clen_i,
            name=(
                "dsv4_write_kv_owner_clen_"
                f"nnew{n_new}_d{cache_shape[1]}_clen{clen_i}_"
                f"stride{stride_i}_c{max_clen_i}"
            ),
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "kv_cache.must_alias_input": kv_cache,
            "kv_new": kv_new,
            "owner_ids": owner_ids,
        },
        outputs={"kv_cache": kv_cache},
    )
    return kv_cache


def run_write_kv_owner_pos_device(
    *,
    kv_cache: Any,
    kv_new: Any,
    owner_ids: Any,
    positions: Any,
    ratio: int,
    max_clen: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Device scatter into ``kv_cache[owner_ids * max_clen + positions // ratio]``."""
    cache_shape = tuple(int(dim) for dim in getattr(kv_cache, "shape"))
    new_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))

    if len(cache_shape) != 2:
        raise ValueError(f"kv_cache must be [num_slots, d], got {cache_shape}")
    if len(new_shape) != 2:
        raise ValueError(f"kv_new must be [n_new, d], got {new_shape}")
    if cache_shape[1] != new_shape[1]:
        raise ValueError(
            f"head_dim mismatch: kv_cache={cache_shape[1]}, kv_new={new_shape[1]}"
        )
    n_new = new_shape[0]
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    ratio_i = int(ratio)
    max_clen_i = int(max_clen)
    if ratio_i <= 0 or max_clen_i <= 0:
        raise ValueError(
            f"ratio and max_clen must be positive, got {ratio_i}/{max_clen_i}"
        )
    if n_new == 0:
        return kv_cache

    cache = _WRITE_KV_OWNER_POS_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    cache_key = (
        "write_kv_owner_pos",
        cache_shape,
        new_shape,
        owner_shape,
        pos_shape,
        str(_dtype_like(kv_cache)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
        ratio_i,
        max_clen_i,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            _write_kv_owner_pos_entry,
            _sample_like(kv_cache),
            _sample_like(kv_new),
            _sample_like(owner_ids),
            _sample_like(positions),
            ratio=ratio_i,
            max_clen=max_clen_i,
            name=(
                "dsv4_write_kv_owner_pos_"
                f"nnew{n_new}_d{cache_shape[1]}_p{pos_shape[0]}_"
                f"r{ratio_i}_c{max_clen_i}"
            ),
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "kv_cache.must_alias_input": kv_cache,
            "kv_new": kv_new,
            "owner_ids": owner_ids,
            "positions": positions,
        },
        outputs={"kv_cache": kv_cache},
    )
    return kv_cache


def run_write_kv_owner_window_device(
    *,
    kv_cache: Any,
    kv_new: Any,
    owner_ids: Any,
    positions: Any,
    live_rows: Any | None = None,
    window_size: int,
    guard_owner: int | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Device scatter into ``kv_cache[owner_ids * window + positions % window]``.

    Bucket padding rows are redirected to a guard-owner window. The SWA cache
    can also include a final single-row padding sink, so its row count does not
    need to be an exact multiple of ``window_size``.
    """
    cache_shape = tuple(int(dim) for dim in getattr(kv_cache, "shape"))
    new_shape = tuple(int(dim) for dim in getattr(kv_new, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    pos_shape = tuple(int(dim) for dim in getattr(positions, "shape"))
    if live_rows is None:
        live_rows = _device_scalar_i32(
            int(new_shape[0]),
            name="dsv4_write_kv_owner_window_live_rows",
        )
    live_shape = tuple(int(dim) for dim in getattr(live_rows, "shape"))

    if len(cache_shape) != 2:
        raise ValueError(f"kv_cache must be [num_slots, d], got {cache_shape}")
    if len(new_shape) != 2:
        raise ValueError(f"kv_new must be [n_new, d], got {new_shape}")
    if cache_shape[1] != new_shape[1]:
        raise ValueError(
            f"head_dim mismatch: kv_cache={cache_shape[1]}, kv_new={new_shape[1]}"
        )
    n_new = new_shape[0]
    if owner_shape != (n_new,):
        raise ValueError(f"owner_ids must be [{n_new}], got {owner_shape}")
    if not pos_shape or int(pos_shape[0]) < int(n_new):
        raise ValueError(
            f"positions must have first dim >= {int(n_new)}, got {pos_shape}"
        )
    if len(pos_shape) > 2 or (len(pos_shape) == 2 and int(pos_shape[1]) != 1):
        raise ValueError(f"positions must be [N] or [N, 1], got {pos_shape}")
    if live_shape != (1, 1):
        raise ValueError(f"live_rows must be [1, 1], got {live_shape}")
    window_i = int(window_size)
    if window_i <= 0:
        raise ValueError(f"window_size must be positive, got {window_i}")
    if guard_owner is None:
        guard_owner_i = int(cache_shape[0]) // window_i - 1
    else:
        guard_owner_i = int(guard_owner)
    if guard_owner_i < 0 or (guard_owner_i + 1) * window_i > int(cache_shape[0]):
        raise ValueError(
            f"guard_owner={guard_owner_i} outside SWA cache "
            f"(rows={cache_shape[0]}, window={window_i})"
        )
    if n_new == 0:
        return kv_cache

    cache = (
        _WRITE_KV_OWNER_WINDOW_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    )
    cache_key = (
        "write_kv_owner_window",
        cache_shape,
        new_shape,
        owner_shape,
        pos_shape,
        str(_dtype_like(kv_cache)),
        str(_dtype_like(kv_new)),
        str(_dtype_like(owner_ids)),
        str(_dtype_like(positions)),
        live_shape,
        str(_dtype_like(live_rows)),
        window_i,
        guard_owner_i,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = _compile_and_load_with_lock(
            _device_kernel_cls,
            _write_kv_owner_window_entry,
            _sample_like(kv_cache),
            _sample_like(kv_new),
            _sample_like(owner_ids),
            _sample_like(positions),
            _sample_like(live_rows),
            window_size=window_i,
            guard_owner=guard_owner_i,
            name=(
                "dsv4_write_kv_owner_window_"
                f"nbucket{n_new}_d{cache_shape[1]}_p{pos_shape[0]}_"
                f"w{window_i}_g{guard_owner_i}"
            ),
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "kv_cache.must_alias_input": kv_cache,
            "kv_new": kv_new,
            "owner_ids": owner_ids,
            "positions": positions,
            "live_rows": live_rows,
        },
        outputs={"kv_cache": kv_cache},
    )
    return kv_cache
