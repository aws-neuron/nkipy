"""Row-copy kernels for DSV4 owner-state checkpoint/restore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nkipy_serving.runtime.device_tensor import dtype_like as _dtype_like
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.device_tensor import sample_like as _sample_like
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

try:
    import neuronxcc.nki as _nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt

    _NKI_AVAILABLE = True
except ImportError:
    _nki = None
    nl = None
    nt = None
    _NKI_AVAILABLE = False


_GATHER_STATE_ROWS_KERNEL_CACHE: dict[tuple, Any] = {}


def gather_state_rows_oracle(src: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return ``src[rows]`` with validation for state checkpoint tests."""

    src_arr = np.asarray(src)
    row_ids = np.asarray(rows, dtype=np.int32).reshape(-1)
    if src_arr.ndim != 2:
        raise ValueError(f"src must be [num_rows, width], got {src_arr.shape}")
    if row_ids.size == 0:
        return np.empty((0, src_arr.shape[1]), dtype=src_arr.dtype)
    if np.any(row_ids < 0) or int(row_ids.max()) >= src_arr.shape[0]:
        raise ValueError("row ids outside source state buffer")
    return np.asarray(src_arr[row_ids.astype(np.int64)]).copy()


if _NKI_AVAILABLE:

    @_nki.jit
    def _gather_state_rows_kernel(
        state: "nt.tensor",
        row_mapping: "nt.tensor",
        out: "nt.tensor[nt.mutable]",
    ):
        """Gather ``state[row_mapping[i]]`` into dense ``out[i]``."""
        n_rows, width = out.shape
        max_rows = 128
        n_tiles = (n_rows + max_rows - 1) // max_rows
        last_tile = n_rows - (n_tiles - 1) * max_rows
        rows_2d = row_mapping.reshape((n_rows, 1))

        for tile_idx in nl.static_range(n_tiles):
            if tile_idx < n_tiles - 1:
                cur_rows = max_rows
            else:
                cur_rows = last_tile
            row_start = tile_idx * max_rows

            row_p = nl.arange(cur_rows)[:, None]
            width_f = nl.arange(width)[None, :]
            rows_sb = nl.load(rows_2d[row_start : row_start + cur_rows])
            vals = nl.load(state[rows_sb[row_p, 0], width_f])
            nl.store(
                dst=out[row_start + row_p, width_f],
                value=vals[row_p, width_f],
            )
        return out


def _gather_state_rows_entry(state, row_mapping, out):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _gather_state_rows_kernel(state, row_mapping, out)


def run_gather_state_rows_device(
    *,
    src: Any,
    rows: Any,
    out: Any | None = None,
    name: str = "dsv4_state_rows_checkpoint",
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Gather selected rows from a 2-D DeviceTensor into a dense DeviceTensor."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    src_shape = tuple(int(dim) for dim in getattr(src, "shape"))
    row_shape = tuple(int(dim) for dim in getattr(rows, "shape"))
    if len(src_shape) != 2:
        raise ValueError(f"src must be [num_rows, width], got {src_shape}")
    if len(row_shape) != 1:
        raise ValueError(f"rows must be [n_rows], got {row_shape}")
    n_rows = int(row_shape[0])
    if n_rows == 0:
        if out is not None:
            return out

        return get_device_tensor_cls().from_numpy(
            np.empty((0, src_shape[1]), dtype=_dtype_like(src)),
            name=name,
        )
    if out is None:
        out = get_device_tensor_cls().from_numpy(
            np.empty((n_rows, src_shape[1]), dtype=_dtype_like(src)),
            name=name,
        )
    out_shape = tuple(int(dim) for dim in getattr(out, "shape"))
    if out_shape != (n_rows, src_shape[1]):
        raise ValueError(f"out must be [{n_rows}, {src_shape[1]}], got {out_shape}")

    cache = _GATHER_STATE_ROWS_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    cache_key = (
        "gather_state_rows",
        src_shape,
        row_shape,
        out_shape,
        str(_dtype_like(src)),
        str(_dtype_like(rows)),
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = compile_and_load_with_lock(
            _device_kernel_cls,
            _gather_state_rows_entry,
            _sample_like(src),
            _sample_like(rows),
            _sample_like(out),
            name=f"dsv4_gather_state_rows_n{n_rows}_w{src_shape[1]}",
            build_dir=artifacts_dir,
            namespace="dsv4_state_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "state": src,
            "row_mapping": rows,
            "out.must_alias_input": out,
        },
        outputs={"out": out},
    )
    return out


__all__ = [
    "gather_state_rows_oracle",
    "run_gather_state_rows_device",
]
