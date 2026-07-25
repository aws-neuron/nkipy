"""Device mutation helpers for DeepSeek-V4 sliding-window KV state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from nkipy_serving.attention.deepseek_v4.kernels import (
    run_write_kv_owner_window_device,
    run_write_kv_slots_device,
)
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls, is_device_tensor

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
    key = (int(value), str(name))
    cached = _DEVICE_SCALAR_I32_CACHE.get(key)
    if cached is not None:
        return cached
    DeviceTensor = get_device_tensor_cls()
    arr = np.asarray([[int(value)]], dtype=np.int32)
    dev = DeviceTensor.from_numpy(np.ascontiguousarray(arr), name=name)
    _DEVICE_SCALAR_I32_CACHE[key] = dev
    return dev


def swa_kv_cache_slots(
    owner_ids: np.ndarray,
    positions: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    """Return flat SWA cache rows for ``owner * window + position % window``."""

    owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    window = int(window_size)
    if window <= 0:
        raise ValueError("window_size must be positive")
    if owners.shape != pos.shape:
        raise ValueError(
            f"owner_ids and positions must match, got {owners.shape}/{pos.shape}"
        )
    if np.any(owners < 0):
        raise ValueError("owner_ids must be non-negative")
    if np.any(pos < 0):
        raise ValueError("positions must be non-negative")
    return (owners * np.int64(window) + (pos % np.int64(window))).astype(np.int32)


def write_swa_kv_cache_oracle(
    swa_kv_cache: np.ndarray,
    kv_new: np.ndarray,
    owner_ids: np.ndarray,
    positions: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    """CPU reference for SWA rolling-cache scatter."""

    cache = np.asarray(swa_kv_cache)
    rows = np.asarray(kv_new)
    if cache.ndim != 2 or rows.ndim != 2:
        raise ValueError(f"cache/kv_new must be 2-D, got {cache.shape}/{rows.shape}")
    if cache.shape[1] != rows.shape[1]:
        raise ValueError(f"head_dim mismatch: {cache.shape[1]} vs {rows.shape[1]}")
    slots = swa_kv_cache_slots(owner_ids, positions, window_size=window_size)
    if slots.shape != (rows.shape[0],):
        raise ValueError(f"slots must be [{rows.shape[0]}], got {slots.shape}")
    if np.any(slots >= cache.shape[0]):
        raise ValueError(
            f"SWA slot outside flat cache: max={int(slots.max())}, rows={cache.shape[0]}"
        )
    cache[slots] = rows.astype(cache.dtype, copy=False)
    return cache


def run_write_swa_kv_cache_device(
    *,
    swa_kv_cache: Any,
    kv_new: np.ndarray,
    owner_ids: np.ndarray,
    positions: np.ndarray,
    window_size: int,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
    live_rows: int | Any | None = None,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Write fresh attention KV rows into the device SWA ring cache.

    The underlying NKI primitive is the generic aliased flat-KV scatter.  This
    wrapper owns the DSV4 ring address calculation and chunks large prefill
    writes into scatter-kernel supported batches.
    """

    device_rows = hasattr(kv_new, "shape") and not isinstance(kv_new, np.ndarray)
    rows = kv_new if device_rows else np.asarray(kv_new)
    cache_shape = tuple(int(dim) for dim in getattr(swa_kv_cache, "shape"))
    rows_shape = tuple(int(dim) for dim in getattr(rows, "shape", ()))
    rows_ndim = len(rows_shape)
    if len(cache_shape) != 2 or rows_ndim != 2:
        raise ValueError(
            f"swa_kv_cache/kv_new must be 2-D, got {cache_shape}/{rows_shape}"
        )
    if cache_shape[1] != rows_shape[1]:
        raise ValueError(f"head_dim mismatch: {cache_shape[1]} vs {rows_shape[1]}")
    n_rows = int(rows_shape[0])
    live_rows_i: int | None = None
    live_rows_dev = live_rows
    if live_rows is not None:
        if isinstance(live_rows, (int, np.integer)):
            live_rows_i = int(live_rows)
            live_rows_dev = _device_scalar_i32(
                live_rows_i,
                name="dsv4_swa_owner_window_live_rows",
            )
        elif isinstance(live_rows, np.ndarray):
            live_rows_i = int(np.asarray(live_rows, dtype=np.int32).reshape(-1)[0])
        if live_rows_i is not None and (live_rows_i < 0 or live_rows_i > n_rows):
            raise ValueError(f"live_rows={live_rows_i} outside row count {n_rows}")

    owners = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
    pos = np.asarray(positions, dtype=np.int32).reshape(-1)
    if live_rows is not None and owners.shape[0] < n_rows:
        owners = _pad_i32_vector(owners, rows=n_rows)
    if live_rows is not None and pos.shape[0] < n_rows:
        pos = _pad_i32_vector(pos, rows=n_rows, fill=0)
    if (
        device_rows
        and is_device_tensor(swa_kv_cache)
        and owner_ids_dev is not None
        and positions_dev is not None
    ):
        DeviceTensor = get_device_tensor_cls()
        owner_dev_shape = tuple(int(dim) for dim in getattr(owner_ids_dev, "shape"))
        pos_dev_shape = tuple(int(dim) for dim in getattr(positions_dev, "shape"))
        if owner_dev_shape != (n_rows,):
            if owners.shape != (n_rows,):
                raise ValueError(
                    f"owner_ids_dev must be [{n_rows}], got {owner_dev_shape}"
                )
            owner_ids_dev = DeviceTensor.from_numpy(
                np.ascontiguousarray(owners.astype(np.int32, copy=False)),
                name=f"dsv4_swa_owner_window_owners_{n_rows}",
            )
            owner_dev_shape = tuple(int(dim) for dim in getattr(owner_ids_dev, "shape"))
        if not pos_dev_shape or int(pos_dev_shape[0]) < n_rows:
            if pos.shape != (n_rows,):
                raise ValueError(
                    "positions_dev must have first dim >= "
                    f"{n_rows}, got {pos_dev_shape}"
                )
            positions_dev = DeviceTensor.from_numpy(
                np.ascontiguousarray(pos.astype(np.int32, copy=False)),
                name=f"dsv4_swa_owner_window_positions_{n_rows}",
            )
            pos_dev_shape = tuple(int(dim) for dim in getattr(positions_dev, "shape"))
        if len(pos_dev_shape) > 2 or (
            len(pos_dev_shape) == 2 and int(pos_dev_shape[1]) != 1
        ):
            raise ValueError(
                f"positions_dev must be [N] or [N, 1], got {pos_dev_shape}"
            )
        run_write_kv_owner_window_device(
            kv_cache=swa_kv_cache,
            kv_new=rows,
            owner_ids=owner_ids_dev,
            positions=positions_dev,
            live_rows=live_rows_dev,
            window_size=int(window_size),
            artifacts_dir=artifacts_dir,
            _device_kernel_cls=_device_kernel_cls,
            _kernel_cache=_kernel_cache,
        )
        return swa_kv_cache

    if owners.shape != (rows_shape[0],) or pos.shape != (rows_shape[0],):
        raise ValueError(
            f"owner_ids/positions must be [{rows_shape[0]}], got "
            f"{owners.shape}/{pos.shape}"
        )
    if live_rows_i is not None:
        rows = rows[:live_rows_i]
        owners = owners[:live_rows_i]
        pos = pos[:live_rows_i]

    slots = swa_kv_cache_slots(owners, pos, window_size=window_size)
    if slots.size and int(slots.max()) >= cache_shape[0]:
        raise ValueError(
            f"SWA slot outside flat cache: max={int(slots.max())}, rows={cache_shape[0]}"
        )
    if rows_shape[0] == 0:
        return swa_kv_cache
    if not is_device_tensor(swa_kv_cache, require_numpy=True):
        if device_rows:
            raise RuntimeError("device SWA rows require a device SWA cache")
        return write_swa_kv_cache_oracle(
            swa_kv_cache,
            rows,
            owners,
            pos,
            window_size=window_size,
        )

    DeviceTensor = get_device_tensor_cls()

    if device_rows:
        if slots.size and np.unique(slots).size != slots.size:
            raise RuntimeError(
                "device SWA write with duplicate ring slots requires a "
                "pre-deduped row tensor"
            )
        if rows_shape[0] > 128:
            raise RuntimeError(
                "device SWA row tensor writes currently require <=128 rows; "
                f"got {rows_shape[0]}"
            )
        run_write_kv_slots_device(
            kv_cache=swa_kv_cache,
            kv_new=rows,
            slot_mapping=DeviceTensor.from_numpy(
                np.ascontiguousarray(slots.astype(np.int32)),
                name="dsv4_swa_kv_slots",
            ),
            artifacts_dir=artifacts_dir,
            _device_kernel_cls=_device_kernel_cls,
            _kernel_cache=_kernel_cache,
        )
        return swa_kv_cache

    # Duplicate slots appear in long prefill.  A single scatter kernel has no
    # ordering guarantee for repeated destinations, so keep only the last row
    # for each slot before launching.
    if slots.size:
        _, keep_rev = np.unique(slots[::-1], return_index=True)
        keep = (slots.size - 1 - keep_rev).astype(np.int64)
        keep.sort()
        rows = rows[keep]
        slots = slots[keep]

    for offset in range(0, rows.shape[0], 128):
        end = min(offset + 128, rows.shape[0])
        run_write_kv_slots_device(
            kv_cache=swa_kv_cache,
            kv_new=DeviceTensor.from_numpy(
                np.ascontiguousarray(rows[offset:end].astype(ml_dtypes.bfloat16)),
                name="dsv4_swa_kv_rows",
            ),
            slot_mapping=DeviceTensor.from_numpy(
                np.ascontiguousarray(slots[offset:end].astype(np.int32)),
                name="dsv4_swa_kv_slots",
            ),
            artifacts_dir=artifacts_dir,
            _device_kernel_cls=_device_kernel_cls,
            _kernel_cache=_kernel_cache,
        )
    return swa_kv_cache


def mirror_swa_kv_to_device_cache(
    kv: Any,
    start_pos: int,
    *,
    window_size: int,
    device_layer_state: Any,
    build_dir: str | Path | None = None,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    positions_dev: Any | None = None,
    live_rows: int | Any | None = None,
    bsz: int | None = None,
    seqlen: int | None = None,
) -> None:
    """Scatter fresh attention KV rows into the device SWA ring cache.

    ``kv`` may be a numpy array ``[bsz, seqlen, d]`` or an already-flattened
    device tensor ``[bsz * seqlen, d]``. Device callers should flatten through
    a graph/math fragment before this state wrapper; this function only owns
    ring addressing and the aliased scatter launch.

    Moved from ``models/deepseek_v4/sampled_forward.py`` (Stage 3) so
    state-bridging logic lives with the kernel wrapper it invokes.
    """
    shape = tuple(int(dim) for dim in getattr(kv, "shape"))
    device_rows = hasattr(kv, "shape") and not isinstance(kv, np.ndarray)
    if len(shape) == 3:
        bsz_i, seqlen_i, d = shape
        if device_rows:
            raise RuntimeError(
                "device SWA mirror expects flattened [bsz*seqlen, d] rows; "
                "flatten in the graph/math function before state scatter"
            )
    elif len(shape) == 2 and bsz is not None and seqlen is not None:
        bsz_i = int(bsz)
        seqlen_i = int(seqlen)
        d = shape[1]
        logical_rows = int(bsz_i) * int(seqlen_i)
        if shape[0] < logical_rows:
            raise ValueError(
                f"flat kv rows {shape[0]} do not match bsz*seqlen={logical_rows}"
            )
        if shape[0] > logical_rows and live_rows is None:
            live_rows = logical_rows
    else:
        raise ValueError(
            "kv must be [bsz, seqlen, d] or flat [bsz*seqlen, d] with "
            f"bsz/seqlen, got {shape}"
        )
    n_rows = int(shape[0]) if len(shape) == 2 else int(bsz_i) * int(seqlen_i)
    logical_rows = int(bsz_i) * int(seqlen_i)
    if owner_ids is None:
        owner_ids_arr = np.repeat(
            np.arange(int(bsz_i), dtype=np.int32),
            int(seqlen_i),
        )
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape not in ((logical_rows,), (n_rows,)):
            raise ValueError(
                f"owner_ids must be [{logical_rows}]"
                + (f" or [{n_rows}]" if n_rows != logical_rows else "")
                + f", got {owner_ids_arr.shape}"
            )
    positions = np.tile(
        np.arange(int(start_pos), int(start_pos) + int(seqlen_i), dtype=np.int32),
        int(bsz_i),
    )
    if n_rows > logical_rows:
        owner_ids_arr = _pad_i32_vector(owner_ids_arr, rows=n_rows)
        positions = _pad_i32_vector(positions, rows=n_rows, fill=0)
    if len(shape) == 2:
        rows = kv
    else:
        rows = np.ascontiguousarray(
            np.asarray(kv).reshape(int(bsz_i) * int(seqlen_i), int(d))
        )
    run_write_swa_kv_cache_device(
        swa_kv_cache=device_layer_state.swa_kv_cache,
        kv_new=rows,
        owner_ids=owner_ids_arr,
        positions=positions,
        window_size=int(window_size),
        owner_ids_dev=owner_ids_dev,
        positions_dev=positions_dev,
        live_rows=live_rows,
        artifacts_dir=build_dir,
    )


__all__ = [
    "mirror_swa_kv_to_device_cache",
    "run_write_swa_kv_cache_device",
    "swa_kv_cache_slots",
    "write_swa_kv_cache_oracle",
]
