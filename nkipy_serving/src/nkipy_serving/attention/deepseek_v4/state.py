"""Device-owned DeepSeek-V4 attention state.

This module is the state ABI for moving DSV4 cache mutation out of model-local
numpy objects.  It intentionally keeps allocation separate from scheduling:
the CPU scheduler still owns request/page assignment, while these dataclasses
own the device buffers and the address math used by NKI kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import ml_dtypes
import numpy as np

from nkipy_serving.runtime.device_tensor import get_device_tensor_cls

DeviceAllocator = Callable[..., Any]
StateWriter = Callable[[Any, np.ndarray], None]


@dataclass(frozen=True)
class Dsv4KVFormat:
    """Logical format of one KV row.

    The first implementation keeps semantic rows (one row per token, one
    column per head-dim element).  Packed V4 rows can be added behind this
    format boundary without changing the model/backend call sites.
    """

    name: str
    dtype: Any
    head_dim: int
    packed_bytes_per_token: int | None = None

    @property
    def semantic(self) -> bool:
        return self.packed_bytes_per_token is None

    @property
    def row_shape(self) -> tuple[int, ...]:
        if self.semantic:
            return (int(self.head_dim),)
        return (int(self.packed_bytes_per_token),)


SEMANTIC_BF16_KV = Dsv4KVFormat(
    name="semantic_bf16",
    dtype=ml_dtypes.bfloat16,
    head_dim=0,
)


@dataclass(frozen=True)
class Dsv4CompressorStateSpec:
    """Shape/address contract for one layer's compressor rolling state.

    ``kv_score_state`` is flat by owner and ring offset:

        row = owner_id * ring_size + (position % ring_size)

    For c4a overlap layers every input row carries ``2 * head_dim`` KV and
    score lanes.  Compression reads the previous group from the first half and
    the current group from the second half.  This avoids the CPU implementation's
    physical shift after every compressed decode output.
    """

    layer_id: int
    compress_ratio: int
    head_dim: int
    max_seq_len: int
    num_state_owners: int
    num_compressed_slots: int
    overlap: bool | None = None
    state_dtype: Any = np.float32
    cache_dtype: Any = ml_dtypes.bfloat16
    # When True, ``num_state_owners`` / ``num_compressed_slots`` already include
    # one extra owner block reserved as the bucketed-prefill padding sink: the
    # fused NKI write redirects masked/padding rows to owner ``num_real_owners``
    # (the last block), which is never read back. Real owners stay in
    # ``[0, num_real_owners)``. See dsv4_nki_writeswa_plan.
    has_guard_owner: bool = False

    def __post_init__(self) -> None:
        if self.compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_state_owners <= 0:
            raise ValueError("num_state_owners must be positive")
        if self.num_compressed_slots <= 0:
            raise ValueError("num_compressed_slots must be positive")
        if self.overlap is None:
            object.__setattr__(self, "overlap", self.compress_ratio == 4)

    @property
    def num_real_owners(self) -> int:
        """Owner ids that hold live state: ``[0, num_real_owners)``.

        Equals ``num_state_owners`` unless a guard owner is reserved, in which
        case the last owner block (index ``num_real_owners``) is the
        bucketed-prefill padding sink and is never read.
        """
        if bool(self.has_guard_owner):
            return int(self.num_state_owners) - 1
        return int(self.num_state_owners)

    @property
    def guard_owner(self) -> int:
        """Owner index of the padding sink (== ``num_real_owners``)."""
        return int(self.num_real_owners)

    @property
    def coff(self) -> int:
        return 2 if bool(self.overlap) else 1

    @property
    def ring_size(self) -> int:
        return self.coff * int(self.compress_ratio)

    @property
    def state_width(self) -> int:
        return self.coff * int(self.head_dim)

    @property
    def packed_width(self) -> int:
        return 2 * self.state_width

    @property
    def max_compressed_len(self) -> int:
        return int(self.max_seq_len) // int(self.compress_ratio)

    @property
    def state_shape(self) -> tuple[int, int]:
        return (
            int(self.num_state_owners) * self.ring_size,
            self.packed_width,
        )

    @property
    def compressed_cache_shape(self) -> tuple[int, int]:
        return (int(self.num_compressed_slots), int(self.head_dim))

    def state_row(self, owner_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Return flat ring rows for ``owner_ids`` at token ``positions``."""
        owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
        pos = np.asarray(positions, dtype=np.int64).reshape(-1)
        if owners.shape != pos.shape:
            raise ValueError(
                f"owner_ids and positions must match, got {owners.shape}/{pos.shape}"
            )
        if np.any(owners < 0) or np.any(owners >= self.num_state_owners):
            raise ValueError("owner_ids outside compressor state owner range")
        return (
            owners * np.int64(self.ring_size) + (pos % np.int64(self.ring_size))
        ).astype(np.int32)


@dataclass
class Dsv4DeviceCompressorState:
    """Device buffers for one compressor-bearing layer."""

    spec: Dsv4CompressorStateSpec
    kv_score_state: Any
    compressed_kv_cache: Any

    @property
    def ring_size(self) -> int:
        return self.spec.ring_size

    @property
    def state_width(self) -> int:
        return self.spec.state_width

    def state_rows(
        self,
        owner_ids: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        return self.spec.state_row(owner_ids, positions)


@dataclass
class Dsv4DeviceLayerState:
    """All device-owned mutable state for one attention layer."""

    layer_id: int
    swa_kv_cache: Any
    compressor: Dsv4DeviceCompressorState | None = None
    indexer: Dsv4DeviceCompressorState | None = None

    @property
    def has_compressor(self) -> bool:
        return self.compressor is not None

    @property
    def has_indexer(self) -> bool:
        return self.indexer is not None


@dataclass
class Dsv4DeviceState:
    """Top-level DSV4 device state owned by one worker/lane."""

    layers: tuple[Dsv4DeviceLayerState, ...]
    num_slots_per_layer: int
    head_dim: int
    window_size: int
    max_seq_len: int
    max_batch_size: int

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Dsv4DeviceState requires at least one layer")
        for expected, layer in enumerate(self.layers):
            if int(layer.layer_id) != expected:
                raise ValueError(
                    "layers must be ordered by layer_id; "
                    f"expected {expected}, got {layer.layer_id}"
                )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def swa_kv_caches(self) -> list[Any]:
        return [layer.swa_kv_cache for layer in self.layers]

    def layer(self, layer_id: int) -> Dsv4DeviceLayerState:
        return self.layers[int(layer_id)]

    def compressor(self, layer_id: int) -> Dsv4DeviceCompressorState | None:
        return self.layer(layer_id).compressor

    def indexer(self, layer_id: int) -> Dsv4DeviceCompressorState | None:
        return self.layer(layer_id).indexer


@dataclass(frozen=True)
class Dsv4DeviceCompressorCheckpoint:
    """Snapshot of one owner-local compressor/indexer state."""

    kv_score_rows: np.ndarray
    kv_score_state: Any
    compressed_kv_rows: np.ndarray
    compressed_kv_cache: Any


@dataclass(frozen=True)
class Dsv4DeviceLayerCheckpoint:
    """Snapshot of one owner-local layer state."""

    layer_id: int
    swa_rows: np.ndarray
    swa_kv_cache: Any
    compressor: Dsv4DeviceCompressorCheckpoint | None = None
    indexer: Dsv4DeviceCompressorCheckpoint | None = None


@dataclass(frozen=True)
class Dsv4DeviceStateCheckpoint:
    """Checkpoint of all mutable DSV4 rows owned by one request-pool slot.

    Production DeviceTensor rollback must use bounded row sets.  Full-owner
    snapshots are kept for host-visible tests only because a c4 compressed KV
    owner can be huge at full V4 context.
    """

    owner_id: int
    seq_len: int | None
    layers: tuple[Dsv4DeviceLayerCheckpoint, ...]


def _alloc_zero(
    allocator: DeviceAllocator,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    name: str,
) -> Any:
    """Allocate a zero-initialized device buffer through ``allocator``.

    Existing backend allocators accept ``(shape, dtype, name=...)`` and return
    either a DeviceTensor or a test double.  If the returned object is a numpy
    array, keep it zero-filled for host tests.
    """

    obj = allocator(shape, dtype, name=name)
    if isinstance(obj, np.ndarray):
        obj[...] = np.asarray(0, dtype=obj.dtype)
    return obj


def _seed_kv_score_state(obj: Any, spec: Dsv4CompressorStateSpec) -> Any:
    """Seed packed state to CPU Compressor semantics when host-visible.

    Device allocators in this repo currently expose only shape/dtype creation,
    so true DeviceTensor seeding is handled by the first write/pool kernels.
    Host test doubles are initialized exactly: KV lanes zero, score lanes -inf.
    """

    if isinstance(obj, np.ndarray):
        obj[:, : spec.state_width] = 0
        obj[:, spec.state_width :] = -np.inf
    return obj


def _default_state_writer(dst: Any, src: np.ndarray) -> None:
    """Write host reset values into numpy test doubles or DeviceTensors."""
    if isinstance(dst, np.ndarray):
        dst[...] = np.asarray(src, dtype=dst.dtype)
        return
    from nkipy_serving.models.reload_utils import overwrite_device_tensor

    overwrite_device_tensor(dst, np.asarray(src))


def _zero_like_state_buffer(
    obj: Any, fallback_shape: tuple[int, ...], fallback_dtype: Any
) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(obj, "shape", fallback_shape))
    dtype = getattr(obj, "dtype", fallback_dtype)
    return np.zeros(shape, dtype=dtype)


def _owner_ids_array(owner_ids: Iterable[int], *, max_owners: int) -> np.ndarray:
    owners = np.asarray(tuple(int(v) for v in owner_ids), dtype=np.int32).reshape(-1)
    if owners.size == 0:
        return owners
    if np.any(owners < 0) or np.any(owners >= int(max_owners)):
        raise ValueError(
            "owner_ids outside DSV4 device-state owner range: "
            f"owners={owners.tolist()}, max_owners={int(max_owners)}"
        )
    return np.unique(owners).astype(np.int32, copy=False)


def _owner_swa_rows(state: Dsv4DeviceState, owners: np.ndarray) -> np.ndarray:
    return (
        (
            owners[:, None].astype(np.int64) * np.int64(state.window_size)
            + np.arange(int(state.window_size), dtype=np.int64)[None, :]
        )
        .reshape(-1)
        .astype(np.int32)
    )


def _owner_ring_rows(
    comp: Dsv4DeviceCompressorState,
    owners: np.ndarray,
) -> np.ndarray:
    spec = comp.spec
    return (
        (
            owners[:, None].astype(np.int64) * np.int64(spec.ring_size)
            + np.arange(int(spec.ring_size), dtype=np.int64)[None, :]
        )
        .reshape(-1)
        .astype(np.int32)
    )


def _owner_compressed_cache_rows(
    comp: Dsv4DeviceCompressorState,
    owners: np.ndarray,
) -> np.ndarray:
    spec = comp.spec
    return (
        (
            owners[:, None].astype(np.int64) * np.int64(spec.max_compressed_len)
            + np.arange(int(spec.max_compressed_len), dtype=np.int64)[None, :]
        )
        .reshape(-1)
        .astype(np.int32)
    )


def _unique_i32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int32)
    return np.unique(arr).astype(np.int32)


def _checkpoint_positions(seq_len: int, num_tokens: int) -> np.ndarray:
    if int(seq_len) < 0:
        raise ValueError("seq_len must be non-negative")
    if int(num_tokens) < 0:
        raise ValueError("num_tokens must be non-negative")
    return (np.arange(int(num_tokens), dtype=np.int64) + np.int64(seq_len)).astype(
        np.int64
    )


def _owner_swa_rows_for_positions(
    state: Dsv4DeviceState,
    owner_id: int,
    positions: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(positions, dtype=np.int64) % np.int64(state.window_size)
    return _unique_i32(np.int64(owner_id) * np.int64(state.window_size) + offsets)


def _owner_ring_rows_for_positions(
    comp: Dsv4DeviceCompressorState,
    owner_id: int,
    positions: np.ndarray,
) -> np.ndarray:
    spec = comp.spec
    offsets = np.asarray(positions, dtype=np.int64) % np.int64(spec.ring_size)
    return _unique_i32(np.int64(owner_id) * np.int64(spec.ring_size) + offsets)


def _owner_compressed_cache_rows_for_positions(
    comp: Dsv4DeviceCompressorState,
    owner_id: int,
    positions: np.ndarray,
) -> np.ndarray:
    spec = comp.spec
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        return np.empty((0,), dtype=np.int32)
    compressed = pos[(pos + 1) % np.int64(spec.compress_ratio) == 0]
    if compressed.size == 0:
        return np.empty((0,), dtype=np.int32)
    cpos = compressed // np.int64(spec.compress_ratio)
    return _unique_i32(np.int64(owner_id) * np.int64(spec.max_compressed_len) + cpos)


def _state_has_device_buffers(state: Dsv4DeviceState) -> bool:
    for layer in state.layers:
        if hasattr(layer.swa_kv_cache, "tensor_ref"):
            return True
        for comp in (layer.compressor, layer.indexer):
            if comp is None:
                continue
            if hasattr(comp.kv_score_state, "tensor_ref"):
                return True
            if hasattr(comp.compressed_kv_cache, "tensor_ref"):
                return True
    return False


def _write_state_rows(
    dst: Any,
    rows: np.ndarray,
    values: Any,
    *,
    artifacts_dir: str | Path | None = None,
) -> None:
    row_ids = np.asarray(rows, dtype=np.int32).reshape(-1)
    if row_ids.size == 0:
        return
    shape = tuple(int(dim) for dim in getattr(dst, "shape"))
    if int(row_ids.min()) < 0 or int(row_ids.max()) >= shape[0]:
        raise ValueError(
            "DSV4 owner row outside buffer: "
            f"min_row={int(row_ids.min())}, max_row={int(row_ids.max())}, "
            f"rows={shape[0]}"
        )
    if hasattr(values, "tensor_ref"):
        value_shape = tuple(int(dim) for dim in getattr(values, "shape"))
        if row_ids.shape != (value_shape[0],):
            raise ValueError(
                f"rows/values mismatch: rows={row_ids.shape}, values={value_shape}"
            )
        if len(shape) != 2 or len(value_shape) != 2 or value_shape[1] != shape[1]:
            raise ValueError(
                f"bad state row write shape: dst={shape}, values={value_shape}"
            )
        if not hasattr(dst, "tensor_ref"):
            raise RuntimeError(
                "refusing to download DeviceTensor state checkpoint into a "
                "host-visible buffer"
            )

        from nkipy_serving.attention.deepseek_v4.kernels import (
            run_write_kv_slots_device,
        )

        DeviceTensor = get_device_tensor_cls()

        run_write_kv_slots_device(
            kv_cache=dst,
            kv_new=values,
            slot_mapping=DeviceTensor.from_numpy(
                np.ascontiguousarray(row_ids),
                name="dsv4_restore_state_row_ids",
            ),
            artifacts_dir=artifacts_dir,
        )
        return

    vals = np.asarray(values)
    if row_ids.shape != (vals.shape[0],):
        raise ValueError(
            f"rows/values mismatch: rows={row_ids.shape}, values={vals.shape}"
        )
    if len(shape) != 2 or vals.ndim != 2 or vals.shape[1] != shape[1]:
        raise ValueError(f"bad state row write shape: dst={shape}, values={vals.shape}")
    if isinstance(dst, np.ndarray):
        dst[row_ids.astype(np.int64)] = vals.astype(dst.dtype, copy=False)
        return
    if not hasattr(dst, "tensor_ref"):
        raise TypeError(f"unsupported DSV4 state buffer type {type(dst)!r}")

    from nkipy_serving.attention.deepseek_v4.kernels import (
        run_write_kv_slots_device,
    )

    DeviceTensor = get_device_tensor_cls()

    for offset in range(0, row_ids.size, 128):
        end = min(offset + 128, row_ids.size)
        run_write_kv_slots_device(
            kv_cache=dst,
            kv_new=DeviceTensor.from_numpy(
                np.ascontiguousarray(vals[offset:end].astype(dst.dtype)),
                name="dsv4_clear_state_rows",
            ),
            slot_mapping=DeviceTensor.from_numpy(
                np.ascontiguousarray(row_ids[offset:end]),
                name="dsv4_clear_state_row_ids",
            ),
            artifacts_dir=artifacts_dir,
        )


def _read_state_rows_for_checkpoint(
    src: Any,
    rows: np.ndarray,
    *,
    name: str,
    artifacts_dir: str | Path | None = None,
) -> Any:
    row_ids = np.asarray(rows, dtype=np.int32).reshape(-1)
    if row_ids.size == 0:
        width = int(getattr(src, "shape")[1])
        dtype = getattr(src, "dtype")
        return np.zeros((0, width), dtype=dtype)
    shape = tuple(int(dim) for dim in getattr(src, "shape"))
    if len(shape) != 2:
        raise ValueError(f"bad state row read shape: src={shape}")
    if int(row_ids.min()) < 0 or int(row_ids.max()) >= shape[0]:
        raise ValueError(
            "DSV4 owner checkpoint row outside buffer: "
            f"min_row={int(row_ids.min())}, max_row={int(row_ids.max())}, "
            f"rows={shape[0]}"
        )
    if isinstance(src, np.ndarray):
        return np.asarray(src[row_ids.astype(np.int64)]).copy()
    if not hasattr(src, "tensor_ref"):
        raise TypeError(f"unsupported DSV4 state buffer type {type(src)!r}")

    from nkipy_serving.ops.deepseek_v4.state_copy import (
        run_gather_state_rows_device,
    )

    DeviceTensor = get_device_tensor_cls()

    return run_gather_state_rows_device(
        src=src,
        rows=DeviceTensor.from_numpy(
            np.ascontiguousarray(row_ids),
            name=f"{name}.checkpoint_rows",
        ),
        name=f"{name}.checkpoint_values",
        artifacts_dir=artifacts_dir,
    )


def _checkpoint_compressor_owner(
    comp: Dsv4DeviceCompressorState | None,
    owners: np.ndarray,
    *,
    name: str,
    positions: np.ndarray | None = None,
    artifacts_dir: str | Path | None = None,
) -> Dsv4DeviceCompressorCheckpoint | None:
    if comp is None:
        return None
    if positions is None:
        ring_rows = _owner_ring_rows(comp, owners)
        cache_rows = _owner_compressed_cache_rows(comp, owners)
    else:
        owner_id = int(owners[0])
        ring_rows = _owner_ring_rows_for_positions(comp, owner_id, positions)
        cache_rows = _owner_compressed_cache_rows_for_positions(
            comp,
            owner_id,
            positions,
        )
    return Dsv4DeviceCompressorCheckpoint(
        kv_score_rows=ring_rows,
        kv_score_state=_read_state_rows_for_checkpoint(
            comp.kv_score_state,
            ring_rows,
            name=f"{name}.kv_score_state",
            artifacts_dir=artifacts_dir,
        ),
        compressed_kv_rows=cache_rows,
        compressed_kv_cache=_read_state_rows_for_checkpoint(
            comp.compressed_kv_cache,
            cache_rows,
            name=f"{name}.compressed_kv_cache",
            artifacts_dir=artifacts_dir,
        ),
    )


def checkpoint_dsv4_device_state_owner(
    state: Dsv4DeviceState,
    owner_id: int,
    *,
    seq_len: int | None = None,
    num_tokens: int | None = None,
    artifacts_dir: str | Path | None = None,
) -> Dsv4DeviceStateCheckpoint:
    """Snapshot all mutable rows for one request-state owner.

    If ``num_tokens`` is provided, only rows that can be overwritten by
    speculative positions ``[seq_len, seq_len + num_tokens)`` are copied.
    DeviceTensor checkpoints require this bounded form to avoid full-owner
    compressed-KV snapshots at V4 context length.
    """

    owners = _owner_ids_array((int(owner_id),), max_owners=int(state.max_batch_size))
    if owners.size != 1:
        raise ValueError("checkpoint requires exactly one owner_id")
    if seq_len is not None and int(seq_len) < 0:
        raise ValueError("seq_len must be non-negative")
    if num_tokens is not None and int(num_tokens) < 0:
        raise ValueError("num_tokens must be non-negative")
    if num_tokens is not None and seq_len is None:
        raise ValueError("seq_len is required when num_tokens is provided")
    bounded = seq_len is not None and num_tokens is not None
    if _state_has_device_buffers(state) and not bounded:
        raise RuntimeError(
            "DSV4 DeviceTensor checkpoint requires seq_len and num_tokens; "
            "refusing full-owner snapshot on the production path"
        )
    positions = (
        _checkpoint_positions(int(seq_len), int(num_tokens)) if bounded else None
    )

    layers: list[Dsv4DeviceLayerCheckpoint] = []
    for layer in state.layers:
        if positions is None:
            swa_rows = _owner_swa_rows(state, owners)
        else:
            swa_rows = _owner_swa_rows_for_positions(
                state,
                int(owner_id),
                positions,
            )
        layers.append(
            Dsv4DeviceLayerCheckpoint(
                layer_id=int(layer.layer_id),
                swa_rows=swa_rows,
                swa_kv_cache=_read_state_rows_for_checkpoint(
                    layer.swa_kv_cache,
                    swa_rows,
                    name=f"layer{int(layer.layer_id)}.swa_kv_cache",
                    artifacts_dir=artifacts_dir,
                ),
                compressor=_checkpoint_compressor_owner(
                    layer.compressor,
                    owners,
                    name=f"layer{int(layer.layer_id)}.compressor",
                    positions=positions,
                    artifacts_dir=artifacts_dir,
                ),
                indexer=_checkpoint_compressor_owner(
                    layer.indexer,
                    owners,
                    name=f"layer{int(layer.layer_id)}.indexer",
                    positions=positions,
                    artifacts_dir=artifacts_dir,
                ),
            )
        )

    return Dsv4DeviceStateCheckpoint(
        owner_id=int(owner_id),
        seq_len=None if seq_len is None else int(seq_len),
        layers=tuple(layers),
    )


def _restore_compressor_checkpoint(
    comp: Dsv4DeviceCompressorState | None,
    checkpoint: Dsv4DeviceCompressorCheckpoint | None,
    *,
    name: str,
    artifacts_dir: str | Path | None,
) -> None:
    if checkpoint is None:
        if comp is not None:
            raise ValueError(f"checkpoint missing {name} state")
        return
    if comp is None:
        raise ValueError(f"checkpoint has {name} state but target does not")
    _write_state_rows(
        comp.kv_score_state,
        checkpoint.kv_score_rows,
        checkpoint.kv_score_state,
        artifacts_dir=artifacts_dir,
    )
    _write_state_rows(
        comp.compressed_kv_cache,
        checkpoint.compressed_kv_rows,
        checkpoint.compressed_kv_cache,
        artifacts_dir=artifacts_dir,
    )


def restore_dsv4_device_state_owner(
    state: Dsv4DeviceState,
    checkpoint: Dsv4DeviceStateCheckpoint,
    *,
    artifacts_dir: str | Path | None = None,
) -> None:
    """Restore one owner-local checkpoint into an existing DSV4 device state."""

    _owner_ids_array((int(checkpoint.owner_id),), max_owners=int(state.max_batch_size))
    if len(checkpoint.layers) != state.num_layers:
        raise ValueError(
            "checkpoint layer count does not match target state: "
            f"{len(checkpoint.layers)} != {state.num_layers}"
        )
    for layer, layer_checkpoint in zip(state.layers, checkpoint.layers, strict=True):
        if int(layer.layer_id) != int(layer_checkpoint.layer_id):
            raise ValueError(
                "checkpoint layer_id does not match target state: "
                f"{layer_checkpoint.layer_id} != {layer.layer_id}"
            )
        _write_state_rows(
            layer.swa_kv_cache,
            layer_checkpoint.swa_rows,
            layer_checkpoint.swa_kv_cache,
            artifacts_dir=artifacts_dir,
        )
        _restore_compressor_checkpoint(
            layer.compressor,
            layer_checkpoint.compressor,
            name=f"layer{int(layer.layer_id)}.compressor",
            artifacts_dir=artifacts_dir,
        )
        _restore_compressor_checkpoint(
            layer.indexer,
            layer_checkpoint.indexer,
            name=f"layer{int(layer.layer_id)}.indexer",
            artifacts_dir=artifacts_dir,
        )


def reset_dsv4_device_state(
    state: Dsv4DeviceState,
    *,
    writer: StateWriter | None = None,
) -> None:
    """Reset all persistent DSV4 device-state buffers to their empty values."""
    write = writer or _default_state_writer
    for layer in state.layers:
        swa_zero = _zero_like_state_buffer(
            layer.swa_kv_cache,
            (int(state.num_slots_per_layer), int(state.head_dim)),
            SEMANTIC_BF16_KV.dtype,
        )
        write(layer.swa_kv_cache, swa_zero)

        for comp in (layer.compressor, layer.indexer):
            if comp is None:
                continue
            spec = comp.spec
            kv_score = np.zeros(spec.state_shape, dtype=spec.state_dtype)
            _seed_kv_score_state(kv_score, spec)
            write(comp.kv_score_state, kv_score)
            write(
                comp.compressed_kv_cache,
                np.zeros(spec.compressed_cache_shape, dtype=spec.cache_dtype),
            )


def clear_dsv4_device_state_owners(
    state: Dsv4DeviceState,
    owner_ids: Iterable[int],
    *,
    artifacts_dir: str | Path | None = None,
) -> None:
    """Reset host-visible DSV4 rows owned by ``owner_ids``.

    DeviceTensor state intentionally does no per-request clearing here. Serving
    reads SWA/compressed caches through current sequence-length bounds, and
    compressor/indexer ring rows are written before they are consumed. Avoiding
    full owner wipes keeps request teardown from allocating temporary tensors or
    late-loading row-scatter kernels when HBM is already at peak. Full reset is
    still available through ``reset_dsv4_device_state`` for flush/warmup.
    """

    owners = _owner_ids_array(owner_ids, max_owners=int(state.max_batch_size))
    if owners.size == 0:
        return
    if _state_has_device_buffers(state):
        return

    for layer in state.layers:
        swa_rows = _owner_swa_rows(state, owners)
        swa_zero = np.zeros(
            (swa_rows.shape[0], int(state.head_dim)),
            dtype=getattr(layer.swa_kv_cache, "dtype", SEMANTIC_BF16_KV.dtype),
        )
        _write_state_rows(
            layer.swa_kv_cache,
            swa_rows,
            swa_zero,
            artifacts_dir=artifacts_dir,
        )

        for comp in (layer.compressor, layer.indexer):
            if comp is None:
                continue
            spec = comp.spec
            ring_rows = _owner_ring_rows(comp, owners)
            kv_score = np.zeros(
                (ring_rows.shape[0], int(spec.packed_width)),
                dtype=spec.state_dtype,
            )
            kv_score[:, int(spec.state_width) :] = -np.inf
            _write_state_rows(
                comp.kv_score_state,
                ring_rows,
                kv_score,
                artifacts_dir=artifacts_dir,
            )

            cache_rows = _owner_compressed_cache_rows(comp, owners)
            cache_zero = np.zeros(
                (cache_rows.shape[0], int(spec.head_dim)),
                dtype=getattr(comp.compressed_kv_cache, "dtype", spec.cache_dtype),
            )
            _write_state_rows(
                comp.compressed_kv_cache,
                cache_rows,
                cache_zero,
                artifacts_dir=artifacts_dir,
            )


def allocate_dsv4_device_state(
    alloc_device_cache: DeviceAllocator,
    *,
    layer_compress_ratios: Iterable[int],
    layer_has_indexer: Iterable[bool] | None = None,
    num_slots_per_layer: int,
    head_dim: int,
    indexer_head_dim: int | None = None,
    window_size: int,
    max_seq_len: int,
    max_batch_size: int,
    prefix: str = "dsv4",
    state_dtype: Any = np.float32,
    cache_dtype: Any = ml_dtypes.bfloat16,
    reserve_guard_owner: bool = False,
) -> Dsv4DeviceState:
    """Allocate all persistent DSV4 device state for one worker/lane.

    ``head_dim`` is the primary attention/KV width.  V4 indexer compressors
    can have a narrower local width, so their rolling state uses
    ``indexer_head_dim`` when provided.
    """

    ratios = tuple(int(r) for r in layer_compress_ratios)
    if not ratios:
        raise ValueError("layer_compress_ratios must be non-empty")
    if layer_has_indexer is None:
        has_indexer = tuple(r == 4 for r in ratios)
    else:
        has_indexer = tuple(bool(v) for v in layer_has_indexer)
    if len(has_indexer) != len(ratios):
        raise ValueError("layer_has_indexer length must match layer_compress_ratios")
    if int(num_slots_per_layer) < int(max_batch_size) * int(window_size):
        raise ValueError(
            "num_slots_per_layer must cover owner-local SWA rows: "
            f"num_slots_per_layer={int(num_slots_per_layer)}, "
            f"max_batch_size={int(max_batch_size)}, window_size={int(window_size)}"
        )

    layers: list[Dsv4DeviceLayerState] = []
    # Reserve one extra owner block as the bucketed-prefill padding sink. The
    # fused NKI prefill write compiles at the token bucket and redirects masked
    # / padding rows to this guard owner (index == max_batch_size); real owners
    # stay in [0, max_batch_size). The guard block is never read. See
    # dsv4_nki_writeswa_plan. SWA already has its own +1 sink via
    # num_slots_per_layer (executor.py), so only the compressor/indexer
    # ring-state and compressed caches need the guard owner here.
    real_owner_count = max(1, int(max_batch_size))
    # The guard owner is only allocated for the bucketed-prefill write path;
    # otherwise cache shapes stay byte-identical to the legacy layout.
    compressed_owner_count = (
        real_owner_count + 1 if bool(reserve_guard_owner) else real_owner_count
    )
    idx_head_dim = (
        int(indexer_head_dim) if indexer_head_dim is not None else int(head_dim)
    )
    for layer_id, (ratio, has_idx) in enumerate(zip(ratios, has_indexer)):
        swa = _alloc_zero(
            alloc_device_cache,
            (int(num_slots_per_layer), int(head_dim)),
            cache_dtype,
            name=f"{prefix}_layer{layer_id}_swa_kv",
        )
        compressor_state: Dsv4DeviceCompressorState | None = None
        indexer_state: Dsv4DeviceCompressorState | None = None
        if ratio > 0:
            comp_slots = compressed_owner_count * (int(max_seq_len) // ratio)
            spec = Dsv4CompressorStateSpec(
                layer_id=layer_id,
                compress_ratio=ratio,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                num_state_owners=compressed_owner_count,
                num_compressed_slots=comp_slots,
                state_dtype=state_dtype,
                cache_dtype=cache_dtype,
                has_guard_owner=bool(reserve_guard_owner),
            )
            compressor_state = Dsv4DeviceCompressorState(
                spec=spec,
                kv_score_state=_seed_kv_score_state(
                    _alloc_zero(
                        alloc_device_cache,
                        spec.state_shape,
                        state_dtype,
                        name=f"{prefix}_layer{layer_id}_compressor_kv_score",
                    ),
                    spec,
                ),
                compressed_kv_cache=_alloc_zero(
                    alloc_device_cache,
                    spec.compressed_cache_shape,
                    cache_dtype,
                    name=f"{prefix}_layer{layer_id}_compressed_kv",
                ),
            )
            if has_idx:
                indexer_spec = Dsv4CompressorStateSpec(
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    head_dim=idx_head_dim,
                    max_seq_len=max_seq_len,
                    num_state_owners=compressed_owner_count,
                    num_compressed_slots=comp_slots,
                    overlap=True,
                    state_dtype=state_dtype,
                    cache_dtype=cache_dtype,
                    has_guard_owner=bool(reserve_guard_owner),
                )
                indexer_state = Dsv4DeviceCompressorState(
                    spec=indexer_spec,
                    kv_score_state=_seed_kv_score_state(
                        _alloc_zero(
                            alloc_device_cache,
                            indexer_spec.state_shape,
                            state_dtype,
                            name=f"{prefix}_layer{layer_id}_indexer_kv_score",
                        ),
                        indexer_spec,
                    ),
                    compressed_kv_cache=_alloc_zero(
                        alloc_device_cache,
                        indexer_spec.compressed_cache_shape,
                        cache_dtype,
                        name=f"{prefix}_layer{layer_id}_indexer_kv",
                    ),
                )
        layers.append(
            Dsv4DeviceLayerState(
                layer_id=layer_id,
                swa_kv_cache=swa,
                compressor=compressor_state,
                indexer=indexer_state,
            )
        )

    return Dsv4DeviceState(
        layers=tuple(layers),
        num_slots_per_layer=int(num_slots_per_layer),
        head_dim=int(head_dim),
        window_size=int(window_size),
        max_seq_len=int(max_seq_len),
        max_batch_size=int(max_batch_size),
    )


__all__ = [
    "Dsv4CompressorStateSpec",
    "Dsv4DeviceCompressorCheckpoint",
    "Dsv4DeviceCompressorState",
    "Dsv4DeviceLayerCheckpoint",
    "Dsv4DeviceLayerState",
    "Dsv4DeviceState",
    "Dsv4DeviceStateCheckpoint",
    "Dsv4KVFormat",
    "SEMANTIC_BF16_KV",
    "allocate_dsv4_device_state",
    "checkpoint_dsv4_device_state_owner",
    "clear_dsv4_device_state_owners",
    "reset_dsv4_device_state",
    "restore_dsv4_device_state_owner",
]
