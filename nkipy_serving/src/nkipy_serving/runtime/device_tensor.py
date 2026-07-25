"""Shared DeviceTensor metadata and compile-sample utilities."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np


def get_device_tensor_cls() -> Any:
    """Lazy import of NKIPy DeviceTensor, cached by Python's import system."""

    from nkipy.runtime import DeviceTensor

    return DeviceTensor


def is_device_tensor(value: Any, *, require_numpy: bool = False) -> bool:
    """Return whether ``value`` looks like an NKIPy DeviceTensor."""

    if not hasattr(value, "tensor_ref"):
        return False
    return not require_numpy or hasattr(value, "numpy")


def is_device_array_like(value: Any) -> bool:
    """Return whether ``value`` is non-host array metadata with a shape."""

    return hasattr(value, "shape") and not isinstance(value, np.ndarray)


def normalize_dtype(dtype: Any, fallback: Any | None = None) -> Any:
    """Return a NumPy-compatible dtype object, using ``fallback`` if needed."""

    if dtype is None:
        if fallback is None:
            raise ValueError("dtype is required")
        dtype = fallback
    dtype_s = str(dtype)
    if dtype_s in ("bfloat16", "bf16"):
        return ml_dtypes.bfloat16
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        if "bfloat16" in dtype_s or dtype_s == "bf16":
            return ml_dtypes.bfloat16
        if "float32" in dtype_s or dtype_s == "f32":
            return np.float32
        if "float16" in dtype_s or dtype_s == "f16":
            return np.float16
        if "int32" in dtype_s or dtype_s == "i32":
            return np.int32
        if fallback is None:
            raise
        return np.dtype(fallback)


def dtype_like(value: Any, fallback: Any | None = None) -> Any:
    """Return a NumPy-compatible dtype for an ndarray or DeviceTensor-like value."""

    try:
        return normalize_dtype(getattr(value, "dtype", None), fallback)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tensor {value!r} has no valid dtype") from exc


def sample_like(
    value: Any,
    dtype: Any | None = None,
    *,
    fill: str = "empty",
    fallback_dtype: Any = ml_dtypes.bfloat16,
) -> np.ndarray:
    """Return a host array with the same logical shape for kernel tracing."""

    shape = tuple(int(dim) for dim in getattr(value, "shape"))
    sample_dtype = normalize_dtype(
        dtype if dtype is not None else getattr(value, "dtype", None),
        fallback_dtype,
    )
    try:
        if fill == "zeros":
            return np.zeros(shape, dtype=sample_dtype)
        if fill == "empty":
            return np.empty(shape, dtype=sample_dtype)
    except (TypeError, ValueError):
        if fill == "zeros":
            return np.zeros(shape, dtype=fallback_dtype)
        if fill == "empty":
            return np.empty(shape, dtype=fallback_dtype)
    raise ValueError(f"unsupported sample fill mode: {fill!r}")


def alias_device_value_shape(
    value: Any,
    shape: tuple[int, ...],
    *,
    default_name: str = "dsv4_shape_alias",
) -> Any | None:
    """Return a DeviceTensor with a different logical shape and same buffer."""

    shape_t = tuple(int(dim) for dim in shape)
    if tuple(int(dim) for dim in getattr(value, "shape", ())) == shape_t:
        return value
    if not hasattr(value, "tensor_ref"):
        return None
    try:
        DeviceTensor = get_device_tensor_cls()
    except ImportError:
        return None
    return DeviceTensor(
        tensor_ref=value.tensor_ref,
        shape=shape_t,
        dtype=value.dtype,
        name=getattr(value, "name", default_name),
    )


def alias_device_value_first_dim_slice(
    value: Any,
    *,
    start: int,
    size: int,
    default_name: str = "dsv4",
) -> Any | None:
    """Return a DeviceTensor alias for a contiguous first-dimension slice."""

    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    start_i = int(start)
    size_i = int(size)
    if len(shape) == 0 or start_i < 0 or size_i < 0 or start_i + size_i > shape[0]:
        return None
    target_shape = (size_i, *shape[1:])
    if start_i == 0:
        return alias_device_value_shape(
            value,
            target_shape,
            default_name=f"{default_name}_shape_alias",
        )
    if not hasattr(value, "tensor_ref"):
        return None
    try:
        DeviceTensor = get_device_tensor_cls()
        from spike.spike_singleton import get_spike_singleton
    except ImportError:
        return None
    row_elements = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    itemsize = int(np.dtype(value.dtype).itemsize)
    byte_offset = int(start_i * row_elements * itemsize)
    byte_size = int(size_i * row_elements * itemsize)
    alias_name = f"{getattr(value, 'name', default_name)}_slice_{start_i}_{size_i}"
    try:
        tensor_ref = get_spike_singleton().slice_from_tensor(
            value.tensor_ref,
            byte_offset,
            byte_size,
            alias_name,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return DeviceTensor(
        tensor_ref=tensor_ref,
        shape=target_shape,
        dtype=value.dtype,
        name=alias_name,
    )
