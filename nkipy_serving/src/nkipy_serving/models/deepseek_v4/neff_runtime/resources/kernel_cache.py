"""Product-kernel cache key canonicalization for DSV4 NEFF runtime."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from nkipy_serving.runtime.device_tensor import normalize_dtype as _normalize_dtype


def _canonical_dtype_key(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        dtype_s = dtype.strip()
        exact_aliases = {
            "bfloat16",
            "bf16",
            "float32",
            "f32",
            "float16",
            "f16",
            "int32",
            "i32",
            "int64",
            "i64",
            "uint32",
            "u32",
            "bool",
        }
        class_repr = dtype_s.startswith("<class '") and dtype_s.endswith("'>")
        if dtype_s not in exact_aliases and not class_repr:
            return None
    else:
        module = str(getattr(dtype, "__module__", ""))
        if not module.startswith(("ml_dtypes", "numpy")) and not isinstance(
            dtype,
            np.dtype,
        ):
            return None
    try:
        return str(np.dtype(_normalize_dtype(dtype)))
    except (TypeError, ValueError):
        return None


def _canonical_product_kernel_cache_key(value: Any) -> Any:
    dtype_key = _canonical_dtype_key(value)
    if dtype_key is not None:
        return dtype_key
    if isinstance(value, tuple):
        return tuple(_canonical_product_kernel_cache_key(v) for v in value)
    return value


class _ProductKernelCache(dict[tuple[Any, ...], Any]):
    """Dict that canonicalizes dtype-bearing product kernel cache keys."""

    @staticmethod
    def _key(key: Any) -> Any:
        return _canonical_product_kernel_cache_key(key)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._key(key), default)

    def __setitem__(self, key: tuple[Any, ...], value: Any) -> None:
        super().__setitem__(self._key(key), value)

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(self._key(key))

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._key(key))

    def pop(self, key: Any, default: Any = ...) -> Any:
        if default is ...:
            return super().pop(self._key(key))
        return super().pop(self._key(key), default)

    def setdefault(self, key: tuple[Any, ...], default: Any = None) -> Any:
        return super().setdefault(self._key(key), default)


def _kernel_cache() -> _ProductKernelCache:
    return _ProductKernelCache()


def _product_kernel_cache_digest(key: tuple[Any, ...]) -> str:
    canonical_key = _canonical_product_kernel_cache_key(key)
    return hashlib.sha1(repr(canonical_key).encode("utf-8")).hexdigest()[:16]


def _product_canonical_neff_cache_key(
    namespace: str,
    version: str,
    key: tuple[Any, ...],
) -> str:
    return f"{namespace}:{version}:{_product_kernel_cache_digest(key)}"
