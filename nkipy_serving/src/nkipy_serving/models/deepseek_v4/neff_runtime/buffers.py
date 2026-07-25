"""Preallocated buffer accessors for the DSV4 NEFF runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from nkipy_serving.models.deepseek_v4.neff_runtime.resources.types import (
    Dsv4ProductBucket,
)
from nkipy_serving.runtime.device_tensor import (
    alias_device_value_shape as _alias_device_value_shape,
)
from nkipy_serving.runtime.device_tensor import (
    normalize_dtype as _normalize_dtype,
)


class Dsv4ProductBuffersMixin:
    def _active_moe_ep_output_for(self, *, rows: int, dim: int) -> Any | None:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            return None
        layer_id = getattr(self, "_active_moe_layer_id", None)
        is_decode = bool(getattr(self, "_active_moe_is_decode", False))
        if layer_id is None:
            return None
        outputs = (
            bucket.moe_decode_ep_outputs if is_decode else bucket.moe_prefill_ep_outputs
        )
        idx = int(layer_id)
        if idx < 0 or idx >= len(outputs):
            return None
        out = outputs[idx]
        if tuple(int(axis) for axis in getattr(out, "shape", ())) != (
            int(rows),
            int(dim),
        ):
            return None
        return out

    def _moe_ep_output_for_layer(
        self,
        *,
        layer_id: int,
        is_decode: bool,
        rows: int,
        dim: int,
    ) -> Any | None:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            return None
        outputs = (
            bucket.moe_decode_ep_outputs
            if bool(is_decode)
            else bucket.moe_prefill_ep_outputs
        )
        idx = int(layer_id)
        if idx < 0 or idx >= len(outputs):
            return None
        out = outputs[idx]
        if tuple(int(axis) for axis in getattr(out, "shape", ())) != (
            int(rows),
            int(dim),
        ):
            return None
        return out

    def _head_hidden_alias_for(
        self,
        *,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Any | None:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            return None
        base = getattr(bucket, "head_hidden_output", None)
        if base is None:
            return None
        if _normalize_dtype(getattr(base, "dtype", None)) != _normalize_dtype(dtype):
            return None
        base_elems = int(np.prod(tuple(int(dim) for dim in getattr(base, "shape", ()))))
        shape_t = tuple(int(dim) for dim in shape)
        if base_elems != int(np.prod(shape_t)):
            return None
        return _alias_device_value_shape(base, shape_t)

    def _attention_output_for(
        self, layer_id: int, token_bucket: int | None
    ) -> Any | None:
        active = getattr(self, "_active_product_bucket", None)
        if active is None and token_bucket is None:
            return None
        bucket = self._ensure_product_bucket(
            int(token_bucket) if token_bucket is not None else int(active.token_bucket)
        )
        idx = int(layer_id)
        if idx < 0 or idx >= len(bucket.attention_outputs):
            raise RuntimeError(
                "DSV4 product attention output missing for layer "
                f"{idx}; bucket has {len(bucket.attention_outputs)} layers"
            )
        return bucket.attention_outputs[idx]

    def _attention_output_scratch_for(
        self,
        bucket: Dsv4ProductBucket | None,
        *,
        rows: int,
        n_heads: int,
        head_dim: int,
    ) -> Any | None:
        if bucket is None:
            return None
        return self._bucket_scratch(
            bucket,
            "attention_out",
            (int(rows), int(n_heads), int(head_dim)),
            np.float32,
        )

    def _attention_owner_buffers_for(
        self,
        bucket: Dsv4ProductBucket | None,
        *,
        rows: int,
        primary: bool = False,
    ) -> tuple[np.ndarray | None, Any | None]:
        if bucket is None:
            return (None, None)
        rows_i = int(rows)
        if rows_i == int(bucket.token_bucket) and not bool(primary):
            return (bucket.owner_ids_host, bucket.owner_ids_dev)
        kind = "primary_owner_ids" if bool(primary) else "owner_ids"
        cache = getattr(self, "_attention_owner_host_cache", None)
        if cache is None:
            cache = {}
            self._attention_owner_host_cache = cache
        key = (int(bucket.token_bucket), kind, rows_i)
        host = cache.get(key)
        if host is None:
            host = np.zeros((rows_i,), dtype=np.int32)
            cache[key] = host
        dev = self._bucket_scratch(bucket, kind, (rows_i,), np.int32)
        return host, dev

    def _moe_outputs_for(
        self, layer_id: int, *, is_decode: bool
    ) -> tuple[Any, Any, Any]:
        bucket = getattr(self, "_active_product_bucket", None)
        if bucket is None:
            return (None, None, None)
        idx = int(layer_id)
        outputs = (
            bucket.moe_decode_outputs if bool(is_decode) else bucket.moe_prefill_outputs
        )
        ep_outputs = (
            bucket.moe_decode_ep_outputs
            if bool(is_decode)
            else bucket.moe_prefill_ep_outputs
        )
        tp_outputs = (
            bucket.moe_decode_tp_outputs
            if bool(is_decode)
            else bucket.moe_prefill_tp_outputs
        )
        if idx < 0 or idx >= len(outputs):
            raise RuntimeError(
                "DSV4 product MoE output missing for layer "
                f"{idx}; bucket has {len(outputs)} layers"
            )
        return outputs[idx], ep_outputs[idx], tp_outputs[idx]

    def _attention_bucket_for(
        self,
        y: Any,
        token_bucket: int | None,
    ) -> int | None:
        candidates: list[int] = []
        if token_bucket is not None:
            candidates.append(int(token_bucket))
        active = getattr(self, "_active_product_bucket", None)
        if active is not None:
            candidates.append(int(active.token_bucket))
        backend_bucket = getattr(
            getattr(self, "attention_backend", None),
            "active_bucket",
            None,
        )
        if backend_bucket is not None:
            candidates.append(int(backend_bucket))
        shape = tuple(int(dim) for dim in getattr(y, "shape", ()))
        if len(shape) >= 2:
            candidates.append(int(shape[0]) * int(shape[1]))
        if not candidates:
            return None
        return max(candidates)

    def _compressed_attention_bucket_for_tokens(
        self,
        y_tokens: int,
        token_bucket: int | None,
    ) -> int:
        min_rows = max(2, int(y_tokens))
        if int(y_tokens) <= 0:
            fallback = self._attention_bucket_for(None, token_bucket)
            return min_rows if fallback is None else max(min_rows, int(fallback))

        full_buckets: list[int] = []
        if token_bucket is not None:
            full_buckets.append(int(token_bucket))
        active = getattr(self, "_active_product_bucket", None)
        if active is not None:
            full_buckets.append(int(active.token_bucket))
        backend_bucket = getattr(
            getattr(self, "attention_backend", None),
            "active_bucket",
            None,
        )
        if backend_bucket is not None:
            full_buckets.append(int(backend_bucket))

        max_requests = max(1, int(getattr(self, "max_requests_per_step", 1)))
        candidates: list[int] = [2]
        for bucket in full_buckets:
            if bucket <= 0:
                continue
            lane_bucket = max(2, (int(bucket) + max_requests - 1) // max_requests)
            candidates.extend((lane_bucket, int(bucket)))

        viable = sorted({int(v) for v in candidates if int(v) >= min_rows})
        if viable:
            return viable[0]
        return min_rows

    def _compressed_attention_bucket_for(
        self,
        y: Any,
        token_bucket: int | None,
    ) -> int:
        shape = tuple(int(dim) for dim in getattr(y, "shape", ()))
        y_tokens = int(shape[0] * int(shape[1])) if len(shape) >= 2 else 0
        return self._compressed_attention_bucket_for_tokens(y_tokens, token_bucket)
