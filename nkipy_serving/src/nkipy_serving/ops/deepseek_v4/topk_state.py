"""Device-side mutation helpers for DSV4 sparse top-k metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nkipy_serving.runtime.device_tensor import dtype_like as _dtype_like
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


_TOPK_TAIL_INSERT_KERNEL_CACHE: dict[tuple, Any] = {}


def topk_tail_insert_oracle(
    topk_global_t: np.ndarray,
    topk_mask: np.ndarray,
    topk_lens: np.ndarray,
    safe_extra: np.ndarray,
    extra_mask: np.ndarray,
    extra_lens: np.ndarray,
    *,
    tail_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU reference for appending compressed/indexer top-k into K-tail."""

    topk_t = np.asarray(topk_global_t)
    mask = np.asarray(topk_mask)
    lens = np.asarray(topk_lens)
    safe = np.asarray(safe_extra, dtype=np.int32)
    extra_m = np.asarray(extra_mask)
    extra_l = np.asarray(extra_lens, dtype=np.int32).reshape(-1, 1)
    if safe.ndim != 2:
        raise ValueError(f"safe_extra must be [tokens, k_extra], got {safe.shape}")
    total, k_extra = safe.shape
    if topk_t.ndim != 2 or topk_t.shape[1] < total:
        raise ValueError(f"topk_global_t must be [max_k, >=tokens], got {topk_t.shape}")
    if mask.shape[0] < total or mask.shape[1] < int(tail_start) + k_extra:
        raise ValueError(f"topk_mask shape too small: {mask.shape}")
    if lens.shape[0] < total:
        raise ValueError(f"topk_lens shape too small: {lens.shape}")
    if extra_m.shape != (total, k_extra):
        raise ValueError(f"extra_mask must be {safe.shape}, got {extra_m.shape}")
    if extra_l.shape != (total, 1):
        raise ValueError(f"extra_lens must be [{total}, 1], got {extra_l.shape}")

    topk_t[int(tail_start) : int(tail_start) + k_extra, :total] = safe.T
    mask[:total, int(tail_start) : int(tail_start) + k_extra] = extra_m.astype(
        mask.dtype,
        copy=False,
    )
    lens[:total, :1] = lens[:total, :1] + extra_l.astype(lens.dtype, copy=False)
    return topk_t, mask, lens


if _NKI_AVAILABLE:

    def _make_topk_tail_insert_kernel(tail_start: int):
        tail = int(tail_start)

        @_nki.jit
        def topk_tail_insert_kernel(
            topk_global_t: "nt.tensor[nt.mutable]",
            topk_mask: "nt.tensor[nt.mutable]",
            topk_lens: "nt.tensor[nt.mutable]",
            safe_extra: "nt.tensor",
            extra_mask: "nt.tensor",
            extra_lens: "nt.tensor",
        ):
            """Append ``safe_extra`` into ``topk_*`` K-tail in place."""

            total, k_extra = safe_extra.shape
            MAX_T = 128
            n_tiles = (total + MAX_T - 1) // MAX_T

            for ti in nl.affine_range(n_tiles):
                if total <= MAX_T:
                    cur = total
                    t0 = 0
                else:
                    cur = MAX_T
                    t0 = ti * MAX_T

                i_t = nl.arange(cur)[:, None]
                lens_sb = nl.load(topk_lens[t0 : t0 + cur, 0:1])
                extra_lens_sb = nl.load(extra_lens[t0 : t0 + cur, 0:1])
                new_lens = nl.add(lens_sb, extra_lens_sb)

                for kk in nl.affine_range(k_extra):
                    safe_col = nl.load(safe_extra[t0 : t0 + cur, kk : kk + 1])
                    mask_col = nl.load(extra_mask[t0 : t0 + cur, kk : kk + 1])
                    nl.store(
                        dst=topk_global_t[tail + kk, t0 + i_t],
                        value=safe_col,
                    )
                    nl.store(
                        dst=topk_mask[t0 + i_t, tail + kk],
                        value=mask_col,
                    )
                nl.store(
                    dst=topk_lens[t0 : t0 + cur, 0:1],
                    value=new_lens,
                )
            return topk_global_t, topk_mask, topk_lens

        return topk_tail_insert_kernel


def _make_topk_tail_insert_entry(tail_start: int):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    kernel = _make_topk_tail_insert_kernel(int(tail_start))

    def _entry(
        topk_global_t,
        topk_mask,
        topk_lens,
        safe_extra,
        extra_mask,
        extra_lens,
    ):
        return kernel(
            topk_global_t,
            topk_mask,
            topk_lens,
            safe_extra,
            extra_mask,
            extra_lens,
        )

    return _entry


def run_topk_tail_insert_device(
    *,
    topk_global_t: Any,
    topk_mask: Any,
    topk_lens: Any,
    safe_extra: Any,
    extra_mask: Any,
    extra_lens: Any,
    tail_start: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Device in-place K-tail insert for compressed/indexer top-k."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    topk_shape = tuple(int(dim) for dim in getattr(topk_global_t, "shape"))
    mask_shape = tuple(int(dim) for dim in getattr(topk_mask, "shape"))
    lens_shape = tuple(int(dim) for dim in getattr(topk_lens, "shape"))
    safe_shape = tuple(int(dim) for dim in getattr(safe_extra, "shape"))
    extra_mask_shape = tuple(int(dim) for dim in getattr(extra_mask, "shape"))
    extra_lens_shape = tuple(int(dim) for dim in getattr(extra_lens, "shape"))
    if len(topk_shape) != 2 or len(mask_shape) != 2:
        raise ValueError(f"bad topk/mask shapes: {topk_shape}/{mask_shape}")
    if len(safe_shape) != 2:
        raise ValueError(f"safe_extra must be [tokens, k_extra], got {safe_shape}")
    total, k_extra = safe_shape
    if extra_mask_shape != safe_shape:
        raise ValueError(
            f"extra_mask shape {extra_mask_shape} != safe_extra {safe_shape}"
        )
    if extra_lens_shape != (total, 1):
        raise ValueError(f"extra_lens must be [{total}, 1], got {extra_lens_shape}")
    if lens_shape[0] != total or lens_shape[1] != 1:
        raise ValueError(f"topk_lens must be [{total}, 1], got {lens_shape}")
    if mask_shape[0] != total:
        raise ValueError(f"topk_mask token dim {mask_shape[0]} != {total}")
    if int(tail_start) < 0 or int(tail_start) + k_extra > topk_shape[0]:
        raise ValueError(
            f"tail_start={tail_start}, k_extra={k_extra} exceed topk K={topk_shape[0]}"
        )
    if mask_shape[1] < int(tail_start) + k_extra:
        raise ValueError(
            f"tail_start={tail_start}, k_extra={k_extra} exceed mask K={mask_shape[1]}"
        )
    if total == 0:
        return topk_global_t, topk_mask, topk_lens
    if total > 128 and total % 128 != 0:
        raise NotImplementedError(
            "topk_tail_insert supports tokens <= 128 or a multiple of 128"
        )

    cache = _TOPK_TAIL_INSERT_KERNEL_CACHE if _kernel_cache is None else _kernel_cache
    cache_key = (
        "topk_tail_insert",
        topk_shape,
        mask_shape,
        lens_shape,
        safe_shape,
        int(tail_start),
        str(_dtype_like(topk_global_t)),
        str(_dtype_like(topk_mask)),
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = compile_and_load_with_lock(
            _device_kernel_cls,
            _make_topk_tail_insert_entry(int(tail_start)),
            _sample_like(topk_global_t),
            _sample_like(topk_mask),
            _sample_like(topk_lens),
            _sample_like(safe_extra),
            _sample_like(extra_mask),
            _sample_like(extra_lens),
            name=f"dsv4_topk_tail_insert_t{total}_k{k_extra}_at{int(tail_start)}",
            build_dir=artifacts_dir,
            namespace="dsv4_attention_kernels",
        )
        cache[cache_key] = kernel

    kernel(
        inputs={
            "topk_global_t.must_alias_input": topk_global_t,
            "topk_mask.must_alias_input": topk_mask,
            "topk_lens.must_alias_input": topk_lens,
            "safe_extra": safe_extra,
            "extra_mask": extra_mask,
            "extra_lens": extra_lens,
        },
        outputs={
            "topk_global_t": topk_global_t,
            "topk_mask": topk_mask,
            "topk_lens": topk_lens,
        },
    )
    return topk_global_t, topk_mask, topk_lens


__all__ = [
    "run_topk_tail_insert_device",
    "topk_tail_insert_oracle",
]
