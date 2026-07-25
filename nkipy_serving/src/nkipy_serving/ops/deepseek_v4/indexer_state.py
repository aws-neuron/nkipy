"""Device-cache consumers for DeepSeek-V4 indexer state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from nkipy_serving.runtime.device_tensor import dtype_like as _dtype_like
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.device_tensor import sample_like as _sample_like
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

P_MAX = 128
D_BLOCK = 128


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


_INDEXER_SCORE_FROM_CACHE_KERNEL_CACHE: dict[tuple, Any] = {}


def _indexer_score_from_cache_cache_key(
    *,
    q_T: Any,
    kv_cache: Any,
    owner_ids: Any,
    w: Any,
    kv_len: int,
    max_compressed_len: int,
) -> tuple[Any, ...]:
    return (
        "indexer_score_from_cache",
        tuple(int(dim) for dim in getattr(q_T, "shape")),
        tuple(int(dim) for dim in getattr(kv_cache, "shape")),
        tuple(int(dim) for dim in getattr(owner_ids, "shape")),
        tuple(int(dim) for dim in getattr(w, "shape")),
        int(kv_len),
        int(max_compressed_len),
        str(_dtype_like(kv_cache)),
    )


def indexer_score_from_cache_oracle(
    q: np.ndarray,
    kv_cache: np.ndarray,
    owner_ids: np.ndarray,
    w: np.ndarray,
    *,
    kv_len: int,
    max_compressed_len: int,
) -> np.ndarray:
    """Reference for indexer score reading a flat persistent KV cache."""

    if q.ndim != 3:
        raise ValueError(f"q must be [B,h,d], got {q.shape}")
    if kv_cache.ndim != 2:
        raise ValueError(f"kv_cache must be [slots,d], got {kv_cache.shape}")
    owners = np.asarray(owner_ids, dtype=np.int64).reshape(-1)
    if owners.shape != (q.shape[0],):
        raise ValueError(f"owner_ids must be [B={q.shape[0]}], got {owners.shape}")
    if w.shape != (q.shape[0], q.shape[1]):
        raise ValueError(f"w shape mismatch: got {w.shape}, expected {q.shape[:2]}")
    if kv_cache.shape[1] != q.shape[2]:
        raise ValueError(f"q/kv d mismatch: {q.shape[2]} vs {kv_cache.shape[1]}")
    t = int(kv_len)
    max_t = int(max_compressed_len)
    if t < 0 or t > max_t:
        raise ValueError(f"kv_len={t} must be in [0,{max_t}]")

    rows = owners[:, None] * np.int64(max_t) + np.arange(t, dtype=np.int64)[None, :]
    kv = np.asarray(kv_cache, dtype=np.float32)[rows]
    qf = np.asarray(q, dtype=np.float32)
    wf = np.asarray(w, dtype=np.float32)
    idx = np.einsum("bhd,btd->bht", qf, kv)
    idx = np.maximum(idx, 0.0) * wf[..., None]
    return idx.sum(axis=1).astype(np.float32)


if _NKI_AVAILABLE:

    def _make_indexer_score_from_cache_kernel(
        *,
        kv_len: int,
        max_compressed_len: int,
    ):
        out_t = int(kv_len)
        max_t = int(max_compressed_len)

        @_nki.jit
        def indexer_score_from_cache_kernel(
            q_T: "nt.tensor",
            kv_cache: "nt.tensor",
            owner_ids: "nt.tensor",
            w: "nt.tensor",
        ):
            """Indexer score with in-kernel compressed-cache gather.

            ``q_T`` is ``[B,d,h]`` with ``d`` on partition. ``kv_cache`` is
            flat ``[owners * max_compressed_len, d]``.
            """

            B = q_T.shape[0]
            d = q_T.shape[1]
            h = q_T.shape[2]
            if d != D_BLOCK:
                raise ValueError(f"d={d} must equal D_BLOCK={D_BLOCK}")
            if h > P_MAX:
                raise ValueError(f"h={h} must fit partition size {P_MAX}")

            n_tiles = (out_t + P_MAX - 1) // P_MAX
            last_tile = out_t - (n_tiles - 1) * P_MAX
            out = nl.ndarray((B, out_t), dtype=nl.float32, buffer=nl.shared_hbm)
            ones_h = nl.ndarray((par_dim(h), 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            ones_h[...] = nisa.memset(shape=(h, 1), value=1.0, dtype=nl.bfloat16)

            for bi in nl.affine_range(B):
                q_sb = nl.load(q_T[bi, :, :])
                owner = nl.load(owner_ids[bi : bi + 1])
                base = nl.multiply(owner, nl.int32(max_t))

                w_slice = nl.load(w[bi : bi + 1, :])
                w_on_part_psum = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                w_on_part_psum[...] = nisa.nc_transpose(
                    w_slice,
                    engine=nisa.tensor_engine,
                )
                w_on_part = nl.ndarray(
                    (par_dim(h), 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                w_on_part[...] = nl.copy(w_on_part_psum)

                for kt in nl.static_range(n_tiles):
                    if kt < n_tiles - 1:
                        tile_t = P_MAX
                    else:
                        tile_t = last_tile
                    tile_start = kt * P_MAX
                    i_t = nl.arange(tile_t)[:, None]
                    i_d = nl.arange(d)[None, :]
                    rows = nl.add(
                        nl.broadcast_to(
                            base.reshape((1, 1)),
                            shape=(tile_t, 1),
                        ),
                        nl.add(i_t, nl.int32(tile_start)),
                    )
                    kv_gathered = nl.ndarray(
                        (par_dim(tile_t), d),
                        dtype=kv_cache.dtype,
                        buffer=nl.sbuf,
                    )
                    kv_gathered[i_t, i_d] = nl.load(kv_cache[rows[i_t, 0], i_d])
                    kv_T_psum = nl.ndarray(
                        (par_dim(d), tile_t),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    kv_T_psum[...] = nisa.nc_transpose(
                        kv_gathered,
                        engine=nisa.tensor_engine,
                    )
                    kv_T = nl.ndarray(
                        (par_dim(d), tile_t),
                        dtype=kv_cache.dtype,
                        buffer=nl.sbuf,
                    )
                    kv_T[...] = nl.copy(kv_T_psum, dtype=kv_cache.dtype)

                    qk_psum = nl.zeros(
                        (par_dim(h), tile_t),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    qk_psum[...] = nisa.nc_matmul(q_sb, kv_T)

                    relu_qk = nl.ndarray(
                        (par_dim(h), tile_t),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    relu_qk[...] = nl.maximum(qk_psum, nl.float32(0.0))
                    scored = nl.ndarray(
                        (par_dim(h), tile_t),
                        dtype=nl.float32,
                        buffer=nl.sbuf,
                    )
                    scored[...] = nisa.tensor_scalar(
                        data=relu_qk,
                        op0=nl.multiply,
                        operand0=w_on_part,
                        dtype=nl.float32,
                    )

                    scored_bf = nl.ndarray(
                        (par_dim(h), tile_t),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    scored_bf[...] = nl.copy(scored, dtype=nl.bfloat16)
                    score_psum = nl.zeros(
                        (par_dim(1), tile_t),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    score_psum[...] = nisa.nc_matmul(ones_h, scored_bf)
                    nl.store(
                        out[bi : bi + 1, tile_start : tile_start + tile_t],
                        score_psum,
                    )

            return out

        return indexer_score_from_cache_kernel


def _make_indexer_score_from_cache_entry(*, kv_len: int, max_compressed_len: int):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    kernel = _make_indexer_score_from_cache_kernel(
        kv_len=kv_len,
        max_compressed_len=max_compressed_len,
    )

    def _entry(q_T, kv_cache, owner_ids, w):
        return kernel(q_T, kv_cache, owner_ids, w)

    return _entry


def precompile_indexer_score_from_cache_device(
    *,
    q_T_shape: tuple[int, ...],
    q_T_dtype: Any,
    kv_cache_shape: tuple[int, ...],
    kv_cache_dtype: Any,
    owner_ids_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    w_dtype: Any,
    kv_len: int,
    max_compressed_len: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
) -> Any:
    """Compile/load the device-cache indexer score kernel without executing it."""

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    q_T = SimpleNamespace(shape=tuple(int(dim) for dim in q_T_shape), dtype=q_T_dtype)
    kv_cache = SimpleNamespace(
        shape=tuple(int(dim) for dim in kv_cache_shape),
        dtype=kv_cache_dtype,
    )
    owner_ids = SimpleNamespace(
        shape=tuple(int(dim) for dim in owner_ids_shape),
        dtype=np.dtype(np.int32),
    )
    w = SimpleNamespace(shape=tuple(int(dim) for dim in w_shape), dtype=w_dtype)

    q_T_shape_t = tuple(int(dim) for dim in q_T.shape)
    cache_shape = tuple(int(dim) for dim in kv_cache.shape)
    owner_shape = tuple(int(dim) for dim in owner_ids.shape)
    w_shape_t = tuple(int(dim) for dim in w.shape)
    if len(q_T_shape_t) != 3:
        raise ValueError(f"q_T must be [B,d,h], got {q_T_shape_t}")
    B, d, h = q_T_shape_t
    t = int(kv_len)
    max_t = int(max_compressed_len)
    if d != D_BLOCK or h > P_MAX:
        raise ValueError(f"unsupported q_T shape {q_T_shape_t}")
    if len(cache_shape) != 2 or cache_shape[1] != d:
        raise ValueError(f"kv_cache shape {cache_shape} incompatible with d={d}")
    if owner_shape != (B,):
        raise ValueError(f"owner_ids must be [{B}], got {owner_shape}")
    if w_shape_t != (B, h):
        raise ValueError(f"w must be [{B},{h}], got {w_shape_t}")
    if t <= 0 or t > max_t:
        raise ValueError(f"kv_len={t} must be in [1,{max_t}]")

    cache = (
        _INDEXER_SCORE_FROM_CACHE_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = _indexer_score_from_cache_cache_key(
        q_T=q_T,
        kv_cache=kv_cache,
        owner_ids=owner_ids,
        w=w,
        kv_len=t,
        max_compressed_len=max_t,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = compile_and_load_with_lock(
            _device_kernel_cls,
            _make_indexer_score_from_cache_entry(kv_len=t, max_compressed_len=max_t),
            _sample_like(q_T),
            _sample_like(kv_cache),
            _sample_like(owner_ids),
            _sample_like(w),
            name=f"dsv4_indexer_score_from_cache_b{B}_h{h}_t{t}",
            build_dir=artifacts_dir,
            namespace="dsv4_indexer_kernels",
        )
        cache[cache_key] = kernel
    return kernel


def run_indexer_score_from_cache_device(
    *,
    q_T: Any,
    kv_cache: Any,
    owner_ids: Any,
    w: Any,
    kv_len: int,
    max_compressed_len: int,
    artifacts_dir: str | Path | None = None,
    _device_kernel_cls: Any | None = None,
    _kernel_cache: dict[tuple, Any] | None = None,
    output: Any | None = None,
) -> Any:
    """Device indexer score that reads persistent compressed KV cache.

    ``q_T`` is expected as ``[B, d, h] bf16`` (``d``-on-partition). ``w`` is
    ``[B, h] fp32``. Both are typically the output of the
    ``indexer_score_qw_prep`` trace function, so they stay on device — no
    host round-trip is needed to call this kernel. ``owner_ids`` is
    ``[B]`` int32.
    """

    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    q_T_shape = tuple(int(dim) for dim in getattr(q_T, "shape"))
    cache_shape = tuple(int(dim) for dim in getattr(kv_cache, "shape"))
    owner_shape = tuple(int(dim) for dim in getattr(owner_ids, "shape"))
    w_shape = tuple(int(dim) for dim in getattr(w, "shape"))
    if len(q_T_shape) != 3:
        raise ValueError(f"q_T must be [B,d,h], got {q_T_shape}")
    B, d, h = q_T_shape
    t = int(kv_len)
    max_t = int(max_compressed_len)
    if d != D_BLOCK or h > P_MAX:
        raise ValueError(f"unsupported q_T shape {q_T_shape}")
    if len(cache_shape) != 2 or cache_shape[1] != d:
        raise ValueError(f"kv_cache shape {cache_shape} incompatible with d={d}")
    if owner_shape != (B,):
        raise ValueError(f"owner_ids must be [{B}], got {owner_shape}")
    if w_shape != (B, h):
        raise ValueError(f"w must be [{B},{h}], got {w_shape}")
    if t <= 0 or t > max_t:
        raise ValueError(f"kv_len={t} must be in [1,{max_t}]")

    DeviceTensor = get_device_tensor_cls()

    owner_ids_arg = owner_ids
    if not hasattr(owner_ids_arg, "tensor_ref"):
        owner_ids_arg = DeviceTensor.from_numpy(
            np.ascontiguousarray(np.asarray(owner_ids_arg, dtype=np.int32)),
            name="dsv4_indexer_owner_ids",
        )

    cache = (
        _INDEXER_SCORE_FROM_CACHE_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = _indexer_score_from_cache_cache_key(
        q_T=q_T,
        kv_cache=kv_cache,
        owner_ids=owner_ids_arg,
        w=w,
        kv_len=t,
        max_compressed_len=max_t,
    )
    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        kernel = compile_and_load_with_lock(
            _device_kernel_cls,
            _make_indexer_score_from_cache_entry(kv_len=t, max_compressed_len=max_t),
            _sample_like(q_T),
            _sample_like(kv_cache),
            _sample_like(owner_ids_arg),
            _sample_like(w),
            name=f"dsv4_indexer_score_from_cache_b{B}_h{h}_t{t}",
            build_dir=artifacts_dir,
            namespace="dsv4_indexer_kernels",
        )
        cache[cache_key] = kernel

    out_dev = output
    if out_dev is None:
        out_dev = DeviceTensor.from_numpy(
            np.zeros((B, t), dtype=np.float32),
            name="dsv4_indexer_score_from_cache_out",
        )
    kernel(
        inputs={
            "q_T": q_T,
            "kv_cache": kv_cache,
            "owner_ids": owner_ids_arg,
            "w": w,
        },
        outputs={"output0": out_dev},
    )
    return out_dev


# ---------------------------------------------------------------------------
# Stage 3 state-bridging adapters (moved from sampled_forward.py).
# ---------------------------------------------------------------------------


def indexer_score_kernel_adapter(
    q: np.ndarray,
    kv_block: np.ndarray,
    weights: np.ndarray,
    *,
    build_dir: str | Path | None = None,
) -> np.ndarray:
    """Score indexer queries against an ephemeral compressed-KV block.

    Dispatches to the device-path ``indexer_score``. ``head_dim`` must match
    ``D_BLOCK``; callers that need a CPU reference must call the oracle
    explicitly.
    Shapes: ``q [bsz, seqlen, n_heads, head_dim]``,
    ``kv_block [bsz, kv_len, head_dim]``,
    ``weights [bsz, seqlen, n_heads]``. Returns ``[bsz, seqlen, kv_len]``.
    """
    from nkipy_serving.ops.attention.indexer import (
        D_BLOCK,
        indexer_score,
    )

    bsz, seqlen, n_heads, head_dim = q.shape
    n_tokens = bsz * seqlen
    q_flat = q.reshape(n_tokens, n_heads, head_dim)
    w_flat = weights.reshape(n_tokens, n_heads)
    kv_len = kv_block.shape[1]
    kv_flat = np.broadcast_to(
        kv_block[:, None, :, :],
        (bsz, seqlen, kv_len, head_dim),
    ).reshape(n_tokens, kv_len, head_dim)
    kv_flat = np.ascontiguousarray(kv_flat)
    if head_dim != D_BLOCK:
        raise ValueError(
            "indexer_score_kernel_adapter requires "
            f"head_dim == {D_BLOCK}; got head_dim={head_dim}"
        )
    out = indexer_score(
        q_flat,
        kv_flat,
        w_flat,
        use_device=True,
        artifacts_dir=build_dir,
    )
    return out.reshape(bsz, seqlen, kv_len)


def indexer_score_from_device_cache_adapter(
    q_T: Any,
    w: Any,
    *,
    device_state: Any,
    bsz: int,
    seqlen: int,
    kv_len: int,
    build_dir: str | Path | None = None,
    return_device: bool = False,
    owner_ids: np.ndarray | None = None,
    owner_ids_dev: Any | None = None,
    output: Any | None = None,
) -> Any:
    """Score indexer queries against the persistent device compressed-KV cache.

    Expects ``q_T`` and ``w`` as either DeviceTensors (typical production
    path — produced by the ``indexer_score_qw_prep`` trace function) or
    numpy arrays (debug/test path). Shapes: ``q_T [B=bsz*seqlen, d, h] bf16``,
    ``w [B, h] fp32``.
    """
    n = int(bsz) * int(seqlen)
    if owner_ids is None:
        owner_ids_arr = np.repeat(np.arange(int(bsz), dtype=np.int32), int(seqlen))
    else:
        owner_ids_arr = np.asarray(owner_ids, dtype=np.int32).reshape(-1)
        if owner_ids_arr.shape != (n,):
            raise ValueError(f"owner_ids must be [{n}], got {owner_ids_arr.shape}")
    if hasattr(device_state.compressed_kv_cache, "tensor_ref"):
        owner_ids_arg: Any
        if owner_ids_dev is None:
            owner_ids_arg = np.ascontiguousarray(owner_ids_arr)
        else:
            owner_dev_shape = tuple(int(dim) for dim in getattr(owner_ids_dev, "shape"))
            if owner_dev_shape != (n,):
                raise ValueError(f"owner_ids_dev must be [{n}], got {owner_dev_shape}")
            owner_ids_arg = owner_ids_dev
        out_dev = run_indexer_score_from_cache_device(
            q_T=q_T,
            kv_cache=device_state.compressed_kv_cache,
            owner_ids=owner_ids_arg,
            w=w,
            kv_len=int(kv_len),
            max_compressed_len=int(device_state.spec.max_compressed_len),
            artifacts_dir=build_dir,
            output=output,
        )
        if return_device:
            return out_dev
        out = out_dev.numpy()
    else:
        if return_device:
            raise ValueError("return_device=True requires device compressed KV cache")
        # Host oracle reconstructs q [B,h,d] fp32 from q_T [B,d,h] bf16.
        q_T_np = q_T.numpy() if hasattr(q_T, "numpy") else np.asarray(q_T)
        w_np = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
        q_flat = np.ascontiguousarray(q_T_np.astype(np.float32).transpose(0, 2, 1))
        w_flat = np.ascontiguousarray(w_np.astype(np.float32))
        out = indexer_score_from_cache_oracle(
            q_flat,
            device_state.compressed_kv_cache,
            owner_ids_arr,
            w_flat,
            kv_len=int(kv_len),
            max_compressed_len=int(device_state.spec.max_compressed_len),
        )
    return np.asarray(out, dtype=np.float32).reshape(bsz, seqlen, kv_len)


__all__ = [
    "indexer_score_from_cache_oracle",
    "indexer_score_from_device_cache_adapter",
    "indexer_score_kernel_adapter",
    "precompile_indexer_score_from_cache_device",
    "run_indexer_score_from_cache_device",
]
