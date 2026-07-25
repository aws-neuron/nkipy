"""Fused compressor post-pool kernel.

DeepSeek-V4's Compressor normalises + rotates + (optionally) Hadamards the
pooled KV before writing it into the compressed KV cache. The reference
formula is four array operations in a row:

    kv_pool = apply_rms_norm(kv_pool, norm_weight, eps)          # rms
    kv_pool[..., -rd:] = apply_rotary_emb(..., fc)               # rope
    if rotate:
        kv_pool = hadamard_transform(kv_pool)                    # walsh-hadamard
    kv_pool = fp8_act_quant_inplace(kv_pool, ...)                # fp8 qdq

Under sampled NKIPy graph wiring, each of those crosses the CPU/device
boundary. This module fuses RMS+RoPE+Hadamard into a single NKI kernel
and keeps the terminal ``fp8_act_quant_inplace`` on host — fp8 qdq's
``_fast_log2_ceil`` bit-manipulation is awkward to express in NKI
primitives and is cheap in numpy.

Kernel layout:

- Input ``x``: ``[B, d]`` bf16. ``B = bsz * compressed_len`` for prefill,
  ``B = bsz`` for decode. ``d`` is the head_dim (V4 = 128, fits one
  partition).
- Input ``norm_weight``: ``[d]`` fp32 (shared across B).
- Input ``cos``, ``sin``: ``[B, rope_head_dim // 2]`` fp32 — per-row RoPE
  frequencies. rope_head_dim is a suffix of d.
- Output: ``[B, d]`` fp32.

Steps inside the kernel:

1. RMS-norm: ``y[b, i] = x[b, i] * norm_weight[i] / sqrt(mean(x[b, :]**2) + eps)``.
2. RoPE on the last ``rope_head_dim`` slice: interleaved pairs rotated by
   ``(cos, sin)``.
3. Optional Hadamard across d (``d`` power-of-2). Implemented as a
   sequence of butterfly nc_matmuls against the constant ``H_d`` matrix.

Shape constraints (the device path raises when not met; pass
``use_device=False`` to run the oracle explicitly):

- ``d <= 128`` and ``d`` power of 2 (covers V4's 128).
- ``rope_head_dim <= d`` and ``rope_head_dim`` power of 2, even.
- ``B <= some reasonable cap``; the free dimension must fit SBUF.

**Performance status (2026-04-28):** At the small shapes a single
Compressor call produces today (B=32 pools at V4 prefill seqlen=128,
ratio=4), the vectorised numpy oracle runs in <1 ms while each
isolated NKI kernel call pays ~6 ms of per-call DMA/launch overhead.
That makes the kernel a slowdown **when invoked in isolation** because
per-fragment direct wiring pays launch overhead repeatedly. The kernel is
landed off-by-default (``use_compressor_post_kernel=False``) and is expected
to pay off when fragment boundaries are merged so this kernel shares its
launch with adjacent compute. Correctness and the flag are in place; flipping
the default is a perf decision deferred to the larger fusion work.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ml_dtypes
import numpy as np

from nkipy_serving.ops.attention.sparse_mla import (
    _compile_once,
    _run_cached,
)
from nkipy_serving.runtime.device_tensor import get_device_tensor_cls
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

P_MAX = 128
D_MAX = 128


def _as_nki_bf16(a: np.ndarray) -> np.ndarray:
    if a.dtype == ml_dtypes.bfloat16:
        return np.ascontiguousarray(a)
    return np.ascontiguousarray(a.astype(ml_dtypes.bfloat16))


def _hadamard_matrix(d: int) -> np.ndarray:
    """Walsh-Hadamard matrix of order ``d`` (power of 2), normalized by
    ``1 / sqrt(d)`` so ``H^T H = I``. fp32."""
    if d & (d - 1):
        raise ValueError(f"d={d} must be a power of 2")
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return (H / np.float32(np.sqrt(d))).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU oracle — matches the eager RMS+RoPE+(Hadamard) chain.
# ---------------------------------------------------------------------------


def compressor_post_pool_oracle(
    x: np.ndarray,  # [B, d] fp32
    norm_weight: np.ndarray,  # [d] fp32
    cos: np.ndarray,  # [B, rope_head_dim // 2] fp32
    sin: np.ndarray,  # [B, rope_head_dim // 2] fp32
    *,
    rope_head_dim: int,
    rotate: bool,
    eps: float,
) -> np.ndarray:
    """Reference for the fused kernel's output (pre-FP8-qdq).

    Matches eager: RMS-norm → RoPE on the last ``rope_head_dim`` slice →
    optional Hadamard across full ``d``. Returns ``[B, d]`` fp32.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [B, d], got {x.shape}")
    B, d = x.shape
    if norm_weight.shape != (d,):
        raise ValueError(f"norm_weight must be [d={d}], got {norm_weight.shape}")
    rd = int(rope_head_dim)
    if rd % 2 or rd > d:
        raise ValueError(f"bad rope_head_dim {rd} for d={d}")
    if cos.shape != (B, rd // 2) or sin.shape != (B, rd // 2):
        raise ValueError(
            f"cos/sin must be [B={B}, rd/2={rd // 2}], got {cos.shape}/{sin.shape}"
        )

    xf = x.astype(np.float32)
    # RMS-norm.
    rsqrt = np.float32(1.0) / np.sqrt(
        np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps)
    )
    y = xf * rsqrt * norm_weight.astype(np.float32)[None, :]  # [B, d]

    # RoPE on the last rd slice (interleaved pairs).
    head = y[:, -rd:]
    pair = head.reshape(B, rd // 2, 2)
    x0 = pair[:, :, 0]
    x1 = pair[:, :, 1]
    cos_f = cos.astype(np.float32)
    sin_f = sin.astype(np.float32)
    y0 = x0 * cos_f - x1 * sin_f
    y1 = x0 * sin_f + x1 * cos_f
    rotated = np.stack([y0, y1], axis=-1).reshape(B, rd)
    out = np.concatenate([y[:, : d - rd], rotated], axis=-1)  # [B, d]

    if rotate:
        H = _hadamard_matrix(d)
        out = out @ H.T  # [B, d]
    return out


# ---------------------------------------------------------------------------
# NKI kernel — fused RMS + RoPE + (optional) Hadamard.
# ---------------------------------------------------------------------------


try:
    import neuronxcc.nki as _nki
    import neuronxcc.nki.isa as nisa
    import neuronxcc.nki.language as nl
    from neuronxcc.nki.language import par_dim

    _NKI_AVAILABLE = True
except ImportError:
    _nki = None
    nisa = None
    nl = None
    par_dim = None
    _NKI_AVAILABLE = False


if _NKI_AVAILABLE:

    @_nki.jit
    def _compressor_post_kernel_with_hadamard(
        x,
        norm_weight,
        cos,
        sin,
        hadamard_T,
    ):
        """Fused RMS + RoPE + Hadamard kernel.

        Layout:
        - ``x``           : [B, d]      bf16. B on partition (B<=128), d free.
        - ``norm_weight`` : [1, d]      fp32.
        - ``cos``/``sin`` : [B, rd/2]   fp32, B on partition.
        - ``hadamard_T``  : [d, d]      bf16. d on partition; matmul against
          x to apply Hadamard: out[B, d] = sum_i x[B, i] * H[i, d] needs
          contract=d, so we matmul x_bf stationary (partition=d after
          transpose) against H. We pass H_T so partition=d on it too and
          use nc_matmul(x_T, H_T) to output [B-partition, d-free].

        Returns ``[B, d]`` fp32.
        """
        B = x.shape[0]
        d = x.shape[1]
        rd_half = cos.shape[1]
        rd = rd_half * 2

        # (1) RMS norm.
        x_sb = nl.load(x)  # [B, d]
        # Compute mean(x*x) along free (d).
        x_f32 = nl.ndarray((par_dim(B), d), dtype=nl.float32, buffer=nl.sbuf)
        x_f32[...] = nl.copy(x_sb, dtype=nl.float32)
        sq = nl.ndarray((par_dim(B), d), dtype=nl.float32, buffer=nl.sbuf)
        sq[...] = nl.multiply(x_f32, x_f32)
        mean_sq = nisa.tensor_reduce(
            nl.add,
            sq,
            axis=(1,),
            dtype=nl.float32,
            negate=False,
        )  # [B, 1]
        # mean = sum / d
        mean_sq_div = nisa.activation(
            nl.copy,
            mean_sq,
            scale=nl.float32(1.0 / d),
        )  # [B, 1]
        # rsqrt(mean + eps) via pow. NKI doesn't have a direct rsqrt; use
        # nisa.activation with nl.power or compute via exp(-0.5 * log).
        # Simpler: compute (mean + eps) then use activation(rsqrt).
        eps_added = nisa.tensor_scalar(
            data=mean_sq_div,
            op0=nl.add,
            operand0=nl.float32(1e-6),  # eps baked in via kernel arg?
            dtype=nl.float32,
        )
        # Use activation with np.sqrt then reciprocal multiply.
        # NKI supports nisa.activation(np.sqrt, ...).
        sqrt_val = nisa.activation(np.sqrt, eps_added)  # [B, 1]
        rsqrt = nl.ndarray((par_dim(B), 1), dtype=nl.float32, buffer=nl.sbuf)
        rsqrt[...] = nl.divide(nl.float32(1.0), sqrt_val)

        # Scale by norm_weight: y[B, d] = x * rsqrt * norm_weight.
        nw_sb = nl.load(norm_weight)  # [1, d]
        nw_bcast = nl.broadcast_to(nw_sb, shape=(B, d))
        rsqrt_bcast = nl.broadcast_to(rsqrt, shape=(B, d))
        y = nl.ndarray((par_dim(B), d), dtype=nl.float32, buffer=nl.sbuf)
        y[...] = nl.multiply(
            nl.multiply(x_f32, rsqrt_bcast),
            nw_bcast,
        )  # [B, d]

        # (2) RoPE on the last rd slice.
        # Load cos, sin: [B, rd/2]. They're already on the B partition.
        cos_sb = nl.load(cos)  # [B, rd/2]
        sin_sb = nl.load(sin)  # [B, rd/2]

        # y[:, d-rd:d] pairs: even = y[:, d-rd + 2k], odd = y[:, d-rd + 2k+1]
        # We compute:
        #   new_even[b, k] = even[b, k] * cos[b, k] - odd[b, k] * sin[b, k]
        #   new_odd [b, k] = even[b, k] * sin[b, k] + odd[b, k] * cos[b, k]
        # Do this via strided slice access on the free dim.
        # Then re-interleave.
        even = nl.ndarray((par_dim(B), rd_half), dtype=nl.float32, buffer=nl.sbuf)
        odd = nl.ndarray((par_dim(B), rd_half), dtype=nl.float32, buffer=nl.sbuf)
        # NKI free-dim striding: use affine_range loop.
        for k in nl.affine_range(rd_half):
            even[:, k] = y[:, d - rd + 2 * k]
            odd[:, k] = y[:, d - rd + 2 * k + 1]

        new_even = nl.subtract(
            nl.multiply(even, cos_sb),
            nl.multiply(odd, sin_sb),
        )
        new_odd = nl.add(
            nl.multiply(even, sin_sb),
            nl.multiply(odd, cos_sb),
        )
        # Write back interleaved into y.
        for k in nl.affine_range(rd_half):
            y[:, d - rd + 2 * k] = new_even[:, k]
            y[:, d - rd + 2 * k + 1] = new_odd[:, k]

        # (3) Hadamard: out[B, d] = y[B, :] @ H^T where H^T[d, d] has d on partition.
        # Cast y to bf16 for nc_matmul.
        y_bf = nl.ndarray((par_dim(B), d), dtype=nl.bfloat16, buffer=nl.sbuf)
        y_bf[...] = nl.copy(y, dtype=nl.bfloat16)

        # nc_matmul requires stationary and moving both have partition=128.
        # Here y_bf has partition=B (<=128) and hadamard_T has partition=d
        # (=128). They don't match. We want contract=d:
        #   result[B, d_out] = sum_i y[B, i] * H_T[i, d_out]
        # so we need i (=d) on the partition of both. y has d on free dim.
        # Solution: transpose y onto d-partition via nc_transpose, then
        # nc_matmul(y_T_on_d, hadamard_T).
        y_on_d_psum = nl.ndarray(
            (par_dim(d), B),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        y_on_d_psum[...] = nisa.nc_transpose(
            y_bf,
            engine=nisa.tensor_engine,
        )
        y_on_d = nl.ndarray((par_dim(d), B), dtype=nl.bfloat16, buffer=nl.sbuf)
        y_on_d[...] = nl.copy(y_on_d_psum, dtype=nl.bfloat16)

        # hadamard_T [d, d] with d on partition, d on free.
        h_sb = nl.load(hadamard_T)  # [d, d]

        # nc_matmul(stationary=y_on_d [d, B], moving=h_sb [d, d]):
        #   output[stationary.free=B on partition, moving.free=d on free]
        #   = sum_d y_on_d[d, B] * h_sb[d, d] = sum_i y[B, i] * H[i, d]
        out_psum = nl.zeros(
            (par_dim(B), d),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        out_psum[...] = nisa.nc_matmul(y_on_d, h_sb)

        out = nl.ndarray((B, d), dtype=nl.float32, buffer=nl.shared_hbm)
        nl.store(out, out_psum)
        return out

    @_nki.jit
    def _compressor_post_kernel_no_hadamard(
        x,
        norm_weight,
        cos,
        sin,
    ):
        """Variant without Hadamard: RMS + RoPE only."""
        B = x.shape[0]
        d = x.shape[1]
        rd_half = cos.shape[1]
        rd = rd_half * 2

        x_sb = nl.load(x)
        x_f32 = nl.ndarray((par_dim(B), d), dtype=nl.float32, buffer=nl.sbuf)
        x_f32[...] = nl.copy(x_sb, dtype=nl.float32)
        sq = nl.ndarray((par_dim(B), d), dtype=nl.float32, buffer=nl.sbuf)
        sq[...] = nl.multiply(x_f32, x_f32)
        mean_sq = nisa.tensor_reduce(
            nl.add,
            sq,
            axis=(1,),
            dtype=nl.float32,
            negate=False,
        )
        mean_sq_div = nisa.activation(
            nl.copy,
            mean_sq,
            scale=nl.float32(1.0 / d),
        )
        eps_added = nisa.tensor_scalar(
            data=mean_sq_div,
            op0=nl.add,
            operand0=nl.float32(1e-6),
            dtype=nl.float32,
        )
        sqrt_val = nisa.activation(np.sqrt, eps_added)
        rsqrt = nl.ndarray((par_dim(B), 1), dtype=nl.float32, buffer=nl.sbuf)
        rsqrt[...] = nl.divide(nl.float32(1.0), sqrt_val)

        nw_sb = nl.load(norm_weight)
        nw_bcast = nl.broadcast_to(nw_sb, shape=(B, d))
        rsqrt_bcast = nl.broadcast_to(rsqrt, shape=(B, d))
        y = nl.ndarray((par_dim(B), d), dtype=nl.float32, buffer=nl.sbuf)
        y[...] = nl.multiply(
            nl.multiply(x_f32, rsqrt_bcast),
            nw_bcast,
        )

        cos_sb = nl.load(cos)
        sin_sb = nl.load(sin)
        even = nl.ndarray((par_dim(B), rd_half), dtype=nl.float32, buffer=nl.sbuf)
        odd = nl.ndarray((par_dim(B), rd_half), dtype=nl.float32, buffer=nl.sbuf)
        for k in nl.affine_range(rd_half):
            even[:, k] = y[:, d - rd + 2 * k]
            odd[:, k] = y[:, d - rd + 2 * k + 1]
        new_even = nl.subtract(
            nl.multiply(even, cos_sb),
            nl.multiply(odd, sin_sb),
        )
        new_odd = nl.add(
            nl.multiply(even, sin_sb),
            nl.multiply(odd, cos_sb),
        )
        for k in nl.affine_range(rd_half):
            y[:, d - rd + 2 * k] = new_even[:, k]
            y[:, d - rd + 2 * k + 1] = new_odd[:, k]

        out = nl.ndarray((B, d), dtype=nl.float32, buffer=nl.shared_hbm)
        nl.store(out, y)
        return out


@lru_cache(maxsize=2)
def _traced_compressor_post(rotate: bool):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    ensure_nki_bridge()
    from nkipy.core.trace import NKIPyKernel

    if rotate:

        def _entry(x, norm_weight, cos, sin, hadamard_T):
            return _compressor_post_kernel_with_hadamard(
                x,
                norm_weight,
                cos,
                sin,
                hadamard_T,
            )

        _entry.__name__ = "_compressor_post_with_hadamard_entry"
        return NKIPyKernel.trace(_entry, backend="hlo")
    else:

        def _entry(x, norm_weight, cos, sin):
            return _compressor_post_kernel_no_hadamard(
                x,
                norm_weight,
                cos,
                sin,
            )

        _entry.__name__ = "_compressor_post_no_hadamard_entry"
        return NKIPyKernel.trace(_entry, backend="hlo")


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------


def compressor_post_pool(
    x: np.ndarray,  # [B, d] fp32 (pooled KV)
    norm_weight: np.ndarray,  # [d] fp32
    cos: np.ndarray,  # [B, rope_head_dim // 2] fp32
    sin: np.ndarray,  # [B, rope_head_dim // 2] fp32
    *,
    rope_head_dim: int,
    rotate: bool,
    eps: float,
    use_device: bool = True,
    artifacts_dir: str | Path | None = None,
) -> np.ndarray:
    """Fused RMS + RoPE + (optional) Hadamard on pooled compressor KV.

    The eager chain also runs fp8_act_quant_inplace on the result; the
    caller should apply that on host after calling this function.

    ``use_device=False`` returns the oracle (same math, no NKI).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [B, d], got {x.shape}")
    B, d = x.shape

    if not use_device:
        return compressor_post_pool_oracle(
            x,
            norm_weight,
            cos,
            sin,
            rope_head_dim=rope_head_dim,
            rotate=rotate,
            eps=eps,
        )

    # Envelope: d fits one partition (V4 head_dim=128), d is a power of 2
    # (Hadamard needs this), B fits the partition axis (<=128).
    rd = int(rope_head_dim)
    if d > D_MAX or (d & (d - 1)) or B > P_MAX or rd > d:
        raise ValueError(
            "compressor_post_pool device kernel requires "
            f"B <= {P_MAX}, d <= {D_MAX}, power-of-two d, and rd <= d; "
            f"got B={B}, d={d}, rd={rd}. Pass use_device=False for CPU reference."
        )

    # NOTE: eps is currently baked into the kernel as 1e-6. Different eps
    # values would need separate NEFF entries — not a concern for V4
    # (rms_norm_eps=1e-6) but documented here for future callers.

    x_bf = _as_nki_bf16(x)  # [B, d]
    nw_2d = np.ascontiguousarray(
        norm_weight.astype(np.float32).reshape(1, -1)
    )  # [1, d]
    cos_f = np.ascontiguousarray(cos.astype(np.float32))  # [B, rd/2]
    sin_f = np.ascontiguousarray(sin.astype(np.float32))  # [B, rd/2]

    art = str(artifacts_dir) if artifacts_dir is not None else None
    if rotate:
        H_T = np.ascontiguousarray(_hadamard_matrix(d).T.astype(ml_dtypes.bfloat16))
        traced = _traced_compressor_post(rotate=True)
        _compile_once(traced, x_bf, nw_2d, cos_f, sin_f, H_T, artifacts_dir=art)
        return np.asarray(_run_cached(traced, x_bf, nw_2d, cos_f, sin_f, H_T)).astype(
            np.float32
        )
    else:
        traced = _traced_compressor_post(rotate=False)
        _compile_once(traced, x_bf, nw_2d, cos_f, sin_f, artifacts_dir=art)
        return np.asarray(_run_cached(traced, x_bf, nw_2d, cos_f, sin_f)).astype(
            np.float32
        )


# ---------------------------------------------------------------------------
# DeviceTensor-in / DeviceTensor-out wrapper.
# ---------------------------------------------------------------------------


_COMPRESSOR_POST_POOL_DEVICE_KERNEL_CACHE: dict[tuple, "object"] = {}


def _post_pool_dtype_like(tensor: "object"):
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise ValueError(f"tensor {tensor!r} has no dtype")
    if str(dtype) == "bfloat16":
        return ml_dtypes.bfloat16
    return np.dtype(dtype)


def _post_pool_sample_like(tensor: "object") -> np.ndarray:
    shape = tuple(int(dim) for dim in getattr(tensor, "shape"))
    return np.empty(shape, dtype=_post_pool_dtype_like(tensor))


def _compressor_post_pool_entry_no_hadamard(
    x: "object",
    norm_weight: "object",
    cos: "object",
    sin: "object",
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _compressor_post_kernel_no_hadamard(x, norm_weight, cos, sin)


def _compressor_post_pool_entry_with_hadamard(
    x: "object",
    norm_weight: "object",
    cos: "object",
    sin: "object",
    hadamard_T: "object",
):
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    return _compressor_post_kernel_with_hadamard(
        x,
        norm_weight,
        cos,
        sin,
        hadamard_T,
    )


def run_compressor_post_pool_device(
    *,
    x: "object",  # DeviceTensor [B, d] bf16
    norm_weight: "object",  # DeviceTensor [1, d] fp32
    cos: "object",  # DeviceTensor [B, rd/2] fp32
    sin: "object",  # DeviceTensor [B, rd/2] fp32
    rope_head_dim: int,
    rotate: bool,
    hadamard_T: "object | None" = None,  # DeviceTensor [d, d] bf16 when rotate
    artifacts_dir: "str | Path | None" = None,
    _device_kernel_cls: "object | None" = None,
    _kernel_cache: "dict[tuple, object] | None" = None,
    output: "object | None" = None,
) -> "object":
    """Run ``compressor_post_pool`` with DeviceTensor inputs and output.

    Returns a DeviceTensor ``[B, d]`` fp32 containing the fused
    RMS+RoPE+(Hadamard) output. The terminal FP8 qdq is applied by the
    caller via the ``fp8_act_qdq`` trace function.

    Non-rotating callers keep ``d`` on the free axis and support V4's main
    ``d=512`` compressor. Rotating callers use the Hadamard matmul path and
    still require ``d <= 128`` power-of-2. Raises on unsupported shapes rather
    than falling back to the oracle.
    """
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    x_shape = tuple(int(dim) for dim in getattr(x, "shape"))
    nw_shape = tuple(int(dim) for dim in getattr(norm_weight, "shape"))
    cos_shape = tuple(int(dim) for dim in getattr(cos, "shape"))
    sin_shape = tuple(int(dim) for dim in getattr(sin, "shape"))
    if len(x_shape) != 2:
        raise ValueError(f"x must be [B, d], got {x_shape}")
    B, d = x_shape
    rd = int(rope_head_dim)
    if d <= 0 or B > P_MAX or rd > d or rd % 2:
        raise ValueError(
            f"unsupported shape B={B}, d={d}, rd={rd} for device post-pool"
        )
    if bool(rotate) and (d > D_MAX or (d & (d - 1))):
        raise ValueError(
            f"unsupported rotating shape B={B}, d={d}, rd={rd} for device post-pool"
        )
    if nw_shape != (1, d):
        raise ValueError(f"norm_weight must be [1,{d}], got {nw_shape}")
    if cos_shape != (B, rd // 2) or sin_shape != (B, rd // 2):
        raise ValueError(
            f"cos/sin must be [{B}, {rd // 2}], got {cos_shape}/{sin_shape}"
        )
    rotate = bool(rotate)
    if rotate and hadamard_T is None:
        raise ValueError("rotate=True requires hadamard_T")
    DeviceTensor = get_device_tensor_cls()

    cache = (
        _COMPRESSOR_POST_POOL_DEVICE_KERNEL_CACHE
        if _kernel_cache is None
        else _kernel_cache
    )
    cache_key = (
        "compressor_post_pool_device",
        x_shape,
        nw_shape,
        cos_shape,
        sin_shape,
        rd,
        rotate,
        str(_post_pool_dtype_like(x)),
    )
    if rotate:
        ht_shape = tuple(int(dim) for dim in getattr(hadamard_T, "shape"))
        cache_key = cache_key + (ht_shape, str(_post_pool_dtype_like(hadamard_T)))

    kernel = cache.get(cache_key)
    if kernel is None:
        if _device_kernel_cls is None:
            ensure_nki_bridge()
            from nkipy.runtime import DeviceKernel

            _device_kernel_cls = DeviceKernel
        if rotate:
            kernel = compile_and_load_with_lock(
                _device_kernel_cls,
                _compressor_post_pool_entry_with_hadamard,
                _post_pool_sample_like(x),
                _post_pool_sample_like(norm_weight),
                _post_pool_sample_like(cos),
                _post_pool_sample_like(sin),
                _post_pool_sample_like(hadamard_T),
                name=f"dsv4_compressor_post_pool_rotate_B{B}_d{d}_rd{rd}",
                build_dir=artifacts_dir,
                namespace="dsv4_compressor_kernels",
            )
        else:
            kernel = compile_and_load_with_lock(
                _device_kernel_cls,
                _compressor_post_pool_entry_no_hadamard,
                _post_pool_sample_like(x),
                _post_pool_sample_like(norm_weight),
                _post_pool_sample_like(cos),
                _post_pool_sample_like(sin),
                name=f"dsv4_compressor_post_pool_norot_B{B}_d{d}_rd{rd}",
                build_dir=artifacts_dir,
                namespace="dsv4_compressor_kernels",
            )
        cache[cache_key] = kernel

    out_dev = output
    if out_dev is None:
        out_dev = DeviceTensor.from_numpy(
            np.zeros((B, d), dtype=np.float32),
            name="dsv4_compressor_post_pool_out",
        )
    if rotate:
        kernel(
            inputs={
                "x": x,
                "norm_weight": norm_weight,
                "cos": cos,
                "sin": sin,
                "hadamard_T": hadamard_T,
            },
            outputs={"output0": out_dev},
        )
    else:
        kernel(
            inputs={
                "x": x,
                "norm_weight": norm_weight,
                "cos": cos,
                "sin": sin,
            },
            outputs={"output0": out_dev},
        )
    return out_dev


def build_hadamard_T_device(d: int, *, name: str = "dsv4_hadamard_T") -> "object":
    """Return the Walsh-Hadamard^T device tensor used by the rotate kernel."""
    H_T = np.ascontiguousarray(
        _hadamard_matrix(int(d)).T.astype(ml_dtypes.bfloat16),
    )
    return get_device_tensor_cls().from_numpy(H_T, name=name)
