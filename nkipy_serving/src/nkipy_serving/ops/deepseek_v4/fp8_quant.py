"""Host-side block-wise FP8 quant→dequant roundtrip (HF act_quant parity)."""

from __future__ import annotations

import ml_dtypes
import numpy as np


def _fast_pow2(exponent: np.ndarray) -> np.ndarray:
    """2**e for integer e as fp32 via bit manipulation (matches `fast_pow2`)."""
    bits = ((exponent.astype(np.int32) + 127) << 23).astype(np.uint32)
    return bits.view(np.float32)


def _fast_log2_ceil(x: np.ndarray) -> np.ndarray:
    """ceil(log2(x)) for fp32 x via exponent extraction (matches `fast_log2_ceil`)."""
    u = x.astype(np.float32).view(np.uint32)
    exp = ((u >> 23) & 0xFF).astype(np.int32) - 127
    mantissa = u & ((1 << 23) - 1)
    return exp + (mantissa != 0).astype(np.int32)


_FP8_MAX = np.float32(448.0)
_FP8_MAX_INV = np.float32(1.0 / 448.0)


def fp8_act_quant_inplace(
    x: np.ndarray,
    block_size: int = 128,
    scale_fmt: str | None = "ue8m0",
) -> np.ndarray:
    """Block-wise FP8 quant → dequant roundtrip along the last dim.

    Mirrors HF `act_quant(..., inplace=True)`. Returns array in the same
    dtype as `x` with FP8 rounding applied. `scale_fmt="ue8m0"` rounds the
    per-block scale up to a power of two (E8M0), else uses the raw fp32 scale.
    """
    original_dtype = x.dtype
    xf = x.astype(np.float32)
    n = xf.shape[-1]
    if n % block_size != 0:
        raise RuntimeError(f"Last dim {n} must be divisible by block_size={block_size}")
    nblocks = n // block_size
    flat = xf.reshape(-1, nblocks, block_size)
    amax = np.maximum(np.abs(flat).max(axis=-1), np.float32(1e-4))
    if scale_fmt == "ue8m0":
        scale = _fast_pow2(_fast_log2_ceil(amax * _FP8_MAX_INV))
    else:
        scale = amax * _FP8_MAX_INV
    # quant: clip(x / s, -448, 448) then cast to fp8
    scaled = flat / scale[..., None]
    clipped = np.clip(scaled, -_FP8_MAX, _FP8_MAX)
    fp8 = clipped.astype(ml_dtypes.float8_e4m3fn)
    # dequant: fp8.astype(fp32) * scale
    dequant = fp8.astype(np.float32) * scale[..., None]
    dequant = dequant.reshape(xf.shape)
    return dequant.astype(original_dtype)
