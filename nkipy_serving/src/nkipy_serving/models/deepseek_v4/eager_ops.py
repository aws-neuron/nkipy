"""V4-specific numpy reference primitives.

Each function mirrors a HF `inference/model.py` or `inference/kernel.py`
construct bit-exactly (within fp32 rounding) so we can diff logits
against the torch reference. Trivial ops (RMSNorm, SiLU) live in
`nkipy_serving/ops/nn.py`; this file owns the V4-specific CPU arithmetic.
"""

from __future__ import annotations

import math

import numpy as np

# -- YaRN RoPE --------------------------------------------------------------
#
# Follows `precompute_freqs_cis` in HF inference/model.py. The frequency
# schedule is standard RoPE (base = rope_theta) with YaRN interpolation
# applied when `original_seq_len > 0`. Output is stored as a complex fp32
# tensor of shape [seqlen, head_dim // 2] so `apply_rotary_emb` matches the
# reference `view_as_complex` semantics.


def _yarn_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_seq_len: int,
) -> tuple[int, int]:
    def correction_dim(num_rot: float) -> float:
        return (
            dim * math.log(max_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
        )

    low = max(int(math.floor(correction_dim(low_rot))), 0)
    high = min(int(math.ceil(correction_dim(high_rot))), dim - 1)
    return low, high


def _linear_ramp(low: int, high: int, dim: int) -> np.ndarray:
    if low == high:
        high = low + 1
    r = (np.arange(dim, dtype=np.float32) - low) / (high - low)
    return np.clip(r, 0.0, 1.0)


def precompute_freqs_cis_yarn(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> np.ndarray:
    """Matches HF `precompute_freqs_cis` exactly. Returns complex64 `[seqlen, dim//2]`.

    When `original_seq_len == 0`, pure RoPE (no YaRN interpolation). When
    positive, apply the YaRN schedule with `factor`, `beta_fast`, `beta_slow`.
    """
    half = dim // 2
    freqs = 1.0 / (np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:
        low, high = _yarn_correction_range(
            float(beta_fast), float(beta_slow), dim, float(base), original_seq_len
        )
        smooth = 1.0 - _linear_ramp(low, high, half)
        freqs = freqs / np.float32(factor) * (1.0 - smooth) + freqs * smooth
    t = np.arange(seqlen, dtype=np.float32)
    theta = np.outer(t, freqs)  # [seqlen, half]
    return (np.cos(theta) + 1j * np.sin(theta)).astype(np.complex64)


def apply_rotary_emb(
    x: np.ndarray,
    freqs_cis: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    """Interleaved RoPE. `x` is `[..., head_dim]`; rotates in pairs.

    Matches HF `apply_rotary_emb`: pair is `(x[..., 2k], x[..., 2k+1])` →
    complex `x[..., 2k] + j * x[..., 2k+1]`, then multiply by `freqs_cis`
    (conjugated if `inverse=True`). Returns fp32 same shape as `x`.
    """
    original_dtype = x.dtype
    xf = x.astype(np.float32)
    head_dim = xf.shape[-1]
    half = head_dim // 2
    # reinterpret [..., head_dim] as complex [..., half]
    cx = xf.reshape(*xf.shape[:-1], half, 2)
    cx = cx[..., 0] + 1j * cx[..., 1]
    fc = freqs_cis
    if inverse:
        fc = np.conj(fc)
    # Broadcast freqs_cis: HF code views it as [1, seqlen, 1, half] for 4D q/k,
    # and [1, seqlen, half] for 3D. We do the same by inserting singleton axes.
    if cx.ndim == 3:  # [b, s, half]
        fc = fc.reshape(1, fc.shape[0], fc.shape[1])
    elif cx.ndim == 4:  # [b, s, h, half]
        fc = fc.reshape(1, fc.shape[0], 1, fc.shape[1])
    else:
        raise RuntimeError(f"apply_rotary_emb: unexpected x.ndim={cx.ndim}")
    rotated = cx * fc
    # back to real, interleaved
    out = np.empty((*rotated.shape, 2), dtype=np.float32)
    out[..., 0] = rotated.real
    out[..., 1] = rotated.imag
    out = out.reshape(*xf.shape[:-1], head_dim)
    return out.astype(original_dtype)


# -- Activations ------------------------------------------------------------


def sqrtsoftplus(x: np.ndarray) -> np.ndarray:
    """softplus(x).sqrt() in fp32, result cast back to input dtype.

    Matches HF `Gate.forward` scoring path for `score_func="sqrtsoftplus"`.
    softplus(x) = log(1 + exp(x)); we use np.logaddexp(0, x) for numerical
    stability across the full fp32 range.
    """
    original_dtype = x.dtype
    xf = x.astype(np.float32)
    sp = np.logaddexp(np.float32(0.0), xf)
    out = np.sqrt(sp)
    return out.astype(original_dtype)


def swiglu_with_limit(
    gate: np.ndarray,
    up: np.ndarray,
    limit: float,
) -> np.ndarray:
    """SwiGLU with optional clamping on both pre-activation legs.

    From HF Expert.forward:
        gate = w1(x).float()
        up   = w3(x).float()
        if limit > 0:
            up   = clamp(up, min=-limit, max=limit)
            gate = clamp(gate, max=limit)
        y = silu(gate) * up
    Inputs assumed fp32; returns fp32.
    """
    gate = gate.astype(np.float32)
    up = up.astype(np.float32)
    if limit > 0:
        up = np.clip(up, -limit, limit)
        gate = np.minimum(gate, np.float32(limit))
    # silu(x) = x * sigmoid(x)
    silu_gate = gate / (np.float32(1.0) + np.exp(-gate))
    return silu_gate * up


# -- Hadamard --------------------------------------------------------------


def hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Randomized Walsh-Hadamard transform along the last dim, normalized by 1/sqrt(d).

    Matches `fast_hadamard_transform.hadamard_transform(x, scale=d**-0.5)`
    (used in HF `rotate_activation`). `d` must be a power of 2.
    """
    original_dtype = x.dtype
    xf = x.astype(np.float32)
    d = xf.shape[-1]
    if d & (d - 1):
        raise RuntimeError(f"Hadamard last-dim must be power of 2, got {d}")
    out = xf.copy()
    step = 1
    while step < d:
        # butterflies of size 2*step
        shape = out.shape
        out = out.reshape(-1, d // (2 * step), 2, step)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = np.stack((a + b, a - b), axis=-2).reshape(shape)
        step *= 2
    out = out * np.float32(d**-0.5)
    return out.astype(original_dtype)


# Host FP8 quant→dequant lives in the ops layer (also used by ops/ kernels).
from nkipy_serving.ops.deepseek_v4.fp8_quant import (  # noqa: F401,E402
    fp8_act_quant_inplace,
)

# -- 4×4 Sinkhorn + HC split --------------------------------------------------


def hc_split_sinkhorn(
    mixes: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matches HF `hc_split_sinkhorn_kernel`. Inputs fp32.

    `mixes` is `[N, (2 + hc_mult) * hc_mult]`, i.e. pre (hc), post (hc),
    comb (hc*hc). Returns `(pre, post, comb)` shaped `[N, hc]`, `[N, hc]`,
    `[N, hc, hc]` respectively.
    """
    if mixes.ndim != 2:
        raise RuntimeError(f"mixes must be 2D, got {mixes.shape}")
    m = mixes.astype(np.float32)
    hc = hc_mult
    scale = hc_scale.astype(np.float32)
    base = hc_base.astype(np.float32)

    pre_raw = m[:, :hc] * scale[0] + base[:hc]
    post_raw = m[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    comb_raw = (
        m[:, 2 * hc : 2 * hc + hc * hc] * scale[2] + base[2 * hc : 2 * hc + hc * hc]
    )
    pre = _sigmoid(pre_raw) + np.float32(eps)
    post = np.float32(2.0) * _sigmoid(post_raw)
    comb = comb_raw.reshape(-1, hc, hc)

    # First iteration: row softmax + eps, then col-normalize.
    row_max = comb.max(axis=2, keepdims=True)
    comb = np.exp(comb - row_max)
    row_sum = comb.sum(axis=2, keepdims=True)
    comb = comb / row_sum + np.float32(eps)
    col_sum = comb.sum(axis=1, keepdims=True)
    comb = comb / (col_sum + np.float32(eps))

    # (sinkhorn_iters - 1) more row/col normalization passes.
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(axis=2, keepdims=True)
        comb = comb / (row_sum + np.float32(eps))
        col_sum = comb.sum(axis=1, keepdims=True)
        comb = comb / (col_sum + np.float32(eps))

    return pre, post, comb


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x))
