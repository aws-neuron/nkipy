"""NKI FP8 matmul kernels (single expert).

Current variants:

1. ``fp8_bf16_matmul_unscaled`` — stable Trn2 correctness path for
   FP8-weight × BF16-activation matmul. It handles one 128-wide K block;
   callers apply UE8M0 weight scales and reduce across K blocks on the host.

2. ``fp8_block_matmul`` — experimental fused scale/reduce path. Kept for
   Phase-H iteration; multi-K reductions currently hit a Trn2 compiler/layout
   instability.

3. ``fp8_matmul_unscaled`` — FP8×FP8 diagnostic kernel that emits per-K-block
   partial sums. Kept for isolating the native FP8 path.

Uses the ``neuronxcc.nki`` frontend because this path depends on current Trn2
``nisa.nc_matmul`` behavior. The matmul requires ``float8_e4m3`` (non-FN dtype
tag); callers should ``.view`` their ``float8_e4m3fn`` arrays into the short
``float8_e4m3`` tag before passing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ml_dtypes
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.language import par_dim

from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

P_MAX = 128  # partition tile (K-block, aligned with UE8M0 128-wide blocks).
M_TILE = 128  # stationary free-axis tile (= weight-scale M block).
N_TILE_MAX = 512  # moving free-axis max tile; clamped to N at trace time.


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _scale_fp8_partial_mn(
    partial_mn: np.ndarray,
    scale_w: np.ndarray,
    kb: int,
    *,
    block_m: int = M_TILE,
) -> np.ndarray:
    """Apply UE8M0/fp32 weight scales to one unscaled ``[M, N]`` partial."""
    out = np.asarray(partial_mn, dtype=np.float32).copy()
    scales = np.asarray(scale_w, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"partial_mn must be 2D, got shape={out.shape}")
    if scales.ndim != 2:
        raise ValueError(f"scale_w must be 2D, got shape={scales.shape}")
    if kb < 0 or kb >= scales.shape[1]:
        raise ValueError(f"kb={kb} out of range for scale_w shape={scales.shape}")
    expected_rows = _div_ceil(out.shape[0], block_m)
    if scales.shape[0] < expected_rows:
        raise ValueError(
            f"scale_w has {scales.shape[0]} M blocks, need {expected_rows}"
        )
    for mi in range(expected_rows):
        m0 = mi * block_m
        m1 = min(m0 + block_m, out.shape[0])
        out[m0:m1] *= scales[mi, kb]
    return out


def _require_host_reduce_shapes(
    x_bf16: np.ndarray,
    w_fp8: np.ndarray,
    scale_w: np.ndarray,
    *,
    block_size: int = P_MAX,
) -> tuple[int, int, int]:
    if x_bf16.ndim != 2:
        raise ValueError(f"x_bf16 must be [N, K], got shape={x_bf16.shape}")
    if w_fp8.ndim != 2:
        raise ValueError(f"w_fp8 must be [M, K], got shape={w_fp8.shape}")
    if scale_w.ndim != 2:
        raise ValueError(f"scale_w must be [M/128, K/128], got shape={scale_w.shape}")
    n_tokens, k = x_bf16.shape
    m, wk = w_fp8.shape
    if wk != k:
        raise ValueError(f"x/w K mismatch: x K={k}, w K={wk}")
    if k % block_size:
        raise ValueError(f"K={k} must be a multiple of {block_size}")
    if m % M_TILE:
        raise ValueError(f"M={m} must be a multiple of {M_TILE}")
    expected_scale = (m // M_TILE, k // block_size)
    if tuple(scale_w.shape) != expected_scale:
        raise ValueError(
            f"scale_w shape={scale_w.shape} must equal {expected_scale} "
            "for [M, K] FP8 weights"
        )
    if w_fp8.dtype not in (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3):
        raise TypeError(
            "w_fp8 must have dtype ml_dtypes.float8_e4m3fn or float8_e4m3, "
            f"got {w_fp8.dtype}"
        )
    return int(n_tokens), int(k), int(m)


def _as_nki_e4m3(w_T: np.ndarray) -> np.ndarray:
    if w_T.dtype == ml_dtypes.float8_e4m3:
        return w_T
    if w_T.dtype == ml_dtypes.float8_e4m3fn:
        return w_T.view(ml_dtypes.float8_e4m3)
    raise TypeError(f"unsupported FP8 dtype for NKI matmul: {w_T.dtype}")


@lru_cache(maxsize=1)
def _traced_fp8_bf16_unscaled():
    ensure_nki_bridge()
    from nkipy.core.trace import NKIPyKernel

    def _kernel(a, b):
        return fp8_bf16_matmul_unscaled(a, b)

    return NKIPyKernel.trace(_kernel, backend="hlo")


def fp8_bf16_block_matmul_host_reduce(
    x_bf16: np.ndarray,
    w_fp8: np.ndarray,
    scale_w: np.ndarray,
    *,
    artifacts_dir: str | Path | None = None,
    warmup: bool = True,
) -> np.ndarray:
    """Stable D2b runtime wrapper: NKI one-K-block matmul + host reduction.

    Args:
        x_bf16: Activation matrix ``[N, K]``. Values should already be BF16
            or small enough for the mixed FP8×BF16 path.
        w_fp8: FP8 E4M3/FN weight matrix ``[M, K]``.
        scale_w: UE8M0 or fp32 weight scales ``[M/128, K/128]``.
        artifacts_dir: Optional compile/run artifact root. Each K block gets
            a subdirectory to avoid trace-output collisions.
        warmup: Run once before collecting output. Kept on by default because
            device tests use this path immediately after tracing.

    Returns:
        fp32 matrix ``[N, M]`` matching ``x @ dequant(w).T``.
    """
    n_tokens, k, m = _require_host_reduce_shapes(x_bf16, w_fp8, scale_w)
    from nkipy.runtime.execute import baremetal_run_traced_kernel

    traced = _traced_fp8_bf16_unscaled()
    out_mn = np.zeros((m, n_tokens), dtype=np.float32)
    root = Path(artifacts_dir) if artifacts_dir is not None else None
    scales = np.asarray(scale_w, dtype=np.float32)

    for kb, k0 in enumerate(range(0, k, P_MAX)):
        x_T = np.ascontiguousarray(x_bf16[:, k0 : k0 + P_MAX].T)
        w_T = _as_nki_e4m3(np.ascontiguousarray(w_fp8[:, k0 : k0 + P_MAX].T))
        kb_artifacts = None
        if root is not None:
            kb_artifacts = root / f"kb{kb:04d}"
            kb_artifacts.mkdir(parents=True, exist_ok=True)
            kb_artifacts = str(kb_artifacts)
        if warmup:
            baremetal_run_traced_kernel(traced, x_T, w_T, artifacts_dir=kb_artifacts)
        partial = baremetal_run_traced_kernel(
            traced, x_T, w_T, artifacts_dir=kb_artifacts
        )
        out_mn += _scale_fp8_partial_mn(np.asarray(partial), scales, kb)

    return out_mn.T.copy()


@nki.jit
def fp8_block_matmul(
    x_T,  # HBM BF16 tensor, shape [K, N].
    w_T,  # HBM FP8 E4M3 tensor, shape [K, M].
    scale_w,  # HBM fp32 tensor, shape [M // M_TILE, K // P_MAX].
):
    """Mixed FP8 (weight) × BF16 (activation) block matmul.

    Returns fp32 ``[M, N]`` = ``sum_kb (w_fp8[kb] · x_bf16[kb]) * sw[kb]``.

    On Trn2 (NeuronCore-v3), ``nisa.nc_matmul`` with both inputs as FP8
    requires the double-row mode shape (``[128, 2, batch]``, free dim = 2).
    For our arbitrary-free-dim use case we run the matmul in mixed
    ``fp8 × bf16`` mode. The NKI docs guarantee fp32-internal accumulation
    and the mixed dtype path is explicitly supported on v3.

    Activation scales are NOT applied inside the kernel. Callers pre-scale
    the BF16 activations so per-block amax ≈ 1.0 and the kernel sees values
    in the bf16-representable range. This leaves only weight-side scales for
    the kernel to multiply in, which is a single fp32 scalar per (mi, kb)
    and a simple fused multiply-add in the inner loop.
    """
    K, N = x_T.shape
    _, M = w_T.shape
    nk = _div_ceil(K, P_MAX)
    nm = _div_ceil(M, M_TILE)
    n_tile = N if N <= N_TILE_MAX else N_TILE_MAX
    nn = _div_ceil(N, n_tile)

    output = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for mi in nl.affine_range(nm):
        for ni in nl.affine_range(nn):
            acc = nl.zeros(
                (par_dim(M_TILE), n_tile),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for kb in nl.sequential_range(nk):
                x_sb = nl.load(
                    x_T[
                        nl.ds(kb * P_MAX, P_MAX),
                        nl.ds(ni * n_tile, n_tile),
                    ]
                )
                w_sb = nl.load(
                    w_T[
                        nl.ds(kb * P_MAX, P_MAX),
                        nl.ds(mi * M_TILE, M_TILE),
                    ]
                )
                sw_scalar = nl.load(scale_w[mi, kb])
                sw_bcast = nl.broadcast_to(sw_scalar, shape=(P_MAX, 1))
                x_scaled = nl.ndarray(
                    (par_dim(P_MAX), n_tile),
                    dtype=x_T.dtype,
                    buffer=nl.sbuf,
                )
                x_scaled[...] = nisa.tensor_scalar(
                    data=x_sb,
                    op0=nl.multiply,
                    operand0=sw_bcast,
                    dtype=x_T.dtype,
                )
                acc += nisa.nc_matmul(w_sb, x_scaled)

            nl.store(
                output[
                    nl.ds(mi * M_TILE, M_TILE),
                    nl.ds(ni * n_tile, n_tile),
                ],
                value=acc,
            )

    return output


@nki.jit
def fp8_bf16_matmul_unscaled(
    x_T,  # HBM BF16/FP32 tensor, shape [128, N].
    w_T,  # HBM FP8 E4M3 tensor, shape [128, M].
):
    """Single-K-block mixed FP8 × activation matmul without scale apply.

    Returns fp32 ``[M, N]``. The caller applies UE8M0 block scales outside the
    kernel. This is the stable Trn2 correctness path for multi-K reductions:
    run one 128-wide K block at a time, scale each M tile on host, then sum.
    """
    K, N = x_T.shape
    _, M = w_T.shape
    nm = _div_ceil(M, M_TILE)
    n_tile = N if N <= N_TILE_MAX else N_TILE_MAX
    nn = _div_ceil(N, n_tile)

    output = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for mi in nl.affine_range(nm):
        for ni in nl.affine_range(nn):
            x_sb = nl.load(
                x_T[
                    nl.ds(0, P_MAX),
                    nl.ds(ni * n_tile, n_tile),
                ]
            )
            w_sb = nl.load(
                w_T[
                    nl.ds(0, P_MAX),
                    nl.ds(mi * M_TILE, M_TILE),
                ]
            )
            psum = nl.zeros(
                (par_dim(M_TILE), n_tile),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            psum[...] = nisa.nc_matmul(w_sb, x_sb)
            nl.store(
                output[
                    nl.ds(mi * M_TILE, M_TILE),
                    nl.ds(ni * n_tile, n_tile),
                ],
                value=psum,
            )

    return output


@nki.jit
def fp8_matmul_unscaled(
    x_T,  # HBM FP8 E4M3 tensor, shape [K, N].
    w_T,  # HBM FP8 E4M3 tensor, shape [K, M].
):
    """Return fp32 per-K-block partial sums, shape ``[nk, M, N]``.

    ``out[kb, m, n] = sum_{j=0..P_MAX-1} x_fp8[kb * P_MAX + j, n] *
                                         w_fp8[kb * P_MAX + j, m]``

    Output layout is ``[nk, M, N]`` to match the native ``nc_matmul`` result
    ordering (stationary=w → M on partition). The host applies UE8M0 block
    scales to reconstruct the full matmul:

        C[n, m] = sum_kb out[kb, m, n] * scale_x[n, kb] * scale_w[m/128, kb]

    Kept minimal (no scale broadcasts, no fused mul-add) so the FP8 matmul
    path is validated in isolation. Phase H may fuse the scale apply.
    """
    K, N = x_T.shape
    _, M = w_T.shape
    nk = _div_ceil(K, P_MAX)
    nm = _div_ceil(M, M_TILE)
    n_tile = N if N <= N_TILE_MAX else N_TILE_MAX
    nn = _div_ceil(N, n_tile)

    output = nl.ndarray((nk, M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for kb in nl.affine_range(nk):
        for mi in nl.affine_range(nm):
            for ni in nl.affine_range(nn):
                x_sb = nl.load(
                    x_T[
                        nl.ds(kb * P_MAX, P_MAX),
                        nl.ds(ni * n_tile, n_tile),
                    ]
                )
                w_sb = nl.load(
                    w_T[
                        nl.ds(kb * P_MAX, P_MAX),
                        nl.ds(mi * M_TILE, M_TILE),
                    ]
                )
                # nc_matmul: stationary=w_sb, moving=x_sb → psum [M_TILE, n_tile]
                # with M on the partition axis.
                psum = nl.zeros(
                    (par_dim(M_TILE), n_tile),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                psum[...] = nisa.nc_matmul(w_sb, x_sb)
                nl.store(
                    output[
                        kb,
                        nl.ds(mi * M_TILE, M_TILE),
                        nl.ds(ni * n_tile, n_tile),
                    ],
                    value=psum,
                )

    return output
