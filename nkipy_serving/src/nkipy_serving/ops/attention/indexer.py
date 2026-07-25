"""Device-side indexer score kernel.

DeepSeek-V4's sparse indexer picks the top-k compressed-KV positions for
each query token by scoring every compressed position against the query's
per-head Q vectors, reducing across heads, then taking top-k over the t
(compressed-time) axis. Reference pseudocode:

    idx[b, s, h, t] = q[b, s, h, :]  @ kv[b, t, :]    # per-head dot product
    score[b, s, t]  = sum_h max(idx[b, s, h, t], 0) * w[b, s, h]
    # optional causal mask, then top-k over t.

This module moves the **score** computation to an NKI kernel. Top-k stays
on host because it lives in the outer Python code that reshapes, slices, and
applies the causal mask.

Kernel inputs (partition-axis-first layout):

- ``q_T``    [B, d, h]   bf16  — Q tensor transposed so ``d`` is on
  the partition axis. B = flattened (b*s) query slots.
- ``kv_T``   [B, d, t]   bf16  — compressed KV transposed the same way.
- ``w``      [B, h]      fp32  — per-head weight.
- Output: ``score [B, t]`` fp32.

V4 shape: h=64, d=128 (index_head_dim), t up to a few thousand. Because
``d=128`` fits a single Trn2 partition, the qk matmul needs no d-block
tiling. Reducing the h axis (which lives on
the qk output's partition dim) uses an ``nc_matmul`` against an all-ones
vector, the same trick the sparse attention kernel uses for sink
transposition.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ml_dtypes
import numpy as np

from nkipy_serving.ops.attention.sparse_mla import _compile_once, _run_cached
from nkipy_serving.runtime.nki_bridge import ensure_nki_bridge

P_MAX = 128
D_BLOCK = 128  # index_head_dim; kernel requires d == D_BLOCK.


def _as_nki_bf16(a: np.ndarray) -> np.ndarray:
    if a.dtype == ml_dtypes.bfloat16:
        return np.ascontiguousarray(a)
    return np.ascontiguousarray(a.astype(ml_dtypes.bfloat16))


# ---------------------------------------------------------------------------
# CPU oracle (matches eager Indexer's score computation bit-close)
# ---------------------------------------------------------------------------


def indexer_score_oracle(
    q: np.ndarray,  # [B, h, d]   fp32 (or bf16)
    kv: np.ndarray,  # [B, t, d]   fp32 (or bf16)
    w: np.ndarray,  # [B, h]      fp32
) -> np.ndarray:
    """Reference score without the causal mask.

    Returns ``[B, t]`` fp32. The caller adds the causal mask and top-k
    on host (same as eager).
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [B, h, d], got {q.shape}")
    if kv.shape[0] != q.shape[0]:
        raise ValueError(f"q/kv B mismatch: q={q.shape[0]} kv={kv.shape[0]}")
    if w.shape != (q.shape[0], q.shape[1]):
        raise ValueError(f"w must be [B={q.shape[0]}, h={q.shape[1]}], got {w.shape}")
    qf = q.astype(np.float32)
    kvf = kv.astype(np.float32)
    idx = np.einsum("bhd,btd->bht", qf, kvf)  # [B, h, t]
    idx = np.maximum(idx, 0.0) * w[..., None]  # [B, h, t]
    return idx.sum(axis=1)  # [B, t]


# ---------------------------------------------------------------------------
# NKI kernel (module-level @nki.jit; see sparse_mla.py for why)
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
    def _indexer_score_batched_kernel(q_T, kv_T, w):
        """Batched indexer score.

        Inputs:
        - ``q_T`` : [B, d, h]  bf16. d on partition (=128, no tiling).
        - ``kv_T``: [B, d, t]  bf16. d on partition.
        - ``w``   : [B, h]     fp32.

        Produces:
        - ``score`` : [B, t]   fp32 = sum_h max(q·kv^T, 0) * w[h].

        Per-query steps:
        1) qk[h, t] = nc_matmul(q, kv)       — contract d on partition.
        2) relu(qk) then tensor_scalar multiply by w[h] (broadcast across t).
        3) score[t] = sum_h scored[h, t] via nc_matmul(ones_h, scored),
           contracting h on partition.
        """
        B = q_T.shape[0]
        d = q_T.shape[1]
        h = q_T.shape[2]
        t = kv_T.shape[2]
        if d != D_BLOCK:
            raise ValueError(f"d={d} must equal D_BLOCK={D_BLOCK}")
        if h > P_MAX:
            raise ValueError(f"h={h} must fit partition size {P_MAX}")

        # Pre-build the ones[h,1] tensor used to reduce h via nc_matmul.
        ones_h = nl.ndarray((par_dim(h), 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        ones_h[...] = nisa.memset(shape=(h, 1), value=1.0, dtype=nl.bfloat16)

        out = nl.ndarray((B, t), dtype=nl.float32, buffer=nl.shared_hbm)

        for bi in nl.affine_range(B):
            # (1) qk[h, t] = q @ kv  — contract on d (partition, =128).
            q_sb = nl.load(q_T[bi, :, :])  # [d, h]
            kv_sb = nl.load(kv_T[bi, :, :])  # [d, t]
            qk_psum = nl.zeros(
                (par_dim(h), t),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            qk_psum[...] = nisa.nc_matmul(q_sb, kv_sb)

            # (2) ReLU, then multiply row-wise by w[bi, h].
            # Move w[bi, :] onto partition h via nc_transpose of the 1×h slice.
            w_slice = nl.load(w[bi : bi + 1, :])  # [1, h]
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

            # ReLU + multiply: scored[h, t] = max(qk, 0) * w[h].
            relu_qk = nl.ndarray(
                (par_dim(h), t),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            relu_qk[...] = nl.maximum(qk_psum, nl.float32(0.0))

            scored = nl.ndarray(
                (par_dim(h), t),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            scored[...] = nisa.tensor_scalar(
                data=relu_qk,
                op0=nl.multiply,
                operand0=w_on_part,
                dtype=nl.float32,
            )

            # (3) Sum over h -> score[1, t]. Use nc_matmul(ones_h, scored)
            # which contracts on partition (h).
            scored_bf = nl.ndarray(
                (par_dim(h), t),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            scored_bf[...] = nl.copy(scored, dtype=nl.bfloat16)
            score_psum = nl.zeros(
                (par_dim(1), t),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            score_psum[...] = nisa.nc_matmul(ones_h, scored_bf)

            nl.store(out[bi : bi + 1, :], score_psum)

        return out


@lru_cache(maxsize=1)
def _traced_indexer_score_batched():
    if not _NKI_AVAILABLE:
        raise RuntimeError("nkipy not available")
    ensure_nki_bridge()
    from nkipy.core.trace import NKIPyKernel

    def _entry(q_T, kv_T, w):
        return _indexer_score_batched_kernel(q_T, kv_T, w)

    return NKIPyKernel.trace(_entry, backend="hlo")


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------


def indexer_score(
    q: np.ndarray,  # [B, h, d]
    kv: np.ndarray,  # [B, t, d]
    w: np.ndarray,  # [B, h]
    *,
    use_device: bool = True,
    artifacts_dir: str | Path | None = None,
) -> np.ndarray:
    """Compute ``score[B, t] = sum_h max(q·kv^T, 0) * w[h]``.

    ``use_device=False`` runs the numpy oracle.

    Kernel constraints:
    - ``d == 128`` (V4 index_head_dim).
    - ``h <= 128``.
    - ``t`` can be arbitrary; Trn2 SBUF sizes t on the free dimension.

    Outside those constraints the caller must fall back to ``indexer_score_oracle``
    explicitly — we don't silently fallback here because the kernel is
    shape-parametrised per NEFF cache entry.

    Returns ``score [B, t]`` fp32.
    """
    if q.ndim != 3 or kv.ndim != 3 or w.ndim != 2:
        raise ValueError(f"bad input shapes: q={q.shape} kv={kv.shape} w={w.shape}")
    B, h, d = q.shape
    B_kv, t, d_kv = kv.shape
    if B != B_kv or d != d_kv:
        raise ValueError(f"shape mismatch: q={q.shape}, kv={kv.shape}")
    if w.shape != (B, h):
        raise ValueError(f"w shape mismatch: w={w.shape}, expected ({B},{h})")

    if not use_device:
        return indexer_score_oracle(q, kv, w)

    if d != D_BLOCK or h > P_MAX:
        raise ValueError(
            "indexer_score device kernel requires "
            f"d == {D_BLOCK} and h <= {P_MAX}; got d={d}, h={h}. "
            "Call indexer_score_oracle or pass use_device=False for CPU reference."
        )

    q_bf = _as_nki_bf16(q)
    kv_bf = _as_nki_bf16(kv)
    # Transpose for partition-on-d layout.
    q_T = np.ascontiguousarray(q_bf.transpose(0, 2, 1))  # [B, d, h]
    kv_T = np.ascontiguousarray(kv_bf.transpose(0, 2, 1))  # [B, d, t]
    w_f = np.ascontiguousarray(w.astype(np.float32))  # [B, h]

    traced = _traced_indexer_score_batched()
    art = str(artifacts_dir) if artifacts_dir is not None else None
    _compile_once(traced, q_T, kv_T, w_f, artifacts_dir=art)
    return np.asarray(_run_cached(traced, q_T, kv_T, w_f)).astype(np.float32)
