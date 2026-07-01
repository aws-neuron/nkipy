"""Direct lowering of tensor IR matmul to NKI IR.

Lowers A[..., M, K] @ B[..., K, N] -> C[..., M, N] for any legal shape
(rank >= 2, with numpy-style batch broadcasting). Generates tiled NKI IR
with K-accumulation in PSUM, M-tiling on the output partition axis, and
N-tiling when N exceeds PSUM_FREE_MAX.

This is a standalone lowering pass that takes tensor IR matmul parameters
directly (no fusion plan or layout solver dependency) and produces an
executable NKI IR graph.
"""

from __future__ import annotations

import math

import numpy as np

from nkigen_lite.core import DType, Graph
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
    PSUM_FREE_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import ceildiv


def lower_matmul(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype: DType = DType.F32,
    tile_m: int = 128,
    tile_n: int = 512,
    tile_k: int = 128,
) -> Graph:
    """Lower matmul A @ B -> C to tiled NKI IR.

    Supports any rank >= 2 operands with numpy-style batch broadcasting.
    A[..., M, K] @ B[..., K, N] -> C[..., M, N]

    Tiling strategy:
      - M (output partition): tiled at tile_m (max 128 = PARTITION_MAX)
      - K (contraction): tiled at tile_k (max 128), accumulated in PSUM
      - N (output free): tiled at tile_n (max 512 = PSUM_FREE_MAX)

    For each output tile (batch, m_tile, n_tile):
      1. Accumulate K-chunks: load A[m, k] -> transpose -> stationary (K, M)
         load B[k, n] -> moving (K, N), matmul into PSUM
      2. Copy PSUM -> SBUF, store to C[batch, m, n]
    """
    a_rank = len(a_shape)
    b_rank = len(b_shape)
    if a_rank < 2 or b_rank < 2:
        raise ValueError("both operands must be rank >= 2")

    M, K = a_shape[-2], a_shape[-1]
    K2, N = b_shape[-2], b_shape[-1]
    if K != K2:
        raise ValueError(f"contraction dim mismatch: {K} vs {K2}")

    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    out_batch = np.broadcast_shapes(a_batch, b_batch) if (a_batch or b_batch) else ()
    out_shape = out_batch + (M, N)

    tile_m = min(tile_m, PARTITION_MAX)
    tile_k = min(tile_k, PARTITION_MAX)
    tile_n = min(tile_n, PSUM_FREE_MAX)

    n_m_tiles = ceildiv(M, tile_m)
    n_k_tiles = ceildiv(K, tile_k)
    n_n_tiles = ceildiv(N, tile_n)

    batch_dims = list(out_batch)
    n_batch = math.prod(batch_dims) if batch_dims else 1

    b_builder = Builder("direct_matmul")
    a_hbm = b_builder.add_input("a", a_shape, dtype)
    b_hbm = b_builder.add_input("b", b_shape, dtype)
    c_hbm = b_builder.add_input("c", out_shape, DType.F32)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        """Convert flat batch index to multi-dimensional batch indices."""
        indices = []
        remaining = flat_idx
        for d in reversed(batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _a_batch_slices(batch_idx: tuple[int, ...]) -> list[DimSlice]:
        """Build batch slices for A, respecting broadcast (size-1 dims)."""
        slices = []
        offset = len(out_batch) - len(a_batch)
        for i, d in enumerate(a_batch):
            bi = batch_idx[i + offset]
            slices.append(DimSlice(0 if d == 1 else bi, 1))
        return slices

    def _b_batch_slices(batch_idx: tuple[int, ...]) -> list[DimSlice]:
        """Build batch slices for B, respecting broadcast (size-1 dims)."""
        slices = []
        offset = len(out_batch) - len(b_batch)
        for i, d in enumerate(b_batch):
            bi = batch_idx[i + offset]
            slices.append(DimSlice(0 if d == 1 else bi, 1))
        return slices

    def _c_batch_slices(batch_idx: tuple[int, ...]) -> list[DimSlice]:
        """Build batch slices for output C."""
        return [DimSlice(bi, 1) for bi in batch_idx]

    # M==1 fast path: A is a single row (1, K), so its transpose to (K, 1) is a
    # pure layout reinterpret — the K elements are contiguous in HBM either way.
    # View A's HBM buffer as (K, 1) (zero-copy) and DMA each K-tile straight into
    # a (k_size, 1) stationary tile, skipping the per-K-tile dma_transpose. See
    # emit_matmul for the full rationale. Only for a non-batched A.
    a_k1_view = b_builder.view(a_hbm, (K, 1)) if (M == 1 and not a_batch) else None

    def _load_a_stats(nb: Builder, batch_idx: tuple[int, ...], m_off: int, m_size: int):
        """Load + transpose A's K-tiles for this (batch, m) once. A's stationary
        operand depends only on (m, k), so it is reused across all N-tiles."""
        stats = []
        for k_i in range(n_k_tiles):
            k_off = k_i * tile_k
            k_size = min(tile_k, K - k_off)
            if a_k1_view is not None:
                stats.append(nb.dma_copy(
                    nb.alloc((k_size, 1), dtype, MemorySpace.SBUF),
                    a_k1_view, (DimSlice(k_off, k_size), DimSlice(0, 1)),
                ))
                continue
            a_slices = _a_batch_slices(batch_idx) + [DimSlice(m_off, m_size), DimSlice(k_off, k_size)]
            a_tile = nb.dma_copy(
                nb.alloc((m_size, k_size), dtype, MemorySpace.SBUF),
                a_hbm,
                tuple(a_slices),
            )
            stats.append(nb.transpose(a_tile, (1, 0)))
        return stats

    def _emit_tile(nb: Builder, batch_idx, a_stats, m_off, m_size, n_off, n_size):
        """Emit one (m_tile, n_tile) output tile with K accumulation, reusing
        the pre-transposed A stationary tiles."""
        psum = nb.alloc((m_size, n_size), DType.F32, MemorySpace.PSUM)
        psum = nb.memset(psum, 0.0)

        for k_i in range(n_k_tiles):
            k_off = k_i * tile_k
            k_size = min(tile_k, K - k_off)

            b_slices = _b_batch_slices(batch_idx) + [DimSlice(k_off, k_size), DimSlice(n_off, n_size)]
            b_mov = nb.dma_copy(
                nb.alloc((k_size, n_size), dtype, MemorySpace.SBUF),
                b_hbm,
                tuple(b_slices),
            )

            nb.matmul(psum, a_stats[k_i], b_mov, accumulate=(k_i > 0))

        c_sbuf = nb.tensor_copy(
            nb.alloc((m_size, n_size), DType.F32, MemorySpace.SBUF), psum
        )
        c_slices = _c_batch_slices(batch_idx) + [DimSlice(m_off, m_size), DimSlice(n_off, n_size)]
        nb.dma_copy(c_hbm, c_sbuf, tuple(c_slices))

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat) if batch_dims else ()
        for m_i in range(n_m_tiles):
            m_off = m_i * tile_m
            m_size = min(tile_m, M - m_off)
            a_stats = _load_a_stats(b_builder, batch_idx, m_off, m_size)
            for n_i in range(n_n_tiles):
                n_off = n_i * tile_n
                n_size = min(tile_n, N - n_off)
                _emit_tile(b_builder, batch_idx, a_stats, m_off, m_size, n_off, n_size)

    b_builder.set_outputs({"c": c_hbm})
    return b_builder.graph


def emit_matmul(
    nb: Builder,
    a_hbm,
    b_hbm,
    c_hbm,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype: DType = DType.F32,
) -> None:
    """Emit matmul tiling into an existing Builder with pre-allocated HBM buffers."""
    M, K = a_shape[-2], a_shape[-1]
    N = b_shape[-1]

    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    out_batch = np.broadcast_shapes(a_batch, b_batch) if (a_batch or b_batch) else ()

    tile_m = min(M, PARTITION_MAX)
    tile_k = min(K, PARTITION_MAX)
    tile_n = min(N, PSUM_FREE_MAX)
    n_m_tiles = ceildiv(M, tile_m)
    n_k_tiles = ceildiv(K, tile_k)
    n_n_tiles = ceildiv(N, tile_n)
    n_batch_total = math.prod(out_batch) if out_batch else 1
    batch_dims = list(out_batch)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _a_batch_slices(batch_idx: tuple[int, ...]) -> list[DimSlice]:
        slices = []
        offset = len(out_batch) - len(a_batch)
        for i, d in enumerate(a_batch):
            bi = batch_idx[i + offset]
            slices.append(DimSlice(0 if d == 1 else bi, 1))
        return slices

    def _b_batch_slices(batch_idx: tuple[int, ...]) -> list[DimSlice]:
        slices = []
        offset = len(out_batch) - len(b_batch)
        for i, d in enumerate(b_batch):
            bi = batch_idx[i + offset]
            slices.append(DimSlice(0 if d == 1 else bi, 1))
        return slices

    # M==1 fast path: the stationary operand A is a single row (1, K), so its
    # transpose to (K, 1) is a pure layout reinterpret — the K elements are
    # contiguous in HBM either way. Reshape A's HBM buffer to put K on the
    # partition (zero-copy row-major view) and DMA each K-tile straight into a
    # (k_size, 1) stationary tile, skipping the per-K-tile dma_transpose. This is
    # how HLO lowers the matrix-vector dots in the per-token MoE experts, and it
    # removes n_k_tiles transposes per matmul (16 per (1,2048)@(2048,384) gate-up
    # GEMM alone). Only for a non-batched A (a batched A's leading dims would sit
    # between the row and K, so the (…,1,K)->(…,K,1) collapse wouldn't be a view).
    a_k1_view = None
    if M == 1 and not a_batch:
        a_k1_view = nb.view(a_hbm, (K, 1))

    for batch_flat in range(n_batch_total):
        batch_idx = _batch_indices(batch_flat) if batch_dims else ()
        for m_i in range(n_m_tiles):
            m_off = m_i * tile_m
            m_size = min(tile_m, M - m_off)

            # Load + transpose the K-tiles of A's stationary operand ONCE per
            # (batch, m). A's transposed tiles depend only on (m, k), not on n,
            # so the old code redundantly re-loaded and re-transposed them for
            # every N-tile (n_n_tiles x too many). Each (k_size, m_size) tile is
            # <= 128x128 (<=512 B/partition), so holding all K-tiles resident
            # stays well within SBUF even for large K.
            a_stats = []
            for k_i in range(n_k_tiles):
                k_off = k_i * tile_k
                k_size = min(tile_k, K - k_off)
                if a_k1_view is not None:
                    # (K,1) view: load k-tile directly with K on the partition.
                    a_stats.append(nb.dma_copy(
                        nb.alloc((k_size, 1), dtype, MemorySpace.SBUF),
                        a_k1_view, (DimSlice(k_off, k_size), DimSlice(0, 1)),
                    ))
                    continue
                a_slices = _a_batch_slices(batch_idx) + [DimSlice(m_off, m_size), DimSlice(k_off, k_size)]
                a_tile = nb.dma_copy(
                    nb.alloc((m_size, k_size), dtype, MemorySpace.SBUF),
                    a_hbm, tuple(a_slices),
                )
                a_stats.append(nb.transpose(a_tile, (1, 0)))

            for n_i in range(n_n_tiles):
                n_off = n_i * tile_n
                n_size = min(tile_n, N - n_off)

                psum = nb.alloc((m_size, n_size), DType.F32, MemorySpace.PSUM)
                psum = nb.memset(psum, 0.0)

                for k_i in range(n_k_tiles):
                    k_off = k_i * tile_k
                    k_size = min(tile_k, K - k_off)

                    b_slices = _b_batch_slices(batch_idx) + [DimSlice(k_off, k_size), DimSlice(n_off, n_size)]
                    b_mov = nb.dma_copy(
                        nb.alloc((k_size, n_size), dtype, MemorySpace.SBUF),
                        b_hbm, tuple(b_slices),
                    )

                    nb.matmul(psum, a_stats[k_i], b_mov, accumulate=(k_i > 0))

                c_sbuf = nb.tensor_copy(
                    nb.alloc((m_size, n_size), DType.F32, MemorySpace.SBUF), psum
                )
                c_slices = [DimSlice(bi, 1) for bi in batch_idx] + [DimSlice(m_off, m_size), DimSlice(n_off, n_size)]
                nb.dma_copy(c_hbm, c_sbuf, tuple(c_slices))
