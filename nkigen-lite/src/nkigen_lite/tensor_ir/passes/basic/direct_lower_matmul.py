"""Direct lowering of tensor IR matmul to NKI IR.

Lowers A[..., M, K] @ B[..., K, N] -> C[..., M, N] for any legal shape
(rank >= 2, with numpy-style batch broadcasting). Generates tiled NKI IR
with K-accumulation in PSUM, M-tiling on the output partition axis, and
N-tiling when N exceeds PSUM_FREE_MAX.

``emit_matmul`` is the single implementation, emitting into an existing
Builder with pre-allocated HBM buffers; ``lower_matmul`` is a thin
standalone-graph wrapper over it.
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

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import ceildiv, unravel
from nkigen_lite.tensor_ir.passes.basic.direct_lower_alloc import Allocator


def emit_matmul(
    nb: Builder,
    a_hbm,
    b_hbm,
    c_hbm,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype: DType = DType.F32,
    alloc: "Allocator | None" = None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
) -> None:
    """Emit matmul tiling into an existing Builder with pre-allocated HBM buffers.

    Tiling strategy:
      - M (output partition): tiled at tile_m (max 128 = PARTITION_MAX)
      - K (contraction): tiled at tile_k (max 128), accumulated in PSUM
      - N (output free): tiled at tile_n (max 512 = PSUM_FREE_MAX)

    For each output tile (batch, m_tile, n_tile):
      1. Accumulate K-chunks: load A[m, k] -> transpose -> stationary (K, M)
         load B[k, n] -> moving (K, N), matmul into PSUM
      2. Copy PSUM -> SBUF, store to C[batch, m, n]

    ``tile_m``/``tile_n``/``tile_k`` default to the hardware maxima; smaller
    explicit values are honored (capped at the maxima).
    """
    if alloc is None:
        alloc = Allocator(nb)

    M, K = a_shape[-2], a_shape[-1]
    N = b_shape[-1]

    # PSUM accumulates in F32; the PSUM->SBUF tensor_copy stages the result in
    # the destination buffer's dtype (tensor_copy casts), so a bf16 c_hbm gets
    # a bf16 store rather than a silent F32/bf16 mismatch at the dma_copy.
    out_dtype = c_hbm.type.dtype

    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    out_batch = np.broadcast_shapes(a_batch, b_batch) if (a_batch or b_batch) else ()

    tile_m = min(tile_m or PARTITION_MAX, PARTITION_MAX, M)
    tile_k = min(tile_k or PARTITION_MAX, PARTITION_MAX, K)
    tile_n = min(tile_n or PSUM_FREE_MAX, PSUM_FREE_MAX, N)
    n_m_tiles = ceildiv(M, tile_m)
    n_k_tiles = ceildiv(K, tile_k)
    n_n_tiles = ceildiv(N, tile_n)
    n_batch_total = math.prod(out_batch) if out_batch else 1
    batch_dims = list(out_batch)

    def _batch_slices(
        operand_batch: tuple[int, ...], batch_idx: tuple[int, ...],
    ) -> list[DimSlice]:
        """Build an operand's batch slices, respecting broadcast (size-1 dims)."""
        offset = len(out_batch) - len(operand_batch)
        return [
            DimSlice(0 if d == 1 else batch_idx[i + offset], 1)
            for i, d in enumerate(operand_batch)
        ]

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
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for m_i in range(n_m_tiles):
            m_off = m_i * tile_m
            m_size = min(tile_m, M - m_off)

            # Load + transpose the K-tiles of A's stationary operand ONCE per
            # (batch, m). A's transposed tiles depend only on (m, k), not on n,
            # so they are reused across all N-tiles. Each (k_size, m_size) tile
            # is <= 128x128 (<=512 B/partition), so holding all K-tiles resident
            # stays well within SBUF even for large K.
            a_stats = []
            for k_i in range(n_k_tiles):
                k_off = k_i * tile_k
                k_size = min(tile_k, K - k_off)
                if a_k1_view is not None:
                    # (K,1) view: load k-tile directly with K on the partition.
                    a_stats.append(alloc.load(
                        a_k1_view, (DimSlice(k_off, k_size), DimSlice(0, 1)),
                        (k_size, 1), dtype,
                    ))
                    continue
                a_slices = _batch_slices(a_batch, batch_idx) + [DimSlice(m_off, m_size), DimSlice(k_off, k_size)]
                a_tile = alloc.load(
                    a_hbm, tuple(a_slices), (m_size, k_size), dtype,
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

                    b_slices = _batch_slices(b_batch, batch_idx) + [DimSlice(k_off, k_size), DimSlice(n_off, n_size)]
                    b_mov = alloc.load(
                        b_hbm, tuple(b_slices), (k_size, n_size), dtype,
                    )

                    nb.matmul(psum, a_stats[k_i], b_mov, accumulate=(k_i > 0))

                c_sbuf = nb.tensor_copy(
                    alloc.sbuf((m_size, n_size), out_dtype), psum
                )
                c_slices = [DimSlice(bi, 1) for bi in batch_idx] + [DimSlice(m_off, m_size), DimSlice(n_off, n_size)]
                nb.dma_copy(c_hbm, c_sbuf, tuple(c_slices))


def lower_matmul(
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    dtype: DType = DType.F32,
    tile_m: int = 128,
    tile_n: int = 512,
    tile_k: int = 128,
) -> Graph:
    """Lower matmul A @ B -> C into a standalone graph.

    Thin wrapper over ``emit_matmul``: HBM inputs ``a``/``b``/``c``, one emit
    call. Supports any rank >= 2 operands with numpy-style batch broadcasting.
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

    b_builder = Builder("direct_matmul")
    a_hbm = b_builder.add_input("a", a_shape, dtype)
    b_hbm = b_builder.add_input("b", b_shape, dtype)
    c_hbm = b_builder.add_input("c", out_shape, DType.F32)
    emit_matmul(
        b_builder, a_hbm, b_hbm, c_hbm, a_shape, b_shape, dtype,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    b_builder.set_outputs({"c": c_hbm})
    return b_builder.graph
