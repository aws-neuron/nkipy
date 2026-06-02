"""Direct lowering of transpose from tensor IR to NKI IR.

Supports arbitrary permutations on any-rank tensors (rank >= 2). Two strategies:

  - DMA transpose: batch dims are reordered via DMA slice remapping. When the
    permutation swaps the last two dims (P↔F), dma_transpose handles the
    on-chip swap. When no P↔F swap is needed, a plain DMA copy suffices.

  - Tensor engine transpose: same batch-dim remapping, but for the P↔F swap
    uses the matmul trick: stat[K, M].T @ I[K, N] materializes the transpose.
    Only needed when the permutation swaps the last two dims.

For any permutation perm, the output shape is [in_shape[perm[i]] for i in range(rank)].
The key observation: on NeuronCore, only the last two dims are "on-chip" (P and F).
Batch dim reordering is just DMA slice coordinate remapping (reading from different
positions in HBM). The only operation that requires on-chip work is swapping P↔F.
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
    MATMUL_STATIONARY_FREE_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import ceildiv


def _needs_pf_swap(perm: tuple[int, ...]) -> bool:
    """Check if the permutation swaps the relative order of the last two source dims.

    If the source dim that maps to output[-2] has a higher index than the one
    that maps to output[-1], the on-chip tile needs a P↔F transpose.
    """
    rank = len(perm)
    return perm[rank - 2] > perm[rank - 1]


def lower_transpose_dma(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower arbitrary transpose via DMA engine.

    Args:
        in_shape: Input tensor shape, rank >= 2.
        perm: Permutation of axes. None defaults to swapping last two dims.
        dtype: Element type.

    Batch dim reordering is handled by reading from remapped HBM coordinates.
    P↔F swap (when needed) uses dma_transpose on-chip. Both P and F tiles
    are capped at 128 since after transposing, either could be a partition dim.
    """
    rank = len(in_shape)
    if rank < 2:
        raise ValueError("input must be rank >= 2")
    if perm is None:
        perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
    if sorted(perm) != list(range(rank)):
        raise ValueError(f"invalid permutation: {perm}")

    out_shape = tuple(in_shape[p] for p in perm)
    swap_pf = _needs_pf_swap(perm)

    # Both tiles capped at 128: after a P↔F swap the F-dim becomes the
    # new partition dim, so it must also fit within PARTITION_MAX.
    tile_p = min(out_shape[-2], PARTITION_MAX)
    tile_f = min(out_shape[-1], PARTITION_MAX)
    n_p_tiles = ceildiv(out_shape[-2], tile_p)
    n_f_tiles = ceildiv(out_shape[-1], tile_f)

    out_batch_dims = list(out_shape[:-2])
    n_batch = math.prod(out_batch_dims) if out_batch_dims else 1

    b = Builder("transpose_dma")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(out_batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _build_src_slices(batch_idx, p_off, p_size, f_off, f_size):
        """Map output tile coordinates back to source HBM slices via perm."""
        out_coords = {}
        for i, bi in enumerate(batch_idx):
            out_coords[i] = (bi, 1)
        out_coords[rank - 2] = (p_off, p_size)
        out_coords[rank - 1] = (f_off, f_size)

        src_slices = [None] * rank
        for out_dim in range(rank):
            src_dim = perm[out_dim]
            src_slices[src_dim] = DimSlice(*out_coords[out_dim])
        return tuple(src_slices)

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat) if out_batch_dims else ()
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, out_shape[-2] - p_off)
            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, out_shape[-1] - f_off)

                src_slices = _build_src_slices(batch_idx, p_off, p_size, f_off, f_size)
                dst_slices = tuple(
                    [DimSlice(bi, 1) for bi in batch_idx]
                    + [DimSlice(p_off, p_size), DimSlice(f_off, f_size)]
                )

                if swap_pf:
                    # Source loads as (f_size, p_size) due to reversed dim order,
                    # then dma_transpose to (p_size, f_size) for the output.
                    tile = b.dma_copy(
                        b.alloc((f_size, p_size), dtype, MemorySpace.SBUF),
                        x_hbm, src_slices,
                    )
                    transposed = b.transpose(tile, (1, 0))
                    b.dealloc(tile)
                    b.dma_copy(y_hbm, transposed, dst_slices)
                    b.dealloc(transposed)
                else:
                    # No P↔F swap needed, just remap batch coordinates
                    tile = b.dma_copy(
                        b.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                        x_hbm, src_slices,
                    )
                    b.dma_copy(y_hbm, tile, dst_slices)
                    b.dealloc(tile)

    b.set_outputs({"y": y_hbm})
    return b.graph


def lower_transpose_te(
    in_shape: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower arbitrary transpose via tensor engine (A.T @ I trick).

    Args:
        in_shape: Input tensor shape, rank >= 2.
        perm: Permutation of axes. None defaults to swapping last two dims.
        dtype: Element type.

    Batch dim reordering is DMA slice remapping (same as DMA strategy). For
    the P↔F swap, uses matmul:
      stat[K=f_size, M=p_size].T @ I[K=f_size, N=f_size] -> dst[p_size, f_size]

    The loaded source tile is (f_size, p_size) — used as stationary with K=f_size,
    M=p_size. The identity I is (f_size, f_size). The result stat.T @ I is
    (p_size, f_size) = the transposed tile.

    Constraints: K=f_size <= 128, M=p_size <= 128, N=f_size <= 512.
    So both tile_p and tile_f are capped at 128.

    When no P↔F swap is needed, falls back to plain DMA copy.

    Requires an identity matrix as HBM input ("eye").
    """
    rank = len(in_shape)
    if rank < 2:
        raise ValueError("input must be rank >= 2")
    if perm is None:
        perm = tuple(range(rank - 2)) + (rank - 1, rank - 2)
    if sorted(perm) != list(range(rank)):
        raise ValueError(f"invalid permutation: {perm}")

    out_shape = tuple(in_shape[p] for p in perm)
    swap_pf = _needs_pf_swap(perm)

    # For TE: K=f_size <= 128 (partition), M=p_size <= 128 (stat free)
    tile_p = min(out_shape[-2], PARTITION_MAX)
    tile_f = min(out_shape[-1], PARTITION_MAX)
    n_p_tiles = ceildiv(out_shape[-2], tile_p)
    n_f_tiles = ceildiv(out_shape[-1], tile_f)

    out_batch_dims = list(out_shape[:-2])
    n_batch = math.prod(out_batch_dims) if out_batch_dims else 1

    # Identity matrix: size = tile_f (the K=N dimension for the matmul)
    eye_size = tile_f if swap_pf else 0

    bld = Builder("transpose_te")
    x_hbm = bld.add_input("x", in_shape, dtype)
    if swap_pf:
        y_hbm = bld.add_input("y", out_shape, DType.F32)
        eye_hbm = bld.add_input("eye", (eye_size, eye_size), dtype)
    else:
        y_hbm = bld.add_input("y", out_shape, dtype)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(out_batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _build_src_slices(batch_idx, p_off, p_size, f_off, f_size):
        out_coords = {}
        for i, bi in enumerate(batch_idx):
            out_coords[i] = (bi, 1)
        out_coords[rank - 2] = (p_off, p_size)
        out_coords[rank - 1] = (f_off, f_size)

        src_slices = [None] * rank
        for out_dim in range(rank):
            src_dim = perm[out_dim]
            src_slices[src_dim] = DimSlice(*out_coords[out_dim])
        return tuple(src_slices)

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat) if out_batch_dims else ()
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, out_shape[-2] - p_off)

            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, out_shape[-1] - f_off)

                src_slices = _build_src_slices(batch_idx, p_off, p_size, f_off, f_size)
                dst_slices = tuple(
                    [DimSlice(bi, 1) for bi in batch_idx]
                    + [DimSlice(p_off, p_size), DimSlice(f_off, f_size)]
                )

                if swap_pf:
                    # Source loads as (f_size, p_size) — reversed dim order
                    # Use as stationary: stat[K=f_size, M=p_size]
                    # stat.T @ I[K=f_size, N=f_size] -> (p_size, f_size)
                    stat = bld.dma_copy(
                        bld.alloc((f_size, p_size), dtype, MemorySpace.SBUF),
                        x_hbm, src_slices,
                    )
                    eye_tile = bld.dma_copy(
                        bld.alloc((f_size, f_size), dtype, MemorySpace.SBUF),
                        eye_hbm,
                        (DimSlice(0, f_size), DimSlice(0, f_size)),
                    )

                    psum = bld.alloc((p_size, f_size), DType.F32, MemorySpace.PSUM)
                    bld.matmul(psum, stat, eye_tile, accumulate=False)
                    bld.dealloc(stat)
                    bld.dealloc(eye_tile)

                    out_sbuf = bld.tensor_copy(
                        bld.alloc((p_size, f_size), DType.F32, MemorySpace.SBUF), psum
                    )
                    bld.dealloc(psum)
                    bld.dma_copy(y_hbm, out_sbuf, dst_slices)
                    bld.dealloc(out_sbuf)
                else:
                    # No swap, plain DMA copy with remapped slices
                    tile = bld.dma_copy(
                        bld.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                        x_hbm, src_slices,
                    )
                    bld.dma_copy(y_hbm, tile, dst_slices)
                    bld.dealloc(tile)

    bld.set_outputs({"y": y_hbm})
    return bld.graph


def emit_transpose(
    nb: Builder,
    x_hbm,
    y_hbm,
    in_shape: tuple[int, ...],
    perm: tuple[int, ...],
    dtype: DType = DType.F32,
) -> None:
    """Emit transpose tiling into an existing Builder (DMA strategy)."""
    rank = len(in_shape)
    out_shape = tuple(in_shape[p] for p in perm)
    swap_pf = _needs_pf_swap(perm)

    tile_p = min(out_shape[-2], PARTITION_MAX)
    tile_f = min(out_shape[-1], PARTITION_MAX)
    n_p_tiles = ceildiv(out_shape[-2], tile_p)
    n_f_tiles = ceildiv(out_shape[-1], tile_f)

    out_batch_dims = list(out_shape[:-2])
    n_batch = math.prod(out_batch_dims) if out_batch_dims else 1

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(out_batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _build_src_slices(batch_idx, p_off, p_size, f_off, f_size):
        out_coords = {}
        for i, bi in enumerate(batch_idx):
            out_coords[i] = (bi, 1)
        out_coords[rank - 2] = (p_off, p_size)
        out_coords[rank - 1] = (f_off, f_size)
        src_slices = [None] * rank
        for out_dim in range(rank):
            src_dim = perm[out_dim]
            src_slices[src_dim] = DimSlice(*out_coords[out_dim])
        return tuple(src_slices)

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat) if out_batch_dims else ()
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, out_shape[-2] - p_off)
            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, out_shape[-1] - f_off)

                src_slices = _build_src_slices(batch_idx, p_off, p_size, f_off, f_size)
                dst_slices = tuple(
                    [DimSlice(bi, 1) for bi in batch_idx]
                    + [DimSlice(p_off, p_size), DimSlice(f_off, f_size)]
                )

                if swap_pf:
                    tile = nb.dma_copy(
                        nb.alloc((f_size, p_size), dtype, MemorySpace.SBUF),
                        x_hbm, src_slices,
                    )
                    transposed = nb.transpose(tile, (1, 0))
                    nb.dma_copy(y_hbm, transposed, dst_slices)
                else:
                    tile = nb.dma_copy(
                        nb.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                        x_hbm, src_slices,
                    )
                    nb.dma_copy(y_hbm, tile, dst_slices)
