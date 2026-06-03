"""Direct lowering of broadcast_to from tensor IR to NKI IR.

Supports broadcasting a single dimension (I, P, or F) from size 1 to size N
within any-rank tensors. The broadcast dimension determines the strategy:

  - I-dim (batch): loop over output batch indices, load source with fixed
    index 0 on the broadcast dim, store to each output batch slice.
  - P-dim (partition): tensor engine trick — construct an all-ones stationary
    vector ones[1, P] and compute ones.T @ src[1, F] -> dst[P, F].
  - F-dim (free): vector engine — use tensor_scalar_arith to multiply a
    full-size ones tile (P, F) by the source (P, 1), broadcasting along F.

The caller specifies which dimension (by axis index) is being broadcast and
to what size. The input must have size 1 on that axis.
"""

from __future__ import annotations

import math

import numpy as np

from nkigen_lite.core import DType, Graph
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    NisaArithOp,
    PARTITION_MAX,
    PSUM_FREE_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import ceildiv


def lower_broadcast(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    broadcast_axis: int,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower broadcast_to where a single axis goes from 1 to N.

    Args:
        in_shape: Input shape with size 1 on broadcast_axis.
        out_shape: Output shape (same as input except broadcast_axis has size N).
        broadcast_axis: Which axis is being broadcast (0-indexed).
        dtype: Element type.

    The axis is classified as I (batch), P (partition), or F (free) based on
    its position relative to the last two dims:
      - axis < rank-2: I-dim (batch)
      - axis == rank-2: P-dim (partition, second-to-last)
      - axis == rank-1: F-dim (free, last)
    """
    rank = len(in_shape)
    if rank < 2:
        raise ValueError("input must be rank >= 2")
    if len(out_shape) != rank:
        raise ValueError("input and output must have same rank")
    if in_shape[broadcast_axis] != 1:
        raise ValueError(
            f"input must have size 1 on broadcast_axis={broadcast_axis}, "
            f"got {in_shape[broadcast_axis]}"
        )
    if broadcast_axis < 0:
        broadcast_axis += rank

    for i in range(rank):
        if i == broadcast_axis:
            continue
        if in_shape[i] != out_shape[i]:
            raise ValueError(
                f"non-broadcast dims must match: axis {i} "
                f"{in_shape[i]} vs {out_shape[i]}"
            )

    if broadcast_axis < rank - 2:
        return _lower_i_broadcast(in_shape, out_shape, broadcast_axis, dtype)
    elif broadcast_axis == rank - 2:
        return _lower_p_broadcast(in_shape, out_shape, dtype)
    else:
        return _lower_f_broadcast(in_shape, out_shape, dtype)


def _lower_i_broadcast(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    broadcast_axis: int,
    dtype: DType,
) -> Graph:
    """I-dim broadcast: loop over output batch, load source with index 0."""
    rank = len(in_shape)
    P = in_shape[-2]
    F = in_shape[-1]
    broadcast_size = out_shape[broadcast_axis]

    batch_dims = list(out_shape[:-2])
    n_batch = math.prod(batch_dims) if batch_dims else 1

    tile_p = min(P, PARTITION_MAX)
    tile_f = min(F, PSUM_FREE_MAX)

    b = Builder("broadcast_i")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _src_slices(batch_idx: tuple[int, ...], p_off: int, p_size: int, f_off: int, f_size: int):
        slices = []
        for i, d in enumerate(in_shape[:-2]):
            if i == broadcast_axis:
                slices.append(DimSlice(0, 1))
            else:
                slices.append(DimSlice(batch_idx[i], 1))
        slices.append(DimSlice(p_off, p_size))
        slices.append(DimSlice(f_off, f_size))
        return tuple(slices)

    def _dst_slices(batch_idx: tuple[int, ...], p_off: int, p_size: int, f_off: int, f_size: int):
        slices = []
        for i in range(rank - 2):
            slices.append(DimSlice(batch_idx[i], 1))
        slices.append(DimSlice(p_off, p_size))
        slices.append(DimSlice(f_off, f_size))
        return tuple(slices)

    n_p_tiles = ceildiv(P, tile_p)
    n_f_tiles = ceildiv(F, tile_f)

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat)
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, P - p_off)
            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, F - f_off)
                tile = b.dma_copy(
                    b.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                    x_hbm,
                    _src_slices(batch_idx, p_off, p_size, f_off, f_size),
                )
                b.dma_copy(y_hbm, tile, _dst_slices(batch_idx, p_off, p_size, f_off, f_size))

    b.set_outputs({"y": y_hbm})
    return b.graph


def _lower_p_broadcast(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: DType,
) -> Graph:
    """P-dim broadcast: ones[1,P].T @ src[1,F] -> dst[P,F] via tensor engine."""
    rank = len(in_shape)
    P_out = out_shape[-2]
    F = in_shape[-1]

    batch_dims = list(out_shape[:-2])
    n_batch = math.prod(batch_dims) if batch_dims else 1

    tile_p = min(P_out, PARTITION_MAX)
    tile_f = min(F, PSUM_FREE_MAX)

    b = Builder("broadcast_p")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, DType.F32)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _src_slices(batch_idx: tuple[int, ...], f_off: int, f_size: int):
        slices = []
        for i in range(rank - 2):
            slices.append(DimSlice(batch_idx[i], 1))
        slices.append(DimSlice(0, 1))
        slices.append(DimSlice(f_off, f_size))
        return tuple(slices)

    def _dst_slices(batch_idx: tuple[int, ...], p_off: int, p_size: int, f_off: int, f_size: int):
        slices = []
        for i in range(rank - 2):
            slices.append(DimSlice(batch_idx[i], 1))
        slices.append(DimSlice(p_off, p_size))
        slices.append(DimSlice(f_off, f_size))
        return tuple(slices)

    n_p_tiles = ceildiv(P_out, tile_p)
    n_f_tiles = ceildiv(F, tile_f)

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat)
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, P_out - p_off)

            # Stationary: ones[1, p_size] — K=1, M=p_size
            ones_stat = b.constant(1.0, (1, p_size), dtype, MemorySpace.SBUF)

            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, F - f_off)

                # Moving: src[1, f_size] — K=1, N=f_size
                src_mov = b.dma_copy(
                    b.alloc((1, f_size), dtype, MemorySpace.SBUF),
                    x_hbm,
                    _src_slices(batch_idx, f_off, f_size),
                )

                # matmul: ones[1, p_size].T @ src[1, f_size] -> psum[p_size, f_size]
                psum = b.alloc((p_size, f_size), DType.F32, MemorySpace.PSUM)
                b.matmul(psum, ones_stat, src_mov, accumulate=False)

                # PSUM -> SBUF -> HBM
                out_sbuf = b.tensor_copy(
                    b.alloc((p_size, f_size), DType.F32, MemorySpace.SBUF), psum
                )
                b.dma_copy(y_hbm, out_sbuf, _dst_slices(batch_idx, p_off, p_size, f_off, f_size))

    b.set_outputs({"y": y_hbm})
    return b.graph


def _lower_f_broadcast(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: DType,
) -> Graph:
    """F-dim broadcast: tensor_scalar_arith(ones(P,F), src(P,1), MULTIPLY)."""
    rank = len(in_shape)
    P = in_shape[-2]
    F_out = out_shape[-1]

    batch_dims = list(out_shape[:-2])
    n_batch = math.prod(batch_dims) if batch_dims else 1

    tile_p = min(P, PARTITION_MAX)
    tile_f = min(F_out, PSUM_FREE_MAX)

    b = Builder("broadcast_f")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    def _batch_indices(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    def _src_slices(batch_idx: tuple[int, ...], p_off: int, p_size: int):
        slices = []
        for i in range(rank - 2):
            slices.append(DimSlice(batch_idx[i], 1))
        slices.append(DimSlice(p_off, p_size))
        slices.append(DimSlice(0, 1))
        return tuple(slices)

    def _dst_slices(batch_idx: tuple[int, ...], p_off: int, p_size: int, f_off: int, f_size: int):
        slices = []
        for i in range(rank - 2):
            slices.append(DimSlice(batch_idx[i], 1))
        slices.append(DimSlice(p_off, p_size))
        slices.append(DimSlice(f_off, f_size))
        return tuple(slices)

    n_p_tiles = ceildiv(P, tile_p)
    n_f_tiles = ceildiv(F_out, tile_f)

    for batch_flat in range(n_batch):
        batch_idx = _batch_indices(batch_flat)
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, P - p_off)

            # Load source (P_tile, 1) — the scalar operand for broadcast
            src_tile = b.dma_copy(
                b.alloc((p_size, 1), dtype, MemorySpace.SBUF),
                x_hbm,
                _src_slices(batch_idx, p_off, p_size),
            )

            for f_i in range(n_f_tiles):
                f_off = f_i * tile_f
                f_size = min(tile_f, F_out - f_off)

                # ones(p_size, f_size) — the "x" tensor in tensor_scalar_arith
                ones_tile = b.constant(1.0, (p_size, f_size), dtype, MemorySpace.SBUF)

                # dst = ones * src (broadcast src along F via tensor_scalar_arith)
                dst = b.alloc((p_size, f_size), dtype, MemorySpace.SBUF)
                dst = b.tensor_scalar_arith(dst, ones_tile, src_tile, NisaArithOp.MULTIPLY)

                b.dma_copy(y_hbm, dst, _dst_slices(batch_idx, p_off, p_size, f_off, f_size))

    b.set_outputs({"y": y_hbm})
    return b.graph


def _emit_broadcast_scalar(nb: Builder, x_hbm, y_hbm, out_shape, dtype) -> None:
    """Broadcast a scalar (rank-0) HBM tensor to an arbitrary output shape."""
    rank = len(out_shape)
    # Load scalar as (1, 1) tile
    src_slices = [DimSlice(0, 1)] * len(x_hbm.type.shape)
    scalar_tile = nb.dma_copy(
        nb.alloc((1, 1), dtype, MemorySpace.SBUF), x_hbm, src_slices
    )

    tile_p = min(out_shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = out_shape[-1] if rank >= 1 else 1
    p_extent = out_shape[-2] if rank >= 2 else 1
    batch_dims = list(out_shape[:-2]) if rank > 2 else []
    n_batch = math.prod(batch_dims) if batch_dims else 1

    for bf in range(n_batch):
        batch_idx = []
        remaining = bf
        for d in reversed(batch_dims):
            batch_idx.append(remaining % d)
            remaining //= d
        batch_idx = tuple(reversed(batch_idx))

        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)

            ones = nb.constant(1.0, (p_size, tile_f), dtype, MemorySpace.SBUF)
            dst = nb.alloc((p_size, tile_f), dtype, MemorySpace.SBUF)
            dst = nb.tensor_scalar_arith(dst, ones, scalar_tile, NisaArithOp.MULTIPLY)

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if rank >= 2:
                dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, tile_f))
            nb.dma_copy(y_hbm, dst, dst_slices)


def emit_broadcast_to(nb: Builder, x_hbm, y_hbm, in_shape, out_shape, dtype) -> None:
    """Emit broadcast_to tiling into an existing Builder."""
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import broadcast_partition

    # Scalar input: load the single element and broadcast to all output tiles
    if len(in_shape) == 0:
        _emit_broadcast_scalar(nb, x_hbm, y_hbm, out_shape, dtype)
        return

    rank = len(out_shape)
    offset = rank - len(in_shape)
    tile_p = min(out_shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = out_shape[-1]
    p_extent = out_shape[-2] if rank >= 2 else 1
    batch_dims = list(out_shape[:-2]) if rank > 2 else []
    n_batch = math.prod(batch_dims) if batch_dims else 1

    def _unravel_idx(flat_idx: int) -> tuple[int, ...]:
        indices = []
        remaining = flat_idx
        for d in reversed(batch_dims):
            indices.append(remaining % d)
            remaining //= d
        return tuple(reversed(indices))

    for bf in range(n_batch):
        batch_idx = _unravel_idx(bf) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)

            src_slices = []
            for i in range(len(in_shape)):
                out_i = i + offset
                if in_shape[i] == 1:
                    src_slices.append(DimSlice(0, 1))
                elif rank > 2 and out_i < rank - 2:
                    src_slices.append(DimSlice(batch_idx[out_i], 1))
                elif out_i == rank - 2:
                    src_slices.append(DimSlice(p_off, p_size))
                else:
                    src_slices.append(DimSlice(0, tile_f))

            src_p = p_size if (len(in_shape) >= 2 and in_shape[-2] > 1) else 1
            src_f = tile_f if in_shape[-1] > 1 else 1
            tile = nb.dma_copy(
                nb.alloc((src_p, src_f), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )

            if src_p == 1 and p_size > 1:
                tile = broadcast_partition(nb, tile, (p_size, src_f))

            if src_f == 1 and tile_f > 1:
                ones = nb.constant(1.0, (tile.type.shape[0], tile_f), dtype, MemorySpace.SBUF)
                dst = nb.alloc((tile.type.shape[0], tile_f), dtype, MemorySpace.SBUF)
                tile = nb.tensor_scalar_arith(dst, ones, tile, NisaArithOp.MULTIPLY)

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if rank >= 2:
                dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, tile_f))
            nb.dma_copy(y_hbm, tile, dst_slices)
