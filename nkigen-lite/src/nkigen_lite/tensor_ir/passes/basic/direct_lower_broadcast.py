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
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import broadcast_partition

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

            # tensor_scalar_arith requires the scalar operand's partition dim to
            # match dst; replicate the (1, 1) scalar to (p_size, 1) first.
            if p_size > 1:
                scalar_operand = broadcast_partition(nb, scalar_tile, (p_size, 1))
            else:
                scalar_operand = scalar_tile
            ones = nb.constant(1.0, (p_size, tile_f), dtype, MemorySpace.SBUF)
            dst = nb.alloc((p_size, tile_f), dtype, MemorySpace.SBUF)
            dst = nb.tensor_scalar_arith(dst, ones, scalar_operand, NisaArithOp.MULTIPLY)

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if rank >= 2:
                dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, tile_f))
            nb.dma_copy(y_hbm, dst, dst_slices)


def _collapse_broadcast(in_shape, out_shape):
    """Collapse a broadcast into a canonical ``(L, B, T)`` decomposition.

    ``in_shape`` is right-aligned to ``out_shape`` (missing leading dims are
    size 1, per numpy broadcasting rules). Every output axis is then either a
    *keep* axis (input extent already matches) or a *broadcast* axis (input
    extent is 1, output > 1).

    When all broadcast axes form a single contiguous run, the whole op reduces
    to three flat extents:
      - ``L`` = product of the keep axes *before* the run,
      - ``B`` = product of the broadcast run,
      - ``T`` = product of the keep axes *after* the run.
    The input is then a contiguous ``(L, 1, T)`` tensor and the output a
    contiguous ``(L, B, T)`` tensor (both row-major, so the reshape is a
    zero-copy HBM view).

    Returns ``(L, B, T)`` for the single-run case, or ``None`` when the
    broadcast axes are non-contiguous (multiple runs) — callers fall back to
    the generic per-tile lowering.
    """
    R = len(out_shape)
    in_al = (1,) * (R - len(in_shape)) + tuple(in_shape)
    b_axes = []
    for i in range(R):
        if in_al[i] == out_shape[i]:
            continue  # keep axis (includes trivial 1 -> 1)
        if in_al[i] == 1 and out_shape[i] > 1:
            b_axes.append(i)
        else:
            raise ValueError(
                f"broadcast: axis {i} not broadcastable {in_al[i]} -> {out_shape[i]}"
            )
    if not b_axes:
        return None  # pure copy / no expansion; let the generic path handle it
    # Require a single contiguous run of broadcast axes.
    if b_axes != list(range(b_axes[0], b_axes[-1] + 1)):
        return None
    bstart, bend = b_axes[0], b_axes[-1]
    L = math.prod(out_shape[:bstart]) if bstart > 0 else 1
    B = math.prod(out_shape[bstart:bend + 1])
    T = math.prod(out_shape[bend + 1:]) if bend + 1 < R else 1
    return L, B, T


def _emit_collapsed_broadcast(nb: Builder, x_hbm, y_hbm, L, B, T, dtype) -> None:
    """Emit a single-run broadcast lowered through the canonical ``(L, B, T)``.

    Both HBM tensors are reshaped to the collapsed form via zero-copy views,
    then one of three partition-efficient strategies is chosen by which of the
    three extents the broadcast sits between:

      - ``T == 1`` (broadcast is the innermost axis): F-broadcast over ``(L, B)``
        — partition over ``L``, replicate the per-row scalar across ``B`` with
        a single ``tensor_scalar`` multiply per tile.
      - ``L == 1`` (broadcast spans the leading axes): P-broadcast over
        ``(B, T)`` — every output row is the same ``(T,)`` source row, so a
        stride-0 partition load fans it across up to 128 rows per store.
      - otherwise (genuine middle broadcast): load each ``(<=128, T)`` block of
        the source once and store it to each of the ``B`` output copies.

    All three pack the 128-wide partition and tile the free dim to the SBUF
    budget, replacing the old per-batch-element DMA loop.
    """
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import max_free_elems

    cap = max_free_elems(dtype)

    if T == 1:
        # F-broadcast on (L, B): out[l, b] = in[l, 0].
        src = nb.view(x_hbm, (L, 1))
        dst = nb.view(y_hbm, (L, B))
        for p_i in range(ceildiv(L, PARTITION_MAX)):
            p_off = p_i * PARTITION_MAX
            p_size = min(PARTITION_MAX, L - p_off)
            src_tile = nb.dma_copy(
                nb.alloc((p_size, 1), dtype, MemorySpace.SBUF),
                src, (DimSlice(p_off, p_size), DimSlice(0, 1)),
            )
            for f_i in range(ceildiv(B, cap)):
                f_off = f_i * cap
                f_size = min(cap, B - f_off)
                ones = nb.constant(1.0, (p_size, f_size), dtype, MemorySpace.SBUF)
                rep = nb.tensor_scalar_arith(
                    nb.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                    ones, src_tile, NisaArithOp.MULTIPLY,
                )
                nb.dma_copy(dst, rep, (DimSlice(p_off, p_size), DimSlice(f_off, f_size)))
        return

    if L == 1:
        # P-broadcast on (B, T): every output row equals the single source row.
        src = nb.view(x_hbm, (1, T))
        dst = nb.view(y_hbm, (B, T))
        for p_i in range(ceildiv(B, PARTITION_MAX)):
            p_off = p_i * PARTITION_MAX
            p_size = min(PARTITION_MAX, B - p_off)
            for f_i in range(ceildiv(T, cap)):
                f_off = f_i * cap
                f_size = min(cap, T - f_off)
                # Stride-0 partition load fans the one source row across p_size rows.
                rep = nb.dma_copy(
                    nb.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                    src, (DimSlice(0, p_size, stride=0), DimSlice(f_off, f_size)),
                )
                nb.dma_copy(dst, rep, (DimSlice(p_off, p_size), DimSlice(f_off, f_size)))
        return

    # Genuine middle broadcast (L, B, T): copy each source (<=128, T) block to
    # all B output slices. Load once per block, store B times.
    src = nb.view(x_hbm, (L, 1, T))
    dst = nb.view(y_hbm, (L, B, T))
    for p_i in range(ceildiv(L, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, L - p_off)
        for f_i in range(ceildiv(T, cap)):
            f_off = f_i * cap
            f_size = min(cap, T - f_off)
            tile = nb.dma_copy(
                nb.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                src, (DimSlice(p_off, p_size), DimSlice(0, 1), DimSlice(f_off, f_size)),
            )
            for b in range(B):
                nb.dma_copy(
                    dst, tile,
                    (DimSlice(p_off, p_size), DimSlice(b, 1), DimSlice(f_off, f_size)),
                )


def emit_broadcast_to(nb: Builder, x_hbm, y_hbm, in_shape, out_shape, dtype) -> None:
    """Emit broadcast_to tiling into an existing Builder."""
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import broadcast_partition

    # Scalar input: load the single element and broadcast to all output tiles
    if len(in_shape) == 0:
        _emit_broadcast_scalar(nb, x_hbm, y_hbm, out_shape, dtype)
        return

    # Fast path: collapse a single contiguous broadcast run into (L, B, T) and
    # lower it with a partition-packed strategy. Falls back to the generic
    # per-tile loop below for multi-run broadcasts (rare).
    collapsed = _collapse_broadcast(in_shape, out_shape)
    if collapsed is not None:
        _emit_collapsed_broadcast(nb, x_hbm, y_hbm, *collapsed, dtype)
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
