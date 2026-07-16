"""Direct lowering of broadcast_to from tensor IR to NKI IR.

``emit_broadcast_to`` is the single implementation. Its main path collapses a
contiguous run of broadcast axes into a canonical ``(L, B, T)`` decomposition
(see ``_collapse_broadcast``) and picks a partition-packed strategy by which
extent the run sits between:

  - innermost (``T == 1``): replicate a per-row scalar across the free dim
    with one ``tensor_scalar`` multiply per tile,
  - leading (``L == 1``): stride-0 partition load fans one source row across
    up to 128 partition rows per store,
  - middle: load each source block once, store it to each of the ``B`` copies.

Non-contiguous (multi-run) broadcasts fall back to a generic per-tile loop.
``lower_broadcast`` is a thin standalone-graph wrapper for the single-axis
1 -> N case.
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

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    collapse_view,
    unravel,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_schedule import TileSchedule
from nkigen_lite.tensor_ir.passes.basic.direct_lower_alloc import Scratch


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

    Thin wrapper over ``emit_broadcast_to``.
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

    b = Builder("broadcast")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)
    emit_broadcast_to(b, x_hbm, y_hbm, in_shape, out_shape, dtype)
    b.set_outputs({"y": y_hbm})
    return b.graph


def _emit_broadcast_scalar(nb: Builder, x_hbm, y_hbm, out_shape, dtype, scratch) -> None:
    """Broadcast a scalar (rank-0) HBM tensor to an arbitrary output shape."""
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import broadcast_partition

    rank = len(out_shape)
    # Load scalar as (1, 1) tile
    src_slices = [DimSlice(0, 1)] * len(x_hbm.type.shape)
    scalar_tile = scratch.load(x_hbm, src_slices, (1, 1), dtype)

    tile_p = min(out_shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = out_shape[-1] if rank >= 1 else 1
    p_extent = out_shape[-2] if rank >= 2 else 1
    batch_dims = list(out_shape[:-2]) if rank > 2 else []
    n_batch = math.prod(batch_dims) if batch_dims else 1

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()

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


def _emit_collapsed_broadcast(nb: Builder, x_hbm, y_hbm, L, B, T, dtype, scratch) -> None:
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
    if T == 1:
        # F-broadcast on (L, B): out[l, b] = in[l, 0].
        src = collapse_view(nb, x_hbm, L, 1)
        dst = collapse_view(nb, y_hbm, L, B)
        src_tiles: dict[int, object] = {}  # per-partition-row scalar, keyed by p_off
        for p_off, p_size, f_off, f_size in TileSchedule.pf(L, B, dtype).pf_tiles():
            if p_off not in src_tiles:
                src_tiles[p_off] = nb.dma_copy(
                    nb.alloc((p_size, 1), dtype, MemorySpace.SBUF),
                    src, (DimSlice(p_off, p_size), DimSlice(0, 1)),
                )
            ones = nb.constant(1.0, (p_size, f_size), dtype, MemorySpace.SBUF)
            rep = nb.tensor_scalar_arith(
                nb.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
                ones, src_tiles[p_off], NisaArithOp.MULTIPLY,
            )
            nb.dma_copy(dst, rep, (DimSlice(p_off, p_size), DimSlice(f_off, f_size)))
        return

    if L == 1:
        # P-broadcast on (B, T): every output row equals the single source row.
        src = collapse_view(nb, x_hbm, 1, T)
        dst = collapse_view(nb, y_hbm, B, T)
        for p_off, p_size, f_off, f_size in TileSchedule.pf(B, T, dtype).pf_tiles():
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
    for p_off, p_size, f_off, f_size in TileSchedule.pf(L, T, dtype).pf_tiles():
        tile = nb.dma_copy(
            nb.alloc((p_size, f_size), dtype, MemorySpace.SBUF),
            src, (DimSlice(p_off, p_size), DimSlice(0, 1), DimSlice(f_off, f_size)),
        )
        for b in range(B):
            nb.dma_copy(
                dst, tile,
                (DimSlice(p_off, p_size), DimSlice(b, 1), DimSlice(f_off, f_size)),
            )


def emit_broadcast_to(nb: Builder, x_hbm, y_hbm, in_shape, out_shape, dtype, scratch=None) -> None:
    """Emit broadcast_to tiling into an existing Builder."""
    from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import broadcast_partition
    if scratch is None:
        scratch = Scratch(nb)

    # Scalar input: load the single element and broadcast to all output tiles
    if len(in_shape) == 0:
        _emit_broadcast_scalar(nb, x_hbm, y_hbm, out_shape, dtype, scratch)
        return

    # Fast path: collapse a single contiguous broadcast run into (L, B, T) and
    # lower it with a partition-packed strategy. Falls back to the generic
    # per-tile loop below for multi-run broadcasts (rare).
    collapsed = _collapse_broadcast(in_shape, out_shape)
    if collapsed is not None:
        _emit_collapsed_broadcast(nb, x_hbm, y_hbm, *collapsed, dtype, scratch)
        return

    rank = len(out_shape)
    offset = rank - len(in_shape)
    tile_p = min(out_shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = out_shape[-1]
    p_extent = out_shape[-2] if rank >= 2 else 1
    batch_dims = list(out_shape[:-2]) if rank > 2 else []
    n_batch = math.prod(batch_dims) if batch_dims else 1

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()
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
            tile = scratch.load(x_hbm, src_slices, (src_p, src_f), dtype)

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
