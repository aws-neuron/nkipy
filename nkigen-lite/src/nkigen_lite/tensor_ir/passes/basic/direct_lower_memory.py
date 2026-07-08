"""Direct lowering of memory/shape ops (reshape, slice, concat) to NKI IR.

These are pure data-movement ops with no compute. Each is lowered to tiled
DMA copies between HBM source and destination with appropriate indexing.

  - reshape: reinterprets the HBM buffer layout. When the total element count
    is preserved, this is a tiled copy with different source/destination
    indexing derived from the shape change.

  - slice: extracts a contiguous sub-tensor from the source at given
    start/stop/stride offsets. Lowered as DMA loads from offset positions.

  - concat: assembles multiple source tensors along a given axis into one
    output tensor. Lowered as DMA copies from each source into the
    appropriate offset of the destination.

``emit_reshape`` / ``emit_slice`` / ``emit_concat`` are the single
implementations; the ``lower_*`` entry points are thin standalone-graph
wrappers over them.
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import DType, Graph
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ceildiv,
    collapse_view,
    flat_range_to_src_chunks,
    iter_pf_tiles,
    max_free_elems,
    prefix_row_segments,
    row_major_strides,
    unravel,
)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


def lower_reshape(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: DType = DType.F32,
) -> Graph:
    """Lower reshape into a standalone graph (thin wrapper over ``emit_reshape``)."""
    if prod(in_shape) != prod(out_shape):
        raise ValueError(
            f"reshape: element count mismatch {prod(in_shape)} vs {prod(out_shape)}"
        )

    b = Builder("reshape")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)
    emit_reshape(b, x_hbm, y_hbm, in_shape, out_shape, dtype)
    b.set_outputs({"y": y_hbm})
    return b.graph


def _largest_common_prefix(in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> int:
    """Largest leading prefix-product common to both shapes (excluding the last
    dim of each, which becomes the per-row free dimension).

    e.g. (1152,2,16,16,3) and (1152,1536) share prefix product 1152; the
    remaining free dims are 2*16*16*3=1536 and 1536, which match by construction
    (total element counts are equal).
    """
    def prefixes(shape):
        out = {1}
        p = 1
        for s in shape[:-1]:
            p *= s
            out.add(p)
        return out

    common = prefixes(in_shape) & prefixes(out_shape)
    return max(common)


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def lower_slice(
    in_shape: tuple[int, ...],
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower slice into a standalone graph (thin wrapper over ``emit_slice``).

    Extracts elements from in_shape[starts[i]:stops[i]:strides[i]] per dim.
    """
    rank = len(in_shape)
    if len(starts) != rank or len(stops) != rank:
        raise ValueError("starts/stops must match input rank")
    if strides is None:
        strides = (1,) * rank

    out_shape = tuple(
        ceildiv(stop - start, stride)
        for start, stop, stride in zip(starts, stops, strides)
    )
    for i, s in enumerate(out_shape):
        if s <= 0:
            raise ValueError(f"empty slice on axis {i}")

    b = Builder("slice")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)
    emit_slice(b, x_hbm, y_hbm, in_shape, out_shape, starts, dtype,
               strides=strides)
    b.set_outputs({"y": y_hbm})
    return b.graph


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


def lower_concat(
    input_shapes: list[tuple[int, ...]],
    axis: int,
    dtype: DType = DType.F32,
) -> Graph:
    """Lower concat into a standalone graph (thin wrapper over ``emit_concat``).

    All inputs must have the same shape except on the concat axis.
    """
    if len(input_shapes) < 2:
        raise ValueError("concat needs at least 2 inputs")

    rank = len(input_shapes[0])
    for s in input_shapes:
        if len(s) != rank:
            raise ValueError("all inputs must have the same rank")
    if axis < 0:
        axis += rank

    # Validate non-concat dims match
    for i in range(rank):
        if i == axis:
            continue
        ref = input_shapes[0][i]
        for s in input_shapes[1:]:
            if s[i] != ref:
                raise ValueError(
                    f"shape mismatch on non-concat axis {i}: {ref} vs {s[i]}"
                )

    out_shape = list(input_shapes[0])
    out_shape[axis] = sum(s[axis] for s in input_shapes)
    out_shape = tuple(out_shape)

    b = Builder("concat")
    x_hbms = [b.add_input(f"x{i}", s, dtype) for i, s in enumerate(input_shapes)]
    y_hbm = b.add_input("y", out_shape, dtype)
    emit_concat(b, x_hbms, y_hbm, input_shapes, axis, dtype)
    b.set_outputs({"y": y_hbm})
    return b.graph


def _emit_concat_input(
    b: Builder,
    x_hbm,
    y_hbm,
    inp_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    axis: int,
    concat_offset: int,
    dtype: DType,
) -> None:
    """Emit tiled DMA copies for one concat input into the output (generic
    rank<=2 / non-collapsible fallback used by ``emit_concat``)."""
    rank = len(inp_shape)
    tile_p = min(inp_shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = inp_shape[-1] if rank >= 2 else inp_shape[0]
    n_p_tiles = ceildiv(inp_shape[-2], tile_p) if rank >= 2 else 1

    batch_dims = list(inp_shape[:-2]) if rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    for batch_flat in range(n_batch):
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, inp_shape[-2] - p_off) if rank >= 2 else tile_p

            # Source slices (from the input tensor)
            src_slices = []
            for i in range(rank):
                if rank > 2 and i < rank - 2:
                    src_slices.append(DimSlice(batch_idx[i], 1))
                elif i == rank - 2:
                    src_slices.append(DimSlice(p_off, p_size))
                else:
                    src_slices.append(DimSlice(0, tile_f))

            # Destination slices (into the output tensor, shifted by concat_offset)
            dst_slices = []
            for i in range(rank):
                if rank > 2 and i < rank - 2:
                    offset = batch_idx[i] + (concat_offset if i == axis else 0)
                    dst_slices.append(DimSlice(offset, 1))
                elif i == rank - 2:
                    offset = p_off + (concat_offset if i == axis else 0)
                    dst_slices.append(DimSlice(offset, p_size))
                else:  # i == rank - 1
                    offset = concat_offset if i == axis else 0
                    dst_slices.append(DimSlice(offset, tile_f))

            tile = b.dma_copy(
                b.alloc((p_size, tile_f), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            b.dma_copy(y_hbm, tile, dst_slices)


# ---------------------------------------------------------------------------
# Emit functions for use by the orchestrator
# ---------------------------------------------------------------------------


def emit_reshape(nb: Builder, x_hbm, y_hbm, in_shape, out_shape, dtype) -> None:
    """Emit reshape tiling into an existing Builder."""
    if len(out_shape) == 0 or len(in_shape) == 0:
        _emit_reshape_scalar(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)
        return

    # When a path's natural tile would be wider than one SBUF partition can
    # hold (e.g. a vocab-sized [1, 128256] row), fall back to a bounded flat
    # copy — always correct since a reshape preserves row-major flat order.
    cap = max_free_elems(dtype)

    # Prefix path: when shapes share a leading prefix-product, emit one
    # load + view + store per partition tile. Handles (Co,*K,Ci)->(Co,K*Ci),
    # trailing-1 additions, and single-partition-row reshapes.
    p_common = _largest_common_prefix(in_shape, out_shape)
    if p_common > 1 or (p_common == 1 and in_shape[0] == 1 and out_shape[0] == 1):
        if prod(in_shape) // p_common > cap:
            _emit_reshape_flat_copy(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)
        else:
            _emit_reshape_via_prefix(nb, x_hbm, y_hbm, in_shape, out_shape, p_common, dtype)
    elif in_shape[-1] == out_shape[-1]:
        if in_shape[-1] > cap:
            _emit_reshape_flat_copy(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)
        else:
            _emit_reshape_same_f(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)
    else:
        if in_shape[-1] > cap or out_shape[-1] > cap:
            _emit_reshape_flat_copy(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)
        else:
            _emit_reshape_diff_f(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)


def _emit_reshape_flat_copy(nb, x_hbm, y_hbm, in_shape, out_shape, dtype):
    """Reshape via a bounded flat copy, used when a row is too wide for SBUF.

    A reshape preserves row-major flat order, so element ``i`` of the source
    maps to element ``i`` of the output. We walk the flat range in chunks no
    wider than one SBUF partition can hold, and split each chunk into segments
    that are a single rectangle on *both* the source and destination (a chunk
    may straddle a leading-dim boundary of either shape). Each segment is one
    (1, seg) load + store — no on-chip reshape needed since flat positions
    already coincide.
    """
    total = prod(in_shape)
    cap = max_free_elems(dtype)
    in_strides = row_major_strides(in_shape)
    out_strides = row_major_strides(out_shape)

    pos = 0
    while pos < total:
        budget = min(cap, total - pos)
        # First chunk's `covered` is the largest single rectangle starting at
        # `pos` within `budget` elements, on each side independently. Taking
        # the smaller makes the segment one rectangle on *both* sides.
        src_first = flat_range_to_src_chunks(pos, budget, in_shape, in_strides)[0]
        dst_first = flat_range_to_src_chunks(pos, budget, out_shape, out_strides)[0]
        seg = min(src_first[1], dst_first[1])
        (src_slices, _), = flat_range_to_src_chunks(pos, seg, in_shape, in_strides)
        (dst_slices, _), = flat_range_to_src_chunks(pos, seg, out_shape, out_strides)
        tile = nb.dma_copy(
            nb.alloc((1, seg), dtype, MemorySpace.SBUF), x_hbm, src_slices)
        nb.dma_copy(y_hbm, tile, dst_slices)
        pos += seg


def _emit_reshape_via_prefix(nb, x_hbm, y_hbm, in_shape, out_shape, p_common, dtype):
    """Emit prefix-path reshape: partition fixed, free-dim regrouped via view."""
    total = prod(in_shape)
    in_free = total // p_common
    out_free = total // p_common

    in_strides = row_major_strides(in_shape)
    out_strides = row_major_strides(out_shape)

    for r0 in range(0, p_common, PARTITION_MAX):
        p = min(PARTITION_MAX, p_common - r0)
        # A tile of p rows may straddle leading-dim boundaries of either shape,
        # in which case it is not a single rectangle. Split it into segments
        # that are one rectangle on both sides and emit a load+view+store each.
        for rows, src_slices, dst_slices in prefix_row_segments(
            r0, p, in_free, in_shape, in_strides, out_shape, out_strides):
            tile = nb.dma_copy(
                nb.alloc((rows, in_free), dtype, MemorySpace.SBUF), x_hbm, src_slices)
            tile = nb.view(tile, (rows, out_free))
            nb.dma_copy(y_hbm, tile, dst_slices)


def _emit_reshape_scalar(nb, x_hbm, y_hbm, in_shape, out_shape, dtype):
    """Handle reshape to/from scalar (rank-0) tensors.

    HBM buffers may have been promoted from () to (1,) by the lowering,
    so use the actual HBM tensor rank for slice construction.
    """
    src_rank = len(x_hbm.type.shape)
    dst_rank = len(y_hbm.type.shape)
    src_slices = [DimSlice(0, 1)] * src_rank
    dst_slices = [DimSlice(0, 1)] * dst_rank
    tile = nb.dma_copy(nb.alloc((1, 1), dtype, MemorySpace.SBUF), x_hbm, src_slices)
    nb.dma_copy(y_hbm, tile, dst_slices)


def _emit_reshape_same_f(nb, x_hbm, y_hbm, in_shape, out_shape, dtype):
    out_rank = len(out_shape)
    tile_f = out_shape[-1]
    p_extent = out_shape[-2] if out_rank >= 2 else 1
    tile_p = min(p_extent, PARTITION_MAX)
    batch_dims = list(out_shape[:-2]) if out_rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    out_strides = row_major_strides(out_shape)
    in_strides = row_major_strides(in_shape)

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)

            flat_offset = sum(bi * out_strides[i] for i, bi in enumerate(batch_idx))
            if out_rank >= 2:
                flat_offset += p_off * out_strides[-2]

            # The tile's flat range may cross a source leading-dim boundary, so
            # split it into maximal rectangles (one chunk for the aligned fast
            # path). Each chunk is a whole number of rows mapping 1:1 to
            # consecutive output rows.
            chunks = flat_range_to_src_chunks(
                flat_offset, p_size * tile_f, in_shape, in_strides
            )
            row_cursor = 0
            for src_slices, covered in chunks:
                chunk_rows = covered // tile_f
                dst_slices = []
                for bi in batch_idx:
                    dst_slices.append(DimSlice(bi, 1))
                if out_rank >= 2:
                    dst_slices.append(DimSlice(p_off + row_cursor, chunk_rows))
                dst_slices.append(DimSlice(0, tile_f))

                tile = nb.dma_copy(
                    nb.alloc((chunk_rows, tile_f), dtype, MemorySpace.SBUF),
                    x_hbm, src_slices,
                )
                nb.dma_copy(y_hbm, tile, dst_slices)
                row_cursor += chunk_rows


def _emit_reshape_diff_f(nb, x_hbm, y_hbm, in_shape, out_shape, dtype):
    """Reshape with different last dim via scratch buffer."""
    total = prod(in_shape)
    out_f = out_shape[-1]
    out_rank = len(out_shape)

    # For 1D inputs or inputs whose last dim exceeds SBUF capacity,
    # re-interpret the source as 2D with a bounded row width.
    MAX_F_BYTES = 128 * 1024  # conservative SBUF tile limit
    ELEM_BYTES = 4  # f32
    max_f = MAX_F_BYTES // ELEM_BYTES

    in_f = in_shape[-1]
    if len(in_shape) == 1 and in_f > max_f:
        # Re-interpret as 2D: (total/max_f, max_f) — pick a row width
        # that divides the total evenly and is <= max_f
        row_width = out_f if out_f <= max_f else max_f
        while total % row_width != 0:
            row_width -= 1
        effective_in_shape = (total // row_width, row_width)
    else:
        effective_in_shape = in_shape
        row_width = in_f

    scratch_shape = (total // row_width, row_width)
    scratch_hbm = nb.alloc(scratch_shape, dtype, MemorySpace.HBM)

    # Phase 1: copy source into scratch
    eff_f = effective_in_shape[-1]
    eff_p = effective_in_shape[-2] if len(effective_in_shape) >= 2 else 1
    eff_batch_dims = list(effective_in_shape[:-2]) if len(effective_in_shape) > 2 else []
    eff_n_batch = prod(eff_batch_dims) if eff_batch_dims else 1
    tile_p_in = min(eff_p, PARTITION_MAX)
    row_offset = 0
    for bf in range(eff_n_batch):
        batch_idx = unravel(bf, eff_batch_dims) if eff_batch_dims else ()
        for p_i in range(ceildiv(eff_p, tile_p_in)):
            p_off = p_i * tile_p_in
            p_size = min(tile_p_in, eff_p - p_off)
            # Source slices use the original shape
            src_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if len(in_shape) == 1:
                # 1D source: use flat offset into the single dim
                flat_off = row_offset * eff_f
                src_slices = [DimSlice(flat_off, p_size * eff_f)]
            else:
                if len(in_shape) >= 2:
                    src_slices.append(DimSlice(p_off, p_size))
                src_slices.append(DimSlice(0, eff_f))
            tile = nb.dma_copy(nb.alloc((p_size, eff_f), dtype, MemorySpace.SBUF), x_hbm, src_slices)
            nb.dma_copy(scratch_hbm, tile, [DimSlice(row_offset, p_size), DimSlice(0, eff_f)])
            row_offset += p_size

    # Phase 2: reload from scratch per output row.
    # scratch_shape = (total // row_width, row_width)
    scratch_f = row_width
    out_p = out_shape[-2] if out_rank >= 2 else 1
    out_batch_dims = list(out_shape[:-2]) if out_rank > 2 else []
    out_n_batch = prod(out_batch_dims) if out_batch_dims else 1
    out_strides = row_major_strides(out_shape)

    for bf in range(out_n_batch):
        batch_idx = unravel(bf, out_batch_dims) if out_batch_dims else ()
        for p_i in range(out_p):
            flat_offset = sum(bi * out_strides[i] for i, bi in enumerate(batch_idx))
            if out_rank >= 2:
                flat_offset += p_i * out_strides[-2]

            scratch_row = flat_offset // scratch_f
            scratch_col = flat_offset % scratch_f

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if out_rank >= 2:
                dst_slices.append(DimSlice(p_i, 1))
            dst_slices.append(DimSlice(0, out_f))

            if scratch_col == 0 and out_f <= scratch_f:
                s_sl = [DimSlice(scratch_row, 1), DimSlice(0, out_f)]
                tile = nb.dma_copy(nb.alloc((1, out_f), dtype, MemorySpace.SBUF), scratch_hbm, s_sl)
                nb.dma_copy(y_hbm, tile, dst_slices)
            elif scratch_col + out_f <= scratch_f:
                s_sl = [DimSlice(scratch_row, 1), DimSlice(scratch_col, out_f)]
                tile = nb.dma_copy(nb.alloc((1, out_f), dtype, MemorySpace.SBUF), scratch_hbm, s_sl)
                nb.dma_copy(y_hbm, tile, dst_slices)
            else:
                remaining = out_f
                out_col = 0
                cur_row, cur_col = scratch_row, scratch_col
                while remaining > 0:
                    chunk = min(remaining, scratch_f - cur_col)
                    s_sl = [DimSlice(cur_row, 1), DimSlice(cur_col, chunk)]
                    tile = nb.dma_copy(nb.alloc((1, chunk), dtype, MemorySpace.SBUF), scratch_hbm, s_sl)
                    d_sl = [DimSlice(bi, 1) for bi in batch_idx]
                    if out_rank >= 2:
                        d_sl.append(DimSlice(p_i, 1))
                    d_sl.append(DimSlice(out_col, chunk))
                    nb.dma_copy(y_hbm, tile, d_sl)
                    remaining -= chunk
                    out_col += chunk
                    cur_row += 1
                    cur_col = 0


def _emit_2d_window_copy(
    nb: Builder, src_2d, dst_2d, src_f_off: int, dst_f_off: int, f_width: int,
    dtype,
) -> None:
    """Copy a full-height column window ``[f_off, f_off+f_width)`` from a 2D HBM
    source to a 2D HBM destination, tiling the partition at 128 and the free
    dim at the per-partition SBUF budget. One load + store per (P, F) tile.

    Used by the collapsed slice/concat fast paths: both reduce to "copy a
    contiguous last-axis window across all the (collapsed) leading rows".
    """
    P = src_2d.type.shape[0]
    for p_off, p_size, fo, fw in iter_pf_tiles(P, f_width, dtype):
        tile = nb.dma_copy(
            nb.alloc((p_size, fw), dtype, MemorySpace.SBUF),
            src_2d, (DimSlice(p_off, p_size), DimSlice(src_f_off + fo, fw)),
        )
        nb.dma_copy(
            dst_2d, tile, (DimSlice(p_off, p_size), DimSlice(dst_f_off + fo, fw))
        )


def _emit_2d_rows_copy(
    nb: Builder, src_2d, dst_2d, src_r_off: int, dst_r_off: int, n_rows: int,
    width: int, dtype,
) -> None:
    """Copy a full-width row window ``[r_off, r_off+n_rows)`` from a 2D HBM
    source to a 2D HBM destination, tiling the partition (row axis) at 128 and
    the free dim at the per-partition SBUF budget. One load + store per tile.

    The counterpart to ``_emit_2d_window_copy``: that windows the free (column)
    axis; this windows the partition (row) axis. Used by the non-last-axis
    slice/concat fast paths, where the operated axis folds onto the partition so
    the copy packs all 128 lanes instead of unrolling the leading rows one at a
    time.
    """
    for p_off, p_size, fo, fw in iter_pf_tiles(n_rows, width, dtype):
        tile = nb.dma_copy(
            nb.alloc((p_size, fw), dtype, MemorySpace.SBUF),
            src_2d, (DimSlice(src_r_off + p_off, p_size), DimSlice(fo, fw)),
        )
        nb.dma_copy(
            dst_2d, tile, (DimSlice(dst_r_off + p_off, p_size), DimSlice(fo, fw))
        )


def _emit_axis_window_copy(
    nb, src_hbm, dst_hbm, in_shape, out_shape, axis, src_axis_off, dst_axis_off,
    axis_extent, dtype,
) -> None:
    """Copy a window along a non-last ``axis``, all other axes kept full.

    Handles the ``cache[:, off:off+n] = x`` / concat-on-a-leading-axis pattern.
    Folds the tensor into a 3D ``(L, A, T)`` grouping — ``L = prod(shape[:axis])``
    (axes before the operated one), ``A`` = that axis's extent, ``T =
    prod(shape[axis+1:])`` (everything after, incl. the last dim) — then, per
    outer index, views the ``(A, T)`` block as 2D and copies a row window that
    packs the partition. When ``L == 1`` (the common KV-cache / batch-1 case)
    this is a single partition-tiled row-window copy instead of one DMA per
    leading row.
    """
    L = prod(in_shape[:axis]) if axis > 0 else 1
    T = prod(in_shape[axis + 1:]) if axis + 1 < len(in_shape) else 1
    A_in = in_shape[axis]
    A_out = out_shape[axis]
    src_2d = collapse_view(nb, src_hbm, L * A_in, T)
    dst_2d = collapse_view(nb, dst_hbm, L * A_out, T)
    for outer in range(L):
        _emit_2d_rows_copy(
            nb, src_2d, dst_2d,
            outer * A_in + src_axis_off, outer * A_out + dst_axis_off,
            axis_extent, T, dtype,
        )


def emit_slice(nb: Builder, x_hbm, y_hbm, in_shape, out_shape, starts, dtype,
               strides=None) -> None:
    """Emit slice tiling into an existing Builder."""
    rank = len(in_shape)
    if strides is None:
        strides = (1,) * rank

    has_non_unit_stride = any(s != 1 for s in strides)
    if has_non_unit_stride:
        _emit_strided_slice(nb, x_hbm, y_hbm, in_shape, out_shape, starts, strides, dtype)
        return

    # Fast path: a rank>=3 slice that touches only the last axis (all leading
    # dims kept in full) collapses to a 2D last-axis window. The leading dims
    # fold onto the partition via a zero-copy view, so a single 2D tiled loop
    # packs all 128 lanes instead of unrolling prod(out[:-2]) one-row copies.
    if (rank >= 3
            and all(starts[i] == 0 and out_shape[i] == in_shape[i] for i in range(rank - 1))):
        lead = prod(in_shape[:-1])
        src_2d = collapse_view(nb, x_hbm, lead, in_shape[-1])
        dst_2d = collapse_view(nb, y_hbm, lead, out_shape[-1])
        _emit_2d_window_copy(
            nb, src_2d, dst_2d, starts[-1], 0, out_shape[-1], dtype)
        return

    # Fast path: a rank>=3 slice that touches only ONE non-last axis (every
    # other axis, incl. the last, kept full) folds the operated axis onto the
    # partition. This is the read side of a KV-cache update — slicing
    # (1,4096,1,128) on axis 1 otherwise tiles shape[-2]=1 lane and unrolls the
    # 4096 rows one at a time.
    if rank >= 3:
        cut = [i for i in range(rank)
               if not (starts[i] == 0 and out_shape[i] == in_shape[i])]
        if len(cut) == 1 and cut[0] < rank - 1:
            a = cut[0]
            _emit_axis_window_copy(
                nb, x_hbm, y_hbm, in_shape, out_shape, a,
                starts[a], 0, out_shape[a], dtype)
            return

    tile_p = min(out_shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = out_shape[-1] if rank >= 2 else out_shape[0]
    p_extent = out_shape[-2] if rank >= 2 else 1
    batch_dims = list(out_shape[:-2]) if rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)

            src_slices = []
            for i in range(rank):
                if rank > 2 and i < rank - 2:
                    src_slices.append(DimSlice(starts[i] + batch_idx[i], 1))
                elif i == rank - 2:
                    src_slices.append(DimSlice(starts[i] + p_off, p_size))
                else:
                    src_slices.append(DimSlice(starts[i], tile_f))

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if rank >= 2:
                dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, tile_f))

            tile = nb.dma_copy(nb.alloc((p_size, tile_f), dtype, MemorySpace.SBUF), x_hbm, src_slices)
            nb.dma_copy(y_hbm, tile, dst_slices)


def _emit_strided_slice(nb, x_hbm, y_hbm, in_shape, out_shape, starts, strides, dtype):
    """Emit a strided slice as tiled strided-DMA descriptors.

    A strided slice reads every ``stride``-th element along each axis. The DMA
    engine expresses this natively via per-dimension ``DimSlice`` strides, so
    we tile the output like the contiguous slice path (P at ``min(out_p, 128)``,
    F full) and emit a single strided load + contiguous store per tile. The
    earlier implementation copied one element at a time when the free-dim
    stride was non-unit, which produced ``O(num_elements)`` DMAs (e.g. ~9.4k
    ops for a single strided conv im2col slice).
    """
    rank = len(in_shape)

    # Rank 1: one strided 1D load into a (1, out_f) tile.
    if rank == 1:
        out_f = out_shape[0]
        src_slices = [DimSlice(starts[0], out_f, stride=strides[0])]
        dst_slices = [DimSlice(0, out_f)]
        tile = nb.dma_copy(
            nb.alloc((1, out_f), dtype, MemorySpace.SBUF), x_hbm, src_slices)
        nb.dma_copy(y_hbm, tile, dst_slices)
        return

    # Rank >= 2: tile the output P-dim; load each tile with strided source
    # descriptors on the P and F axes (and constant batch indices).
    p_stride = strides[-2]
    f_stride = strides[-1]
    out_p = out_shape[-2]
    out_f = out_shape[-1]
    tile_p = min(out_p, PARTITION_MAX)
    batch_dims = list(out_shape[:-2]) if rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1
    batch_strides = strides[:-2] if rank > 2 else ()

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(out_p, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, out_p - p_off)

            src_slices = []
            for i, bi in enumerate(batch_idx):
                src_slices.append(DimSlice(starts[i] + bi * batch_strides[i], 1))
            src_slices.append(
                DimSlice(starts[-2] + p_off * p_stride, p_size, stride=p_stride))
            src_slices.append(DimSlice(starts[-1], out_f, stride=f_stride))

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, out_f))

            tile = nb.dma_copy(
                nb.alloc((p_size, out_f), dtype, MemorySpace.SBUF), x_hbm, src_slices)
            nb.dma_copy(y_hbm, tile, dst_slices)


def emit_concat(nb: Builder, input_hbms: list, y_hbm, input_shapes: list, axis: int,
                dtype) -> None:
    """Emit concat tiling into an existing Builder."""
    rank = len(input_shapes[0])
    if axis < 0:
        axis += rank
    out_shape = list(input_shapes[0])
    out_shape[axis] = sum(s[axis] for s in input_shapes)
    out_shape = tuple(out_shape)

    # Fast path: a rank>=3 concat on the last axis (all non-axis dims match by
    # definition) collapses to a 2D last-axis assembly. Each input becomes a
    # column window of the 2D output; leading dims fold onto the partition via
    # zero-copy views, so each input is one 2D tiled loop over all 128 lanes
    # instead of unrolling prod(shape[:-2]) one-row copies.
    if rank >= 3 and axis == rank - 1:
        lead = prod(out_shape[:-1])
        dst_2d = collapse_view(nb, y_hbm, lead, out_shape[-1])
        dst_f_off = 0
        for inp_idx, inp_shape in enumerate(input_shapes):
            w = inp_shape[-1]
            src_2d = collapse_view(nb, input_hbms[inp_idx], lead, w)
            _emit_2d_window_copy(nb, src_2d, dst_2d, 0, dst_f_off, w, dtype)
            dst_f_off += w
        return

    # Fast path: a rank>=3 concat on a non-last axis folds that axis onto the
    # partition (all other axes match by definition). Each input becomes a row
    # window of the (L, A_out, T) output. This is the write side of a KV-cache
    # update — concat on axis 1 of (1,4096,1,128) otherwise tiles shape[-2]=1
    # lane and unrolls the 4096 rows one at a time.
    if rank >= 3 and axis < rank - 1:
        axis_offset = 0
        for inp_idx, inp_shape in enumerate(input_shapes):
            _emit_axis_window_copy(
                nb, input_hbms[inp_idx], y_hbm, inp_shape, out_shape, axis,
                0, axis_offset, inp_shape[axis], dtype)
            axis_offset += inp_shape[axis]
        return

    concat_offset = 0
    for inp_idx, inp_shape in enumerate(input_shapes):
        _emit_concat_input(nb, input_hbms[inp_idx], y_hbm, inp_shape, out_shape,
                           axis, concat_offset, dtype)
        concat_offset += inp_shape[axis]
