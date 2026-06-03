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
    build_out_slices,
    ceildiv,
    flat_range_to_src_slices,
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
    """Lower reshape as tiled DMA copies with linearized index remapping.

    Both shapes must have the same total element count. Since both are
    row-major in HBM, we iterate per output row (the F-dim), compute the
    flat offset of that row, and express it as a source coordinate for DMA.

    Tiling: output P-dim tiled at min(out[-2], 128). Each row of the output
    tile maps to a contiguous range in the source (of length out[-1]),
    which we express as source coordinates per row.
    """
    if prod(in_shape) != prod(out_shape):
        raise ValueError(
            f"reshape: element count mismatch {prod(in_shape)} vs {prod(out_shape)}"
        )

    out_rank = len(out_shape)
    in_rank = len(in_shape)

    # If the last dim matches, we can load multi-row tiles directly
    if in_shape[-1] == out_shape[-1]:
        return _lower_reshape_same_last_dim(in_shape, out_shape, dtype)

    # General case: use an HBM scratch buffer. Load source rows into scratch
    # in flat order, then reload from scratch in output shape. Since both
    # shapes describe the same flat data in row-major order, the scratch
    # buffer (treated as flat) bridges the two interpretations.
    return _lower_reshape_via_scratch(in_shape, out_shape, dtype)


def _lower_reshape_via_scratch(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: DType,
) -> Graph:
    """Reshape when inner dims differ, using a flat HBM scratch buffer.

    Strategy: copy the entire source into a 1D scratch buffer (preserving
    flat order), then reload from scratch using output coordinates. Both
    source and output are row-major views of the same flat data, so the
    scratch buffer (shaped as (total_rows, max_F)) bridges between them.

    We use a scratch with shape (N, F) where F = in_F, then reload with
    output's coordinate mapping.
    """
    total = prod(in_shape)
    out_rank = len(out_shape)
    in_f = in_shape[-1]
    out_f = out_shape[-1]

    # Scratch: 2D with the source's row width, flattened leading dims
    total_rows_in = total // in_f
    scratch_shape = (total_rows_in, in_f)

    # Output iteration
    total_rows_out = total // out_f
    out_p_extent = out_shape[-2] if out_rank >= 2 else 1
    batch_dims = list(out_shape[:-2]) if out_rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    b = Builder("reshape")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)
    scratch_hbm = b.add_input("scratch", scratch_shape, dtype)

    # Phase 1: copy source into scratch (same row width, just flatten leading dims)
    in_p_extent = in_shape[-2] if len(in_shape) >= 2 else 1
    in_batch_dims = list(in_shape[:-2]) if len(in_shape) > 2 else []
    in_n_batch = prod(in_batch_dims) if in_batch_dims else 1
    tile_p_in = min(in_p_extent, PARTITION_MAX)
    n_p_tiles_in = ceildiv(in_p_extent, tile_p_in)

    row_offset = 0
    for batch_flat in range(in_n_batch):
        batch_idx = unravel(batch_flat, in_batch_dims) if in_batch_dims else ()
        for p_i in range(n_p_tiles_in):
            p_off = p_i * tile_p_in
            p_size = min(tile_p_in, in_p_extent - p_off)

            src_slices = []
            for bi in batch_idx:
                src_slices.append(DimSlice(bi, 1))
            if len(in_shape) >= 2:
                src_slices.append(DimSlice(p_off, p_size))
            src_slices.append(DimSlice(0, in_f))

            scratch_slices = [DimSlice(row_offset, p_size), DimSlice(0, in_f)]

            tile = b.dma_copy(
                b.alloc((p_size, in_f), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            b.dma_copy(scratch_hbm, tile, scratch_slices)
            row_offset += p_size

    # Phase 2: reload from scratch using output coordinates.
    # Each output row of out_f elements maps to a contiguous range in scratch.
    # If out_f <= in_f and aligned, one load suffices. If out_f > in_f, the
    # output row spans multiple scratch rows — copy each chunk separately.
    out_strides = row_major_strides(out_shape)
    for batch_flat in range(n_batch):
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for p_i in range(out_p_extent):
            flat_offset = 0
            for i, bi in enumerate(batch_idx):
                flat_offset += bi * out_strides[i]
            if out_rank >= 2:
                flat_offset += p_i * out_strides[-2]

            row_flat = flat_offset
            scratch_row = row_flat // in_f
            scratch_col = row_flat % in_f

            if scratch_col == 0 and out_f <= in_f:
                scratch_slices = [DimSlice(scratch_row, 1), DimSlice(0, out_f)]
                tile = b.dma_copy(
                    b.alloc((1, out_f), dtype, MemorySpace.SBUF),
                    scratch_hbm, scratch_slices,
                )
                dst_slices = build_out_slices(batch_idx, p_i, 1, out_f, out_rank)
                b.dma_copy(y_hbm, tile, dst_slices)
            elif scratch_col + out_f <= in_f:
                scratch_slices = [DimSlice(scratch_row, 1), DimSlice(scratch_col, out_f)]
                tile = b.dma_copy(
                    b.alloc((1, out_f), dtype, MemorySpace.SBUF),
                    scratch_hbm, scratch_slices,
                )
                dst_slices = build_out_slices(batch_idx, p_i, 1, out_f, out_rank)
                b.dma_copy(y_hbm, tile, dst_slices)
            else:
                # Output row spans multiple scratch rows — copy chunk by chunk
                remaining = out_f
                out_col = 0
                cur_row = scratch_row
                cur_col = scratch_col
                while remaining > 0:
                    chunk = min(remaining, in_f - cur_col)
                    scratch_slices = [DimSlice(cur_row, 1), DimSlice(cur_col, chunk)]
                    tile = b.dma_copy(
                        b.alloc((1, chunk), dtype, MemorySpace.SBUF),
                        scratch_hbm, scratch_slices,
                    )
                    dst_slices = []
                    for bi in batch_idx:
                        dst_slices.append(DimSlice(bi, 1))
                    if out_rank >= 2:
                        dst_slices.append(DimSlice(p_i, 1))
                    dst_slices.append(DimSlice(out_col, chunk))
                    b.dma_copy(y_hbm, tile, dst_slices)
                    remaining -= chunk
                    out_col += chunk
                    cur_row += 1
                    cur_col = 0

    b.set_outputs({"y": y_hbm})
    return b.graph


def _lower_reshape_same_last_dim(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: DType,
) -> Graph:
    """Optimized reshape when the last dim is unchanged (common case).

    When in_shape[-1] == out_shape[-1], each output row maps directly to a
    source row (just at a different multi-dimensional index). We can load
    multi-row tiles since consecutive output rows are also consecutive in
    the source.
    """
    out_rank = len(out_shape)
    tile_f = out_shape[-1]
    p_extent = out_shape[-2] if out_rank >= 2 else 1
    tile_p = min(p_extent, PARTITION_MAX)
    n_p_tiles = ceildiv(p_extent, tile_p)
    batch_dims = list(out_shape[:-2]) if out_rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    out_strides = row_major_strides(out_shape)
    in_strides = row_major_strides(in_shape)

    b = Builder("reshape")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    for batch_flat in range(n_batch):
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)

            flat_offset = 0
            for i, bi in enumerate(batch_idx):
                flat_offset += bi * out_strides[i]
            if out_rank >= 2:
                flat_offset += p_off * out_strides[-2]

            n_elements = p_size * tile_f
            src_slices = flat_range_to_src_slices(flat_offset, n_elements, in_shape, in_strides)
            dst_slices = build_out_slices(batch_idx, p_off, p_size, tile_f, out_rank)

            tile = b.dma_copy(
                b.alloc((p_size, tile_f), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            b.dma_copy(y_hbm, tile, dst_slices)

    b.set_outputs({"y": y_hbm})
    return b.graph


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
    """Lower slice (sub-tensor extraction) as tiled DMA copies from offsets.

    Extracts elements from in_shape[starts[i]:stops[i]:strides[i]] per dim.
    Only stride=1 is supported (contiguous slices).

    Tiling: output is tiled with P=min(out[-2], 128), F=out[-1] (full).
    """
    rank = len(in_shape)
    if len(starts) != rank or len(stops) != rank:
        raise ValueError("starts/stops must match input rank")
    if strides is None:
        strides = (1,) * rank
    if any(s != 1 for s in strides):
        raise NotImplementedError("only stride=1 is supported")

    out_shape = tuple(stop - start for start, stop in zip(starts, stops))
    for i, s in enumerate(out_shape):
        if s <= 0:
            raise ValueError(f"empty slice on axis {i}")

    out_rank = len(out_shape)
    tile_p = min(out_shape[-2], PARTITION_MAX) if out_rank >= 2 else 1
    tile_f = out_shape[-1] if out_rank >= 2 else out_shape[0]
    n_p_tiles = ceildiv(out_shape[-2], tile_p) if out_rank >= 2 else 1

    batch_dims = list(out_shape[:-2]) if out_rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    b = Builder("slice")
    x_hbm = b.add_input("x", in_shape, dtype)
    y_hbm = b.add_input("y", out_shape, dtype)

    for batch_flat in range(n_batch):
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for p_i in range(n_p_tiles):
            p_off = p_i * tile_p
            p_size = min(tile_p, out_shape[-2] - p_off) if out_rank >= 2 else tile_p

            # Source slices: offset by starts + current tile position
            src_slices = []
            for i in range(rank):
                if out_rank > 2 and i < rank - 2:
                    src_slices.append(DimSlice(starts[i] + batch_idx[i], 1))
                elif i == rank - 2:
                    src_slices.append(DimSlice(starts[i] + p_off, p_size))
                else:  # i == rank - 1
                    src_slices.append(DimSlice(starts[i], tile_f))

            # Destination slices
            dst_slices = build_out_slices(batch_idx, p_off, p_size, tile_f, out_rank)

            tile = b.dma_copy(
                b.alloc((p_size, tile_f), dtype, MemorySpace.SBUF),
                x_hbm, src_slices,
            )
            b.dma_copy(y_hbm, tile, dst_slices)

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
    """Lower concat (tensor assembly along an axis) as tiled DMA copies.

    Each input tensor is copied into the output at the appropriate offset
    along the concat axis. All inputs must have the same shape except on
    the concat axis.

    Tiling: each input is tiled with P=min(shape[-2], 128), F=shape[-1].
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

    # Compute output shape
    out_shape = list(input_shapes[0])
    out_shape[axis] = sum(s[axis] for s in input_shapes)
    out_shape = tuple(out_shape)

    b = Builder("concat")
    x_hbms = [b.add_input(f"x{i}", s, dtype) for i, s in enumerate(input_shapes)]
    y_hbm = b.add_input("y", out_shape, dtype)

    # Copy each input into the output at increasing offsets along concat axis
    concat_offset = 0
    for inp_idx, inp_shape in enumerate(input_shapes):
        _emit_concat_input(b, x_hbms[inp_idx], y_hbm, inp_shape, out_shape,
                           axis, concat_offset, dtype)
        concat_offset += inp_shape[axis]

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
    """Emit tiled DMA copies for one concat input into the output."""
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
    if in_shape[-1] == out_shape[-1]:
        _emit_reshape_same_f(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)
    else:
        _emit_reshape_diff_f(nb, x_hbm, y_hbm, in_shape, out_shape, dtype)


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

            src_slices = flat_range_to_src_slices(flat_offset, p_size * tile_f, in_shape, in_strides)
            dst_slices = []
            for bi in batch_idx:
                dst_slices.append(DimSlice(bi, 1))
            if out_rank >= 2:
                dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, tile_f))

            tile = nb.dma_copy(nb.alloc((p_size, tile_f), dtype, MemorySpace.SBUF), x_hbm, src_slices)
            nb.dma_copy(y_hbm, tile, dst_slices)


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


def emit_slice(nb: Builder, x_hbm, y_hbm, in_shape, out_shape, starts, dtype) -> None:
    """Emit slice tiling into an existing Builder."""
    rank = len(in_shape)
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


def emit_concat(nb: Builder, input_hbms: list, y_hbm, input_shapes: list, axis: int, dtype) -> None:
    """Emit concat tiling into an existing Builder."""
    rank = len(input_shapes[0])
    out_shape = list(input_shapes[0])
    out_shape[axis] = sum(s[axis] for s in input_shapes)
    out_shape = tuple(out_shape)

    concat_offset = 0
    for inp_idx, inp_shape in enumerate(input_shapes):
        _emit_concat_input(nb, input_hbms[inp_idx], y_hbm, inp_shape, out_shape,
                           axis, concat_offset, dtype)
        concat_offset += inp_shape[axis]
