"""Orchestrated direct lowering: tensor IR → NKI IR with HBM boundaries.

Lowers a complete tensor IR graph (after canonicalize + decompose) to a single
NKI IR graph. Consecutive elementwise ops are grouped and lowered together
(intermediates stay on-chip); all other ops (reduce, matmul, transpose,
reshape, slice, concat, broadcast_to) get their own load→compute→store
sequence with HBM boundaries.

Usage:
    graph = build_some_pattern(...)
    canonicalize(graph)
    decompose(graph)
    layouts = solve_graph(graph)
    nki_graph = lower_graph(graph, layouts)
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import DType, Graph, Value
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
)
from nkigen_lite.nki_ir import ir as nki_ir
from nkigen_lite.nki_ir.insert_deallocs import insert_deallocs
from nkigen_lite.tensor_ir.passes.layout_solver import Layout

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    BINARY_OPS,
    BITWISE_OPS,
    COMPARE_OPS,
    ELEMENTWISE_OPCODES,
    UNARY_OPS,
    ceildiv,
    compute_tile_sizes,
    emit_binary_op,
    emit_unary_op,
    hbm_slices,
    max_free_elems,
    on_chip_shape,
    unravel,
)


# ---------------------------------------------------------------------------
# Graph segmentation
# ---------------------------------------------------------------------------


def _segment_ops(graph: Graph, layouts: dict[str, Layout]) -> list[list]:
    """Segment graph ops into elementwise groups and individual non-elementwise ops.

    Elementwise ops are grouped only if their output layouts are compatible
    (same P/F dim assignment). A layout flip breaks the group.

    Returns a list of segments. Each segment is either:
      - A list of consecutive elementwise ops (grouped)
      - A list with a single non-elementwise op
    """
    segments = []
    current_ew = []
    current_pf = None  # (p_dims, f_dims) of current group

    for op in graph.ops:
        if op.opcode in ELEMENTWISE_OPCODES:
            out_name = op.results[0].name
            if out_name in layouts:
                out_layout = layouts[out_name]
                pf = (out_layout.p_dims, out_layout.f_dims)
            else:
                pf = current_pf

            if current_ew and current_pf is not None and pf != current_pf:
                segments.append(current_ew)
                current_ew = []

            current_ew.append(op)
            current_pf = pf
        else:
            if current_ew:
                segments.append(current_ew)
                current_ew = []
                current_pf = None
            segments.append([op])

    if current_ew:
        segments.append(current_ew)

    return segments




# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def lower_graph(graph: Graph, layouts: dict[str, Layout]) -> nki_ir.Graph:
    """Lower a full tensor IR graph to NKI IR with HBM boundaries."""
    nb = Builder("direct_lower")
    hbm_map: dict[str, Value] = {}

    def _nki_shape(shape):
        """NKI requires at least rank-1 tensors."""
        return shape if len(shape) > 0 else (1,)

    # Allocate HBM inputs
    for v in graph.inputs:
        hbm_map[v.name] = nb.add_input(v.name, _nki_shape(v.type.shape), v.type.dtype)

    # Allocate HBM output buffers
    for out_name, out_val in graph.outputs.items():
        key = f"{out_name}_out"
        if key not in hbm_map:
            hbm_map[key] = nb.add_input(key, _nki_shape(out_val.type.shape), out_val.type.dtype)

    # Allocate HBM intermediates for all op results.  Skip reshape results:
    # a reshape preserves row-major flat order, so its result is emitted as a
    # zero-copy view of the (row-major contiguous) input HBM buffer rather than
    # a fresh allocation + copy.  Allocating here would leave a large dead HBM
    # buffer (an expert weight reshape is ~100 MB).  Scalar reshapes still need
    # a real buffer (() promotion), so don't skip those.
    for op in graph.ops:
        if op.opcode == "reshape" and _is_view_reshape(op):
            continue
        for r in op.results:
            if r.name not in hbm_map:
                hbm_map[r.name] = nb.alloc(
                    _nki_shape(r.type.shape), r.type.dtype, MemorySpace.HBM
                )

    # Segment and lower
    segments = _segment_ops(graph, layouts)
    for segment in segments:
        if segment[0].opcode in ELEMENTWISE_OPCODES:
            # Further split if any input has an incompatible layout
            sub_segments = _split_on_layout_conflict(segment, layouts, hbm_map)
            for sub_seg in sub_segments:
                _emit_elementwise_segment(nb, sub_seg, layouts, hbm_map)
        elif segment[0].opcode == "reduce":
            _emit_reduce_op(nb, segment[0], layouts, hbm_map)
        elif segment[0].opcode == "matmul":
            _emit_matmul_op(nb, segment[0], layouts, hbm_map)
        elif segment[0].opcode == "transpose":
            _emit_transpose_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "reshape":
            _emit_reshape_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "slice":
            _emit_slice_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "concat":
            _emit_concat_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "broadcast_to":
            _emit_broadcast_op(nb, segment[0], layouts, hbm_map)
        elif segment[0].opcode == "iota":
            _emit_iota_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "topk":
            _emit_topk_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "gather_along_axis":
            _emit_gather_along_axis_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "scatter_rows":
            _emit_scatter_rows_op(nb, segment[0], hbm_map)
        elif segment[0].opcode == "gather_rows":
            _emit_gather_rows_op(nb, segment[0], hbm_map)
        elif segment[0].opcode in COLLECTIVE_OPCODES:
            _emit_collective_op(nb, segment[0], hbm_map)
        else:
            raise NotImplementedError(f"Op {segment[0].opcode!r} not supported")

    # Copy final results to output buffers
    for out_name, out_val in graph.outputs.items():
        src = hbm_map[out_val.name]
        dst = hbm_map[f"{out_name}_out"]
        if src is not dst:
            _emit_hbm_copy(nb, src, dst, out_val.type.shape)

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_hbm_copy(nb: Builder, src: Value, dst: Value, shape: tuple[int, ...]):
    """Copy an entire HBM tensor to another HBM tensor, tiled."""
    if len(shape) == 0:
        # HBM buffers may be promoted from () to (1,)
        src_slices = [DimSlice(0, 1)] * len(src.type.shape)
        dst_slices = [DimSlice(0, 1)] * len(dst.type.shape)
        tile = nb.dma_copy(nb.alloc((1, 1), src.type.dtype, MemorySpace.SBUF), src, src_slices)
        nb.dma_copy(dst, tile, dst_slices)
        return
    tile_p = min(shape[-2], PARTITION_MAX) if len(shape) >= 2 else 1
    f_extent = shape[-1] if len(shape) >= 2 else shape[0]
    p_extent = shape[-2] if len(shape) >= 2 else 1
    batch_dims = list(shape[:-2]) if len(shape) > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1
    # Cap the free extent so a (tile_p, tile_f) tile fits one SBUF partition;
    # a vocab-wide row (e.g. [1, 128256]) otherwise blows the per-partition cap.
    tile_f = min(f_extent, max_free_elems(src.type.dtype))

    for batch_flat in range(n_batch):
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)
            for f_i in range(ceildiv(f_extent, tile_f)):
                f_off = f_i * tile_f
                f_size = min(tile_f, f_extent - f_off)
                slices = []
                for bi in batch_idx:
                    slices.append(DimSlice(bi, 1))
                if len(shape) >= 2:
                    slices.append(DimSlice(p_off, p_size))
                slices.append(DimSlice(f_off, f_size))
                tile = nb.dma_copy(
                    nb.alloc((p_size, f_size), src.type.dtype, MemorySpace.SBUF),
                    src, slices,
                )
                nb.dma_copy(dst, tile, slices)


def _split_on_layout_conflict(
    ops: list, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> list[list]:
    """Split an elementwise segment when an input has an incompatible layout.

    An input is incompatible if its P/F dims differ from the segment's rep
    AND its shape (after considering the layout) would produce a tile with
    different (P, F) dimensions that can't be aligned via broadcasting.
    """
    if len(ops) <= 1:
        return [ops]

    rep_layout = layouts[ops[-1].results[0].name]
    segment_results = {r.name for op in ops for r in op.results}

    # Check each op's inputs for layout conflicts
    sub_segments = []
    current = []
    for op in ops:
        has_conflict = False
        for inp in op.inputs:
            if inp.name in segment_results:
                continue
            if inp.name not in layouts:
                continue
            inp_layout = layouts[inp.name]
            if (inp_layout.p_dims != rep_layout.p_dims and
                inp_layout.f_dims != rep_layout.f_dims):
                # Check if it's a broadcast (size-1 dim) — those are OK
                inp_shape = inp.type.shape
                inp_p_ext = prod(inp_shape[d] for d in inp_layout.p_dims) if inp_layout.p_dims else 1
                inp_f_ext = prod(inp_shape[d] for d in inp_layout.f_dims) if inp_layout.f_dims else 1
                if inp_p_ext > 1 and inp_f_ext > 1:
                    has_conflict = True
                    break

        if has_conflict:
            if current:
                sub_segments.append(current)
                current = []
            sub_segments.append([op])
        else:
            current.append(op)

    if current:
        sub_segments.append(current)
    return sub_segments


# ---------------------------------------------------------------------------
# Elementwise segment emission
# ---------------------------------------------------------------------------


def _canonical_layout(rank: int) -> Layout:
    """Return a canonical row-major layout: last dim = F, penultimate = P, rest = I."""
    if rank == 0:
        return Layout(i_dims=(), p_dims=(), f_dims=())
    if rank == 1:
        return Layout(i_dims=(), p_dims=(), f_dims=(0,))
    f_dims = (rank - 1,)
    p_dims = (rank - 2,)
    i_dims = tuple(range(rank - 2))
    return Layout(i_dims=i_dims, p_dims=p_dims, f_dims=f_dims)


def _collapse_ew_shape(shape: tuple[int, ...], rep_shape: tuple[int, ...]):
    """Collapse a value's shape to the 2D ``(P, F)`` form of an elementwise
    segment whose representative output is ``rep_shape``.

    The segment loops a canonical row-major layout: the last axis is the free
    dim and *all* leading axes fold into the partition. When every value in the
    segment shares the rep's free extent (or is free-size-1), the whole op
    reduces to one ``(prod(rep[:-1]), rep[-1])`` loop — packing the 128-lane
    partition instead of iterating the leading dims one element at a time.

    Returns the collapsed ``(P, F)`` for *shape*, or ``None`` if it can't align
    to the rep (different free extent and not free-size-1; or rank mismatch
    that isn't a clean broadcast). ``None`` makes the caller fall back to the
    generic per-tile path.
    """
    if len(rep_shape) < 3:
        return None  # 0/1/2-D already packs the partition; no collapse needed
    rep_P = prod(rep_shape[:-1])
    rep_F = rep_shape[-1]

    if shape == rep_shape:
        return (rep_P, rep_F)
    # Right-align (numpy broadcast rules): a lower-rank operand broadcasts over
    # the rep's leading axes.
    if len(shape) > len(rep_shape):
        return None
    al = (1,) * (len(rep_shape) - len(shape)) + tuple(shape)
    P = prod(al[:-1])
    F = al[-1]
    # Free dim must match the rep (full) or be 1 (per-row scalar broadcast).
    if F != rep_F and F != 1:
        return None
    # Partition must match the rep (full) or be 1 (broadcast across all rows).
    if P != rep_P and P != 1:
        return None
    return (P, F)


def _try_emit_collapsed_ew(
    nb: Builder, ops: list, hbm_map: dict[str, Value], rep_shape: tuple[int, ...],
) -> bool:
    """Emit a rank>=3 elementwise segment via leading-dims-onto-partition
    collapse. Returns False (emitting nothing) when the segment can't be
    cleanly collapsed, so the caller runs the generic per-tile path.

    Every external input and every op result must collapse to the rep's
    ``(P, F)`` (see ``_collapse_ew_shape``). When it does, we reinterpret each
    HBM buffer to its collapsed 2D shape with a zero-copy view, build a 2D
    ``hbm_map`` / ``layouts`` view, and reuse the existing 2D tile machinery
    (``_emit_ew_tile``) — which already handles F/P broadcasting of size-1
    operands via ``emit_binary_op``.
    """
    if len(rep_shape) < 3:
        return False

    segment_results = {r.name for op in ops for r in op.results}

    # Gather every value that needs a collapsed shape: external inputs and all
    # results. Validate each collapses; bail (no emission) on any mismatch.
    values: dict[str, Value] = {}
    for op in ops:
        for inp in op.inputs:
            if inp.name not in segment_results:
                values[inp.name] = inp
        for r in op.results:
            values[r.name] = r

    collapsed: dict[str, tuple[int, int]] = {}
    for name, v in values.items():
        pf = _collapse_ew_shape(v.type.shape, rep_shape)
        if pf is None:
            return False
        collapsed[name] = pf

    # Build 2D HBM views and a parallel hbm_map / layouts keyed by the same
    # names the tile machinery uses.
    view_map: dict[str, Value] = {}
    view_layouts: dict[str, Layout] = {}
    rep_2d = (prod(rep_shape[:-1]), rep_shape[-1])
    for name, pf in collapsed.items():
        hbm_val = hbm_map[name]
        # A genuine reshape (collapse) needs a view; an already-2D buffer of the
        # right shape is used directly.
        if tuple(hbm_val.type.shape) == pf:
            view_map[name] = hbm_val
        else:
            view_map[name] = nb.view(hbm_val, pf)
        view_layouts[name] = _canonical_layout(2)

    rep_layout = _canonical_layout(2)
    tile_sizes = compute_tile_sizes(rep_2d, rep_layout)
    loop_dims = [(d, rep_2d[d], tile_sizes[d])
                 for d in sorted(tile_sizes.keys())
                 if tile_sizes[d] < rep_2d[d]]

    shape_override = {name: pf for name, pf in collapsed.items()}

    def _emit_nested(depth: int, indices: dict[int, int]):
        if depth >= len(loop_dims):
            _emit_ew_tile(nb, ops, view_layouts, view_map, rep_layout, rep_2d,
                          tile_sizes, indices, segment_results, shape_override)
            return
        d, extent, ts = loop_dims[depth]
        for i in range(ceildiv(extent, ts)):
            _emit_nested(depth + 1, {**indices, d: i})

    _emit_nested(0, {})
    return True


def _emit_elementwise_segment(
    nb: Builder, ops: list, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    """Emit a fused elementwise segment: one tiled loop for all ops."""
    # Use a canonical row-major layout for elementwise segments. HBM is
    # layout-agnostic, so all loads/stores address data by logical dimension
    # coordinates — the declared layout of individual values is irrelevant.
    rep_val = ops[-1].results[0]
    rep_shape = rep_val.type.shape

    # Fast path: collapse leading dims onto the partition. A rank>=3 segment
    # otherwise tiles only out[-2] as the partition and unrolls prod(out[:-2])
    # leading-dim iterations one at a time (e.g. (1,128,16,64) uses 16 of 128
    # lanes and unrolls 128 times). Collapsing to (prod(shape[:-1]), shape[-1])
    # packs the partition and runs a single 2D loop.
    if _try_emit_collapsed_ew(nb, ops, hbm_map, rep_shape):
        return

    rep_layout = _canonical_layout(len(rep_shape))
    tile_sizes = compute_tile_sizes(rep_shape, rep_layout)

    # Which values are produced within this segment (stay on-chip)
    segment_results = {r.name for op in ops for r in op.results}

    # Loop dims
    loop_dims = [(d, rep_shape[d], tile_sizes[d])
                 for d in sorted(tile_sizes.keys())
                 if tile_sizes[d] < rep_shape[d]]

    def _emit_nested(depth: int, indices: dict[int, int]):
        if depth >= len(loop_dims):
            _emit_ew_tile(nb, ops, layouts, hbm_map, rep_layout, rep_shape,
                          tile_sizes, indices, segment_results)
            return
        d, extent, ts = loop_dims[depth]
        for i in range(ceildiv(extent, ts)):
            _emit_nested(depth + 1, {**indices, d: i})

    _emit_nested(0, {})


def _emit_ew_tile(
    nb: Builder, ops: list, layouts: dict[str, Layout], hbm_map: dict[str, Value],
    rep_layout: Layout, rep_shape: tuple[int, ...],
    tile_sizes: dict[int, int], indices: dict[int, int],
    segment_results: set[str],
    shape_override: dict[str, tuple[int, ...]] | None = None,
) -> None:
    """Emit one tile of fused elementwise computation.

    ``shape_override`` (collapsed-path only) maps a value name to the shape the
    tile machinery should treat it as, instead of its declared N-D shape. The
    collapsed path reshapes HBM buffers to 2D views but the op results still
    carry their original N-D types, so ``constant`` (which sizes its tile from
    the result type) needs the 2D shape to match the rest of the tile.
    """
    shape_override = shape_override or {}
    tile_map: dict[str, Value] = {}
    rep_tile = on_chip_shape(rep_shape, rep_layout, tile_sizes, indices)

    # Load external inputs — use canonical layout for the input's own rank
    # since HBM is layout-agnostic (row-major).
    for op in ops:
        for inp in op.inputs:
            if inp.name in tile_map or inp.name in segment_results:
                continue
            hbm_val = hbm_map[inp.name]
            val_layout = _canonical_layout(len(hbm_val.type.shape))
            val_tile_sizes = compute_tile_sizes(hbm_val.type.shape, val_layout)
            val_tile = on_chip_shape(hbm_val.type.shape, val_layout, val_tile_sizes, indices)
            slices = hbm_slices(hbm_val.type.shape, val_layout, val_tile_sizes,
                                 indices, rep_layout)
            dst = nb.alloc(val_tile, hbm_val.type.dtype, MemorySpace.SBUF)
            tile_map[inp.name] = nb.dma_copy(dst, hbm_val, slices)

    # Compute
    for op in ops:
        out_name = op.results[0].name
        out_dtype = op.results[0].type.dtype

        if op.opcode in BINARY_OPS or op.opcode in BITWISE_OPS or op.opcode in COMPARE_OPS:
            lhs = tile_map[op.inputs[0].name]
            rhs = tile_map[op.inputs[1].name]
            tile_map[out_name] = emit_binary_op(nb, out_dtype, lhs, rhs, op.opcode)
        elif op.opcode in UNARY_OPS:
            src = tile_map[op.inputs[0].name]
            tile_map[out_name] = emit_unary_op(nb, out_dtype, src, op.opcode)
        elif op.opcode == "cast":
            src = tile_map[op.inputs[0].name]
            dst = nb.alloc(src.type.shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_copy(dst, src)
            tile_map[out_name] = dst
        elif op.opcode == "where":
            cond = tile_map[op.inputs[0].name]
            x_true = tile_map[op.inputs[1].name]
            y_false = tile_map[op.inputs[2].name]
            from nkigen_lite.nki_ir.ir import NisaArithOp
            # NKI pattern: result = cond*x + (1-cond)*y
            # cond is float (1.0/0.0), same dtype as x/y
            shape = x_true.type.shape
            # inv_cond = 1.0 - cond
            ones = nb.constant(1.0, shape, out_dtype, MemorySpace.SBUF)
            inv_cond = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_tensor_arith(inv_cond, ones, cond, NisaArithOp.SUBTRACT)
            # mask_x = cond * x
            mask_x = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_tensor_arith(mask_x, cond, x_true, NisaArithOp.MULTIPLY)
            # mask_y = inv_cond * y
            mask_y = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_tensor_arith(mask_y, inv_cond, y_false, NisaArithOp.MULTIPLY)
            # result = mask_x + mask_y
            result = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_tensor_arith(result, mask_x, mask_y, NisaArithOp.ADD)
            tile_map[out_name] = result
        elif op.opcode == "constant":
            out_shape = shape_override.get(out_name, op.results[0].type.shape)
            const_layout = _canonical_layout(len(out_shape))
            const_tile_sizes = compute_tile_sizes(out_shape, const_layout)
            const_tile = on_chip_shape(out_shape, const_layout, const_tile_sizes, indices)
            tile_map[out_name] = nb.constant(
                op.attrs["value"], const_tile, out_dtype, MemorySpace.SBUF
            )

    # Store results — use canonical layout of the output's own rank
    for op in ops:
        out_name = op.results[0].name
        if out_name in tile_map and out_name in hbm_map:
            hbm_dst = hbm_map[out_name]
            out_layout = _canonical_layout(len(hbm_dst.type.shape))
            out_tile_sizes = compute_tile_sizes(hbm_dst.type.shape, out_layout)
            slices = hbm_slices(hbm_dst.type.shape, out_layout, out_tile_sizes,
                                 indices, rep_layout)
            nb.dma_copy(hbm_dst, tile_map[out_name], slices)




# ---------------------------------------------------------------------------
# Op-specific emission — delegates to standalone modules
# ---------------------------------------------------------------------------


from nkigen_lite.tensor_ir.passes.basic.direct_lower_matmul import emit_matmul
from nkigen_lite.tensor_ir.passes.basic.direct_lower_transpose import emit_transpose
from nkigen_lite.tensor_ir.passes.basic.direct_lower_memory import (
    emit_reshape, emit_slice, emit_concat,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_broadcast import emit_broadcast_to
from nkigen_lite.tensor_ir.passes.basic.direct_lower_reduce import (
    emit_reduce,
)


def _emit_reduce_op(
    nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    emit_reduce(nb, op, layouts, hbm_map)


def _emit_matmul_op(
    nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    a_val, b_val = op.inputs
    c_val = op.results[0]
    emit_matmul(
        nb, hbm_map[a_val.name], hbm_map[b_val.name], hbm_map[c_val.name],
        a_val.type.shape, b_val.type.shape, a_val.type.dtype,
    )


COLLECTIVE_OPCODES = frozenset(
    {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}
)


def _emit_collective_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower a collective op to an nki_ir collective node.

    The compiler forbids collectives from reading/writing kernel IO tensors
    directly, so we stage through internal HBM scratch buffers:
    IO/result HBM -> src scratch -> collective -> dst scratch -> result HBM.
    """
    inp_val = op.inputs[0]
    out_val = op.results[0]
    src_hbm = hbm_map[inp_val.name]
    dst_hbm = hbm_map[out_val.name]

    src_scratch = nb.alloc(src_hbm.type.shape, src_hbm.type.dtype, MemorySpace.HBM)
    dst_scratch = nb.alloc(dst_hbm.type.shape, dst_hbm.type.dtype, MemorySpace.HBM)

    _emit_hbm_copy(nb, src_hbm, src_scratch, inp_val.type.shape)
    nb.collective(op.opcode, dst_scratch, src_scratch, op.attrs)
    _emit_hbm_copy(nb, dst_scratch, dst_hbm, out_val.type.shape)


def _emit_transpose_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_transpose(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, op.attrs["perm"], inp_val.type.dtype,
    )


def _is_view_reshape(op) -> bool:
    """True if a reshape can be lowered as a zero-copy HBM view.

    A reshape preserves row-major flat order, and HBM buffers are row-major
    contiguous, so reinterpreting one to a new shape needs no data movement.
    The only exception is reshapes involving a rank-0 (scalar) shape, which
    the lowering promotes () -> (1,); those keep the explicit copy path so the
    byte sizes the view checks still line up.
    """
    inp_val = op.inputs[0]
    out_val = op.results[0]
    return len(inp_val.type.shape) > 0 and len(out_val.type.shape) > 0


def _emit_reshape_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    if _is_view_reshape(op):
        # Zero-copy: reinterpret the input HBM buffer to the output shape.
        # The result was deliberately not pre-allocated (see lower_graph), so
        # downstream consumers read this view directly.
        hbm_map[out_val.name] = nb.view(hbm_map[inp_val.name], out_val.type.shape)
        return
    emit_reshape(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, out_val.type.shape, inp_val.type.dtype,
    )


def _emit_slice_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_slice(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, out_val.type.shape, op.attrs["starts"],
        inp_val.type.dtype,
        strides=op.attrs.get("strides"),
    )


def _emit_concat_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    out_val = op.results[0]
    axis = op.attrs["axis"]
    rank = len(out_val.type.shape)
    if axis < 0:
        axis += rank
    input_hbms = [hbm_map[v.name] for v in op.inputs]
    input_shapes = [v.type.shape for v in op.inputs]
    emit_concat(
        nb, input_hbms, hbm_map[out_val.name],
        input_shapes, axis, op.inputs[0].type.dtype,
    )


def _emit_broadcast_op(nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value]) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_broadcast_to(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, out_val.type.shape, inp_val.type.dtype,
    )


def _emit_iota_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower iota: an index ramp along ``dim``, broadcast over other axes.

    Tiled with a canonical row-major layout (last dim = free, penultimate =
    partition, earlier = batch).  ``nisa.iota`` produces, per SBUF tile,
    ``offset + p * channel_multiplier + f * step``.  We pick those so the
    value equals the global index along ``dim``:

      - dim is the free axis:      step = 1,  channel_multiplier = 0, offset = f_off
      - dim is the partition axis: step = 0,  channel_multiplier = 1, offset = p_off
      - dim is a batch axis:       constant per tile = batch index on that axis
    """
    out_val = op.results[0]
    dim = op.attrs["dim"]
    dst_hbm = hbm_map[out_val.name]
    dtype = out_val.type.dtype
    shape = out_val.type.shape
    rank = len(shape)

    tile_p = min(shape[-2], PARTITION_MAX) if rank >= 2 else 1
    tile_f = shape[-1] if rank >= 2 else shape[0]
    p_extent = shape[-2] if rank >= 2 else 1
    batch_dims = list(shape[:-2]) if rank > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    f_axis = rank - 1
    p_axis = rank - 2  # only meaningful when rank >= 2

    for bf in range(n_batch):
        batch_idx = unravel(bf, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)

            if dim == f_axis:
                pattern, ch_mul, offset = [[1, tile_f]], 0, 0
            elif rank >= 2 and dim == p_axis:
                pattern, ch_mul, offset = [[0, tile_f]], 1, p_off
            else:
                # batch axis: every element in this tile shares the index
                pattern, ch_mul, offset = [[0, tile_f]], 0, int(batch_idx[dim])

            tile = nb.alloc((p_size, tile_f), dtype, MemorySpace.SBUF)
            tile = nb.iota(tile, pattern=pattern, offset=offset, channel_multiplier=ch_mul)

            dst_slices = [DimSlice(bi, 1) for bi in batch_idx]
            if rank >= 2:
                dst_slices.append(DimSlice(p_off, p_size))
            dst_slices.append(DimSlice(0, tile_f))
            nb.dma_copy(dst_hbm, tile, dst_slices)


# max8 / match_replace8 read at most this many free elements per call (a real
# hardware limit the MLIR verifier enforces).  Wider rows are tiled.
TOPK_FREE_MAX = 16384


def _topk_scan(nb: Builder, data: Value, P: int, k: int, vdtype, idtype):
    """Run the max8 + match_replace8 scan over a resident (P, W>=8) SBUF tile.

    Returns ``(vals_sbuf (P, kp), idx_sbuf (P, kp))`` where ``kp = ceil(k/8)*8``
    (the fold-aligned width); callers slice the first ``k`` columns.  Indices
    are positions within ``data``.  Assembling the per-fold 8-wide results into
    one wide SBUF tile needs sub-tile column writes, which nki_ir lacks, so the
    folds round-trip through an HBM scratch buffer.
    """
    n_fold = (k + 7) // 8
    kp = n_fold * 8
    if n_fold == 1:
        val8 = nb.max8(nb.alloc((P, 8), vdtype, MemorySpace.SBUF), data)
        idx8 = nb.alloc((P, 8), idtype, MemorySpace.SBUF)
        _, idx8 = nb.match_replace8(data, idx8, data, val8, float("-inf"))
        return val8, idx8
    val_scratch = nb.alloc((P, kp), vdtype, MemorySpace.HBM)
    idx_scratch = nb.alloc((P, kp), idtype, MemorySpace.HBM)
    for fold in range(n_fold):
        val8 = nb.max8(nb.alloc((P, 8), vdtype, MemorySpace.SBUF), data)
        idx8 = nb.alloc((P, 8), idtype, MemorySpace.SBUF)
        data, idx8 = nb.match_replace8(data, idx8, data, val8, float("-inf"))
        col = DimSlice(fold * 8, 8)
        nb.dma_copy(val_scratch, val8, (DimSlice(0, P), col))
        nb.dma_copy(idx_scratch, idx8, (DimSlice(0, P), col))
    vals = nb.dma_copy(nb.alloc((P, kp), vdtype, MemorySpace.SBUF),
                       val_scratch, (DimSlice(0, P), DimSlice(0, kp)))
    idxs = nb.dma_copy(nb.alloc((P, kp), idtype, MemorySpace.SBUF),
                       idx_scratch, (DimSlice(0, P), DimSlice(0, kp)))
    return vals, idxs


def _load_topk_data(nb: Builder, src_hbm, P: int, off: int, f: int, vdtype):
    """Load src_hbm[:, off:off+f] into a (P, max(f,8)) SBUF tile, padding the
    tail with -inf so max8's >=8 free-dim requirement always holds."""
    width = max(f, 8)
    if width == f:
        return nb.dma_copy(nb.alloc((P, width), vdtype, MemorySpace.SBUF),
                           src_hbm, (DimSlice(0, P), DimSlice(off, f)))
    padded = nb.memset(nb.alloc((P, width), vdtype, MemorySpace.SBUF), float("-inf"))
    loaded = nb.dma_copy(nb.alloc((P, f), vdtype, MemorySpace.SBUF),
                         src_hbm, (DimSlice(0, P), DimSlice(off, f)))
    return _overlay_columns(nb, padded, loaded, f)


def _emit_topk_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower topk via the canonical hardware scan (max8 + match_replace8).

    The source (P, F) tile is loaded into SBUF; each fold reads the next 8
    largest (max8) and masks them to -inf in place (match_replace8), which also
    yields their indices.  ceil(k/8) folds cover any k.

    When F exceeds the max8 free-dim limit, the row is split into chunks of
    <= TOPK_FREE_MAX: each chunk is scanned for its local top-k (indices
    rebased to global), then the chunk winners (n_chunks * k candidates) are
    merged with a second scan, and a gather maps the merged positions back to
    the global indices.
    """
    src_val = op.inputs[0]
    val_out, idx_out = op.results[0], op.results[1]
    P, F = src_val.type.shape
    k = op.attrs["k"]
    src_hbm = hbm_map[src_val.name]
    val_hbm = hbm_map[val_out.name]
    idx_hbm = hbm_map[idx_out.name]
    vdtype = val_out.type.dtype
    idtype = idx_out.type.dtype

    if F <= TOPK_FREE_MAX:
        data = _load_topk_data(nb, src_hbm, P, 0, F, vdtype)
        vals, idxs = _topk_scan(nb, data, P, k, vdtype, idtype)
        v_store = vals if k == _aligned(k) else _first_cols(nb, vals, k)
        i_store = idxs if k == _aligned(k) else _first_cols(nb, idxs, k)
        nb.dma_copy(val_hbm, v_store, (DimSlice(0, P), DimSlice(0, k)))
        nb.dma_copy(idx_hbm, i_store, (DimSlice(0, P), DimSlice(0, k)))
        return

    n_chunks = ceildiv(F, TOPK_FREE_MAX)
    cand = n_chunks * k
    if cand > TOPK_FREE_MAX:
        raise NotImplementedError(
            f"topk: F={F}, k={k} needs {cand} merge candidates, exceeds "
            f"{TOPK_FREE_MAX}; a multi-level merge is not implemented"
        )

    # Per-chunk local top-k, with indices rebased to global, gathered into
    # candidate buffers (P, cand).
    cand_vals_hbm = nb.alloc((P, cand), vdtype, MemorySpace.HBM)
    cand_idx_hbm = nb.alloc((P, cand), idtype, MemorySpace.HBM)
    for c in range(n_chunks):
        off = c * TOPK_FREE_MAX
        fc = min(TOPK_FREE_MAX, F - off)
        data = _load_topk_data(nb, src_hbm, P, off, fc, vdtype)
        vals, idxs = _topk_scan(nb, data, P, k, vdtype, idtype)
        if off != 0:
            # Rebase local indices to global: idx += off.
            off_tile = nb.constant(float(off), (P, 1), idtype, MemorySpace.SBUF)
            idxs = nb.tensor_scalar_arith(
                nb.alloc(idxs.type.shape, idtype, MemorySpace.SBUF),
                idxs, off_tile, nki_ir.NisaArithOp.ADD,
            )
        col = DimSlice(c * k, k)
        v_store = vals if k == _aligned(k) else _first_cols(nb, vals, k)
        i_store = idxs if k == _aligned(k) else _first_cols(nb, idxs, k)
        nb.dma_copy(cand_vals_hbm, v_store, (DimSlice(0, P), col))
        nb.dma_copy(cand_idx_hbm, i_store, (DimSlice(0, P), col))

    # Merge: scan the candidate values for the global top-k, then gather the
    # corresponding global indices by the merged positions.
    cand_data = _load_topk_data(nb, cand_vals_hbm, P, 0, cand, vdtype)
    width = max(cand, 8)
    cand_idx_sbuf = nb.memset(
        nb.alloc((P, width), idtype, MemorySpace.SBUF), 0.0)
    cand_idx_sbuf = _overlay_columns(
        nb,
        cand_idx_sbuf,
        nb.dma_copy(nb.alloc((P, cand), idtype, MemorySpace.SBUF),
                    cand_idx_hbm, (DimSlice(0, P), DimSlice(0, cand))),
        cand,
    )
    mvals, mpos = _topk_scan(nb, cand_data, P, k, vdtype, idtype)
    # gather: global_idx[p, i] = cand_idx_sbuf[p, mpos[p, i]]
    gidx = nb.gather(
        nb.alloc(mpos.type.shape, idtype, MemorySpace.SBUF),
        cand_idx_sbuf, mpos,
    )
    v_store = mvals if k == _aligned(k) else _first_cols(nb, mvals, k)
    i_store = gidx if k == _aligned(k) else _first_cols(nb, gidx, k)
    nb.dma_copy(val_hbm, v_store, (DimSlice(0, P), DimSlice(0, k)))
    nb.dma_copy(idx_hbm, i_store, (DimSlice(0, P), DimSlice(0, k)))


def _aligned(k: int) -> int:
    """Fold-aligned width ceil(k/8)*8 (the column count _topk_scan returns)."""
    return ((k + 7) // 8) * 8


def _emit_gather_along_axis_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower gather_along_axis via the hardware per-partition gather.

    ``out[p, i] = data[p, idx[p, i]]``.  Each partition chunk (up to
    PARTITION_MAX rows) loads its data and index rows into SBUF, runs
    ``nisa.gather``, and stores the gathered row back to HBM.  The free
    dims of data and idx differ (F_data vs F_idx); the gather dst matches
    the idx shape.
    """
    data_val, idx_val = op.inputs[0], op.inputs[1]
    out_val = op.results[0]
    P, F_data = data_val.type.shape
    F_idx = idx_val.type.shape[1]
    data_hbm = hbm_map[data_val.name]
    idx_hbm = hbm_map[idx_val.name]
    out_hbm = hbm_map[out_val.name]
    vdtype = out_val.type.dtype
    idtype = idx_val.type.dtype

    for p_i in range(ceildiv(P, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, P - p_off)

        data_tile = nb.dma_copy(
            nb.alloc((p_size, F_data), vdtype, MemorySpace.SBUF),
            data_hbm, (DimSlice(p_off, p_size), DimSlice(0, F_data)),
        )
        idx_tile = nb.dma_copy(
            nb.alloc((p_size, F_idx), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(p_off, p_size), DimSlice(0, F_idx)),
        )
        out_tile = nb.gather(
            nb.alloc((p_size, F_idx), vdtype, MemorySpace.SBUF),
            data_tile, idx_tile,
        )
        nb.dma_copy(out_hbm, out_tile, (DimSlice(p_off, p_size), DimSlice(0, F_idx)))


def _emit_scatter_rows_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower scatter_rows: ``out = base.copy(); out[idx[r], :] = updates[r, :]``.

    First copy ``base`` HBM -> result HBM (tiled by N rows, the unchanged
    backdrop), then scatter the M update rows into the result via the indirect
    DMA store (``dma_copy_indirect``), tiled by M update rows.  The index tile
    is (m_size, 1) U32: 1-D SBUF index tiles are rejected by the hardware.
    """
    base_val, idx_val, upd_val = op.inputs[0], op.inputs[1], op.inputs[2]
    out_val = op.results[0]
    N, W = base_val.type.shape
    M = upd_val.type.shape[0]
    base_hbm = hbm_map[base_val.name]
    idx_hbm = hbm_map[idx_val.name]
    upd_hbm = hbm_map[upd_val.name]
    out_hbm = hbm_map[out_val.name]
    vdtype = out_val.type.dtype
    idtype = idx_val.type.dtype

    # Tile the row width so wide rows never need a (P, W) SBUF tile.
    w_tile = min(W, max_free_elems(vdtype))

    # Backdrop: copy base -> result, tiled over N rows and W columns.
    for p_i in range(ceildiv(N, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, N - p_off)
        for w_i in range(ceildiv(W, w_tile)):
            w_off = w_i * w_tile
            w_size = min(w_tile, W - w_off)
            tile = nb.dma_copy(
                nb.alloc((p_size, w_size), vdtype, MemorySpace.SBUF),
                base_hbm, (DimSlice(p_off, p_size), DimSlice(w_off, w_size)),
            )
            nb.dma_copy(
                out_hbm, tile, (DimSlice(p_off, p_size), DimSlice(w_off, w_size))
            )

    # Scatter the M update rows, tiled over M and W.  dma_copy_indirect
    # addresses whole rows of the result HBM tensor via the per-row index;
    # row_width keeps the full row stride while a column window is written.
    for m_i in range(ceildiv(M, PARTITION_MAX)):
        m_off = m_i * PARTITION_MAX
        m_size = min(PARTITION_MAX, M - m_off)
        idx_tile = nb.dma_copy(
            nb.alloc((m_size, 1), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(m_off, m_size), DimSlice(0, 1)),
        )
        for w_i in range(ceildiv(W, w_tile)):
            w_off = w_i * w_tile
            w_size = min(w_tile, W - w_off)
            upd_tile = nb.dma_copy(
                nb.alloc((m_size, w_size), vdtype, MemorySpace.SBUF),
                upd_hbm, (DimSlice(m_off, m_size), DimSlice(w_off, w_size)),
            )
            nb.dma_copy_indirect(
                out_hbm, upd_tile, idx_tile, row_width=W, free_offset=w_off,
            )


def _emit_gather_rows_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    """Lower gather_rows: ``out[r, :] = src[idx[r], :]`` via the indirect DMA
    load (``dma_copy_indirect``), gathering whole rows from the (N, W) src HBM
    tensor into the (M, W) result.  Tiled over M gathered rows.  The index tile
    is (m_size, 1) U32.  Avoids materializing the full (N, W) table on chip, so
    it scales to tall tables (e.g. embedding (128256, 2048))."""
    src_val, idx_val = op.inputs[0], op.inputs[1]
    out_val = op.results[0]
    N, W = src_val.type.shape
    M = out_val.type.shape[0]
    src_hbm = hbm_map[src_val.name]
    idx_hbm = hbm_map[idx_val.name]
    out_hbm = hbm_map[out_val.name]
    vdtype = out_val.type.dtype
    idtype = idx_val.type.dtype

    # Tile the row width: a single (m_size, W) tile would overflow SBUF for
    # wide rows (e.g. an MoE expert's flattened weight, W=786432). Gather a
    # column window [w_off, w_off+w_size) per DMA, addressing the full row
    # stride W via row_width so the index still selects whole source rows.
    w_tile = min(W, max_free_elems(vdtype))
    for m_i in range(ceildiv(M, PARTITION_MAX)):
        m_off = m_i * PARTITION_MAX
        m_size = min(PARTITION_MAX, M - m_off)
        idx_tile = nb.dma_copy(
            nb.alloc((m_size, 1), idtype, MemorySpace.SBUF),
            idx_hbm, (DimSlice(m_off, m_size), DimSlice(0, 1)),
        )
        for w_i in range(ceildiv(W, w_tile)):
            w_off = w_i * w_tile
            w_size = min(w_tile, W - w_off)
            # Indirect load: gather a column window of m_size rows of src.
            out_tile = nb.dma_copy_indirect(
                nb.alloc((m_size, w_size), vdtype, MemorySpace.SBUF),
                src_hbm, idx_tile, row_width=W, free_offset=w_off,
            )
            nb.dma_copy(
                out_hbm, out_tile, (DimSlice(m_off, m_size), DimSlice(w_off, w_size))
            )


def _first_cols(nb: Builder, tile: Value, keep: int) -> Value:
    """Return a (P, keep) SBUF tile holding the first ``keep`` columns of an
    8-wide tile, via an HBM scratch round-trip (nki_ir has no SBUF sub-view)."""
    P = tile.type.shape[0]
    scratch = nb.alloc((P, 8), tile.type.dtype, MemorySpace.HBM)
    nb.dma_copy(scratch, tile, (DimSlice(0, P), DimSlice(0, 8)))
    return nb.dma_copy(
        nb.alloc((P, keep), tile.type.dtype, MemorySpace.SBUF),
        scratch, (DimSlice(0, P), DimSlice(0, keep)),
    )


def _overlay_columns(nb: Builder, base: Value, cols: Value, n: int) -> Value:
    """Write the first ``n`` columns of ``cols`` over ``base`` (P, W>=n) via an
    HBM scratch round-trip, returning the merged SBUF tile."""
    P, W = base.type.shape
    scratch = nb.alloc((P, W), base.type.dtype, MemorySpace.HBM)
    nb.dma_copy(scratch, base, (DimSlice(0, P), DimSlice(0, W)))
    nb.dma_copy(scratch, cols, (DimSlice(0, P), DimSlice(0, n)))
    return nb.dma_copy(
        nb.alloc((P, W), base.type.dtype, MemorySpace.SBUF),
        scratch, (DimSlice(0, P), DimSlice(0, W)),
    )
