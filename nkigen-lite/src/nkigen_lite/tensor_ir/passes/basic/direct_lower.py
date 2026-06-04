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

from nkigen_lite.core import Graph, Value
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
    ELEMENTWISE_OPCODES,
    UNARY_OPS,
    ceildiv,
    compute_tile_sizes,
    emit_binary_op,
    emit_unary_op,
    hbm_slices,
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

    # Allocate HBM intermediates for all op results
    for op in graph.ops:
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
    tile_f = shape[-1] if len(shape) >= 2 else shape[0]
    p_extent = shape[-2] if len(shape) >= 2 else 1
    batch_dims = list(shape[:-2]) if len(shape) > 2 else []
    n_batch = prod(batch_dims) if batch_dims else 1

    for batch_flat in range(n_batch):
        batch_idx = unravel(batch_flat, batch_dims) if batch_dims else ()
        for p_i in range(ceildiv(p_extent, tile_p)):
            p_off = p_i * tile_p
            p_size = min(tile_p, p_extent - p_off)
            slices = []
            for bi in batch_idx:
                slices.append(DimSlice(bi, 1))
            if len(shape) >= 2:
                slices.append(DimSlice(p_off, p_size))
            slices.append(DimSlice(0, tile_f))
            tile = nb.dma_copy(
                nb.alloc((p_size, tile_f), src.type.dtype, MemorySpace.SBUF),
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


def _emit_elementwise_segment(
    nb: Builder, ops: list, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    """Emit a fused elementwise segment: one tiled loop for all ops."""
    # Use a canonical row-major layout for elementwise segments. HBM is
    # layout-agnostic, so all loads/stores address data by logical dimension
    # coordinates — the declared layout of individual values is irrelevant.
    rep_val = ops[-1].results[0]
    rep_shape = rep_val.type.shape
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
) -> None:
    """Emit one tile of fused elementwise computation."""
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

        if op.opcode in BINARY_OPS or op.opcode in BITWISE_OPS:
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
        elif op.opcode == "constant":
            out_shape = op.results[0].type.shape
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


def _emit_transpose_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_transpose(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, op.attrs["perm"], inp_val.type.dtype,
    )


def _emit_reshape_op(nb: Builder, op, hbm_map: dict[str, Value]) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
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
