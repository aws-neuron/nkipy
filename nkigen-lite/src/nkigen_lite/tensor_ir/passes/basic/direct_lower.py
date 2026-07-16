"""Orchestrated direct lowering: tensor IR → NKI IR with HBM boundaries.

Lowers a complete tensor IR graph (after canonicalize + decompose) to a single
NKI IR graph. Every op (elementwise, reduce, matmul, transpose, reshape, slice,
concat, broadcast_to) gets its own load→compute→store sequence with HBM
boundaries — elementwise through the linearize-and-tile loop, the rest through
their per-opcode emitters. (Fusing consecutive elementwise ops so intermediates
stay on-chip is the deferred Phase-2 work.)

Layouts are decided per op, not per value: every op boundary round-trips
through HBM, which is layout-agnostic (row-major), so each emitter picks the
layout for its own tile loop (see passes/layout.py).

Usage:
    graph = build_some_pattern(...)
    canonicalize(graph)
    decompose(graph)
    nki_graph = lower_graph(graph)
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

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    ELEMENTWISE_OPCODES,
    ceildiv,
    collapse_view,
    max_free_elems,
    unravel,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_schedule import TileSchedule
from nkigen_lite.tensor_ir.passes.basic.direct_lower_elementwise import (
    _emit_elementwise_op,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_alloc import Allocator


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def lower_graph(graph: Graph) -> nki_ir.Graph:
    """Lower a full tensor IR graph to NKI IR with HBM boundaries."""
    nb = Builder("direct_lower")
    alloc = Allocator(nb)
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

    # Allocate an HBM buffer for every op result.  Skip reshape results:
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

    # Lower each op.  An elementwise op is emitted through the tiled
    # load→compute→store loop; every other op reads its inputs' HBM buffers
    # directly through its per-opcode emitter (grouping consecutive elementwise
    # ops into one fused loop is to be reintroduced as the Phase-2
    # fusion-compatibility predicate).
    for op in graph.ops:
        opcode = op.opcode

        if opcode in ELEMENTWISE_OPCODES:
            _emit_elementwise_op(nb, op, hbm_map, alloc)
            continue

        emitter = _OP_EMITTERS.get(opcode)
        if emitter is None:
            raise NotImplementedError(f"Op {opcode!r} not supported")
        emitter(nb, op, hbm_map, alloc)

    # Copy final results to output buffers
    for out_name, out_val in graph.outputs.items():
        src = hbm_map[out_val.name]
        dst = hbm_map[f"{out_name}_out"]
        if src is not dst:
            _emit_hbm_copy(nb, src, dst, out_val.type.shape, alloc)

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_hbm_copy(nb: Builder, src: Value, dst: Value, shape: tuple[int, ...], alloc: Allocator):
    """Copy an entire HBM tensor to another HBM tensor, tiled."""
    if len(shape) == 0:
        # HBM buffers may be promoted from () to (1,)
        src_slices = [DimSlice(0, 1)] * len(src.type.shape)
        dst_slices = [DimSlice(0, 1)] * len(dst.type.shape)
        tile = alloc.load(src, src_slices, (1, 1), src.type.dtype)
        nb.dma_copy(dst, tile, dst_slices)
        return

    # Fast path: a full copy preserves row-major flat order, so a rank>=3 tensor
    # collapses to 2D (prod(shape[:-1]), shape[-1]) via zero-copy views and tiles
    # all leading axes onto the 128-lane partition. The generic path below tiles
    # only shape[-2] as the partition and unrolls prod(shape[:-2]) leading-dim
    # iterations one at a time — e.g. a (1,4096,1,128) KV-cache copy uses 1 of
    # 128 lanes and unrolls 4096 times. Only when the row width shape[-1] alone
    # fits the SBUF free budget (else the flat collapse would need per-row
    # sub-tiling the generic path already handles via tile_f).
    if len(shape) >= 3 and shape[-1] <= max_free_elems(src.type.dtype):
        lead = prod(shape[:-1])
        src_2d = collapse_view(nb, src, lead, shape[-1])
        dst_2d = collapse_view(nb, dst, lead, shape[-1])
        for p_off, p_size, f_off, f_size in TileSchedule.pf(
            lead, shape[-1], src.type.dtype
        ).pf_tiles():
            tile = alloc.load(
                src_2d, (DimSlice(p_off, p_size), DimSlice(f_off, f_size)),
                (p_size, f_size), src.type.dtype,
            )
            nb.dma_copy(dst_2d, tile, (DimSlice(p_off, p_size), DimSlice(f_off, f_size)))
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
                tile = alloc.load(src, slices, (p_size, f_size), src.type.dtype)
                nb.dma_copy(dst, tile, slices)


# ---------------------------------------------------------------------------
# Op-specific emission — delegates to standalone modules
# ---------------------------------------------------------------------------


from nkigen_lite.tensor_ir.passes.basic.direct_lower_matmul import emit_matmul
from nkigen_lite.tensor_ir.passes.basic.direct_lower_transpose import emit_transpose
from nkigen_lite.tensor_ir.passes.basic.direct_lower_memory import (
    emit_reshape, emit_slice, emit_concat,
)
from nkigen_lite.tensor_ir.passes.basic.direct_lower_broadcast import emit_broadcast_to
from nkigen_lite.tensor_ir.passes.basic.direct_lower_reduce import emit_reduce
from nkigen_lite.tensor_ir.passes.basic.direct_lower_iota import emit_iota
from nkigen_lite.tensor_ir.passes.basic.direct_lower_topk import emit_topk
from nkigen_lite.tensor_ir.passes.basic.direct_lower_gather import (
    emit_gather_along_axis, emit_scatter_rows, emit_gather_rows,
)


def _emit_matmul_op(
    nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator,
) -> None:
    a_val, b_val = op.inputs
    c_val = op.results[0]
    emit_matmul(
        nb, hbm_map[a_val.name], hbm_map[b_val.name], hbm_map[c_val.name],
        a_val.type.shape, b_val.type.shape, a_val.type.dtype, alloc,
    )


COLLECTIVE_OPCODES = frozenset(
    {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}
)


def _emit_collective_op(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
    """Lower a collective op to an nki_ir collective node.

    The compiler forbids collectives from reading/writing kernel IO tensors
    directly, so we stage through internal HBM scratch buffers:
    IO/result HBM -> src alloc -> collective -> dst alloc -> result HBM.
    """
    inp_val = op.inputs[0]
    out_val = op.results[0]
    src_hbm = hbm_map[inp_val.name]
    dst_hbm = hbm_map[out_val.name]

    src_scratch = alloc.hbm(src_hbm.type.shape, src_hbm.type.dtype)
    dst_scratch = alloc.hbm(dst_hbm.type.shape, dst_hbm.type.dtype)

    _emit_hbm_copy(nb, src_hbm, src_scratch, inp_val.type.shape, alloc)
    nb.collective(op.opcode, dst_scratch, src_scratch, op.attrs)
    _emit_hbm_copy(nb, dst_scratch, dst_hbm, out_val.type.shape, alloc)


def _emit_transpose_op(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_transpose(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, op.attrs["perm"], inp_val.type.dtype, alloc,
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


def _emit_reshape_op(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
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
        inp_val.type.shape, out_val.type.shape, inp_val.type.dtype, alloc,
    )


def _emit_slice_op(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_slice(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, out_val.type.shape, op.attrs["starts"],
        inp_val.type.dtype,
        strides=op.attrs.get("strides"), alloc=alloc,
    )


def _emit_concat_op(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
    out_val = op.results[0]
    axis = op.attrs["axis"]
    rank = len(out_val.type.shape)
    if axis < 0:
        axis += rank
    input_hbms = [hbm_map[v.name] for v in op.inputs]
    input_shapes = [v.type.shape for v in op.inputs]
    emit_concat(
        nb, input_hbms, hbm_map[out_val.name],
        input_shapes, axis, op.inputs[0].type.dtype, alloc,
    )


def _emit_broadcast_op(nb: Builder, op, hbm_map: dict[str, Value], alloc: Allocator) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    emit_broadcast_to(
        nb, hbm_map[inp_val.name], hbm_map[out_val.name],
        inp_val.type.shape, out_val.type.shape, inp_val.type.dtype, alloc,
    )


# Per-opcode emitters, all ``(nb, op, hbm_map) -> None``.
# Elementwise ops are not dispatched here — they run through the fused tile loop.
_OP_EMITTERS: dict[str, object] = {
    "reduce": emit_reduce,
    "matmul": _emit_matmul_op,
    "transpose": _emit_transpose_op,
    "reshape": _emit_reshape_op,
    "slice": _emit_slice_op,
    "concat": _emit_concat_op,
    "broadcast_to": _emit_broadcast_op,
    "iota": emit_iota,
    "topk": emit_topk,
    "gather_along_axis": emit_gather_along_axis,
    "scatter_rows": emit_scatter_rows,
    "gather_rows": emit_gather_rows,
    **{opcode: _emit_collective_op for opcode in COLLECTIVE_OPCODES},
}


