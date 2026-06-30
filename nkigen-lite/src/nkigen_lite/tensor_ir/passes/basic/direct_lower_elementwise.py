"""Direct lowering of tensor IR elementwise ops to NKI IR.

Lowers elementwise unary and binary ops from tensor IR to tiled NKI IR,
given the tensor IR graph and layout solver results (I/P/F classification
per value). This is a standalone lowering pass — no fusion plan or fusion
analysis dependency.

Supported ops:
  - binary: add, sub, mul, maximum, minimum
  - unary:  neg, exp, log, sqrt, rsqrt, tanh, relu, gelu, sigmoid, silu,
            reciprocal
  - constant

Tiling strategy:
  - I-dims: iterated one-at-a-time (outermost loops)
  - P-dims: innermost P-dim tiled at min(extent, 128), outer P-dims iterated
  - F-dims: taken at full extent (no F-tiling)

Broadcasting: operands with size-1 P or F dims are broadcast to the tile
shape of the group representative (partition broadcast via HBM scratch
round-trip, free-axis broadcast via tensor_scalar_arith).
"""

from __future__ import annotations

from nkigen_lite.core import DType, Graph, Value
from nkigen_lite.nki_ir.ir import (
    Builder,
    MemorySpace,
)
from nkigen_lite.nki_ir import ir as nki_ir
from nkigen_lite.nki_ir.insert_deallocs import insert_deallocs
from nkigen_lite.tensor_ir.passes.layout_solver import Layout

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    BINARY_OPS,
    UNARY_OPS,
    ceildiv,
    compute_tile_sizes,
    emit_binary_op,
    emit_unary_op,
    hbm_slices,
    map_indices,
    on_chip_shape,
)

_SUPPORTED_OPCODES = frozenset(BINARY_OPS.keys() | UNARY_OPS.keys() | {"constant"})




# ---------------------------------------------------------------------------
# Main lowering
# ---------------------------------------------------------------------------


def lower_elementwise(
    graph: Graph,
    layouts: dict[str, Layout],
) -> nki_ir.Graph:
    """Lower a tensor IR graph of elementwise ops to NKI IR.

    Args:
        graph: Tensor IR graph containing only elementwise ops (binary, unary,
               constant). All ops must have compatible layouts.
        layouts: Layout solver results mapping value names to Layout.

    Returns:
        An NKI IR graph ready for interpretation or hardware execution.

    Raises:
        NotImplementedError: If the graph contains unsupported ops.
    """
    for op in graph.ops:
        if op.opcode not in _SUPPORTED_OPCODES:
            raise NotImplementedError(
                f"Op {op.opcode!r} not supported by direct_lower_elementwise"
            )

    # Find the representative layout from the first output
    first_output = next(iter(graph.outputs.values()))
    rep_layout = layouts[first_output.name]
    rep_shape = first_output.type.shape

    # Compute tile sizes from the representative
    tile_sizes = compute_tile_sizes(rep_shape, rep_layout)

    nb = Builder("direct_elementwise")
    hbm_map: dict[str, Value] = {}

    # HBM inputs
    for v in graph.inputs:
        hbm_map[v.name] = nb.add_input(v.name, v.type.shape, v.type.dtype)

    # HBM output buffers
    for out_name, out_val in graph.outputs.items():
        hbm_map[f"{out_name}_out"] = nb.add_input(
            f"{out_name}_out", out_val.type.shape, out_val.type.dtype
        )

    # Determine which dims need loops
    loop_dims = []
    for d in sorted(tile_sizes.keys()):
        ts = tile_sizes[d]
        if ts < rep_shape[d]:
            loop_dims.append((d, rep_shape[d], ts))

    # Nested iteration over all tiled dimensions
    def _emit_nested(depth: int, indices: dict[int, int]):
        if depth >= len(loop_dims):
            _emit_tile_body(nb, graph, layouts, hbm_map, rep_layout,
                            rep_shape, tile_sizes, indices)
            return
        d, extent, ts = loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _emit_nested(depth + 1, {**indices, d: i})

    _emit_nested(0, {})

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_tile_body(
    nb: Builder,
    graph: Graph,
    layouts: dict[str, Layout],
    hbm_map: dict[str, Value],
    rep_layout: Layout,
    rep_shape: tuple[int, ...],
    tile_sizes: dict[int, int],
    indices: dict[int, int],
) -> None:
    """Emit loads -> compute -> stores for one tile iteration."""
    tile_map: dict[str, Value] = {}

    # Compute the representative tile shape for this iteration
    rep_tile = on_chip_shape(rep_shape, rep_layout, tile_sizes, indices)

    # Identify which values are graph inputs (need HBM loads)
    group_results = {r.name for op in graph.ops for r in op.results}

    # Load inputs
    for op in graph.ops:
        for inp in op.inputs:
            if inp.name in tile_map or inp.name in group_results:
                continue
            if inp.name not in hbm_map:
                raise ValueError(f"Input {inp.name!r} not found in HBM map")
            hbm_val = hbm_map[inp.name]
            val_layout = layouts[inp.name]
            val_tile_sizes = compute_tile_sizes(hbm_val.type.shape, val_layout)
            val_tile = on_chip_shape(
                hbm_val.type.shape, val_layout, val_tile_sizes, indices
            )
            slices = hbm_slices(
                hbm_val.type.shape, val_layout, val_tile_sizes,
                indices, rep_layout,
            )
            dst = nb.alloc(val_tile, hbm_val.type.dtype, MemorySpace.SBUF)
            tile_map[inp.name] = nb.dma_copy(dst, hbm_val, slices)

    # Compute ops
    for op in graph.ops:
        out_name = op.results[0].name
        out_dtype = op.results[0].type.dtype

        if op.opcode in BINARY_OPS:
            lhs = tile_map[op.inputs[0].name]
            rhs = tile_map[op.inputs[1].name]
            tile_map[out_name] = emit_binary_op(nb, out_dtype, lhs, rhs, op.opcode)

        elif op.opcode in UNARY_OPS:
            src = tile_map[op.inputs[0].name]
            tile_map[out_name] = emit_unary_op(nb, out_dtype, src, op.opcode)

        elif op.opcode == "constant":
            tile_map[out_name] = nb.constant(
                op.attrs["value"], rep_tile, out_dtype, MemorySpace.SBUF
            )

        else:
            raise NotImplementedError(f"Op {op.opcode!r} not supported")

    # Store outputs
    for out_name, out_val in graph.outputs.items():
        if out_val.name in tile_map:
            hbm_dst = hbm_map[f"{out_name}_out"]
            out_layout = layouts[out_val.name]
            out_tile_sizes = compute_tile_sizes(hbm_dst.type.shape, out_layout)
            slices = hbm_slices(
                hbm_dst.type.shape, out_layout, out_tile_sizes,
                indices, rep_layout,
            )
            nb.dma_copy(hbm_dst, tile_map[out_val.name], slices)
