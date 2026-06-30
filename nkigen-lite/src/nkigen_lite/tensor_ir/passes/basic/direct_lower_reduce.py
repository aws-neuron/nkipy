"""Direct lowering of tensor IR reduce ops to NKI IR.

Lowers reduce ops from tensor IR to tiled NKI IR, given the tensor IR graph
and layout solver results. Supports two classes of reduction:

1. P-dim reduction (cross-lane): reduces along partition dimensions.
   Two strategies:
     - GpSimd: cross_lane_reduce_arith (P,F) -> (1,F). Fast but only works
       when the full P extent fits in a single tile (<=128).
     - Matmul trick: ones[P,1].T @ x[P,F] -> dst[1,F]. Uses the tensor
       engine to sum across partitions. Works for any P extent via tiling
       with PSUM accumulation.

2. F-dim reduction (last N dims): reduces the rightmost N free dimensions.
   Uses tensor_reduce_arith which operates on the vector engine.
   Supports reducing all F-dims or a suffix of F-dims (partial F).

The input graph is expected to contain a single reduce op (with optional
preceding elementwise ops that feed into it). All values must have layouts
assigned by the layout solver (keepdims=True required).
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
    COMBINE_INIT,
    COMBINE_OPS,
    REDUCE_OPS,
    build_slices,
    ceildiv,
    clamped_extent,
    max_free_elems,
)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def lower_reduce(
    graph: Graph,
    layouts: dict[str, Layout],
    strategy: str = "gpsimd",
) -> nki_ir.Graph:
    """Lower a reduce op handling all legal axis combinations.

    Decomposes the reduction into up to two phases:
      1. F-phase: reduce any F-dims via tensor_reduce_arith
      2. P-phase: reduce any P-dims via gpsimd or matmul

    Args:
        strategy: "gpsimd" (default) or "matmul" for the P-dim phase.
                  "gpsimd" supports all kinds; "matmul" only sum/mean.
    """
    reduce_op = _find_reduce_op(graph)
    inp_val = reduce_op.inputs[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    kind = reduce_op.attrs["kind"]
    axis = set(reduce_op.attrs["axis"])

    f_axes = axis & set(inp_layout.f_dims)
    p_axes = axis & set(inp_layout.p_dims)

    if not f_axes:
        # Pure P-reduce
        if strategy == "matmul":
            return lower_p_reduce_matmul(graph, layouts)
        return lower_p_reduce_gpsimd(graph, layouts)

    if not p_axes:
        # Pure F-reduce
        return lower_f_reduce(graph, layouts)

    # Mixed P/F: decompose into F-reduce then P-reduce.
    # For mean: decompose as sum on F, sum on P, then divide by total count.
    # For sum/max/min: both phases use the same kind.
    f_kind = "sum" if kind == "mean" else kind
    p_kind = "sum" if kind == "mean" else kind

    nb = Builder("direct_reduce_mixed")
    hbm_map: dict[str, Value] = {}
    for v in graph.inputs:
        hbm_map[v.name] = nb.add_input(v.name, v.type.shape, v.type.dtype)
    for out_name, oval in graph.outputs.items():
        hbm_map[f"{out_name}_out"] = nb.add_input(
            f"{out_name}_out", oval.type.shape, oval.type.dtype
        )

    out_shape = reduce_op.results[0].type.shape

    # Classify dims
    reduced_f_dims = tuple(d for d in inp_layout.f_dims if d in f_axes)
    kept_f_dims = tuple(d for d in inp_layout.f_dims if d not in f_axes)
    reduced_p_dims = tuple(d for d in inp_layout.p_dims if d in p_axes)
    kept_p_dims = tuple(d for d in inp_layout.p_dims if d not in p_axes)

    # Tile sizes for input: iterate I, kept-P, kept-F; full on reduced-F;
    # reduced-P: innermost at min(ext, 128), outer at 1 (so product <= 128)
    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    for d in kept_p_dims:
        inp_tile_sizes[d] = 1
    for i, d in enumerate(reduced_p_dims):
        if i == len(reduced_p_dims) - 1:
            inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX)
        else:
            inp_tile_sizes[d] = 1
    for d in kept_f_dims:
        inp_tile_sizes[d] = 1
    for d in reduced_f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    # Output tile sizes
    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for d in kept_p_dims:
        out_tile_sizes[d] = 1
    for d in reduced_p_dims:
        out_tile_sizes[d] = out_shape[d]  # 1 (keepdims)
    for d in kept_f_dims:
        out_tile_sizes[d] = 1
    for d in reduced_f_dims:
        out_tile_sizes[d] = out_shape[d]  # 1 (keepdims)

    # Outer loops: I + kept-P + kept-F
    outer_loop_dims = [
        (d, inp_shape[d], inp_tile_sizes[d])
        for d in sorted(set(inp_layout.i_dims) | set(kept_p_dims) | set(kept_f_dims))
        if inp_tile_sizes[d] < inp_shape[d]
    ]
    # Inner P accumulation
    p_accum_dims = [
        (d, inp_shape[d], inp_tile_sizes[d])
        for d in sorted(reduced_p_dims)
        if inp_tile_sizes[d] < inp_shape[d]
    ]

    f_reduced_ext = prod(inp_shape[d] for d in reduced_f_dims)
    total_reduced = prod(inp_shape[d] for d in axis)
    dtype = inp_val.type.dtype
    reduce_nki_op = REDUCE_OPS[f_kind]
    combine_op = COMBINE_OPS[p_kind]

    def _emit_outer_nested(depth: int, outer_indices: dict[int, int]):
        if depth >= len(outer_loop_dims):
            _emit_mixed_reduce(
                nb, reduce_op, inp_layout, inp_shape, out_shape,
                inp_tile_sizes, out_tile_sizes, outer_indices,
                p_accum_dims, hbm_map, f_kind, p_kind, kind,
                reduced_p_dims, reduced_f_dims, f_reduced_ext,
                total_reduced, dtype, reduce_nki_op, combine_op, graph,
            )
            return
        d, extent, ts = outer_loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _emit_outer_nested(depth + 1, {**outer_indices, d: i})

    _emit_outer_nested(0, {})

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_mixed_reduce(
    nb: Builder,
    reduce_op,
    inp_layout: Layout,
    inp_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    inp_tile_sizes: dict[int, int],
    out_tile_sizes: dict[int, int],
    outer_indices: dict[int, int],
    p_accum_dims: list[tuple[int, int, int]],
    hbm_map: dict[str, Value],
    f_kind: str,
    p_kind: str,
    original_kind: str,
    reduced_p_dims: tuple[int, ...],
    reduced_f_dims: tuple[int, ...],
    f_reduced_ext: int,
    total_reduced: int,
    dtype: DType,
    reduce_nki_op: NisaReduceOp,
    combine_op: nki_ir.NisaArithOp,
    graph: Graph,
) -> None:
    """Emit mixed P/F reduction: F-reduce each P-chunk, then combine across P."""
    inp_val = reduce_op.inputs[0]

    # Accumulator for P-reduce: (1, 1) — after F-reduce each chunk is (P,1),
    # then cross-lane gives (1,1)
    accum = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
    accum = nb.memset(accum, COMBINE_INIT[p_kind])

    def _p_accum_nested(depth: int, p_indices: dict[int, int]):
        nonlocal accum
        if depth >= len(p_accum_dims):
            indices = {**outer_indices, **p_indices}
            p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, indices)

            # Load (P_chunk, F_reduced)
            slices = build_slices(inp_shape, inp_tile_sizes, indices)
            src = nb.alloc((p_ext, f_reduced_ext), dtype, MemorySpace.SBUF)
            src = nb.dma_copy(src, hbm_map[inp_val.name], slices)

            # F-reduce: (P_chunk, F) -> (P_chunk, 1)
            f_reduced = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
            f_reduced = nb.tensor_reduce_arith(f_reduced, src, reduce_nki_op,
                                               num_r_dim=1, keepdims=True)

            # P-reduce: (P_chunk, 1) -> (1, 1)
            p_reduced = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
            p_reduced = nb.cross_lane_reduce_arith(
                p_reduced, f_reduced, REDUCE_OPS[p_kind]
            )

            # Combine with accumulator
            new_accum = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
            accum = nb.tensor_tensor_arith(new_accum, accum, p_reduced, combine_op)
            return
        d, extent, ts = p_accum_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _p_accum_nested(depth + 1, {**p_indices, d: i})

    if p_accum_dims:
        _p_accum_nested(0, {})
    else:
        # Reduced P fits in one tile
        p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, outer_indices)
        slices = build_slices(inp_shape, inp_tile_sizes, outer_indices)
        src = nb.alloc((p_ext, f_reduced_ext), dtype, MemorySpace.SBUF)
        src = nb.dma_copy(src, hbm_map[inp_val.name], slices)

        f_reduced = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
        f_reduced = nb.tensor_reduce_arith(f_reduced, src, reduce_nki_op,
                                           num_r_dim=1, keepdims=True)

        accum = nb.cross_lane_reduce_arith(accum, f_reduced, REDUCE_OPS[p_kind])

    # Mean: divide by total count of reduced elements
    if original_kind == "mean":
        scale = nb.constant(1.0 / float(total_reduced), (1, 1), dtype, MemorySpace.SBUF)
        result = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
        accum = nb.tensor_tensor_arith(result, accum, scale, nki_ir.NisaArithOp.MULTIPLY)

    # Store
    out_slices = build_slices(out_shape, out_tile_sizes, outer_indices)
    out_key = f"{_out_name(reduce_op, graph)}_out"
    nb.dma_copy(hbm_map[out_key], accum, out_slices)


# ---------------------------------------------------------------------------
# F-dim reduction: tensor_reduce_arith on the vector engine
# ---------------------------------------------------------------------------


def lower_f_reduce(
    graph: Graph,
    layouts: dict[str, Layout],
) -> nki_ir.Graph:
    """Lower a graph with a reduce op over F-dims to NKI IR.

    Supports reducing any subset of F-dims (suffix, prefix, or middle).
    All kept F-dims are iterated one-at-a-time so the on-chip tile only
    contains the reduced F-dims as its free axis.

    Tiling: I-dims iterate one-at-a-time, innermost P-dim tiled at 128,
    outer P-dims one-at-a-time, kept F-dims iterated one-at-a-time,
    reduced F-dims taken at full extent.
    """
    reduce_op = _find_reduce_op(graph)
    inp_val = reduce_op.inputs[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = reduce_op.results[0].type.shape
    kind = reduce_op.attrs["kind"]
    axis = set(reduce_op.attrs["axis"])

    if not axis <= set(inp_layout.f_dims):
        raise ValueError(
            f"F-reduce requires axes {axis} to be F-dims, "
            f"but layout has f_dims={inp_layout.f_dims}"
        )

    f_dims = inp_layout.f_dims
    # Determine which F-dims are reduced and which are kept.
    # tensor_reduce_arith reduces the rightmost N free dims of the 2D tile.
    # If the reduced axes form the suffix of f_dims, we take the reduced dims
    # at full extent and reduce them all at once.
    # If the reduced axes are a prefix or middle (non-suffix), we iterate over
    # the non-reduced trailing F-dims so each tile contains only the reduced
    # portion as its free axis.

    kept_f_dims = tuple(d for d in f_dims if d not in axis)
    reduced_f_dims = tuple(d for d in f_dims if d in axis)

    nb = Builder("direct_f_reduce")
    hbm_map: dict[str, Value] = {}

    for v in graph.inputs:
        hbm_map[v.name] = nb.add_input(v.name, v.type.shape, v.type.dtype)
    for out_name, oval in graph.outputs.items():
        hbm_map[f"{out_name}_out"] = nb.add_input(
            f"{out_name}_out", oval.type.shape, oval.type.dtype
        )

    # Input tile sizes
    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    p_dims = inp_layout.p_dims
    for i, d in enumerate(p_dims):
        inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX) if i == len(p_dims) - 1 else 1
    for d in kept_f_dims:
        inp_tile_sizes[d] = 1
    for d in reduced_f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    # Output tile sizes
    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for i, d in enumerate(p_dims):
        out_tile_sizes[d] = min(out_shape[d], PARTITION_MAX) if i == len(p_dims) - 1 else 1
    for d in kept_f_dims:
        out_tile_sizes[d] = 1
    for d in reduced_f_dims:
        out_tile_sizes[d] = 1

    loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                 for d in sorted(inp_tile_sizes.keys())
                 if inp_tile_sizes[d] < inp_shape[d]]

    def _emit_nested(depth: int, indices: dict[int, int]):
        if depth >= len(loop_dims):
            _emit_f_reduce_tile(nb, graph, reduce_op, inp_layout, inp_shape,
                                out_shape, inp_tile_sizes, out_tile_sizes,
                                indices, hbm_map, kind, reduced_f_dims)
            return
        d, extent, ts = loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _emit_nested(depth + 1, {**indices, d: i})

    _emit_nested(0, {})

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_f_reduce_tile(
    nb: Builder,
    graph: Graph,
    reduce_op,
    inp_layout: Layout,
    inp_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    inp_tile_sizes: dict[int, int],
    out_tile_sizes: dict[int, int],
    indices: dict[int, int],
    hbm_map: dict[str, Value],
    kind: str,
    reduced_f_dims: tuple[int, ...],
) -> None:
    """Emit one tile of F-dim reduction."""
    inp_val = reduce_op.inputs[0]
    dtype = inp_val.type.dtype

    p_ext = clamped_extent(inp_layout.p_dims, inp_shape, inp_tile_sizes, indices)
    f_reduced_ext = prod(inp_shape[d] for d in reduced_f_dims)
    tile_shape = (p_ext, f_reduced_ext)

    # Load input tile
    slices = build_slices(inp_shape, inp_tile_sizes, indices)
    src_tile = nb.alloc(tile_shape, dtype, MemorySpace.SBUF)
    src_tile = nb.dma_copy(src_tile, hbm_map[inp_val.name], slices)

    # Reduce all free dims -> (P, 1)
    reduce_nki_op = REDUCE_OPS[kind]
    dst_tile = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
    dst_tile = nb.tensor_reduce_arith(dst_tile, src_tile, reduce_nki_op,
                                      num_r_dim=1, keepdims=True)

    # Mean: divide by count
    if kind == "mean":
        scale = nb.constant(1.0 / float(f_reduced_ext), (p_ext, 1), dtype, MemorySpace.SBUF)
        result = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
        dst_tile = nb.tensor_tensor_arith(
            result, dst_tile, scale, nki_ir.NisaArithOp.MULTIPLY
        )

    # Store output tile
    out_slices = build_slices(out_shape, out_tile_sizes, indices)
    out_key = f"{_out_name(reduce_op, graph)}_out"
    nb.dma_copy(hbm_map[out_key], dst_tile, out_slices)


# ---------------------------------------------------------------------------
# P-dim reduction: GpSimd strategy
# ---------------------------------------------------------------------------


def lower_p_reduce_gpsimd(
    graph: Graph,
    layouts: dict[str, Layout],
) -> nki_ir.Graph:
    """Lower P-dim reduction using GpSimd cross_lane_reduce_arith.

    Works for any P extent. When the reduced P extent exceeds PARTITION_MAX
    (128), tiles P at 128 and combines partial cross-lane reductions with the
    appropriate element-wise op (add for sum/mean, max for max, min for min).
    Non-reduced P-dims (if any) are iterated one-at-a-time.
    """
    reduce_op = _find_reduce_op(graph)
    inp_val = reduce_op.inputs[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = reduce_op.results[0].type.shape
    kind = reduce_op.attrs["kind"]
    axis = set(reduce_op.attrs["axis"])

    if not axis <= set(inp_layout.p_dims):
        raise ValueError(
            f"P-reduce (gpsimd) requires axes {axis} to be P-dims, "
            f"but layout has p_dims={inp_layout.p_dims}"
        )

    reduced_p_dims = tuple(d for d in inp_layout.p_dims if d in axis)
    kept_p_dims = tuple(d for d in inp_layout.p_dims if d not in axis)

    nb = Builder("direct_p_reduce_gpsimd")
    hbm_map: dict[str, Value] = {}
    for v in graph.inputs:
        hbm_map[v.name] = nb.add_input(v.name, v.type.shape, v.type.dtype)
    for out_name, oval in graph.outputs.items():
        hbm_map[f"{out_name}_out"] = nb.add_input(
            f"{out_name}_out", oval.type.shape, oval.type.dtype
        )

    # Input tile sizes: I=1, kept P=1, reduced P: innermost at min(ext,128)
    # outer at 1 (ensures product <= 128), F=full
    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    for d in kept_p_dims:
        inp_tile_sizes[d] = 1
    for i, d in enumerate(reduced_p_dims):
        if i == len(reduced_p_dims) - 1:
            inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX)
        else:
            inp_tile_sizes[d] = 1
    for d in inp_layout.f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    # Output tile sizes
    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for d in kept_p_dims:
        out_tile_sizes[d] = 1
    for d in reduced_p_dims:
        out_tile_sizes[d] = out_shape[d]  # keepdims: 1
    for d in inp_layout.f_dims:
        out_tile_sizes[d] = out_shape[d]

    # Outer loops: I-dims + kept P-dims (iterate per output element)
    outer_loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                       for d in sorted(set(inp_layout.i_dims) | set(kept_p_dims))
                       if inp_tile_sizes[d] < inp_shape[d]]
    # Inner accumulation: reduced P-dims (combined across tiles)
    accum_loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                       for d in sorted(reduced_p_dims)
                       if inp_tile_sizes[d] < inp_shape[d]]

    def _emit_outer_nested(depth: int, outer_indices: dict[int, int]):
        if depth >= len(outer_loop_dims):
            _emit_p_reduce_gpsimd_accumulate(
                nb, graph, reduce_op, inp_layout, inp_shape, out_shape,
                inp_tile_sizes, out_tile_sizes, outer_indices, accum_loop_dims,
                hbm_map, kind, reduced_p_dims,
            )
            return
        d, extent, ts = outer_loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _emit_outer_nested(depth + 1, {**outer_indices, d: i})

    _emit_outer_nested(0, {})

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_p_reduce_gpsimd_accumulate(
    nb: Builder,
    graph: Graph,
    reduce_op,
    inp_layout: Layout,
    inp_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    inp_tile_sizes: dict[int, int],
    out_tile_sizes: dict[int, int],
    outer_indices: dict[int, int],
    accum_loop_dims: list[tuple[int, int, int]],
    hbm_map: dict[str, Value],
    kind: str,
    reduced_p_dims: tuple[int, ...],
) -> None:
    """Accumulate cross-lane partial reductions across P-tiles."""
    inp_val = reduce_op.inputs[0]
    dtype = inp_val.type.dtype
    f_ext = clamped_extent(inp_layout.f_dims, inp_shape, inp_tile_sizes, outer_indices)
    reduce_nki_op = REDUCE_OPS[kind]
    combine_op = COMBINE_OPS[kind]
    total_p = prod(inp_shape[d] for d in reduced_p_dims)

    # Accumulator: (1, F) initialized to identity for the combine op
    accum = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
    accum = nb.memset(accum, COMBINE_INIT[kind])

    def _accum_nested(depth: int, accum_indices: dict[int, int]):
        if depth >= len(accum_loop_dims):
            nonlocal accum
            indices = {**outer_indices, **accum_indices}
            p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, indices)

            # Load (P_chunk, F)
            slices = build_slices(inp_shape, inp_tile_sizes, indices)
            src_tile = nb.alloc((p_ext, f_ext), dtype, MemorySpace.SBUF)
            src_tile = nb.dma_copy(src_tile, hbm_map[inp_val.name], slices)

            # Cross-lane reduce this chunk: (P_chunk, F) -> (1, F)
            partial = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
            partial = nb.cross_lane_reduce_arith(partial, src_tile, reduce_nki_op)

            # Combine with accumulator
            new_accum = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
            accum = nb.tensor_tensor_arith(new_accum, accum, partial, combine_op)
            return
        d, extent, ts = accum_loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _accum_nested(depth + 1, {**accum_indices, d: i})

    if accum_loop_dims:
        _accum_nested(0, {})
    else:
        # Single tile: no accumulation needed, just reduce directly
        p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, outer_indices)
        slices = build_slices(inp_shape, inp_tile_sizes, outer_indices)
        src_tile = nb.alloc((p_ext, f_ext), dtype, MemorySpace.SBUF)
        src_tile = nb.dma_copy(src_tile, hbm_map[inp_val.name], slices)
        accum = nb.cross_lane_reduce_arith(accum, src_tile, reduce_nki_op)

    # Mean: divide accumulated sum by total P extent
    if kind == "mean":
        scale = nb.constant(1.0 / float(total_p), (1, f_ext), dtype, MemorySpace.SBUF)
        result = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
        accum = nb.tensor_tensor_arith(result, accum, scale, nki_ir.NisaArithOp.MULTIPLY)

    # Store (1, F)
    out_slices = build_slices(out_shape, out_tile_sizes, outer_indices)
    out_key = f"{_out_name(reduce_op, graph)}_out"
    nb.dma_copy(hbm_map[out_key], accum, out_slices)


# ---------------------------------------------------------------------------
# P-dim reduction: matmul trick (ones.T @ x)
# ---------------------------------------------------------------------------


def lower_p_reduce_matmul(
    graph: Graph,
    layouts: dict[str, Layout],
) -> nki_ir.Graph:
    """Lower P-dim reduction using the matmul trick: ones.T @ x.

    Works for any P extent by tiling P at PARTITION_MAX with PSUM accumulation.

    The matmul computes: stationary[K,M].T @ moving[K,N] = dst[M,N]
    For P-dim sum: ones[P,1].T @ x[P,F] = sum_over_P[1,F]
    where K=P (contraction/partition), M=1 (stationary free), N=F (moving free).
    """
    reduce_op = _find_reduce_op(graph)
    inp_val = reduce_op.inputs[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = reduce_op.results[0].type.shape
    kind = reduce_op.attrs["kind"]
    axis = set(reduce_op.attrs["axis"])

    if kind not in ("sum", "mean"):
        raise ValueError(
            f"Matmul trick only supports sum/mean reduction, got {kind!r}. "
            f"Use gpsimd for max/min."
        )

    if not axis <= set(inp_layout.p_dims):
        raise ValueError(
            f"P-reduce (matmul) requires axes {axis} to be P-dims, "
            f"but layout has p_dims={inp_layout.p_dims}"
        )

    nb = Builder("direct_p_reduce_matmul")
    hbm_map: dict[str, Value] = {}
    for v in graph.inputs:
        hbm_map[v.name] = nb.add_input(v.name, v.type.shape, v.type.dtype)
    for out_name, oval in graph.outputs.items():
        hbm_map[f"{out_name}_out"] = nb.add_input(
            f"{out_name}_out", oval.type.shape, oval.type.dtype
        )

    reduced_p_dims = tuple(d for d in inp_layout.p_dims if d in axis)
    kept_p_dims = tuple(d for d in inp_layout.p_dims if d not in axis)
    f_extent = prod(inp_shape[d] for d in inp_layout.f_dims)

    # Input tile sizes: I=1, kept P=1, reduced P: innermost at min(ext,128)
    # outer at 1 (ensures product <= 128), F=full
    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    for d in kept_p_dims:
        inp_tile_sizes[d] = 1
    for i, d in enumerate(reduced_p_dims):
        if i == len(reduced_p_dims) - 1:
            inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX)
        else:
            inp_tile_sizes[d] = 1
    for d in inp_layout.f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    # Outer loops (I-dims + kept P-dims) vs inner accumulation (reduced P-dims)
    outer_loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                       for d in sorted(set(inp_layout.i_dims) | set(kept_p_dims))
                       if inp_tile_sizes[d] < inp_shape[d]]
    p_loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                   for d in sorted(reduced_p_dims)
                   if inp_tile_sizes[d] < inp_shape[d]]

    # Output tile sizes
    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for d in kept_p_dims:
        out_tile_sizes[d] = 1
    for d in reduced_p_dims:
        out_tile_sizes[d] = out_shape[d]  # keepdims: 1
    for d in inp_layout.f_dims:
        out_tile_sizes[d] = out_shape[d]

    def _emit_outer_nested(depth: int, outer_indices: dict[int, int]):
        if depth >= len(outer_loop_dims):
            _emit_p_reduce_matmul_accumulate(
                nb, graph, reduce_op, inp_layout, inp_shape, out_shape,
                inp_tile_sizes, out_tile_sizes, outer_indices, p_loop_dims,
                hbm_map, f_extent, reduced_p_dims,
            )
            return
        d, extent, ts = outer_loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _emit_outer_nested(depth + 1, {**outer_indices, d: i})

    _emit_outer_nested(0, {})

    nb.set_outputs({name: hbm_map[f"{name}_out"] for name in graph.outputs})
    insert_deallocs(nb.graph)
    return nb.graph


def _emit_p_reduce_matmul_accumulate(
    nb: Builder,
    graph: Graph,
    reduce_op,
    inp_layout: Layout,
    inp_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    inp_tile_sizes: dict[int, int],
    out_tile_sizes: dict[int, int],
    i_indices: dict[int, int],
    p_loop_dims: list[tuple[int, int, int]],
    hbm_map: dict[str, Value],
    f_extent: int,
    reduced_p_dims: tuple[int, ...],
) -> None:
    """Accumulate across P-tiles using matmul: ones[P_chunk,1].T @ x[P_chunk,F]."""
    inp_val = reduce_op.inputs[0]
    dtype = inp_val.type.dtype
    kind = reduce_op.attrs["kind"]
    total_p = prod(inp_shape[d] for d in reduced_p_dims)

    # PSUM accumulator: (M=1, N=F)
    psum = nb.alloc((1, f_extent), DType.F32, MemorySpace.PSUM)
    psum = nb.memset(psum, 0.0)

    def _p_nested(depth: int, p_indices: dict[int, int]):
        if depth >= len(p_loop_dims):
            indices = {**i_indices, **p_indices}
            p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, indices)

            # Load input tile (P_chunk, F)
            slices = build_slices(inp_shape, inp_tile_sizes, indices)
            src_tile = nb.alloc((p_ext, f_extent), dtype, MemorySpace.SBUF)
            src_tile = nb.dma_copy(src_tile, hbm_map[inp_val.name], slices)

            # Stationary: ones[K=P_chunk, M=1]
            ones = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
            ones = nb.memset(ones, 1.0)

            # matmul: ones[K,M=1].T @ src[K,N=F] -> psum[M=1,N=F]
            nb.matmul(psum, ones, src_tile, accumulate=True)
            return
        d, extent, ts = p_loop_dims[depth]
        n_tiles = ceildiv(extent, ts)
        for i in range(n_tiles):
            _p_nested(depth + 1, {**p_indices, d: i})

    if p_loop_dims:
        _p_nested(0, {})
    else:
        # Reduced P fits in one tile
        p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, i_indices)
        slices = build_slices(inp_shape, inp_tile_sizes, i_indices)
        src_tile = nb.alloc((p_ext, f_extent), dtype, MemorySpace.SBUF)
        src_tile = nb.dma_copy(src_tile, hbm_map[inp_val.name], slices)
        ones = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
        ones = nb.memset(ones, 1.0)
        nb.matmul(psum, ones, src_tile, accumulate=True)

    # Copy PSUM -> SBUF
    sbuf_out = nb.alloc((1, f_extent), DType.F32, MemorySpace.SBUF)
    sbuf_out = nb.tensor_copy(sbuf_out, psum)

    # Mean: divide by total reduced P extent
    if kind == "mean":
        scale = nb.constant(1.0 / float(total_p), (1, f_extent), DType.F32, MemorySpace.SBUF)
        mean_dst = nb.alloc((1, f_extent), DType.F32, MemorySpace.SBUF)
        sbuf_out = nb.tensor_tensor_arith(mean_dst, sbuf_out, scale, nki_ir.NisaArithOp.MULTIPLY)

    # Cast if needed
    out_dtype = reduce_op.results[0].type.dtype
    if out_dtype != DType.F32:
        cast_dst = nb.alloc((1, f_extent), out_dtype, MemorySpace.SBUF)
        sbuf_out = nb.cast(cast_dst, sbuf_out)

    # Store
    out_slices = build_slices(out_shape, out_tile_sizes, i_indices)
    out_key = f"{_out_name(reduce_op, graph)}_out"
    nb.dma_copy(hbm_map[out_key], sbuf_out, out_slices)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _find_reduce_op(graph: Graph):
    """Find the single reduce op in the graph."""
    reduce_ops = [op for op in graph.ops if op.opcode == "reduce"]
    if len(reduce_ops) == 0:
        raise ValueError("No reduce op found in graph")
    if len(reduce_ops) > 1:
        raise ValueError(f"Expected 1 reduce op, found {len(reduce_ops)}")
    return reduce_ops[0]


def _out_name(reduce_op, graph: Graph) -> str:
    """Find the output name for the reduce op's result."""
    result_name = reduce_op.results[0].name
    for name, val in graph.outputs.items():
        if val.name == result_name:
            return name
    raise ValueError(f"Reduce result {result_name!r} not in graph outputs")


# ---------------------------------------------------------------------------
# Emit function for use by the orchestrator
# ---------------------------------------------------------------------------


def emit_reduce(
    nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    """Emit a reduce op into an existing Builder with pre-allocated HBM buffers."""
    inp_val = op.inputs[0]
    out_val = op.results[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = out_val.type.shape
    kind = op.attrs["kind"]
    axis = set(op.attrs["axis"])
    dtype = inp_val.type.dtype

    f_axes = axis & set(inp_layout.f_dims)
    p_axes = axis & set(inp_layout.p_dims)

    if f_axes and not p_axes:
        _emit_f_reduce_inline(nb, op, layouts, hbm_map)
    elif p_axes and not f_axes:
        _emit_p_reduce_inline(nb, op, layouts, hbm_map)
    else:
        _emit_mixed_reduce_inline(nb, op, layouts, hbm_map)


def _try_emit_collapsed_f_reduce(
    nb: Builder, op, hbm_map: dict[str, Value],
) -> bool:
    """Reduce over a trailing-suffix free axis by collapsing all leading dims
    onto the partition. Returns False (emitting nothing) when not applicable.

    A reduce like ``(1, 16, 128, 128) -> (1, 16, 128, 1)`` over the last axis
    is logically ``(prod(shape[:-1]), shape[-1]) -> (prod(shape[:-1]), 1)``: a
    2-D reduce of the free axis. The old per-tile path instead put only
    ``shape[-2]`` on the partition (16 lanes for the 4-D attention tensor) and
    iterated the other leading dims one row at a time (~640 nki ops/reduce).
    Collapsing folds ``1*16*128 = 2048`` rows onto the partition, so it tiles
    at 128 (16 full tiles) and reduces each in one ``tensor_reduce_arith`` —
    matching how HLO lowers softmax's reduce (reduce then reshape to 2-D).

    Only fires for ``sum``/``max``/``min`` over a contiguous trailing run of
    axes with ``keepdims=True`` (the canonical/decomposed form; ``mean`` has
    already been rewritten to ``sum`` + scale by the decompose pass). Anything
    else falls through to the general per-tile path.
    """
    inp_val = op.inputs[0]
    out_val = op.results[0]
    inp_shape = inp_val.type.shape
    rank = len(inp_shape)
    axis = sorted(op.attrs["axis"])
    kind = op.attrs["kind"]

    # Require: keepdims, a known reduce op, and the reduced axes form the
    # trailing suffix of the shape (so leading dims collapse cleanly).
    if not op.attrs.get("keepdims", False) or kind not in REDUCE_OPS:
        return False
    if rank < 3:
        return False  # 2-D already packs the partition
    if axis != list(range(rank - len(axis), rank)):
        return False  # reduced axes are not the trailing suffix

    lead = prod(inp_shape[: axis[0]])     # collapsed partition extent
    red = prod(inp_shape[axis[0]:])       # reduced free extent
    dtype = inp_val.type.dtype

    # The reduced free extent must fit one SBUF partition row; bail otherwise
    # (the general path's finer F tiling handles the wide-row case).
    if red > max_free_elems(dtype):
        return False

    src_2d = nb.view(hbm_map[inp_val.name], (lead, red))
    dst_2d = nb.view(hbm_map[out_val.name], (lead, 1))
    reduce_nki_op = REDUCE_OPS[kind]

    for p_i in range(ceildiv(lead, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, lead - p_off)
        src = nb.dma_copy(
            nb.alloc((p_size, red), dtype, MemorySpace.SBUF),
            src_2d, (DimSlice(p_off, p_size), DimSlice(0, red)),
        )
        dst = nb.tensor_reduce_arith(
            nb.alloc((p_size, 1), dtype, MemorySpace.SBUF),
            src, reduce_nki_op, num_r_dim=1, keepdims=True,
        )
        nb.dma_copy(dst_2d, dst, (DimSlice(p_off, p_size), DimSlice(0, 1)))
    return True


def _emit_f_reduce_inline(
    nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    # Fast path: collapse leading dims onto the partition for a trailing-axis
    # reduce (packs all 128 lanes; mirrors HLO's reduce-then-reshape). Falls
    # back to the general per-tile path below when not applicable.
    if _try_emit_collapsed_f_reduce(nb, op, hbm_map):
        return

    inp_val = op.inputs[0]
    out_val = op.results[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = out_val.type.shape
    kind = op.attrs["kind"]
    axis = set(op.attrs["axis"])
    dtype = inp_val.type.dtype

    f_dims = inp_layout.f_dims
    kept_f_dims = tuple(d for d in f_dims if d not in axis)
    reduced_f_dims = tuple(d for d in f_dims if d in axis)

    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    p_dims = inp_layout.p_dims
    for i, d in enumerate(p_dims):
        inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX) if i == len(p_dims) - 1 else 1
    for d in kept_f_dims:
        inp_tile_sizes[d] = 1
    for d in reduced_f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for i, d in enumerate(p_dims):
        out_tile_sizes[d] = min(out_shape[d], PARTITION_MAX) if i == len(p_dims) - 1 else 1
    for d in kept_f_dims:
        out_tile_sizes[d] = 1
    for d in reduced_f_dims:
        out_tile_sizes[d] = 1

    loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                 for d in sorted(inp_tile_sizes.keys())
                 if inp_tile_sizes[d] < inp_shape[d]]

    def _nested(depth: int, indices: dict[int, int]):
        if depth >= len(loop_dims):
            p_ext = clamped_extent(inp_layout.p_dims, inp_shape, inp_tile_sizes, indices)
            f_ext = prod(inp_shape[d] for d in reduced_f_dims)

            slices = build_slices(inp_shape, inp_tile_sizes, indices)
            src = nb.alloc((p_ext, f_ext), dtype, MemorySpace.SBUF)
            src = nb.dma_copy(src, hbm_map[inp_val.name], slices)

            reduce_nki_op = REDUCE_OPS[kind]
            dst = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
            dst = nb.tensor_reduce_arith(dst, src, reduce_nki_op, num_r_dim=1, keepdims=True)

            if kind == "mean":
                scale = nb.constant(1.0 / float(f_ext), (p_ext, 1), dtype, MemorySpace.SBUF)
                result = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
                dst = nb.tensor_tensor_arith(result, dst, scale, nki_ir.NisaArithOp.MULTIPLY)

            out_slices = build_slices(out_shape, out_tile_sizes, indices)
            nb.dma_copy(hbm_map[out_val.name], dst, out_slices)
            return
        d, extent, ts = loop_dims[depth]
        for i in range(ceildiv(extent, ts)):
            _nested(depth + 1, {**indices, d: i})

    _nested(0, {})


def _emit_p_reduce_inline(
    nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = out_val.type.shape
    kind = op.attrs["kind"]
    axis = set(op.attrs["axis"])
    dtype = inp_val.type.dtype

    reduced_p_dims = tuple(d for d in inp_layout.p_dims if d in axis)
    kept_p_dims = tuple(d for d in inp_layout.p_dims if d not in axis)

    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    for d in kept_p_dims:
        inp_tile_sizes[d] = 1
    for i, d in enumerate(reduced_p_dims):
        inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX) if i == len(reduced_p_dims) - 1 else 1
    for d in inp_layout.f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for d in kept_p_dims:
        out_tile_sizes[d] = 1
    for d in reduced_p_dims:
        out_tile_sizes[d] = out_shape[d]
    for d in inp_layout.f_dims:
        out_tile_sizes[d] = out_shape[d]

    outer_loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                       for d in sorted(set(inp_layout.i_dims) | set(kept_p_dims))
                       if inp_tile_sizes[d] < inp_shape[d]]
    accum_loop_dims = [(d, inp_shape[d], inp_tile_sizes[d])
                       for d in sorted(reduced_p_dims)
                       if inp_tile_sizes[d] < inp_shape[d]]

    total_p = prod(inp_shape[d] for d in reduced_p_dims)
    reduce_nki_op = REDUCE_OPS[kind]
    combine_op = COMBINE_OPS[kind]

    def _outer_nested(depth: int, outer_indices: dict[int, int]):
        if depth >= len(outer_loop_dims):
            f_ext = clamped_extent(inp_layout.f_dims, inp_shape, inp_tile_sizes, outer_indices)
            accum = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
            accum = nb.memset(accum, COMBINE_INIT[kind])

            def _accum_nested(depth2: int, accum_indices: dict[int, int]):
                nonlocal accum
                if depth2 >= len(accum_loop_dims):
                    indices = {**outer_indices, **accum_indices}
                    p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, indices)
                    slices = build_slices(inp_shape, inp_tile_sizes, indices)
                    src = nb.alloc((p_ext, f_ext), dtype, MemorySpace.SBUF)
                    src = nb.dma_copy(src, hbm_map[inp_val.name], slices)
                    partial = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
                    partial = nb.cross_lane_reduce_arith(partial, src, reduce_nki_op)
                    new_accum = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
                    accum = nb.tensor_tensor_arith(new_accum, accum, partial, combine_op)
                    return
                d, extent, ts = accum_loop_dims[depth2]
                for i in range(ceildiv(extent, ts)):
                    _accum_nested(depth2 + 1, {**accum_indices, d: i})

            if accum_loop_dims:
                _accum_nested(0, {})
            else:
                p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, outer_indices)
                slices = build_slices(inp_shape, inp_tile_sizes, outer_indices)
                src = nb.alloc((p_ext, f_ext), dtype, MemorySpace.SBUF)
                src = nb.dma_copy(src, hbm_map[inp_val.name], slices)
                accum = nb.cross_lane_reduce_arith(accum, src, reduce_nki_op)

            if kind == "mean":
                scale = nb.constant(1.0 / float(total_p), (1, f_ext), dtype, MemorySpace.SBUF)
                result = nb.alloc((1, f_ext), dtype, MemorySpace.SBUF)
                accum = nb.tensor_tensor_arith(result, accum, scale, nki_ir.NisaArithOp.MULTIPLY)

            out_slices = build_slices(out_shape, out_tile_sizes, outer_indices)
            nb.dma_copy(hbm_map[out_val.name], accum, out_slices)
            return
        d, extent, ts = outer_loop_dims[depth]
        for i in range(ceildiv(extent, ts)):
            _outer_nested(depth + 1, {**outer_indices, d: i})

    _outer_nested(0, {})


def _emit_mixed_reduce_inline(
    nb: Builder, op, layouts: dict[str, Layout], hbm_map: dict[str, Value],
) -> None:
    inp_val = op.inputs[0]
    out_val = op.results[0]
    inp_layout = layouts[inp_val.name]
    inp_shape = inp_val.type.shape
    out_shape = out_val.type.shape
    kind = op.attrs["kind"]
    axis = set(op.attrs["axis"])
    dtype = inp_val.type.dtype

    f_axes = axis & set(inp_layout.f_dims)
    p_axes = axis & set(inp_layout.p_dims)
    reduced_f_dims = tuple(d for d in inp_layout.f_dims if d in f_axes)
    kept_f_dims = tuple(d for d in inp_layout.f_dims if d not in f_axes)
    reduced_p_dims = tuple(d for d in inp_layout.p_dims if d in p_axes)
    kept_p_dims = tuple(d for d in inp_layout.p_dims if d not in p_axes)

    f_kind = "sum" if kind == "mean" else kind
    p_kind = "sum" if kind == "mean" else kind
    total_reduced = prod(inp_shape[d] for d in axis)
    f_reduced_ext = prod(inp_shape[d] for d in reduced_f_dims)

    inp_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        inp_tile_sizes[d] = 1
    for d in kept_p_dims:
        inp_tile_sizes[d] = 1
    for i, d in enumerate(reduced_p_dims):
        inp_tile_sizes[d] = min(inp_shape[d], PARTITION_MAX) if i == len(reduced_p_dims) - 1 else 1
    for d in kept_f_dims:
        inp_tile_sizes[d] = 1
    for d in reduced_f_dims:
        inp_tile_sizes[d] = inp_shape[d]

    out_tile_sizes: dict[int, int] = {}
    for d in inp_layout.i_dims:
        out_tile_sizes[d] = 1
    for d in kept_p_dims:
        out_tile_sizes[d] = 1
    for d in reduced_p_dims:
        out_tile_sizes[d] = out_shape[d]
    for d in kept_f_dims:
        out_tile_sizes[d] = 1
    for d in reduced_f_dims:
        out_tile_sizes[d] = out_shape[d]

    outer_loop_dims = [
        (d, inp_shape[d], inp_tile_sizes[d])
        for d in sorted(set(inp_layout.i_dims) | set(kept_p_dims) | set(kept_f_dims))
        if inp_tile_sizes[d] < inp_shape[d]
    ]
    p_accum_dims = [
        (d, inp_shape[d], inp_tile_sizes[d])
        for d in sorted(reduced_p_dims)
        if inp_tile_sizes[d] < inp_shape[d]
    ]

    reduce_nki_op = REDUCE_OPS[f_kind]
    combine_op = COMBINE_OPS[p_kind]

    def _outer_nested(depth: int, outer_indices: dict[int, int]):
        if depth >= len(outer_loop_dims):
            accum = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
            accum = nb.memset(accum, COMBINE_INIT[p_kind])

            def _p_nested(depth2: int, p_indices: dict[int, int]):
                nonlocal accum
                if depth2 >= len(p_accum_dims):
                    indices = {**outer_indices, **p_indices}
                    p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, indices)
                    slices = build_slices(inp_shape, inp_tile_sizes, indices)
                    src = nb.alloc((p_ext, f_reduced_ext), dtype, MemorySpace.SBUF)
                    src = nb.dma_copy(src, hbm_map[inp_val.name], slices)
                    f_red = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
                    f_red = nb.tensor_reduce_arith(f_red, src, reduce_nki_op, num_r_dim=1, keepdims=True)
                    p_red = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
                    p_red = nb.cross_lane_reduce_arith(p_red, f_red, REDUCE_OPS[p_kind])
                    new_accum = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
                    accum = nb.tensor_tensor_arith(new_accum, accum, p_red, combine_op)
                    return
                d, extent, ts = p_accum_dims[depth2]
                for i in range(ceildiv(extent, ts)):
                    _p_nested(depth2 + 1, {**p_indices, d: i})

            if p_accum_dims:
                _p_nested(0, {})
            else:
                p_ext = clamped_extent(reduced_p_dims, inp_shape, inp_tile_sizes, outer_indices)
                slices = build_slices(inp_shape, inp_tile_sizes, outer_indices)
                src = nb.alloc((p_ext, f_reduced_ext), dtype, MemorySpace.SBUF)
                src = nb.dma_copy(src, hbm_map[inp_val.name], slices)
                f_red = nb.alloc((p_ext, 1), dtype, MemorySpace.SBUF)
                f_red = nb.tensor_reduce_arith(f_red, src, reduce_nki_op, num_r_dim=1, keepdims=True)
                accum = nb.cross_lane_reduce_arith(accum, f_red, REDUCE_OPS[p_kind])

            if kind == "mean":
                scale = nb.constant(1.0 / float(total_reduced), (1, 1), dtype, MemorySpace.SBUF)
                result = nb.alloc((1, 1), dtype, MemorySpace.SBUF)
                accum = nb.tensor_tensor_arith(result, accum, scale, nki_ir.NisaArithOp.MULTIPLY)

            out_slices = build_slices(out_shape, out_tile_sizes, outer_indices)
            nb.dma_copy(hbm_map[out_val.name], accum, out_slices)
            return
        d, extent, ts = outer_loop_dims[depth]
        for i in range(ceildiv(extent, ts)):
            _outer_nested(depth + 1, {**outer_indices, d: i})

    _outer_nested(0, {})
