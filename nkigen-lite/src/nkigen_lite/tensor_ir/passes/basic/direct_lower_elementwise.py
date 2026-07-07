"""Elementwise tile emission for tensor IR.

An elementwise segment (a list of elementwise ops) is lowered as one tiled
load→compute→store loop, keeping intermediates on-chip within the segment (no
HBM round-trip between them):

  - ``_emit_elementwise_segment`` emits one tiled loop for a segment: the last
    axis is the free dim, the penultimate is the partition, and the leading
    axes are unrolled one tile at a time.

Segmentation (grouping consecutive elementwise ops into one segment) currently
lives in ``direct_lower.lower_graph``, which passes one op per segment — the
grouping heuristic is to be reintroduced as the Phase-2 fusion-compatibility
predicate, at which point a segment carries more than one op again.

Layouts are canonical row-major throughout: HBM boundaries are layout-agnostic,
so every load/store addresses data by logical coordinate and no per-value
layout has to be agreed on. ``direct_lower.lower_graph`` calls
``_emit_elementwise_segment``; everything else here is a helper for it.
"""

from __future__ import annotations

from nkigen_lite.core import DType, Value, _DTYPE_BYTES
from nkigen_lite.nki_ir.ir import (
    Builder,
    MemorySpace,
)
from nkigen_lite.tensor_ir.passes.layout import Layout

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    BINARY_OPS,
    BITWISE_OPS,
    COMPARE_OPS,
    UNARY_OPS,
    canonical_layout,
    ceildiv,
    compute_tile_sizes,
    emit_binary_op,
    emit_unary_op,
    load_input_tile,
    on_chip_shape,
    store_output_tile,
)


# ---------------------------------------------------------------------------
# Elementwise segment emission
# ---------------------------------------------------------------------------


def _segment_dtype(ops: list) -> DType:
    """Widest dtype touched by a segment.

    Every value in a segment shares the rep's tile loop, so all
    ``compute_tile_sizes`` calls must use one byte budget — offsets are
    ``idx * tile_size`` and would misalign if values tiled differently. The
    binding constraint is the largest element, so tile to that; an all-bf16
    segment gets twice the F32-default tile width.
    """
    best = DType.F32
    best_bytes = 0
    for op in ops:
        for v in list(op.inputs) + list(op.results):
            b = _DTYPE_BYTES[v.type.dtype]
            if b > best_bytes:
                best, best_bytes = v.type.dtype, b
    return best


def _emit_elementwise_segment(
    nb: Builder, ops: list, hbm_map: dict[str, Value],
    slice_views: dict | None = None, resolve=None,
) -> None:
    """Emit a fused elementwise segment: one tiled loop for all ops.

    ``slice_views``/``resolve`` support slice-as-view: a foldable slice input
    (in ``slice_views``) is read from its source buffer with the slice's base
    offset added into the load, so no copy or HBM buffer is emitted for it.
    """
    slice_views = slice_views or {}

    # Use a canonical row-major layout for elementwise segments. HBM is
    # layout-agnostic, so all loads/stores address data by logical dimension
    # coordinates — the declared layout of individual values is irrelevant.
    rep_shape = ops[-1].results[0].type.shape

    seg_dtype = _segment_dtype(ops)
    rep_layout = canonical_layout(len(rep_shape))
    tile_sizes = compute_tile_sizes(rep_shape, rep_layout, seg_dtype)

    # Which values are produced within this segment (stay on-chip)
    segment_results = {r.name for op in ops for r in op.results}

    # Slice-view inputs read by this segment: map name -> (source HBM buffer,
    # per-dim base offsets) so the tile load adds the offset instead of copying.
    slice_srcs: dict[str, tuple[Value, tuple[int, ...]]] = {}
    for op in ops:
        for inp in op.inputs:
            if inp.name in slice_views and inp.name not in segment_results:
                sop = slice_views[inp.name]
                src_hbm = resolve(sop.inputs[0].name) if resolve else hbm_map[sop.inputs[0].name]
                slice_srcs[inp.name] = (src_hbm, tuple(sop.attrs["starts"]))

    # Loop dims
    loop_dims = [(d, rep_shape[d], tile_sizes[d])
                 for d in sorted(tile_sizes.keys())
                 if tile_sizes[d] < rep_shape[d]]

    def _emit_nested(depth: int, indices: dict[int, int]):
        if depth >= len(loop_dims):
            _emit_ew_tile(nb, ops, hbm_map, rep_layout,
                          indices, segment_results, seg_dtype,
                          slice_srcs=slice_srcs)
            return
        d, extent, ts = loop_dims[depth]
        for i in range(ceildiv(extent, ts)):
            _emit_nested(depth + 1, {**indices, d: i})

    _emit_nested(0, {})


def _emit_ew_tile(
    nb: Builder, ops: list, hbm_map: dict[str, Value],
    rep_layout: Layout, indices: dict[int, int],
    segment_results: set[str],
    seg_dtype: DType,
    slice_srcs: dict[str, tuple[Value, tuple[int, ...]]] | None = None,
) -> None:
    """Emit one tile of fused elementwise computation.

    Each input/output value is loaded/stored via ``load_input_tile`` /
    ``store_output_tile``, which tile the value in its own canonical row-major
    layout mapped onto the rep loop (``indices`` / ``rep_layout``) — the rep's
    shape and tile sizes never need to be threaded in.

    ``seg_dtype`` is the segment-wide tiling dtype (see ``_segment_dtype``);
    every ``compute_tile_sizes`` call in the segment must use it so the
    per-value slices align with the rep's loop.

    ``slice_srcs`` (slice-as-view) maps an input name to ``(source_buffer,
    starts)``: the input is a static-start, stride-1 slice with no buffer of
    its own, so its tile loads from the source with each per-dim start added
    into the load offset. The slice preserves rank, so the tile layout is
    computed from the *slice* shape (keeping the rep-loop alignment) and only
    the offsets shift.
    """
    slice_srcs = slice_srcs or {}
    tile_map: dict[str, Value] = {}

    # Load external inputs through the one shared helper (canonical row-major
    # for the input's own rank; HBM is layout-agnostic). A slice-as-view input
    # has no buffer of its own: it reads from the slice's source with the
    # slice's per-dim starts composed into the load offset.
    for op in ops:
        for inp in op.inputs:
            if inp.name in tile_map or inp.name in segment_results:
                continue
            if inp.name in slice_srcs:
                src_hbm, starts = slice_srcs[inp.name]
                tile_map[inp.name] = load_input_tile(
                    nb, src_hbm, inp.type.shape, inp.type.dtype, seg_dtype,
                    indices, rep_layout, offsets=starts,
                )
                continue
            hbm_val = hbm_map[inp.name]
            tile_map[inp.name] = load_input_tile(
                nb, hbm_val, hbm_val.type.shape, hbm_val.type.dtype, seg_dtype,
                indices, rep_layout,
            )

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
            # result = y (copy), then overwrite with x where cond > 0.
            # copy_predicated never evaluates arithmetic on the unselected
            # branch, so where(mask, scores, -inf) stays -inf/scores — the
            # old cond*x + (1-cond)*y form produced NaN whenever the
            # unselected branch was inf (0 * inf), which is exactly the
            # masked-attention pattern.
            shape = x_true.type.shape
            result = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_copy(result, y_false)
            # copy_predicated requires an integer pred_mask; cond is float
            # (1.0/0.0) here, so cast it through a u8 scratch tile.
            if cond.type.dtype not in (DType.U8, DType.U16, DType.U32):
                pred_u8 = nb.alloc(cond.type.shape, DType.U8, MemorySpace.SBUF)
                nb.tensor_copy(pred_u8, cond)
                nb.copy_predicated(result, pred_u8, x_true)
            else:
                nb.copy_predicated(result, cond, x_true)
            tile_map[out_name] = result
        elif op.opcode == "constant":
            out_shape = op.results[0].type.shape
            const_layout = canonical_layout(len(out_shape))
            const_tile_sizes = compute_tile_sizes(out_shape, const_layout, seg_dtype)
            const_tile = on_chip_shape(out_shape, const_layout, const_tile_sizes, indices)
            tile_map[out_name] = nb.constant(
                op.attrs["value"], const_tile, out_dtype, MemorySpace.SBUF
            )

    # Store each result through the shared helper (canonical layout of the
    # output's own rank).
    for op in ops:
        out_name = op.results[0].name
        if out_name in tile_map:
            store_output_tile(nb, hbm_map[out_name], tile_map[out_name],
                              seg_dtype, indices, rep_layout)
