"""Elementwise tile emission for tensor IR — linearize-and-tile flow.

An elementwise group (a list of elementwise ops sharing a broadcast result
shape) is lowered exactly the way an NKI kernel for elementwise ops is written:

  1. **Linearize**: collapse every tensor to 2D ``(P, F)`` where ``F`` is the
     last (contiguous) axis and ``P`` is the product of all leading axes. HBM
     buffers are row-major contiguous, so this is a zero-copy ``view`` — no data
     movement, no ND index bookkeeping.
  2. **Tile**: iterate the 2D domain in ``(128, free_tile)`` blocks — the
     partition axis at the 128-lane hardware max, the free axis at the largest
     power-of-two ``<= 512``.
  3. **Compute**: for each tile, load the operands, run the group's ops on
     the SBUF tiles, and store the results.

Why a flat 2D collapse is sound
-------------------------------
Every ``broadcast_to`` is materialized to its own HBM buffer before it reaches
here, so each operand reaching an elementwise op, when right-aligned to the
group's result shape and collapsed to 2D, has ``P in {1, rep_P}`` and
``F in {1, rep_F}``. A genuine *middle* broadcast (e.g. GQA head expansion) is
also materialized to its own HBM buffer. So each operand is one of: the full
``(rep_P, rep_F)`` tensor, a per-row scalar ``(rep_P, 1)``, a partition-broadcast
row ``(1, rep_F)``, or a scalar ``(1, 1)`` — exactly the four cases the shared
``emit_binary_op`` already broadcasts on-chip. The contract is asserted per
operand in ``_collapse_operand`` so a violation fails loudly rather than
miscomputing.
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import DType, Value
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
)

from nkigen_lite.tensor_ir.passes.basic.direct_lower_utils import (
    BINARY_OPS,
    BITWISE_OPS,
    COMPARE_OPS,
    UNARY_OPS,
    ceildiv,
    collapse_view,
    emit_binary_op,
    emit_unary_op,
    _materialize_broadcast,
)


# Largest free-dim tile: a power of two, capped at 512 (the tensor/vector
# engine's comfortable free width). Anything wider is just more iterations.
_FREE_TILE_MAX = 512


# ---------------------------------------------------------------------------
# Heuristic linearization + tiling
# ---------------------------------------------------------------------------


def _pf(shape: tuple[int, ...]) -> tuple[int, int]:
    """Collapse a logical shape to 2D ``(P, F)``: F = last axis, P = the rest.

    Rank 0 → (1, 1); rank 1 → (1, F). Matches the row-major flat order of the
    HBM buffer, so materializing this shape is a zero-copy view.
    """
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    return prod(shape[:-1]), shape[-1]


def _free_tile(free: int) -> int:
    """Largest power of two ``<= min(free, 512)`` (and ``>= 1``)."""
    cap = min(free, _FREE_TILE_MAX)
    t = 1
    while t * 2 <= cap:
        t *= 2
    return t


# ---------------------------------------------------------------------------
# Slice-as-view offset mapping
# ---------------------------------------------------------------------------


def _slice_2d_offset(
    src_shape: tuple[int, ...],
    slice_shape: tuple[int, ...],
    starts: tuple[int, ...],
) -> tuple[int, int]:
    """Map a stride-1 static slice to ``(row_off, col_off)`` in the source's
    collapsed 2D ``(P, F)`` view.

    ``col_off`` is the last-axis start. ``row_off`` is the flat row index of the
    slice's leading origin within the source's leading axes. This is only a
    valid 2D window when the slice's leading region is a contiguous run of
    source rows — i.e. once a leading axis is partially sliced, every inner
    leading axis is full. Anything else raises rather than silently
    miscomputing.
    """
    lead = src_shape[:-1]
    col_off = starts[-1] if starts else 0
    row_off = 0
    partial_seen = False
    for d in range(len(lead)):
        full = slice_shape[d] == src_shape[d] and starts[d] == 0
        if partial_seen and not full:
            raise NotImplementedError(
                f"slice-as-view into elementwise: non-contiguous leading slice "
                f"src={src_shape} slice={slice_shape} starts={starts}"
            )
        row_off += starts[d] * prod(src_shape[d + 1:-1])
        if not full:
            partial_seen = True
    return row_off, col_off


# ---------------------------------------------------------------------------
# Group emission
# ---------------------------------------------------------------------------


def _emit_elementwise_group(
    nb: Builder, ops: list, hbm_map: dict[str, Value],
    slice_views: dict | None = None, resolve=None,
) -> None:
    """Emit a fused elementwise group as one linearized, tiled loop.

    ``slice_views``/``resolve`` support slice-as-view: a foldable slice input
    (in ``slice_views``) is read from its source buffer with the slice's window
    composed into the load, so no copy or HBM buffer is emitted for it.
    """
    slice_views = slice_views or {}

    rep_shape = ops[-1].results[0].type.shape
    rep_P, rep_F = _pf(rep_shape)

    group_results = {r.name for op in ops for r in op.results}

    # Resolve every external input to a load plan once (the 2D ``view`` of its
    # HBM buffer is emitted a single time, not per tile). Each plan is
    # ``(src_2d, opP, opF, row_off, col_off)``; a size-1 collapsed axis is a
    # broadcast the load fans out on-chip, and ``row_off``/``col_off`` carry a
    # slice-as-view base window.
    load_plan: dict[str, tuple[Value, int, int, int, int]] = {}
    for op in ops:
        for inp in op.inputs:
            if inp.name in load_plan or inp.name in group_results:
                continue
            if inp.name in slice_views:
                sop = slice_views[inp.name]
                src_name = sop.inputs[0].name
                src_hbm = resolve(src_name) if resolve else hbm_map[src_name]
                src_shape = sop.inputs[0].type.shape
                row_off, col_off = _slice_2d_offset(
                    src_shape, inp.type.shape, tuple(sop.attrs["starts"]),
                )
                srcP, srcF = _pf(src_shape)
                opP, opF = _pf(inp.type.shape)
                load_plan[inp.name] = (
                    collapse_view(nb, src_hbm, srcP, srcF), opP, opF,
                    row_off, col_off,
                )
            else:
                hbm_val = hbm_map[inp.name]
                opP, opF = _collapse_operand(inp, rep_P, rep_F)
                load_plan[inp.name] = (
                    collapse_view(nb, hbm_val, *_pf(hbm_val.type.shape)),
                    opP, opF, 0, 0,
                )

    # Resolve each result to its collapsed-2D store view once.
    store_plan: dict[str, Value] = {}
    for op in ops:
        for r in op.results:
            hbm_dst = hbm_map[r.name]
            store_plan[r.name] = collapse_view(nb, hbm_dst, *_pf(hbm_dst.type.shape))

    p_tile = min(rep_P, PARTITION_MAX)
    f_tile = _free_tile(rep_F)

    for p_i in range(ceildiv(rep_P, p_tile)):
        p_off = p_i * p_tile
        p_size = min(p_tile, rep_P - p_off)
        for f_i in range(ceildiv(rep_F, f_tile)):
            f_off = f_i * f_tile
            f_size = min(f_tile, rep_F - f_off)
            _emit_ew_tile(
                nb, ops, group_results, load_plan, store_plan,
                p_off, p_size, f_off, f_size,
            )


def _collapse_operand(val: Value, rep_P: int, rep_F: int) -> tuple[int, int]:
    """Collapsed ``(P, F)`` of an operand, with the collapse-clean contract
    asserted (see module docstring)."""
    P, F = _pf(val.type.shape)
    assert P in (1, rep_P) and F in (1, rep_F), (
        f"elementwise operand {val.name} shape {val.type.shape} collapses to "
        f"({P}, {F}), not broadcast-compatible with rep ({rep_P}, {rep_F})"
    )
    return P, F


def _load_tile(
    nb: Builder, hbm_2d: Value, opP: int, opF: int,
    p_off: int, p_size: int, f_off: int, f_size: int,
    dtype: DType, row_off: int, col_off: int,
) -> Value:
    """Load one ``(P, F)`` tile from a collapsed-2D HBM view, honoring
    broadcast (size-1) axes and an optional slice-view base offset."""
    ps = 1 if opP == 1 else p_size
    fs = 1 if opF == 1 else f_size
    p = row_off + (0 if opP == 1 else p_off)
    f = col_off + (0 if opF == 1 else f_off)
    dst = nb.alloc((ps, fs), dtype, MemorySpace.SBUF)
    return nb.dma_copy(dst, hbm_2d, (DimSlice(p, ps), DimSlice(f, fs)))


def _emit_ew_tile(
    nb: Builder, ops: list,
    group_results: set[str],
    load_plan: dict[str, tuple[Value, int, int, int, int]],
    store_plan: dict[str, Value],
    p_off: int, p_size: int, f_off: int, f_size: int,
) -> None:
    """Emit one ``(p_size, f_size)`` tile of the fused group."""
    tile_map: dict[str, Value] = {}

    # Load external inputs (collapsed to 2D, broadcast axes honored).
    for op in ops:
        for inp in op.inputs:
            if inp.name in tile_map or inp.name in group_results:
                continue
            src_2d, opP, opF, row_off, col_off = load_plan[inp.name]
            tile_map[inp.name] = _load_tile(
                nb, src_2d, opP, opF, p_off, p_size, f_off, f_size,
                inp.type.dtype, row_off, col_off,
            )

    # Compute.
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
            # copy_predicated needs exactly-matching operand shapes; broadcast
            # any size-1 operand up to the tile shape first.
            out_shape = (p_size, f_size)
            cond = _materialize_broadcast(nb, cond, out_shape)
            x_true = _materialize_broadcast(nb, x_true, out_shape)
            y_false = _materialize_broadcast(nb, y_false, out_shape)
            # result = y (copy), then overwrite with x where cond > 0.
            # copy_predicated never evaluates arithmetic on the unselected
            # branch, so where(mask, scores, -inf) stays -inf/scores.
            result = nb.alloc(out_shape, out_dtype, MemorySpace.SBUF)
            nb.tensor_copy(result, y_false)
            if cond.type.dtype not in (DType.U8, DType.U16, DType.U32):
                pred_u8 = nb.alloc(cond.type.shape, DType.U8, MemorySpace.SBUF)
                nb.tensor_copy(pred_u8, cond)
                nb.copy_predicated(result, pred_u8, x_true)
            else:
                nb.copy_predicated(result, cond, x_true)
            tile_map[out_name] = result
        elif op.opcode == "constant":
            cP, cF = _pf(op.results[0].type.shape)
            shape = (1 if cP == 1 else p_size, 1 if cF == 1 else f_size)
            tile_map[out_name] = nb.constant(
                op.attrs["value"], shape, out_dtype, MemorySpace.SBUF
            )

    # Store each result to its collapsed-2D HBM buffer.
    for op in ops:
        out_name = op.results[0].name
        if out_name not in tile_map:
            continue
        nb.dma_copy(
            store_plan[out_name], tile_map[out_name],
            (DimSlice(p_off, p_size), DimSlice(f_off, f_size)),
        )
