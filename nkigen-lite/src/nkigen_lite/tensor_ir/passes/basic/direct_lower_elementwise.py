"""Elementwise tile emission for tensor IR — linearize-and-tile flow.

A single elementwise op is lowered exactly the way an NKI kernel for an
elementwise op is written:

  1. **Linearize**: collapse every tensor to 2D ``(P, F)`` where ``F`` is the
     last (contiguous) axis and ``P`` is the product of all leading axes. HBM
     buffers are row-major contiguous, so this is a zero-copy ``view`` — no data
     movement, no ND index bookkeeping.
  2. **Tile**: iterate the 2D domain in ``(128, free_tile)`` blocks — the
     partition axis at the 128-lane hardware max, the free axis at the largest
     power-of-two ``<= 512``.
  3. **Compute**: for each tile, load the operands, run the op on the SBUF
     tiles, and store the result.

Why a flat 2D collapse is sound
-------------------------------
Every ``broadcast_to`` is materialized to its own HBM buffer before it reaches
here, so each operand reaching an elementwise op, when right-aligned to the
result shape and collapsed to 2D, has ``P in {1, rep_P}`` and
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
from typing import NamedTuple

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
from nkigen_lite.tensor_ir.passes.basic.direct_lower_alloc import Scratch


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
# Op emission
# ---------------------------------------------------------------------------


class LoadPlan(NamedTuple):
    """How to load one input, resolved once and reused per tile.

    ``src_2d`` is the collapsed-2D ``view`` of the input's HBM buffer. A size-1
    ``opP``/``opF`` marks a collapsed axis the load fans out on-chip (broadcast).
    """

    src_2d: Value
    opP: int
    opF: int


def _emit_elementwise_op(nb: Builder, op, hbm_map: dict[str, Value], scratch: Scratch) -> None:
    """Emit one elementwise op as a linearized, tiled load→compute→store loop."""
    rep_P, rep_F = _pf(op.results[0].type.shape)

    # Resolve every input to a load plan once (the 2D ``view`` of its HBM buffer
    # is emitted a single time, not per tile).
    load_plan: dict[str, LoadPlan] = {}
    for inp in op.inputs:
        if inp.name in load_plan:
            continue
        hbm_val = hbm_map[inp.name]
        opP, opF = _collapse_operand(inp, rep_P, rep_F)
        load_plan[inp.name] = LoadPlan(
            collapse_view(nb, hbm_val, *_pf(hbm_val.type.shape)), opP, opF,
        )

    # Resolve the result to its collapsed-2D store view once.
    hbm_dst = hbm_map[op.results[0].name]
    store_2d = collapse_view(nb, hbm_dst, *_pf(hbm_dst.type.shape))

    p_tile = min(rep_P, PARTITION_MAX)
    f_tile = _free_tile(rep_F)

    for p_i in range(ceildiv(rep_P, p_tile)):
        p_off = p_i * p_tile
        p_size = min(p_tile, rep_P - p_off)
        for f_i in range(ceildiv(rep_F, f_tile)):
            f_off = f_i * f_tile
            f_size = min(f_tile, rep_F - f_off)
            _emit_ew_tile(
                nb, op, load_plan, store_2d,
                p_off, p_size, f_off, f_size, scratch,
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
    scratch: Scratch, hbm_2d: Value, opP: int, opF: int,
    p_off: int, p_size: int, f_off: int, f_size: int,
    dtype: DType,
) -> Value:
    """Load one ``(P, F)`` tile from a collapsed-2D HBM view, honoring
    broadcast (size-1) axes."""
    ps = 1 if opP == 1 else p_size
    fs = 1 if opF == 1 else f_size
    p = 0 if opP == 1 else p_off
    f = 0 if opF == 1 else f_off
    return scratch.load(hbm_2d, (DimSlice(p, ps), DimSlice(f, fs)), (ps, fs), dtype)


def _emit_ew_tile(
    nb: Builder, op,
    load_plan: dict[str, LoadPlan],
    store_2d: Value,
    p_off: int, p_size: int, f_off: int, f_size: int,
    scratch: Scratch,
) -> None:
    """Emit one ``(p_size, f_size)`` tile: load inputs, compute, store."""
    out_dtype = op.results[0].type.dtype

    # Load inputs (collapsed to 2D, broadcast axes honored).
    tile_map: dict[str, Value] = {}
    for inp in op.inputs:
        if inp.name in tile_map:
            continue
        plan = load_plan[inp.name]
        tile_map[inp.name] = _load_tile(
            scratch, plan.src_2d, plan.opP, plan.opF, p_off, p_size, f_off, f_size,
            inp.type.dtype,
        )

    # Compute.
    if op.opcode in BINARY_OPS or op.opcode in BITWISE_OPS or op.opcode in COMPARE_OPS:
        lhs = tile_map[op.inputs[0].name]
        rhs = tile_map[op.inputs[1].name]
        result = emit_binary_op(nb, out_dtype, lhs, rhs, op.opcode)
    elif op.opcode in UNARY_OPS:
        src = tile_map[op.inputs[0].name]
        result = emit_unary_op(nb, out_dtype, src, op.opcode)
    elif op.opcode == "cast":
        src = tile_map[op.inputs[0].name]
        result = nb.alloc(src.type.shape, out_dtype, MemorySpace.SBUF)
        nb.tensor_copy(result, src)
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
    elif op.opcode == "constant":
        cP, cF = _pf(op.results[0].type.shape)
        shape = (1 if cP == 1 else p_size, 1 if cF == 1 else f_size)
        result = nb.constant(
            op.attrs["value"], shape, out_dtype, MemorySpace.SBUF
        )
    else:
        raise NotImplementedError(f"elementwise opcode {op.opcode!r} not supported")

    # Store the result to its collapsed-2D HBM buffer.
    nb.dma_copy(
        store_2d, result,
        (DimSlice(p_off, p_size), DimSlice(f_off, f_size)),
    )
