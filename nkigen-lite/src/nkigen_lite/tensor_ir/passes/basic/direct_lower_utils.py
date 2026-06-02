"""Shared utilities for direct lowering modules.

Contains tiling helpers, HBM slice computation, op tables, and common
data-movement patterns used across elementwise, reduce, matmul, transpose,
reshape, slice, concat, and broadcast lowering.
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
from nkigen_lite.nki_ir import ir as nki_ir
from nkigen_lite.tensor_ir.passes.layout_solver import Layout


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------


def unravel(flat_idx: int, dims: list[int]) -> tuple[int, ...]:
    """Convert flat index to multi-dimensional indices (row-major)."""
    indices = []
    remaining = flat_idx
    for d in reversed(dims):
        indices.append(remaining % d)
        remaining //= d
    return tuple(reversed(indices))


def row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major strides for a shape."""
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return tuple(reversed(strides))


def flat_range_to_src_slices(
    flat_offset: int,
    n_elements: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> list[DimSlice]:
    """Convert a flat element range to source DimSlices.

    Given a contiguous range [flat_offset, flat_offset + n_elements) in
    row-major order, express it as a rectangular slice in the source shape.
    """
    rank = len(shape)
    start_indices = []
    remaining = flat_offset
    for s in strides:
        start_indices.append(remaining // s)
        remaining %= s

    inner_product = 1
    split_dim = rank - 1
    for d in range(rank - 1, -1, -1):
        if n_elements >= inner_product * shape[d] and start_indices[d] == 0:
            inner_product *= shape[d]
            split_dim = d - 1
        else:
            split_dim = d
            break

    slices = []
    for d in range(rank):
        if d < split_dim:
            slices.append(DimSlice(start_indices[d], 1))
        elif d == split_dim:
            size_on_dim = n_elements // (prod(shape[d + 1:]) if d < rank - 1 else 1)
            slices.append(DimSlice(start_indices[d], size_on_dim))
        else:
            slices.append(DimSlice(0, shape[d]))
    return slices


# ---------------------------------------------------------------------------
# Tiling utilities
# ---------------------------------------------------------------------------


def compute_tile_sizes(shape: tuple[int, ...], layout: Layout) -> dict[int, int]:
    """Compute per-dimension tile sizes.

    I-dims: 1, outer P-dims: 1, innermost P-dim: min(extent, 128), F-dims: full.
    """
    tiles: dict[int, int] = {}
    for d in layout.i_dims:
        tiles[d] = 1
    p_dims = layout.p_dims
    for i, d in enumerate(p_dims):
        tiles[d] = min(shape[d], PARTITION_MAX) if i == len(p_dims) - 1 else 1
    for d in layout.f_dims:
        tiles[d] = shape[d]
    return tiles


def on_chip_shape(
    shape: tuple[int, ...],
    layout: Layout,
    tile_sizes: dict[int, int],
    indices: dict[int, int],
) -> tuple[int, int]:
    """Compute the 2D on-chip tile shape (P, F) for the current iteration."""
    def _extent(d: int) -> int:
        ts = tile_sizes[d]
        if ts >= shape[d]:
            return shape[d]
        idx = indices.get(d, 0)
        return min(ts, shape[d] - idx * ts)

    p = prod(_extent(d) for d in layout.p_dims) if layout.p_dims else 1
    f = prod(_extent(d) for d in layout.f_dims) if layout.f_dims else 1
    return (p, f)


def clamped_extent(
    dims: tuple[int, ...],
    shape: tuple[int, ...],
    tile_sizes: dict[int, int],
    indices: dict[int, int],
) -> int:
    """Product of per-dim extents, clamped on boundaries."""
    result = 1
    for d in dims:
        ts = tile_sizes[d]
        if ts >= shape[d]:
            result *= shape[d]
        else:
            idx = indices.get(d, 0)
            result *= min(ts, shape[d] - idx * ts)
    return result


def build_slices(
    shape: tuple[int, ...],
    tile_sizes: dict[int, int],
    indices: dict[int, int],
) -> list[DimSlice]:
    """Build DimSlice list for DMA, one per dimension."""
    slices = []
    for d in range(len(shape)):
        ts = tile_sizes.get(d, shape[d])
        if ts >= shape[d]:
            slices.append(DimSlice(0, shape[d]))
        else:
            idx = indices.get(d, 0)
            off = idx * ts
            size = min(ts, shape[d] - off)
            slices.append(DimSlice(off, size))
    return slices


def map_indices(
    val_layout: Layout, rep_layout: Layout, indices: dict[int, int],
) -> dict[int, int]:
    """Map loop indices from the rep's dim positions to a value's dim positions."""
    mapped: dict[int, int] = {}
    for val_group, rep_group in [
        (val_layout.i_dims, rep_layout.i_dims),
        (val_layout.p_dims, rep_layout.p_dims),
        (val_layout.f_dims, rep_layout.f_dims),
    ]:
        for k, val_d in enumerate(val_group):
            if k < len(rep_group) and rep_group[k] in indices:
                mapped[val_d] = indices[rep_group[k]]
    return mapped


def hbm_slices(
    shape: tuple[int, ...],
    layout: Layout,
    tile_sizes: dict[int, int],
    indices: dict[int, int],
    rep_layout: Layout,
) -> list[DimSlice]:
    """Build DimSlice list for a DMA copy, mapping rep loop indices to value dims."""
    val_indices = map_indices(layout, rep_layout, indices)
    slices = []
    for d in range(len(shape)):
        ts = tile_sizes[d]
        idx = val_indices.get(d, 0)
        if ts >= shape[d]:
            slices.append(DimSlice(0, shape[d]))
        else:
            off = idx * ts
            size = min(ts, shape[d] - off)
            slices.append(DimSlice(off, size))
    return slices


# ---------------------------------------------------------------------------
# Output slice helper
# ---------------------------------------------------------------------------


def build_out_slices(
    batch_idx: tuple[int, ...],
    p_off: int,
    p_size: int,
    f_size: int,
    out_rank: int,
) -> list[DimSlice]:
    """Build destination DimSlice for an output tile."""
    slices = []
    for bi in batch_idx:
        slices.append(DimSlice(bi, 1))
    if out_rank >= 2:
        slices.append(DimSlice(p_off, p_size))
        slices.append(DimSlice(0, f_size))
    else:
        slices.append(DimSlice(p_off, p_size))
    return slices


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


def broadcast_partition(nb: Builder, src: Value, target_shape: tuple[int, int]) -> Value:
    """Replicate a (1, F) tile to (P, F) via HBM scratch round-trip."""
    p, f = target_shape
    scratch = nb.alloc((1, f), src.type.dtype, MemorySpace.HBM)
    nb.dma_copy(scratch, src, (DimSlice(0, 1), DimSlice(0, f)))
    dst = nb.alloc((p, f), src.type.dtype, MemorySpace.SBUF)
    return nb.dma_copy(dst, scratch, (DimSlice(0, p, stride=0), DimSlice(0, f)))


# ---------------------------------------------------------------------------
# Op tables
# ---------------------------------------------------------------------------


BINARY_OPS: dict[str, nki_ir.NisaArithOp] = {
    "add": nki_ir.NisaArithOp.ADD,
    "sub": nki_ir.NisaArithOp.SUBTRACT,
    "mul": nki_ir.NisaArithOp.MULTIPLY,
    "maximum": nki_ir.NisaArithOp.MAXIMUM,
    "minimum": nki_ir.NisaArithOp.MINIMUM,
}

COMMUTATIVE_OPS = {
    nki_ir.NisaArithOp.ADD,
    nki_ir.NisaArithOp.MULTIPLY,
    nki_ir.NisaArithOp.MAXIMUM,
    nki_ir.NisaArithOp.MINIMUM,
}

UNARY_OPS: dict[str, nki_ir.NisaActivationOp | None] = {
    "neg": None,
    "exp": nki_ir.NisaActivationOp.EXP,
    "log": nki_ir.NisaActivationOp.LOG,
    "sqrt": nki_ir.NisaActivationOp.SQRT,
    "rsqrt": nki_ir.NisaActivationOp.RSQRT,
    "tanh": nki_ir.NisaActivationOp.TANH,
    "relu": nki_ir.NisaActivationOp.RELU,
    "gelu": nki_ir.NisaActivationOp.GELU,
    "sigmoid": nki_ir.NisaActivationOp.SIGMOID,
    "silu": nki_ir.NisaActivationOp.SILU,
    "reciprocal": nki_ir.NisaActivationOp.RECIPROCAL,
}

REDUCE_OPS: dict[str, nki_ir.NisaReduceOp] = {
    "sum": nki_ir.NisaReduceOp.ADD,
    "max": nki_ir.NisaReduceOp.MAX,
    "min": nki_ir.NisaReduceOp.MIN,
    "mean": nki_ir.NisaReduceOp.ADD,
}

COMBINE_OPS: dict[str, nki_ir.NisaArithOp] = {
    "sum": nki_ir.NisaArithOp.ADD,
    "mean": nki_ir.NisaArithOp.ADD,
    "max": nki_ir.NisaArithOp.MAXIMUM,
    "min": nki_ir.NisaArithOp.MINIMUM,
}

COMBINE_INIT: dict[str, float] = {
    "sum": 0.0,
    "mean": 0.0,
    "max": float("-inf"),
    "min": float("inf"),
}

ELEMENTWISE_OPCODES = frozenset({
    "add", "sub", "mul", "maximum", "minimum",
    "neg", "exp", "log", "sqrt", "rsqrt", "tanh", "relu", "gelu",
    "sigmoid", "silu", "reciprocal", "constant",
})


# ---------------------------------------------------------------------------
# Compute emission helpers
# ---------------------------------------------------------------------------


def emit_binary_op(nb: Builder, out_dtype: DType, a: Value, b: Value, opcode: str) -> Value:
    """Emit a binary elementwise op with broadcast alignment."""
    arith_op = BINARY_OPS[opcode]
    if a.type.shape == b.type.shape:
        dst = nb.alloc(a.type.shape, out_dtype, MemorySpace.SBUF)
        return nb.tensor_tensor_arith(dst, a, b, arith_op)

    ap, af = a.type.shape
    bp, bf = b.type.shape
    out_shape = (max(ap, bp), max(af, bf))

    if bf == 1 and (bp == ap or bp == 1):
        if bp == 1 and out_shape[0] > 1:
            b = broadcast_partition(nb, b, (out_shape[0], bf))
        dst = nb.alloc(out_shape, out_dtype, MemorySpace.SBUF)
        return nb.tensor_scalar_arith(dst, a, b, arith_op)

    if af == 1 and (ap == bp or ap == 1):
        if arith_op in COMMUTATIVE_OPS:
            if ap == 1 and out_shape[0] > 1:
                a = broadcast_partition(nb, a, (out_shape[0], af))
            dst = nb.alloc(out_shape, out_dtype, MemorySpace.SBUF)
            return nb.tensor_scalar_arith(dst, b, a, arith_op)
        if ap == 1 and out_shape[0] > 1:
            a = broadcast_partition(nb, a, out_shape)
        else:
            a = nb.broadcast(a, out_shape)
        dst = nb.alloc(out_shape, out_dtype, MemorySpace.SBUF)
        return nb.tensor_tensor_arith(dst, a, b, arith_op)

    if ap == 1 and bp > 1 and af == bf:
        a = broadcast_partition(nb, a, out_shape)
    elif bp == 1 and ap > 1 and af == bf:
        b = broadcast_partition(nb, b, out_shape)
    else:
        raise NotImplementedError(
            f"binary shapes {a.type.shape} / {b.type.shape} not alignable"
        )
    dst = nb.alloc(out_shape, out_dtype, MemorySpace.SBUF)
    return nb.tensor_tensor_arith(dst, a, b, arith_op)


def emit_unary_op(nb: Builder, out_dtype: DType, src: Value, opcode: str) -> Value:
    """Emit a unary elementwise op."""
    act_op = UNARY_OPS[opcode]
    dst = nb.alloc(src.type.shape, out_dtype, MemorySpace.SBUF)
    if act_op is None:
        p = src.type.shape[0]
        neg_one = nb.constant(-1.0, (p, 1), src.type.dtype, MemorySpace.SBUF)
        return nb.tensor_scalar_arith(dst, src, neg_one, nki_ir.NisaArithOp.MULTIPLY)
    return nb.activation(dst, src, act_op)
