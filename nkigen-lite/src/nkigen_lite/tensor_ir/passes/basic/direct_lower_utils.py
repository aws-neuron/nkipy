"""Shared utilities for direct lowering modules.

Contains tiling helpers, HBM slice computation, op tables, and common
data-movement patterns used across elementwise, reduce, matmul, transpose,
reshape, slice, concat, and broadcast lowering.
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import DType, Value, _DTYPE_BYTES
from nkigen_lite.nki_ir.ir import (
    Builder,
    DimSlice,
    MemorySpace,
    PARTITION_MAX,
    SBUF_PER_PARTITION_BYTES,
)
from nkigen_lite.nki_ir import ir as nki_ir
from nkigen_lite.tensor_ir.passes.layout import Layout


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def max_free_elems(dtype: DType) -> int:
    """Largest free-dim extent (per partition row) for one data-movement tile.

    Data-movement tiles span the full partition (up to 128 rows), so the
    free dim is what determines per-partition byte usage. Several such tiles
    can be live at once (load + store, double-buffering, plus the segment's
    working set), and the compiler's allocator must fit them all in one
    partition's SBUF. Budget a conservative fraction of capacity so a handful
    of concurrent tiles stay well under the limit. Returns at least 1.
    """
    elem_bytes = _DTYPE_BYTES[dtype]
    return max(1, (SBUF_PER_PARTITION_BYTES // 4) // elem_bytes)


def collapse_view(nb: Builder, hbm: Value, lead: int, last: int) -> Value:
    """Reinterpret a row-major HBM buffer as 2D ``(lead, last)``.

    ``lead * last`` must equal the buffer's element count. A reshape preserves
    row-major flat order and HBM buffers are contiguous, so this is a
    zero-copy view (no data movement). Returns the original buffer untouched
    when it already has the target 2D shape.
    """
    if tuple(hbm.type.shape) == (lead, last):
        return hbm
    return nb.view(hbm, (lead, last))


def iter_pf_tiles(P: int, F: int, dtype: DType):
    """Yield ``(p_off, p_size, f_off, f_size)`` over a 2D ``(P, F)`` extent.

    The single place that encodes the data-movement tile budget: partition
    tiled at PARTITION_MAX (128 lanes), free dim tiled at
    ``max_free_elems(dtype)`` so each tile fits the per-partition SBUF
    budget. Every collapse-onto-partition fast path should loop with this
    rather than hand-rolling its own bounds.
    """
    cap = max_free_elems(dtype)
    for p_i in range(ceildiv(P, PARTITION_MAX)):
        p_off = p_i * PARTITION_MAX
        p_size = min(PARTITION_MAX, P - p_off)
        for f_i in range(ceildiv(F, cap)):
            f_off = f_i * cap
            f_size = min(cap, F - f_off)
            yield p_off, p_size, f_off, f_size


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


def flat_range_to_src_chunks(
    flat_offset: int,
    n_elements: int,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> list[tuple[list[DimSlice], int]]:
    """Decompose a contiguous flat range into maximal rectangular sub-slices.

    A contiguous flat range is a *single* rectangle only when it starts at a
    leading-dim boundary and stays within one. A range that crosses such a
    boundary (e.g. collapsing ``(3, 100, 8)`` into ``(300, 8)`` and loading a
    128-row tile) cannot be a single rectangle.

    This splits ``[flat_offset, flat_offset + n_elements)`` into a list of
    ``(src_slices, covered)`` pairs, each a maximal rectangle, that together
    cover the whole range. Returns a single chunk for the aligned fast path,
    so callers pay no extra DMAs when the range already is a rectangle.
    """
    rank = len(shape)
    chunks: list[tuple[list[DimSlice], int]] = []
    pos = flat_offset
    end = flat_offset + n_elements
    while pos < end:
        budget = end - pos
        start_indices = []
        remaining = pos
        for s in strides:
            start_indices.append(remaining // s)
            remaining %= s

        # Grow the largest rectangle from the innermost dim: absorb whole dims
        # while they start at 0 and fit the budget. ``split_dim`` is the first
        # dim (from the right) we can only partially traverse; -1 means the
        # remaining range is itself one full-array rectangle.
        inner = 1
        split_dim = -1
        for d in range(rank - 1, -1, -1):
            if start_indices[d] == 0 and inner * shape[d] <= budget:
                inner *= shape[d]
            else:
                split_dim = d
                break

        slices = []
        if split_dim < 0:
            for d in range(rank):
                slices.append(DimSlice(0, shape[d]))
            covered = inner
        else:
            avail = shape[split_dim] - start_indices[split_dim]
            count = min(budget // inner, avail)
            for d in range(rank):
                if d < split_dim:
                    slices.append(DimSlice(start_indices[d], 1))
                elif d == split_dim:
                    slices.append(DimSlice(start_indices[d], count))
                else:
                    slices.append(DimSlice(0, shape[d]))
            covered = inner * count

        chunks.append((slices, covered))
        pos += covered
    return chunks


def prefix_row_segments(r0, p, free, in_shape, in_strides, out_shape, out_strides):
    """Split partition rows [r0, r0+p) into segments that are a single
    rectangle on *both* the source and destination.

    Both shapes are treated as ``(rows, free)`` row-major data. A 128-row tile
    only forms one source/destination rectangle when its row boundaries
    coincide with a leading-dim boundary of *each* shape. When the collapsed
    row count is not itself such a boundary (e.g. reshaping (10,30,20) ->
    (300,20): tiles cut at rows 128/256, mid-way through the 30-row source
    blocks), the tile spans several rectangles. We find the row-boundary cut
    points each side induces, merge them, and yield ``(rows, src_slices,
    dst_slices)`` per segment — each a single rectangle on both sides.
    """
    def cut_rows(chunks):
        # Cumulative covered rows at each chunk boundary (relative to r0).
        cuts, acc = [], 0
        for _slices, covered in chunks:
            acc += covered // free
            cuts.append(acc)
        return cuts

    src_chunks = flat_range_to_src_chunks(r0 * free, p * free, in_shape, in_strides)
    dst_chunks = flat_range_to_src_chunks(r0 * free, p * free, out_shape, out_strides)
    bounds = sorted(set(cut_rows(src_chunks) + cut_rows(dst_chunks)))

    lo = 0
    for hi in bounds:
        rows = hi - lo
        (src_slices, _), = flat_range_to_src_chunks(
            (r0 + lo) * free, rows * free, in_shape, in_strides)
        (dst_slices, _), = flat_range_to_src_chunks(
            (r0 + lo) * free, rows * free, out_shape, out_strides)
        yield rows, src_slices, dst_slices
        lo = hi


# ---------------------------------------------------------------------------
# Tiling utilities
# ---------------------------------------------------------------------------


def compute_tile_sizes(
    shape: tuple[int, ...], layout: Layout, dtype: "DType | None" = None
) -> dict[int, int]:
    """Compute per-dimension tile sizes.

    I-dims: 1, outer P-dims: 1, innermost P-dim: min(extent, 128). F-dims are
    full except the innermost, which is capped so one tile row fits a single
    SBUF partition (a vocab-wide row, e.g. [1, 37984], would otherwise need a
    150 KB tile and overflow once a few are live).  The lowering loops already
    iterate ceil(extent / tile) tiles, so a smaller innermost-F tile is just
    more iterations.  ``dtype`` selects the byte budget; when omitted, the
    conservative F32 budget is used.
    """
    tiles: dict[int, int] = {}
    for d in layout.i_dims:
        tiles[d] = 1
    p_dims = layout.p_dims
    for i, d in enumerate(p_dims):
        tiles[d] = min(shape[d], PARTITION_MAX) if i == len(p_dims) - 1 else 1
    f_dims = layout.f_dims
    cap = max_free_elems(dtype) if dtype is not None else max_free_elems(DType.F32)
    for i, d in enumerate(f_dims):
        if i == len(f_dims) - 1:
            # Innermost F: cap to the per-partition budget divided by the outer
            # F extents already committed (so the full on-chip free size stays
            # within budget).
            outer_f = 1
            for j, dd in enumerate(f_dims):
                if j != i:
                    outer_f *= shape[dd]
            tiles[d] = max(1, min(shape[d], cap // max(1, outer_f)))
        else:
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
# Canonical row-major layout + the shared tile load/store helpers
# ---------------------------------------------------------------------------


def canonical_layout(rank: int) -> Layout:
    """Canonical row-major layout: last dim = F, penultimate = P, rest = I.

    HBM is layout-agnostic (row-major contiguous), so every elementwise
    load/store addresses data by logical coordinates in this one layout — the
    declared layout of individual values is irrelevant across a segment
    boundary. This is the normal form the tile machinery loops over.
    """
    if rank == 0:
        return Layout(i_dims=(), p_dims=(), f_dims=())
    if rank == 1:
        return Layout(i_dims=(), p_dims=(), f_dims=(0,))
    f_dims = (rank - 1,)
    p_dims = (rank - 2,)
    i_dims = tuple(range(rank - 2))
    return Layout(i_dims=i_dims, p_dims=p_dims, f_dims=f_dims)


def load_input_tile(
    nb: Builder,
    src_hbm: Value,
    val_shape: tuple[int, ...],
    val_dtype: DType,
    seg_dtype: DType,
    indices: dict[int, int],
    rep_layout: Layout,
    offsets: tuple[int, ...] | None = None,
) -> Value:
    """Materialize one tile of a viewed value from HBM into SBUF.

    The single load path for elementwise emission: it computes the value's own
    canonical row-major layout, tiles it with the segment dtype, maps the rep
    loop indices onto its dims, and DMAs the tile in. ``offsets`` composes an
    access descriptor into the load — a slice-as-view input passes its per-dim
    ``starts`` so the tile reads ``src_hbm`` shifted by the slice window with no
    copy of its own (the slice preserves rank, so the tile layout still comes
    from ``val_shape`` and only the base offset shifts).
    """
    val_layout = canonical_layout(len(val_shape))
    val_tile_sizes = compute_tile_sizes(val_shape, val_layout, seg_dtype)
    val_tile = on_chip_shape(val_shape, val_layout, val_tile_sizes, indices)
    slices = hbm_slices(val_shape, val_layout, val_tile_sizes, indices, rep_layout)
    if offsets is not None:
        slices = [DimSlice(s.offset + off, s.size, s.stride)
                  for s, off in zip(slices, offsets)]
    dst = nb.alloc(val_tile, val_dtype, MemorySpace.SBUF)
    return nb.dma_copy(dst, src_hbm, slices)


def store_output_tile(
    nb: Builder,
    hbm_dst: Value,
    tile: Value,
    seg_dtype: DType,
    indices: dict[int, int],
    rep_layout: Layout,
) -> None:
    """Store one SBUF tile back to its HBM buffer — the store counterpart to
    ``load_input_tile``, using the destination's own canonical layout."""
    out_layout = canonical_layout(len(hbm_dst.type.shape))
    out_tile_sizes = compute_tile_sizes(hbm_dst.type.shape, out_layout, seg_dtype)
    slices = hbm_slices(hbm_dst.type.shape, out_layout, out_tile_sizes,
                        indices, rep_layout)
    nb.dma_copy(hbm_dst, tile, slices)


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

BITWISE_OPS: dict[str, nki_ir.NisaBitvecOp] = {
    "bitwise_and": nki_ir.NisaBitvecOp.AND,
    "bitwise_or": nki_ir.NisaBitvecOp.OR,
    "bitwise_xor": nki_ir.NisaBitvecOp.XOR,
}

COMPARE_OPS: dict[str, nki_ir.NisaArithOp] = {
    "equal": nki_ir.NisaArithOp.IS_EQ,
    "not_equal": nki_ir.NisaArithOp.IS_NE,
    "greater": nki_ir.NisaArithOp.IS_GT,
    "greater_equal": nki_ir.NisaArithOp.IS_GE,
    "less": nki_ir.NisaArithOp.IS_LT,
    "less_equal": nki_ir.NisaArithOp.IS_LE,
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
    "abs": nki_ir.NisaActivationOp.ABS,
    "sign": nki_ir.NisaActivationOp.SIGN,
    "sin": nki_ir.NisaActivationOp.SIN,
    "arctan": nki_ir.NisaActivationOp.ARCTAN,
    "floor": None,  # handled by _emit_floor special case
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
    "sigmoid", "silu", "reciprocal", "abs", "sign", "sin", "arctan", "floor",
    "constant", "cast",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "where",
})


# ---------------------------------------------------------------------------
# Compute emission helpers
# ---------------------------------------------------------------------------


def _materialize_broadcast(nb: Builder, x: Value, out_shape: tuple[int, int]) -> Value:
    """Broadcast a tile to ``out_shape`` on both axes, materializing the result.

    Partition (P) broadcast goes through the HBM scratch round-trip
    (``broadcast_partition``); free (F) broadcast replicates the per-row
    scalar with a ones-multiply (``tensor_scalar_arith``). Used by the
    compare/bitwise paths, whose tensor_tensor ops require exactly matching
    shapes — unlike arith, which has a native tensor_scalar form.
    """
    p, f = x.type.shape
    out_p, out_f = out_shape
    if p == 1 and out_p > 1:
        x = broadcast_partition(nb, x, (out_p, f))
        p = out_p
    if f == 1 and out_f > 1:
        ones = nb.constant(1.0, (p, out_f), x.type.dtype, MemorySpace.SBUF)
        rep = nb.alloc((p, out_f), x.type.dtype, MemorySpace.SBUF)
        x = nb.tensor_scalar_arith(rep, ones, x, nki_ir.NisaArithOp.MULTIPLY)
    return x


def emit_binary_op(nb: Builder, out_dtype: DType, a: Value, b: Value, opcode: str) -> Value:
    """Emit a binary elementwise op with broadcast alignment."""
    if opcode in COMPARE_OPS or opcode in BITWISE_OPS:
        # tensor_tensor_compare / _bitvec require exactly matching shapes, so
        # materialize size-1 operands on both axes (the arith path below can
        # instead use the native tensor_scalar form for F-size-1 operands).
        if a.type.shape != b.type.shape:
            ap, af = a.type.shape
            bp, bf = b.type.shape
            out_shape = (max(ap, bp), max(af, bf))
            a = _materialize_broadcast(nb, a, out_shape)
            b = _materialize_broadcast(nb, b, out_shape)
        if opcode in COMPARE_OPS:
            # Comparison ops produce same dtype as input (1.0/0.0 float)
            dst = nb.alloc(a.type.shape, a.type.dtype, MemorySpace.SBUF)
            return nb.tensor_tensor_compare(dst, a, b, COMPARE_OPS[opcode])
        dst = nb.alloc(a.type.shape, out_dtype, MemorySpace.SBUF)
        return nb.tensor_tensor_bitvec(dst, a, b, BITWISE_OPS[opcode])
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
    if opcode == "floor":
        return _emit_floor(nb, out_dtype, src)
    if opcode == "neg":
        dst = nb.alloc(src.type.shape, out_dtype, MemorySpace.SBUF)
        p = src.type.shape[0]
        neg_one = nb.constant(-1.0, (p, 1), src.type.dtype, MemorySpace.SBUF)
        return nb.tensor_scalar_arith(dst, src, neg_one, nki_ir.NisaArithOp.MULTIPLY)
    act_op = UNARY_OPS[opcode]
    dst = nb.alloc(src.type.shape, out_dtype, MemorySpace.SBUF)
    return nb.activation(dst, src, act_op)


def _emit_floor(nb: Builder, out_dtype: DType, src: Value) -> Value:
    """Emit floor(x) using the NKI compiler's compare+select pattern.

    Pattern (from nki.language.operators.floor):
      1. casted      = tensor_copy(x → i32)    — truncate toward zero
      2. casted_back = tensor_copy(casted → f)  — back to float
      3. condition   = casted_back > x          — true when trunc overshot
      4. cond_not    = condition XOR 1          — logical NOT
      5. casted_m1   = casted - 1              — trunc minus one (int)
      6. larger      = condition * casted_m1    — selected when overshot
      7. smaller     = cond_not * casted        — selected otherwise
      8. result      = larger + smaller         — final floor (cast to out_dtype)

    Uses integer arithmetic for the conditional select to avoid float
    precision issues in the correction step.
    """
    shape = src.type.shape
    p = shape[0]

    # trunc(x) via int32 cast (rounds toward zero)
    casted = nb.alloc(shape, DType.I32, MemorySpace.SBUF)
    nb.tensor_copy(casted, src)

    # cast back to float for comparison
    casted_back = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
    nb.tensor_copy(casted_back, casted)

    # condition = (casted_back > x): uint8 predicate, 1 when trunc overshot
    condition = nb.alloc(shape, DType.U8, MemorySpace.SBUF)
    nb.tensor_tensor_compare(condition, casted_back, src, nki_ir.NisaArithOp.IS_GT)

    # cond_not = condition XOR 1 (logical NOT)
    one_u8 = nb.constant(1.0, (p, 1), DType.U8, MemorySpace.SBUF)
    cond_not = nb.alloc(shape, DType.U8, MemorySpace.SBUF)
    nb.tensor_scalar_bitvec(cond_not, condition, one_u8, nki_ir.NisaBitvecOp.XOR)

    # casted_m1 = casted - 1 (integer subtraction)
    one_i32 = nb.constant(1.0, (p, 1), DType.I32, MemorySpace.SBUF)
    casted_m1 = nb.alloc(shape, DType.I32, MemorySpace.SBUF)
    nb.tensor_scalar_arith(casted_m1, casted, one_i32, nki_ir.NisaArithOp.SUBTRACT)

    # larger = condition * casted_m1 (mixed-dtype: u8 × i32 → out_dtype)
    larger = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
    nb.tensor_tensor_compare(larger, condition, casted_m1, nki_ir.NisaArithOp.MULTIPLY)

    # smaller = cond_not * casted (mixed-dtype: u8 × i32 → out_dtype)
    smaller = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
    nb.tensor_tensor_compare(smaller, cond_not, casted, nki_ir.NisaArithOp.MULTIPLY)

    # result = larger + smaller
    result = nb.alloc(shape, out_dtype, MemorySpace.SBUF)
    nb.tensor_tensor_arith(result, larger, smaller, nki_ir.NisaArithOp.ADD)
    return result
