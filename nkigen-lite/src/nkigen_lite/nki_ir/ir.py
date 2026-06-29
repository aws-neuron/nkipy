"""NKI-level IR for NeuronCore targets.

Bridges tensor_ir (logical, whole-tensor) and NISA (hardware instructions).
Makes tiling, layout, and memory placement explicit while remaining
verifiable and executable via numpy.

Key differences from tensor_ir:
  - Every value carries a MemorySpace (HBM, SBUF, PSUM).
  - Dim 0 of on-chip tiles is the partition dimension (max 128).
  - Explicit memory management: alloc/dealloc + dma_copy for data movement.
  - All compute ops take a pre-allocated dst as first parameter.
  - Matmul computes stationary[K, M].T @ moving[K, N] -> dst[M, N]:
    K is partition dim (contraction), M is stationary free (output partition),
    N is moving free (output free). Due to systolic array design.
  - DimSlice-based indexing mirrors Kernel Builder's nb.ts/nb.ds:
    ts(tile_i, size, total) and ds(offset, size) bundle offset + extent.
  - fori_loop for explicit tile iteration (static or dynamic bounds).
  - Verifier checks hardware tile constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Callable, Sequence

from nkigen_lite.core import (
    DType,
    Graph as _BaseGraph,
    Op,
    Value,
    ValueCounter,
    _DTYPE_BYTES,
)


# ===========================
# Types
# ===========================

class MemorySpace(str, Enum):
    HBM = "hbm"
    SBUF = "sbuf"
    PSUM = "psum"
    REG = "reg"


# -- Hardware constraints (gen2 defaults; gen3/gen4 have larger SBUF and free dims) --

PARTITION_MAX = 128
PSUM_FREE_MAX = 512                     # gen2/gen3; gen4: 4096 (fp32), 8192 (bf16)
MATMUL_STATIONARY_FREE_MAX = 128        # gemm_stationary_fmax, all gens
MATMUL_MOVING_FREE_MAX = 512            # gen2/gen3; gen4: 4096 (fp32), 8192 (bf16)
SBUF_PER_PARTITION_BYTES = 180_224      # gen2: 192KB - 16KB reserved
PSUM_PER_PARTITION_BYTES = 16 * 1024    # 16 KB, all gens
PSUM_BANKS = 8
PSUM_BANK_ELEMENTS = 512               # FP32 elements per bank


@dataclass(frozen=True)
class TileType:
    """Type of a tile value: shape, dtype, memory location.

    Convention for on-chip tiles (SBUF/PSUM):
      dim 0 = partition dimension (max 128)
      dim 1+ = free dimensions
    HBM tensors have no partition/free distinction.
    """
    shape: tuple[int, ...]
    dtype: DType
    memory: MemorySpace

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def partition_size(self) -> int:
        return self.shape[0] if self.rank > 0 else 1

    @property
    def free_shape(self) -> tuple[int, ...]:
        return self.shape[1:] if self.rank > 1 else ()

    @property
    def free_size(self) -> int:
        return prod(self.free_shape) if self.free_shape else 1

    @property
    def num_elements(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def size_bytes(self) -> int:
        return self.num_elements * _DTYPE_BYTES[self.dtype]

    def __str__(self) -> str:
        shape_str = "x".join(str(s) for s in self.shape)
        return f"tile<{shape_str}x{self.dtype.value}@{self.memory.value}>"


# ===========================
# Tile indexing (mirrors KB's nb.ts / nb.ds)
# ===========================

@dataclass(frozen=True)
class DimSlice:
    """One dimension's slice into an HBM tensor.

    Mirrors Kernel Builder's ``nb.ds(offset, size)``.  Bundles the byte
    offset and the tile extent so they stay in sync.

    *offset* may be an ``int`` (compile-time known, used after loop
    unrolling) or an IR ``Value`` (dynamic, from a loop index).
    *size* is always a static ``int`` — matching KB's restriction that
    slice extents are compile-time constants.
    *stride* defaults to 1 (contiguous). When > 1, accesses every
    stride-th element (maps to ``nb.coords`` with affine expression
    ``offset + idx * stride``).
    """
    offset: int | Value
    size: int
    stride: int = 1

    def __repr__(self) -> str:
        if self.stride == 1:
            return f"DimSlice(offset={self.offset}, size={self.size})"
        return f"DimSlice(offset={self.offset}, size={self.size}, stride={self.stride})"


# ===========================
# NISA enums (hardware engine grouping)
# ===========================

class NisaActivationOp(str, Enum):
    """Scalar engine activation functions (maps to nisa.activation_function)."""
    # Standard activations
    RELU = "relu"
    GELU = "gelu"
    GELU_APPRX_TANH = "gelu_apprx_tanh"
    GELU_APPRX_SIGMOID = "gelu_apprx_sigmoid"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SILU = "silu"
    SOFTPLUS = "softplus"
    MISH = "mish"
    # Math functions
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    RSQRT = "rsqrt"
    RECIPROCAL = "reciprocal"
    ABS = "abs"
    SQUARE = "square"
    SIGN = "sign"
    SIN = "sin"
    ARCTAN = "arctan"
    ERF = "erf"
    # Utility
    COPY = "copy"


class NisaArithOp(str, Enum):
    """Vector engine arithmetic ops (maps to nisa.arith_op)."""
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    MAXIMUM = "Maximum"
    MINIMUM = "Minimum"
    POW = "Pow"
    # Comparison ops (produce uint8 predicate output)
    IS_GT = "IsGT"
    IS_GE = "IsGE"
    IS_LT = "IsLT"
    IS_LE = "IsLE"
    IS_EQ = "IsEQ"
    IS_NE = "IsNE"
    # Logical ops (operate on uint8 predicates)
    LOGICAL_XOR = "LogicalXor"
    LOGICAL_AND = "LogicalAnd"
    LOGICAL_OR = "LogicalOr"


class NisaReduceOp(str, Enum):
    """Vector engine reduction ops (maps to nisa.tensor_reduce_arith)."""
    ADD = "Add"
    MAX = "Max"
    MIN = "Min"


class NisaBitvecOp(str, Enum):
    """Bitwise ops (maps to nisa.bitvec_op)."""
    AND = "BitwiseAnd"
    OR = "BitwiseOr"
    XOR = "BitwiseXor"
    NOT = "BitwiseNot"


class NisaRangeSelectCmp(str, Enum):
    """Comparison ops for range_select (maps to nisa.range_select_cmp)."""
    IS_EQ = "IsEq"
    IS_GT = "IsGt"
    IS_GE = "IsGe"
    IS_LE = "IsLe"
    IS_LT = "IsLt"


# ===========================
# Graph (tile-specific)
# ===========================

class Graph(_BaseGraph):
    """Ordered list of tile ops forming a tiled program."""

    _graph_label = "nki_graph"
    _SIDE_EFFECT_OPCODES = {"dma_copy", "dealloc", "fori_loop", "if_else"}

    def __init__(self, name: str = "main"):
        super().__init__(name)
        self.counter = ValueCounter(prefix="t")

    def verify(self) -> list[str]:
        """Check graph invariants plus hardware tile constraints."""
        errors = super().verify()
        for op in self.ops:
            for r in op.results:
                errors.extend(
                    _check_tile_constraints(r.type, f"{op.opcode} result {r!r}")
                )
        return errors


def _check_tile_constraints(tt: TileType, context: str) -> list[str]:
    """Validate hardware tile constraints for on-chip tiles."""
    errors: list[str] = []
    if tt.memory in (MemorySpace.HBM, MemorySpace.REG):
        return errors
    if tt.rank < 2:
        errors.append(
            f"{context}: on-chip tiles must be >= 2D "
            f"(partition + free), got rank {tt.rank}"
        )
        return errors
    if tt.partition_size > PARTITION_MAX:
        errors.append(
            f"{context}: partition size {tt.partition_size} "
            f"exceeds max {PARTITION_MAX}"
        )
    if tt.memory == MemorySpace.PSUM and tt.free_size > PSUM_FREE_MAX:
        errors.append(
            f"{context}: PSUM free size {tt.free_size} "
            f"exceeds max {PSUM_FREE_MAX}"
        )
    if tt.memory == MemorySpace.SBUF and tt.size_bytes > SBUF_PER_PARTITION_BYTES:
        errors.append(
            f"{context}: SBUF tile {tt.size_bytes} bytes "
            f"exceeds per-partition capacity {SBUF_PER_PARTITION_BYTES}"
        )
    if tt.memory == MemorySpace.PSUM and tt.size_bytes > PSUM_PER_PARTITION_BYTES:
        errors.append(
            f"{context}: PSUM tile {tt.size_bytes} bytes "
            f"exceeds capacity {PSUM_PER_PARTITION_BYTES}"
        )
    return errors


# ===========================
# Builder
# ===========================

class Builder:
    """Construct a NKI IR graph with explicit tiling, layout, and memory."""

    def __init__(self, name: str = "main"):
        self.graph = Graph(name)

    @classmethod
    def _from_graph(cls, graph: Graph) -> Builder:
        """Wrap an existing graph (used by unroll pass)."""
        b = cls.__new__(cls)
        b.graph = graph
        return b

    def _emit(
        self,
        opcode: str,
        inputs: Sequence[Value],
        result_types: Sequence[TileType],
        attrs: dict[str, Any] | None = None,
    ) -> Op:
        op = Op(opcode, inputs, result_types, attrs, counter=self.graph.counter)
        self.graph.append(op)
        return op

    # -- graph inputs (HBM tensors) --

    def add_input(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType = DType.F32,
    ) -> Value:
        """Declare an HBM tensor input."""
        v = Value(name=name, type=TileType(shape, dtype, MemorySpace.HBM))
        self.graph.add_input(v)
        return v

    # -- scalar index arithmetic --

    def scalar_const(self, value: int) -> Value:
        """Create a constant scalar index (register-like)."""
        rt = TileType((), DType.I32, MemorySpace.REG)
        return self._emit("scalar_const", [], [rt], {"value": value}).result

    def affine(self, index: int | Value, scale: int, base: int = 0) -> int | Value:
        """Compute base + index * scale.

        Polymorphic: returns int when index is int (unroll mode),
        returns Value when index is Value (graph construction mode).

        Prefer ``ts()`` / ``ds()`` for tile indexing — this is the
        low-level primitive they delegate to for dynamic offsets.
        """
        if isinstance(index, int):
            return base + index * scale
        rt = TileType((), DType.I32, MemorySpace.REG)
        return self._emit(
            "affine", [index], [rt], {"scale": scale, "base": base}
        ).result

    # -- tile indexing (mirrors KB's nb.ts / nb.ds) --

    def ts(self, tile_i: int | Value, size: int, total: int | None = None) -> DimSlice:
        """Tile-index slice — mirrors ``nb.ts(tile_i, size)``.

        Computes ``offset = tile_i * size``.  When *total* is provided
        and *tile_i* is a concrete ``int``, the extent is clamped to
        ``min(size, total - offset)`` so remainder tiles get the
        correct size.

        When *tile_i* is a dynamic ``Value`` (inside a loop body
        graph), the offset becomes an ``affine`` op and the extent
        is the full *size* (the body graph is a template for the
        common case).
        """
        if isinstance(tile_i, int):
            offset = tile_i * size
            extent = min(size, total - offset) if total is not None else size
            return DimSlice(offset, extent)
        offset = self.affine(tile_i, size, 0)
        return DimSlice(offset, size)

    @staticmethod
    def ds(offset: int | Value, size: int) -> DimSlice:
        """Dynamic slice — mirrors ``nb.ds(offset, size)``."""
        return DimSlice(offset, size)

    @staticmethod
    def full(size: int) -> DimSlice:
        """Full-dimension slice (offset 0, full extent)."""
        return DimSlice(0, size)

    # -- data movement --

    def dma_copy(
        self,
        dst: Value,
        src: Value,
        slices: tuple[DimSlice | int | Value, ...],
    ) -> Value | None:
        """DMA copy between HBM and on-chip memory.

        Direction inferred from memory spaces:
          Load  (HBM->on-chip): src is HBM, dst is pre-allocated on-chip tile.
                slices index into src. Returns a Value (SSA result with dst's type).
          Store (on-chip->HBM): src is on-chip, dst is HBM.
                slices index into dst. Returns None (side-effect).

        Each element of *slices* is a ``DimSlice`` (preferred) or a
        bare ``int``/``Value`` offset for backward compatibility (in
        which case the extent is inferred from the on-chip tile shape).
        """
        src_hbm = src.type.memory == MemorySpace.HBM
        dst_hbm = dst.type.memory == MemorySpace.HBM
        if src_hbm == dst_hbm:
            raise ValueError(
                f"dma_copy: exactly one of src/dst must be HBM, "
                f"got src={src.type.memory} dst={dst.type.memory}"
            )

        if src_hbm:
            hbm_tensor = src
            direction = "load"
        else:
            hbm_tensor = dst
            direction = "store"

        if len(slices) != hbm_tensor.type.rank:
            raise ValueError(
                f"dma_copy: slices rank {len(slices)} != "
                f"HBM tensor rank {hbm_tensor.type.rank}"
            )

        # Normalise: extract raw offsets, strides, and per-HBM-dim
        # extents from DimSlice (or bare int/Value, in which case
        # extent is inferred later from the on-chip tile).
        offsets: list[int | Value] = []
        strides: list[int] = []
        sizes: list[int | None] = []
        for s in slices:
            if isinstance(s, DimSlice):
                offsets.append(s.offset)
                strides.append(s.stride)
                sizes.append(s.size)
            else:
                offsets.append(s)
                strides.append(1)
                sizes.append(None)

        has_strides = any(s != 1 for s in strides)
        # Only persist `sizes` when at least one dim has an explicit
        # extent — for back-compat with older graphs that stored just
        # offsets and inferred extent from the on-chip tile shape.
        explicit_sizes = any(s is not None for s in sizes)

        if direction == "load":
            attrs: dict[str, Any] = {"direction": "load"}
            if has_strides:
                attrs["strides"] = tuple(strides)
            if explicit_sizes:
                # Fall back to inferring missing entries from the dst shape.
                inferred = [
                    sz if sz is not None else dst.type.shape[i]
                    for i, sz in enumerate(sizes)
                ]
                attrs["sizes"] = tuple(inferred)
            if any(isinstance(o, Value) for o in offsets):
                offset_vals = [
                    self.scalar_const(o) if isinstance(o, int) else o
                    for o in offsets
                ]
                attrs["dynamic_offsets"] = True
                return self._emit(
                    "dma_copy", [dst, src] + offset_vals, [dst.type], attrs,
                ).result
            attrs["offsets"] = tuple(offsets)
            return self._emit(
                "dma_copy", [dst, src], [dst.type], attrs,
            ).result
        else:
            attrs = {"direction": "store"}
            if has_strides:
                attrs["strides"] = tuple(strides)
            if explicit_sizes:
                inferred = [
                    sz if sz is not None else src.type.shape[i]
                    for i, sz in enumerate(sizes)
                ]
                attrs["sizes"] = tuple(inferred)
            if any(isinstance(o, Value) for o in offsets):
                offset_vals = [
                    self.scalar_const(o) if isinstance(o, int) else o
                    for o in offsets
                ]
                attrs["dynamic_offsets"] = True
                self._emit(
                    "dma_copy", [src, dst] + offset_vals, [], attrs,
                )
                return None
            attrs["offsets"] = tuple(offsets)
            self._emit(
                "dma_copy", [src, dst], [], attrs,
            )
            return None

    def collective(
        self,
        kind: str,
        dst: Value,
        src: Value,
        attrs: dict[str, Any],
    ) -> None:
        """Emit a collective communication op (HBM -> HBM, side-effect).

        ``kind`` is one of ``all_reduce``, ``all_gather``, ``reduce_scatter``,
        ``all_to_all``. ``attrs`` carries the per-collective parameters
        (replica_groups, reduce_op, dims) straight from the tensor_ir op.
        """
        if src.type.memory != MemorySpace.HBM or dst.type.memory != MemorySpace.HBM:
            raise ValueError(
                f"{kind}: collective operands must be HBM, got "
                f"src={src.type.memory} dst={dst.type.memory}"
            )
        self._emit(kind, [dst, src], [], dict(attrs))

    def tensor_copy(self, dst: Value, src: Value) -> Value:
        """Copy between on-chip memories (e.g. PSUM -> SBUF).

        dst is a pre-allocated Value. Both must be on-chip.
        Shapes must match. Returns the op's result with dst.type.
        Maps to nisa.tensor_copy. Vector engine.
        """
        if src.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_copy: src must be on-chip, use dma_copy for HBM")
        if dst.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_copy: dst must be on-chip, use dma_copy for HBM")
        if src.type.shape != dst.type.shape:
            raise ValueError(
                f"tensor_copy: shapes must match, "
                f"got src={src.type.shape} vs dst={dst.type.shape}"
            )
        return self._emit("tensor_copy", [dst, src], [dst.type]).result

    def access_pattern(
        self,
        src: Value,
        pattern: list[list[int]],
        offset: int | Value = 0,
        register_offsets: tuple[Value | None, ...] | None = None,
        vector_offset: Value | None = None,
    ) -> Value:
        """Create a strided view of an on-chip tile.

        Maps to KB's ``tile.ap(pattern, offset, register_offsets, vector_offset)``.

        *pattern* is a list of ``[stride, count]`` pairs, one per
        dimension.  The partition dim stride must equal the product
        of free dims (mandatory for SBUF/PSUM layout).

        The result is a view with shape derived from the counts:
        ``(count_0, count_1, ...)``.

        Args:
            src: On-chip tile to create a view of.
            pattern: [[stride, count], ...] per dimension.
            offset: Static or dynamic (Reg Value) base offset.
            register_offsets: Per-dimension dynamic offsets (Reg Values).
                              Tuple of (Value | None) matching pattern rank.
            vector_offset: Per-element indirect offset tile. When provided,
                           creates an indirect access pattern (gather-like).

        Example::

            # src is (128, 512). Access every 2nd free element:
            view = b.access_pattern(src, [[512, 128], [2, 256]])

            # With offset=1: picks elements 1, 3, 5, ...
            view = b.access_pattern(src, [[512, 128], [2, 256]], offset=1)

            # With dynamic offset from a loop index:
            view = b.access_pattern(src, [[512, 128], [1, 256]], offset=idx_reg)
        """
        if src.type.memory == MemorySpace.HBM:
            raise ValueError("access_pattern: src must be on-chip")
        out_shape = tuple(p[1] for p in pattern)
        rt = TileType(out_shape, src.type.dtype, src.type.memory)

        inputs = [src]
        attrs: dict[str, Any] = {"pattern": pattern}

        if isinstance(offset, Value):
            inputs.append(offset)
            attrs["dynamic_offset"] = True
        else:
            attrs["offset"] = offset

        if register_offsets is not None:
            reg_inputs = []
            for r in register_offsets:
                if r is not None:
                    reg_inputs.append(r)
            inputs.extend(reg_inputs)
            attrs["register_offsets"] = tuple(
                True if r is not None else False for r in register_offsets
            )

        if vector_offset is not None:
            inputs.append(vector_offset)
            attrs["vector_offset"] = True

        return self._emit("access_pattern", inputs, [rt], attrs).result

    def copy_predicated(self, dst: Value, pred: Value, src: Value) -> Value:
        """Conditional tensor copy: dst[i] = src[i] where pred[i] > 0.

        Maps to ``nisa.copy_predicated``. All operands must be on-chip
        with matching shapes (pred may be uint8).
        """
        if any(v.type.memory == MemorySpace.HBM for v in (dst, pred, src)):
            raise ValueError("copy_predicated: operands must be on-chip")
        if src.type.shape != dst.type.shape:
            raise ValueError(
                f"copy_predicated: src shape {src.type.shape} != dst shape {dst.type.shape}"
            )
        return self._emit("copy_predicated", [dst, pred, src], [dst.type]).result

    def gather(self, dst: Value, src: Value, indices: Value) -> Value:
        """Per-partition index-based gather: dst[p,i] = src[p, indices[p,i]].

        Maps to ``nisa.gather``.  All operands must be in SBUF.
        ``dst`` and ``indices`` must have the same shape.
        """
        if any(v.type.memory != MemorySpace.SBUF for v in (dst, src, indices)):
            raise ValueError("gather: all operands must be in SBUF")
        if dst.type.shape != indices.type.shape:
            raise ValueError(
                f"gather: dst shape {dst.type.shape} != indices shape {indices.type.shape}"
            )
        return self._emit("gather", [dst, src, indices], [dst.type]).result

    # -- on-chip allocation --

    def alloc(
        self,
        shape: tuple[int, ...],
        dtype: DType,
        memory: MemorySpace,
        num_buffers: int = 1,
    ) -> Value:
        """Allocate an uninitialized tile.

        *num_buffers* > 1 enables multi-buffering for pipelined
        double/triple buffering.  Use ``rotate()`` to advance to the
        next buffer slot.  Maps to
        ``nb.compiler.alloc(..., num_buffers=N)``.

        ``MemorySpace.HBM`` allocates a device-memory scratch buffer
        (not a graph input).  Maps to
        ``nb.compiler.alloc(..., space=nb.hbm)``.
        """
        rt = TileType(shape, dtype, memory)
        attrs: dict[str, Any] = {}
        if num_buffers > 1:
            attrs["num_buffers"] = num_buffers
        return self._emit("alloc", [], [rt], attrs or None).result

    def rotate(self, tile: Value) -> Value:
        """Advance to the next slot of a multi-buffered allocation.

        Maps to ``nb.compiler.rotate(tile)``.  Returns a Value
        referencing the new buffer slot (same type as *tile*).
        The interpreter ignores buffering and returns the same array.
        """
        return self._emit("rotate", [tile], [tile.type]).result

    def dealloc(self, tile: Value) -> None:
        """Deallocate a previously allocated tile."""
        self._emit("dealloc", [tile], [])

    def constant(
        self,
        value: float,
        shape: tuple[int, ...],
        dtype: DType,
        memory: MemorySpace = MemorySpace.SBUF,
    ) -> Value:
        """Create a constant tile (convenience: alloc + memset)."""
        tile = self.alloc(shape, dtype, memory)
        return self.memset(tile, value)

    # ===========================
    # Tensor Engine: matmul
    # ===========================

    def matmul(
        self,
        dst: Value,
        stationary: Value,
        moving: Value,
        accumulate: bool = False,
        is_transpose: bool = False,
    ) -> Value:
        """Tile-level matmul on Tensor Engine (NeuronCore systolic array).

        Always computes stationary.T @ moving:
          stationary: tile<K, M @ sbuf>  K=partition (contraction), M=free (max 128)
          moving:     tile<K, N @ sbuf>  K=partition (contraction), N=free (max 512)
          dst:        tile<M, N @ psum>  M=output partition, N=output free (FP32)

        *is_transpose* is a hardware precision hint — when ``True`` the
        tensor engine uses a numerically more accurate path for the
        implicit transpose of the stationary operand.  It does NOT
        change the mathematical semantics (always ``stat.T @ mov``).

        When accumulate=True, the matmul accumulates into dst.
        """
        if stationary.type.memory != MemorySpace.SBUF:
            raise ValueError(
                f"matmul: stationary must be in SBUF, "
                f"got {stationary.type.memory}"
            )
        if moving.type.memory != MemorySpace.SBUF:
            raise ValueError(
                f"matmul: moving must be in SBUF, got {moving.type.memory}"
            )
        if stationary.type.rank != 2 or moving.type.rank != 2:
            raise ValueError(
                "matmul: operands must be 2D [contraction, partition/free]"
            )
        # Always stat[K, M].T @ mov[K, N] → dst[M, N]
        c_stat, p = stationary.type.shape
        c_mov, f = moving.type.shape
        if c_stat != c_mov:
            raise ValueError(
                f"matmul: contraction dim mismatch: {c_stat} vs {c_mov}"
            )
        if p > PARTITION_MAX:
            raise ValueError(
                f"matmul: partition dim {p} exceeds max {PARTITION_MAX}"
            )
        if p > MATMUL_STATIONARY_FREE_MAX:
            raise ValueError(
                f"matmul: stationary free (P={p}) exceeds "
                f"max {MATMUL_STATIONARY_FREE_MAX}"
            )

        out_shape = (p, f)
        if dst.type.memory != MemorySpace.PSUM:
            raise ValueError(
                f"matmul: dst must be in PSUM, got {dst.type.memory}"
            )
        if dst.type.shape != out_shape:
            raise ValueError(
                f"matmul: dst shape {dst.type.shape} != "
                f"expected output {out_shape}"
            )
        if dst.type.dtype != DType.F32:
            raise ValueError(
                f"matmul: dst dtype must be F32, got {dst.type.dtype}"
            )

        attrs: dict[str, Any] = {}
        if accumulate:
            attrs["accumulate"] = True
        if is_transpose:
            attrs["is_transpose"] = True

        return self._emit(
            "matmul", [dst, stationary, moving], [dst.type], attrs,
        ).result

    # -- cross-partition reduction (GpSimd Engine) --

    def cross_lane_reduce_arith(
        self,
        dst: Value,
        x: Value,
        op: NisaReduceOp = NisaReduceOp.ADD,
    ) -> Value:
        """Reduce along partition dim (axis 0) via GpSimd Engine.

        Maps to ``nisa.cross_lane_reduce_arith``.  Reduces across
        partitions (unlike ``tensor_reduce_arith`` which reduces free
        axes via the Vector Engine).

        dst must be pre-allocated with partition dim = 1.

        MIN is decomposed to negate→MAX→negate since hardware only
        supports Add and Max for cross-lane reduction.
        """
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("cross_lane_reduce_arith: operand must be on-chip")
        if x.type.rank < 2:
            raise ValueError(
                "cross_lane_reduce_arith: need at least 2D (partition + free)"
            )
        expected_shape = (1,) + x.type.shape[1:]
        if dst.type.shape != expected_shape:
            raise ValueError(
                f"cross_lane_reduce_arith: dst shape {dst.type.shape} != "
                f"expected {expected_shape}"
            )
        if op == NisaReduceOp.MIN:
            # min(x) = -max(-x)
            p_size = x.type.shape[0]
            neg_const = self.constant(-1.0, (p_size, 1), x.type.dtype, MemorySpace.SBUF)
            neg_x = self.alloc(x.type.shape, x.type.dtype, MemorySpace.SBUF)
            neg_x = self.tensor_scalar_arith(
                neg_x, x, neg_const, NisaArithOp.MULTIPLY
            )
            max_neg = self._emit(
                "cross_lane_reduce_arith", [dst, neg_x], [dst.type],
                {"op": NisaReduceOp.MAX},
            ).result
            neg_const_out = self.constant(-1.0, (1, 1), dst.type.dtype, MemorySpace.SBUF)
            neg_result = self.alloc(dst.type.shape, dst.type.dtype, MemorySpace.SBUF)
            return self.tensor_scalar_arith(
                neg_result, max_neg, neg_const_out, NisaArithOp.MULTIPLY
            )
        return self._emit(
            "cross_lane_reduce_arith", [dst, x], [dst.type], {"op": op},
        ).result

    # -- GpSimd utilities --

    def iota(
        self,
        dst: Value,
        pattern: list[list[int]] | None = None,
        offset: int = 0,
        channel_multiplier: int = 0,
    ) -> Value:
        """Generate index pattern tile.

        Without arguments: element [p, f] = f (free-dim index).
        With pattern/offset/channel_multiplier: matches KB's ``nisa.iota``.

        ``pattern`` is a list of ``[step, count]`` pairs whose counts
        multiply to the free dimension size.  ``channel_multiplier``
        scales the partition index contribution.  Final value:
        ``offset + p * channel_multiplier + sum(digit_i * step_i)``.
        """
        if dst.type.memory == MemorySpace.HBM:
            raise ValueError("iota: dst must be on-chip")
        if pattern is None:
            pattern = [[1, dst.type.shape[-1]]]
        attrs: dict[str, Any] = {
            "pattern": pattern,
            "offset": offset,
            "channel_multiplier": channel_multiplier,
        }
        return self._emit("iota", [dst], [dst.type], attrs).result

    def max8(self, dst: Value, src: Value) -> Value:
        """8 largest values per partition, descending. Maps to ``nisa.max8``.

        ``src`` is [par_dim, F] (8 <= F <= 16384); ``dst`` is [par_dim, 8].
        """
        if src.type.memory == MemorySpace.HBM or dst.type.memory == MemorySpace.HBM:
            raise ValueError("max8: operands must be on-chip")
        return self._emit("max8", [dst, src], [dst.type]).result

    def find_index8(self, dst: Value, src: Value, vals: Value) -> Value:
        """Indices of each of ``vals`` (first match) within ``src`` per partition.

        Maps to ``nisa.find_index8``.  ``src`` is [par_dim, F], ``vals`` and
        ``dst`` are [par_dim, 8]; ``dst`` is integer.

        NOTE: ``find_index8`` is gen2-only; on gen3+ targets it fails the
        compiler's ISA check.  Use ``match_replace8`` (with ``dst_idx``)
        instead for index recovery on current hardware.
        """
        for v in (dst, src, vals):
            if v.type.memory == MemorySpace.HBM:
                raise ValueError("find_index8: operands must be on-chip")
        return self._emit("find_index8", [dst, src, vals], [dst.type]).result

    def match_replace8(
        self, dst: Value, dst_idx: Value, data: Value, vals: Value, imm: float,
    ) -> tuple[Value, Value]:
        """For each of the 8 ``vals``, find its first match in ``data``, record
        the index in ``dst_idx``, and replace that position with ``imm``.

        Maps to ``nisa.nc_match_replace8`` (gen3+).  Returns ``(masked_data,
        indices)``.  ``data``/``dst`` are [par_dim, F]; ``vals``/``dst_idx``
        are [par_dim, 8] (``dst_idx`` integer).  This is the workhorse of the
        scanning top-k loop: it yields indices *and* masks taken values so the
        next ``max8`` finds the following 8.
        """
        for v in (dst, dst_idx, data, vals):
            if v.type.memory == MemorySpace.HBM:
                raise ValueError("match_replace8: operands must be on-chip")
        op = self._emit(
            "match_replace8", [dst, dst_idx, data, vals],
            [dst.type, dst_idx.type], {"imm": imm},
        )
        return op.results[0], op.results[1]

    def stream_shuffle(
        self,
        dst: Value,
        x: Value,
        shuffle_mask: list[int],
    ) -> Value:
        """Cross-partition data shuffle within SBUF.

        Rearranges data across partitions according to shuffle_mask.
        shuffle_mask[i] specifies which source partition supplies
        destination partition i. Used to broadcast after cross_lane_reduce.
        Maps to nisa.stream_shuffle.
        """
        if x.type.memory != MemorySpace.SBUF:
            raise ValueError("stream_shuffle: operand must be in SBUF")
        return self._emit("stream_shuffle", [dst, x], [dst.type], {
            "shuffle_mask": list(shuffle_mask),
        }).result

    # -- additional ISA ops --

    def affine_select(
        self,
        dst: Value,
        pred: Value,
        on_true: Value,
        on_false: Value,
    ) -> Value:
        """Conditional select per element.  Maps to ``nisa.affine_select``.

        ``dst[i] = on_true[i] if pred[i] else on_false[i]``.
        *pred* should contain boolean / mask values.
        """
        if any(v.type.memory == MemorySpace.HBM for v in (pred, on_true, on_false)):
            raise ValueError("affine_select: operands must be on-chip")
        return self._emit(
            "affine_select", [dst, pred, on_true, on_false], [dst.type],
        ).result

    def dma_copy_indirect(
        self,
        dst: Value,
        src: Value,
        index: Value,
        row_width: int | None = None,
        free_offset: int = 0,
    ) -> Value | None:
        """Indirect DMA copy with vector offset.  Maps to ``nisa.dma_copy_indirect``.

        *index* is an SBUF tile of integer offsets used to gather/scatter.
        Direction inferred from memory spaces like ``dma_copy``.

        By default each gathered/scattered row spans the full HBM row (the
        on-chip tile's free size). To process a *column window* of a wide row —
        so a huge row can be tiled across several DMAs and never materialised in
        one oversized SBUF tile — pass ``row_width`` (the HBM tensor's true row
        stride) and ``free_offset`` (the starting column). The on-chip tile then
        holds only ``free_offset:free_offset+free`` of each row.
        """
        src_hbm = src.type.memory == MemorySpace.HBM
        dst_hbm = dst.type.memory == MemorySpace.HBM
        if src_hbm == dst_hbm:
            raise ValueError("dma_copy_indirect: exactly one of src/dst must be HBM")
        attrs: dict[str, Any] = {}
        if row_width is not None:
            attrs["row_width"] = int(row_width)
        if free_offset:
            attrs["free_offset"] = int(free_offset)
        if src_hbm:
            attrs["direction"] = "load"
            return self._emit(
                "dma_copy_indirect", [dst, src, index], [dst.type], attrs,
            ).result
        else:
            attrs["direction"] = "store"
            self._emit(
                "dma_copy_indirect", [src, dst, index], [], attrs,
            )
            return None

    def tensor_tensor_scan(
        self,
        dst: Value,
        data0: Value,
        data1: Value,
        initial: Value,
        op0: NisaArithOp,
        op1: NisaArithOp,
    ) -> Value:
        """Two-input scan along the free dimension.  Maps to ``nisa.tensor_tensor_scan``.

        Computes per partition::

            result[:, 0] = op1(op0(data0[:, 0], initial), data1[:, 0])
            result[:, i] = op1(op0(data0[:, i], result[:, i-1]), data1[:, i])

        ``data0`` and ``data1`` must have the same shape.
        ``initial`` has free_size=1 (one element per partition) or is scalar.
        """
        if any(v.type.memory == MemorySpace.HBM for v in (data0, data1)):
            raise ValueError("tensor_tensor_scan: operands must be on-chip")
        return self._emit(
            "tensor_tensor_scan", [dst, data0, data1, initial], [dst.type],
            {"op0": op0, "op1": op1},
        ).result

    def sequence_bounds(
        self,
        dst: Value,
        segment_ids: Value,
    ) -> Value:
        """Compute segment boundaries from segment IDs.

        Maps to ``nisa.sequence_bounds``.  Given segment IDs of shape
        ``(1, F)``, outputs ``(1, 2, F)`` where ``dst[0, 0, f]`` is the
        start index and ``dst[0, 1, f]`` is the end index of the segment
        that element ``f`` belongs to.  Partition dim must be 1.
        Elements with segment ID 0 are treated as padding.
        """
        if segment_ids.type.memory == MemorySpace.HBM:
            raise ValueError("sequence_bounds: segment_ids must be on-chip")
        return self._emit(
            "sequence_bounds", [dst, segment_ids], [dst.type],
        ).result

    def dma_gather_transpose(
        self,
        dst: Value,
        src: Value,
        gather_index: Value,
    ) -> Value:
        """Fused gather + transpose via DMA.  Maps to ``nisa.dma_gather_transpose``."""
        return self._emit(
            "dma_gather_transpose", [dst, src, gather_index], [dst.type],
        ).result

    def exponential(
        self,
        dst: Value,
        src: Value,
        max_value: Value | None = None,
    ) -> Value:
        """Numerically-stable exp: dst = exp(src - max_value).

        Maps to ``nisa.exponential``.  If *max_value* is ``None``,
        computes ``exp(src)`` directly.  *max_value* should have
        free_size=1 for per-partition broadcast.
        """
        if src.type.memory == MemorySpace.HBM:
            raise ValueError("exponential: src must be on-chip")
        if dst.type.shape != src.type.shape:
            raise ValueError(
                f"exponential: dst shape {dst.type.shape} != src shape {src.type.shape}"
            )
        inputs = [dst, src]
        if max_value is not None:
            inputs.append(max_value)
        return self._emit("exponential", inputs, [dst.type]).result

    def range_select(
        self,
        dst: Value,
        src: Value,
        bound0: Value,
        bound1: Value,
        fill_value: float,
        comp_op0: NisaRangeSelectCmp,
        comp_op1: NisaRangeSelectCmp,
    ) -> Value:
        """Conditional range selection on free-dim index.

        For each element at free-dim position ``j``:
        ``dst[p,j] = src[p,j]`` if ``j comp_op0 bound0[p]`` AND
        ``j comp_op1 bound1[p]``, else ``fill_value``.

        Maps to ``nisa.range_select``.
        """
        if any(v.type.memory == MemorySpace.HBM for v in (dst, src, bound0, bound1)):
            raise ValueError("range_select: operands must be on-chip")
        return self._emit("range_select", [dst, src, bound0, bound1], [dst.type], {
            "fill_value": fill_value,
            "comp_op0": comp_op0,
            "comp_op1": comp_op1,
        }).result

    def select_reduce(
        self,
        dst: Value,
        pred: Value,
        on_true: Value,
        on_false: float | Value,
        reduce_dst: Value | None = None,
        reduce_op: NisaReduceOp | None = None,
    ) -> Value:
        """Fused predicated select + optional reduction.

        ``dst = where(pred > 0, on_true, on_false)``.
        When *reduce_dst* and *reduce_op* are given, also writes the
        reduction of the selected result into *reduce_dst*.

        Maps to ``nisa.select_reduce``.
        """
        if any(v.type.memory == MemorySpace.HBM for v in (dst, pred, on_true)):
            raise ValueError("select_reduce: operands must be on-chip")
        inputs = [dst, pred, on_true]
        attrs: dict[str, Any] = {}
        if isinstance(on_false, (int, float)):
            attrs["on_false_scalar"] = float(on_false)
        else:
            inputs.append(on_false)
        if reduce_dst is not None:
            if reduce_op is None:
                raise ValueError("select_reduce: reduce_op required when reduce_dst given")
            inputs.append(reduce_dst)
            attrs["reduce_op"] = reduce_op
        return self._emit("select_reduce", inputs, [dst.type], attrs).result

    # ===========================
    # Transpose (DMA engine / Vector engine)
    # ===========================

    def dma_transpose(self, dst: Value, src: Value, perm: tuple[int, ...]) -> Value:
        """Transpose via DMA engine.  Maps to ``nisa.dma_transpose``.

        dst must be pre-allocated with the transposed shape.
        Supports any on-chip tile size (unlike stream_transpose).
        Not supported on TRN3.
        """
        if src.type.memory == MemorySpace.HBM:
            raise ValueError("dma_transpose: src must be on-chip")
        if sorted(perm) != list(range(src.type.rank)):
            raise ValueError(f"dma_transpose: invalid perm {perm}")
        expected_shape = tuple(src.type.shape[p] for p in perm)
        if dst.type.shape != expected_shape:
            raise ValueError(
                f"dma_transpose: dst shape {dst.type.shape} != "
                f"expected {expected_shape}"
            )
        return self._emit(
            "dma_transpose", [dst, src], [dst.type], {"perm": perm}
        ).result

    def stream_transpose(self, dst: Value, src: Value) -> Value:
        """Small partition-free transpose via Vector engine.

        Maps to ``nisa.stream_transpose``.  Partition dim <= 32,
        free dim <= 32.  Always swaps axes (0, 1).
        """
        if src.type.memory == MemorySpace.HBM:
            raise ValueError("stream_transpose: src must be on-chip")
        if src.type.rank != 2:
            raise ValueError("stream_transpose: src must be 2D")
        if src.type.shape[0] > 32 or src.type.shape[1] > 32:
            raise ValueError(
                f"stream_transpose: max 32x32, got {src.type.shape}"
            )
        expected_shape = (src.type.shape[1], src.type.shape[0])
        if dst.type.shape != expected_shape:
            raise ValueError(
                f"stream_transpose: dst shape {dst.type.shape} != "
                f"expected {expected_shape}"
            )
        return self._emit(
            "stream_transpose", [dst, src], [dst.type]
        ).result

    def transpose(self, x: Value, perm: tuple[int, ...]) -> Value:
        """Convenience: auto-selects dma_transpose with implicit dst alloc.

        Kept for backward compatibility with tiling pass and examples.
        """
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("transpose: operand must be on-chip")
        if sorted(perm) != list(range(x.type.rank)):
            raise ValueError(f"transpose: invalid perm {perm}")
        new_shape = tuple(x.type.shape[p] for p in perm)
        dst = self.alloc(new_shape, x.type.dtype, x.type.memory)
        return self.dma_transpose(dst, x, perm)

    # ===========================
    # Shape manipulation
    # ===========================

    def broadcast(self, x: Value, shape: tuple[int, ...]) -> Value:
        """Broadcast a tile to a larger shape within the same memory."""
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("broadcast: operand must be on-chip")
        offset = len(shape) - x.type.rank
        if offset < 0:
            raise ValueError("broadcast: target rank must be >= source rank")
        for i, src_dim in enumerate(x.type.shape):
            tgt_dim = shape[offset + i]
            if src_dim != 1 and src_dim != tgt_dim:
                raise ValueError(
                    f"broadcast: dim {i} (size {src_dim}) not "
                    f"broadcastable to {tgt_dim}"
                )
        rt = TileType(shape, x.type.dtype, x.type.memory)
        return self._emit("broadcast", [x], [rt], {"shape": shape}).result

    def reshape(self, x: Value, new_shape: tuple[int, ...]) -> Value:
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("reshape: operand must be on-chip")
        if prod(x.type.shape) != prod(new_shape):
            raise ValueError(
                f"reshape: size mismatch {x.type.shape} -> {new_shape}"
            )
        rt = TileType(new_shape, x.type.dtype, x.type.memory)
        return self._emit("reshape", [x], [rt], {"shape": new_shape}).result

    def view(self, x: Value, new_shape: tuple[int, ...], dtype: DType | None = None) -> Value:
        """Reinterpret memory with new shape and optionally new dtype.

        Maps to KB's ``TileView.view(new_shape, dtype)``.
        Total byte size must match.  Zero-copy in KB (view transform).
        """
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("view: operand must be on-chip")
        out_dtype = dtype if dtype is not None else x.type.dtype
        old_bytes = prod(x.type.shape) * _DTYPE_BYTES[x.type.dtype]
        new_bytes = prod(new_shape) * _DTYPE_BYTES[out_dtype]
        if old_bytes != new_bytes:
            raise ValueError(
                f"view: byte size mismatch {old_bytes} -> {new_bytes}"
            )
        rt = TileType(new_shape, out_dtype, x.type.memory)
        return self._emit("view", [x], [rt], {
            "shape": new_shape, "dtype": out_dtype,
        }).result

    # -- type cast --

    def cast(self, x: Value, dtype: DType) -> Value:
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("cast: operand must be on-chip")
        rt = TileType(x.type.shape, dtype, x.type.memory)
        return self._emit("cast", [x], [rt], {"dtype": dtype}).result

    # ===========================
    # NISA-grouped ops (hardware engine mapping)
    # ===========================

    def activation(
        self,
        dst: Value,
        src: Value,
        op: NisaActivationOp,
        bias: Value | None = None,
        scale: float = 1.0,
        reduce_dst: Value | None = None,
        reduce_op: NisaReduceOp | None = None,
    ) -> Value:
        """Scalar engine activation: dst = act(src * scale + bias).

        When *reduce_dst* and *reduce_op* are given, also writes the
        reduction of the activated result into *reduce_dst* (fused
        activation+reduce, matching KB's ``nisa.activation(reduce_res=...,
        reduce_op=...)``).
        """
        if src.type.memory == MemorySpace.HBM:
            raise ValueError("activation: operand must be on-chip")
        inputs = [dst, src] if bias is None else [dst, src, bias]
        if bias is not None:
            if bias.type.memory == MemorySpace.HBM:
                raise ValueError("activation: bias must be on-chip")
            if bias.type.dtype != src.type.dtype:
                raise ValueError(
                    f"activation: dtype mismatch src={src.type.dtype} vs bias={bias.type.dtype}"
                )
            if (bias.type.partition_size != src.type.partition_size
                    and bias.type.partition_size != 1):
                raise ValueError(
                    f"activation: partition dim mismatch "
                    f"src={src.type.partition_size} vs bias={bias.type.partition_size}"
                )
            if bias.type.free_size != 1:
                raise ValueError(
                    f"activation: bias must have free_size=1, "
                    f"got shape {bias.type.shape} (free_size={bias.type.free_size})"
                )
        if dst.type.shape != src.type.shape:
            raise ValueError(
                f"activation: dst shape {dst.type.shape} != "
                f"src shape {src.type.shape}"
            )
        attrs: dict[str, Any] = {"op": op, "scale": scale}
        if reduce_dst is not None:
            if reduce_op is None:
                raise ValueError("activation: reduce_op required when reduce_dst is given")
            inputs.append(reduce_dst)
            attrs["reduce_op"] = reduce_op
        emit_op = self._emit("activation", inputs, [dst.type], attrs)
        return emit_op.result

    def tensor_tensor_arith(self, dst: Value, a: Value, b: Value, op: NisaArithOp) -> Value:
        """Vector engine tensor-tensor arithmetic: dst = a op b.

        Requires exact shape match (use tensor_scalar_arith for broadcasting).
        dst must be pre-allocated with the correct shape.
        """
        if a.type.memory == MemorySpace.HBM or b.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_tensor_arith: operands must be on-chip")
        if a.type.dtype != b.type.dtype:
            raise ValueError(
                f"tensor_tensor_arith: dtype mismatch {a.type.dtype} vs {b.type.dtype}"
            )
        if a.type.shape != b.type.shape:
            raise ValueError(
                f"tensor_tensor_arith: shapes must match exactly, "
                f"got {a.type.shape} vs {b.type.shape} "
                f"(use tensor_scalar_arith for broadcasting)"
            )
        if a.type.memory == MemorySpace.PSUM and b.type.memory == MemorySpace.PSUM:
            raise ValueError(
                "tensor_tensor_arith: both operands in PSUM not supported "
                "(move one to SBUF first)"
            )
        # Validate dst shape
        if dst.type.shape != a.type.shape:
            raise ValueError(
                f"tensor_tensor_arith: dst shape {dst.type.shape} != "
                f"operand shape {a.type.shape}"
            )
        return self._emit("tensor_tensor_arith", [dst, a, b], [dst.type], {"op": op}).result

    def tensor_tensor_bitvec(self, dst: Value, a: Value, b: Value, op: NisaBitvecOp) -> Value:
        """Vector engine tensor-tensor bitwise operation: dst = a op b.

        Requires exact shape match. dst must be pre-allocated.
        """
        if a.type.memory == MemorySpace.HBM or b.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_tensor_bitvec: operands must be on-chip")
        if a.type.shape != b.type.shape:
            raise ValueError(
                f"tensor_tensor_bitvec: shapes must match exactly, "
                f"got {a.type.shape} vs {b.type.shape}"
            )
        if dst.type.shape != a.type.shape:
            raise ValueError(
                f"tensor_tensor_bitvec: dst shape {dst.type.shape} != "
                f"operand shape {a.type.shape}"
            )
        return self._emit("tensor_tensor_bitvec", [dst, a, b], [dst.type], {"op": op}).result

    def tensor_tensor_compare(self, dst: Value, a: Value, b: Value, op: NisaArithOp) -> Value:
        """Vector engine tensor-tensor comparison: dst = a op b (predicate output).

        Unlike tensor_tensor_arith, allows dtype mismatch: inputs can be float
        while dst is uint8 (predicate). Uses the same nisa.tensor_tensor_arith
        instruction with comparison ops (IsGT, IsGE, etc.).
        """
        if a.type.memory == MemorySpace.HBM or b.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_tensor_compare: operands must be on-chip")
        if a.type.shape != b.type.shape:
            raise ValueError(
                f"tensor_tensor_compare: shapes must match, "
                f"got {a.type.shape} vs {b.type.shape}"
            )
        if dst.type.shape != a.type.shape:
            raise ValueError(
                f"tensor_tensor_compare: dst shape {dst.type.shape} != "
                f"operand shape {a.type.shape}"
            )
        return self._emit("tensor_tensor_arith", [dst, a, b], [dst.type], {"op": op}).result

    def tensor_scalar_bitvec(
        self, dst: Value, x: Value, operand0: Value, op0: NisaBitvecOp,
    ) -> Value:
        """Vector engine tensor-scalar bitwise operation: dst = x op0 operand0.

        Scalar operand must have free_size=1 (broadcast along free dims).
        Maps to nisa.tensor_scalar_bitvec.
        """
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_scalar_bitvec: operands must be on-chip")
        if operand0.type.free_size != 1:
            raise ValueError(
                f"tensor_scalar_bitvec: operand0 must have free_size=1, "
                f"got shape {operand0.type.shape}"
            )
        return self._emit("tensor_scalar_bitvec", [dst, x, operand0], [dst.type], {"op0": op0}).result

    def tensor_scalar_arith(
        self,
        dst: Value,
        x: Value,
        operand0: Value,
        op0: NisaArithOp,
        operand1: Value | None = None,
        op1: NisaArithOp | None = None,
    ) -> Value:
        """Vector engine tensor-scalar arithmetic.

        Single-stage: dst = x op0 operand0
        Two-stage:    dst = (x op0 operand0) op1 operand1

        Scalar operands must have free_size=1 (broadcast along free dims).
        dst must be pre-allocated. Maps to nisa.tensor_scalar_arith.
        """
        if x.type.memory == MemorySpace.HBM or operand0.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_scalar_arith: operands must be on-chip")
        if x.type.dtype != operand0.type.dtype:
            raise ValueError(
                f"tensor_scalar_arith: dtype mismatch {x.type.dtype} vs {operand0.type.dtype}"
            )
        if operand0.type.free_size != 1:
            raise ValueError(
                f"tensor_scalar_arith: operand0 must have free_size=1, "
                f"got shape {operand0.type.shape} (free_size={operand0.type.free_size})"
            )
        if (operand0.type.partition_size != x.type.partition_size
                and operand0.type.partition_size != 1):
            raise ValueError(
                f"tensor_scalar_arith: partition dim mismatch "
                f"x={x.type.partition_size} vs operand0={operand0.type.partition_size}"
            )
        inputs = [dst, x, operand0]
        attrs: dict[str, Any] = {"op0": op0}
        if operand1 is not None:
            if op1 is None:
                raise ValueError("tensor_scalar_arith: op1 required when operand1 is provided")
            if operand1.type.free_size != 1:
                raise ValueError(
                    f"tensor_scalar_arith: operand1 must have free_size=1, "
                    f"got shape {operand1.type.shape}"
                )
            inputs.append(operand1)
            attrs["op1"] = op1
        return self._emit("tensor_scalar_arith", inputs, [dst.type], attrs).result

    def scalar_tensor_tensor_arith(
        self,
        dst: Value,
        src0: Value,
        src1: Value,
        imm0: Value,
        op0: NisaArithOp,
        op1: NisaArithOp,
    ) -> Value:
        """Vector engine three-operand fused: dst = (src0 op0 imm0) op1 src1.

        imm0 must have free_size=1 (scalar broadcast).
        src0 and src1 must have the same shape.
        dst must be pre-allocated. Maps to nisa.scalar_tensor_tensor_arith.
        """
        if any(v.type.memory == MemorySpace.HBM for v in (src0, src1, imm0)):
            raise ValueError("scalar_tensor_tensor_arith: operands must be on-chip")
        if src0.type.shape != src1.type.shape:
            raise ValueError(
                f"scalar_tensor_tensor_arith: src0 and src1 shapes must match, "
                f"got {src0.type.shape} vs {src1.type.shape}"
            )
        if imm0.type.free_size != 1:
            raise ValueError(
                f"scalar_tensor_tensor_arith: imm0 must have free_size=1, "
                f"got shape {imm0.type.shape}"
            )
        return self._emit("scalar_tensor_tensor_arith", [dst, src0, src1, imm0], [dst.type], {
            "op0": op0, "op1": op1,
        }).result

    def tensor_reduce_arith(
        self,
        dst: Value,
        x: Value,
        op: NisaReduceOp,
        num_r_dim: int,
        keepdims: bool = True,
    ) -> Value:
        """Vector engine reduction: reduce the rightmost num_r_dim free dims.

        Default keepdims=True matches NISA's output shape convention (P,1).
        num_r_dim must be >= 1 and <= rank-1 (cannot reduce partition dim).
        dst must be pre-allocated with the expected reduced shape.
        """
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("tensor_reduce_arith: operand must be on-chip")
        if x.type.rank == 0:
            raise ValueError("tensor_reduce_arith: cannot reduce a scalar (rank 0)")
        if num_r_dim < 1:
            raise ValueError(
                f"tensor_reduce_arith: num_r_dim must be >= 1, got {num_r_dim}"
            )
        if num_r_dim >= x.type.rank:
            raise ValueError(
                f"tensor_reduce_arith: num_r_dim={num_r_dim} must be < rank={x.type.rank} "
                f"(cannot reduce partition dim; use cross_lane_reduce_arith)"
            )
        if not keepdims and x.type.rank - num_r_dim < 2:
            raise ValueError(
                "tensor_reduce_arith: reducing all free dims with keepdims=False "
                "would leave rank < 2, violating on-chip 2D tile convention"
            )
        # Compute expected reduced shape
        if keepdims:
            expected_shape = x.type.shape[:x.type.rank - num_r_dim] + (1,) * num_r_dim
        else:
            expected_shape = x.type.shape[:x.type.rank - num_r_dim]
        if dst.type.shape != expected_shape:
            raise ValueError(
                f"tensor_reduce_arith: dst shape {dst.type.shape} != "
                f"expected {expected_shape}"
            )
        return self._emit("tensor_reduce_arith", [dst, x], [dst.type], {
            "op": op, "num_r_dim": num_r_dim, "keepdims": keepdims,
        }).result

    def activation_reduce(
        self,
        dst: Value,
        x: Value,
        act_op: NisaActivationOp,
        reduce_op: NisaReduceOp,
        num_r_dim: int,
        keepdims: bool = True,
    ) -> Value:
        """Fused scalar engine activation + reduction of rightmost free dims.

        dst must be pre-allocated with the expected reduced shape.
        """
        if x.type.memory == MemorySpace.HBM:
            raise ValueError("activation_reduce: operand must be on-chip")
        if x.type.rank == 0:
            raise ValueError("activation_reduce: cannot reduce a scalar (rank 0)")
        if num_r_dim < 1:
            raise ValueError(
                f"activation_reduce: num_r_dim must be >= 1, got {num_r_dim}"
            )
        if num_r_dim >= x.type.rank:
            raise ValueError(
                f"activation_reduce: num_r_dim={num_r_dim} must be < rank={x.type.rank} "
                f"(cannot reduce partition dim; use cross_lane_reduce_arith)"
            )
        if not keepdims and x.type.rank - num_r_dim < 2:
            raise ValueError(
                "activation_reduce: reducing all free dims with keepdims=False "
                "would leave rank < 2, violating on-chip 2D tile convention"
            )
        # Compute expected reduced shape
        if keepdims:
            expected_shape = x.type.shape[:x.type.rank - num_r_dim] + (1,) * num_r_dim
        else:
            expected_shape = x.type.shape[:x.type.rank - num_r_dim]
        if dst.type.shape != expected_shape:
            raise ValueError(
                f"activation_reduce: dst shape {dst.type.shape} != "
                f"expected {expected_shape}"
            )
        return self._emit("activation_reduce", [dst, x], [dst.type], {
            "act_op": act_op, "reduce_op": reduce_op,
            "num_r_dim": num_r_dim, "keepdims": keepdims,
        }).result

    # ===========================
    # Control flow
    # ===========================

    def fori_loop(
        self,
        name: str,
        extent: int | Value,
        step: int,
        body_fn: Callable[..., None],
    ) -> None:
        """Side-effect loop, no carries. body_fn(b, index) -> None.

        Maps to ``nb.fori_loop``.  *extent* may be a static ``int`` or a
        dynamic ``Value`` (register loaded at runtime).  HBM buffers
        captured from outer scope are mutated via side-effect DMA stores.
        """
        body = Builder(f"{name}_body")
        body.graph.counter = self.graph.counter

        idx = Value(
            name=self.graph.counter.fresh(),
            type=TileType((), DType.I32, MemorySpace.REG),
        )
        body.graph.add_input(idx)

        body_fn(body, idx)

        inputs: list[Value] = []
        if isinstance(extent, Value):
            inputs.append(extent)

        self._emit("fori_loop", inputs, [], {
            "name": name,
            "extent": extent if isinstance(extent, int) else None,
            "step": step,
            "body": body.graph,
            "body_fn": body_fn,
        })

    def if_else(
        self,
        cond: Value,
        then_fn: Callable[..., None],
        else_fn: Callable[..., None] | None = None,
    ) -> None:
        """Dynamic conditional branching.  Maps to ``nb.if_else``.

        *cond* must be a scalar register (REG) with boolean semantics
        (typically from a comparison like ``i > 0``).
        *then_fn* and *else_fn* receive a Builder and emit ops into
        their respective branches.
        """
        then_b = Builder(f"if_then")
        then_b.graph.counter = self.graph.counter
        then_fn(then_b)

        else_graph = None
        else_body_fn = None
        if else_fn is not None:
            else_b = Builder(f"if_else")
            else_b.graph.counter = self.graph.counter
            else_fn(else_b)
            else_graph = else_b.graph
            else_body_fn = else_fn

        self._emit("if_else", [cond], [], {
            "then_body": then_b.graph,
            "then_fn": then_fn,
            "else_body": else_graph,
            "else_fn": else_body_fn,
        })

    def while_loop(
        self,
        init: Value,
        cond_fn: Callable[..., tuple[Value, Value]],
        body_fn: Callable[..., Value],
    ) -> Value:
        """Dynamic while loop with single carry register.

        Maps to ``nb.while_loop``.

        *init*: initial Reg value.
        *cond_fn(b, carry) -> (condition, output)*: returns bool Reg
        and the value to pass to body.
        *body_fn(b, carry) -> new_carry*: loop body.

        Returns the final carry value.
        """
        cond_b = Builder("while_cond")
        cond_b.graph.counter = self.graph.counter
        cond_ph = Value(name=self.graph.counter.fresh(), type=init.type)
        cond_b.graph.add_input(cond_ph)
        cond_result = cond_fn(cond_b, cond_ph)
        if isinstance(cond_result, tuple):
            cond_val, output_val = cond_result
        else:
            cond_val = cond_result
            output_val = cond_ph
        cond_b.set_outputs({"cond": cond_val, "output": output_val})

        body_b = Builder("while_body")
        body_b.graph.counter = self.graph.counter
        body_ph = Value(name=self.graph.counter.fresh(), type=init.type)
        body_b.graph.add_input(body_ph)
        new_carry = body_fn(body_b, body_ph)
        body_b.set_outputs({"carry": new_carry})

        op = self._emit("while_loop", [init], [init.type], {
            "cond_body": cond_b.graph,
            "cond_fn": cond_fn,
            "body_body": body_b.graph,
            "body_fn": body_fn,
        })
        return op.result

    # -- scalar register ops (maps to KB Reg arithmetic) --

    def reg_compare(self, a: Value, b: Value | int, op: str) -> Value:
        """Compare two scalar register values.  Returns bool-typed REG.

        *op* is one of: ``"<"``, ``"<="``, ``">"``, ``">="``, ``"!="``.
        Maps to Reg comparison operators in KB.
        """
        if isinstance(b, int):
            b = self.scalar_const(b)
        rt = TileType((), DType.BOOL, MemorySpace.REG)
        return self._emit("reg_compare", [a, b], [rt], {"op": op}).result

    def load_register(self, tile: Value) -> Value:
        """Load a scalar value from a tile into a register.

        Maps to ``nisa.load_register``.  Reads the element at index [0]
        of the tile.
        """
        rt = TileType((), tile.type.dtype, MemorySpace.REG)
        return self._emit("load_register", [tile], [rt]).result

    def store_register(self, dst: Value, reg: Value) -> Value:
        """Store a scalar register value into a tile.

        Maps to ``nisa.store_register``.
        """
        return self._emit("store_register", [dst, reg], [dst.type]).result

    # -- sugar --

    def neg(self, dst: Value, x: Value) -> Value:
        """Negate: dst = -x. Lowered to activation(COPY, scale=-1.0)."""
        return self.activation(dst, x, NisaActivationOp.COPY, scale=-1.0)

    # -- memset --

    def memset(self, tile: Value, value: float) -> Value:
        """Set all elements of a tile to a constant value."""
        if tile.type.memory == MemorySpace.HBM:
            raise ValueError("memset: tile must be on-chip")
        rt = tile.type
        return self._emit("memset", [tile], [rt], {"value": value}).result

    # -- graph outputs --

    def set_outputs(self, values: dict[str, Value]) -> None:
        self.graph.set_outputs(values)


# ===========================
# Passes
# ===========================

_LOOP_OPCODES = {"fori_loop", "tile_loop", "while_loop"}


def unroll_tile_loops(graph: Graph) -> int:
    """Unroll all tile_loop / fori_loop ops into flat op sequences.

    The pass re-calls each loop's body_fn with concrete int indices,
    which naturally produces ops with concrete offsets (no constant folding
    needed). Handles nested loops by iterating until none remain.

    Returns the number of loops unrolled.
    """
    count = 0
    while True:
        loop_op = None
        for op in graph.ops:
            if op.opcode in _LOOP_OPCODES:
                loop_op = op
                break
        if loop_op is None:
            break
        _unroll_one_loop(graph, loop_op)
        count += 1
    graph.toposort()
    # Clean up dead ops from the body graph representation (scalar_const
    # ops emitted for dynamic offsets in the body that are no longer used).
    graph.dce()
    return count


def _unroll_one_loop(graph: Graph, loop_op: Op) -> None:
    """Unroll a single loop by re-calling body_fn with concrete indices.

    body_fn must be a re-callable closure that captures outer Values (stable
    references) and Python ints (from already-unrolled outer loops). It must
    not capture or mutate any external state.
    """
    body_fn = loop_op.attrs["body_fn"]
    extent = loop_op.attrs["extent"]
    step = loop_op.attrs["step"]

    b = Builder._from_graph(graph)

    if loop_op.opcode == "fori_loop":
        # No carries — just call body_fn for each iteration
        for i in range(0, extent, step):
            body_fn(b, i)
        graph.erase_op(loop_op)
    else:
        # tile_loop: thread carried state
        carried = list(loop_op.inputs)
        for i in range(0, extent, step):
            results = body_fn(b, i, *carried)
            if isinstance(results, Value):
                carried = [results]
            else:
                carried = list(results)
        for old_r, new_val in zip(loop_op.results, carried):
            graph.replace_value(old_r, new_val)
        graph.erase_op(loop_op)
