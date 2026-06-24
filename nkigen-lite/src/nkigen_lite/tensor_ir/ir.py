"""Tensor-level IR with numpy-like builder and simulation.

Design goals:
  - SSA-based IR: every operation produces new Value(s), enabling clean
    transformation and analysis passes.
  - Numpy-like builder API: users write kernels in a familiar style.
  - Numpy interpreter: execute the IR graph with real data for correctness
    checking and rapid prototyping.
  - Minimal and extensible: easy to add new ops, write lowering passes,
    or convert to/from other IRs (e.g. design_lab.ir).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Sequence

import numpy as np

from nkigen_lite.core import (
    DType,
    Graph,
    Op,
    Value,
    ValueCounter,
    to_np_dtype,
    eval_common_op,
)


# ===========================
# Types
# ===========================

@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: DType

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __str__(self) -> str:
        dims = 'x'.join(str(s) for s in self.shape)
        return f"<{dims}x{self.dtype.value}>" if dims else f"<{self.dtype.value}>"


# ===========================
# Builder (numpy-like API)
# ===========================

class Builder:
    """Construct a tensor IR graph with a numpy-like API."""

    def __init__(self, name: str = "main"):
        self.graph = Graph(name)

    def _emit(
        self,
        opcode: str,
        inputs: Sequence[Value],
        result_types: Sequence[TensorType],
        attrs: dict[str, Any] | None = None,
    ) -> Op:
        op = Op(opcode, inputs, result_types, attrs, counter=self.graph.counter)
        self.graph.append(op)
        return op

    # -- graph inputs --

    def add_input(self, name: str, shape: tuple[int, ...], dtype: DType = DType.F32) -> Value:
        v = Value(name=name, type=TensorType(shape, dtype))
        self.graph.add_input(v)
        return v

    # -- elementwise unary --

    def _unary(self, opcode: str, x: Value) -> Value:
        return self._emit(opcode, [x], [x.type]).result

    def neg(self, x: Value) -> Value:
        return self._unary("neg", x)

    def exp(self, x: Value) -> Value:
        return self._unary("exp", x)

    def log(self, x: Value) -> Value:
        return self._unary("log", x)

    def sqrt(self, x: Value) -> Value:
        return self._unary("sqrt", x)

    def rsqrt(self, x: Value) -> Value:
        return self._unary("rsqrt", x)

    def reciprocal(self, x: Value) -> Value:
        return self._unary("reciprocal", x)

    def tanh(self, x: Value) -> Value:
        return self._unary("tanh", x)

    def relu(self, x: Value) -> Value:
        return self._unary("relu", x)

    def gelu(self, x: Value) -> Value:
        return self._unary("gelu", x)

    def sigmoid(self, x: Value) -> Value:
        return self._unary("sigmoid", x)

    def silu(self, x: Value) -> Value:
        return self._unary("silu", x)

    def sin(self, x: Value) -> Value:
        return self._unary("sin", x)

    def cos(self, x: Value) -> Value:
        return self._unary("cos", x)

    def arctan(self, x: Value) -> Value:
        return self._unary("arctan", x)

    def abs(self, x: Value) -> Value:
        return self._unary("abs", x)

    def sign(self, x: Value) -> Value:
        return self._unary("sign", x)

    def floor(self, x: Value) -> Value:
        return self._unary("floor", x)

    def ceil(self, x: Value) -> Value:
        return self._unary("ceil", x)

    # -- comparison (returns bool) --

    def _compare(self, opcode: str, a: Value, b: Value) -> Value:
        if a.type.dtype != b.type.dtype:
            raise ValueError(f"{opcode}: dtype mismatch {a.type.dtype} vs {b.type.dtype}")
        try:
            out_shape = np.broadcast_shapes(a.type.shape, b.type.shape)
        except ValueError:
            raise ValueError(
                f"{opcode}: shapes {a.type.shape} and {b.type.shape} are not broadcastable"
            )
        # Produce same dtype as input (1.0/0.0) — matches NKI convention
        rt = TensorType(out_shape, a.type.dtype)
        return self._emit(opcode, [a, b], [rt]).result

    def equal(self, a: Value, b: Value) -> Value:
        return self._compare("equal", a, b)

    def not_equal(self, a: Value, b: Value) -> Value:
        return self._compare("not_equal", a, b)

    def greater(self, a: Value, b: Value) -> Value:
        return self._compare("greater", a, b)

    def greater_equal(self, a: Value, b: Value) -> Value:
        return self._compare("greater_equal", a, b)

    def less(self, a: Value, b: Value) -> Value:
        return self._compare("less", a, b)

    def less_equal(self, a: Value, b: Value) -> Value:
        return self._compare("less_equal", a, b)

    # -- elementwise binary --

    def _binary(self, opcode: str, a: Value, b: Value) -> Value:
        if a.type.dtype != b.type.dtype:
            raise ValueError(f"{opcode}: dtype mismatch {a.type.dtype} vs {b.type.dtype}")
        try:
            out_shape = np.broadcast_shapes(a.type.shape, b.type.shape)
        except ValueError:
            raise ValueError(
                f"{opcode}: shapes {a.type.shape} and {b.type.shape} are not broadcastable"
            )
        rt = TensorType(out_shape, a.type.dtype)
        return self._emit(opcode, [a, b], [rt]).result

    def add(self, a: Value, b: Value) -> Value:
        return self._binary("add", a, b)

    def sub(self, a: Value, b: Value) -> Value:
        return self._binary("sub", a, b)

    def mul(self, a: Value, b: Value) -> Value:
        return self._binary("mul", a, b)

    def div(self, a: Value, b: Value) -> Value:
        return self._binary("div", a, b)

    def maximum(self, a: Value, b: Value) -> Value:
        return self._binary("maximum", a, b)

    def minimum(self, a: Value, b: Value) -> Value:
        return self._binary("minimum", a, b)

    def power(self, a: Value, b: Value) -> Value:
        return self._binary("power", a, b)

    def floor_divide(self, a: Value, b: Value) -> Value:
        return self._binary("floor_divide", a, b)

    def mod(self, a: Value, b: Value) -> Value:
        return self._binary("mod", a, b)

    # -- bitwise --

    def bitwise_and(self, a: Value, b: Value) -> Value:
        return self._binary("bitwise_and", a, b)

    def bitwise_or(self, a: Value, b: Value) -> Value:
        return self._binary("bitwise_or", a, b)

    def bitwise_xor(self, a: Value, b: Value) -> Value:
        return self._binary("bitwise_xor", a, b)

    # -- ternary --

    def where(self, cond: Value, a: Value, b: Value) -> Value:
        if a.type.dtype != b.type.dtype:
            raise ValueError(f"where: dtype mismatch {a.type.dtype} vs {b.type.dtype}")
        try:
            out_shape = np.broadcast_shapes(cond.type.shape, a.type.shape, b.type.shape)
        except ValueError:
            raise ValueError(
                f"where: shapes {cond.type.shape}, {a.type.shape}, and "
                f"{b.type.shape} are not broadcastable"
            )
        rt = TensorType(out_shape, a.type.dtype)
        return self._emit("where", [cond, a, b], [rt]).result

    # -- constants / creation --

    def constant(self, value: float, shape: tuple[int, ...], dtype: DType = DType.F32) -> Value:
        rt = TensorType(shape, dtype)
        return self._emit("constant", [], [rt], {"value": value}).result

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: DType = DType.F32) -> Value:
        return self.constant(fill_value, shape, dtype)

    def zeros(self, shape: tuple[int, ...], dtype: DType = DType.F32) -> Value:
        return self.constant(0.0, shape, dtype)

    def iota(self, shape: tuple[int, ...], dim: int = 0, dtype: DType = DType.I32) -> Value:
        """Index-ramp tensor: ``out[..., i, ...] == i`` along ``dim``.

        The value at each position equals its index along ``dim`` (a 0-based
        ramp), broadcast across all other axes — matching ``np.arange`` placed
        on ``dim``.  Maps to ``nisa.iota`` during lowering.
        """
        rank = len(shape)
        if rank == 0:
            raise ValueError("iota: shape must have rank >= 1")
        if dim < -rank or dim >= rank:
            raise ValueError(f"iota: dim {dim} out of range for rank {rank}")
        dim = dim % rank
        rt = TensorType(tuple(shape), dtype)
        return self._emit("iota", [], [rt], {"dim": dim}).result

    # -- reductions --

    def reduce(self, x: Value, axis: int | tuple[int, ...], kind: str = "sum", keepdims: bool = False) -> Value:
        if kind not in ("sum", "max", "min", "mean"):
            raise ValueError(f"reduce: unsupported kind {kind!r}")
        if x.type.rank == 0:
            raise ValueError(f"reduce: cannot reduce a scalar (rank 0)")
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        for a in axes:
            if a < -x.type.rank or a >= x.type.rank:
                raise ValueError(f"reduce: axis {a} out of range for rank {x.type.rank}")
        axes = tuple(a % x.type.rank for a in axes)
        if keepdims:
            new_shape = tuple(1 if i in axes else s for i, s in enumerate(x.type.shape))
        else:
            new_shape = tuple(s for i, s in enumerate(x.type.shape) if i not in axes)
        rt = TensorType(new_shape, x.type.dtype)
        return self._emit("reduce", [x], [rt], {"axis": axes, "keepdims": keepdims, "kind": kind}).result


    # -- shape manipulation --

    def transpose(self, x: Value, perm: tuple[int, ...]) -> Value:
        for p in perm:
            if p < -x.type.rank or p >= x.type.rank:
                raise ValueError(f"transpose: axis {p} out of range for rank {x.type.rank}")
        perm = tuple(p % x.type.rank for p in perm)
        if sorted(perm) != list(range(x.type.rank)):
            raise ValueError(f"transpose: invalid perm {perm} for rank {x.type.rank}")
        new_shape = tuple(x.type.shape[p] for p in perm)
        rt = TensorType(new_shape, x.type.dtype)
        return self._emit("transpose", [x], [rt], {"perm": perm}).result

    def reshape(self, x: Value, new_shape: tuple[int, ...]) -> Value:
        if prod(x.type.shape) != prod(new_shape):
            raise ValueError(f"reshape: size mismatch {x.type.shape} -> {new_shape}")
        rt = TensorType(new_shape, x.type.dtype)
        return self._emit("reshape", [x], [rt], {"shape": new_shape}).result

    def broadcast_to(self, x: Value, shape: tuple[int, ...]) -> Value:
        if len(shape) < x.type.rank:
            raise ValueError(f"broadcast_to: target rank must be >= source rank")
        offset = len(shape) - x.type.rank
        for i, src_dim in enumerate(x.type.shape):
            tgt_dim = shape[offset + i]
            if src_dim != 1 and src_dim != tgt_dim:
                raise ValueError(
                    f"broadcast_to: source dim {i} (size {src_dim}) is not "
                    f"broadcastable to target size {tgt_dim}"
                )
        rt = TensorType(shape, x.type.dtype)
        return self._emit("broadcast_to", [x], [rt], {"shape": shape}).result

    def expand_dims(self, x: Value, axis: int) -> Value:
        ndim = len(x.type.shape) + 1  # rank after insertion
        if axis < 0:
            axis = ndim + axis
        new_shape = list(x.type.shape)
        new_shape.insert(axis, 1)
        return self.reshape(x, tuple(new_shape))

    def squeeze(self, x: Value, axis: int) -> Value:
        if x.type.shape[axis] != 1:
            raise ValueError(f"squeeze: axis {axis} has size {x.type.shape[axis]}, expected 1")
        new_shape = list(x.type.shape)
        new_shape.pop(axis)
        return self.reshape(x, tuple(new_shape))

    def slice(
        self,
        x: Value,
        starts: tuple[int, ...],
        stops: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
    ) -> Value:
        rank = x.type.rank
        if len(starts) != rank or len(stops) != rank:
            raise ValueError(f"slice: starts/stops length must match rank {rank}")
        if strides is None:
            strides = (1,) * rank
        elif len(strides) != rank:
            raise ValueError(f"slice: strides length must match rank {rank}")
        new_shape = tuple(
            (stop - start + stride - 1) // stride
            for start, stop, stride in zip(starts, stops, strides)
        )
        for i, s in enumerate(new_shape):
            if s <= 0:
                raise ValueError(f"slice: empty or negative extent on axis {i}: "
                                 f"start={starts[i]}, stop={stops[i]}, stride={strides[i]}")
        rt = TensorType(new_shape, x.type.dtype)
        return self._emit("slice", [x], [rt], {
            "starts": starts, "stops": stops, "strides": strides,
        }).result

    def split(self, x: Value, num_or_sizes: int | Sequence[int], axis: int = 0) -> list[Value]:
        axis = axis % x.type.rank
        if isinstance(num_or_sizes, int):
            n = num_or_sizes
            if x.type.shape[axis] % n != 0:
                raise ValueError(
                    f"split: axis {axis} size {x.type.shape[axis]} not divisible by {n}"
                )
            chunk = x.type.shape[axis] // n
            sizes = [chunk] * n
        else:
            sizes = list(num_or_sizes)
            if sum(sizes) != x.type.shape[axis]:
                raise ValueError(
                    f"split: sizes {sizes} don't sum to axis {axis} size {x.type.shape[axis]}"
                )
        # Emit a sequence of slice ops — easier to pattern-match in lowering.
        rank = x.type.rank
        results: list[Value] = []
        offset = 0
        for s in sizes:
            starts = tuple(0 if i != axis else offset for i in range(rank))
            stops = tuple(x.type.shape[i] if i != axis else offset + s for i in range(rank))
            results.append(self.slice(x, starts, stops))
            offset += s
        return results

    def concat(self, inputs: Sequence[Value], axis: int) -> Value:
        if len(inputs) < 2:
            raise ValueError("concat: need at least 2 inputs")
        ref = inputs[0]
        for v in inputs[1:]:
            if v.type.rank != ref.type.rank:
                raise ValueError(f"concat: rank mismatch {ref.type.rank} vs {v.type.rank}")
            if v.type.dtype != ref.type.dtype:
                raise ValueError(f"concat: dtype mismatch {ref.type.dtype} vs {v.type.dtype}")
            for i, (s1, s2) in enumerate(zip(ref.type.shape, v.type.shape)):
                if i != axis and s1 != s2:
                    raise ValueError(f"concat: shape mismatch on axis {i}: {s1} vs {s2}")
        new_shape = list(ref.type.shape)
        new_shape[axis] = sum(v.type.shape[axis] for v in inputs)
        rt = TensorType(tuple(new_shape), ref.type.dtype)
        return self._emit("concat", list(inputs), [rt], {"axis": axis}).result

    # -- top-8 selection (hardware max8 / find_index8) --

    def max8(self, x: Value) -> Value:
        """8 largest values per row (last axis), descending. ``x`` is 2-D
        ``(P, F)`` with 8 <= F <= 16384; result is ``(P, 8)``."""
        if x.type.rank != 2:
            raise ValueError(f"max8: input must be 2-D, got rank {x.type.rank}")
        if not (8 <= x.type.shape[1] <= 16384):
            raise ValueError(
                f"max8: free dim must be in [8, 16384], got {x.type.shape[1]}"
            )
        rt = TensorType((x.type.shape[0], 8), x.type.dtype)
        return self._emit("max8", [x], [rt]).result

    def find_index8(self, x: Value, vals: Value) -> Value:
        """First-match index of each of the 8 ``vals`` within each row of
        ``x``.  ``x`` is ``(P, F)``, ``vals`` is ``(P, 8)``; result ``(P, 8)``
        int32."""
        if x.type.rank != 2 or vals.type.rank != 2:
            raise ValueError("find_index8: inputs must be 2-D")
        rt = TensorType((x.type.shape[0], 8), DType.I32)
        return self._emit("find_index8", [x, vals], [rt]).result

    # -- matmul --

    def matmul(self, a: Value, b: Value) -> Value:
        if a.type.rank < 1 or b.type.rank < 1:
            raise TypeError("matmul: inputs must be at least 1-D")
        if a.type.dtype != b.type.dtype:
            raise TypeError(f"matmul: dtype mismatch {a.type.dtype} vs {b.type.dtype}")
        if a.type.shape[-1] != b.type.shape[-2 if b.type.rank >= 2 else 0]:
            raise TypeError(
                f"matmul: contraction dim mismatch: "
                f"{a.type.shape[-1]} vs {b.type.shape[-2 if b.type.rank >= 2 else 0]}"
            )
        a_batch = a.type.shape[:-2] if a.type.rank > 2 else ()
        b_batch = b.type.shape[:-2] if b.type.rank > 2 else ()
        try:
            batch = np.broadcast_shapes(a_batch, b_batch) if (a_batch or b_batch) else ()
        except ValueError:
            raise TypeError(
                f"matmul: batch shapes {a_batch} and {b_batch} are not broadcastable"
            )
        if a.type.rank >= 2 and b.type.rank >= 2:
            out_shape = batch + (a.type.shape[-2], b.type.shape[-1])
        elif a.type.rank == 1 and b.type.rank >= 2:
            out_shape = b_batch + (b.type.shape[-1],)
        elif b.type.rank == 1:
            out_shape = a.type.shape[:-1]
        else:
            out_shape = ()
        rt = TensorType(out_shape, a.type.dtype)
        return self._emit("matmul", [a, b], [rt]).result

    # -- collective communication --

    def all_reduce(self, x: Value, replica_groups, reduce_op: str = "add") -> Value:
        """All-reduce across the replica group; output shape == input shape."""
        rt = TensorType(x.type.shape, x.type.dtype)
        return self._emit(
            "all_reduce", [x], [rt],
            {"replica_groups": replica_groups, "reduce_op": reduce_op},
        ).result

    def all_gather(self, x: Value, all_gather_dim: int, replica_groups) -> Value:
        """All-gather; the gather dim grows by the replica-group size."""
        world = len(replica_groups[0])
        dim = all_gather_dim % x.type.rank
        out_shape = tuple(
            s * world if i == dim else s for i, s in enumerate(x.type.shape)
        )
        rt = TensorType(out_shape, x.type.dtype)
        return self._emit(
            "all_gather", [x], [rt],
            {"all_gather_dim": dim, "replica_groups": replica_groups},
        ).result

    def reduce_scatter(
        self, x: Value, reduce_scatter_dim: int, replica_groups, reduce_op: str = "add"
    ) -> Value:
        """Reduce-scatter; the scatter dim shrinks by the replica-group size."""
        world = len(replica_groups[0])
        dim = reduce_scatter_dim % x.type.rank
        if x.type.shape[dim] % world != 0:
            raise ValueError(
                f"reduce_scatter: dim {dim} size {x.type.shape[dim]} not "
                f"divisible by world size {world}"
            )
        out_shape = tuple(
            s // world if i == dim else s for i, s in enumerate(x.type.shape)
        )
        rt = TensorType(out_shape, x.type.dtype)
        return self._emit(
            "reduce_scatter", [x], [rt],
            {"reduce_scatter_dim": dim, "replica_groups": replica_groups,
             "reduce_op": reduce_op},
        ).result

    def all_to_all(
        self, x: Value, split_dimension: int, concat_dimension: int, replica_groups
    ) -> Value:
        """All-to-all; split dim shrinks and concat dim grows by world size."""
        world = len(replica_groups[0])
        rank = x.type.rank
        split_dim = split_dimension % rank
        concat_dim = concat_dimension % rank
        if x.type.shape[split_dim] % world != 0:
            raise ValueError(
                f"all_to_all: split dim {split_dim} size "
                f"{x.type.shape[split_dim]} not divisible by world size {world}"
            )
        out = list(x.type.shape)
        out[split_dim] //= world
        out[concat_dim] *= world
        rt = TensorType(tuple(out), x.type.dtype)
        return self._emit(
            "all_to_all", [x], [rt],
            {"split_dimension": split_dim, "concat_dimension": concat_dim,
             "replica_groups": replica_groups},
        ).result

    # -- type cast --

    def cast(self, x: Value, dtype: DType) -> Value:
        rt = TensorType(x.type.shape, dtype)
        return self._emit("cast", [x], [rt], {"dtype": dtype}).result

    # -- graph outputs --

    def set_outputs(self, values: dict[str, Value]) -> None:
        self.graph.set_outputs(values)

    # -- control flow --

    @staticmethod
    def _trace_body(
        name: str,
        input_types: Sequence[tuple[str, TensorType]],
        body_fn: Callable[..., Value | Sequence[Value]],
    ) -> Graph:
        """Trace body_fn into a sub-graph by calling it with add_input Values."""
        body = Builder(name)
        add_inputs = [body.add_input(n, t.shape, t.dtype) for n, t in input_types]
        results = body_fn(body, *add_inputs)
        if isinstance(results, Value):
            results = [results]
        else:
            results = list(results)
        body.set_outputs({f"out_{j}": v for j, v in enumerate(results)})
        return body.graph

    def for_loop(
        self,
        trip_count: int,
        init: Sequence[Value],
        body_fn: Callable[..., Value | Sequence[Value]],
    ) -> tuple[Value, ...]:
        """Fixed-trip-count loop with carried state.

        body_fn(b: Builder, i: Value, *carried) -> new_carried
        """
        input_types: list[tuple[str, TensorType]] = [
            ("i", TensorType((), DType.I32)),
        ]
        for j, v in enumerate(init):
            input_types.append((f"carry_{j}", v.type))

        body_graph = self._trace_body("for_body", input_types, body_fn)
        result_types = [v.type for v in init]
        op = self._emit("for_loop", list(init), result_types, {
            "trip_count": trip_count,
            "body": body_graph,
        })
        return tuple(op.results)



# ===========================
# Numpy interpreter
# ===========================

def interpret(
    graph: Graph,
    inputs: dict[str, np.ndarray],
    outer_env: dict[str, np.ndarray] | None = None,
    extra_eval: Callable | None = None,
) -> dict[str, np.ndarray]:
    """Execute a Graph with numpy, returning a map of value-name -> ndarray.

    ``outer_env``, when provided, makes captured (free) values from an
    enclosing graph available to ops in this graph.

    ``extra_eval``, when provided, is called as ``extra_eval(op, get, env)``
    before the default dispatch.  It should return True if it handled the op.
    This allows extension modules (e.g. nisa_ir) to add interpreter support
    for their opcodes without modifying this file.
    """
    env: dict[str, np.ndarray] = {}
    if outer_env is not None:
        env.update(outer_env)

    for v in graph.inputs:
        if v.name not in inputs:
            raise ValueError(f"Missing input: {v.name}")
        arr = inputs[v.name]
        if tuple(arr.shape) != v.type.shape:
            raise ValueError(f"Shape mismatch for {v.name}: expected {v.type.shape}, got {arr.shape}")
        env[v.name] = arr.astype(to_np_dtype(v.type.dtype))

    def _get(v: Value) -> np.ndarray:
        return env[v.name]

    for op in graph.ops:
        if extra_eval is not None and extra_eval(op, _get, env):
            pass
        elif eval_common_op(op, _get, env):
            pass
        elif op.opcode == "broadcast_to":
            env[op.result.name] = np.broadcast_to(_get(op.inputs[0]), op.attrs["shape"]).copy()
        elif op.opcode == "slice":
            slices = tuple(
                slice(s, e, st)
                for s, e, st in zip(op.attrs["starts"], op.attrs["stops"], op.attrs["strides"])
            )
            env[op.result.name] = _get(op.inputs[0])[slices].copy()
        elif op.opcode == "matmul":
            env[op.result.name] = np.matmul(_get(op.inputs[0]), _get(op.inputs[1]))
        elif op.opcode == "concat":
            env[op.result.name] = np.concatenate(
                [_get(v) for v in op.inputs], axis=op.attrs["axis"]
            )
        elif op.opcode == "max8":
            src = _get(op.inputs[0]).astype(np.float32)
            out = np.sort(src, axis=1)[:, ::-1][:, :8]
            env[op.result.name] = out.astype(to_np_dtype(op.result.type.dtype))
        elif op.opcode == "find_index8":
            src = _get(op.inputs[0]).astype(np.float32)
            vals = _get(op.inputs[1]).astype(np.float32)
            out = np.zeros((src.shape[0], 8), dtype=np.int64)
            for p in range(src.shape[0]):
                for i in range(min(8, vals.shape[1])):
                    m = np.where(src[p] == vals[p, i])[0]
                    if len(m) > 0:
                        out[p, i] = m[0]
            env[op.result.name] = out.astype(to_np_dtype(op.result.type.dtype))
        elif op.opcode == "for_loop":
            body = op.attrs["body"]
            trip_count = op.attrs["trip_count"]
            carried = [_get(v) for v in op.inputs]
            for i in range(trip_count):
                body_inputs = {body.inputs[0].name: np.array(i, dtype=np.int32)}
                for j, bv in enumerate(body.inputs[1:]):
                    body_inputs[bv.name] = carried[j]
                body_env = interpret(body, body_inputs, outer_env=env, extra_eval=extra_eval)
                carried = [body_env[bv.name] for bv in body.output_values]
            for j, rv in enumerate(op.results):
                env[rv.name] = carried[j]
        else:
            raise NotImplementedError(f"Interpreter: unknown opcode {op.opcode!r}")

        # Validate interpreter results match declared types
        for r in op.results:
            if r.name in env:
                actual = env[r.name].shape
                expected = r.type.shape
                if tuple(actual) != expected:
                    raise RuntimeError(
                        f"Interpreter bug: {op.opcode} result {r.name} "
                        f"has shape {tuple(actual)}, expected {expected}"
                    )

    return env


def run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute and return named output arrays."""
    if not graph.outputs:
        raise ValueError("Graph has no outputs. Call builder.set_outputs().")
    env = interpret(graph, inputs)
    return {name: env[v.name] for name, v in graph.outputs.items()}
