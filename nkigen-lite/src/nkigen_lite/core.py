"""Shared IR infrastructure for tensor_ir and nki_ir.

Provides the common SSA-based IR core:
  - DType enum and numpy dtype mapping
  - ValueCounter, Value, Op (SSA primitives)
  - Graph (ordered op list with mutation helpers, DCE, verify, toposort)
  - Common numpy interpreter dispatch tables and helpers
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import ml_dtypes


# ===========================
# Types
# ===========================

class DType(str, Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    TF32 = "tf32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E4M3_IEEE = "fp8_e4m3_ieee"
    FP8_E5M2 = "fp8_e5m2"
    FP8_E3M4 = "fp8_e3m4"
    I32 = "i32"
    I16 = "i16"
    I8 = "i8"
    U32 = "u32"
    U16 = "u16"
    U8 = "u8"
    BOOL = "bool"

_DTYPE_TO_NP = {
    DType.F32: np.float32,
    DType.F16: np.float16,
    DType.BF16: ml_dtypes.bfloat16,
    DType.TF32: np.float32,
    DType.FP8_E4M3: ml_dtypes.float8_e4m3fn,
    DType.FP8_E4M3_IEEE: ml_dtypes.float8_e4m3,
    DType.FP8_E5M2: ml_dtypes.float8_e5m2,
    DType.FP8_E3M4: ml_dtypes.float8_e3m4,
    DType.I32: np.int32,
    DType.I16: np.int16,
    DType.I8: np.int8,
    DType.U32: np.uint32,
    DType.U16: np.uint16,
    DType.U8: np.uint8,
    DType.BOOL: np.bool_,
}

_DTYPE_BYTES = {
    DType.F32: 4,
    DType.F16: 2,
    DType.BF16: 2,
    DType.TF32: 4,
    DType.FP8_E4M3: 1,
    DType.FP8_E4M3_IEEE: 1,
    DType.FP8_E5M2: 1,
    DType.FP8_E3M4: 1,
    DType.I32: 4,
    DType.I16: 2,
    DType.I8: 1,
    DType.U32: 4,
    DType.U16: 2,
    DType.U8: 1,
    DType.BOOL: 1,
}


def to_np_dtype(dtype: DType) -> np.dtype:
    return np.dtype(_DTYPE_TO_NP[dtype])


# ===========================
# Values and Ops (SSA core)
# ===========================

class ValueCounter:
    """Per-graph counter for generating unique value names."""

    def __init__(self, prefix: str = "v") -> None:
        self._prefix = prefix
        self._count = 0

    def fresh(self) -> str:
        self._count += 1
        return f"{self._prefix}{self._count}"


@dataclass
class Value:
    name: str
    type: Any  # TensorType or TileType — kept generic for reuse
    producer: Op | None = None
    _uses: list[Op] = field(default_factory=list, repr=False, compare=False)

    @property
    def uses(self) -> list[Op]:
        """Snapshot of consuming ops, safe to iterate during mutation."""
        return list(self._uses)

    @property
    def has_uses(self) -> bool:
        return len(self._uses) > 0

    def replace_all_uses_with(self, new: Value) -> None:
        """Replace this value with *new* in every consuming op's inputs."""
        for op in dict.fromkeys(self._uses):  # each op visited once
            count = sum(1 for v in op.inputs if v is self)
            op.inputs = [new if v is self else v for v in op.inputs]
            for _ in range(count):
                new._uses.append(op)
        self._uses.clear()

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"%{self.name}"

    def __str__(self) -> str:
        return f"%{self.name}: {self.type}"


class Op:
    """A single operation in the IR graph.

    Important: do not mutate ``op.inputs`` directly — use
    ``Value.replace_all_uses_with`` or ``Graph.replace_value`` so that
    use-lists stay consistent.
    """

    def __init__(
        self,
        opcode: str,
        inputs: Sequence[Value],
        result_types: Sequence[Any],
        attrs: dict[str, Any] | None = None,
        *,
        counter: ValueCounter | None = None,
    ):
        self.opcode = opcode
        self.inputs = list(inputs)
        self.attrs = attrs or {}
        self._counter = counter or ValueCounter()
        self.results: list[Value] = []
        for rt in result_types:
            v = Value(name=self._counter.fresh(), type=rt, producer=self)
            self.results.append(v)
        for v in self.inputs:
            v._uses.append(self)

    @property
    def result(self) -> Value:
        assert len(self.results) == 1
        return self.results[0]

    def __str__(self) -> str:
        outs = ", ".join(str(v) for v in self.results)
        ins = ", ".join(repr(v) for v in self.inputs)
        a_parts = []
        for k, val in self.attrs.items():
            if isinstance(val, Graph):
                a_parts.append(f"{k}=<graph @{val.name}>")
            elif callable(val):
                continue  # skip non-serializable callables (e.g. body_fn)
            else:
                a_parts.append(f"{k}={val}")
        a = f" {{{', '.join(a_parts)}}}" if a_parts else ""
        return f"{outs} = {self.opcode}({ins}){a}"


# ===========================
# Graph
# ===========================

class Graph:
    """Ordered list of ops forming an IR program."""

    # Subclasses can override for dump output (e.g. "nki_graph")
    _graph_label = "graph"

    def __init__(self, name: str = "main"):
        self.name = name
        self.counter = ValueCounter()
        self.inputs: list[Value] = []
        self.ops: list[Op] = []
        self.outputs: dict[str, Value] = {}

    def add_input(self, v: Value) -> None:
        self.inputs.append(v)

    def append(self, op: Op) -> None:
        self.ops.append(op)

    def set_outputs(self, values: dict[str, Value]) -> None:
        self.outputs = dict(values)

    # -- mutation helpers --

    def insert_before(self, ref: Op, new_op: Op) -> None:
        """Insert *new_op* immediately before *ref* in the op list."""
        idx = self.ops.index(ref)
        self.ops.insert(idx, new_op)

    def insert_after(self, ref: Op, new_op: Op) -> None:
        """Insert *new_op* immediately after *ref* in the op list."""
        idx = self.ops.index(ref)
        self.ops.insert(idx + 1, new_op)

    def erase_op(self, op: Op) -> None:
        """Remove *op* from the graph.

        Raises ValueError if any of op's results still have uses.
        """
        for r in op.results:
            if r.has_uses:
                raise ValueError(
                    f"Cannot erase {op.opcode}: result {r!r} still has "
                    f"{len(r._uses)} use(s)"
                )
        for v in op.inputs:
            if op not in v._uses:
                raise ValueError(
                    f"use-list inconsistency: {op.opcode} not in {v!r}._uses"
                )
            v._uses.remove(op)
        self.ops.remove(op)

    def replace_value(self, old: Value, new: Value) -> None:
        """Replace *old* with *new* everywhere: op inputs and graph outputs."""
        old.replace_all_uses_with(new)
        for name in self.outputs:
            if self.outputs[name] is old:
                self.outputs[name] = new

    # -- passes --

    # Opcodes that are side-effecting (no results, but must not be DCE'd).
    # Empty in the base class; subclasses (e.g. nki_ir.Graph) override.
    _SIDE_EFFECT_OPCODES: set[str] = set()

    def dce(self) -> int:
        """Dead code elimination. Returns number of ops removed."""
        live_outputs = {id(v) for v in self.outputs.values()}
        dead: list[Op] = []
        for op in reversed(self.ops):
            if op.opcode in self._SIDE_EFFECT_OPCODES:
                continue
            alive = any(
                id(r) in live_outputs or r.has_uses
                for r in op.results
            )
            if alive:
                continue
            for v in op.inputs:
                v._uses.remove(op)
            dead.append(op)
        if dead:
            dead_ids = {id(op) for op in dead}
            self.ops = [op for op in self.ops if id(op) not in dead_ids]
        return len(dead)

    def toposort(self) -> None:
        """Re-sort ops into a valid topological (def-before-use) order."""
        producer_of: dict[str, Op] = {}
        for op in self.ops:
            for r in op.results:
                producer_of[r.name] = op

        op_ids = {id(op) for op in self.ops}

        in_degree: dict[int, int] = {id(op): 0 for op in self.ops}
        rdeps: dict[int, list[int]] = {id(op): [] for op in self.ops}
        for op in self.ops:
            seen: set[int] = set()
            for v in op.inputs:
                dep = producer_of.get(v.name)
                if dep is not None and id(dep) in op_ids and id(dep) != id(op):
                    if id(dep) not in seen:
                        seen.add(id(dep))
                        in_degree[id(op)] += 1
                        rdeps[id(dep)].append(id(op))

        id_to_op = {id(op): op for op in self.ops}
        queue = deque(oid for oid, deg in in_degree.items() if deg == 0)
        sorted_ops: list[Op] = []
        while queue:
            oid = queue.popleft()
            sorted_ops.append(id_to_op[oid])
            for succ in rdeps[oid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(sorted_ops) != len(self.ops):
            raise ValueError("toposort: cycle detected in graph")
        self.ops = sorted_ops

    def verify(self) -> list[str]:
        """Check graph invariants. Returns a list of error strings (empty = valid)."""
        errors: list[str] = []
        defined: dict[str, Value] = {}

        for v in self.inputs:
            if v.name in defined:
                errors.append(f"Duplicate input name: {v.name!r}")
            defined[v.name] = v

        for op in self.ops:
            for v in op.inputs:
                if v.name not in defined:
                    errors.append(
                        f"{op.opcode}: input {v!r} used before definition"
                    )
            for r in op.results:
                if r.name in defined:
                    errors.append(
                        f"{op.opcode}: result {r!r} shadows existing value"
                    )
                defined[r.name] = r
            for r in op.results:
                if r.producer is not op:
                    errors.append(
                        f"{op.opcode}: result {r!r} producer mismatch"
                    )

        for name, v in self.outputs.items():
            if v.name not in defined:
                errors.append(
                    f"Output {name!r} references undefined value {v!r}"
                )

        # Use-list consistency (identity-based to handle sub-graph scopes)
        def _collect_ops(g: Graph) -> list[Op]:
            ops = list(g.ops)
            for op in g.ops:
                for attr_val in op.attrs.values():
                    if isinstance(attr_val, Graph):
                        ops.extend(_collect_ops(attr_val))
            return ops

        expected_uses: dict[int, set[int]] = {}
        for op in _collect_ops(self):
            for v in op.inputs:
                expected_uses.setdefault(id(v), set()).add(id(op))

        for v in list(defined.values()):
            actual = {id(op) for op in v._uses}
            expected = expected_uses.get(id(v), set())
            if actual != expected:
                errors.append(
                    f"Value {v!r}: use-list inconsistent "
                    f"(expected {len(expected)} uses, got {len(actual)})"
                )

        return errors

    # -- accessors --

    @property
    def all_values(self) -> dict[str, Value]:
        vals: dict[str, Value] = {}
        for v in self.inputs:
            vals[v.name] = v
        for op in self.ops:
            for v in op.results:
                vals[v.name] = v
        return vals

    @property
    def output_values(self) -> list[Value]:
        """Output Values in insertion order (for positional access)."""
        return list(self.outputs.values())

    def dump(self, indent: int = 0) -> str:
        pad = "  " * indent
        out_sig = ", ".join(f"{k}: {v.type}" for k, v in self.outputs.items())
        ret_vals = ", ".join(f"{k}={v!r}" for k, v in self.outputs.items())
        lines = [f"{pad}{self._graph_label} @{self.name}("]
        for v in self.inputs:
            lines.append(f"{pad}  {v},")
        lines.append(f"{pad}) -> ({out_sig}) {{")
        for op in self.ops:
            lines.append(f"{pad}  {op}")
            for key in ("body", "true_body", "false_body"):
                if key in op.attrs and isinstance(op.attrs[key], Graph):
                    lines.append(op.attrs[key].dump(indent + 2))
        lines.append(f"{pad}  return {ret_vals}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.dump()

    def _repr_html_(self) -> str:
        """Rich display in Jupyter notebooks."""
        import html
        return f"<pre style='font-size:13px;line-height:1.4'>{html.escape(self.dump())}</pre>"


# ===========================
# Numpy interpreter helpers
# ===========================

NP_UNARY = {
    "neg": np.negative,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "reciprocal": np.reciprocal,
    "tanh": np.tanh,
    "sin": np.sin,
    "cos": np.cos,
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
}

NP_BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.true_divide,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "power": np.power,
    "floor_divide": np.floor_divide,
    "mod": np.mod,
}

NP_COMPARE = {
    "equal": np.equal,
    "not_equal": np.not_equal,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
    "less": np.less,
    "less_equal": np.less_equal,
}

NP_REDUCE = {
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
}


def eval_common_op(op: Op, get: callable, env: dict[str, np.ndarray]) -> bool:
    """Try to evaluate a common opcode, storing into env. Returns True if handled."""
    if op.opcode in NP_UNARY:
        env[op.result.name] = NP_UNARY[op.opcode](get(op.inputs[0]))
    elif op.opcode == "rsqrt":
        env[op.result.name] = 1.0 / np.sqrt(get(op.inputs[0]))
    elif op.opcode == "relu":
        env[op.result.name] = np.maximum(get(op.inputs[0]), 0)
    elif op.opcode == "gelu":
        orig = get(op.inputs[0])
        x = orig.astype(np.float64)
        env[op.result.name] = (
            0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        ).astype(orig.dtype)
    elif op.opcode == "sigmoid":
        env[op.result.name] = 1.0 / (1.0 + np.exp(-get(op.inputs[0])))
    elif op.opcode == "silu":
        x = get(op.inputs[0])
        env[op.result.name] = x / (1.0 + np.exp(-x))
    elif op.opcode in NP_BINARY:
        env[op.result.name] = NP_BINARY[op.opcode](get(op.inputs[0]), get(op.inputs[1]))
    elif op.opcode in NP_COMPARE:
        env[op.result.name] = NP_COMPARE[op.opcode](get(op.inputs[0]), get(op.inputs[1]))
    elif op.opcode == "constant":
        env[op.result.name] = np.full(
            op.result.type.shape,
            op.attrs["value"],
            dtype=to_np_dtype(op.result.type.dtype),
        )
    elif op.opcode == "reduce":
        env[op.result.name] = NP_REDUCE[op.attrs["kind"]](
            get(op.inputs[0]), axis=op.attrs["axis"], keepdims=op.attrs["keepdims"],
        )
    elif op.opcode == "transpose":
        env[op.result.name] = np.transpose(get(op.inputs[0]), op.attrs["perm"])
    elif op.opcode == "reshape":
        env[op.result.name] = np.reshape(get(op.inputs[0]), op.attrs["shape"])
    elif op.opcode == "cast":
        env[op.result.name] = get(op.inputs[0]).astype(to_np_dtype(op.attrs["dtype"]))
    elif op.opcode == "where":
        env[op.result.name] = np.where(get(op.inputs[0]), get(op.inputs[1]), get(op.inputs[2]))
    else:
        return False
    return True
