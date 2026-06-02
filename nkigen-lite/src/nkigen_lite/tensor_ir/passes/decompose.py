"""Decomposition pass for tensor IR.

Lowers ops that have no direct NISA equivalent into supported primitives:
  - div(a, b)                    → mul(a, reciprocal(b))
  - reduce(x, kind="mean")       → mul(reduce(x, kind="sum"), 1/N)

Pipeline:
  tensor_ir graph (canonical ops)
    → decompose()        # lower unsupported ops
  tensor_ir graph (decomposed ops)
    → tiling / legalize_to_nisa
"""

from __future__ import annotations

from math import prod

from nkigen_lite.core import Graph, Op


class DecomposePattern:
    """Base class for decomposition patterns.

    Mirrors ``CanonPattern`` from ``canonicalize``.
    """

    def match(self, op: Op) -> dict | None:
        raise NotImplementedError

    def rewrite(self, op: Op, data: dict, graph: Graph) -> None:
        raise NotImplementedError


class DivPattern(DecomposePattern):
    """div(a, b) → mul(a, reciprocal(b))"""

    def match(self, op):
        if op.opcode != "div":
            return None
        return {"a": op.inputs[0], "b": op.inputs[1]}

    def rewrite(self, op, data, graph):
        a, b = data["a"], data["b"]
        recip_op = Op("reciprocal", [b], [b.type], counter=graph.counter)
        graph.insert_before(op, recip_op)
        mul_op = Op("mul", [a, recip_op.result], [op.result.type], counter=graph.counter)
        graph.insert_before(op, mul_op)
        graph.replace_value(op.result, mul_op.result)


class ReduceMeanPattern(DecomposePattern):
    """reduce(x, kind="mean") → mul(reduce(x, kind="sum"), constant(1/N))"""

    def match(self, op):
        if op.opcode != "reduce" or op.attrs.get("kind") != "mean":
            return None
        x = op.inputs[0]
        axes = op.attrs["axis"]
        keepdims = op.attrs["keepdims"]
        n = prod(x.type.shape[a] for a in axes)
        return {"x": x, "axes": axes, "keepdims": keepdims, "inv_n": 1.0 / n}

    def rewrite(self, op, data, graph):
        sum_op = Op(
            "reduce", [data["x"]], [op.result.type],
            {"axis": data["axes"], "keepdims": data["keepdims"], "kind": "sum"},
            counter=graph.counter,
        )
        graph.insert_before(op, sum_op)
        inv_n_op = Op(
            "constant", [], [op.result.type],
            {"value": data["inv_n"]},
            counter=graph.counter,
        )
        graph.insert_before(op, inv_n_op)
        mul_op = Op(
            "mul", [sum_op.result, inv_n_op.result], [op.result.type],
            counter=graph.counter,
        )
        graph.insert_before(op, mul_op)
        graph.replace_value(op.result, mul_op.result)


DECOMPOSE_PATTERNS: list[DecomposePattern] = [
    DivPattern(),
    ReduceMeanPattern(),
]


def decompose(graph: Graph) -> int:
    """Lower ops that have no direct NISA equivalent into supported primitives.

    Must run **after** ``canonicalize`` so that patterns like
    ``div(1, sqrt(x)) → rsqrt`` fire first.

    Mutates *graph* in place. Returns the number of rewrites applied.
    """
    rewrites = 0
    for op in list(graph.ops):
        for pattern in DECOMPOSE_PATTERNS:
            data = pattern.match(op)
            if data is not None:
                pattern.rewrite(op, data, graph)
                rewrites += 1
                break
    graph.dce()
    return rewrites
