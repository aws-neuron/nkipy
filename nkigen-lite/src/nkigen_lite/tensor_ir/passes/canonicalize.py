"""Canonicalization pass for tensor IR.

Recomposes primitive-op chains into high-level ops:
  - div(1, sqrt(x))                       → rsqrt(x)
  - div(1, add(1, exp(neg(x))))           → sigmoid(x)
  - div(x, add(1, exp(neg(x))))           → silu(x)
  - mul(x, div(1, add(1, exp(neg(x)))))   → silu(x)

Pipeline:
  tensor_ir graph (primitive ops)
    → canonicalize()     # recompose high-level ops
    → decompose()        # lower unsupported ops  (see nkigen_lite.decompose)
  tensor_ir graph (canonical + decomposed ops)
    → tiling / legalize_to_nisa
"""

from __future__ import annotations

from nkigen_lite.core import Graph, Op, Value


# ===========================
# Helpers
# ===========================

def _is_constant(v: Value, val: float) -> bool:
    """True if v is produced by a constant op with the given value.

    Uses Python ``==`` for comparison, which is correct for the 0.0 and 1.0
    checks used by current patterns (note: -0.0 == 0.0 is True, NaN == NaN
    is False under these semantics).
    """
    return (
        v.producer is not None
        and v.producer.opcode == "constant"
        and v.producer.attrs["value"] == val
    )


def _extract_exp_neg_input(v: Value) -> Value | None:
    """If v = add(1, exp(neg(x))), return x. Otherwise None."""
    if v.producer is None or v.producer.opcode != "add":
        return None
    add_op = v.producer
    if _is_constant(add_op.inputs[0], 1.0):
        exp_v = add_op.inputs[1]
    elif _is_constant(add_op.inputs[1], 1.0):
        exp_v = add_op.inputs[0]
    else:
        return None

    if exp_v.producer is None or exp_v.producer.opcode != "exp":
        return None

    neg_v = exp_v.producer.inputs[0]
    if neg_v.producer is None or neg_v.producer.opcode != "neg":
        return None

    return neg_v.producer.inputs[0]


def _is_sigmoid_chain(v: Value, x: Value) -> bool:
    """True if v computes sigmoid(x) = 1 / (1 + exp(-x)).

    Walks backward: v → div(1, .) → 1+exp(-x).
    """
    if v.producer is None or v.producer.opcode != "div":
        return False
    if not _is_constant(v.producer.inputs[0], 1.0):
        return False
    return _extract_exp_neg_input(v.producer.inputs[1]) is x


def _insert_canonical(graph: Graph, root: Op, opcode: str, inputs: list[Value]) -> None:
    """Create a canonical op, insert before root, and RAUW root's result."""
    # Use graph.counter so the new value gets a unique SSA name
    new_op = Op(opcode, inputs, [root.result.type], counter=graph.counter)
    graph.insert_before(root, new_op)
    graph.replace_value(root.result, new_op.result)


# ===========================
# Canonicalization patterns
# ===========================

class CanonPattern:
    """Base class for canonicalization patterns.

    Subclasses implement ``match`` and ``rewrite`` in one place.
    ``match`` walks backward through producers to check structural patterns.
    ``rewrite`` creates a canonical op and RAUWs the root's result.
    Dead intermediate ops are cleaned up by DCE after all patterns run.
    """

    def match(self, op: Op) -> dict | None:
        """Try to match this pattern at *op*.

        Returns a dict of captured data, or None if no match.
        """
        raise NotImplementedError

    def rewrite(self, op: Op, data: dict, graph: Graph) -> None:
        """Create canonical op, insert before *op*, RAUW *op*'s result."""
        raise NotImplementedError


class RsqrtPattern(CanonPattern):
    """div(constant(1), sqrt(x)) → rsqrt(x)"""

    def match(self, op):
        if op.opcode != "div":
            return None
        if not _is_constant(op.inputs[0], 1.0):
            return None
        sqrt_v = op.inputs[1]
        if sqrt_v.producer is None or sqrt_v.producer.opcode != "sqrt":
            return None
        return {"x": sqrt_v.producer.inputs[0]}

    def rewrite(self, op, data, graph):
        _insert_canonical(graph, op, "rsqrt", [data["x"]])


class SigmoidPrimitivePattern(CanonPattern):
    """div(1, add(1, exp(neg(x)))) → sigmoid(x)"""

    def match(self, op):
        if op.opcode != "div":
            return None
        if not _is_constant(op.inputs[0], 1.0):
            return None
        x = _extract_exp_neg_input(op.inputs[1])
        if x is None:
            return None
        return {"x": x}

    def rewrite(self, op, data, graph):
        _insert_canonical(graph, op, "sigmoid", [data["x"]])


class SiluPrimitivePattern(CanonPattern):
    """Recognizes two primitive forms of SiLU:

    Form 1: div(x, add(1, exp(neg(x))))           — x / (1 + exp(-x))
    Form 2: mul(x, div(1, add(1, exp(neg(x)))))   — x * sigmoid(x)
    """

    def match(self, op):
        # Form 1: div(x, 1+exp(-x))
        if op.opcode == "div":
            x = op.inputs[0]
            denom = op.inputs[1]
            if _extract_exp_neg_input(denom) is x:
                return {"x": x}

        # Form 2: mul(x, sigmoid_chain)
        if op.opcode == "mul":
            a, b = op.inputs
            if _is_sigmoid_chain(b, a):
                return {"x": a}
            if _is_sigmoid_chain(a, b):
                return {"x": b}

        return None

    def rewrite(self, op, data, graph):
        _insert_canonical(graph, op, "silu", [data["x"]])


CANON_PATTERNS: list[CanonPattern] = [
    RsqrtPattern(),
    # SiLU before sigmoid: silu Form 2 (mul(x, sigmoid(x))) contains a sigmoid
    # sub-expression. If sigmoid fires first, silu Form 2 can no longer match.
    SiluPrimitivePattern(),
    SigmoidPrimitivePattern(),
]


# ===========================
# Main pass
# ===========================

def canonicalize(graph: Graph) -> int:
    """Rewrite primitive-op chains into canonical high-level ops.

    Mutates *graph* in place. Returns the number of rewrites applied.
    Single pass — patterns don't create new opportunities for each other.
    """
    rewrites = 0
    for pattern in CANON_PATTERNS:
        for op in list(graph.ops):  # snapshot, safe during mutation
            data = pattern.match(op)
            if data is not None:
                pattern.rewrite(op, data, graph)
                rewrites += 1
        # Clean dead ops between patterns so later patterns don't waste
        # work matching ops that were part of an already-rewritten chain.
        graph.dce()
    return rewrites
