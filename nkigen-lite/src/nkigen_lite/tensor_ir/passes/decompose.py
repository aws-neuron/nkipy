"""Decomposition pass for tensor IR.

Lowers ops that have no direct NISA equivalent into supported primitives:
  - div(a, b)           → mul(a, reciprocal(b))
  - floor_divide(a, b)  → floor(div(a,b)) + verify-and-correct
  - mod(a, b)           → a - b * floor_divide(a, b)
  - power(a, b)         → exp(b * log(a))
  - ceil(x)             → neg(floor(neg(x)))
  - reduce(kind="mean") → mul(reduce(kind="sum"), 1/N)

Pipeline:
  tensor_ir graph (canonical ops)
    → decompose()        # lower unsupported ops
  tensor_ir graph (decomposed ops, only NISA-supported opcodes)
    → layout_solver → direct_lower

Floor-divide precision strategy
================================
NeuronCore has no native division instruction — only ``reciprocal`` (NISA
scalar engine), which gives ~23-bit precision.  A naive ``floor(a *
reciprocal(b))`` produces wrong results when ``a/b`` lands within 1 ULP of
an exact integer (e.g. ``0.6 / 0.2`` computes as ``2.9999...`` → floor gives
2 instead of 3).

We adopt the same **divide-then-verify-and-correct** strategy used by
neuronx-cc's tensorizer for HLO ``floor_divide``, verified by inspecting
the generated BIR (``penguin.py`` + ``bir.json``):

  1. Approximate: ``q = floor(a * reciprocal(b))``
  2. Back-verify: ``rem = a - b * q``
  3. Correct down: if ``sign(rem) ≠ sign(b)`` → ``q -= 1``
     (reciprocal over-estimated, quotient too high)
  4. Correct up: if ``|rem| ≥ |b|`` → ``q += 1``
     (reciprocal under-estimated, quotient too low)

This eliminates >99.99% of precision errors.  A residual ~1/65536 error
rate can occur when operands are broadcast via HBM scratch *before*
division (the extra DMA round-trip introduces additional float rounding).
The HLO compiler avoids this by fusing broadcast into the division's DMA
schedule at the instruction level — a lowering optimization not yet
implemented in nkigen-lite.
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


class ReduceKeepdimsFalsePattern(DecomposePattern):
    """reduce(x, keepdims=False) → reshape(reduce(x, keepdims=True), squeezed_shape)

    The layout solver and direct lowering require keepdims=True so that the
    reduce output retains the same rank/layout as its input. We decompose
    keepdims=False into a keepdims=True reduce followed by a reshape that
    drops the reduced dimensions.
    """

    def match(self, op):
        if op.opcode != "reduce":
            return None
        if op.attrs.get("keepdims", False):
            return None
        x = op.inputs[0]
        axes = op.attrs["axis"]
        kind = op.attrs["kind"]
        keepdims_shape = tuple(
            1 if i in axes else s for i, s in enumerate(x.type.shape)
        )
        return {"x": x, "axes": axes, "kind": kind, "keepdims_shape": keepdims_shape}

    def rewrite(self, op, data, graph):
        from nkigen_lite.tensor_ir.ir import TensorType

        keepdims_type = TensorType(data["keepdims_shape"], op.result.type.dtype)
        reduce_op = Op(
            "reduce", [data["x"]], [keepdims_type],
            {"axis": data["axes"], "keepdims": True, "kind": data["kind"]},
            counter=graph.counter,
        )
        graph.insert_before(op, reduce_op)
        reshape_op = Op(
            "reshape", [reduce_op.result], [op.result.type],
            {"shape": op.result.type.shape},
            counter=graph.counter,
        )
        graph.insert_before(op, reshape_op)
        graph.replace_value(op.result, reshape_op.result)


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


class FloorDividePattern(DecomposePattern):
    """floor_divide(a, b) → divide-then-verify-and-correct.

    Mirrors the strategy used by neuronx-cc's tensorizer (verified via BIR
    inspection of HLO floor_divide compilation artifacts on trn2).

    The BIR sequence from neuronx-cc is:
      [0-1] Load a, b
      [2]   Reciprocal(b)              — approximate 1/b
      [3]   TensorTensor(a, 1/b)       — q_approx = a * (1/b)
      [4]   GenericCopy(q_approx)      — f32 → f32 (for floor)
      [5]   GenericCopy(q_approx)      — f32 → i32 (truncate to int)
      [6]   TensorTensor(b, trunc_q)   — b * trunc_q (back-verify)
      [7]   TensorScalarPtr(xor)       — sign bit comparison
      [8]   TensorScalarPtr(mult,add)  — conditional correction
      [9-11] TensorTensor              — final result assembly
      [12]  Save

    Our decomposition emits the equivalent logic at tensor IR level:
      1. q = floor(a * reciprocal(b))
      2. rem = a - b * q
      3. corr_down = max(0, -(sign(rem) * sign(b)))  [signs differ → 1]
      4. corr_up = max(0, sign(|rem| - |b|))         [|rem| ≥ |b| → 1]
      5. result = q - corr_down + corr_up
    """

    def match(self, op):
        if op.opcode != "floor_divide":
            return None
        return {"a": op.inputs[0], "b": op.inputs[1]}

    def rewrite(self, op, data, graph):
        a, b = data["a"], data["b"]
        rt = op.result.type

        # Step 1: approximate quotient q = floor(a / b)
        div_op = Op("div", [a, b], [rt], counter=graph.counter)
        graph.insert_before(op, div_op)
        floor_op = Op("floor", [div_op.result], [rt], counter=graph.counter)
        graph.insert_before(op, floor_op)

        # Step 2: remainder = a - b * q
        mul_bq = Op("mul", [b, floor_op.result], [rt], counter=graph.counter)
        graph.insert_before(op, mul_bq)
        rem = Op("sub", [a, mul_bq.result], [rt], counter=graph.counter)
        graph.insert_before(op, rem)

        # Step 3: two corrections using sign comparison (matches neuronx-cc BIR)
        # Correction 1: if sign(rem) != sign(b) and rem != 0, subtract 1
        #   (floor was too high — remainder went negative for positive b)
        # Correction 2: if sign(rem) == sign(b) and abs(rem) >= abs(b), add 1
        #   (floor was too low — remainder exceeds divisor)
        sign_rem = Op("sign", [rem.result], [rt], counter=graph.counter)
        graph.insert_before(op, sign_rem)
        sign_b = Op("sign", [b], [rt], counter=graph.counter)
        graph.insert_before(op, sign_b)

        # sign_prod = sign(rem) * sign(b): negative when signs differ
        sign_prod = Op("mul", [sign_rem.result, sign_b.result], [rt], counter=graph.counter)
        graph.insert_before(op, sign_prod)

        # corr_down = max(0, -sign_prod): 1 when remainder has wrong sign
        neg_one = Op("constant", [], [rt], {"value": -1.0}, counter=graph.counter)
        graph.insert_before(op, neg_one)
        neg_sp = Op("mul", [sign_prod.result, neg_one.result], [rt], counter=graph.counter)
        graph.insert_before(op, neg_sp)
        zero = Op("constant", [], [rt], {"value": 0.0}, counter=graph.counter)
        graph.insert_before(op, zero)
        corr_down = Op("maximum", [neg_sp.result, zero.result], [rt], counter=graph.counter)
        graph.insert_before(op, corr_down)

        # corr_up: check if |rem| >= |b| (floor was too low)
        abs_rem = Op("abs", [rem.result], [rt], counter=graph.counter)
        graph.insert_before(op, abs_rem)
        abs_b = Op("abs", [b], [rt], counter=graph.counter)
        graph.insert_before(op, abs_b)
        # corr_up = (|rem| >= |b|) -> 1.0/0.0.
        #
        # Must be an INCLUSIVE compare: when the true quotient is an exact
        # integer N, the reciprocal-based divide undershoots to N-eps so
        # floor gives N-1, leaving rem == b exactly (i.e. |rem| == |b|). A
        # genuine remainder is always strictly < |b|, so |rem| == |b| can
        # only mean undershoot. The previous `max(0, sign(|rem|-|b|))` form
        # returned 0 at that boundary (sign(0)==0) and missed the correction.
        corr_up = Op("greater_equal", [abs_rem.result, abs_b.result], [rt],
                     counter=graph.counter)
        graph.insert_before(op, corr_up)

        # Step 4: result = q - corr_down + corr_up
        q_corrected = Op("sub", [floor_op.result, corr_down.result], [rt], counter=graph.counter)
        graph.insert_before(op, q_corrected)
        result = Op("add", [q_corrected.result, corr_up.result], [rt], counter=graph.counter)
        graph.insert_before(op, result)
        graph.replace_value(op.result, result.result)


class ModPattern(DecomposePattern):
    """mod(a, b) → a - b * floor_divide(a, b)

    Uses the corrected floor_divide (which will be further decomposed).
    """

    def match(self, op):
        if op.opcode != "mod":
            return None
        return {"a": op.inputs[0], "b": op.inputs[1]}

    def rewrite(self, op, data, graph):
        a, b = data["a"], data["b"]
        rt = op.result.type
        # floor_divide(a, b) — will be decomposed by FloorDividePattern
        fdiv_op = Op("floor_divide", [a, b], [rt], counter=graph.counter)
        graph.insert_before(op, fdiv_op)
        # b * floor_divide(a, b)
        mul_bq = Op("mul", [b, fdiv_op.result], [rt], counter=graph.counter)
        graph.insert_before(op, mul_bq)
        # a - b * q
        sub_op = Op("sub", [a, mul_bq.result], [rt], counter=graph.counter)
        graph.insert_before(op, sub_op)
        graph.replace_value(op.result, sub_op.result)


class CeilPattern(DecomposePattern):
    """ceil(x) → neg(floor(neg(x)))"""

    def match(self, op):
        if op.opcode != "ceil":
            return None
        return {"x": op.inputs[0]}

    def rewrite(self, op, data, graph):
        x = data["x"]
        neg1_op = Op("neg", [x], [x.type], counter=graph.counter)
        graph.insert_before(op, neg1_op)
        floor_op = Op("floor", [neg1_op.result], [x.type], counter=graph.counter)
        graph.insert_before(op, floor_op)
        neg2_op = Op("neg", [floor_op.result], [x.type], counter=graph.counter)
        graph.insert_before(op, neg2_op)
        graph.replace_value(op.result, neg2_op.result)


class PowerPattern(DecomposePattern):
    """power(a, b) → exp(mul(b, log(a)))

    NISA POW only supports scalar exponents via tensor_scalar_arith.
    For general tensor-tensor power, decompose into exp/log.
    """

    def match(self, op):
        if op.opcode != "power":
            return None
        return {"a": op.inputs[0], "b": op.inputs[1]}

    def rewrite(self, op, data, graph):
        a, b = data["a"], data["b"]
        log_op = Op("log", [a], [a.type], counter=graph.counter)
        graph.insert_before(op, log_op)
        mul_op = Op("mul", [b, log_op.result], [op.result.type], counter=graph.counter)
        graph.insert_before(op, mul_op)
        exp_op = Op("exp", [mul_op.result], [op.result.type], counter=graph.counter)
        graph.insert_before(op, exp_op)
        graph.replace_value(op.result, exp_op.result)


class CosPattern(DecomposePattern):
    """cos(x) → sin(x + π/2)"""

    def match(self, op):
        if op.opcode != "cos":
            return None
        return {"x": op.inputs[0]}

    def rewrite(self, op, data, graph):
        import math
        x = data["x"]
        rt = op.result.type
        half_pi = Op("constant", [], [rt], {"value": math.pi / 2}, counter=graph.counter)
        graph.insert_before(op, half_pi)
        shifted = Op("add", [x, half_pi.result], [rt], counter=graph.counter)
        graph.insert_before(op, shifted)
        sin_op = Op("sin", [shifted.result], [rt], counter=graph.counter)
        graph.insert_before(op, sin_op)
        graph.replace_value(op.result, sin_op.result)


class SinRangeReductionPattern(DecomposePattern):
    """sin(x) → sin(x - 2π·round(x / 2π))

    The hardware SIN activation is only accurate for arguments near
    [-π, π]; outside that the polynomial approximation diverges wildly
    (cos(x) for x≈500 returns ~2e7 instead of a value in [-1, 1]).  Reduce
    the argument modulo 2π first.  round(y) = floor(y + 0.5).

    The emitted inner ``sin`` carries ``range_reduced`` so the pattern does
    not re-match it (which would loop forever).
    """

    TWO_PI = 6.283185307179586
    INV_TWO_PI = 0.15915494309189535  # 1 / (2π)

    def match(self, op):
        if op.opcode != "sin" or op.attrs.get("range_reduced"):
            return None
        return {"x": op.inputs[0]}

    def rewrite(self, op, data, graph):
        x = data["x"]
        rt = op.result.type

        inv = Op("constant", [], [rt], {"value": self.INV_TWO_PI}, counter=graph.counter)
        graph.insert_before(op, inv)
        scaled = Op("mul", [x, inv.result], [rt], counter=graph.counter)
        graph.insert_before(op, scaled)
        # round-to-nearest: floor(y + 0.5)
        half = Op("constant", [], [rt], {"value": 0.5}, counter=graph.counter)
        graph.insert_before(op, half)
        biased = Op("add", [scaled.result, half.result], [rt], counter=graph.counter)
        graph.insert_before(op, biased)
        k = Op("floor", [biased.result], [rt], counter=graph.counter)
        graph.insert_before(op, k)
        # x_reduced = x - k * 2π
        two_pi = Op("constant", [], [rt], {"value": self.TWO_PI}, counter=graph.counter)
        graph.insert_before(op, two_pi)
        k2pi = Op("mul", [k.result, two_pi.result], [rt], counter=graph.counter)
        graph.insert_before(op, k2pi)
        x_red = Op("sub", [x, k2pi.result], [rt], counter=graph.counter)
        graph.insert_before(op, x_red)

        sin_op = Op(
            "sin", [x_red.result], [rt], {"range_reduced": True}, counter=graph.counter
        )
        graph.insert_before(op, sin_op)
        graph.replace_value(op.result, sin_op.result)


DECOMPOSE_PATTERNS: list[DecomposePattern] = [
    # ReduceKeepdimsFalse must run before ReduceMean so keepdims=False reduces
    # become keepdims=True+reshape before mean decomposition fires.
    ReduceKeepdimsFalsePattern(),
    # FloorDivide/Mod must run before DivPattern since they emit 'div' nodes
    # that DivPattern will decompose in a subsequent iteration.
    FloorDividePattern(),
    ModPattern(),
    PowerPattern(),
    CeilPattern(),
    CosPattern(),
    # After CosPattern so cos→sin first, then both sins get range-reduced.
    SinRangeReductionPattern(),
    DivPattern(),
    ReduceMeanPattern(),
]


def decompose(graph: Graph) -> int:
    """Lower ops that have no direct NISA equivalent into supported primitives.

    Must run **after** ``canonicalize`` so that patterns like
    ``div(1, sqrt(x)) → rsqrt`` fire first.

    Iterates until no more patterns match (fixed-point), since some
    decompositions (e.g. floor_divide → div → mul+reciprocal) are multi-step.

    Mutates *graph* in place. Returns the number of rewrites applied.
    """
    total_rewrites = 0
    max_iterations = 10
    for _ in range(max_iterations):
        rewrites = 0
        for op in list(graph.ops):
            for pattern in DECOMPOSE_PATTERNS:
                data = pattern.match(op)
                if data is not None:
                    pattern.rewrite(op, data, graph)
                    rewrites += 1
                    break
        total_rewrites += rewrites
        graph.dce()
        if rewrites == 0:
            break
    else:
        raise RuntimeError(
            f"decompose: failed to converge after {max_iterations} iterations "
            f"({total_rewrites} rewrites applied). This indicates a cycle in "
            f"decomposition patterns."
        )
    return total_rewrites
