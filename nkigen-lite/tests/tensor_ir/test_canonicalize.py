"""Tests for canonicalize: recompose primitive ops into high-level activations."""

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir import Builder, run
from nkigen_lite.tensor_ir.passes.canonicalize import canonicalize
from nkigen_lite.tensor_ir.passes.decompose import decompose


# ===========================
# Helpers
# ===========================

def _run_and_compare(graph, inputs, rtol=1e-5, atol=1e-6):
    """Run graph before and after canonicalization, assert outputs match."""
    expected = run(graph, inputs)
    n = canonicalize(graph)
    actual = run(graph, inputs)
    assert set(expected.keys()) == set(actual.keys())
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=atol,
                                   err_msg=f"output {name!r} mismatch after canonicalize")
    return n


# ===========================
# RsqrtPattern: div(1, sqrt(x)) → rsqrt(x)
# ===========================

class TestRsqrtPattern:
    def test_basic(self):
        """div(1, sqrt(x)) → rsqrt(x)."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        one = b.constant(1.0, x.type.shape, DType.F32)
        b.set_outputs({"y": b.div(one, b.sqrt(x))})
        inputs = {"x": np.array([1, 4, 9, 16], dtype=np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "rsqrt" in opcodes
        assert "div" not in opcodes
        assert "sqrt" not in opcodes

    def test_sqrt_multi_use(self):
        """div(1, sqrt(x)) with sqrt used elsewhere — rsqrt created, sqrt stays alive."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        one = b.constant(1.0, x.type.shape, DType.F32)
        s = b.sqrt(x)
        b.set_outputs({"rsqrt": b.div(one, s), "sqrt": s})
        inputs = {"x": np.array([1, 4, 9, 16], dtype=np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "rsqrt" in opcodes
        assert "sqrt" in opcodes  # sqrt still alive for its other use

    def test_no_match_div_by_non_sqrt(self):
        """div(1, x) should NOT become rsqrt."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        one = b.constant(1.0, (4,), DType.F32)
        b.set_outputs({"y": b.div(one, x)})
        inputs = {"x": np.array([1, 2, 4, 8], dtype=np.float32)}
        n = _run_and_compare(b.graph, inputs)
        assert n == 0

    def test_no_match_non_one_numerator(self):
        """div(2, sqrt(x)) should NOT become rsqrt."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        two = b.constant(2.0, x.type.shape, DType.F32)
        b.set_outputs({"y": b.div(two, b.sqrt(x))})
        inputs = {"x": np.array([1, 4, 9, 16], dtype=np.float32)}
        n = _run_and_compare(b.graph, inputs)
        assert n == 0


# ===========================
# SigmoidPrimitivePattern: div(1, add(1, exp(neg(x)))) → sigmoid(x)
# ===========================

class TestSigmoidPrimitivePattern:
    def test_basic(self):
        """div(1, add(1, exp(neg(x)))) → sigmoid(x)."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        b.set_outputs({"y": b.div(one, denom)})
        inputs = {"x": np.random.randn(4, 8).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n >= 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "sigmoid" in opcodes
        assert "div" not in opcodes
        assert "exp" not in opcodes

    def test_add_reversed(self):
        """div(1, add(exp(neg(x)), 1)) — add operands reversed."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(exp_neg, one)  # reversed
        b.set_outputs({"y": b.div(one, denom)})
        inputs = {"x": np.random.randn(4).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n >= 1
        assert any(op.opcode == "sigmoid" for op in b.graph.ops)

    def test_no_match_non_one_numerator(self):
        """div(2, add(1, exp(neg(x)))) should NOT become sigmoid."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        two = b.constant(2.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        b.set_outputs({"y": b.div(two, denom)})
        inputs = {"x": np.random.randn(4).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 0
        assert not any(op.opcode == "sigmoid" for op in b.graph.ops)

    def test_silu_mul_form_not_stolen(self):
        """mul(x, div(1, 1+exp(-x))) should become silu, not sigmoid+mul."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        sigmoid = b.div(one, denom)
        b.set_outputs({"y": b.mul(x, sigmoid)})
        inputs = {"x": np.random.randn(4).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        opcodes = [op.opcode for op in b.graph.ops]
        assert "silu" in opcodes
        # sigmoid should not appear — silu matched first
        assert "sigmoid" not in opcodes


# ===========================
# SiluPrimitivePattern
# ===========================

class TestSiluPrimitivePattern:
    def _build_silu_div_form(self):
        """Build silu as x / (1 + exp(-x))."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        b.set_outputs({"y": b.div(x, denom)})
        return b

    def _build_silu_mul_form(self):
        """Build silu as x * (1 / (1 + exp(-x)))."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        sigmoid = b.div(one, denom)
        b.set_outputs({"y": b.mul(x, sigmoid)})
        return b

    def test_div_form(self):
        """x / (1 + exp(-x)) → silu(x)."""
        b = self._build_silu_div_form()
        inputs = {"x": np.random.randn(4, 8).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "silu" in opcodes
        assert "neg" not in opcodes
        assert "exp" not in opcodes

    def test_mul_form(self):
        """x * (1 / (1 + exp(-x))) → silu(x)."""
        b = self._build_silu_mul_form()
        inputs = {"x": np.random.randn(4, 8).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "silu" in opcodes
        assert "sigmoid" not in opcodes

    def test_mul_reversed_operands(self):
        """(1 / (1 + exp(-x))) * x → silu(x)."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        sigmoid = b.div(one, denom)
        b.set_outputs({"y": b.mul(sigmoid, x)})  # sigmoid first
        inputs = {"x": np.random.randn(4).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "silu" in opcodes
        assert "sigmoid" not in opcodes

    def test_add_reversed_operands(self):
        """x / (exp(-x) + 1) — add operands reversed."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(exp_neg, one)  # reversed
        b.set_outputs({"y": b.div(x, denom)})
        inputs = {"x": np.random.randn(4).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        assert any(op.opcode == "silu" for op in b.graph.ops)

    def test_intermediate_multi_use(self):
        """Sigmoid chain with exp(-x) used elsewhere — silu still canonicalizes."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        b.set_outputs({
            "silu": b.div(x, denom),
            "exp_neg": exp_neg,  # extra use of exp(-x)
        })
        inputs = {"x": np.random.randn(4).astype(np.float32)}
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "silu" in opcodes
        # neg and exp stay alive because exp_neg is a graph output
        assert "neg" in opcodes
        assert "exp" in opcodes

    def test_no_match_different_x(self):
        """div(x, 1+exp(-y)) where y != x — should NOT match."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        neg_y = b.neg(y)
        exp_neg = b.exp(neg_y)
        one = b.constant(1.0, (4,), DType.F32)
        denom = b.add(one, exp_neg)
        b.set_outputs({"r": b.div(x, denom)})
        inputs = {
            "x": np.random.randn(4).astype(np.float32),
            "y": np.random.randn(4).astype(np.float32),
        }
        n = _run_and_compare(b.graph, inputs, rtol=1e-5)
        assert n == 0


# ===========================
# Graph integrity
# ===========================

class TestGraphIntegrity:
    def test_verify_after_canonicalize(self):
        """Graph.verify() should pass after canonicalization."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        one = b.constant(1.0, (4,), DType.F32)
        b.set_outputs({"y": b.div(one, b.sqrt(x))})
        canonicalize(b.graph)
        errors = b.graph.verify()
        assert errors == [], f"Graph verification failed: {errors}"

    def test_verify_silu_after_canonicalize(self):
        """Graph.verify() should pass after SiLU canonicalization."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        neg_x = b.neg(x)
        exp_neg = b.exp(neg_x)
        one = b.constant(1.0, x.type.shape, DType.F32)
        denom = b.add(one, exp_neg)
        b.set_outputs({"y": b.div(x, denom)})
        canonicalize(b.graph)
        errors = b.graph.verify()
        assert errors == [], f"Graph verification failed: {errors}"


# ===========================
# Decompose: div(a, b) → mul(a, reciprocal(b))
# ===========================

class TestDivDecompose:
    def _run_and_compare_decompose(self, graph, inputs, rtol=1e-5, atol=1e-6):
        expected = run(graph, inputs)
        n = decompose(graph)
        actual = run(graph, inputs)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=atol,
                                       err_msg=f"output {name!r} mismatch after decompose")
        return n

    def test_basic(self):
        """div(a, b) → mul(a, reciprocal(b))."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        y = b.add_input("y", (4, 8), DType.F32)
        b.set_outputs({"r": b.div(x, y)})
        inputs = {
            "x": np.random.randn(4, 8).astype(np.float32),
            "y": np.random.uniform(0.5, 2.0, (4, 8)).astype(np.float32),
        }
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "div" not in opcodes
        assert "reciprocal" in opcodes
        assert "mul" in opcodes

    def test_broadcast(self):
        """div with broadcast: div(x[4,8], y[1,8]) → mul(x, reciprocal(y))."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        y = b.add_input("y", (1, 8), DType.F32)
        b.set_outputs({"r": b.div(x, y)})
        inputs = {
            "x": np.random.randn(4, 8).astype(np.float32),
            "y": np.random.uniform(0.5, 2.0, (1, 8)).astype(np.float32),
        }
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 1
        opcodes = [op.opcode for op in b.graph.ops]
        assert "div" not in opcodes

    def test_multiple_divs(self):
        """Multiple div ops all get decomposed."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        z = b.add_input("z", (4,), DType.F32)
        d1 = b.div(x, y)
        d2 = b.div(d1, z)
        b.set_outputs({"r": d2})
        inputs = {
            "x": np.random.randn(4).astype(np.float32),
            "y": np.random.uniform(0.5, 2.0, (4,)).astype(np.float32),
            "z": np.random.uniform(0.5, 2.0, (4,)).astype(np.float32),
        }
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 2
        opcodes = [op.opcode for op in b.graph.ops]
        assert "div" not in opcodes

    def test_canonicalize_then_decompose(self):
        """canonicalize turns div(1,sqrt(x)) into rsqrt; remaining divs get decomposed."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        one = b.constant(1.0, (4,), DType.F32)
        rsqrt_chain = b.div(one, b.sqrt(x))  # should become rsqrt
        result = b.div(rsqrt_chain, y)        # should become mul+reciprocal
        b.set_outputs({"r": result})
        inputs = {
            "x": np.random.uniform(0.5, 4.0, (4,)).astype(np.float32),
            "y": np.random.uniform(0.5, 2.0, (4,)).astype(np.float32),
        }
        expected = run(b.graph, inputs)
        canonicalize(b.graph)
        # rsqrt pattern fired, one div remains
        assert any(op.opcode == "rsqrt" for op in b.graph.ops)
        decompose(b.graph)
        actual = run(b.graph, inputs)
        np.testing.assert_allclose(actual["r"], expected["r"], rtol=1e-5)
        opcodes = [op.opcode for op in b.graph.ops]
        assert "div" not in opcodes
        assert "rsqrt" in opcodes
        assert "reciprocal" in opcodes

    def test_verify_after_decompose(self):
        """Graph.verify() should pass after decomposition."""
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        b.set_outputs({"r": b.div(x, y)})
        decompose(b.graph)
        errors = b.graph.verify()
        assert errors == [], f"Graph verification failed: {errors}"


# ===========================
# Decompose: reduce(kind="mean") → reduce(kind="sum") * (1/N)
# ===========================

class TestReduceMeanDecompose:
    def _run_and_compare_decompose(self, graph, inputs, rtol=1e-5, atol=1e-6):
        expected = run(graph, inputs)
        n = decompose(graph)
        actual = run(graph, inputs)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=atol,
                                       err_msg=f"output {name!r} mismatch after decompose")
        return n

    def test_basic(self):
        """reduce(x, kind="mean") → reduce(x, kind="sum") * (1/N)."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=1, kind="mean")})
        inputs = {"x": np.random.randn(4, 8).astype(np.float32)}
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 1
        reduce_ops = [op for op in b.graph.ops if op.opcode == "reduce"]
        assert all(op.attrs["kind"] != "mean" for op in reduce_ops)
        assert any(op.attrs["kind"] == "sum" for op in reduce_ops)
        opcodes = [op.opcode for op in b.graph.ops]
        assert "constant" in opcodes
        assert "mul" in opcodes

    def test_keepdims(self):
        """reduce with kind="mean" and keepdims=True."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=1, keepdims=True, kind="mean")})
        inputs = {"x": np.random.randn(4, 8).astype(np.float32)}
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 1
        reduce_ops = [op for op in b.graph.ops if op.opcode == "reduce"]
        assert all(op.attrs["kind"] != "mean" for op in reduce_ops)

    def test_multi_axis(self):
        """reduce(kind="mean") over multiple axes."""
        b = Builder()
        x = b.add_input("x", (2, 4, 8), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=(1, 2), kind="mean")})
        inputs = {"x": np.random.randn(2, 4, 8).astype(np.float32)}
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 1
        reduce_ops = [op for op in b.graph.ops if op.opcode == "reduce"]
        assert all(op.attrs["kind"] != "mean" for op in reduce_ops)

    def test_axis_0(self):
        """reduce_mean along axis 0."""
        b = Builder()
        x = b.add_input("x", (16, 4), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=0, kind="mean")})
        inputs = {"x": np.random.randn(16, 4).astype(np.float32)}
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 1

    def test_mixed_div_and_reduce_mean(self):
        """Both div and reduce(kind="mean") get decomposed in a single pass."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        y = b.add_input("y", (4, 1), DType.F32)
        mean = b.reduce(x, axis=1, keepdims=True, kind="mean")
        result = b.div(mean, y)
        b.set_outputs({"r": result})
        inputs = {
            "x": np.random.randn(4, 8).astype(np.float32),
            "y": np.random.uniform(0.5, 2.0, (4, 1)).astype(np.float32),
        }
        n = self._run_and_compare_decompose(b.graph, inputs)
        assert n == 2
        opcodes = [op.opcode for op in b.graph.ops]
        assert "div" not in opcodes
        reduce_ops = [op for op in b.graph.ops if op.opcode == "reduce"]
        assert all(op.attrs["kind"] != "mean" for op in reduce_ops)

    def test_verify_after_decompose(self):
        """Graph.verify() should pass after reduce_mean decomposition."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=1, keepdims=True, kind="mean")})
        decompose(b.graph)
        errors = b.graph.verify()
        assert errors == [], f"Graph verification failed: {errors}"
