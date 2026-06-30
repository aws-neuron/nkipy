"""Tests for direct_lower_elementwise.

Verifies that the direct elementwise lowering produces correct NKI IR by
running the numpy interpreter then executing on real Trainium hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir.ir import Builder as TensorBuilder, run as tensor_run
from nkigen_lite.tensor_ir.passes.layout_solver import solve_graph
from nkigen_lite.nki_ir import run as nki_run
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower_elementwise import lower_elementwise

try:
    import nki.compiler.kernel_builder as nb_kb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

pytestmark = pytest.mark.hw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower_and_check(build_fn, inputs, atol=1e-5):
    """Build graph, lower via direct elementwise, verify interpreter then HW."""
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required, no simulator")

    b = TensorBuilder("t")
    build_fn(b)
    graph = b.graph

    layouts = solve_graph(graph)
    nki_graph = lower_elementwise(graph, layouts)

    ref = tensor_run(graph, inputs)

    # Interpreter gate
    nki_inputs = dict(inputs)
    for out_name, out_val in graph.outputs.items():
        nki_inputs[f"{out_name}_out"] = np.zeros(out_val.type.shape, dtype=np.float32)
    interp = nki_run(nki_graph, nki_inputs)
    for k in ref:
        np.testing.assert_allclose(
            interp[k], ref[k], atol=atol, rtol=atol,
            err_msg=f"Interpreter mismatch on {k!r} (must pass before HW)",
        )

    # Real hardware execution
    opts = nb_kb.CompileOptions(target="trn2")
    kernel_fn = build_kb_kernel(nki_graph)
    hw_inputs = dict(inputs)
    hw_outputs = {
        f"{out_name}_out": np.zeros(out_val.type.shape, dtype=np.float32)
        for out_name, out_val in graph.outputs.items()
    }
    nb_kb.compile_and_execute(
        kernel_fn, inputs=hw_inputs, outputs=hw_outputs, compile_opts=opts,
    )
    for k in ref:
        np.testing.assert_allclose(
            hw_outputs[f"{k}_out"], ref[k], atol=atol, rtol=atol,
            err_msg=f"HW mismatch on {k!r}",
        )


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


class TestUnaryOps:
    @pytest.mark.parametrize("opcode", [
        "neg", "exp", "log", "sqrt", "rsqrt", "tanh",
        "relu", "gelu", "sigmoid", "silu", "reciprocal",
    ])
    def test_unary_basic(self, opcode):
        rng = np.random.default_rng(42)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y = getattr(b, opcode)(x)
            b.set_outputs({"y": y})

        x = rng.uniform(0.1, 2.0, (128, 256)).astype(np.float32)
        _lower_and_check(build, {"x": x}, atol=1e-3)

    def test_unary_chain(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (64, 128), DType.F32)
            y = b.tanh(b.sigmoid(x))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((64, 128)).astype(np.float32),
        })

    def test_neg(self):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", (128, 64), DType.F32)
            y = b.neg(x)
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((128, 64)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------


class TestBinaryOps:
    @pytest.mark.parametrize("opcode", ["add", "sub", "mul", "maximum", "minimum"])
    def test_binary_basic(self, opcode):
        rng = np.random.default_rng(42)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y_in = b.add_input("y_in", (128, 256), DType.F32)
            z = getattr(b, opcode)(x, y_in)
            b.set_outputs({"z": z})

        _lower_and_check(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
            "y_in": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_add_relu(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            bias = b.add_input("bias", (128, 256), DType.F32)
            y = b.relu(b.add(x, bias))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
            "bias": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_mul_exp(self):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", (64, 64), DType.F32)
            s = b.add_input("s", (64, 64), DType.F32)
            y = b.exp(b.mul(x, s))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((64, 64)).astype(np.float32) * 0.1,
            "s": rng.standard_normal((64, 64)).astype(np.float32) * 0.1,
        })

    def test_silu_chain(self):
        rng = np.random.default_rng(2)

        def build(b):
            x = b.add_input("x", (256, 512), DType.F32)
            scale = b.add_input("scale", (256, 512), DType.F32)
            bias = b.add_input("bias", (256, 512), DType.F32)
            y = b.silu(b.add(b.mul(x, scale), bias))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((256, 512)).astype(np.float32),
            "scale": rng.standard_normal((256, 512)).astype(np.float32),
            "bias": rng.standard_normal((256, 512)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Shape coverage (tiling boundary conditions)
# ---------------------------------------------------------------------------


class TestShapeCoverage:
    @pytest.mark.parametrize("shape", [
        (1, 1),           # single element
        (1, 700),         # single partition, large free
        (128, 1),         # full partition, single free
        (128, 256),       # single P-tile
        (129, 33),        # P-remainder
        (300, 700),       # multiple P-tiles + remainder
        (7, 13, 5),       # rank-3
        (5, 200, 97),     # rank-3 with P-tiling
        (4, 128, 256),    # rank-3, batch + full tile
        (2, 3, 64, 50),   # rank-4
        (1, 1, 1, 512),   # rank-4, all-I except F
    ])
    def test_add_relu_shapes(self, shape):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", shape, DType.F32)
            bias = b.add_input("bias", shape, DType.F32)
            y = b.relu(b.add(x, bias))
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal(shape).astype(np.float32),
            "bias": rng.standard_normal(shape).astype(np.float32),
        })

    @pytest.mark.parametrize("shape", [
        (64, 64),
        (256, 128),
        (512, 1024),
    ])
    def test_unary_shapes(self, shape):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", shape, DType.F32)
            y = b.gelu(x)
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal(shape).astype(np.float32),
        }, atol=1e-3)


# ---------------------------------------------------------------------------
# Constant ops
# ---------------------------------------------------------------------------


class TestConstant:
    def test_constant_add(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            c = b.constant(2.0, (128, 256), DType.F32)
            y = b.mul(x, c)
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_constant_chain(self):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", (64, 128), DType.F32)
            c1 = b.constant(0.5, (64, 128), DType.F32)
            c2 = b.constant(1.0, (64, 128), DType.F32)
            y = b.add(b.mul(x, c1), c2)
            b.set_outputs({"y": y})

        _lower_and_check(build, {
            "x": rng.standard_normal((64, 128)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Multi-output graphs
# ---------------------------------------------------------------------------


class TestMultiOutput:
    def test_two_outputs(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y1 = b.relu(x)
            y2 = b.sigmoid(x)
            b.set_outputs({"y1": y1, "y2": y2})

        _lower_and_check(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_shared_intermediate(self):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", (64, 128), DType.F32)
            intermediate = b.tanh(x)
            y1 = b.relu(intermediate)
            y2 = b.neg(intermediate)
            b.set_outputs({"y1": y1, "y2": y2})

        _lower_and_check(build, {
            "x": rng.standard_normal((64, 128)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Unsupported op rejection
# ---------------------------------------------------------------------------


class TestUnsupported:
    def test_matmul_rejected(self):
        def build(b):
            a = b.add_input("a", (128, 256), DType.F32)
            w = b.add_input("w", (256, 128), DType.F32)
            b.set_outputs({"y": b.matmul(a, w)})

        b = TensorBuilder("t")
        build(b)
        layouts = solve_graph(b.graph)
        with pytest.raises(NotImplementedError, match="matmul"):
            lower_elementwise(b.graph, layouts)

    def test_reduce_rejected(self):
        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            b.set_outputs({"y": b.reduce(x, axis=-1, kind="sum", keepdims=True)})

        b = TensorBuilder("t")
        build(b)
        layouts = solve_graph(b.graph)
        with pytest.raises(NotImplementedError, match="reduce"):
            lower_elementwise(b.graph, layouts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-q"])
