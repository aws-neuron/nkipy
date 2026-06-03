"""Tests for direct_lower_reduce.

Verifies P-dim reduction (both GpSimd and matmul strategies) and F-dim
reduction by running the numpy interpreter then executing on real Trainium
hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir.ir import Builder as TensorBuilder, run as tensor_run
from nkigen_lite.tensor_ir.passes.layout_solver import solve_graph
from nkigen_lite.nki_ir import run as nki_run
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

from nkigen_lite.tensor_ir.passes.basic.direct_lower_reduce import (
    lower_f_reduce,
    lower_p_reduce_gpsimd,
    lower_p_reduce_matmul,
    lower_reduce,
)

try:
    import nki.compiler.kernel_builder as nb_kb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False

pytestmark = pytest.mark.hw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(nki_graph, graph, inputs, atol=1e-5):
    """Verify NKI IR graph: interpreter gate then real HW execution."""
    if not HAS_NKI:
        pytest.skip("nki not installed — HW execution required, no simulator")

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


def _build_and_lower(build_fn, inputs, atol=1e-5):
    """Build graph, lower via unified lower_reduce, verify."""
    b = TensorBuilder("t")
    build_fn(b)
    graph = b.graph
    layouts = solve_graph(graph)
    nki_graph = lower_reduce(graph, layouts)
    _check(nki_graph, graph, inputs, atol)


def _build_and_lower_f(build_fn, inputs, atol=1e-5):
    """Build graph with F-reduce, lower, verify."""
    b = TensorBuilder("t")
    build_fn(b)
    graph = b.graph
    layouts = solve_graph(graph)
    nki_graph = lower_f_reduce(graph, layouts)
    _check(nki_graph, graph, inputs, atol)


def _build_and_lower_p_gpsimd(build_fn, inputs, atol=1e-5):
    """Build graph with P-reduce, lower via gpsimd, verify."""
    b = TensorBuilder("t")
    build_fn(b)
    graph = b.graph
    layouts = solve_graph(graph)
    nki_graph = lower_p_reduce_gpsimd(graph, layouts)
    _check(nki_graph, graph, inputs, atol)


def _build_and_lower_p_matmul(build_fn, inputs, atol=1e-4):
    """Build graph with P-reduce, lower via matmul trick, verify."""
    b = TensorBuilder("t")
    build_fn(b)
    graph = b.graph
    layouts = solve_graph(graph)
    nki_graph = lower_p_reduce_matmul(graph, layouts)
    _check(nki_graph, graph, inputs, atol)


# ---------------------------------------------------------------------------
# F-dim reduction tests
# ---------------------------------------------------------------------------


class TestFReduceAllF:
    """Reduce all F-dims (the common case: e.g. sum over last axis)."""

    @pytest.mark.parametrize("kind", ["sum", "max", "min"])
    def test_basic_kinds(self, kind):
        rng = np.random.default_rng(42)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y = b.reduce(x, axis=-1, kind=kind, keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_mean(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (64, 128), DType.F32)
            y = b.reduce(x, axis=-1, kind="mean", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((64, 128)).astype(np.float32),
        })

    @pytest.mark.parametrize("shape", [
        (1, 64),
        (128, 1),
        (128, 512),
        (300, 100),
    ])
    def test_shapes(self, shape):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", shape, DType.F32)
            y = b.reduce(x, axis=-1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal(shape).astype(np.float32),
        })

    def test_rank3_reduce_last(self):
        rng = np.random.default_rng(2)

        def build(b):
            x = b.add_input("x", (4, 128, 64), DType.F32)
            y = b.reduce(x, axis=-1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((4, 128, 64)).astype(np.float32),
        })

    def test_rank4_reduce_last(self):
        rng = np.random.default_rng(3)

        def build(b):
            x = b.add_input("x", (2, 4, 64, 32), DType.F32)
            y = b.reduce(x, axis=-1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((2, 4, 64, 32)).astype(np.float32),
        })


class TestFReducePartialF:
    """Reduce only the last N of multiple F-dims."""

    def test_reduce_last_of_two_f_dims(self):
        """Shape (4, 8, 32): layout I=(0,), P=(1,), F=(2,) -> reduce axis=2."""
        rng = np.random.default_rng(10)

        def build(b):
            # (4, 8, 32) with layout I=(0,), P=(1,), F=(2,)
            # Reduce the single F-dim
            x = b.add_input("x", (4, 8, 32), DType.F32)
            y = b.reduce(x, axis=2, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((4, 8, 32)).astype(np.float32),
        })

    def test_reduce_last_two_f_dims(self):
        """Shape (2, 128, 8, 32): layout P=(0,1), F=(2,3) -> reduce axis=(2,3)."""
        rng = np.random.default_rng(11)

        def build(b):
            x = b.add_input("x", (2, 128, 8, 32), DType.F32)
            y = b.reduce(x, axis=(2, 3), kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((2, 128, 8, 32)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# P-dim reduction tests — GpSimd strategy
# ---------------------------------------------------------------------------


class TestPReduceGpsimd:
    """P-dim reduction via cross_lane_reduce_arith."""

    @pytest.mark.parametrize("kind", ["sum", "max", "min"])
    def test_basic_kinds(self, kind):
        rng = np.random.default_rng(42)

        def build(b):
            # (128, 256): P=(0,), F=(1,) -> reduce axis=0
            x = b.add_input("x", (128, 256), DType.F32)
            y = b.reduce(x, axis=0, kind=kind, keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_mean(self):
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (64, 128), DType.F32)
            y = b.reduce(x, axis=0, kind="mean", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((64, 128)).astype(np.float32),
        })

    @pytest.mark.parametrize("shape", [
        (1, 64),
        (64, 128),
        (128, 256),
        (128, 512),
    ])
    def test_shapes(self, shape):
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", shape, DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal(shape).astype(np.float32),
        })

    def test_rank3_batch(self):
        """(4, 64, 128): I=(0,), P=(1,), F=(2,) -> reduce axis=1."""
        rng = np.random.default_rng(2)

        def build(b):
            x = b.add_input("x", (4, 64, 128), DType.F32)
            y = b.reduce(x, axis=1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((4, 64, 128)).astype(np.float32),
        })

    def test_large_p_sum(self):
        """P > 128: tiles and combines partial reductions (sum)."""
        rng = np.random.default_rng(3)

        def build(b):
            x = b.add_input("x", (256, 64), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((256, 64)).astype(np.float32),
        })

    def test_large_p_max(self):
        """P > 128: tiles and combines partial reductions (max)."""
        rng = np.random.default_rng(4)

        def build(b):
            x = b.add_input("x", (300, 100), DType.F32)
            y = b.reduce(x, axis=0, kind="max", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((300, 100)).astype(np.float32),
        })

    def test_large_p_min(self):
        """P > 128: tiles and combines partial reductions (min)."""
        rng = np.random.default_rng(5)

        def build(b):
            x = b.add_input("x", (200, 50), DType.F32)
            y = b.reduce(x, axis=0, kind="min", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((200, 50)).astype(np.float32),
        })

    def test_large_p_mean(self):
        """P > 128: tiles partial sums then divides (mean)."""
        rng = np.random.default_rng(6)

        def build(b):
            x = b.add_input("x", (256, 128), DType.F32)
            y = b.reduce(x, axis=0, kind="mean", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((256, 128)).astype(np.float32),
        })

    def test_large_p_remainder(self):
        """P=300: 2 full tiles of 128 + 1 remainder tile of 44."""
        rng = np.random.default_rng(7)

        def build(b):
            x = b.add_input("x", (300, 64), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((300, 64)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# P-dim reduction tests — matmul trick (ones.T @ x)
# ---------------------------------------------------------------------------


class TestPReduceMatmul:
    """P-dim reduction via matmul trick: works for any P extent."""

    def test_basic_sum(self):
        rng = np.random.default_rng(42)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_matmul(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_large_p(self):
        """P > 128: requires multiple matmul tiles with accumulation."""
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (256, 64), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_matmul(build, {
            "x": rng.standard_normal((256, 64)).astype(np.float32),
        })

    def test_large_p_remainder(self):
        """P=300: 2 full tiles of 128 + 1 remainder tile of 44."""
        rng = np.random.default_rng(1)

        def build(b):
            x = b.add_input("x", (300, 100), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_matmul(build, {
            "x": rng.standard_normal((300, 100)).astype(np.float32),
        })

    def test_rank3_batch(self):
        """(4, 128, 64): I=(0,), P=(1,), F=(2,) -> reduce axis=1."""
        rng = np.random.default_rng(2)

        def build(b):
            x = b.add_input("x", (4, 128, 64), DType.F32)
            y = b.reduce(x, axis=1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_matmul(build, {
            "x": rng.standard_normal((4, 128, 64)).astype(np.float32),
        })

    def test_rank3_large_p(self):
        """(2, 256, 128): batched, P>128."""
        rng = np.random.default_rng(3)

        def build(b):
            x = b.add_input("x", (2, 256, 128), DType.F32)
            y = b.reduce(x, axis=1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_matmul(build, {
            "x": rng.standard_normal((2, 256, 128)).astype(np.float32),
        })

    def test_mean(self):
        """Matmul trick supports mean (sum + divide)."""
        rng = np.random.default_rng(4)

        def build(b):
            x = b.add_input("x", (256, 64), DType.F32)
            y = b.reduce(x, axis=0, kind="mean", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_matmul(build, {
            "x": rng.standard_normal((256, 64)).astype(np.float32),
        })

    def test_non_sum_mean_rejected(self):
        """Matmul trick only supports sum/mean, not max/min."""
        def build(b):
            x = b.add_input("x", (128, 64), DType.F32)
            y = b.reduce(x, axis=0, kind="max", keepdims=True)
            b.set_outputs({"y": y})

        b = TensorBuilder("t")
        build(b)
        layouts = solve_graph(b.graph)
        with pytest.raises(ValueError, match="sum/mean"):
            lower_p_reduce_matmul(b.graph, layouts)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_element_f_reduce(self):
        """F-dim of size 1 — reduce is a no-op but should still work."""
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (128, 1), DType.F32)
            y = b.reduce(x, axis=-1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((128, 1)).astype(np.float32),
        })

    def test_single_element_p_reduce(self):
        """P-dim of size 1 — reduce is a no-op but should still work."""
        rng = np.random.default_rng(0)

        def build(b):
            x = b.add_input("x", (1, 256), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_p_gpsimd(build, {
            "x": rng.standard_normal((1, 256)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Non-suffix F-reduce
# ---------------------------------------------------------------------------


class TestFReduceNonSuffix:
    """Reduce a prefix or middle F-dim (not the trailing suffix)."""

    def test_reduce_first_of_two_f_dims(self):
        """F=(2,3), reduce axis=2 only (non-suffix)."""
        rng = np.random.default_rng(20)

        def build(b):
            x = b.add_input("x", (2, 128, 8, 32), DType.F32)
            y = b.reduce(x, axis=2, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower_f(build, {
            "x": rng.standard_normal((2, 128, 8, 32)).astype(np.float32),
        })


# ---------------------------------------------------------------------------
# Unified lower_reduce (handles all cases)
# ---------------------------------------------------------------------------


class TestUnifiedReduce:
    """Tests for the unified lower_reduce entry point."""

    @pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
    def test_all_dims_rank2(self, kind):
        """Reduce all dims of a rank-2 tensor (mixed P/F)."""
        rng = np.random.default_rng(30)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y = b.reduce(x, axis=(0, 1), kind=kind, keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        }, atol=1e-3)

    def test_mixed_pf_rank3(self):
        """Mixed P/F reduce on rank-3."""
        rng = np.random.default_rng(31)

        def build(b):
            x = b.add_input("x", (4, 128, 64), DType.F32)
            y = b.reduce(x, axis=(0, 2), kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((4, 128, 64)).astype(np.float32),
        })

    def test_mixed_all_dims_rank3(self):
        """Reduce all dims of rank-3."""
        rng = np.random.default_rng(32)

        def build(b):
            x = b.add_input("x", (4, 128, 64), DType.F32)
            y = b.reduce(x, axis=(0, 1, 2), kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((4, 128, 64)).astype(np.float32),
        })

    def test_mixed_large_p(self):
        """Mixed P/F with P > 128."""
        rng = np.random.default_rng(33)

        def build(b):
            x = b.add_input("x", (300, 100), DType.F32)
            y = b.reduce(x, axis=(0, 1), kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((300, 100)).astype(np.float32),
        })

    def test_mixed_rank4(self):
        """Mixed P/F on rank-4 tensor."""
        rng = np.random.default_rng(34)

        def build(b):
            x = b.add_input("x", (2, 128, 8, 32), DType.F32)
            y = b.reduce(x, axis=(0, 2, 3), kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((2, 128, 8, 32)).astype(np.float32),
        })

    def test_pure_f_delegated(self):
        """Pure F-reduce goes through lower_reduce correctly."""
        rng = np.random.default_rng(35)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y = b.reduce(x, axis=-1, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })

    def test_pure_p_delegated(self):
        """Pure P-reduce goes through lower_reduce correctly."""
        rng = np.random.default_rng(36)

        def build(b):
            x = b.add_input("x", (128, 256), DType.F32)
            y = b.reduce(x, axis=0, kind="sum", keepdims=True)
            b.set_outputs({"y": y})

        _build_and_lower(build, {
            "x": rng.standard_normal((128, 256)).astype(np.float32),
        })


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-q"])
