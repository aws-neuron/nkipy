"""Tests for fold_transpose: compose chained ``transpose`` ops into one.

Correctness is checked two ways: the fold preserves the tensor_ir reference
(``run`` before/after), and the fully-lowered nki_ir still matches numerically
(the composed single transpose replaces the two-step chain).
"""

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir import Builder, run
from nkigen_lite.tensor_ir.passes.fold_transpose import fold_transpose, _compose
from nkigen_lite.tensor_ir.passes.lower_to_nki import lower_to_nki
from nkigen_lite.nki_ir.interpret import interpret
from nkigen_lite.nki_ir.emit_to_kb import build_kb_kernel

try:
    import nki.compiler.kernel_builder as nb
    HAS_NKI = True
except ImportError:
    HAS_NKI = False


@pytest.fixture
def compile_and_run():
    """Compile and execute the lowered graph on Trainium hardware."""
    if not HAS_NKI:
        pytest.skip("nki not installed")
    opts = nb.CompileOptions(target="trn2")

    def _run(graph, inputs, outputs):
        kernel_fn = build_kb_kernel(graph)
        nb.compile_and_execute(
            kernel_fn, inputs=inputs, outputs=outputs, compile_opts=opts,
        )
        return outputs

    return _run


def _n_transpose(graph):
    return sum(1 for op in graph.ops if op.opcode == "transpose")


def _fold_preserves(build_fn, inputs, expect_folds):
    """Assert the fold count and that tensor_ir semantics are unchanged."""
    b = Builder()
    build_fn(b)
    ref = run(b.graph, inputs)
    n = fold_transpose(b.graph)
    assert n == expect_folds, f"expected {expect_folds} folds, got {n}"
    after = run(b.graph, inputs)
    for name in ref:
        np.testing.assert_allclose(after[name], ref[name], rtol=1e-6, atol=1e-6)
    return b.graph


def _lowered_matches(build_fn, inputs, out_shapes, atol=1e-4):
    """Lower through the full pipeline (fold included) and check numerics."""
    b = Builder()
    build_fn(b)
    ref = run(b.graph, inputs)
    nki = lower_to_nki(b.graph)
    feeds = dict(inputs)
    for name, shp in out_shapes.items():
        feeds[f"{name}_out"] = np.zeros(shp, dtype=np.float32)
    env = interpret(nki, feeds)
    for name in ref:
        got = env[nki.outputs[name].name]
        np.testing.assert_allclose(got, ref[name], rtol=1e-4, atol=atol)


def _lowered_matches_hw(compile_and_run, build_fn, inputs, out_shapes, atol=1e-3):
    """Lower through the full pipeline (fold included), compile to Trainium,
    and check numerics against the tensor_ir reference."""
    b = Builder()
    build_fn(b)
    ref = run(b.graph, inputs)
    nki = lower_to_nki(b.graph)
    nki_outputs = {
        f"{name}_out": np.zeros(shp, dtype=np.float32)
        for name, shp in out_shapes.items()
    }
    hw = compile_and_run(nki, dict(inputs), nki_outputs)
    for name in ref:
        np.testing.assert_allclose(hw[f"{name}_out"], ref[name], rtol=1e-3, atol=atol)


def test_compose_helper():
    # transpose(transpose(x, p1), p2) == transpose(x, compose)
    for p1, p2 in [((0, 2, 1, 3), (0, 1, 3, 2)), ((1, 0), (1, 0)),
                   ((0, 2, 1), (2, 1, 0))]:
        x = np.random.randn(*range(2, 2 + len(p1))).astype(np.float32)
        chain = np.transpose(np.transpose(x, p1), p2)
        composed = np.transpose(x, _compose(p1, p2))
        np.testing.assert_array_equal(chain, composed)


class TestFolds:
    def test_batch_then_pf_swap(self):
        """The qwen3 attention chain: (0,2,1,3) then (0,1,3,2) on
        (1,4096,8,128) — folds to a single transpose."""
        def build(b):
            x = b.add_input("x", (1, 16, 8, 4), DType.F32)
            t1 = b.transpose(x, (0, 2, 1, 3))   # (1,8,16,4)
            b.set_outputs({"y": b.transpose(t1, (0, 1, 3, 2))})  # (1,8,4,16)
        _fold_preserves(build, {
            "x": np.random.randn(1, 16, 8, 4).astype(np.float32),
        }, expect_folds=1)
        _lowered_matches(build, {
            "x": np.random.randn(1, 16, 8, 4).astype(np.float32),
        }, {"y": (1, 8, 4, 16)})

    def test_double_pf_swap_is_identity(self):
        """Two P<->F swaps compose to the identity permutation."""
        def build(b):
            x = b.add_input("x", (8, 16), DType.F32)
            t1 = b.transpose(x, (1, 0))
            b.set_outputs({"y": b.transpose(t1, (1, 0))})
        g = _fold_preserves(build, {
            "x": np.random.randn(8, 16).astype(np.float32),
        }, expect_folds=1)
        # The surviving transpose is the identity (0, 1).
        (tp,) = [op for op in g.ops if op.opcode == "transpose"]
        assert tp.attrs["perm"] == (0, 1)
        _lowered_matches(build, {
            "x": np.random.randn(8, 16).astype(np.float32),
        }, {"y": (8, 16)})

    def test_three_chained_transposes(self):
        """A chain of three folds pairwise into a single transpose."""
        def build(b):
            x = b.add_input("x", (2, 3, 4), DType.F32)
            t1 = b.transpose(x, (1, 0, 2))
            t2 = b.transpose(t1, (0, 2, 1))
            b.set_outputs({"y": b.transpose(t2, (2, 1, 0))})
        _fold_preserves(build, {
            "x": np.random.randn(2, 3, 4).astype(np.float32),
        }, expect_folds=2)


class TestNoFold:
    def test_inner_has_other_consumer_still_folds_outer(self):
        """When the inner transpose feeds another consumer, the outer still
        composes; the inner stays live for its other use (one fold)."""
        def build(b):
            x = b.add_input("x", (2, 3, 4), DType.F32)
            t1 = b.transpose(x, (1, 0, 2))       # (3,2,4)
            outer = b.transpose(t1, (0, 2, 1))   # (3,4,2)
            b.set_outputs({"y": outer, "z": b.relu(t1)})
        g = _fold_preserves(build, {
            "x": np.random.randn(2, 3, 4).astype(np.float32),
        }, expect_folds=1)
        # Inner survives (feeds relu); composed outer reads x directly.
        assert _n_transpose(g) == 2

    def test_no_chain_no_fold(self):
        def build(b):
            x = b.add_input("x", (4, 5), DType.F32)
            b.set_outputs({"y": b.transpose(b.relu(x), (1, 0))})
        _fold_preserves(build, {
            "x": np.random.randn(4, 5).astype(np.float32),
        }, expect_folds=0)


@pytest.mark.hw
class TestFoldTransposeHW:
    """Compile the composed transpose to Trainium and check it against the
    tensor_ir reference — validates the fold's real lowering, not just interp."""

    def test_batch_then_pf_swap_hw(self, compile_and_run):
        """The qwen3 attention chain (0,2,1,3) then (0,1,3,2), folded and run
        on device."""
        def build(b):
            x = b.add_input("x", (1, 16, 8, 4), DType.F32)
            t1 = b.transpose(x, (0, 2, 1, 3))
            b.set_outputs({"y": b.transpose(t1, (0, 1, 3, 2))})
        _lowered_matches_hw(compile_and_run, build, {
            "x": np.random.randn(1, 16, 8, 4).astype(np.float32),
        }, {"y": (1, 8, 4, 16)})

    def test_double_pf_swap_is_identity_hw(self, compile_and_run):
        """Two P<->F swaps compose to identity; the folded graph is a no-op
        transpose on device."""
        def build(b):
            x = b.add_input("x", (8, 16), DType.F32)
            t1 = b.transpose(x, (1, 0))
            b.set_outputs({"y": b.transpose(t1, (1, 0))})
        _lowered_matches_hw(compile_and_run, build, {
            "x": np.random.randn(8, 16).astype(np.float32),
        }, {"y": (8, 16)})
