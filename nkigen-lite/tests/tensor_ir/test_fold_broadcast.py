"""Tests for fold_broadcast: rewire elementwise consumers of collapse-safe
``broadcast_to`` ops to their un-broadcast source and drop the broadcast.

Correctness is checked two ways: the fold preserves the tensor_ir reference
(``run`` before/after), and the fully-lowered nki_ir still matches numerically
(the elementwise emitter broadcasts the un-materialized operand on-chip).
"""

import numpy as np
import pytest

from nkigen_lite.core import DType
from nkigen_lite.tensor_ir import Builder, run
from nkigen_lite.tensor_ir.passes.fold_broadcast import fold_broadcast
from nkigen_lite.tensor_ir.passes.lower_to_nki import lower_to_nki
from nkigen_lite.nki_ir.interpret import interpret


def _n_bcast(graph):
    return sum(1 for op in graph.ops if op.opcode == "broadcast_to")


def _fold_preserves(build_fn, inputs, expect_folds):
    """Assert the fold count and that tensor_ir semantics are unchanged."""
    b = Builder()
    build_fn(b)
    ref = run(b.graph, inputs)
    before = _n_bcast(b.graph)
    n = fold_broadcast(b.graph)
    assert n == expect_folds, f"expected {expect_folds} folds, got {n}"
    assert _n_bcast(b.graph) == before - expect_folds
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


class TestFolds:
    def test_free_dim_broadcast_mul(self):
        """(1,8,1) -> (1,8,2048) feeding mul folds (native F broadcast)."""
        def build(b):
            x = b.add_input("x", (1, 8, 2048), DType.F32)
            s = b.add_input("s", (1, 8, 1), DType.F32)
            b.set_outputs({"y": b.mul(x, b.broadcast_to(s, (1, 8, 2048)))})
        _fold_preserves(build, {
            "x": np.random.randn(1, 8, 2048).astype(np.float32),
            "s": np.random.randn(1, 8, 1).astype(np.float32),
        }, expect_folds=1)

    def test_free_dim_broadcast_numeric(self):
        def build(b):
            x = b.add_input("x", (1, 8, 2048), DType.F32)
            s = b.add_input("s", (1, 8, 1), DType.F32)
            b.set_outputs({"y": b.mul(x, b.broadcast_to(s, (1, 8, 2048)))})
        _lowered_matches(build, {
            "x": np.random.randn(1, 8, 2048).astype(np.float32),
            "s": np.random.randn(1, 8, 1).astype(np.float32),
        }, {"y": (1, 8, 2048)})

    def test_div_denominator_broadcast(self):
        """div(x, broadcast_to(s)) folds so reciprocal is on the small operand."""
        def build(b):
            x = b.add_input("x", (8, 2048), DType.F32)
            s = b.add_input("s", (8, 1), DType.F32)
            b.set_outputs({"y": b.div(x, b.broadcast_to(s, (8, 2048)))})
        _fold_preserves(build, {
            "x": np.random.randn(8, 2048).astype(np.float32),
            "s": np.random.randn(8, 1).astype(np.float32) + 2.0,
        }, expect_folds=1)
        _lowered_matches(build, {
            "x": np.random.randn(8, 2048).astype(np.float32),
            "s": np.random.randn(8, 1).astype(np.float32) + 2.0,
        }, {"y": (8, 2048)})

    def test_partition_broadcast_rank2(self):
        """(1,2048) -> (8,2048) feeding add folds (partition broadcast)."""
        def build(b):
            x = b.add_input("x", (8, 2048), DType.F32)
            s = b.add_input("s", (1, 2048), DType.F32)
            b.set_outputs({"y": b.add(x, b.broadcast_to(s, (8, 2048)))})
        _fold_preserves(build, {
            "x": np.random.randn(8, 2048).astype(np.float32),
            "s": np.random.randn(1, 2048).astype(np.float32),
        }, expect_folds=1)
        _lowered_matches(build, {
            "x": np.random.randn(8, 2048).astype(np.float32),
            "s": np.random.randn(1, 2048).astype(np.float32),
        }, {"y": (8, 2048)})


class TestNoFold:
    def test_middle_broadcast_not_folded(self):
        """(1,8,1,64) -> (1,8,8,64) is not collapse-safe: stays materialized."""
        def build(b):
            x = b.add_input("x", (1, 8, 8, 64), DType.F32)
            s = b.add_input("s", (1, 8, 1, 64), DType.F32)
            b.set_outputs({"y": b.mul(x, b.broadcast_to(s, (1, 8, 8, 64)))})
        _fold_preserves(build, {
            "x": np.random.randn(1, 8, 8, 64).astype(np.float32),
            "s": np.random.randn(1, 8, 1, 64).astype(np.float32),
        }, expect_folds=0)

    def test_broadcast_as_graph_output_not_folded(self):
        def build(b):
            s = b.add_input("s", (1, 8, 1), DType.F32)
            b.set_outputs({"y": b.broadcast_to(s, (1, 8, 2048))})
        _fold_preserves(build, {
            "s": np.random.randn(1, 8, 1).astype(np.float32),
        }, expect_folds=0)

    def test_non_elementwise_consumer_not_folded(self):
        """A broadcast also read by reshape can't drop its materialized buffer."""
        def build(b):
            s = b.add_input("s", (1, 8, 1), DType.F32)
            x = b.add_input("x", (1, 8, 2048), DType.F32)
            sb = b.broadcast_to(s, (1, 8, 2048))
            b.set_outputs({"y": b.mul(x, sb), "z": b.reshape(sb, (8, 2048))})
        _fold_preserves(build, {
            "s": np.random.randn(1, 8, 1).astype(np.float32),
            "x": np.random.randn(1, 8, 2048).astype(np.float32),
        }, expect_folds=0)
