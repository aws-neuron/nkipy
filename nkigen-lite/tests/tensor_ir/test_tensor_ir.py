"""Tests for tensor_ir: types, values, ops, builder, graph infra, and interpreter."""

import numpy as np
import pytest
from scipy.special import softmax as scipy_softmax

from nkigen_lite.tensor_ir import (
    Builder, DType, Graph, Op, TensorType, Value, ValueCounter, run,
)
from nkigen_lite.tensor_ir.examples import softmax, layer_norm


# ===========================
# TensorType
# ===========================

class TestTensorType:
    def test_shape_and_dtype(self):
        t = TensorType((2, 3), DType.F32)
        assert t.shape == (2, 3)
        assert t.dtype == DType.F32

    def test_rank(self):
        assert TensorType((4, 8, 16), DType.F16).rank == 3
        assert TensorType((), DType.I32).rank == 0

    def test_str(self):
        assert str(TensorType((2, 3), DType.F32)) == "<2x3xf32>"
        assert str(TensorType((), DType.I32)) == "<i32>"

    def test_frozen(self):
        t1 = TensorType((4,), DType.F16)
        t2 = TensorType((4,), DType.F16)
        assert t1 == t2
        assert hash(t1) == hash(t2)


# ===========================
# ValueCounter
# ===========================

class TestValueCounter:
    def test_fresh_names(self):
        c = ValueCounter()
        assert c.fresh() == "v1"
        assert c.fresh() == "v2"

    def test_independent_counters(self):
        c1 = ValueCounter()
        c2 = ValueCounter()
        c1.fresh()
        c1.fresh()
        assert c2.fresh() == "v1"  # independent


# ===========================
# Value
# ===========================

class TestValue:
    def test_repr_and_str(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        assert repr(v) == "%x"
        assert str(v) == "%x: <4xf32>"

    def test_uses_empty_initially(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        assert not v.has_uses
        assert v.uses == []

    def test_uses_populated_by_op(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        op = Op("neg", [v], [v.type])
        assert v.has_uses
        assert op in v.uses

    def test_multi_consumer_uses(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        op1 = Op("neg", [v], [v.type])
        op2 = Op("exp", [v], [v.type])
        assert len(v.uses) == 2
        assert op1 in v.uses
        assert op2 in v.uses

    def test_replace_all_uses_with(self):
        v_old = Value(name="x", type=TensorType((4,), DType.F32))
        v_new = Value(name="y", type=TensorType((4,), DType.F32))
        op = Op("neg", [v_old], [v_old.type])
        v_old.replace_all_uses_with(v_new)
        assert not v_old.has_uses
        assert v_new.has_uses
        assert op.inputs[0] is v_new

    def test_uses_snapshot_is_copy(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        Op("neg", [v], [v.type])
        snapshot = v.uses
        snapshot.clear()
        assert v.has_uses  # internal list unchanged


# ===========================
# Op
# ===========================

class TestOp:
    def test_single_result(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        op = Op("neg", [v], [v.type])
        assert op.result is op.results[0]
        assert op.result.producer is op

    def test_multiple_results(self):
        v = Value(name="x", type=TensorType((8,), DType.F32))
        rt1 = TensorType((4,), DType.F32)
        rt2 = TensorType((4,), DType.F32)
        op = Op("split", [v], [rt1, rt2])
        assert len(op.results) == 2
        with pytest.raises(AssertionError):
            _ = op.result  # should fail for multi-result

    def test_shared_counter(self):
        c = ValueCounter()
        v = Value(name="x", type=TensorType((4,), DType.F32))
        op1 = Op("neg", [v], [v.type], counter=c)
        op2 = Op("exp", [v], [v.type], counter=c)
        assert op1.result.name == "v1"
        assert op2.result.name == "v2"

    def test_str(self):
        v = Value(name="x", type=TensorType((4,), DType.F32))
        op = Op("neg", [v], [v.type])
        s = str(op)
        assert "neg" in s
        assert "%x" in s


# ===========================
# Graph (per-graph counters)
# ===========================

class TestGraphCounter:
    def test_per_graph_numbering(self):
        b1 = Builder("g1")
        x = b1.add_input("x", (4,), DType.F32)
        b1.neg(x)

        b2 = Builder("g2")
        y = b2.add_input("y", (4,), DType.F32)
        r = b2.neg(y)
        # Second graph starts at v1, not continuing from first
        assert r.name == "v1"

    def test_dump_round_trip(self):
        b = Builder("test")
        x = b.add_input("x", (2, 3), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})
        dump = b.graph.dump()
        assert "@test" in dump
        assert "neg" in dump
        assert "return y=" in dump


# ===========================
# Builder — elementwise ops
# ===========================

class TestBuilderUnary:
    @pytest.fixture
    def bx(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        return b, x

    @pytest.mark.parametrize("op_name", [
        "neg", "exp", "log", "sqrt", "rsqrt", "tanh",
        "relu", "gelu", "sigmoid", "sin", "cos",
    ])
    def test_unary_shape_preserved(self, bx, op_name):
        b, x = bx
        result = getattr(b, op_name)(x)
        assert result.type == x.type

    def test_cast(self, bx):
        b, x = bx
        y = b.cast(x, DType.F16)
        assert y.type.dtype == DType.F16
        assert y.type.shape == x.type.shape


class TestBuilderBinary:
    @pytest.fixture
    def bxy(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        y = b.add_input("y", (2, 3), DType.F32)
        return b, x, y

    @pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div", "maximum", "minimum"])
    def test_binary_shape_preserved(self, bxy, op_name):
        b, x, y = bxy
        result = getattr(b, op_name)(x, y)
        assert result.type == x.type

    def test_binary_not_broadcastable(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        y = b.add_input("y", (2, 4), DType.F32)
        with pytest.raises(ValueError, match="not broadcastable"):
            b.add(x, y)

    def test_binary_dtype_mismatch(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        y = b.add_input("y", (2, 3), DType.I32)
        with pytest.raises(ValueError, match="dtype mismatch"):
            b.add(x, y)

    def test_binary_broadcast_keepdims(self):
        """(2, 3) + (2, 1) -> (2, 3) — the most common broadcast pattern."""
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        y = b.add_input("y", (2, 1), DType.F32)
        r = b.add(x, y)
        assert r.type.shape == (2, 3)

    def test_binary_broadcast_rank_extension(self):
        """(2, 3) * (3,) -> (2, 3) — weight broadcast."""
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        w = b.add_input("w", (3,), DType.F32)
        r = b.mul(x, w)
        assert r.type.shape == (2, 3)

    def test_binary_broadcast_scalar(self):
        """(4, 8) + (1,) -> (4, 8) — scalar broadcast."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        s = b.constant(1.0, (1,), DType.F32)
        r = b.add(x, s)
        assert r.type.shape == (4, 8)

    def test_binary_broadcast_both_expand(self):
        """(1, 3) + (2, 1) -> (2, 3) — both inputs expand."""
        b = Builder()
        a = b.add_input("a", (1, 3), DType.F32)
        c = b.add_input("c", (2, 1), DType.F32)
        r = b.add(a, c)
        assert r.type.shape == (2, 3)


class TestBuilderComparison:
    @pytest.mark.parametrize("op_name", [
        "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    ])
    def test_comparison_returns_bool(self, op_name):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        result = getattr(b, op_name)(x, y)
        assert result.type.dtype == DType.BOOL
        assert result.type.shape == (4,)

    def test_comparison_not_broadcastable(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (5,), DType.F32)
        with pytest.raises(ValueError, match="not broadcastable"):
            b.equal(x, y)

    def test_comparison_broadcast(self):
        """(3, 1) > (1, 4) -> (3, 4) — e.g. causal mask construction."""
        b = Builder()
        row = b.add_input("row", (3, 1), DType.F32)
        col = b.add_input("col", (1, 4), DType.F32)
        r = b.greater(row, col)
        assert r.type.shape == (3, 4)
        assert r.type.dtype == DType.BOOL


class TestBuilderWhere:
    def test_where(self):
        b = Builder()
        c = b.add_input("c", (4,), DType.BOOL)
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        r = b.where(c, x, y)
        assert r.type == x.type

    def test_where_not_broadcastable(self):
        b = Builder()
        c = b.add_input("c", (3,), DType.BOOL)
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        with pytest.raises(ValueError, match="not broadcastable"):
            b.where(c, x, y)

    def test_where_broadcast(self):
        """where(cond:(4,1), a:(4,3), b:(1,3)) -> (4, 3)."""
        b = Builder()
        c = b.add_input("c", (4, 1), DType.BOOL)
        x = b.add_input("x", (4, 3), DType.F32)
        y = b.add_input("y", (1, 3), DType.F32)
        r = b.where(c, x, y)
        assert r.type.shape == (4, 3)

    def test_where_non_bool_cond(self):
        b = Builder()
        c = b.add_input("c", (4,), DType.F32)
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        with pytest.raises(ValueError, match="cond must be bool"):
            b.where(c, x, y)


# ===========================
# Builder — constants
# ===========================

class TestBuilderConstants:
    def test_constant(self):
        b = Builder()
        c = b.constant(3.14, (2, 2), DType.F32)
        assert c.type == TensorType((2, 2), DType.F32)
        assert c.producer.attrs["value"] == 3.14

    def test_zeros(self):
        b = Builder()
        z = b.zeros((4,), DType.F32)
        assert z.producer.attrs["value"] == 0.0

    def test_full(self):
        b = Builder()
        f = b.full((3,), 7.0, DType.F16)
        assert f.type.dtype == DType.F16
        assert f.producer.attrs["value"] == 7.0


# ===========================
# Builder — reductions
# ===========================

class TestBuilderReduce:
    @pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
    def test_reduce_no_keepdims(self, kind):
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        r = b.reduce(x, axis=1, kind=kind)
        assert r.type.shape == (4,)

    @pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
    def test_reduce_keepdims(self, kind):
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        r = b.reduce(x, axis=1, kind=kind, keepdims=True)
        assert r.type.shape == (4, 1)

    def test_reduce_negative_axis(self):
        b = Builder()
        x = b.add_input("x", (4, 8, 16), DType.F32)
        r = b.reduce(x, axis=-1, keepdims=True, kind="sum")
        assert r.type.shape == (4, 8, 1)

    def test_reduce_multi_axis(self):
        b = Builder()
        x = b.add_input("x", (2, 3, 4), DType.F32)
        r = b.reduce(x, axis=(0, 2), kind="sum")
        assert r.type.shape == (3,)

    def test_reduce_invalid_axis(self):
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        with pytest.raises(ValueError, match="out of range"):
            b.reduce(x, axis=5, kind="sum")

    def test_reduce_positive_out_of_range(self):
        """axis=2 on rank-2 should raise, not silently wrap."""
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        with pytest.raises(ValueError, match="out of range"):
            b.reduce(x, axis=2, kind="sum")

    def test_reduce_scalar_raises(self):
        b = Builder()
        x = b.add_input("x", (), DType.F32)
        with pytest.raises(ValueError, match="rank 0"):
            b.reduce(x, axis=0, kind="sum")


# ===========================
# Builder — shape ops
# ===========================

class TestBuilderShape:
    def test_transpose(self):
        b = Builder()
        x = b.add_input("x", (2, 3, 4), DType.F32)
        r = b.transpose(x, (2, 0, 1))
        assert r.type.shape == (4, 2, 3)

    def test_transpose_negative_perm(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        r = b.transpose(x, (-1, -2))
        assert r.type.shape == (3, 2)

    def test_transpose_invalid_perm(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        with pytest.raises(ValueError, match="invalid perm"):
            b.transpose(x, (0, 0))

    def test_transpose_out_of_range(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        with pytest.raises(ValueError, match="out of range"):
            b.transpose(x, (0, 5))

    def test_reshape(self):
        b = Builder()
        x = b.add_input("x", (2, 6), DType.F32)
        r = b.reshape(x, (3, 4))
        assert r.type.shape == (3, 4)

    def test_reshape_size_mismatch(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        with pytest.raises(ValueError, match="size mismatch"):
            b.reshape(x, (2, 4))

    def test_broadcast_to(self):
        b = Builder()
        x = b.add_input("x", (1, 4), DType.F32)
        r = b.broadcast_to(x, (3, 4))
        assert r.type.shape == (3, 4)

    def test_broadcast_to_higher_rank(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        r = b.broadcast_to(x, (2, 3, 4))
        assert r.type.shape == (2, 3, 4)

    def test_broadcast_to_invalid(self):
        b = Builder()
        x = b.add_input("x", (3,), DType.F32)
        with pytest.raises(ValueError, match="not broadcastable"):
            b.broadcast_to(x, (4,))

    def test_expand_dims(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        r = b.expand_dims(x, axis=1)
        assert r.type.shape == (2, 1, 3)

    def test_squeeze(self):
        b = Builder()
        x = b.add_input("x", (2, 1, 3), DType.F32)
        r = b.squeeze(x, axis=1)
        assert r.type.shape == (2, 3)

    def test_squeeze_invalid(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        with pytest.raises(ValueError, match="expected 1"):
            b.squeeze(x, axis=1)

    def test_slice(self):
        b = Builder()
        x = b.add_input("x", (10, 20), DType.F32)
        r = b.slice(x, starts=(2, 4), stops=(8, 16))
        assert r.type.shape == (6, 12)

    def test_slice_with_strides(self):
        b = Builder()
        x = b.add_input("x", (10,), DType.F32)
        r = b.slice(x, starts=(0,), stops=(10,), strides=(2,))
        assert r.type.shape == (5,)

    def test_split_even(self):
        b = Builder()
        x = b.add_input("x", (12,), DType.F32)
        parts = b.split(x, 3, axis=0)
        assert len(parts) == 3
        for p in parts:
            assert p.type.shape == (4,)

    def test_split_sizes(self):
        b = Builder()
        x = b.add_input("x", (10,), DType.F32)
        a, b_val = b.split(x, [3, 7], axis=0)
        assert a.type.shape == (3,)
        assert b_val.type.shape == (7,)

    def test_concat(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        y = b.add_input("y", (2, 5), DType.F32)
        r = b.concat([x, y], axis=1)
        assert r.type.shape == (2, 8)

    def test_concat_requires_two_inputs(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        with pytest.raises(ValueError, match="at least 2"):
            b.concat([x], axis=0)


# ===========================
# Builder — matmul
# ===========================

class TestBuilderMatmul:
    def test_2d_matmul(self):
        b = Builder()
        a = b.add_input("a", (4, 8), DType.F32)
        w = b.add_input("w", (8, 16), DType.F32)
        r = b.matmul(a, w)
        assert r.type.shape == (4, 16)

    def test_batched_matmul(self):
        b = Builder()
        a = b.add_input("a", (2, 4, 8), DType.F32)
        w = b.add_input("w", (2, 8, 16), DType.F32)
        r = b.matmul(a, w)
        assert r.type.shape == (2, 4, 16)

    def test_contraction_dim_mismatch(self):
        b = Builder()
        a = b.add_input("a", (4, 8), DType.F32)
        w = b.add_input("w", (9, 16), DType.F32)
        with pytest.raises(TypeError, match="contraction dim"):
            b.matmul(a, w)

    def test_matmul_dtype_mismatch(self):
        b = Builder()
        a = b.add_input("a", (4, 8), DType.F32)
        w = b.add_input("w", (8, 16), DType.F16)
        with pytest.raises(TypeError, match="dtype mismatch"):
            b.matmul(a, w)

    def test_matmul_batch_mismatch(self):
        b = Builder()
        a = b.add_input("a", (2, 4, 8), DType.F32)
        w = b.add_input("w", (3, 8, 16), DType.F32)
        with pytest.raises(TypeError, match="batch shapes.*not broadcastable"):
            b.matmul(a, w)


# ===========================
# Builder — composites
# ===========================

class TestBuilderComposites:
    def test_softmax_shape(self):
        b = Builder()
        x = b.add_input("x", (2, 8), DType.F32)
        r = softmax(b, x, axis=-1)
        assert r.type == x.type

    def test_softmax_decomposes(self):
        b = Builder()
        x = b.add_input("x", (2, 8), DType.F32)
        softmax(b, x, axis=-1)
        opcodes = [op.opcode for op in b.graph.ops]
        assert "reduce" in opcodes
        assert "exp" in opcodes
        assert "div" in opcodes
        # No explicit broadcast_to — binary ops broadcast implicitly
        assert "broadcast_to" not in opcodes

    def test_layer_norm_shape(self):
        b = Builder()
        x = b.add_input("x", (2, 8), DType.F32)
        w = b.add_input("w", (8,), DType.F32)
        bias = b.add_input("bias", (8,), DType.F32)
        r = layer_norm(b, x, w, bias, axis=-1)
        assert r.type == x.type


# ===========================
# Builder — control flow
# ===========================

class TestBuilderForLoop:
    def test_for_loop_single_carry(self):
        b = Builder()
        init = b.constant(0.0, (4,), DType.F32)

        def body(lb, _i, acc):
            one = lb.constant(1.0, (4,), DType.F32)
            return lb.add(acc, one)

        (result,) = b.for_loop(trip_count=10, init=[init], body_fn=body)
        assert result.type == TensorType((4,), DType.F32)

    def test_for_loop_multi_carry(self):
        b = Builder()
        a = b.constant(0.0, (2,), DType.F32)
        c = b.constant(1.0, (3,), DType.F32)

        def body(lb, _i, x, y):
            return lb.neg(x), lb.neg(y)

        r1, r2 = b.for_loop(trip_count=5, init=[a, c], body_fn=body)
        assert r1.type.shape == (2,)
        assert r2.type.shape == (3,)


# ===========================
# Graph — use-lists
# ===========================

class TestUseLists:
    def test_add_input_no_uses(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        assert not x.has_uses

    def test_single_use(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.neg(x)
        assert len(x.uses) == 1

    def test_multi_use(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        a = b.neg(x)
        c = b.add(x, a)
        # x used by neg and add
        assert len(x.uses) == 2

    def test_rauw_updates_uses(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        r = b.add(x, y)
        b.set_outputs({"r": r})

        # Replace x with y everywhere
        b.graph.replace_value(x, y)
        assert not x.has_uses
        assert r.producer.inputs == [y, y]
        # Graph output unchanged (it was r, not x)

    def test_rauw_updates_graph_outputs(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        z = b.exp(x)
        b.set_outputs({"out": y})

        b.graph.replace_value(y, z)
        assert b.graph.outputs["out"] is z


# ===========================
# Graph — mutation helpers
# ===========================

class TestGraphMutation:
    def test_insert_before(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})

        new_op = Op("exp", [x], [x.type], counter=b.graph.counter)
        b.graph.insert_before(y.producer, new_op)
        assert b.graph.ops.index(new_op) < b.graph.ops.index(y.producer)

    def test_insert_after(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})

        new_op = Op("exp", [x], [x.type], counter=b.graph.counter)
        b.graph.insert_after(y.producer, new_op)
        assert b.graph.ops.index(new_op) > b.graph.ops.index(y.producer)

    def test_erase_op(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        z = b.exp(y)
        b.set_outputs({"z": z})

        # Can't erase y's producer — z uses y
        with pytest.raises(ValueError, match="still has"):
            b.graph.erase_op(y.producer)

        # Replace z's input, then erase
        b.graph.replace_value(y, x)
        b.graph.erase_op(y.producer)
        assert y.producer not in b.graph.ops

    def test_erase_op_asserts_use_list_consistency(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        z = b.exp(y)
        b.set_outputs({"z": z})

        # Corrupt the use-list manually
        b.graph.replace_value(y, x)
        y.producer.inputs = []  # bypass use-list
        # x._uses still references neg_op but neg_op no longer has x as input
        # This shouldn't matter for erase since neg_op.inputs is now empty
        # The assert is about op being in its *own* inputs' use-lists
        # After clearing inputs, erase should work since there's nothing to unhook
        # But we also cleared the inputs list, so there's nothing to remove from
        b.graph.erase_op(y.producer)
        assert y.producer not in b.graph.ops


# ===========================
# Graph — DCE
# ===========================

class TestDCE:
    def test_dce_removes_dead_ops(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)  # dead — not an output
        z = b.exp(x)
        b.set_outputs({"z": z})

        removed = b.graph.dce()
        assert removed == 1
        assert y.producer not in b.graph.ops
        assert z.producer in b.graph.ops

    def test_dce_chain(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        a = b.neg(x)
        b_val = b.exp(a)
        c = b.log(b_val)  # entire chain is dead
        live = b.relu(x)
        b.set_outputs({"live": live})

        removed = b.graph.dce()
        assert removed == 3
        assert len(b.graph.ops) == 1  # only relu

    def test_dce_nothing_to_remove(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})
        assert b.graph.dce() == 0

    def test_dce_preserves_verify(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.neg(x)
        b.exp(x)
        y = b.relu(x)
        b.set_outputs({"y": y})
        b.graph.dce()
        assert b.graph.verify() == []


# ===========================
# Graph — toposort
# ===========================

class TestToposort:
    def test_toposort_maintains_order(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        z = b.exp(y)
        b.set_outputs({"z": z})
        b.graph.toposort()

        idx_neg = b.graph.ops.index(y.producer)
        idx_exp = b.graph.ops.index(z.producer)
        assert idx_neg < idx_exp

    def test_toposort_fixes_misordered_ops(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        z = b.exp(y)
        b.set_outputs({"z": z})

        # Manually reverse the ops
        b.graph.ops.reverse()
        assert b.graph.ops[0] is z.producer  # exp before neg — wrong

        b.graph.toposort()
        idx_neg = b.graph.ops.index(y.producer)
        idx_exp = b.graph.ops.index(z.producer)
        assert idx_neg < idx_exp

    def test_toposort_after_rewrite(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})

        # Insert a new op that should come before neg's consumer
        new_op = Op("relu", [x], [x.type], counter=b.graph.counter)
        b.graph.insert_after(y.producer, new_op)  # after neg
        b.graph.replace_value(y, new_op.result)
        b.graph.dce()
        b.graph.toposort()
        assert b.graph.verify() == []


# ===========================
# Graph — verify
# ===========================

class TestVerify:
    def test_clean_graph_passes(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})
        assert b.graph.verify() == []

    def test_detects_use_before_def(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        z = b.exp(y)
        b.set_outputs({"z": z})

        # Swap ops so exp comes before neg
        b.graph.ops.reverse()
        errors = b.graph.verify()
        assert any("used before definition" in e for e in errors)

    def test_detects_undefined_output(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})

        # Remove the only op — output now references undefined value
        # First clear uses so erase_op allows it
        b.graph.replace_value(y, x)
        b.graph.erase_op(y.producer)
        # Manually set output back to the now-orphaned y
        b.graph.outputs["y"] = y
        errors = b.graph.verify()
        assert any("undefined value" in e for e in errors)

    def test_detects_use_list_inconsistency(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.neg(x)
        b.set_outputs({"y": y})

        # Corrupt use-list
        x._uses.clear()
        errors = b.graph.verify()
        assert any("use-list inconsistent" in e for e in errors)


# ===========================
# Interpreter — elementwise
# ===========================

class TestInterpreterElementwise:
    def test_neg(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.neg(x)})
        outs = run(b.graph, {"x": np.array([1, -2, 3, -4], dtype=np.float32)})
        np.testing.assert_allclose(outs["y"], [-1, 2, -3, 4])

    def test_exp(self):
        b = Builder()
        x = b.add_input("x", (3,), DType.F32)
        b.set_outputs({"y": b.exp(x)})
        x_np = np.array([0, 1, 2], dtype=np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["y"], np.exp(x_np), rtol=1e-6)

    def test_relu(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.relu(x)})
        outs = run(b.graph, {"x": np.array([-1, 0, 1, 2], dtype=np.float32)})
        np.testing.assert_allclose(outs["y"], [0, 0, 1, 2])

    def test_gelu_dtype_preserved(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F16)
        b.set_outputs({"y": b.gelu(x)})
        x_np = np.array([0, 1, -1, 0.5], dtype=np.float16)
        outs = run(b.graph, {"x": x_np})
        assert outs["y"].dtype == np.float16

    def test_sigmoid(self):
        b = Builder()
        x = b.add_input("x", (3,), DType.F32)
        b.set_outputs({"y": b.sigmoid(x)})
        x_np = np.array([0, 10, -10], dtype=np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["y"], 1.0 / (1.0 + np.exp(-x_np)), rtol=1e-6)

    def test_rsqrt(self):
        b = Builder()
        x = b.add_input("x", (3,), DType.F32)
        b.set_outputs({"y": b.rsqrt(x)})
        x_np = np.array([1, 4, 9], dtype=np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["y"], 1.0 / np.sqrt(x_np), rtol=1e-6)

    def test_maximum_minimum(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        y = b.add_input("y", (4,), DType.F32)
        b.set_outputs({"max": b.maximum(x, y), "min": b.minimum(x, y)})
        x_np = np.array([1, 5, 3, 7], dtype=np.float32)
        y_np = np.array([4, 2, 6, 0], dtype=np.float32)
        outs = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(outs["max"], [4, 5, 6, 7])
        np.testing.assert_allclose(outs["min"], [1, 2, 3, 0])

    def test_add_sub_mul_div(self):
        b = Builder()
        x = b.add_input("x", (3,), DType.F32)
        y = b.add_input("y", (3,), DType.F32)
        b.set_outputs({
            "add": b.add(x, y),
            "sub": b.sub(x, y),
            "mul": b.mul(x, y),
            "div": b.div(x, y),
        })
        x_np = np.array([6, 8, 10], dtype=np.float32)
        y_np = np.array([2, 4, 5], dtype=np.float32)
        outs = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_allclose(outs["add"], [8, 12, 15])
        np.testing.assert_allclose(outs["sub"], [4, 4, 5])
        np.testing.assert_allclose(outs["mul"], [12, 32, 50])
        np.testing.assert_allclose(outs["div"], [3, 2, 2])

    def test_comparison_ops(self):
        b = Builder()
        x = b.add_input("x", (3,), DType.F32)
        y = b.add_input("y", (3,), DType.F32)
        b.set_outputs({"gt": b.greater(x, y), "eq": b.equal(x, y)})
        x_np = np.array([1, 2, 3], dtype=np.float32)
        y_np = np.array([3, 2, 1], dtype=np.float32)
        outs = run(b.graph, {"x": x_np, "y": y_np})
        np.testing.assert_array_equal(outs["gt"], [False, False, True])
        np.testing.assert_array_equal(outs["eq"], [False, True, False])

    def test_where(self):
        b = Builder()
        c = b.add_input("c", (3,), DType.BOOL)
        x = b.add_input("x", (3,), DType.F32)
        y = b.add_input("y", (3,), DType.F32)
        b.set_outputs({"r": b.where(c, x, y)})
        outs = run(b.graph, {
            "c": np.array([True, False, True]),
            "x": np.array([10, 20, 30], dtype=np.float32),
            "y": np.array([1, 2, 3], dtype=np.float32),
        })
        np.testing.assert_allclose(outs["r"], [10, 2, 30])

    def test_add_broadcast_keepdims(self):
        """(2, 3) + (2, 1) broadcasts — the common reduce+broadcast pattern."""
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        bias = b.add_input("bias", (2, 1), DType.F32)
        b.set_outputs({"r": b.add(x, bias)})
        x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        bias_np = np.array([[10], [20]], dtype=np.float32)
        outs = run(b.graph, {"x": x_np, "bias": bias_np})
        np.testing.assert_allclose(outs["r"], x_np + bias_np)

    def test_mul_broadcast_rank_extension(self):
        """(2, 3) * (3,) broadcasts — weight vector applied to batched data."""
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        w = b.add_input("w", (3,), DType.F32)
        b.set_outputs({"r": b.mul(x, w)})
        x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        w_np = np.array([10, 100, 1000], dtype=np.float32)
        outs = run(b.graph, {"x": x_np, "w": w_np})
        np.testing.assert_allclose(outs["r"], x_np * w_np)

    def test_comparison_broadcast(self):
        """(3, 1) > (1, 4) broadcasts — causal mask pattern."""
        b = Builder()
        row = b.add_input("row", (3, 1), DType.F32)
        col = b.add_input("col", (1, 4), DType.F32)
        b.set_outputs({"r": b.greater(row, col)})
        row_np = np.array([[0], [1], [2]], dtype=np.float32)
        col_np = np.array([[0, 1, 2, 3]], dtype=np.float32)
        outs = run(b.graph, {"row": row_np, "col": col_np})
        np.testing.assert_array_equal(outs["r"], row_np > col_np)

    def test_where_broadcast(self):
        """where with broadcast shapes."""
        b = Builder()
        c = b.add_input("c", (3, 1), DType.BOOL)
        x = b.add_input("x", (1, 4), DType.F32)
        y = b.add_input("y", (3, 4), DType.F32)
        b.set_outputs({"r": b.where(c, x, y)})
        c_np = np.array([[True], [False], [True]])
        x_np = np.array([[10, 20, 30, 40]], dtype=np.float32)
        y_np = np.ones((3, 4), dtype=np.float32)
        outs = run(b.graph, {"c": c_np, "x": x_np, "y": y_np})
        np.testing.assert_allclose(outs["r"], np.where(c_np, x_np, y_np))


# ===========================
# Interpreter — constants and cast
# ===========================

class TestInterpreterConstants:
    def test_constant(self):
        b = Builder()
        b.set_outputs({"c": b.constant(42.0, (2, 3), DType.F32)})
        outs = run(b.graph, {})
        assert outs["c"].shape == (2, 3)
        np.testing.assert_allclose(outs["c"], 42.0)

    def test_cast(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.cast(x, DType.I32)})
        outs = run(b.graph, {"x": np.array([1.7, 2.3, -0.5, 0.0], dtype=np.float32)})
        assert outs["y"].dtype == np.int32
        np.testing.assert_array_equal(outs["y"], [1, 2, 0, 0])


# ===========================
# Interpreter — reductions
# ===========================

class TestInterpreterReduce:
    def test_reduce_sum(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=1, kind="sum")})
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["r"], x_np.sum(axis=1))

    def test_reduce_mean_keepdims(self):
        b = Builder()
        x = b.add_input("x", (4, 8), DType.F32)
        b.set_outputs({"r": b.reduce(x, axis=-1, keepdims=True, kind="mean")})
        x_np = np.random.randn(4, 8).astype(np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["r"], x_np.mean(axis=1, keepdims=True), rtol=1e-5)


# ===========================
# Interpreter — shape ops
# ===========================

class TestInterpreterShape:
    def test_transpose(self):
        b = Builder()
        x = b.add_input("x", (2, 3), DType.F32)
        b.set_outputs({"r": b.transpose(x, (1, 0))})
        x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["r"], x_np.T)

    def test_reshape(self):
        b = Builder()
        x = b.add_input("x", (2, 6), DType.F32)
        b.set_outputs({"r": b.reshape(x, (3, 4))})
        x_np = np.arange(12, dtype=np.float32).reshape(2, 6)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["r"], x_np.reshape(3, 4))

    def test_broadcast_to(self):
        b = Builder()
        x = b.add_input("x", (1, 4), DType.F32)
        b.set_outputs({"r": b.broadcast_to(x, (3, 4))})
        x_np = np.array([[1, 2, 3, 4]], dtype=np.float32)
        outs = run(b.graph, {"x": x_np})
        assert outs["r"].shape == (3, 4)
        np.testing.assert_allclose(outs["r"][0], outs["r"][2])

    def test_slice(self):
        b = Builder()
        x = b.add_input("x", (10,), DType.F32)
        b.set_outputs({"r": b.slice(x, starts=(2,), stops=(7,))})
        x_np = np.arange(10, dtype=np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["r"], x_np[2:7])

    def test_split_and_concat(self):
        b = Builder()
        x = b.add_input("x", (6,), DType.F32)
        a, c = b.split(x, 2, axis=0)
        r = b.concat([c, a], axis=0)  # swap halves
        b.set_outputs({"r": r})
        x_np = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["r"], [4, 5, 6, 1, 2, 3])

    def test_matmul(self):
        b = Builder()
        a = b.add_input("a", (2, 3), DType.F32)
        w = b.add_input("w", (3, 4), DType.F32)
        b.set_outputs({"r": b.matmul(a, w)})
        a_np = np.random.randn(2, 3).astype(np.float32)
        w_np = np.random.randn(3, 4).astype(np.float32)
        outs = run(b.graph, {"a": a_np, "w": w_np})
        np.testing.assert_allclose(outs["r"], a_np @ w_np, rtol=1e-5)


# ===========================
# Interpreter — composites
# ===========================

class TestInterpreterComposites:
    def test_softmax(self):
        b = Builder()
        x = b.add_input("x", (2, 8), DType.F32)
        b.set_outputs({"p": softmax(b, x, axis=-1)})
        x_np = np.random.randn(2, 8).astype(np.float32)
        outs = run(b.graph, {"x": x_np})
        np.testing.assert_allclose(outs["p"], scipy_softmax(x_np, axis=-1), rtol=1e-5)
        np.testing.assert_allclose(outs["p"].sum(axis=1), [1.0, 1.0], rtol=1e-5)

    def test_layer_norm(self):
        b = Builder()
        x = b.add_input("x", (2, 8), DType.F32)
        w = b.add_input("w", (8,), DType.F32)
        bias = b.add_input("bias", (8,), DType.F32)
        b.set_outputs({"y": layer_norm(b, x, w, bias, axis=-1)})
        x_np = np.random.randn(2, 8).astype(np.float32)
        outs = run(b.graph, {
            "x": x_np,
            "w": np.ones(8, dtype=np.float32),
            "bias": np.zeros(8, dtype=np.float32),
        })
        # With w=1, bias=0: output should have mean~0, std~1 per row
        y = outs["y"]
        np.testing.assert_allclose(y.mean(axis=1), [0, 0], atol=1e-5)
        np.testing.assert_allclose(y.std(axis=1), [1, 1], atol=0.05)


# ===========================
# Interpreter — control flow
# ===========================

class TestInterpreterForLoop:
    def test_accumulate(self):
        b = Builder()
        init = b.constant(0.0, (2,), DType.F32)

        def body(lb, _i, acc):
            one = lb.constant(1.0, (2,), DType.F32)
            return lb.add(acc, one)

        (result,) = b.for_loop(trip_count=50, init=[init], body_fn=body)
        b.set_outputs({"r": result})
        outs = run(b.graph, {})
        np.testing.assert_allclose(outs["r"], [50, 50])


# ===========================
# Interpreter — error handling
# ===========================

class TestInterpreterErrors:
    def test_missing_input(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.neg(x)})
        with pytest.raises(ValueError, match="Missing input"):
            run(b.graph, {})

    def test_shape_mismatch(self):
        b = Builder()
        x = b.add_input("x", (4,), DType.F32)
        b.set_outputs({"y": b.neg(x)})
        with pytest.raises(ValueError, match="Shape mismatch"):
            run(b.graph, {"x": np.zeros((3,), dtype=np.float32)})

    def test_no_outputs(self):
        b = Builder()
        b.add_input("x", (4,), DType.F32)
        with pytest.raises(ValueError, match="no outputs"):
            run(b.graph, {"x": np.zeros((4,), dtype=np.float32)})


# ===========================
# End-to-end: rewrite + verify
# ===========================

class TestEndToEnd:
    def test_softmax_fusion_rewrite(self):
        """Match softmax pattern, replace with fused op, DCE, verify."""
        B, H, S, D = 1, 2, 4, 8
        b = Builder("attn")
        q = b.add_input("q", (B, H, S, D), DType.F32)
        k = b.add_input("k", (B, H, S, D), DType.F32)
        v = b.add_input("v", (B, H, S, D), DType.F32)
        kt = b.transpose(k, (0, 1, 3, 2))
        scores = b.matmul(q, kt)
        scale = b.constant(1.0 / (D ** 0.5), scores.type.shape, DType.F32)
        scores_s = b.mul(scores, scale)
        probs = softmax(b, scores_s, axis=-1)
        out = b.matmul(probs, v)
        b.set_outputs({"r": out})
        g = b.graph

        assert g.verify() == []
        ops_before = len(g.ops)

        # Find and replace the div (tail of softmax)
        div_op = next(op for op in g.ops if op.opcode == "div")
        sub_op = div_op.inputs[0].producer.inputs[0].producer
        x_input = sub_op.inputs[0]

        fused = Op("softmax", [x_input], [div_op.result.type],
                    {"axis": (3,)}, counter=g.counter)
        g.insert_before(div_op, fused)
        g.replace_value(div_op.result, fused.result)
        removed = g.dce()

        assert removed == 5  # reduce(max), sub, exp, reduce(sum), div
        assert len(g.ops) == ops_before - 5 + 1
        assert any(op.opcode == "softmax" for op in g.ops)
        assert not any(op.opcode == "div" for op in g.ops)

        g.toposort()
        assert g.verify() == []

    def test_rmsnorm_example(self):
        """Build and run RMSNorm, verify against numpy."""
        b = Builder("rmsnorm")
        x = b.add_input("x", (2, 8), DType.F32)
        w = b.add_input("w", (8,), DType.F32)
        xsq = b.mul(x, x)
        mean_sq = b.reduce(xsq, axis=-1, keepdims=True, kind="mean")
        eps = b.constant(1e-5, mean_sq.type.shape, DType.F32)
        rstd = b.rsqrt(b.add(mean_sq, eps))
        normed = b.mul(x, rstd)    # (2,8) * (2,1) broadcasts
        out = b.mul(normed, w)       # (2,8) * (8,) broadcasts
        b.set_outputs({"r": out})

        assert b.graph.verify() == []

        np.random.seed(0)
        x_np = np.random.randn(2, 8).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)
        outs = run(b.graph, {"x": x_np, "w": w_np})

        # Reference
        xsq_ref = x_np ** 2
        rstd_ref = 1.0 / np.sqrt(xsq_ref.mean(axis=-1, keepdims=True) + 1e-5)
        expected = x_np * rstd_ref * w_np
        np.testing.assert_allclose(outs["r"], expected, rtol=1e-5)
