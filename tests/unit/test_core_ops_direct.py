# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that call core ops directly (not through numpy interception).

Covers: Op registry, CPU backends for creation ops, HLO edge cases,
and transform error paths.
"""

import numpy as np
import pytest
from nkipy.core.ops import creation
from nkipy.core.ops._registry import Op
from utils import trace_and_compile


# ---------------------------------------------------------------------------
# Op registry tests
# ---------------------------------------------------------------------------
class TestOpRegistry:
    def test_op_call_unregistered_backend(self):
        """Calling an Op with no matching backend raises NotImplementedError."""
        op = Op("dummy_op")

        @op.impl("hlo")
        def _hlo_impl():
            return "hlo"

        # Outside of tracing, backend is "cpu"; no cpu impl registered.
        with pytest.raises(
            NotImplementedError, match="not implemented for backend 'cpu'"
        ):
            op()

    def test_op_repr_with_backends(self):
        """Op.__repr__ lists registered backends."""
        op = Op("my_op")

        @op.impl("hlo")
        def _hlo(x):
            pass

        @op.impl("cpu")
        def _cpu(x):
            pass

        r = repr(op)
        assert "my_op" in r
        assert "hlo" in r
        assert "cpu" in r

    def test_op_repr_empty(self):
        """Op.__repr__ shows 'none' when no backends are registered."""
        op = Op("empty_op")
        assert "none" in repr(op)


# ---------------------------------------------------------------------------
# CPU backend tests for creation ops
# ---------------------------------------------------------------------------
class TestCreationCPU:
    def test_zeros_cpu(self):
        result = creation.zeros((4, 8), np.float32)
        assert result.shape == (4, 8)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.zeros((4, 8), dtype=np.float32))

    def test_zeros_like_cpu(self):
        x = np.ones((3, 5), dtype=np.float64)
        result = creation.zeros_like(x)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_ones_like_cpu(self):
        x = np.zeros((2, 4), dtype=np.int32)
        result = creation.ones_like(x)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
        np.testing.assert_array_equal(result, np.ones_like(x))

    def test_empty_like_cpu(self):
        x = np.ones((6,), dtype=np.float32)
        result = creation.empty_like(x)
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_full_like_cpu(self):
        x = np.zeros((3, 3), dtype=np.float32)
        result = creation.full_like(x, 7.5)
        assert result.shape == x.shape
        np.testing.assert_array_equal(result, np.full_like(x, 7.5))

    def test_full_cpu(self):
        result = creation.full((2, 3), 42.0, np.float32)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.full((2, 3), 42.0, dtype=np.float32))

    def test_constant_cpu(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = creation.constant(arr)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# HLO edge cases for creation ops
# ---------------------------------------------------------------------------
class TestCreationHLO:
    def test_constant_hlo_passthrough(self, trace_mode):
        """constant with NKIPyTensorRef is idempotent."""

        def kernel(x):
            from nkipy.core.ops.creation import constant

            return constant(x)

        in0 = np.ones((16, 16), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_constant_hlo_scalar_types(self, trace_mode):
        """constant with int/float/bool scalars."""

        def kernel(x):
            from nkipy.core.ops.creation import constant

            c = constant(1.0)
            return x + c

        in0 = np.ones((16, 16), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_constant_hlo_list_tuple(self, trace_mode):
        """constant with a list/tuple promotes to tensor."""

        def kernel(x):
            from nkipy.core.ops.creation import constant

            c = constant([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            # Reshape to broadcast-compatible shape and add
            return x + c

        in0 = np.ones((16, 4), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_zeros_hlo_direct(self, trace_mode):
        """zeros(shape, dtype) inside kernel exercises the HLO path."""

        def kernel(x):
            from nkipy.core.ops.creation import zeros

            z = zeros((32, 32), np.float32)
            return x + z

        in0 = np.ones((32, 32), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_zeros_hlo_int_shape(self, trace_mode):
        """zeros with int (not tuple) shape normalizes to tuple."""

        def kernel(x):
            from nkipy.core.ops.creation import zeros

            z = zeros(32, np.float32)
            return x + z

        in0 = np.ones((32,), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_ones_like_hlo(self, trace_mode):
        """ones_like(x) inside kernel (not registered in numpy dispatch)."""

        def kernel(x):
            from nkipy.core.ops.creation import ones_like

            return ones_like(x)

        in0 = np.ones((16, 16), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_ones_like_hlo_dtype_override(self, trace_mode):
        """ones_like(x, dtype=float16) on float32 input hits convert branch."""

        def kernel(x):
            from nkipy.core.ops.creation import ones_like

            return ones_like(x, dtype=np.float16)

        in0 = np.ones((16, 16), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_full_hlo_direct(self, trace_mode):
        """full(shape, fill_value, dtype) inside kernel exercises HLO path."""

        def kernel(x):
            from nkipy.core.ops.creation import full

            f = full((32, 32), 3.14, np.float32)
            return x + f

        in0 = np.ones((32, 32), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_full_hlo_int_shape(self, trace_mode):
        """full with int shape normalizes to tuple."""

        def kernel(x):
            from nkipy.core.ops.creation import full

            f = full(32, 1.0, np.float32)
            return x + f

        in0 = np.ones((32,), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_constant_hlo_dtype_cast(self, trace_mode):
        """constant(x, dtype=float16) where x is float32 NKIPyTensorRef."""

        def kernel(x):
            from nkipy.core.ops.creation import constant

            return constant(x, dtype=np.float16)

        in0 = np.ones((16, 16), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)

    def test_constant_hlo_int_scalar(self, trace_mode):
        """constant(42) — int scalar dtype detection."""

        def kernel(x):
            from nkipy.core.ops.creation import constant
            from nkipy.core.ops.transform import astype

            c = constant(42)
            c_f32 = astype(c, np.float32)
            return x + c_f32

        in0 = np.ones((16, 16), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)


# ---------------------------------------------------------------------------
# Indexing HLO tests
# ---------------------------------------------------------------------------
class TestIndexingHLO:
    def test_where_scalar_y(self, trace_mode):
        """np.where(cond, x, 0.0) — scalar y promotion."""

        def kernel(cond, x):
            return np.where(cond, x, 0.0)

        cond = np.array([[True, False], [False, True]], dtype=np.bool_)
        x = np.ones((2, 2), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, cond, x)

    def test_where_integer_condition(self, trace_mode):
        """np.where(cond_int32, x, y) — int condition converted to bool."""

        def kernel(x, y):
            from nkipy.core.ops.creation import constant

            cond = np.array([[1, 0], [0, 2]], dtype=np.int32)
            cond_t = constant(cond)
            return np.where(cond_t, x, y)

        in0 = np.ones((2, 2), dtype=np.float32)
        in1 = np.zeros((2, 2), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0, in1)

    def test_where_broadcast_x(self, trace_mode):
        """np.where(cond, x_small, y) where x needs broadcasting."""

        def kernel(cond, x, y):
            return np.where(cond, x, y)

        cond = np.array([[True, False, True, False]], dtype=np.bool_)
        x_small = np.ones((1, 4), dtype=np.float32)
        y = np.zeros((2, 4), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, cond, x_small, y)


# ---------------------------------------------------------------------------
# Conv error tests
# ---------------------------------------------------------------------------
class TestConvErrors:
    def test_normalize_tuple_2d_error(self):
        """_normalize_tuple_2d with wrong length raises ValueError."""
        from nkipy.core.ops.conv import _normalize_tuple_2d

        with pytest.raises(ValueError, match="must be an int or a tuple of 2 ints"):
            _normalize_tuple_2d((1, 2, 3), "test")

    def test_normalize_tuple_3d_error(self):
        """_normalize_tuple_3d with invalid type raises ValueError."""
        from nkipy.core.ops.conv import _normalize_tuple_3d

        with pytest.raises(ValueError, match="must be an int or a tuple of 3 ints"):
            _normalize_tuple_3d("invalid", "test")

    def test_conv2d_cpu_dilation_error(self):
        """conv2d on CPU with dilation != 1 raises NotImplementedError."""
        from nkipy.core.ops.conv import conv2d

        inp = np.random.rand(1, 1, 8, 8).astype(np.float32)
        w = np.random.rand(1, 1, 3, 3).astype(np.float32)
        with pytest.raises(NotImplementedError, match="dilation"):
            conv2d(inp, w, dilation=(2, 2))

    def test_conv2d_cpu_groups_error(self):
        """conv2d on CPU with groups != 1 raises NotImplementedError."""
        from nkipy.core.ops.conv import conv2d

        inp = np.random.rand(1, 2, 8, 8).astype(np.float32)
        w = np.random.rand(2, 1, 3, 3).astype(np.float32)
        with pytest.raises(NotImplementedError, match="groups"):
            conv2d(inp, w, groups=2)

    def test_conv3d_cpu_dilation_error(self):
        """conv3d on CPU with dilation != 1 raises NotImplementedError."""
        from nkipy.core.ops.conv import conv3d

        inp = np.random.rand(1, 1, 8, 8, 8).astype(np.float32)
        w = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
        with pytest.raises(NotImplementedError, match="dilation"):
            conv3d(inp, w, dilation=(2, 2, 2))

    def test_conv3d_cpu_groups_error(self):
        """conv3d on CPU with groups != 1 raises NotImplementedError."""
        from nkipy.core.ops.conv import conv3d

        inp = np.random.rand(1, 2, 8, 8, 8).astype(np.float32)
        w = np.random.rand(2, 1, 3, 3, 3).astype(np.float32)
        with pytest.raises(NotImplementedError, match="groups"):
            conv3d(inp, w, groups=2)


# ---------------------------------------------------------------------------
# Reduce error tests
# ---------------------------------------------------------------------------
class TestReduceErrors:
    def test_reduce_unsupported_op(self, trace_mode):
        """_build_reduction_hlo with unsupported op raises NotImplementedError."""

        def kernel(x):
            from nkipy.core.ops.reduce import _build_reduction_hlo

            return _build_reduction_hlo(x, np.prod)

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(NotImplementedError, match="not yet supported"):
            trace_and_compile(kernel, trace_mode, in0)


# ---------------------------------------------------------------------------
# Linalg HLO tests
# ---------------------------------------------------------------------------
class TestLinalgHLO:
    def test_matmul_vector_dot_product(self, trace_mode):
        """np.matmul(v1, v2) with two 1D vectors — dot product path."""

        def kernel(a, b):
            return np.matmul(a, b)

        in0 = np.random.rand(64).astype(np.float32)
        in1 = np.random.rand(64).astype(np.float32)
        trace_and_compile(kernel, trace_mode, in0, in1)


# ---------------------------------------------------------------------------
# Transform HLO tests
# ---------------------------------------------------------------------------
class TestTransformHLO:
    def test_reshape_int_newshape(self, trace_mode):
        """reshape with int newshape normalizes to tuple."""

        def kernel(x):
            return np.reshape(x, 64)

        in0 = np.ones((8, 8), dtype=np.float32)
        trace_and_compile(kernel, trace_mode, in0)


# ---------------------------------------------------------------------------
# Transform error path tests
# ---------------------------------------------------------------------------
class TestTransformErrors:
    def test_reshape_invalid_size(self, trace_mode):
        """reshape with mismatched total size raises ValueError."""

        def kernel(x):
            return np.reshape(x, (5, 5))

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises((ValueError, AssertionError)):
            trace_and_compile(kernel, trace_mode, in0)

    def test_expand_dims_duplicate_axes(self, trace_mode):
        """expand_dims with duplicate axes raises ValueError."""

        def kernel(x):
            return np.expand_dims(x, axis=[0, 0])

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="repeated axis"):
            trace_and_compile(kernel, trace_mode, in0)

    def test_expand_dims_out_of_bounds(self, trace_mode):
        """expand_dims with axis out of bounds raises ValueError."""

        def kernel(x):
            return np.expand_dims(x, axis=10)

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            trace_and_compile(kernel, trace_mode, in0)

    def test_concatenate_empty(self, trace_mode):
        """concatenate with empty tensor list raises ValueError."""
        from nkipy.core.ops.transform import concatenate as concat_op

        def kernel(x):
            concat_op([])
            return x

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Need at least one tensor"):
            trace_and_compile(kernel, trace_mode, in0)

    def test_split_zero_sections(self, trace_mode):
        """split with sections=0 raises ValueError."""

        def kernel(x):
            return np.split(x, 0, axis=0)

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="larger than 0"):
            trace_and_compile(kernel, trace_mode, in0)

    def test_split_unequal(self, trace_mode):
        """split with unequal division raises ValueError."""

        def kernel(x):
            return np.split(x, 3, axis=0)

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="equal division"):
            trace_and_compile(kernel, trace_mode, in0)

    def test_topk_non_last_axis(self, trace_mode):
        """topk on non-last axis raises NotImplementedError in HLO."""
        from nkipy.core import tensor_apis

        def kernel(x):
            return tensor_apis.topk(x, k=2, axis=0)

        in0 = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(NotImplementedError, match="only supports last axis"):
            trace_and_compile(kernel, trace_mode, in0)

    def test_concatenate_axis_out_of_bounds(self, trace_mode):
        """concatenate with axis out of bounds raises ValueError."""

        def kernel(a, b):
            return np.concatenate([a, b], axis=5)

        in0 = np.ones((4, 4), dtype=np.float32)
        in1 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            trace_and_compile(kernel, trace_mode, in0, in1)

    def test_split_axis_out_of_bounds(self, trace_mode):
        """split with axis out of bounds raises ValueError."""

        def kernel(x):
            return np.split(x, 2, axis=5)

        in0 = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            trace_and_compile(kernel, trace_mode, in0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
