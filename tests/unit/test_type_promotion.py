# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for type promotion rules in HLO backend.
Tests the promotion table and find_common_type_hlo function.
"""

import numpy as np
import pytest
import ml_dtypes

from nkipy.core.backend.hlo import (
    _TYPE_PROMOTION_TABLE,
    _WeakInt,
    _WeakFloat,
    _get_type_key,
    _lookup_promotion,
    find_common_type_hlo,
    scalar_dtype_hlo,
)


class TestTypePromotionTable:
    """Test that the promotion table is complete and consistent."""

    def test_table_is_symmetric(self):
        """Verify that for every (a, b) -> c in table, (b, a) -> c is also valid."""
        for (t0, t1), result in _TYPE_PROMOTION_TABLE.items():
            # Check that lookup works in both directions
            assert _lookup_promotion(t0, t1) == result
            assert _lookup_promotion(t1, t0) == result

    def test_table_entries_are_concrete_types(self):
        """Verify that all table outputs are concrete numpy types, not weak types."""
        for (t0, t1), result in _TYPE_PROMOTION_TABLE.items():
            assert result is not _WeakInt, (
                f"Table entry ({t0}, {t1}) has weak int output"
            )
            assert result is not _WeakFloat, (
                f"Table entry ({t0}, {t1}) has weak float output"
            )
            # Should be a numpy type or ml_dtype
            assert hasattr(result, "__module__"), (
                f"Table entry ({t0}, {t1}) has invalid output type {result}"
            )

    def test_table_has_no_same_type_entries(self):
        """Verify that table doesn't have A+A=A entries (handled by _lookup_promotion)."""
        for t0, t1 in _TYPE_PROMOTION_TABLE.keys():
            assert t0 != t1, f"Table should not have same-type entry ({t0}, {t1})"

    def test_same_type_promotion_works(self):
        """Verify that same-type promotion (A + A = A) works via _lookup_promotion."""
        same_types = [
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            ml_dtypes.bfloat16,
            ml_dtypes.float8_e4m3,
            ml_dtypes.float8_e4m3fn,
            ml_dtypes.float8_e5m2,
        ]
        for t in same_types:
            result = _lookup_promotion(t, t)
            assert result == t, f"Same-type promotion failed for {t}: got {result}"

    @pytest.mark.parametrize(
        "t0,t1,expected",
        [
            # Sample entries from each category to verify table correctness
            # Bool promotions (same-type handled separately)
            (np.bool_, np.int32, np.int32),
            (np.bool_, np.float32, np.float32),
            # Weak type promotions (the bug fix!)
            (np.bool_, _WeakInt, np.int32),
            (np.bool_, _WeakFloat, np.float32),
            # Weak int defers to strong
            (_WeakInt, np.int8, np.int8),
            (_WeakInt, np.float32, np.float32),
            # Weak float defers to strong float, promotes int to float32
            (_WeakFloat, np.int32, np.float32),
            (_WeakFloat, np.float16, np.float16),
            # Integer promotions
            (np.int8, np.int16, np.int16),
            (np.int32, np.int64, np.int64),
            (np.uint8, np.uint32, np.uint32),
            # Mixed signed/unsigned
            (np.int8, np.uint8, np.int16),
            (np.int32, np.uint64, np.float32),
            # Float promotions
            (np.float16, np.float32, np.float32),
            (np.float16, ml_dtypes.bfloat16, np.float32),  # Different semantics
        ],
    )
    def test_table_sample_entries(self, t0, t1, expected):
        """Test sample entries from the promotion table."""
        result = _lookup_promotion(t0, t1)
        assert result == expected, f"Expected {expected}, got {result} for ({t0}, {t1})"


class TestGetTypeKey:
    """Test the _get_type_key function for extracting type keys."""

    def test_python_bool_is_strong(self):
        """Python bool should be treated as strong bool, not weak int."""
        assert _get_type_key(True) == np.bool_
        assert _get_type_key(False) == np.bool_

    def test_python_int_is_weak(self):
        """Python int should be weak int."""
        assert _get_type_key(0) == _WeakInt
        assert _get_type_key(42) == _WeakInt
        assert _get_type_key(-100) == _WeakInt
        assert _get_type_key(1000) == _WeakInt

    def test_python_float_is_weak(self):
        """Python float should be weak float."""
        assert _get_type_key(0.0) == _WeakFloat
        assert _get_type_key(3.14) == _WeakFloat
        assert _get_type_key(-2.5) == _WeakFloat

    def test_numpy_scalars_are_strong(self):
        """Numpy scalars should be strong types."""
        assert _get_type_key(np.int32(5)) == np.int32
        assert _get_type_key(np.float32(3.14)) == np.float32
        assert _get_type_key(np.bool_(True)) == np.bool_
        assert _get_type_key(np.int8(-1)) == np.int8

    def test_numpy_arrays(self):
        """Numpy arrays should use their dtype."""
        assert _get_type_key(np.array([1, 2, 3], dtype=np.int32)) == np.int32
        assert _get_type_key(np.array([1.0, 2.0], dtype=np.float16)) == np.float16
        assert _get_type_key(np.array([True, False])) == np.bool_

    def test_object_with_dtype_as_type_class(self):
        """Objects with dtype stored as type class (not np.dtype) should work."""

        class MockTensor:
            """Mock tensor with dtype as type class."""

            def __init__(self, dtype):
                self.dtype = dtype

        assert _get_type_key(MockTensor(np.bool_)) == np.bool_
        assert _get_type_key(MockTensor(np.float32)) == np.float32
        assert _get_type_key(MockTensor(np.int16)) == np.int16


class TestFindCommonTypeHlo:
    """Test the find_common_type_hlo function."""

    def test_bool_tensor_with_int_scalar_returns_int32(self):
        """The original bug: bool tensor + int scalar should return int32, not bool."""
        bool_array = np.array([True, False])
        result = find_common_type_hlo(bool_array, 1000)
        assert result == np.dtype(np.int32), f"Expected int32, got {result}"

    def test_bool_tensor_with_float_scalar_returns_float32(self):
        """Bool tensor + float scalar should return float32."""
        bool_array = np.array([True, False])
        result = find_common_type_hlo(bool_array, 3.14)
        assert result == np.dtype(np.float32), f"Expected float32, got {result}"

    def test_float32_tensor_with_int_scalar(self):
        """Float32 tensor + int scalar should return float32."""
        float_array = np.array([1.0, 2.0], dtype=np.float32)
        result = find_common_type_hlo(float_array, 5)
        assert result == np.dtype(np.float32)

    def test_float32_tensor_with_float_scalar(self):
        """Float32 tensor + float scalar should return float32."""
        float_array = np.array([1.0, 2.0], dtype=np.float32)
        result = find_common_type_hlo(float_array, 3.14)
        assert result == np.dtype(np.float32)

    def test_int8_tensor_with_int_scalar(self):
        """Int8 tensor + int scalar should return int8 (weak defers to strong)."""
        int_array = np.array([1, 2, 3], dtype=np.int8)
        result = find_common_type_hlo(int_array, 100)
        assert result == np.dtype(np.int8)

    def test_int8_tensor_with_float_scalar(self):
        """Int8 tensor + float scalar should return float32."""
        int_array = np.array([1, 2, 3], dtype=np.int8)
        result = find_common_type_hlo(int_array, 3.14)
        assert result == np.dtype(np.float32)

    def test_two_strong_types(self):
        """Two strong types should use standard promotion."""
        int32_array = np.array([1, 2], dtype=np.int32)
        float32_array = np.array([1.0, 2.0], dtype=np.float32)
        result = find_common_type_hlo(int32_array, float32_array)
        assert result == np.dtype(np.float32)

    def test_same_types(self):
        """Same types should return that type."""
        float32_array = np.array([1.0, 2.0], dtype=np.float32)
        result = find_common_type_hlo(float32_array, float32_array)
        assert result == np.dtype(np.float32)


class TestStrongTypePromotion:
    """Test strong type promotion via _lookup_promotion."""

    def test_same_type(self):
        """Same types should return that type."""
        assert _lookup_promotion(np.float32, np.float32) == np.float32
        assert _lookup_promotion(np.int32, np.int32) == np.int32

    def test_bool_promotes_to_other(self):
        """Bool should promote to the other type."""
        assert _lookup_promotion(np.bool_, np.int32) == np.int32
        assert _lookup_promotion(np.bool_, np.float32) == np.float32

    def test_integer_promotion(self):
        """Integers should promote to wider type."""
        assert _lookup_promotion(np.int8, np.int32) == np.int32
        assert _lookup_promotion(np.int16, np.int64) == np.int64

    def test_float_promotion(self):
        """Floats should promote to wider type."""
        assert _lookup_promotion(np.float16, np.float32) == np.float32

    def test_int_float_promotion(self):
        """Int + float should return float."""
        assert _lookup_promotion(np.int32, np.float32) == np.float32
        assert _lookup_promotion(np.int64, np.float16) == np.float16

    def test_bfloat16_float16_promotes_to_float32(self):
        """bfloat16 + float16 should promote to float32 (different semantics)."""
        assert _lookup_promotion(ml_dtypes.bfloat16, np.float16) == np.float32


class TestFloat8Promotion:
    """Test float8 type promotion (strict - only self-promotion)."""

    def test_float8_self_promotion(self):
        """Float8 types should only promote with themselves."""
        assert (
            _lookup_promotion(ml_dtypes.float8_e4m3, ml_dtypes.float8_e4m3)
            == ml_dtypes.float8_e4m3
        )
        assert (
            _lookup_promotion(ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e4m3fn)
            == ml_dtypes.float8_e4m3fn
        )
        assert (
            _lookup_promotion(ml_dtypes.float8_e5m2, ml_dtypes.float8_e5m2)
            == ml_dtypes.float8_e5m2
        )

    def test_float8_cross_promotion_raises(self):
        """Different float8 types should raise TypeError."""
        with pytest.raises(TypeError, match="No implicit dtype promotion path"):
            find_common_type_hlo(
                np.array([1], dtype=ml_dtypes.float8_e4m3),
                np.array([1], dtype=ml_dtypes.float8_e5m2),
            )


class TestScalarDtypeHlo:
    """Test the scalar_dtype_hlo function."""

    def test_python_bool(self):
        """Python bool should return bool dtype."""
        assert scalar_dtype_hlo(True) == np.dtype(np.bool_)
        assert scalar_dtype_hlo(False) == np.dtype(np.bool_)

    def test_python_int(self):
        """Python int should return int32."""
        assert scalar_dtype_hlo(42) == np.dtype(np.int32)
        assert scalar_dtype_hlo(-100) == np.dtype(np.int32)

    def test_python_float(self):
        """Python float should return float32."""
        assert scalar_dtype_hlo(3.14) == np.dtype(np.float32)
        assert scalar_dtype_hlo(-2.5) == np.dtype(np.float32)

    def test_numpy_scalars(self):
        """Numpy scalars should return their dtype (with 64-bit int downcast)."""
        assert scalar_dtype_hlo(np.int32(5)) == np.dtype(np.int32)
        assert scalar_dtype_hlo(np.float32(3.14)) == np.dtype(np.float32)
        # 64-bit int types are downcast to 32-bit
        assert scalar_dtype_hlo(np.int64(5)) == np.dtype(np.int32)
        assert scalar_dtype_hlo(np.uint64(5)) == np.dtype(np.uint32)

    def test_float64_scalar_raises(self):
        """float64 scalar should raise ValueError."""
        with pytest.raises(ValueError, match="float64 is not supported"):
            scalar_dtype_hlo(np.float64(3.14))


class TestUnsupportedPromotions:
    """Test that unsupported promotions raise TypeError."""

    def test_weak_weak_raises(self):
        """Two weak types should raise TypeError."""
        with pytest.raises(TypeError, match="No implicit dtype promotion path"):
            find_common_type_hlo(42, 3.14)

        with pytest.raises(TypeError, match="No implicit dtype promotion path"):
            find_common_type_hlo(100, 200)

    def test_float64_raises(self):
        """float64 is not supported and should raise TypeError."""
        float64_array = np.array([1.0, 2.0], dtype=np.float64)
        float32_array = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(TypeError, match="Unsupported dtype"):
            find_common_type_hlo(float64_array, float32_array)

        with pytest.raises(TypeError, match="Unsupported dtype"):
            find_common_type_hlo(float32_array, float64_array)

        with pytest.raises(TypeError, match="Unsupported dtype"):
            _lookup_promotion(np.float64, np.float32)


class TestPromotionTableCompleteness:
    """Test that the promotion table covers expected type combinations."""

    # All supported strong types (float64 is NOT supported)
    STRONG_TYPES = [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        ml_dtypes.bfloat16,
    ]

    def test_all_strong_type_pairs_have_promotion(self):
        """All pairs of supported strong types (except float8) should have a promotion path."""
        for t0 in self.STRONG_TYPES:
            for t1 in self.STRONG_TYPES:
                result = _lookup_promotion(t0, t1)
                assert result is not None, f"Missing promotion for ({t0}, {t1})"

    def test_all_weak_strong_pairs_have_promotion(self):
        """All weak + supported strong type pairs should have a promotion path."""
        for weak in [_WeakInt, _WeakFloat]:
            for strong in self.STRONG_TYPES:
                result = _lookup_promotion(weak, strong)
                assert result is not None, f"Missing promotion for ({weak}, {strong})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
