# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for automatic dtype sanitization of unsupported 64-bit types."""

import warnings

import numpy as np
from nkipy.core.trace import NKIPyKernel, _sanitize_array_dtype


class TestSanitizeArrayDtype:
    """Tests for the _sanitize_array_dtype helper."""

    def test_float64_downcast_to_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _sanitize_array_dtype(arr, "x")
            assert len(w) == 1
            assert "float64" in str(w[0].message)
            assert "float32" in str(w[0].message)
        assert result.dtype == np.float32

    def test_int64_downcast_to_int32(self):
        arr = np.array([1, 2], dtype=np.int64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _sanitize_array_dtype(arr, "x")
            assert len(w) == 1
            assert "int64" in str(w[0].message)
        assert result.dtype == np.int32

    def test_uint64_downcast_to_uint32(self):
        arr = np.array([1, 2], dtype=np.uint64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _sanitize_array_dtype(arr, "x")
            assert len(w) == 1
        assert result.dtype == np.uint32

    def test_float32_unchanged(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _sanitize_array_dtype(arr, "x")
            assert len(w) == 0
        assert result is arr  # same object, no copy

    def test_int32_unchanged(self):
        arr = np.array([1, 2], dtype=np.int32)
        result = _sanitize_array_dtype(arr, "x")
        assert result is arr


class TestKernelTracingDtypeSanitization:
    """Tests that NKIPyKernel.specialize auto-downcasts 64-bit dtypes."""

    def test_trace_with_float64_input(self):
        def kernel(a):
            return a

        traced = NKIPyKernel.trace(kernel)
        a = np.ones((2, 2), dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traced.specialize(a)
            assert any("float64" in str(wi.message) for wi in w)
        # The traced parameter should be float32, not float64
        assert traced._code.parameters[0].dtype == np.float32

    def test_trace_with_int64_input(self):
        def kernel(a):
            return a

        traced = NKIPyKernel.trace(kernel)
        a = np.array([[0, 1], [2, 3]], dtype=np.int64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traced.specialize(a)
            assert any("int64" in str(wi.message) for wi in w)
        assert traced._code.parameters[0].dtype == np.int32

    def test_trace_with_default_int_array(self):
        """np.array([0, 1, 2, 3]) defaults to int64 on most platforms."""

        def kernel(a):
            return a

        traced = NKIPyKernel.trace(kernel)
        a = np.array([[0, 1], [2, 3]])  # default dtype
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            traced.specialize(a)
        # Should be int32 regardless of platform default
        assert traced._code.parameters[0].dtype == np.int32
