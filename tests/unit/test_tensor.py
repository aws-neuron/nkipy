# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NKIPyTensorRef properties.

Verifies that tensor properties behave equivalently to numpy counterparts.
"""

import numpy as np
import pytest
from utils import (
    NEURON_AVAILABLE,
    baremetal_assert_allclose,
    on_device_test,
    trace_and_compile,
    trace_mode,  # noqa: F401 - pytest fixture
)

SHAPES = [
    (4,),
    (2, 3),
    (2, 3, 4),
    (1, 1),
    (5, 1, 4),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_shape(trace_mode, shape, dtype):
    """tensor.shape matches numpy."""
    in0 = np.ones(shape, dtype=dtype)

    def kernel(a):
        assert a.shape == in0.shape
        return a + 0

    trace_and_compile(kernel, trace_mode, in0)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_dtype(trace_mode, shape, dtype):
    """tensor.dtype matches numpy."""
    in0 = np.ones(shape, dtype=dtype)

    def kernel(a):
        assert a.dtype == in0.dtype
        return a + 0

    trace_and_compile(kernel, trace_mode, in0)


@pytest.mark.parametrize("shape", SHAPES)
def test_ndim(trace_mode, shape):
    """tensor.ndim matches numpy."""
    in0 = np.ones(shape, dtype=np.float32)

    def kernel(a):
        assert a.ndim == in0.ndim
        return a + 0

    trace_and_compile(kernel, trace_mode, in0)


@pytest.mark.parametrize("shape", SHAPES)
def test_size(trace_mode, shape):
    """tensor.size matches numpy."""
    in0 = np.ones(shape, dtype=np.float32)

    def kernel(a):
        assert a.size == in0.size
        return a + 0

    trace_and_compile(kernel, trace_mode, in0)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4),
        (5, 1),
    ],
)
def test_T(trace_mode, shape):
    """tensor.T is equivalent to numpy's .T (full axis reversal)."""

    def kernel(a):
        return a.T

    in0 = np.random.random_sample(shape).astype(np.float32)
    expected = kernel(in0)

    if NEURON_AVAILABLE:
        out_device = on_device_test(kernel, trace_mode, in0)
        baremetal_assert_allclose(expected, out_device)
    else:
        trace_and_compile(kernel, trace_mode, in0)
