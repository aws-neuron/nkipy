# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Targeted reduction operation tests, inspired by nkipy/tests/unit/test_tensor_api.py.

Tests reduction ops (sum, mean, max, min) with various axis configurations.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode

M, N = 128, 256


# -- Currently supported reductions --

def test_sum_axis_last():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.sum(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_mean_axis_last():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.mean(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_max_axis_last():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.max(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Reductions used in common patterns --

def test_sum_subtract_pattern():
    """sum along axis then broadcast-subtract (log-softmax style)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        s = np.sum(x, axis=-1, keepdims=True)
        return x - s

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_max_normalize_pattern():
    """max along axis then subtract (numerical stability for softmax)."""
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        m = np.max(x, axis=-1, keepdims=True)
        return x - m

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Reductions that need to be added --

def test_min_axis_last():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.min(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_sum_axis_first():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.sum(x, axis=0, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_prod_axis_last():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.prod(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


@pytest.mark.xfail(reason="np.argmax not yet supported in tracer")
def test_argmax():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.argmax(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_std():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.std(x, axis=-1, keepdims=True)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)
