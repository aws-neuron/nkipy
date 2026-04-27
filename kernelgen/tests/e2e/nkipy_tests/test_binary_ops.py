# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Targeted binary operation tests, inspired by nkipy/tests/unit/test_tensor_api.py.

Tests each binary NumPy op through the full NKIPyKernelGen pipeline.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode

M, N = 128, 256


# -- Currently supported binary ops --

def test_add():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.add(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_subtract():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.subtract(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_multiply():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.multiply(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_divide():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.divide(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_maximum():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.maximum(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_minimum():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.minimum(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Scalar-tensor arithmetic --

def test_add_scalar():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return x + 2.0

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_scalar_subtract():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return 5.0 - x

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_multiply_scalar():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return x * 3.0

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_scalar_divide():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return 1.0 / x

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


# -- Binary ops that need to be added --

def test_power():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.power(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-2, atol=1e-2)


def test_power_scalar():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.power(x, 2)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


@pytest.mark.xfail(reason="np.floor_divide requires floor+divide decomposition")
def test_floor_divide():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.floor_divide(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


@pytest.mark.xfail(reason="NISA backend does not support tensor_tensor_arith(op=MOD)")
def test_mod():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.mod(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Comparison ops --

def test_greater_equal():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.greater_equal(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_less():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.less(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_equal():
    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.equal(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


# -- Broadcasting --

def test_broadcast_column():
    """a (M,N) + b (M,1) with broadcasting."""
    @trace(input_specs=[((M, N), "f32"), ((M, 1), "f32")])
    def kernel(a, b):
        return np.add(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_broadcast_row():
    """a (M,N) + b (1,N) with broadcasting."""
    @trace(input_specs=[((M, N), "f32"), ((1, N), "f32")])
    def kernel(a, b):
        return np.add(a, b)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)
