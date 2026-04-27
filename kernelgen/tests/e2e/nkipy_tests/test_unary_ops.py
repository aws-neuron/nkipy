# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Targeted unary operation tests, inspired by nkipy/tests/unit/test_tensor_api.py.

Tests each unary NumPy op through the full NKIPyKernelGen pipeline.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode

M, N = 128, 256
TILE = [128, 128]


def test_exp():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.exp(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_sqrt():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        # sqrt needs positive inputs; tracer uses random [0,1) by default
        return np.sqrt(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_negative():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.negative(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_reciprocal():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.reciprocal(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW, rtol=1e-3, atol=1e-3)


def test_abs():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.abs(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_log():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.log(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_sin():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.sin(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_cos():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.cos(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_tanh():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.tanh(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


@pytest.mark.xfail(reason="No NISA activation for ceil — no hardware support")
def test_ceil():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.ceil(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


@pytest.mark.xfail(reason="No NISA activation for floor — no hardware support")
def test_floor():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.floor(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_square():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.square(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)


def test_sign():
    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        return np.sign(x)

    run_kernel_test(kernel, modes=Mode.BIR_SIM | Mode.HW)
