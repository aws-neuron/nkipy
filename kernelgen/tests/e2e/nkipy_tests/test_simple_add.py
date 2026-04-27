# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/simple.py

Simple tensor addition kernel.
"""

import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


def test_simple_add():
    M, N = 128, 256

    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b):
        return np.add(a, b)

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )
