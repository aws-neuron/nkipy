# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/softmax.py

Softmax kernel: exp(x - max(x)) / sum(exp(x - max(x)))
"""

import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


def test_softmax():
    M, N = 128, 256

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        sum_x = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / sum_x

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )
