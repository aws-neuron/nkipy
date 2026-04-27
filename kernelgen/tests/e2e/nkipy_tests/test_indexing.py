# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/indexing.py

Tensor slicing and addition: add slices [2:4,:] and [0:2,:] of input tensor.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


def test_indexed_add():
    M, N = 256, 128

    @trace(input_specs=[((M, N), "f32")])
    def kernel(input_tensor):
        a = input_tensor[128:256, :]
        b = input_tensor[0:128, :]
        return np.add(a, b)

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )
