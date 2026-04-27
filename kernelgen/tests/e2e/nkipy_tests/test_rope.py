# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/rope_dynamo.py

Rotary Position Embedding (RoPE) generation kernel from torch dynamo graph.
Operations: expand_dims, broadcast_to, reshape, matmul, transpose, concatenate, cos, sin, multiply.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


@pytest.mark.xfail(reason="np.concatenate, np.cos, np.sin, int input dtype not yet supported")
def test_rope():
    @trace(input_specs=[
        ((32,), "f32"),       # rotary embedding base
        ((1, 7), "i32"),      # position ids
    ])
    def kernel(freq_base, pos_ids):
        unsqueeze = np.expand_dims(freq_base, 0)
        slice_1 = unsqueeze[:, 0:]
        unsqueeze_1 = np.expand_dims(slice_1, 2)
        expand = np.broadcast_to(unsqueeze_1, [1, 32, 1])

        slice_2 = pos_ids[0:]
        unsqueeze_2 = np.expand_dims(slice_2, 1)
        slice_3 = unsqueeze_2[:, :, 0:]
        to_float = slice_3.astype(np.float32)

        expand_1 = np.broadcast_to(expand, [1, 32, 1])
        view = np.reshape(expand_1, [1, 32, 1])
        expand_2 = np.broadcast_to(to_float, [1, 1, 7])
        view_1 = np.reshape(expand_2, [1, 1, 7])
        bmm = np.matmul(view, view_1)
        view_2 = np.reshape(bmm, [1, 32, 7])
        permute = np.transpose(view_2, [0, 2, 1])

        cat = np.concatenate([permute, permute], -1)
        cos = np.cos(cat)
        sin = np.sin(cat)

        mul = np.multiply(cos, 1.0)
        mul_1 = np.multiply(sin, 1.0)
        # Original returns tuple; return first output for testing
        return mul

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )
