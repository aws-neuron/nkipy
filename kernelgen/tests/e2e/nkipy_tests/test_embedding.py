# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/embedding_dynamo.py

Embedding lookup with boundary checking and masking.
Operations: greater_equal, less, bitwise_and, bitwise_or, logical_not,
            multiply, add, subtract, take, expand_dims, where.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


def test_embedding_lookup():
    vocab_size = 256
    embed_dim = 128

    @trace(input_specs=[
        ((1, 128), "i32"),                # token indices
        ((vocab_size, embed_dim), "f32"),  # embedding table
    ])
    def kernel(indices, embed_table):
        ge = np.greater_equal(indices, 0)
        lt = np.less(indices, vocab_size)
        bitwise_and = np.bitwise_and(ge, lt)
        ge_1 = np.greater_equal(indices, vocab_size * 2)
        lt_1 = np.less(indices, vocab_size * 2)
        bitwise_and_1 = np.bitwise_and(ge_1, lt_1)
        mul = np.multiply(bitwise_and, 0, dtype=np.int32)
        mul_1 = np.multiply(bitwise_and_1, vocab_size, dtype=np.int32)
        add = np.add(mul, mul_1)
        bitwise_or = np.bitwise_or(bitwise_and, bitwise_and_1)
        sub = np.subtract(indices, add)
        mul_2 = np.multiply(bitwise_or, sub)
        bitwise_not = np.logical_not(bitwise_or)
        embedding = np.take(embed_table, mul_2, axis=0)
        unsqueeze = np.expand_dims(bitwise_not, -1)
        scalar_tensor = np.float32(0.0)
        where = np.where(unsqueeze, scalar_tensor, embedding)
        return where

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
    )
