# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IR content hashing."""

import numpy as np
from nkipy.core.compile import trace


def _trace_and_specialize(kernel_fn, *args, **kwargs):
    """Helper: trace a kernel, specialize with given args, return the IR."""
    traced = trace(kernel_fn)
    traced.specialize(*args, **kwargs)
    return traced._code


def test_hlo_hash_varies_with_shape():
    """Same kernel source with different input shapes produces different hashes."""

    def add_kernel(x, y):
        return np.add(x, y)

    ir_small = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    ir_large = _trace_and_specialize(
        add_kernel,
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
    )

    assert ir_small.content_hash("") != ir_large.content_hash("")


def test_hlo_hash_deterministic():
    """Same kernel with same inputs produces identical hashes across traces."""

    def add_kernel(x, y):
        return np.add(x, y)

    inputs = (
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    ir1 = _trace_and_specialize(add_kernel, *inputs)
    ir2 = _trace_and_specialize(add_kernel, *inputs)

    assert ir1.content_hash("--lnc 1") == ir2.content_hash("--lnc 1")


def test_hlo_hash_varies_with_dtype():
    """Same kernel with different dtypes produces different hashes."""

    def add_kernel(x, y):
        return np.add(x, y)

    ir_f32 = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    ir_f16 = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float16),
        np.zeros((2, 2), dtype=np.float16),
    )

    assert ir_f32.content_hash("") != ir_f16.content_hash("")


def test_hlo_hash_varies_with_compiler_args():
    """Same HLO with different compiler args produces different hashes."""

    def add_kernel(x, y):
        return np.add(x, y)

    ir = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    hash1 = ir.content_hash("--lnc 1")
    hash2 = ir.content_hash("--lnc 2")
    assert hash1 != hash2
