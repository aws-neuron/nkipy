# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HLO-based kernel cache hashing."""

import numpy as np
from nkipy.core.compile import trace
from nkipy.runtime.device_kernel import _hlo_content_hash


def _trace_and_specialize(kernel_fn, *args, **kwargs):
    """Helper: trace a kernel, specialize with given args, return the HLOModule."""
    traced = trace(kernel_fn)
    traced.specialize(*args, **kwargs)
    return traced._code


def test_hlo_hash_varies_with_shape():
    """Same kernel source with different input shapes produces different hashes."""

    def add_kernel(x, y):
        return np.add(x, y)

    hlo_small = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    hlo_large = _trace_and_specialize(
        add_kernel,
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
    )

    assert _hlo_content_hash(hlo_small, "") != _hlo_content_hash(hlo_large, "")


def test_hlo_hash_deterministic():
    """Same kernel with same inputs produces identical hashes across traces."""

    def add_kernel(x, y):
        return np.add(x, y)

    inputs = (
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    hlo1 = _trace_and_specialize(add_kernel, *inputs)
    hlo2 = _trace_and_specialize(add_kernel, *inputs)

    assert _hlo_content_hash(hlo1, "--lnc 1") == _hlo_content_hash(hlo2, "--lnc 1")


def test_hlo_hash_varies_with_dtype():
    """Same kernel with different dtypes produces different hashes."""

    def add_kernel(x, y):
        return np.add(x, y)

    hlo_f32 = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    hlo_f16 = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float16),
        np.zeros((2, 2), dtype=np.float16),
    )

    assert _hlo_content_hash(hlo_f32, "") != _hlo_content_hash(hlo_f16, "")


def test_hlo_hash_varies_with_compiler_args():
    """Same HLO with different compiler args produces different hashes."""

    def add_kernel(x, y):
        return np.add(x, y)

    hlo = _trace_and_specialize(
        add_kernel,
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    hash1 = _hlo_content_hash(hlo, "--lnc 1")
    hash2 = _hlo_content_hash(hlo, "--lnc 2")
    assert hash1 != hash2
