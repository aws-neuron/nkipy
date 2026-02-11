# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test the two execution modes: CPU (direct call) and Device (traced + compiled).
Also test the @baremetal_jit decorator and lower-level traced kernel execution.
"""

import numpy as np
import pytest
from nkipy.core.compile import trace
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import (
    baremetal_jit,
    baremetal_run_traced_kernel,
    is_neuron_compatible,
)


def simple_kernel(A, B):
    """Simple matrix multiplication kernel for testing"""
    A = A.transpose(1, 0)
    return A @ B


def test_direct_cpu_execution():
    """Test that kernels can be called directly for CPU execution."""
    size = 128
    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    # Direct call runs on CPU via default backend
    result = simple_kernel(A, B)

    # Verify correctness
    expected = A.T @ B
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


def test_trace_validation():
    """Test that tracing a kernel to HLO produces valid IR."""

    def matmul_kernel(A, B):
        A = A.transpose(1, 0)
        return A @ B

    # Trace the kernel
    traced_kernel = NKIPyKernel.trace(matmul_kernel, backend="hlo")

    # Create test data
    size = 128
    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    # Specialize with concrete shapes - validates HLO generation
    traced_kernel.specialize(A, B)

    # Verify the traced kernel has generated code
    assert traced_kernel._code is not None


@pytest.mark.skipif(
    not is_neuron_compatible(),
    reason="Need Neuron hardware for @baremetal_jit tests",
)
def test_baremetal_jit():
    """Test the @baremetal_jit decorator"""

    @baremetal_jit
    def matmul_jit(A, B):
        A = A.transpose(1, 0)
        return A @ B

    size = 512
    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    result = matmul_jit(A, B)

    # Verify correctness
    expected = A.T @ B
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(
    not is_neuron_compatible(),
    reason="Need Neuron hardware for baremetal_run_traced_kernel tests",
)
def test_baremetal_run_traced_kernel():
    """Test the baremetal_run_traced_kernel function"""

    def matmul_kernel(A, B):
        A = A.transpose(1, 0)
        return A @ B

    # Trace the kernel
    traced_kernel = trace(matmul_kernel)

    # Create test data
    size = 256
    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    # Specialize with concrete shapes
    traced_kernel.specialize(A, B)

    # Execute with baremetal_run_traced_kernel
    result = baremetal_run_traced_kernel(traced_kernel, A, B)

    # Verify correctness
    expected = A.T @ B
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
