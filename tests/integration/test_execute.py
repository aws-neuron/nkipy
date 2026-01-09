# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test the decorator implementations: @simulate_jit and @baremetal_jit
Also test the lower-level traced kernel execution functions
"""

import numpy as np
import pytest
from nkipy.core.compile import trace
from nkipy.runtime import (
    baremetal_jit,
    baremetal_run_traced_kernel,
    is_neuron_compatible,
    simulate_jit,
    simulate_traced_kernel,
)


def simple_kernel(A, B):
    """Simple matrix multiplication kernel for testing"""
    A = A.transpose(1, 0)
    return A @ B


def test_simulate():
    """Test the @simulate decorator"""

    @simulate_jit
    def matmul_sim(A, B):
        A = A.transpose(1, 0)
        return A @ B

    # Create test data
    size = 128
    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    result = matmul_sim(A, B)

    # Verify correctness
    expected = A.T @ B
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


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


def test_simulate_traced_kernel():
    """Test the simulate_traced_kernel function"""

    def matmul_kernel(A, B):
        A = A.transpose(1, 0)
        return A @ B

    # Trace the kernel
    traced_kernel = trace(matmul_kernel)

    # Create test data
    size = 128
    A = np.random.randn(size, size).astype(np.float16)
    B = np.random.randn(size, size).astype(np.float16)

    # Specialize with concrete shapes
    traced_kernel.specialize(A, B)

    # Execute with simulate_traced_kernel
    result = simulate_traced_kernel(traced_kernel, A, B)

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
