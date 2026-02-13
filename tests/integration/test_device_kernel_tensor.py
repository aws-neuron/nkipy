# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test for DeviceKernel and DeviceTensor APIs

This test validates the core functionality of the DeviceKernel and DeviceTensor
abstractions for compiling, loading, and executing kernels on Neuron hardware.
"""

import numpy as np
import pytest
from nkipy.runtime import is_neuron_compatible


def numpy_matmul(A, B):
    """Simple NKIPy matrix multiplication kernel"""
    A = A.transpose(1, 0)
    return A @ B


@pytest.mark.skipif(
    not is_neuron_compatible(),
    reason="Need at least 1 Neuron core for DeviceKernel tests",
)
class TestDeviceKernelTensor:
    """Tests for DeviceKernel and DeviceTensor APIs"""

    @pytest.fixture
    def test_matrices(self):
        """Create test matrices for matmul"""
        size = 1024
        A = ((np.random.rand(size, size) - 0.5) * 2).astype(np.float16)
        B = ((np.random.rand(size, size) - 0.5) * 2).astype(np.float16)
        return A, B

    @pytest.fixture
    def compiled_kernel(self, test_matrices):
        import os

        from nkipy.runtime.device_kernel import DeviceKernel

        """Compile and load the matmul kernel"""
        A, B = test_matrices

        # Use worker-unique name to avoid xdist conflicts
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        kernel_name = f"test_matmul_kernel_{worker_id}"

        kernel = DeviceKernel.compile_and_load(
            numpy_matmul,
            A,
            B,
            name=kernel_name,
            use_cached_if_exists=False,
        )
        return kernel

    def test_device_tensor_roundtrip(self, test_matrices):
        from nkipy.runtime.device_tensor import DeviceTensor

        """Test DeviceTensor creation and numpy conversion"""
        A, B = test_matrices

        # Create device tensors
        device_A = DeviceTensor.from_numpy(A)
        device_B = DeviceTensor.from_numpy(B)

        # Verify properties
        assert device_A.shape == A.shape
        assert device_A.dtype == A.dtype
        assert device_B.shape == B.shape
        assert device_B.dtype == B.dtype

        # Convert back to numpy and verify
        A_recovered = device_A.numpy()
        B_recovered = device_B.numpy()

        np.testing.assert_array_equal(A, A_recovered)
        np.testing.assert_array_equal(B, B_recovered)

    def test_device_kernel_compile_and_load(self, compiled_kernel):
        """Test kernel compilation and loading"""
        assert compiled_kernel is not None
        assert compiled_kernel.name.startswith("test_matmul_kernel_")
        assert compiled_kernel.model_ref is not None

        # Verify tensor info is populated
        assert len(compiled_kernel.input_tensors_info) > 0
        assert len(compiled_kernel.output_tensors_info) > 0

    def test_device_kernel_execution(self, test_matrices, compiled_kernel):
        """Test kernel execution with DeviceTensors"""
        from nkipy.runtime.device_tensor import DeviceTensor

        A, B = test_matrices
        size = A.shape[0]
        output = np.zeros((size, size), dtype=np.float16)

        # Create device tensors
        device_A = DeviceTensor.from_numpy(A)
        device_B = DeviceTensor.from_numpy(B)
        device_output = DeviceTensor.from_numpy(output)

        # Execute kernel
        compiled_kernel(
            inputs={"A": device_A, "B": device_B}, outputs={"output0": device_output}
        )

        # Get result
        result = device_output.numpy()

        # Verify output shape and dtype
        assert result.shape == (size, size)
        assert result.dtype == np.float16

        # Verify it's not all zeros (computation happened)
        assert not np.allclose(result, 0)

    def test_matmul_correctness(self, test_matrices, compiled_kernel):
        """Test matmul output correctness against NumPy reference"""
        from nkipy.runtime.device_tensor import DeviceTensor

        A, B = test_matrices
        size = A.shape[0]
        output = np.zeros((size, size), dtype=np.float16)

        # Create device tensors
        device_A = DeviceTensor.from_numpy(A)
        device_B = DeviceTensor.from_numpy(B)
        device_output = DeviceTensor.from_numpy(output)

        # Execute kernel
        compiled_kernel(
            inputs={"A": device_A, "B": device_B}, outputs={"output0": device_output}
        )

        # Get result
        result = device_output.numpy()

        # Compute reference using NumPy
        A_transposed = A.transpose(1, 0)
        reference = A_transposed @ B

        # Compare with tolerance appropriate for float16
        np.testing.assert_allclose(result, reference, rtol=1e-2, atol=1e-2)

    def test_device_kernel_benchmark(self, test_matrices, compiled_kernel):
        """Test kernel benchmarking functionality"""
        from nkipy.runtime.device_tensor import DeviceTensor

        A, B = test_matrices
        size = A.shape[0]
        output = np.zeros((size, size), dtype=np.float16)

        # Create device tensors
        device_A = DeviceTensor.from_numpy(A)
        device_B = DeviceTensor.from_numpy(B)
        device_output = DeviceTensor.from_numpy(output)

        # Run benchmark
        stats = compiled_kernel.benchmark(
            inputs={"A": device_A, "B": device_B},
            outputs={"output0": device_output},
            warmup_iter=2,
            benchmark_iter=3,
        )

        # Verify execution time values are reasonable
        assert stats.mean_ms > 0
        assert stats.min_ms > 0
        assert stats.max_ms >= stats.min_ms
        assert stats.std_dev_ms >= 0
        assert stats.iterations == 3
        assert stats.warmup_iterations == 2

        # Verify raw per-iteration durations
        assert isinstance(stats.durations_ms, list)
        assert len(stats.durations_ms) == 3
        assert all(d > 0 for d in stats.durations_ms)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
