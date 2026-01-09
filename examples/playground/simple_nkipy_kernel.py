#!/usr/bin/env python3
"""
Simple Matrix Multiplication Example using DeviceKernel and DeviceTensor

This example demonstrates how to:
1. Define a simple NKIPy kernel (use matrix multiplication as an example)
2. Compile and load it using DeviceKernel
3. Create DeviceTensors for inputs and outputs
4. Execute the kernel on Neuron hardware
5. Validate correctness against NumPy reference
6. Benchmark performance


This serves as a template for creating your own simple kernels.
Modify the nkipy_kernel function to implement your own kernel.
"""

import time

import numpy as np

from nkipy.runtime import DeviceKernel, DeviceTensor


def nkipy_kernel(A, B):
    """
    Simple matrix multiplication kernel using NKIPy.

    This kernel performs: result = A^T @ B
    where A^T is the transpose of A.

    Args:
        A: Input matrix of shape (M, K)
        B: Input matrix of shape (K, N)

    Returns:
        Result matrix of shape (M, N)
    """
    A = A.transpose(1, 0)
    return A @ B


def main():
    print("=" * 80)
    print("Simple Matrix Multiplication Example")
    print("=" * 80)

    # Configuration
    size = 1024  # Matrix size (1024x1024)
    warmup_iterations = 5
    benchmark_iterations = 10

    print("\nConfiguration:")
    print(f"  Matrix size: {size}x{size}")
    print("  Data type: float16")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Benchmark iterations: {benchmark_iterations}")

    print("\n[1/6] Creating test data...")
    np.random.seed(42)
    A = ((np.random.rand(size, size) - 0.5) * 2).astype(np.float16)
    B = ((np.random.rand(size, size) - 0.5) * 2).astype(np.float16)
    output = np.zeros((size, size), dtype=np.float16)
    print(f"  ✓ Created matrices A: {A.shape}, B: {B.shape}")

    print("\n[2/6] Compiling kernel...")
    compile_start = time.time()

    kernel = DeviceKernel.compile_and_load(
        nkipy_kernel,
        A,
        B,
        name="simple_kernel",
        use_cached_if_exists=False,  # Always recompile for this example
    )

    compile_time = time.time() - compile_start
    print(f"  ✓ Kernel compiled in {compile_time:.2f} seconds")
    print(f"  ✓ Kernel name: {kernel.name}")

    print("\n[3/6] Creating device tensors...")
    device_A = DeviceTensor.from_numpy(A)
    device_B = DeviceTensor.from_numpy(B)
    device_output = DeviceTensor.from_numpy(output)
    print("  ✓ Created device tensors for inputs and output")

    print("\n[4/6] Executing kernel...")
    kernel(inputs={"A": device_A, "B": device_B}, outputs={"output0": device_output})
    result = device_output.numpy()
    print("  ✓ Kernel executed successfully")
    print(f"  ✓ Output shape: {result.shape}, dtype: {result.dtype}")

    print("\n[5/6] Validating correctness...")
    reference = nkipy_kernel(A, B)

    try:
        np.testing.assert_allclose(result, reference, rtol=1e-2, atol=1e-2)
        print("  ✓ Output matches NumPy reference within tolerance")

        # Calculate relative error
        rel_error = np.abs(result - reference) / (np.abs(reference) + 1e-8)
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)
        print(f"  ✓ Max relative error: {max_rel_error:.6f}")
        print(f"  ✓ Mean relative error: {mean_rel_error:.6f}")
    except AssertionError as e:
        print(f"  ✗ Validation failed: {e}")
        return

    print("\n[6/6] Benchmarking performance...")
    stats = kernel.benchmark(
        inputs={"A": device_A, "B": device_B},
        outputs={"output0": device_output},
        warmup_iter=warmup_iterations,
        benchmark_iter=benchmark_iterations,
    )

    print("\n  Performance Results:")
    print("  ─────────────────────────────────────")
    print(f"  Mean time:    {stats.mean_ms:.3f} ms")
    print(f"  Min time:     {stats.min_ms:.3f} ms")
    print(f"  Max time:     {stats.max_ms:.3f} ms")
    print(f"  Std dev:      {stats.std_dev_ms:.3f} ms")
    print("  ─────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
