"""
Example demonstrating MLIR execution and verification against NumPy.
"""

import numpy as np
from nkigen import trace, verify_against_numpy


# Example 1: Simple element-wise addition
@trace(input_specs=[((4, 3), "f32"), ((4, 3), "f32")])
def add_arrays(A, B):
    """Simple element-wise addition."""
    return np.add(A, B)


# Example 2: Matrix multiplication
@trace(input_specs=[((4, 3), "f32"), ((3, 5), "f32")])
def matmul_func(A, B):
    """Matrix multiplication."""
    return np.matmul(A, B)


# Example 3: Complex computation
@trace(input_specs=[((4, 4), "f32"), ((4, 4), "f32")])
def complex_func(A, B):
    """Complex computation with multiple operations."""
    temp1 = np.add(A, B)
    temp2 = np.full(temp1.shape, 10)
    result = np.multiply(temp1, temp2)
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("NKIPyKernelGen - MLIR Execution and Verification Examples")
    print("=" * 70)

    # Example 1: Element-wise addition
    print("\n1. Element-wise Addition Verification:")
    print("-" * 70)
    A1 = np.random.randn(4, 3).astype(np.float32)
    B1 = np.random.randn(4, 3).astype(np.float32)

    print(f"Input A shape: {A1.shape}")
    print(f"Input B shape: {B1.shape}")

    matches, mlir_result, numpy_result = verify_against_numpy(
        add_arrays, add_arrays.__wrapped__, [A1, B1]
    )

    if matches is not None:
        print(f"\n✓ Results match: {matches}")
        print(f"Max difference: {np.max(np.abs(mlir_result - numpy_result))}")

    # Example 2: Matrix multiplication
    print("\n\n2. Matrix Multiplication Verification:")
    print("-" * 70)
    A2 = np.random.randn(4, 3).astype(np.float32)
    B2 = np.random.randn(3, 5).astype(np.float32)

    print(f"Input A shape: {A2.shape}")
    print(f"Input B shape: {B2.shape}")

    matches, mlir_result, numpy_result = verify_against_numpy(
        matmul_func, matmul_func.__wrapped__, [A2, B2]
    )

    if matches is not None:
        print(f"\n✓ Results match: {matches}")
        print(f"Max difference: {np.max(np.abs(mlir_result - numpy_result))}")

    # Example 3: Complex computation
    print("\n\n3. Complex Computation Verification:")
    print("-" * 70)
    A3 = np.random.randn(4, 4).astype(np.float32)
    B3 = np.random.randn(4, 4).astype(np.float32)

    print(f"Input A shape: {A3.shape}")
    print(f"Input B shape: {B3.shape}")

    matches, mlir_result, numpy_result = verify_against_numpy(
        complex_func, complex_func.__wrapped__, [A3, B3]
    )

    if matches is not None:
        print(f"\n✓ Results match: {matches}")
        print(f"Max difference: {np.max(np.abs(mlir_result - numpy_result))}")

    print("\n" + "=" * 70)
    print("Verification Complete!")
    print("=" * 70)
