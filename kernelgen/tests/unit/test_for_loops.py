"""
Tests for control flow operations: fori_loop.

These tests verify that MLIR/LLVM execution matches NumPy CPU execution
for control flow operations like fori_loop.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from nkipy_kernelgen.apis import fori_loop
from harness import run_kernel_test, Mode


# ============================================================================
# Basic fori_loop Tests
# ============================================================================

def test_simple_accumulation():
    """Test simple accumulation in fori_loop."""
    @trace(input_specs=[((10,), "f32")])
    def sum_with_loop(x):
        def body(i, acc):
            return acc + x[i]

        init = np.zeros((1,), dtype=np.float32)
        result = fori_loop(0, 10, body, init)
        return result

    run_kernel_test(sum_with_loop, stop_after="trace", modes=Mode.LLVM)


def test_single_accumulator():
    """Test fori_loop with a single tensor accumulator."""
    @trace(input_specs=[((8,), "f32")])
    def scale_accumulate(x):
        def body(i, acc):
            return acc + x[i] * 2.0

        init = np.zeros((1,), dtype=np.float32)
        result = fori_loop(0, 8, body, init)
        return result

    run_kernel_test(scale_accumulate, stop_after="trace", modes=Mode.LLVM)


# ============================================================================
# Multiple Accumulator Tests
# ============================================================================

def test_two_accumulators():
    """Test fori_loop with two tensor accumulators."""
    @trace(input_specs=[((8,), "f32")])
    def dual_accumulate(x):
        def body(i, accs):
            acc1, acc2 = accs
            return (acc1 + x[i], acc2 + x[i] * x[i])

        init1 = np.zeros((1,), dtype=np.float32)
        init2 = np.zeros((1,), dtype=np.float32)
        result1, result2 = fori_loop(0, 8, body, (init1, init2))
        # Return sum of both for testing
        return result1 + result2

    run_kernel_test(
        dual_accumulate, stop_after="trace",
        rtol=1e-4, atol=1e-4, modes=Mode.LLVM,
    )


def test_multiple_tensor_operations():
    """Test fori_loop with multiple tensor operations."""
    @trace(input_specs=[((4, 6), "f32")])
    def multi_tensor_loop(x):
        def body(i, accs):
            acc1, acc2 = accs
            # Simple operations that don't require advanced slicing
            row_sum = np.sum(x[i:i+1, :])
            return (acc1 + row_sum, acc2 + row_sum * 2.0)

        init1 = np.zeros((1,), dtype=np.float32)
        init2 = np.zeros((1,), dtype=np.float32)
        result1, result2 = fori_loop(0, 4, body, (init1, init2))
        return result1 + result2

    run_kernel_test(
        multi_tensor_loop, stop_after="trace",
        rtol=1e-4, atol=1e-4, modes=Mode.LLVM,
    )


# ============================================================================
# Dynamic Slicing Tests
# ============================================================================

def test_tiled_accumulation():
    """Test tiled accumulation with dynamic slicing."""
    @trace(input_specs=[((8,), "f32")])
    def tiled_sum(x, TILE_SIZE=2):
        def body(i, acc):
            # Dynamic slicing with loop index
            chunk = x[i * TILE_SIZE : (i + 1) * TILE_SIZE]
            chunk_sum = np.sum(chunk)
            return acc + chunk_sum

        init = np.zeros((1,), dtype=np.float32)
        result = fori_loop(0, 4, body, init)
        return result

    run_kernel_test(
        tiled_sum, stop_after="trace",
        rtol=1e-4, atol=1e-4, modes=Mode.LLVM,
    )


def test_2d_tiled_operations():
    """Test 2D tiled operations with dynamic slicing."""
    @trace(input_specs=[((4, 8), "f16")])
    def tiled_2d_sum(input_tensor, TILING_FACTOR=2):
        M, K = input_tensor.shape
        TILED_CHUNK = K // TILING_FACTOR

        sum_buffer = np.zeros((M, TILED_CHUNK), dtype=np.float16)

        def body(i, acc):
            input_chunk = input_tensor[:, i * TILED_CHUNK : (i + 1) * TILED_CHUNK]
            return np.add(acc, input_chunk)

        result = fori_loop(0, TILING_FACTOR, body, sum_buffer)
        return np.sum(result)

    run_kernel_test(
        tiled_2d_sum, stop_after="trace",
        rtol=0.01, atol=0.01, modes=Mode.LLVM,
    )


# ============================================================================
# Complex Scenario Tests
# ============================================================================

def test_rmsnorm_pattern():
    """Test RMSNorm-like pattern with fori_loop."""
    @trace(input_specs=[((4, 8), "f16")])
    def simple_rmsnorm(input_tensor, TILING_FACTOR=2):
        M, K = input_tensor.shape
        TILED_CHUNK = K // TILING_FACTOR

        square_sum_buffer = np.zeros((M, TILED_CHUNK), dtype=np.float16)

        def body(i, acc):
            input_chunk = input_tensor[:, i * TILED_CHUNK : (i + 1) * TILED_CHUNK]
            squared_input = np.square(input_chunk)
            scaled_square = np.divide(squared_input, K)
            return np.add(scaled_square, acc)

        square_sum_buffer = fori_loop(0, TILING_FACTOR, body, square_sum_buffer)
        rms_sum = np.sum(square_sum_buffer, axis=1, keepdims=True)
        return rms_sum

    run_kernel_test(
        simple_rmsnorm, stop_after="trace",
        rtol=0.01, atol=0.01, modes=Mode.LLVM,
    )


def test_matmul_with_loop():
    """Test matrix multiplication accumulation with fori_loop."""
    @trace(input_specs=[((4, 8), "f16"), ((8, 5), "f16")])
    def tiled_matmul(input_tensor, weight_matrix, TILING_FACTOR=2):
        M, K = input_tensor.shape
        K_, N = weight_matrix.shape
        assert K == K_
        TILED_CHUNK = K // TILING_FACTOR

        matmul_buffer = np.zeros((M, N), dtype=np.float16)

        def body(i, acc):
            input_chunk = input_tensor[:, i * TILED_CHUNK : (i + 1) * TILED_CHUNK]
            weight_chunk = weight_matrix[i * TILED_CHUNK : (i + 1) * TILED_CHUNK, :]
            matmul_result = np.matmul(input_chunk, weight_chunk)
            return np.add(matmul_result, acc)

        result = fori_loop(0, TILING_FACTOR, body, matmul_buffer)
        return result

    run_kernel_test(
        tiled_matmul, stop_after="trace",
        rtol=0.01, atol=0.01, modes=Mode.LLVM,
    )


# ============================================================================
# MLIR Generation Tests
# ============================================================================

def test_mlir_contains_scf_for():
    """Verify that MLIR contains scf.for operation."""
    @trace(input_specs=[((8,), "f32")])
    def loop_func(x):
        def body(i, acc):
            return acc + x[i]

        init = np.zeros((1,), dtype=np.float32)
        return fori_loop(0, 8, body, init)

    run_kernel_test(
        loop_func, stop_after="trace",
        check_ir_contains=["scf.for", "scf.yield"],
        modes=Mode.STRING_CHECK,
    )


def test_mlir_with_dynamic_slicing():
    """Verify MLIR generation with dynamic slicing."""
    @trace(input_specs=[((4, 8), "f16")])
    def dynamic_slice_func(x, TILE=4):
        def body(i, acc):
            chunk = x[:, i * TILE : (i + 1) * TILE]
            return acc + np.sum(chunk)

        init = np.zeros((1,), dtype=np.float16)
        return fori_loop(0, 2, body, init)

    run_kernel_test(
        dynamic_slice_func, stop_after="trace",
        check_ir_contains=["scf.for", "tensor.extract_slice"],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_single_iteration():
    """Test loop with single iteration."""
    @trace(input_specs=[((4,), "f32")])
    def single_iter_loop(x):
        def body(i, acc):
            return acc + x[0:4]

        init = np.zeros((4,), dtype=np.float32)
        return fori_loop(0, 1, body, init)

    run_kernel_test(single_iter_loop, stop_after="trace", modes=Mode.LLVM)


def test_small_tiling():
    """Test with small tile sizes."""
    @trace(input_specs=[((4, 4), "f32")])
    def small_tile_loop(x):
        def body(i, acc):
            chunk = x[:, i * 1 : (i + 1) * 1]
            return acc + np.sum(chunk)

        init = np.zeros((1,), dtype=np.float32)
        return fori_loop(0, 4, body, init)

    run_kernel_test(
        small_tile_loop, stop_after="trace",
        rtol=1e-4, atol=1e-4, modes=Mode.LLVM,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
