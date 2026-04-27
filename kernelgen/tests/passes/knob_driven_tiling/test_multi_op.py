"""
Tests for knob-driven-tiling with multiple operations.

These tests verify tiling of kernels with multiple linalg ops,
including same-type ops with different tile sizes using nkipy.op_id.
Run with: python -m pytest tests/passes/knob_driven_tiling/test_multi_op.py -v
Or directly: python tests/passes/knob_driven_tiling/test_multi_op.py
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Functions
# ============================================================================

def test_two_matmuls_different_tiles():
    """
    Test with two matmul operations with different tile sizes.

    Pattern:
        C = matmul(A, B)    # tile_size=[128, 128], reduction_tile=[128]
        D = matmul(C, E)    # tile_size=[64, 64], reduction_tile=[64]

    This tests per-instance matching via nkipy.op_id - each matmul
    should get its own transform sequence with its specific tile size.
    """
    M, N, K = 256, 256, 256
    # First matmul: tile 128x128, K=128, block 256x256
    tile1_m, tile1_n, tile1_k = 128, 128, 128
    # Second matmul: tile 64x64, K=64, block 128x128
    tile2_m, tile2_n, tile2_k = 64, 64, 64

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((N, N), "f32")])
    def two_matmul_kernel(a, b, e):
        c = np.matmul(a, b)
        knob.knob(c, tile_size=[128, 128], reduction_tile=[128])

        d = np.matmul(c, e)
        knob.knob(d, tile_size=[64, 64], reduction_tile=[64])
        
        return d

    # FileCheck: verify both matmuls get correct tile sizes
    # First matmul: 128x128x128 tiles, with linalg.transpose for LHS
    # Second matmul: 64x64x64 tiles
    # Transpose output is promoted to SBUF (bufferization.alloc_tensor before linalg.transpose)
    check_patterns = f"""
    CHECK: func.func
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.transpose {{{{.*}}}} permutation = [1, 0]
    CHECK: linalg.matmul_transpose_a {{{{.*}}}}tensor<{tile1_k}x{tile1_m}xf32>, tensor<{tile1_k}x{tile1_n}xf32>
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.transpose {{{{.*}}}} permutation = [1, 0]
    CHECK: linalg.matmul_transpose_a {{{{.*}}}}tensor<{tile2_k}x{tile2_m}xf32>, tensor<{tile2_k}x{tile2_n}xf32>
    """
    run_kernel_test(
        two_matmul_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_two_adds_different_tiles():
    """
    Test with two add operations with different tile sizes.

    Pattern:
        C = A + B    # tile_size=[128, 128]
        D = C + E    # tile_size=[64, 64]

    Both adds will have all operands promoted to SBUF.
    """
    shape = (256, 256)
    tile1 = [128, 128]
    tile2 = [64, 64]

    @trace(input_specs=[(shape, "f32"), (shape, "f32"), (shape, "f32")])
    def two_add_kernel(a, b, e):
        c = a + b
        knob.knob(c, tile_size=[128, 128])

        d = c + e
        knob.knob(d, tile_size=[64, 64])

        return d

    # FileCheck: verify both adds get tiled with different tile sizes
    # With SBUF promotion, each add will have alloc_tensor with memory_space
    check_patterns = f"""
    CHECK: func.func
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.add {{{{.*}}}}tensor<{tile1[0]}x{tile1[1]}xf32>
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.add {{{{.*}}}}tensor<{tile2[0]}x{tile2[1]}xf32>
    """
    run_kernel_test(
        two_add_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_three_matmuls_same_and_different_tiles():
    """
    Test with three matmuls - two with same tile size, one different.

    Pattern:
        C = matmul(A, B)    # tile_size=[128, 128], reduction_tile=[128]
        D = matmul(C, E)    # tile_size=[64, 64], reduction_tile=[64]
        F = matmul(D, G)    # tile_size=[128, 128], reduction_tile=[128]
    """
    M, N, K = 256, 256, 256

    @trace(input_specs=[
        ((M, K), "f32"), ((K, N), "f32"), ((N, N), "f32"), ((N, N), "f32")
    ])
    def three_matmul_kernel(a, b, e, g):
        c = np.matmul(a, b)
        knob.knob(c, tile_size=[128, 128], reduction_tile=[128])

        d = np.matmul(c, e)
        knob.knob(d, tile_size=[64, 64], reduction_tile=[64])

        f = np.matmul(d, g)
        knob.knob(f, tile_size=[128, 128], reduction_tile=[128])
        
        return f

    run_kernel_test(
        three_matmul_kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


def test_mixed_matmul_and_add():
    """
    Test with mixed operations - matmul followed by add with different tiles.

    Pattern:
        C = matmul(A, B)    # tile_size=[128, 128], reduction_tile=[128]
        D = C + E           # tile_size=[64, 64]
    """
    M, N, K = 256, 256, 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def mixed_kernel(a, b, e):
        c = np.matmul(a, b)
        knob.knob(c, tile_size=[128, 128], reduction_tile=[128])
        
        d = c + e
        knob.knob(d, tile_size=[64, 64])

        return d

    run_kernel_test(
        mixed_kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
    )


def test_matmul_add_chain():
    """
    Test typical matmul + bias add pattern with different tile sizes.

    Pattern:
        C = matmul(A, B)    # tile_size=[128, 128], reduction_tile=[128]
        D = C + bias        # tile_size=[128, 128] (same spatial dims)

    Matmul: LHS/RHS promoted to SBUF, output to PSUM
    Add: all operands promoted to SBUF
    """
    M, N, K = 256, 256, 256
    tile_m, tile_n, tile_k = 128, 128, 128
    add_tile = [128, 128]

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        c = np.matmul(a, b)
        knob.knob(c, tile_size=[128, 128], reduction_tile=[128])
        
        # Add bias - using same spatial tile sizes as matmul output
        d = c + bias
        knob.knob(d, tile_size=[128, 128])

        return d

    # FileCheck: matmul tiled then add tiled
    # Transpose output promoted to SBUF, then add operands promoted to SBUF
    check_patterns = f"""
    CHECK: func.func
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.transpose {{{{.*}}}} permutation = [1, 0]
    CHECK: linalg.matmul_transpose_a {{{{.*}}}}tensor<{tile_k}x{tile_m}xf32>, tensor<{tile_k}x{tile_n}xf32>
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32
    CHECK: linalg.add {{{{.*}}}}tensor<{add_tile[0]}x{add_tile[1]}xf32>
    """
    run_kernel_test(
        matmul_add_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
