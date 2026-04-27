"""
Tests for knob-driven-tiling pass with matmul operations.

These tests generate MLIR output files for manual inspection.
Run with: python -m pytest tests/passes/knob_driven_tiling/test_matmul.py -v
Or directly: python tests/passes/knob_driven_tiling/test_matmul.py
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from pass_utils import compile_knob_pipeline
from harness import run_kernel_test, Mode

# ============================================================================
# Test Configurations
# ============================================================================

# (M, N, K, tile_size [M, N], reduction_tile [K])
TEST_CONFIGS = [
    # Square matrices
    pytest.param(256, 256, 256, [128, 128], [128], id="256x256x256_tile128"),
    pytest.param(512, 512, 512, [128, 128], [128], id="512x512x512_tile128"),

    # Different tile sizes
    pytest.param(256, 256, 256, [64, 64], [64], id="256x256x256_tile64"),
    pytest.param(256, 256, 256, [128, 64], [128], id="256x256x256_tile_mnk_128_64_128"),

    # Block size 1 cases: tile == dim, so blocking degenerates (no data reuse)
    pytest.param(128, 128, 128, [128, 128], [128], id="128x128x128_no_blocking"),
    pytest.param(256, 128, 256, [128, 128], [128], id="256x128x256_no_blocking_N"),
    pytest.param(128, 256, 256, [128, 128], [128], id="128x256x256_no_blocking_M"),
]


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.parametrize("M,N,K,tile_size,reduction_tile", TEST_CONFIGS)
def test_matmul_tiling(M, N, K, tile_size, reduction_tile, request):
    """
    Test knob-driven-tiling on simple matmul with 6-level blocking.

    KnobDrivenTiling generates (TILES_IN_BLOCK = 2):
      for block_m in [0, M, BLOCK_M):      // BLOCK_M = TILE_M * 2
        for block_n in [0, N, BLOCK_N):    // BLOCK_N = TILE_N * 2
          for tile_m in [0, BLOCK_M, TILE_M):
            for tile_n in [0, BLOCK_N, TILE_N):
              for k in [0, K, TILE_K):
                matmul_transpose_a

    Args:
        M, N, K: Matrix dimensions (A is MxK, B is KxN)
        tile_size: Tile size for matmul output [tileM, tileN]
        reduction_tile: Tile size for reduction dim [tileK]
    """
    # Extract tile sizes
    tile_m, tile_n = tile_size
    tile_k = reduction_tile[0]

    # Dynamic blocking: use 2 if dim is large enough, else 1
    blocks_m = 2 if M >= tile_m * 2 else 1
    blocks_n = 2 if N >= tile_n * 2 else 1
    block_m = tile_m * blocks_m
    block_n = tile_n * blocks_n

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def matmul_kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    # FileCheck verification for 6-level blocking
    # LHS block size: [K, block_m] (transposed)
    # RHS block size: [K, block_n]
    lhs_block_k = K
    lhs_block_m = block_m
    rhs_block_k = K
    rhs_block_n = block_n

    # LHS is transposed: original [M,K] -> transposed [K,M]
    # FileCheck regex uses {{.*}} - in f-strings we need {{{{.*}}}} to escape the braces
    # Constants may have suffixes like %c0_3, so we use %c0{{.*}} to match %c0, %c0_1, %c0_3, etc.
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{M}{{{{.*}}}} step %c{block_m}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32{{{{.*}}}} : tensor<{lhs_block_k}x{block_m}xf32>
    CHECK: linalg.transpose {{{{.*}}}} outs({{{{.*}}}} : tensor<{lhs_block_k}x{block_m}xf32>) permutation = [1, 0]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32{{{{.*}}}} : tensor<{lhs_block_k}x{lhs_block_m}xf32>
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{N}{{{{.*}}}} step %c{block_n}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32{{{{.*}}}} : tensor<{rhs_block_k}x{rhs_block_n}xf32>
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{block_m}{{{{.*}}}} step %c{tile_m}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{block_n}{{{{.*}}}} step %c{tile_n}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 2 : i32{{{{.*}}}} : tensor<{tile_m}x{tile_n}xf32>
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{K}{{{{.*}}}} step %c{tile_k}
    CHECK: tensor.extract_slice %{{{{.*}}}} [{tile_k}, {tile_m}]
    CHECK: tensor.extract_slice %{{{{.*}}}} [{tile_k}, {tile_n}]
    CHECK: linalg.matmul_transpose_a ins(%{{{{.*}}}}, %{{{{.*}}}} : tensor<{tile_k}x{tile_m}xf32>, tensor<{tile_k}x{tile_n}xf32>)
    CHECK-SAME: outs(%{{{{.*}}}} : tensor<{tile_m}x{tile_n}xf32>)
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        matmul_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


def test_matmul_simple_256():
    """
    Simple test for 256x256 matmul to verify 6-level blocking.

    KnobDrivenTiling generates 6-level blocking for matmul:
      for block_m in [0, M, BLOCK_M):      // BLOCK_M = TILE_M * 2
        for block_n in [0, N, BLOCK_N):    // BLOCK_N = TILE_N * 2
          for tile_m in [0, BLOCK_M, TILE_M):
            for tile_n in [0, BLOCK_N, TILE_N):
              for k in [0, K, TILE_K):
                matmul (linalg.matmul_transpose_a)
    
    For 256x256x256 with tile_size=[128, 128], reduction_tile=[128]:
      BLOCK_M = 256, BLOCK_N = 256
      Steps: 256, 256, 128, 128, 128
    """
    M, N, K = 256, 256, 256
    tile_m, tile_n, tile_k = 128, 128, 128
    tile_size = [tile_m, tile_n]
    reduction_tile = [tile_k]

    # Block size = tile * 2 (dim is large enough for blocking)
    block_m = tile_m * 2  # 256
    block_n = tile_n * 2  # 256

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def matmul_kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    # FileCheck verification for 6-level blocking with input promotion
    # LHS block size: [K, block_m] (transposed)
    # RHS block size: [K, block_n]
    lhs_block_k = K
    lhs_block_m = block_m
    rhs_block_k = K
    rhs_block_n = block_n

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{M}{{{{.*}}}} step %c{block_m}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32{{{{.*}}}} : tensor<{lhs_block_k}x{block_m}xf32>
    CHECK: linalg.transpose {{{{.*}}}} outs({{{{.*}}}} : tensor<{lhs_block_k}x{block_m}xf32>) permutation = [1, 0]
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32{{{{.*}}}} : tensor<{lhs_block_k}x{lhs_block_m}xf32>
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{N}{{{{.*}}}} step %c{block_n}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 3 : i32{{{{.*}}}} : tensor<{rhs_block_k}x{rhs_block_n}xf32>
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{block_m}{{{{.*}}}} step %c{tile_m}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{block_n}{{{{.*}}}} step %c{tile_n}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 2 : i32{{{{.*}}}} : tensor<{tile_m}x{tile_n}xf32>
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{K}{{{{.*}}}} step %c{tile_k}
    CHECK: tensor.extract_slice %{{{{.*}}}} [{tile_k}, {tile_m}]
    CHECK: tensor.extract_slice %{{{{.*}}}} [{tile_k}, {tile_n}]
    CHECK: linalg.matmul_transpose_a ins(%{{{{.*}}}}, %{{{{.*}}}} : tensor<{tile_k}x{tile_m}xf32>, tensor<{tile_k}x{tile_n}xf32>)
    CHECK-SAME: outs(%{{{{.*}}}} : tensor<{tile_m}x{tile_n}xf32>)
    CHECK: tensor.insert_slice
    CHECK: scf.yield
    """
    run_kernel_test(
        matmul_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Batch Matmul Tests
# ============================================================================

BATCH_TEST_CONFIGS = [
    pytest.param(8, 256, 256, 256, [1, 128, 128], [128], id="b8_256x256x256_tile128"),
    pytest.param(4, 512, 512, 512, [1, 128, 128], [128], id="b4_512x512x512_tile128"),
]


@pytest.mark.parametrize("B,M,N,K,tile_size,reduction_tile", BATCH_TEST_CONFIGS)
def test_batch_matmul_tiling(B, M, N, K, tile_size, reduction_tile, request):
    """
    Test knob-driven-tiling on batch_matmul with 6-level blocking.

    Same blocking structure as matmul, with a leading batch dim tiled at 1.
    """
    tile_m = tile_size[-2]
    tile_n = tile_size[-1]
    tile_k = reduction_tile[0]

    block_m = tile_m * 2
    block_n = tile_n * 2

    @trace(input_specs=[((B, M, K), "f32"), ((B, K, N), "f32")])
    def batch_matmul_kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    lhs_block_k = K
    lhs_block_m = block_m
    rhs_block_k = K
    rhs_block_n = block_n

    # Batch matmul: same 6-level blocking but with batch dim
    # After tiling batch dim at 1, inner ops become 2D slices
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{B}{{{{.*}}}} step %c1
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{M}{{{{.*}}}} step %c{block_m}
    CHECK: linalg.transpose
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{N}{{{{.*}}}} step %c{block_n}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{block_m}{{{{.*}}}} step %c{tile_m}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{block_n}{{{{.*}}}} step %c{tile_n}
    CHECK: bufferization.alloc_tensor() {{{{.*}}}}memory_space = 2 : i32{{{{.*}}}}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{K}{{{{.*}}}} step %c{tile_k}
    CHECK: linalg.matmul_transpose_a
    """
    run_kernel_test(
        batch_matmul_kernel,
        stop_after='apply-and-strip-transforms',
        check_patterns=check_patterns,
        modes=Mode.LLVM | Mode.FILECHECK,
    )


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_matmul_k_tile_too_large():
    """
    Test that K tile size larger than K dimension produces an error.
    """
    M, N, K = 256, 256, 64  # K is small
    tile_size = [64, 64]        # output tile
    reduction_tile = [128]      # K tile (128) is larger than K dimension (64)

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def matmul_kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    # This should raise an exception
    with pytest.raises(Exception) as excinfo:
        compile_knob_pipeline(matmul_kernel, stop_after='apply-and-strip-transforms')

    error_msg = str(excinfo.value)

    # Check for the exact error message from KnobDrivenTiling.cpp
    expected_error = "matmul K tile (128) is larger than K dimension (64)"
    assert expected_error in error_msg, \
        f"Expected error message:\n  {expected_error}\nGot:\n  {error_msg}"


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
