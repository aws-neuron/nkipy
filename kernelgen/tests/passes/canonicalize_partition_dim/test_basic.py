"""
Tests for the canonicalize-partition-dim pass.

This pass inserts linalg.transpose operations to ensure partition_dim=0
everywhere. It operates on connected components of elementwise ops that
share a non-zero partition_dim, inserting transposes at component boundaries
and rewriting elementwise ops with permuted shapes.

Run with: python -m pytest tests/passes/canonicalize_partition_dim/ -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test: No-op when partition_dim=0 (default)
# ============================================================================

def test_partition_dim_zero_is_noop():
    """
    When all annotations have partition_dim=0 (or no partition_dim),
    the pass should be a no-op: no transposes inserted.
    """
    shape = (256, 256)

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        y = np.exp(x)
        knob.knob(y, mem_space="Sbuf", tile_size=[128, 128], partition_dim=0)
        return y

    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_not_contains=["linalg.transpose"],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test: Single op with partition_dim=1 (2D)
# ============================================================================

def test_single_op_partition_dim_1():
    """
    A single elementwise op annotated with partition_dim=1 on a 2D tensor.

    Input:  y = exp(x)  : tensor<256x128xf32>, partition_dim=1
    After:  transpose inputs [1,0] -> exp on tensor<128x256xf32> -> transpose output [1,0]
            annotation updated to partition_dim=0, tile_size permuted
    """
    M, N = 256, 128

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        y = np.exp(x)
        knob.knob(y, mem_space="Sbuf", tile_size=[M, N], partition_dim=1)
        return y

    # After the pass, we expect:
    # - linalg.transpose ops inserted at boundaries
    # - partition_dim updated to 0
    # - tile_size permuted from [256, 128] to [128, 256]
    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_contains=[
            "linalg.transpose",
            "permutation = [1, 0]",
            "linalg.exp",
            "tensor<128x256xf32>",
            "partition_dim = 0",
            "tile_size = array<i64: 128, 256>",
        ],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test: Elementwise chain with partition_dim=1
# ============================================================================

def test_elementwise_chain_partition_dim_1():
    """
    A chain of elementwise ops where the final op is annotated with
    partition_dim=1. After infer-layout propagates partition_dim backward,
    the entire chain should be rewritten.

    Input:  y = exp(x), z = y + 1.0, knob on z with partition_dim=1
    After:  transpose x -> exp -> add -> transpose back
            All ops in the chain have permuted shapes.
    """
    M, N = 256, 128

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        y = np.exp(x)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=[M, N], partition_dim=1)
        return z

    # After the pass:
    # - One input transpose at the boundary
    # - Elementwise ops rewritten with permuted shapes
    # - One output transpose at the boundary
    # - partition_dim=0 in all annotations
    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_contains=[
            "linalg.transpose",
            "permutation = [1, 0]",
            "linalg.exp",
            "linalg.generic",
            "tensor<128x256xf32>",
            "partition_dim = 0",
            "tile_size = array<i64: 128, 256>",
        ],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test: Numerical correctness after tiling (partition_dim=1)
# ============================================================================

def test_partition_dim_1_tiling_executes():
    """
    Verify that after canonicalize-partition-dim, the pipeline can tile
    and execute the kernel correctly via LLVM JIT.

    This is the key correctness check: the transposes preserve semantics.
    """
    M, N = 256, 128

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        y = np.exp(x)
        z = y + 1.0
        knob.knob(z, mem_space="Sbuf", tile_size=[M, N], partition_dim=1)
        return z

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test: 3D tensor with partition_dim=2
# ============================================================================

def test_3d_partition_dim_2():
    """
    A 3D tensor annotated with partition_dim=2.

    Input:  y = x + 1.0 : tensor<4x256x128xf32>, partition_dim=2
    After:  permutation [2, 0, 1] moves dim 2 to position 0
            tensor becomes <128x4x256xf32>
            tile_size permuted accordingly
    """
    B, M, N = 4, 256, 128

    @trace(input_specs=[((B, M, N), "f32")])
    def kernel(x):
        y = x + 1.0
        knob.knob(y, mem_space="Sbuf", tile_size=[1, M, N], partition_dim=2)
        return y

    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_contains=[
            "linalg.transpose",
            "permutation = [2, 0, 1]",
            "linalg.generic",
            "tensor<128x4x256xf32>",
            "partition_dim = 0",
            "tile_size = array<i64: 128, 1, 256>",
            "permutation = [1, 2, 0]",
        ],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Test: 3D numerical correctness
# ============================================================================

def test_3d_partition_dim_2_executes():
    """
    Verify 3D tensor with partition_dim=2 executes correctly after tiling.
    """
    B, M, N = 4, 256, 128

    @trace(input_specs=[((B, M, N), "f32")])
    def kernel(x):
        y = x + 1.0
        knob.knob(y, mem_space="Sbuf", tile_size=[1, M, N], partition_dim=2)
        return y

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test: 3D broadcast generic with partition_dim=1
# ============================================================================

def test_3d_broadcast_generic_partition_dim_1():
    """
    Verify that canonicalize-partition-dim correctly permutes indexing maps
    of broadcast linalg.generic ops.

    Input:  a(4,128,64) * b(1,128,64) with partition_dim=1
    After perm=[1,0,2]:
      - a becomes (128,4,64), b becomes (128,1,64)
      - The broadcast generic's indexing map for b must change from
        (d0,d1,d2)->(0,d1,d2) to (d0,d1,d2)->(d0,0,d2)

    Without the fix: shapes are permuted but indexing maps are not,
    causing a verifier error ('inferred shape dimension #1 to be 4,
    but found 1').
    """
    BH, S, D = 4, 128, 64
    tile_size = [1, 128, 64]

    @trace(input_specs=[
        ((BH, S, D), "f32"),
        ((1, S, D), "f32"),
    ])
    def kernel(a, b):
        result = a * b
        knob.knob(result, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)
        return result

    # After the pass, shapes should be permuted and the broadcast generic
    # should verify correctly with updated indexing maps.
    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_contains=[
            "linalg.transpose",
            "permutation = [1, 0, 2]",
            "tensor<128x4x64xf32>",
            "tensor<128x1x64xf32>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_3d_broadcast_generic_executes():
    """
    Verify that a 3D broadcast multiply with partition_dim=1 produces
    correct numerical results through tiling after canonicalization.
    """
    BH, S, D = 4, 128, 64
    tile_size = [1, 128, 64]

    @trace(input_specs=[
        ((BH, S, D), "f32"),
        ((1, S, D), "f32"),
    ])
    def kernel(a, b):
        result = a * b
        knob.knob(result, mem_space="Sbuf", tile_size=tile_size,
                  partition_dim=1)
        return result

    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test: Matmul with partition_dim errors
# ============================================================================

def test_matmul_partition_dim_errors():
    """
    partition_dim on a matmul/bmm should error out.
    Users must split annotations: no partition_dim on the matmul itself.
    """
    M, K, N = 128, 64, 128

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def kernel(a, b):
        c = np.matmul(a, b)
        knob.knob(c, mem_space="Sbuf", tile_size=[M, N],
                  reduction_tile=[K], partition_dim=1)
        return c

    with pytest.raises(RuntimeError, match="matmul"):
        run_kernel_test(
            kernel,
            stop_after='canonicalize-partition-dim',
            check_ir_contains=["should_not_get_here"],
            modes=Mode.STRING_CHECK,
        )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
