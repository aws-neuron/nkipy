"""
Tests for canonicalize-partition-dim pass with reduction ops.

Verifies that reductions (np.max, np.sum) with non-zero partition_dim
get their shapes correctly permuted and produce correct results.

Run with: python -m pytest tests/passes/canonicalize_partition_dim/test_reduce.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test: 3D reduction (max) with partition_dim=1
# ============================================================================

def test_3d_reduction_max_partition_dim_1():
    """
    3D np.max with keepdims=True and partition_dim=1.

    Input: tensor<8x128x64xf32>, reduce axis=-1, keepdims=True
    Output: tensor<8x128x1xf32> with partition_dim=1

    After canonicalization (perm=[1,0,2]):
      - Input becomes tensor<128x8x64xf32>
      - Reduction output becomes tensor<128x8x1xf32>
    """
    B, M, N = 8, 128, 64

    @trace(input_specs=[((B, M, N), "f32")])
    def kernel(x):
        sq = x * x
        knob.knob(sq, mem_space="Sbuf", tile_size=[1, M, N], partition_dim=1)

        sm = np.max(sq, axis=-1, keepdims=True)
        knob.knob(sm, mem_space="SharedHbm", tile_size=[1, M],
                  reduction_tile=[N], partition_dim=1)
        return sm

    # String check: verify transposes and permuted shapes
    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_contains=[
            "linalg.transpose",
            "permutation = [1, 0, 2]",
            "tensor<128x8x1xf32>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )

    # Numerical correctness via LLVM JIT
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test: 3D reduction (sum) with partition_dim=1
# ============================================================================

def test_3d_reduction_sum_partition_dim_1():
    """
    3D np.sum with keepdims=True and partition_dim=1.

    Verifies that reduction with linalg.fill(tensor.empty) init gets
    its shapes correctly permuted, and produces correct results.
    """
    B, M, N = 8, 128, 64

    @trace(input_specs=[((B, M, N), "f32")])
    def kernel(x):
        y = x + 1.0
        knob.knob(y, mem_space="Sbuf", tile_size=[1, M, N], partition_dim=1)

        sm = np.sum(y, axis=-1, keepdims=True)
        knob.knob(sm, mem_space="SharedHbm", tile_size=[1, M],
                  reduction_tile=[N], partition_dim=1)
        return sm

    # String check: verify transposes and permuted shapes
    run_kernel_test(
        kernel,
        stop_after='canonicalize-partition-dim',
        check_ir_contains=[
            "linalg.transpose",
            "permutation = [1, 0, 2]",
            "tensor<128x8x64xf32>",
            "tensor<128x8x1xf32>",
            "partition_dim = 0",
        ],
        modes=Mode.STRING_CHECK,
    )

    # Numerical correctness via LLVM JIT
    run_kernel_test(
        kernel,
        stop_after='apply-and-strip-transforms',
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
