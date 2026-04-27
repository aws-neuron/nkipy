"""
Tests for canonicalize-loop-step pass with multiple operations.

Run with: python -m pytest tests/passes/canonicalize_loop_step/test_multi_op.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Multi-Op Tests: Matmul + Elementwise
# ============================================================================

def test_matmul_add_chain():
    """
    Test matmul followed by elementwise add (common pattern: C = A @ B + bias).

    Both operations should have canonicalized loop steps, and each loop
    should have arith.muli to recover original offsets.
    """
    M, N, K = 256, 256, 256
    matmul_tile = [128, 128]
    matmul_reduction_tile = [128]
    add_tile = [128, 128]

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        c = np.matmul(a, b)
        knob.knob(c, tile_size=matmul_tile, reduction_tile=matmul_reduction_tile)

        result = c + bias
        knob.knob(result, tile_size=add_tile)

        return result

    # Strict checks:
    # - Matmul has 5 nested loops (block-M, block-N, tile-M, tile-N, tile-K)
    # - Add has 2 nested loops (tile-M, tile-N)
    # All loops should have step 1 and arith.muli for offset recovery
    check_patterns = """
    CHECK: func.func
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: linalg.matmul
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: linalg.add
    CHECK: return
    """
    run_kernel_test(
        matmul_add_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_matmul_add_different_tile_sizes():
    """
    Test matmul + add with different tile sizes for each operation.

    Different tile sizes mean different constants in the arith.muli operations.
    """
    M, N, K = 512, 512, 256
    matmul_tile = [128, 128]
    matmul_reduction_tile = [64]
    add_tile = [64, 64]

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        c = np.matmul(a, b)
        knob.knob(c, tile_size=matmul_tile, reduction_tile=matmul_reduction_tile)
        
        result = c + bias
        knob.knob(result, tile_size=add_tile)

        return result

    # Strict checks:
    # - Matmul has 5 nested loops (block-M, block-N, tile-M, tile-N, tile-K)
    # - Add has 2 nested loops (tile-M, tile-N)
    # All loops should have step 1 and arith.muli for offset recovery
    check_patterns = """
    CHECK: func.func
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: linalg.matmul
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: scf.for {{.*}} step %c1{{.*}}
    CHECK: arith.muli {{.*}} : index
    CHECK: linalg.add
    """
    run_kernel_test(
        matmul_add_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Multi-Op Tests: Multiple Elementwise
# ============================================================================

def test_add_add_chain():
    """
    Test two elementwise adds in sequence: C = (A + B) + D.

    Both add operations should have canonicalized loops with arith.muli
    for offset recovery.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    tile_m, tile_n = tile_size

    @trace(input_specs=[(shape, "f32"), (shape, "f32"), (shape, "f32")])
    def add_add_kernel(a, b, d):
        c = a + b
        knob.knob(c, tile_size=tile_size)

        result = c + d
        knob.knob(result, tile_size=tile_size)

        return result

    # Strict checks: Two add operations, each with 2 nested loops
    # Each loop should have step 1 and arith.muli for offset recovery
    # Note: Using flexible pattern to handle various constant naming conventions
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{tile_m}{{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{tile_n}{{{{.*}}}} : index
    CHECK: linalg.add
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: linalg.add
    """
    run_kernel_test(
        add_add_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
