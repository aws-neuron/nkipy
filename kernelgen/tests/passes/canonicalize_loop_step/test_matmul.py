"""
Tests for canonicalize-loop-step pass with matmul operations.

The canonicalize-loop-step pass normalizes loop steps to 1:
- Before: for %i = 0 to 256 step 128  (2 iterations)
- After:  for %i = 0 to 2 step 1      (2 iterations)

The pass requires:
- Constant loop bounds and step
- Upper bound must be evenly divisible by step
- Lower bound must be 0

Run with: python -m pytest tests/passes/canonicalize_loop_step/test_matmul.py -v
Or directly: python tests/passes/canonicalize_loop_step/test_matmul.py
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Configurations
# ============================================================================

# Matmul test configurations: (M, N, K, tile_size, reduction_tile)
# Block size is dynamically chosen: 2 when dim >= tile*2, else 1
MATMUL_CONFIGS = [
    pytest.param(256, 256, 256, [128, 128], [128], id="256x256x256_tile128"),
    pytest.param(512, 512, 512, [128, 128], [128], id="512x512x512_tile128"),
    pytest.param(256, 256, 256, [64, 64], [64], id="256x256x256_tile64"),
    pytest.param(512, 512, 256, [128, 128], [64], id="512x512x256_tile_128_128_64"),
]

# ============================================================================
# Matmul Tests
# ============================================================================

@pytest.mark.parametrize("M,N,K,tile_size,reduction_tile", MATMUL_CONFIGS)
def test_matmul_loop_canonicalization(M, N, K, tile_size, reduction_tile, request):
    """
    Test that loop steps are normalized to 1 after canonicalization.

    After canonicalize-loop-step:
    - Loop bounds change from (0 to X step Y) to (0 to X/Y step 1)
    - Original offsets are recovered via arith.muli: offset = loop_idx * original_step

    Matmul has 5 nested loops after tiling:
    - 2 outer block loops (M, N dimensions)
    - 1 K reduction loop
    - 2 inner tile loops (M, N tile dimensions)
    """
    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32")])
    def matmul_kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    # Strict checks for matmul:
    # 1. All 5 loops should have step 1
    # 2. arith.muli should appear to recover offsets for each loop
    # Note: The exact order and tile sizes vary, so we check for key patterns
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}} : index
    CHECK: linalg.matmul
    """
    run_kernel_test(
        matmul_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
