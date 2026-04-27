"""
Tests for canonicalize-loop-step pass with elementwise operations.

Run with: python -m pytest tests/passes/canonicalize_loop_step/test_elementwise.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Configurations
# ============================================================================

# Elementwise add test configurations: (shape, tile_size)
ADD_CONFIGS = [
    pytest.param((256, 256), [128, 128], id="256x256_tile128"),
    pytest.param((512, 512), [128, 128], id="512x512_tile128"),
    pytest.param((256, 256), [64, 64], id="256x256_tile64"),
    pytest.param((512, 512), [64, 64], id="512x512_tile64_8iters"),
    pytest.param((256, 256), [256, 256], id="256x256_single_iter"),
]

# 3D elementwise configurations
ADD_3D_CONFIGS = [
    pytest.param((128, 256, 64), [64, 128, 32], id="3d_128x256x64"),
    pytest.param((64, 128, 256), [32, 64, 128], id="3d_64x128x256"),
]


# ============================================================================
# 2D Elementwise Tests
# ============================================================================

@pytest.mark.parametrize("shape,tile_size", ADD_CONFIGS)
def test_add_loop_canonicalization(shape, tile_size):
    """
    Test loop step canonicalization on 2D elementwise add.

    After canonicalization:
    - Loop bounds change from (0 to X step Y) to (0 to X/Y step 1)
    - Original offset is recovered via arith.muli: offset = loop_idx * original_step
    """
    M, N = shape
    tile_m, tile_n = tile_size

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def add_kernel(a, b):
        result = a + b
        knob.knob(result, tile_size=tile_size)
        return result

    # Strict checks:
    # 1. Loop bounds: step is always 1
    # 2. arith.muli to recover original offset with tile size constant
    # 3. Loop upper bound matches num_iters (but constant naming may vary with suffix)
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{tile_m}{{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{tile_n}{{{{.*}}}} : index
    CHECK: linalg.add
    """
    run_kernel_test(
        add_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


@pytest.mark.parametrize("shape,tile_size", ADD_CONFIGS)
def test_mul_loop_canonicalization(shape, tile_size):
    """
    Test loop step canonicalization on 2D elementwise mul.

    After canonicalization:
    - Loop bounds change from (0 to X step Y) to (0 to X/Y step 1)
    - Original offset is recovered via arith.muli: offset = loop_idx * original_step
    """
    M, N = shape
    tile_m, tile_n = tile_size

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def mul_kernel(a, b):
        result = a * b
        knob.knob(result, tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{tile_m}{{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{tile_n}{{{{.*}}}} : index
    CHECK: linalg.mul
    """
    run_kernel_test(
        mul_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# 3D Elementwise Tests
# ============================================================================

@pytest.mark.parametrize("shape,tile_size", ADD_3D_CONFIGS)
def test_add_3d_loop_canonicalization(shape, tile_size):
    """
    Test loop step canonicalization on 3D elementwise add.

    For 3D tensors, we expect 3 nested scf.for loops with step 1,
    each with arith.muli to recover original offsets.
    """
    D0, D1, D2 = shape
    t0, t1, t2 = tile_size

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def add_kernel(a, b):
        result = a + b
        knob.knob(result, tile_size=tile_size)
        return result

    # Strict checks: 3 nested loops with arith.muli for each offset recovery
    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{t0}{{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{t1}{{{{.*}}}} : index
    CHECK: scf.for {{{{.*}}}} step %c1{{{{.*}}}}
    CHECK: arith.muli {{{{.*}}}}, %c{t2}{{{{.*}}}} : index
    CHECK: linalg.add
    """
    run_kernel_test(
        add_kernel,
        stop_after='canonicalize-loop-step',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
