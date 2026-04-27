"""
Tests for eliminate-same-memspace-copy pass.

The eliminate-same-memspace-copy pass:
1. Removes memref.copy ops where source and target are subviews of the same base
   with the same offsets (i.e., they're copying to themselves)
2. This commonly occurs when tiling generates copy-back operations for promoted
   buffers that happen to be the same as the original memory region

Run with: python -m pytest tests/passes/eliminate_same_memspace_copy/test_basic.py -v
Or directly: python tests/passes/eliminate_same_memspace_copy/test_basic.py
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Cases
# ============================================================================

def test_matmul_sbuf_add_hbm():
    """
    Test matmul-add chain with:
    - matmul output -> SBUF (intermediate)
    - add output -> SharedHbm (returned result)

    This tests that the pass correctly:
    1. Removes redundant SBUF->SBUF copies where source and target are the same region

    Before the pass (in annotate-memory-space output), we have:
        %subview_4 = memref.subview %alloc[%0, 0] [128, 256] ...
        ... (inner loops) ...
        %subview_5 = memref.subview %alloc[%0, 0] [128, 256] ...  // SAME region!
        memref.copy %subview_4, %subview_5  // REDUNDANT - copying to itself

    After the pass:
        The redundant memref.copy is removed since both subviews access
        the same memory region of %alloc.
    """
    M, N, K = 256, 256, 256
    matmul_tile = [128, 128]       # TILE_M, TILE_N
    matmul_reduction_tile = [128]  # TILE_K
    add_tile = [128, 128]          # TILE_M, TILE_N

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        # Matmul outputs to SBUF for reuse in the add
        c = np.matmul(a, b)
        knob.knob(c, mem_space="Sbuf", tile_size=matmul_tile, reduction_tile=matmul_reduction_tile)

        # Add outputs to SharedHbm (returned from kernel)
        result = c + bias
        knob.knob(result, mem_space="SharedHbm", tile_size=add_tile)

        return result

    # FileCheck patterns to verify:
    #
    # After eliminate-same-memspace-copy, we should see:
    # 1. All the valid copies preserved (HBM->SBUF, SBUF->HBM, PSUM->SBUF, etc.)
    # 2. The redundant SBUF->SBUF self-copy at the end of matmul M-loop should be GONE
    #
    # The matmul loop structure:
    #   scf.for %arg3 (M loop)
    #     %subview = subview LHS promoted
    #     %subview_4 = subview %alloc (matmul output tile)
    #     scf.for %arg4 (N loop)
    #       ... matmul inner loops ...
    #       memref.copy %psum, %subview_8 (writeback from PSUM to SBUF subview)
    #     } end N loop
    #     // REMOVED: memref.copy %subview_4, %subview_5 (was redundant)
    #   } end M loop
    #
    # After the pass runs, the redundant copy should not appear.
    # We verify by checking the structure and that no SBUF->SBUF copy exists
    # right after the inner N loop closes.

    check_patterns = '''
CHECK: func.func @matmul_add_kernel
CHECK-SAME: 4 : i32
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: linalg.transpose{{.*}}outs({{.*}}memref<256x256xf32, 3 : i32>)
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.copy{{.*}}to memref<256x256xf32, 3 : i32>
CHECK-NOT: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 2 : i32>
CHECK: scf.for
CHECK: linalg.matmul
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.copy{{.*}}2 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: }
CHECK: }
CHECK-NOT: memref.copy{{.*}}to memref<128x256xf32{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 4 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}4 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: memref.copy{{.*}}4 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: linalg.add{{.*}}memref<128x128xf32, strided{{.*}}3 : i32>{{.*}}memref<128x128xf32, 3 : i32>{{.*}}memref<128x128xf32, 3 : i32>
CHECK: memref.subview{{.*}}4 : i32>
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}4 : i32>
CHECK: return{{.*}}memref<256x256xf32, 4 : i32>
'''
    run_kernel_test(
        matmul_add_kernel,
        stop_after='eliminate-same-memspace-copy',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
