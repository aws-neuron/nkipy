"""
Tests for eliminate-uninitialized-copies pass.

The eliminate-uninitialized-copies pass:
1. Finds memref.copy ops where source is a fresh alloc with no prior writes
2. Eliminates such copies since they copy undefined values
3. Common pattern: copy from SBUF subview to PSUM for accumulator init

Run with: python -m pytest tests/passes/eliminate_uninitialized_copies/test_basic.py -v
Or directly: python tests/passes/eliminate_uninitialized_copies/test_basic.py
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
    1. Eliminates the copy from uninitialized SBUF subview to PSUM accumulator
       Pattern before:
         %alloc = memref.alloc() : memref<256x256xf32>           // matmul output (uninit)
         %subview_7 = memref.subview %subview_4...               // subview chain to %alloc
         %alloc_8 = memref.alloc() : memref<128x128xf32, #psum>  // PSUM accumulator
         memref.copy %subview_7, %alloc_8                        // THIS GETS ELIMINATED
         linalg.matmul ... outs(%alloc_8)                        // matmul writes to PSUM

       Pattern after:
         %alloc = memref.alloc() : memref<256x256xf32>
         %alloc_8 = memref.alloc() : memref<128x128xf32, #psum>
         linalg.matmul ... outs(%alloc_8)                        // No copy before matmul!

    2. Preserves necessary copies (those with initialized source data)
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
    # NOTE: At this point in the pipeline, annotate-memory-space hasn't run yet.
    # - Matmul output alloc is memref<256x256xf32> (no mem space - nkipy.annotate exists but not applied)
    # - Add output alloc is memref<256x256xf32> (no mem space)
    # - BUT promoted buffers DO have memory spaces (from tiling transform)
    #
    # COPIES THAT SHOULD BE ELIMINATED (uninitialized source):
    # 1. Copy from uninitialized SBUF subview to PSUM accumulator before matmul
    #
    # COPIES THAT MUST BE PRESERVED (initialized source):
    # 2. Copy from transpose temp to SBUF (LHS promote)
    # 3. Copy from arg1 to SBUF (RHS promote)
    # 4. Copy from PSUM to matmul output subview (matmul result writeback)
    # 5. Copy from matmul output subview to add's promoted SBUF input (INITIALIZED!)
    # 6. Copy from bias arg subview to add's promoted SBUF input
    # 7. Copy from add's SBUF output to add output subview
    check_patterns = '''
CHECK: func.func @matmul_add_kernel
CHECK: memref.alloc(){{.*}}: memref<256x256xf32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: linalg.transpose{{.*}}outs({{.*}}memref<256x256xf32, 3 : i32>)
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 3 : i32>
CHECK: memref.copy{{.*}}to memref<256x256xf32, 3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 2 : i32>
CHECK-NOT: memref.copy{{.*}}to memref<128x128xf32, 2 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}3 : i32>
CHECK: linalg.matmul
CHECK: memref.subview
CHECK: memref.copy{{.*}}2 : i32>{{.*}}to{{.*}}memref<128x128xf32
CHECK: nkipy.annotate
CHECK: memref.alloc(){{.*}}: memref<256x256xf32>
CHECK: scf.for
CHECK: scf.for
CHECK: memref.subview
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: memref.copy{{.*}}to{{.*}}memref<128x128xf32, 3 : i32>
CHECK: memref.subview
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: memref.copy{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: linalg.add
CHECK: memref.subview
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}memref<128x128xf32
CHECK: nkipy.annotate
CHECK: return
'''
    run_kernel_test(
        matmul_add_kernel,
        stop_after='eliminate-uninitialized-copies',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
