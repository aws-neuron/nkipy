"""
Tests for legalize-layout pass.

The legalize-layout pass:
1. Transforms SBUF memrefs to (R+2)-D physical layout
   - 2D: [tileM, numTilesM, numTilesN, tileN]       (4D physical)
   - 3D: [tileP, numB0, numB1, numB2, tileF]         (5D physical)
2. Updates memref.subview ops to use physical indexing
3. Tiles HBM↔SBUF memref.copy and linalg.transpose into looped tile transfers
4. Collapses all (R+2)-D and R-D memrefs to 2D for NISA consumption

Run with: python -m pytest tests/passes/legalize_layout/test_basic.py -v
Or directly: python tests/passes/legalize_layout/test_basic.py
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
    1. Transforms 2D SBUF allocs to 4D: memref<256x256xf32, #sbuf> -> memref<128x2x2x128xf32, #sbuf>
    2. Tiles HBM→SBUF transpose into nested loops
    3. Transforms subview ops to use 4D indexing

    Before the pass (annotate-memory-space output):
        %alloc = memref.alloc() : memref<256x256xf32, 3 : i32>
        linalg.transpose ins(%hbm_input) outs(%alloc)
        scf.for ... {
          %subview = memref.subview %alloc[%off_m, 0][128, 256]...
          // ... linalg ops using 2D tiles ...
        }

    After the pass (legalize-layout output):
        %alloc_4d = memref.alloc() : memref<128x2x2x128xf32, 3 : i32>
        scf.for %blk_m = 0 to 2 {
          scf.for %blk_n = 0 to 2 {
            %input_tile = memref.subview %hbm_input[...]
            %output_tile = memref.subview %alloc_4d[0, %blk_m, %blk_n, 0][128, 1, 1, 128]
            linalg.transpose ins(%input_tile) outs(%output_tile)
          }
        }
        scf.for ... {
          %subview_4d = memref.subview %alloc_4d[0, %blk, 0, 0][128, 1, 2, 128]...
          // ... linalg ops using 4D tiles with rank reduction ...
        }
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
    # After legalize-layout, we should see:
    # 1. 4D SBUF allocations: memref<128x2x2x128xf32, 3 : i32>
    # 2. Tiled transpose loops (HBM→SBUF)
    # 3. 4D subview operations with rank reduction

    check_patterns = '''
CHECK: func.func @matmul_add_kernel
CHECK-SAME: 4 : i32
CHECK: memref.alloc(){{.*}}: memref<128x2x2x128xf32, 3 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: linalg.transpose{{.*}}outs({{.*}}memref<128x{{.*}}128xf32{{.*}}3 : i32>)
CHECK: memref.alloc(){{.*}}: memref<128x2x2x128xf32, 3 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: memref.copy{{.*}}to{{.*}}memref<128x{{.*}}128xf32{{.*}}3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}3 : i32>
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 2 : i32>
CHECK: linalg.matmul
CHECK: memref.copy{{.*}}2 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: }
CHECK: }
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 4 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: memref.subview{{.*}}4 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: memref.copy{{.*}}4 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 3 : i32>
CHECK: linalg.add
CHECK: memref.copy{{.*}}3 : i32>{{.*}}to{{.*}}4 : i32>
CHECK: return{{.*}}memref<256x256xf32, 4 : i32>
'''
    run_kernel_test(
        matmul_add_kernel,
        stop_after='legalize-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_3d_add_chain_sbuf():
    """
    Test 3D add chain with SBUF intermediate:
    - Shape: (256, 2, 256) — 3D tensor where dim0 > 128 triggers legalization
    - tile_size: [128, 1, 128] — middle dim must have tile=1 (design constraint)
    - intermediate (a+b) goes to SBUF, result (intermediate+c) goes to SharedHbm

    This tests that the pass correctly handles rank-3 tensors:
    1. Transforms 3D SBUF alloc to 5D physical layout:
       memref<256x2x256xf32, #sbuf> -> memref<128x2x2x2x128xf32, #sbuf>
       Physical shape: [tileP=128, numB0=2, numB1=2, numB2=2, tileF=128]
    2. Tiles HBM↔SBUF memref.copy into 3-level nested loops
    3. Collapses (5D physical) and (3D logical) memrefs to 2D for compute
    4. Reconstructs linalg ops with 2D iteration domain
    """
    B, M, N = 256, 2, 256
    tile_size = [128, 1, 128]

    @trace(input_specs=[((B, M, N), "f32"), ((B, M, N), "f32"), ((B, M, N), "f32")])
    def add_chain_3d(a, b, c):
        # First add: intermediate stored in SBUF
        intermediate = a + b
        knob.knob(intermediate, mem_space="Sbuf", tile_size=tile_size)

        # Second add: result goes to SharedHbm
        result = intermediate + c
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)

        return result

    # FileCheck patterns to verify 3D -> 5D legalization:
    #
    # 1. 5D SBUF allocation: memref<128x2x2x2x128xf32, 3 : i32>
    # 2. 3-level tiled copy loops (HBM→SBUF)
    # 3. memref.collapse_shape to 2D for compute ops
    # 4. 2D linalg.add (after reconstruction from 3D)
    check_patterns = '''
CHECK: func.func @add_chain_3d
CHECK: memref.alloc(){{.*}}: memref<128x2x2x2x128xf32, 3 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: scf.for
CHECK: memref.collapse_shape
CHECK: memref.alloc(){{.*}}: memref<256x2x256xf32, 4 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: scf.for
CHECK: memref.subview{{.*}}3 : i32>
CHECK: linalg.add
CHECK: return{{.*}}memref<256x2x256xf32, 4 : i32>
'''
    run_kernel_test(
        add_chain_3d,
        stop_after='legalize-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
