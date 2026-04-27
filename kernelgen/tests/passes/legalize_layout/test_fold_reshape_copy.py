"""
Tests for legalize-layout handling of Phase 0 (foldReshapeIntoAlloc) patterns.

When an SBUF alloc is followed by a copy then a reshape, Phase 0 folds the
reshape into the alloc, creating a collapse_shape for the copy.  Phases 1-3
must follow through the collapse_shape to:
  - discover tile sizes (Phase 1 -- traceToLinalgOperands)
  - resolve the legalized SBUF alloc (Phase 3 -- tileMemrefCopy)
  - handle the HBM/SBUF rank mismatch when the reshape inserted dims (Phase 3)

These tests use crafted MLIR input (the output of annotate-memory-space) and
run only the legalize-layout pass to verify the transformation in isolation.

Run with: python -m pytest tests/passes/legalize_layout/test_fold_reshape_copy.py -v
"""

import pytest

from nkipy_kernelgen.transforms.nkipy_opt import run_nkipy_opt_passes
from passes.pass_utils import run_filecheck


# ============================================================================
# Helpers
# ============================================================================

def run_legalize_layout(mlir_input: str) -> str:
    """Run only the legalize-layout pass on the given MLIR."""
    return run_nkipy_opt_passes(mlir_input, ['legalize-layout'])


# ============================================================================
# Test: 2D SBUF alloc + copy + reshape to 3D
# ============================================================================

# This is the pattern produced by the upstream pipeline for:
#   cos_sbuf = alloc(256x64, sbuf); copy(hbm -> cos_sbuf); reshape -> 256x1x64
# The reshape is then subviewed [128,1,64] inside a tiled loop.

MLIR_2D_COPY_RESHAPE_3D = '''
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
module {
  memref.global "private" constant @__constant_3xindex : memref<3xindex> = dense<[256, 1, 64]> {alignment = 64 : i64}
  func.func @test_reshape_copy(
      %arg0: memref<256x1x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>,
      %arg1: memref<256x64xf32, strided<[?, ?], offset: ?>, 4 : i32>
  ) -> memref<256x1x64xf32, 4 : i32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_3xindex : memref<3xindex>

    // Pattern under test: 2D SBUF alloc + copy from HBM + reshape to 3D
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x64xf32, 3 : i32>
    memref.copy %arg1, %alloc : memref<256x64xf32, strided<[?, ?], offset: ?>, 4 : i32> to memref<256x64xf32, 3 : i32>
    %reshape = memref.reshape %alloc(%0) : (memref<256x64xf32, 3 : i32>, memref<3xindex>) -> memref<256x1x64xf32, 3 : i32>

    // Output buffer
    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<256x1x64xf32, 3 : i32>

    // Tiled loop using [128, 1, 64] tiles of the reshaped buffer
    // Linalg ops use 3D operands directly (Phase 4 handles collapse to 2D)
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %off = arith.muli %arg3, %c128 : index

      // Load a tile from HBM input
      %sv_hbm = memref.subview %arg0[%off, 0, 0] [128, 1, 64] [1, 1, 1]
        : memref<256x1x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
        to memref<128x1x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
      %tile_in = memref.alloc() {alignment = 64 : i64} : memref<128x1x64xf32, 3 : i32>
      memref.copy %sv_hbm, %tile_in
        : memref<128x1x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
        to memref<128x1x64xf32, 3 : i32>

      // Subview into the reshaped cos/sin buffer (the Phase 0 subject)
      %sv_reshape = memref.subview %reshape[%off, 0, 0] [128, 1, 64] [1, 1, 1]
        : memref<256x1x64xf32, 3 : i32>
        to memref<128x1x64xf32, strided<[64, 64, 1], offset: ?>, 3 : i32>

      // Compute: elementwise multiply using 3D operands
      %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x1x64xf32, 3 : i32>
      linalg.generic {indexing_maps = [#map, #map1, #map],
                       iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%tile_in, %sv_reshape
          : memref<128x1x64xf32, 3 : i32>,
            memref<128x1x64xf32, strided<[64, 64, 1], offset: ?>, 3 : i32>)
        outs(%tile_out : memref<128x1x64xf32, 3 : i32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %mul = arith.mulf %in, %in_1 : f32
        linalg.yield %mul : f32
      }

      // Store result tile
      %sv_out = memref.subview %alloc_out[%off, 0, 0] [128, 1, 64] [1, 1, 1]
        : memref<256x1x64xf32, 3 : i32>
        to memref<128x1x64xf32, strided<[64, 64, 1], offset: ?>, 3 : i32>
      memref.copy %tile_out, %sv_out
        : memref<128x1x64xf32, 3 : i32>
        to memref<128x1x64xf32, strided<[64, 64, 1], offset: ?>, 3 : i32>
    }

    // Copy result to HBM
    %alloc_hbm = memref.alloc() {alignment = 64 : i64} : memref<256x1x64xf32, 4 : i32>
    memref.copy %alloc_out, %alloc_hbm
      : memref<256x1x64xf32, 3 : i32>
      to memref<256x1x64xf32, 4 : i32>
    return %alloc_hbm : memref<256x1x64xf32, 4 : i32>
  }
}
'''


def test_2d_copy_reshape_3d_legalized():
    """
    The 256x64 SBUF alloc + copy + reshape(256x1x64) pattern must be legalized.

    Phase 0 folds the reshape into the alloc:
        alloc(256x1x64, sbuf) + collapse_shape(256x64) + copy(hbm, collapse)

    Then Phases 1-3 must:
    1. Trace through collapse_shape to find tile sizes [128, 1, 64]
    2. Legalize alloc to 128x2x1x1x64 physical layout
    3. Tile the HBM->SBUF copy with a loop, handling HBM rank (2) < SBUF rank (3)

    Checks:
    - The 256x1x64 SBUF alloc is replaced by a legalized 5D alloc
    - The HBM->SBUF copy is tiled (appears inside a scf.for)
    - No un-legalized 256x1x64 SBUF alloc remains
    """
    result = run_legalize_layout(MLIR_2D_COPY_RESHAPE_3D)

    check_patterns = '''
CHECK: func.func @test_reshape_copy
CHECK-NOT: memref.alloc(){{.*}}: memref<256x1x64xf32, 3 : i32>
CHECK-NOT: memref.alloc(){{.*}}: memref<256x64xf32, 3 : i32>
CHECK: memref.alloc(){{.*}}: memref<128x2x1x1x64xf32, 3 : i32>
CHECK: scf.for
CHECK: memref.copy{{.*}}4 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: linalg.generic
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test: 2D SBUF alloc (no reshape) — baseline regression
# ============================================================================

MLIR_2D_SBUF_BASELINE = '''
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_2d_baseline(
      %arg0: memref<256x128xf32, strided<[?, ?], offset: ?>, 4 : i32>,
      %arg1: memref<256x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
  ) -> memref<256x128xf32, 4 : i32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index

    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32, 3 : i32>

    scf.for %iv = %c0 to %c2 step %c1 {
      %off = arith.muli %iv, %c128 : index

      %sv_a = memref.subview %arg0[%off, 0] [128, 128] [1, 1]
        : memref<256x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
        to memref<128x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
      %tile_a = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, 3 : i32>
      memref.copy %sv_a, %tile_a
        : memref<128x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
        to memref<128x128xf32, 3 : i32>

      %sv_b = memref.subview %arg1[%off, 0] [128, 128] [1, 1]
        : memref<256x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
        to memref<128x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
      %tile_b = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, 3 : i32>
      memref.copy %sv_b, %tile_b
        : memref<128x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
        to memref<128x128xf32, 3 : i32>

      %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, 3 : i32>
      linalg.add ins(%tile_a, %tile_b
        : memref<128x128xf32, 3 : i32>,
          memref<128x128xf32, 3 : i32>)
        outs(%tile_out : memref<128x128xf32, 3 : i32>)

      %sv_out = memref.subview %alloc_out[%off, 0] [128, 128] [1, 1]
        : memref<256x128xf32, 3 : i32>
        to memref<128x128xf32, strided<[128, 1], offset: ?>, 3 : i32>
      memref.copy %tile_out, %sv_out
        : memref<128x128xf32, 3 : i32>
        to memref<128x128xf32, strided<[128, 1], offset: ?>, 3 : i32>
    }

    %alloc_hbm = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32, 4 : i32>
    memref.copy %alloc_out, %alloc_hbm
      : memref<256x128xf32, 3 : i32>
      to memref<256x128xf32, 4 : i32>
    return %alloc_hbm : memref<256x128xf32, 4 : i32>
  }
}
'''


def test_2d_sbuf_baseline():
    """
    Baseline: 2D SBUF alloc (256x128) without reshape legalizes normally.

    The 256x128 alloc with tile [128, 128] should become 128x2x1x128 (4D).
    No Phase 0 pattern is involved — this is the standard path.
    """
    result = run_legalize_layout(MLIR_2D_SBUF_BASELINE)

    check_patterns = '''
CHECK: func.func @test_2d_baseline
CHECK: memref.alloc(){{.*}}: memref<128x2x1x128xf32, 3 : i32>
CHECK: scf.for
CHECK: linalg.add
CHECK: return{{.*}}4 : i32
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test: 3D SBUF alloc with full-buffer copy from 3D HBM (no reshape, no rank mismatch)
# ============================================================================

MLIR_3D_SBUF_COPY = '''
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_3d_copy(
      %arg0: memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>,
      %arg1: memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
  ) -> memref<256x2x64xf32, 4 : i32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index

    // Full SBUF alloc loaded from HBM (no reshape -- ranks match)
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x2x64xf32, 3 : i32>
    memref.copy %arg1, %alloc
      : memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
      to memref<256x2x64xf32, 3 : i32>

    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<256x2x64xf32, 3 : i32>

    // Linalg ops use 3D operands directly (Phase 4 handles collapse to 2D)
    scf.for %i = %c0 to %c2 step %c1 {
      %off = arith.muli %i, %c128 : index
      scf.for %j = %c0 to %c2 step %c1 {
        %sv_a = memref.subview %arg0[%off, %j, 0] [128, 1, 64] [1, 1, 1]
          : memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
          to memref<128x1x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
        %tile_a = memref.alloc() {alignment = 64 : i64} : memref<128x1x64xf32, 3 : i32>
        memref.copy %sv_a, %tile_a
          : memref<128x1x64xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
          to memref<128x1x64xf32, 3 : i32>

        %sv_b = memref.subview %alloc[%off, %j, 0] [128, 1, 64] [1, 1, 1]
          : memref<256x2x64xf32, 3 : i32>
          to memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, 3 : i32>

        %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x1x64xf32, 3 : i32>
        linalg.generic {indexing_maps = [#map, #map, #map],
                         iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%tile_a, %sv_b
            : memref<128x1x64xf32, 3 : i32>,
              memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, 3 : i32>)
          outs(%tile_out : memref<128x1x64xf32, 3 : i32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %add = arith.addf %in, %in_1 : f32
          linalg.yield %add : f32
        }

        %sv_out = memref.subview %alloc_out[%off, %j, 0] [128, 1, 64] [1, 1, 1]
          : memref<256x2x64xf32, 3 : i32>
          to memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, 3 : i32>
        memref.copy %tile_out, %sv_out
          : memref<128x1x64xf32, 3 : i32>
          to memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, 3 : i32>
      }
    }

    %hbm = memref.alloc() {alignment = 64 : i64} : memref<256x2x64xf32, 4 : i32>
    memref.copy %alloc_out, %hbm
      : memref<256x2x64xf32, 3 : i32>
      to memref<256x2x64xf32, 4 : i32>
    return %hbm : memref<256x2x64xf32, 4 : i32>
  }
}
'''


def test_3d_sbuf_full_copy():
    """
    3D SBUF alloc (256x2x64) with full-buffer copy from 3D HBM.

    No reshape involved — HBM and SBUF have the same rank (3).
    The alloc should be legalized to 5D: 128x2x2x1x64.
    The full-buffer HBM->SBUF copy should be tiled into a 3-level loop.
    """
    result = run_legalize_layout(MLIR_3D_SBUF_COPY)

    check_patterns = '''
CHECK: func.func @test_3d_copy
CHECK: memref.alloc(){{.*}}: memref<128x2x2x1x64xf32, 3 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: scf.for
CHECK: memref.copy{{.*}}4 : i32>{{.*}}to{{.*}}3 : i32>
CHECK: linalg.generic
CHECK: return{{.*}}4 : i32
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test: expandTileShape with multi-non-unit collapse group (Fix 2)
# ============================================================================

# Pattern: 3D SBUF alloc (128, 2, 128) with collapse_shape [[0],[1,2]] → 2D (128, 256).
# A linalg op uses 2D tiles [128, 128].  expandTileShape must expand
# tile=128 for group [1,2] with srcShape=[2, 128].
#
# Before fix: expanded=[2, 64] → middle tile=2 ≠ 1 → REJECTED by legalize-layout.
# After fix:  expanded=[1, 128] → middle tile=1 → legalized to 5D physical.

MLIR_MULTI_NON_UNIT_COLLAPSE = '''
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_expand_tile_multi_non_unit(
      %arg0: memref<128x2x128xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
  ) -> memref<128x256xf32, 4 : i32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index

    // 3D SBUF alloc — partition at dim 0, batch at dim 1, free at dim 2
    %alloc_3d = memref.alloc() {alignment = 64 : i64} : memref<128x2x128xf32, 3 : i32>

    // Manually tiled HBM→SBUF copy
    scf.for %j = %c0 to %c2 step %c1 {
      %sv_hbm = memref.subview %arg0[0, %j, 0] [128, 1, 128] [1, 1, 1]
        : memref<128x2x128xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
        to memref<128x1x128xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
      %sv_sbuf = memref.subview %alloc_3d[0, %j, 0] [128, 1, 128] [1, 1, 1]
        : memref<128x2x128xf32, 3 : i32>
        to memref<128x1x128xf32, strided<[256, 128, 1], offset: ?>, 3 : i32>
      memref.copy %sv_hbm, %sv_sbuf
        : memref<128x1x128xf32, strided<[?, ?, ?], offset: ?>, 4 : i32>
        to memref<128x1x128xf32, strided<[256, 128, 1], offset: ?>, 3 : i32>
    }

    // collapse_shape [[0],[1,2]] → 2D (128, 256)
    // group [1,2] has srcShape=[2, 128] — multi-non-unit!
    %collapsed = memref.collapse_shape %alloc_3d [[0], [1, 2]]
      : memref<128x2x128xf32, 3 : i32> into memref<128x256xf32, 3 : i32>

    // Output in HBM (avoids legalization issues on the output side)
    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32, 4 : i32>

    // Tiled loop using 2D [128, 128] tiles of the collapsed view
    scf.for %j = %c0 to %c2 step %c1 {
      %off = arith.muli %j, %c128 : index

      %sv_in = memref.subview %collapsed[0, %off] [128, 128] [1, 1]
        : memref<128x256xf32, 3 : i32>
        to memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>

      %sv_out = memref.subview %alloc_out[0, %off] [128, 128] [1, 1]
        : memref<128x256xf32, 4 : i32>
        to memref<128x128xf32, strided<[256, 1], offset: ?>, 4 : i32>

      %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, 3 : i32>
      linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
        ins(%sv_in
          : memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>)
        outs(%tile_out : memref<128x128xf32, 3 : i32>) {
      ^bb0(%in: f32, %out: f32):
        %exp = math.exp %in : f32
        linalg.yield %exp : f32
      }

      memref.copy %tile_out, %sv_out
        : memref<128x128xf32, 3 : i32>
        to memref<128x128xf32, strided<[256, 1], offset: ?>, 4 : i32>
    }

    return %alloc_out : memref<128x256xf32, 4 : i32>
  }
}
'''


def test_expand_tile_multi_non_unit_collapse():
    """
    expandTileShape must handle collapse groups with multiple non-unit dims.

    Alloc: memref<128x2x128, sbuf> collapsed to 2D via [[0],[1,2]].
    Linalg ops use 2D tiles [128, 128].

    expandTileShape must expand tile=128 for group [1,2] (srcShape=[2,128]):
      Correct: [1, 128] — middle tile=1, legalize-layout accepts.
      Old bug: [2, 64]  — middle tile=2, legalize-layout rejects.

    After legalization the 3D alloc becomes 5D physical:
      tile=[128, 1, 128], numBlocks=[1, 2, 1] → [128, 1, 2, 1, 128]
    """
    result = run_legalize_layout(MLIR_MULTI_NON_UNIT_COLLAPSE)

    check_patterns = '''
CHECK: func.func @test_expand_tile_multi_non_unit
CHECK: memref.alloc(){{.*}}: memref<128x1x2x1x128xf32, 3 : i32>
CHECK: scf.for
CHECK: linalg.generic
CHECK: return{{.*}}4 : i32
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
