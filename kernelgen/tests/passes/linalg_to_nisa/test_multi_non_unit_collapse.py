"""
Tests for linalg-to-nisa handling of collapse_shape with multi-non-unit groups.

After legalize-layout, a 4D SBUF tensor (e.g. (128, 2, 2, 128) from a head-
deconcat pattern) gets collapsed to 2D via collapse_shape [[0,1],[2,3]].  Both
groups have multiple non-unit dims ([128,2] and [2,128]).

getBaseAndOffsets must trace through this collapse to reach the SBUF alloc base,
because NCC requires NISA ops to reference alloc results.  For each multi-non-
unit group, the largest dim carries the data (partition or free tile), and
smaller dims are batch loop indices that should be dropped.

Run with: python -m pytest tests/passes/linalg_to_nisa/test_multi_non_unit_collapse.py -v
"""

import pytest

from nkipy_kernelgen.transforms.nkipy_opt import run_nkipy_opt_passes
from nkipy_kernelgen.transforms.linalg_to_nisa_py import linalg_to_nisa
from passes.pass_utils import run_filecheck


# ============================================================================
# Helpers
# ============================================================================

def run_linalg_to_nisa(mlir_input: str) -> str:
    """Run linalg-to-nisa on the given MLIR.

    simplify-linalg still runs in C++ (`nkipy-opt`); the actual linalg→NISA
    lowering was moved to Python as part of open-sourcing so we call the
    Python implementation directly here instead of shelling out to an
    `nkipy-opt --linalg-to-nisa` pass that no longer exists.

    Use ``print_generic=False`` so FileCheck patterns can reference the
    pretty form (e.g. ``nisa.tensor_tensor_arith`` rather than
    ``\"nisa.tensor_tensor_arith\"``).
    """
    simplified = run_nkipy_opt_passes(mlir_input, ['simplify-linalg'])
    return linalg_to_nisa(simplified, target='trn2', print_generic=False)


# ============================================================================
# Test: 4D SBUF alloc collapsed to 2D with multi-non-unit groups
# ============================================================================

# This represents the WA3 (head-deconcat) scenario after legalize-layout:
#   4D SBUF alloc: memref<128x2x2x128xf32, #sbuf>
#   collapse_shape [[0,1],[2,3]] → memref<256x256xf32, #sbuf>
#   subview [128, 128] tiles in a nested loop
#   linalg.add on the tiles
#
# getBaseAndOffsets must decompose the collapsed indices:
#   Group [0,1]: srcShape=[128,2] → dim 0 (128) is primary, dim 1 (2) is dropped
#   Group [2,3]: srcShape=[2,128] → dim 3 (128) is primary, dim 2 (2) is dropped

MLIR_MULTI_NON_UNIT_COLLAPSE = '''
module {
  func.func @test_multi_non_unit_collapse(
      %arg0: memref<256x256xf32, strided<[?, ?], offset: ?>, 4 : i32>
  ) -> memref<256x256xf32, 4 : i32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index

    // 4D SBUF alloc — physical layout from legalize-layout
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x2x2x128xf32, 3 : i32>

    // Load data from HBM into the 4D alloc tile by tile
    scf.for %bm = %c0 to %c2 step %c1 {
      scf.for %bn = %c0 to %c2 step %c1 {
        %hbm_off_m = arith.muli %bm, %c128 : index
        %hbm_off_n = arith.muli %bn, %c128 : index
        %sv_hbm = memref.subview %arg0[%hbm_off_m, %hbm_off_n] [128, 128] [1, 1]
          : memref<256x256xf32, strided<[?, ?], offset: ?>, 4 : i32>
          to memref<128x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
        %sv_sbuf = memref.subview %alloc[0, %bm, %bn, 0] [128, 1, 1, 128] [1, 1, 1, 1]
          : memref<128x2x2x128xf32, 3 : i32>
          to memref<128x1x1x128xf32, strided<[512, 256, 128, 1], offset: ?>, 3 : i32>
        %sv_sbuf_2d = memref.collapse_shape %sv_sbuf [[0], [1, 2, 3]]
          : memref<128x1x1x128xf32, strided<[512, 256, 128, 1], offset: ?>, 3 : i32>
          into memref<128x128xf32, strided<[512, 1], offset: ?>, 3 : i32>
        memref.copy %sv_hbm, %sv_sbuf_2d
          : memref<128x128xf32, strided<[?, ?], offset: ?>, 4 : i32>
          to memref<128x128xf32, strided<[512, 1], offset: ?>, 3 : i32>
      }
    }

    // Collapse the 4D alloc to 2D — creates multi-non-unit groups
    %collapsed = memref.collapse_shape %alloc [[0, 1], [2, 3]]
      : memref<128x2x2x128xf32, 3 : i32>
      into memref<256x256xf32, 3 : i32>

    // Output alloc
    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<128x2x2x128xf32, 3 : i32>
    %collapsed_out = memref.collapse_shape %alloc_out [[0, 1], [2, 3]]
      : memref<128x2x2x128xf32, 3 : i32>
      into memref<256x256xf32, 3 : i32>

    // Tiled computation on collapsed 2D view
    scf.for %bm = %c0 to %c2 step %c1 {
      scf.for %bn = %c0 to %c2 step %c1 {
        %off_m = arith.muli %bm, %c128 : index
        %off_n = arith.muli %bn, %c128 : index

        %sv_in = memref.subview %collapsed[%off_m, %off_n] [128, 128] [1, 1]
          : memref<256x256xf32, 3 : i32>
          to memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>

        %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, 3 : i32>

        linalg.add ins(%sv_in, %sv_in
          : memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>,
            memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>)
          outs(%tile_out : memref<128x128xf32, 3 : i32>)

        %sv_out = memref.subview %collapsed_out[%off_m, %off_n] [128, 128] [1, 1]
          : memref<256x256xf32, 3 : i32>
          to memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>
        memref.copy %tile_out, %sv_out
          : memref<128x128xf32, 3 : i32>
          to memref<128x128xf32, strided<[256, 1], offset: ?>, 3 : i32>
      }
    }

    // Copy back to HBM
    %alloc_hbm = memref.alloc() {alignment = 64 : i64} : memref<256x256xf32, 4 : i32>
    %collapsed_hbm = memref.collapse_shape %alloc_out [[0, 1], [2, 3]]
      : memref<128x2x2x128xf32, 3 : i32>
      into memref<256x256xf32, 3 : i32>
    memref.copy %collapsed_hbm, %alloc_hbm
      : memref<256x256xf32, 3 : i32>
      to memref<256x256xf32, 4 : i32>
    return %alloc_hbm : memref<256x256xf32, 4 : i32>
  }
}
'''


def test_multi_non_unit_collapse_nisa_lowering():
    """
    Verify that linalg-to-nisa correctly handles collapse_shape with multi-
    non-unit groups in SBUF.

    Before fix: getBaseAndOffsets stops at the collapse (can't decompose
    multi-non-unit groups), leaving stale memref as base → wrong NISA map.

    After fix: getBaseAndOffsets traces through the collapse, marking batch
    dims as dropped.  The linalg.add is lowered to nisa.tensor_tensor_arith
    with the 4D alloc as base.
    """
    result = run_linalg_to_nisa(MLIR_MULTI_NON_UNIT_COLLAPSE)

    # The linalg.add should be lowered to nisa.tensor_tensor_arith
    # referencing the 4D SBUF alloc (not the collapsed 2D view).
    check_patterns = '''
CHECK: func.func @test_multi_non_unit_collapse
CHECK: nisa.alloc
CHECK: nisa.tensor_tensor_arith
CHECK-SAME: op=add
CHECK-NOT: linalg.add
CHECK: return
'''
    run_filecheck(result, check_patterns)


def test_multi_non_unit_collapse_correct_indices():
    """
    Verify that the decomposed indices are correct:
    - dim 0 (partition, 128): base 0, iterated by d0
    - dim 1 (batch, 2): batch index = collapsed_offset / 128
    - dim 2 (batch, 2): batch index = collapsed_offset / 128
    - dim 3 (free, 128): base 0, iterated by d1

    The NISA map should show:
      %mem[%c0 + d0, <batch1> + 0, <batch2> + 0, %c0 + d1]
    where <batch1> and <batch2> are divui results.
    """
    result = run_linalg_to_nisa(MLIR_MULTI_NON_UNIT_COLLAPSE)

    # Verify correct index structure in tensor_tensor_arith:
    # - dim 0: constant_0 + d0 (partition tile)
    # - dim 1: divui result + 0 (batch block, dropped)
    # - dim 3: constant_0 + d1 (free tile)
    # The divui decomposes collapsed_offset / primary_size.
    # Constants get unique names when the module is re-serialized, so use
    # a regex wildcard to match `%c0`, `%c0_9`, `%c0_11`, etc.
    check_patterns = '''
CHECK: arith.divui
CHECK: nisa.tensor_tensor_arith
CHECK-SAME: %c0{{.*}} + d0
CHECK-SAME: + 0
CHECK-SAME: + 0
CHECK-SAME: %c0{{.*}} + d1
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
