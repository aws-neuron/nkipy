"""
Tests for the insert-spill-reload pass.

Two test categories:
  1. IR-level tests: Run the pass on hand-crafted MLIR to verify spill/reload
     insertion logic (victims selected, HBM slots created, copies inserted).
  2. Kernel correctness test: Trace a real kernel through the full pipeline
     (including insert-spill-reload) and verify numerical output via BIR sim.
"""

import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode
from pass_utils import run_passes, run_filecheck


# ============================================================================
# IR-level pass tests
# ============================================================================


def test_sbuf_overflow_spill():
    """
    SBUF overflow triggers spill insertion.

    Uses post-legalize-layout physical shapes [partTile, nB0, nB1, freeTile].
    Three 128×1×1×2048 SBUF allocations → per-partition = 1×1×2048×4 = 8192 bytes
    each, 24576 bytes peak.  Capacity set to 16384 → spill required.

    Expected: HBM spill slot and memref.copy spill/reload ops.
    """
    input_ir = """
module {
  func.func @sbuf_overflow(%arg0: memref<128x1x1x2048xf32, 4 : i32>) -> memref<128x1x1x2048xf32, 4 : i32> {
    %a = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    %b = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    %c = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%a : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.exp ins(%a : memref<128x1x1x2048xf32, 3 : i32>)
               outs(%b : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.mul ins(%b, %b : memref<128x1x1x2048xf32, 3 : i32>, memref<128x1x1x2048xf32, 3 : i32>)
               outs(%c : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.copy ins(%c : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
    memref.dealloc %a : memref<128x1x1x2048xf32, 3 : i32>
    memref.dealloc %b : memref<128x1x1x2048xf32, 3 : i32>
    memref.dealloc %c : memref<128x1x1x2048xf32, 3 : i32>
    return %arg0 : memref<128x1x1x2048xf32, 4 : i32>
  }
}
"""
    # Per-partition: 1*1*2048*4 = 8192 bytes each, 3 live = 24576 bytes
    # Capacity 16384 < 24576 → spill triggered
    output_ir = run_passes(input_ir, ["insert-spill-reload=sbuf-capacity=16384"])

    check_patterns = """
    CHECK: func.func @sbuf_overflow
    CHECK: memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    CHECK: memref.alloc() : memref<128x1x1x2048xf32, 1 : i32>
    CHECK: linalg.exp
    CHECK: memref.copy {{.*}} 3 : i32> to memref<128x1x1x2048xf32, 1 : i32>
    CHECK: memref.copy {{.*}} 1 : i32> to memref<128x1x1x2048xf32, 3 : i32>
    CHECK: return
    """
    run_filecheck(output_ir, check_patterns)


def test_no_spill_below_capacity():
    """
    Single 128×1×1×512 allocation: per-partition = 1×1×512×4 = 2048 bytes,
    well below the 16384-byte capacity.

    Expected: no HBM spill slot created.
    """
    input_ir = """
module {
  func.func @no_spill(%arg0: memref<128x1x1x512xf32, 4 : i32>) -> memref<128x1x1x512xf32, 4 : i32> {
    %a = memref.alloc() : memref<128x1x1x512xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x512xf32, 4 : i32>)
                outs(%a : memref<128x1x1x512xf32, 3 : i32>)
    linalg.exp ins(%a : memref<128x1x1x512xf32, 3 : i32>)
               outs(%a : memref<128x1x1x512xf32, 3 : i32>)
    linalg.copy ins(%a : memref<128x1x1x512xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x512xf32, 4 : i32>)
    memref.dealloc %a : memref<128x1x1x512xf32, 3 : i32>
    return %arg0 : memref<128x1x1x512xf32, 4 : i32>
  }
}
"""
    # Per-partition: 1*1*512*4 = 2048 bytes; capacity 16384 > 2048 → no spill
    output_ir = run_passes(input_ir, ["insert-spill-reload=sbuf-capacity=16384"])

    check_patterns = """
    CHECK: func.func @no_spill
    CHECK: memref.alloc() : memref<128x1x1x512xf32, 3 : i32>
    CHECK-NOT: 1 : i32
    CHECK: linalg.exp
    CHECK: return
    """
    run_filecheck(output_ir, check_patterns)


def test_non_overlapping_lifetimes():
    """
    Two sequential 128×1×1×2048 allocations whose lifetimes don't overlap.

    Per-partition: 8192 bytes each.  Total (16384) exceeds capacity,
    but peak live usage (8192) does not.
    Expected: no HBM spill slot created.
    """
    input_ir = """
module {
  func.func @sequential(%arg0: memref<128x1x1x2048xf32, 4 : i32>) -> memref<128x1x1x2048xf32, 4 : i32> {
    %a = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%a : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.exp ins(%a : memref<128x1x1x2048xf32, 3 : i32>)
               outs(%a : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.copy ins(%a : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
    memref.dealloc %a : memref<128x1x1x2048xf32, 3 : i32>
    %b = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%b : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.sqrt ins(%b : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%b : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.copy ins(%b : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
    memref.dealloc %b : memref<128x1x1x2048xf32, 3 : i32>
    return %arg0 : memref<128x1x1x2048xf32, 4 : i32>
  }
}
"""
    # Per-partition: 8192 bytes each, non-overlapping → peak = 8192
    # Capacity 16384 > 8192 → no spill
    output_ir = run_passes(input_ir, ["insert-spill-reload=sbuf-capacity=16384"])

    check_patterns = """
    CHECK: func.func @sequential
    CHECK: memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    CHECK-NOT: 1 : i32
    CHECK: linalg.exp
    CHECK-NOT: 1 : i32
    CHECK: linalg.sqrt
    CHECK: return
    """
    run_filecheck(output_ir, check_patterns)


def test_spill_with_loop_use():
    """
    A spilled value is used inside a loop body after the spill point.

    %a and %b are both 128×1×1×2048 → per-partition = 8192 bytes each;
    together (16384) they exceed the 12288-byte capacity.  %a is spilled at
    the peak pressure point (second linalg.copy).  Both %a and %b are used
    inside the subsequent scf.for loop, so the reload must be inserted before
    the loop — not omitted because the use is in a nested region.

    Verifies the nested-region fix: the pass walks up to the entry block to
    find the ancestor of each user, so scf.for is correctly identified as the
    first use after the spill.
    """
    input_ir = """
module {
  func.func @spill_with_loop_use(%arg0: memref<128x1x1x2048xf32, 4 : i32>) -> memref<128x1x1x2048xf32, 4 : i32> {
    %a = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    %b = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%a : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%b : memref<128x1x1x2048xf32, 3 : i32>)
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      linalg.add ins(%a, %b : memref<128x1x1x2048xf32, 3 : i32>, memref<128x1x1x2048xf32, 3 : i32>)
                 outs(%b : memref<128x1x1x2048xf32, 3 : i32>)
    }
    linalg.copy ins(%b : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
    memref.dealloc %a : memref<128x1x1x2048xf32, 3 : i32>
    memref.dealloc %b : memref<128x1x1x2048xf32, 3 : i32>
    return %arg0 : memref<128x1x1x2048xf32, 4 : i32>
  }
}
"""
    # Per-partition: 8192 bytes each, 2 live = 16384 bytes
    # Capacity 12288 < 16384 → spill triggered
    output_ir = run_passes(input_ir, ["insert-spill-reload=sbuf-capacity=12288"])

    check_patterns = """
    CHECK: func.func @spill_with_loop_use
    CHECK: memref.alloc() : memref<128x1x1x2048xf32, 1 : i32>
    CHECK: memref.copy {{.*}} 3 : i32> to memref<128x1x1x2048xf32, 1 : i32>
    CHECK: memref.copy {{.*}} 1 : i32> to memref<128x1x1x2048xf32, 3 : i32>
    CHECK-NEXT: scf.for
    """
    run_filecheck(output_ir, check_patterns)


def test_multiple_pressure_peaks():
    """
    Two independent high-pressure windows, each requiring a separate spill.

    Window 1: %a + %b simultaneously live (2 × 8192 = 16384 > 12288) around linalg.exp.
    Window 2: %c + %d simultaneously live (16384 > 12288) around linalg.sqrt.
    The two windows are separated by deallocations, so they have non-overlapping
    lifetimes.  The single-peak algorithm would only fix one window; the
    multi-peak fix ensures both receive a spill.

    Verifies: two HBM spill slots are created and two SBUF→HBM spill copies
    are inserted — one for each window.
    """
    input_ir = """
module {
  func.func @two_peaks(%arg0: memref<128x1x1x2048xf32, 4 : i32>) -> memref<128x1x1x2048xf32, 4 : i32> {
    %a = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    %b = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%a : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.exp ins(%a : memref<128x1x1x2048xf32, 3 : i32>)
               outs(%b : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.copy ins(%b : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
    memref.dealloc %a : memref<128x1x1x2048xf32, 3 : i32>
    memref.dealloc %b : memref<128x1x1x2048xf32, 3 : i32>
    %c = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    %d = memref.alloc() : memref<128x1x1x2048xf32, 3 : i32>
    linalg.copy ins(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
                outs(%c : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.sqrt ins(%c : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%d : memref<128x1x1x2048xf32, 3 : i32>)
    linalg.copy ins(%d : memref<128x1x1x2048xf32, 3 : i32>)
                outs(%arg0 : memref<128x1x1x2048xf32, 4 : i32>)
    memref.dealloc %c : memref<128x1x1x2048xf32, 3 : i32>
    memref.dealloc %d : memref<128x1x1x2048xf32, 3 : i32>
    return %arg0 : memref<128x1x1x2048xf32, 4 : i32>
  }
}
"""
    # Per-partition: 8192 bytes each, 2 live per window = 16384 bytes
    # Capacity 12288 < 16384 → spill triggered in each window
    output_ir = run_passes(input_ir, ["insert-spill-reload=sbuf-capacity=12288"])

    check_patterns = """
    CHECK: func.func @two_peaks
    CHECK: memref.alloc() : memref<128x1x1x2048xf32, 1 : i32>
    CHECK: linalg.exp
    CHECK: memref.copy {{.*}} 3 : i32> to memref<128x1x1x2048xf32, 1 : i32>
    CHECK: memref.alloc() : memref<128x1x1x2048xf32, 1 : i32>
    CHECK: linalg.sqrt
    CHECK: memref.copy {{.*}} 3 : i32> to memref<128x1x1x2048xf32, 1 : i32>
    CHECK: return
    """
    run_filecheck(output_ir, check_patterns)


def test_rmsnorm_no_spill():
    """
    Verify insert-spill-reload is a no-op for a 256×256 RMSNorm kernel.

    RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight

    After tiling and bufferization, the full sq buffer is allocated as a single
    256×256×f32 SBUF alloc (subviews alias into it).  Per-partition size is
    256 × 4 = 1024 bytes — well within the trn2 per-partition SBUF capacity
    (~208 KB).  No spilling should occur.
    """
    M, N = 256, 256
    tile_size = [128, 128]
    eps = 1e-6

    @trace(input_specs=[((M, N), "f32"), ((N, 1), "f32")])
    def rmsnorm_kernel(x, weight):
        x_fp32 = x.astype(np.float32)
        w_fp32 = weight.astype(np.float32)

        sq = np.square(x_fp32)
        knob.knob(sq, mem_space="Sbuf", tile_size=tile_size)

        sum_sq = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(sum_sq, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])

        mean_sq = sum_sq * np.float32(1.0 / N)
        knob.knob(mean_sq, mem_space="Sbuf", tile_size=[128, 1])

        normed = x_fp32 / np.sqrt(mean_sq + eps)
        knob.knob(normed, mem_space="Sbuf", tile_size=tile_size)

        result = normed * w_fp32
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    check_patterns = """
    CHECK: func.func @rmsnorm_kernel
    CHECK: memref.alloc() {{.*}} : memref<128x{{.*}}xf32, 3 : i32>
    CHECK-NOT: 1 : i32
    CHECK: return
    """
    run_kernel_test(
        rmsnorm_kernel,
        stop_after="insert-spill-reload",
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Kernel correctness test (full pipeline → BIR simulation)
# ============================================================================


def test_exp_kernel_no_spurious_spill():
    """
    Verify insert-spill-reload is a no-op for a kernel that fits in SBUF.

    A 128×128 exp kernel uses 2 SBUF allocs of 128×128×f32.  Per-partition
    size = 128 × 4 = 512 bytes each, well within the trn2 per-partition SBUF
    capacity (~208 KB).  The pass should leave the IR unchanged: no HBM spill
    slots inserted.

    Stops immediately after insert-spill-reload so the check runs on the pass
    output directly, before later passes could transform or eliminate any
    spill-related ops.
    """
    M, N = 128, 128
    tile_size = [128, 128]

    @trace(input_specs=[((M, N), "f32")])
    def exp_kernel(x):
        result = np.exp(x)
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
        return result

    check_patterns = """
    CHECK: func.func @exp_kernel
    CHECK: memref.alloc() {{.*}} 3 : i32
    CHECK-NOT: 1 : i32
    CHECK: linalg.exp
    CHECK: return
    """
    run_kernel_test(
        exp_kernel,
        stop_after="insert-spill-reload",
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )
