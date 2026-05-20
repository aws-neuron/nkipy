"""
Tests for the knob-driven-fusion pass.

Pattern: two elementwise ops with matching tile_size, annotated with
knob.fuse(...).  After tiling each op gets its own scf.for loop; after
fusion they should collapse into a single loop.

Run with: python -m pytest tests/passes/knob_driven_fusion/ -v
"""

import numpy as np
import pytest

from nkigen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test: two elementwise ops fuse into one loop
# ============================================================================


def test_fuse_two_adds():
    """Two independent adds with matching tile_size fuse to one loop nest."""
    M, N = 256, 256

    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32"),
                        ((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b, c, d):
        x = a + b
        knob.knob(x).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        y = c + d
        knob.knob(y).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        knob.fuse(x, y)
        return x, y

    # Before fusion (post tiling) we'd see two separate `scf.for %arg{ } = %c0
    # to %c256 step %c128 ... }` outer loops, one per add.  After fusion the
    # second loop disappears and a single outer loop yields both results.
    run_kernel_test(
        kernel,
        stop_after="knob-driven-fusion",
        modes=Mode.STRING_CHECK,
        check_ir_contains=[
            # Single outer scf.for that yields both fused results.
            "scf.for",
            "linalg.add",
        ],
        # Two adds before fusion -> only one outer loop after.  Use a
        # FileCheck-style negative: there should be exactly one
        # `scf.for %{{.*}} = %c0 to %c256 step %c128` matching the outer
        # loop.  Easiest sanity is to assert the loop count is 1 by
        # absence of a second scf.yield at the outer level — but that's
        # fragile.  Instead, defer the strict check to the FileCheck-mode
        # test below.
    )


def test_fuse_two_adds_loop_count():
    """Strict structural check: after fusion the kernel has exactly one
    outer scf.for and one inner scf.for (the inner siblings collapse
    into a single nest), with both adds living in that single loop.
    Pre-fusion (just tiling) the kernel would have had two outer loops
    with one inner loop each (4 total scf.for); after fusion it's 2.
    """
    M, N = 256, 256

    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32"),
                        ((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b, c, d):
        x = a + b
        knob.knob(x).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        y = c + d
        knob.knob(y).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        knob.fuse(x, y)
        return x, y

    # One outer scf.for, then one inner scf.for, then both adds, no
    # further scf.for before the return.
    check_patterns = (
        "CHECK-LABEL: func.func\n"
        "CHECK: scf.for\n"
        "CHECK-NEXT: scf.for\n"
        "CHECK-NOT: scf.for\n"
        "CHECK: linalg.add\n"
        "CHECK: linalg.add\n"
        "CHECK-NOT: scf.for\n"
        "CHECK: return\n"
    )
    run_kernel_test(
        kernel,
        stop_after="knob-driven-fusion",
        modes=Mode.FILECHECK,
        check_patterns=check_patterns,
    )


def test_fuse_numerics():
    """End-to-end LLVM JIT: fused result must match unfused NumPy."""
    M, N = 128, 256

    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32"),
                        ((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b, c, d):
        x = a + b
        knob.knob(x).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        y = c + d
        knob.knob(y).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        knob.fuse(x, y)
        return x, y

    run_kernel_test(
        kernel,
        stop_after="knob-driven-fusion",
        modes=Mode.LLVM,
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================================
# Test: mismatched bounds error out cleanly
# ============================================================================


def test_fuse_mismatched_tiles_errors():
    """Different tile_sizes -> different loop bounds -> the pass must
    emit an error rather than silently producing wrong code."""
    M, N = 256, 256

    @trace(input_specs=[((M, N), "f32"), ((M, N), "f32"),
                        ((M, N), "f32"), ((M, N), "f32")])
    def kernel(a, b, c, d):
        x = a + b
        knob.knob(x).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
        y = c + d
        knob.knob(y).tile_op(tile_size=[64, 64]).layout(mem_space="SharedHbm")
        knob.fuse(x, y)
        return x, y

    with pytest.raises(RuntimeError, match="mismatched loop bounds"):
        run_kernel_test(
            kernel,
            stop_after="knob-driven-fusion",
            modes=Mode.STRING_CHECK,
            check_ir_contains=["scf.for"],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
