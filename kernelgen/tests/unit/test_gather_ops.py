"""
Tests for gather / np.take operations.

These tests verify that:
  1. nkipy.gather carries a linalg reference_impl region
  2. The reference region is correctly inlined for LLVM CPU simulation
  3. LLVM JIT execution matches NumPy
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


# ============================================================================
# np.take  (axis=0 gather)
# ============================================================================

@pytest.mark.parametrize("vocab,embed,n_idx", [
    (128, 128, 8),
    (256, 64, 16),
    (128, 256, 4),
])
def test_take_axis0(vocab, embed, n_idx):
    """np.take along axis 0 — basic embedding lookup."""
    @trace(input_specs=[((vocab, embed), "f32"), ((n_idx,), "i32")])
    def kernel(table, indices):
        return np.take(table, indices, axis=0)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["nkipy.gather"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_take_single_index():
    """np.take with a single-element index tensor."""
    @trace(input_specs=[((128, 64), "f32"), ((1,), "i32")])
    def kernel(table, indices):
        return np.take(table, indices, axis=0)

    run_kernel_test(
        kernel, stop_after="trace",
        modes=Mode.LLVM,
    )


def test_take_large_embedding():
    """np.take with a larger table."""
    @trace(input_specs=[((512, 128), "f32"), ((32,), "i32")])
    def kernel(table, indices):
        return np.take(table, indices, axis=0)

    run_kernel_test(
        kernel, stop_after="trace",
        modes=Mode.LLVM,
    )


# ============================================================================
# np.take used in a larger computation
# ============================================================================

def test_take_then_add():
    """np.take followed by elementwise add — verifies gather result feeds
    into downstream ops correctly after inlining."""
    vocab, embed, n_idx = 128, 128, 8

    @trace(input_specs=[
        ((vocab, embed), "f32"),
        ((n_idx,), "i32"),
        ((n_idx, embed), "f32"),
    ])
    def kernel(table, indices, bias):
        gathered = np.take(table, indices, axis=0)
        return np.add(gathered, bias)

    run_kernel_test(
        kernel, stop_after="trace",
        modes=Mode.LLVM,
    )


# ============================================================================
# Test runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
