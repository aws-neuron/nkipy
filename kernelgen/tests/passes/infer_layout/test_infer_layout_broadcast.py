"""
Tests for InferLayout backward propagation through broadcast operations.

When a broadcast division like `x (M,N) / sqrt(y (M,1))` has a knob on
the result, InferLayout should propagate backward to the (M,1)-shaped
producers with tile_size clamped to their dimensions.

This is the pattern from RMSNorm: normed = x / sqrt(mean_sq + eps)

Run with: pytest tests/passes/infer_layout/test_infer_layout_broadcast.py -v
"""

import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


def test_broadcast_div_propagates_clamped_tile():
    """
    Verify InferLayout propagates through broadcast with clamped tile_size.

    Pattern (from RMSNorm):
      intermediate = reduced + eps        # (256,1), no knob
      normed = x / sqrt(intermediate)     # (256,256), knob tile=[128,128]

    Expected after infer-layout:
      intermediate gets tile_size=[128, 1]  (clamped from [128, 128])
      sqrt(intermediate) gets tile_size=[128, 1]  (clamped)
    """
    M, N = 256, 256

    @trace(input_specs=[((M, N), "f32"), ((M, 1), "f32")])
    def kernel(x, reduced):
        intermediate = reduced + np.float32(1e-6)
        normed = x / np.sqrt(intermediate)
        knob.knob(normed, mem_space="Sbuf", tile_size=[128, 128])
        return normed

    # After infer-layout, the (256,1) ops should get clamped tile_size=[128, 1]
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=["tile_size = array<i64: 128, 1>"],
        modes=Mode.STRING_CHECK,
    )


def test_broadcast_div_full_rmsnorm_pattern():
    """
    Full RMSNorm broadcast pattern: reduction output (M,1) flows into
    a broadcast division with (M,N).

    Chain:
      sq = square(x)                     # (256,256), knob tile=[128,128]
      sum_sq = sum(sq, axis=-1)          # (256,1),   knob tile=[128], red=[128]
      mean_sq = sum_sq * (1/N)           # (256,1),   knob tile=[128,1]
      normed = x / sqrt(mean_sq + eps)   # (256,256), knob tile=[128,128]

    The intermediate (mean_sq + eps) and sqrt(...) have shape (256,1)
    but NO knob. InferLayout should infer tile_size=[128, 1] for them.
    """
    M, N = 256, 256
    tile_size = [128, 128]
    eps = 1e-6

    @trace(input_specs=[((M, N), "f32")])
    def kernel(x):
        x_fp32 = x.astype(np.float32)

        sq = np.square(x_fp32)
        knob.knob(sq, mem_space="Sbuf", tile_size=tile_size)

        sum_sq = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(sum_sq, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])

        mean_sq = sum_sq * np.float32(1.0 / N)
        knob.knob(mean_sq, mem_space="Sbuf", tile_size=[128, 1])

        normed = x_fp32 / np.sqrt(mean_sq + eps)
        knob.knob(normed, mem_space="Sbuf", tile_size=tile_size)

        return normed

    # The (mean_sq + eps) intermediate should get inferred tile_size=[128, 1]
    # via backward propagation from normed's [128, 128] clamped to (256,1) shape
    run_kernel_test(
        kernel,
        stop_after='infer-layout',
        check_ir_contains=["tile_size = array<i64: 128, 1>"],
        modes=Mode.STRING_CHECK,
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
