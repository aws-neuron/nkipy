"""
End-to-end tests for RMSNorm kernel.

RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight

This exercises:
1. Element-wise square (multiply)
2. Sum reduction over last axis
3. Scalar multiply for mean (sum * 1/N)
4. Addition with scalar epsilon
5. Square root + division for normalization
6. Element-wise multiply with weight

Run with: pytest tests/e2e/test_rmsnorm.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.parametrize("M, N, tile_size", [
    (256, 256, [128, 128]),
])
def test_rmsnorm(M, N, tile_size):
    """
    Test RMSNorm: x / sqrt(mean(x^2) + eps) * weight.

    Broken down into individual ops with per-op knobs.
    """
    eps = 1e-6

    @trace(input_specs=[((M, N), "f32"), ((N, 1), "f32")])
    def rmsnorm_kernel(x, weight):
        x_fp32 = x.astype(np.float32)
        w_fp32 = weight.astype(np.float32)

        sq = np.square(x_fp32)
        knob.knob(sq).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

        sum_sq = np.sum(sq, axis=-1, keepdims=True)
        knob.knob(sum_sq).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

        mean_sq = sum_sq * np.float32(1.0 / N)
        knob.knob(mean_sq).tile_op(tile_size=[128, 1]).layout(mem_space="Sbuf")

        normed = x_fp32 / np.sqrt(mean_sq + eps)
        knob.knob(normed).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

        result = normed * w_fp32
        knob.knob(result).tile_op(tile_size=tile_size).layout(mem_space="SharedHbm")
        return result

    run_kernel_test(
        rmsnorm_kernel,
        stop_after="legalize-layout",
        modes=Mode.LLVM,
        rtol=1e-3,
        atol=1e-3,
    )

    run_kernel_test(
        rmsnorm_kernel,
        modes=Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
