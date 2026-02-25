"""
Test module for Fused RMSNorm Gemm NKI kernel implementation.
"""

import ml_dtypes
import numpy as np
import pytest

from nkipy.runtime import baremetal_run_traced_kernel
from kernels.fused_rmsnorm_gemm_nki import (
    fused_rmsnorm_gemm_v0_dma_transpose,
    fused_rmsnorm_gemm_v0_nc_transpose,
)
from kernels.rmsnorm import rmsnorm as rmsnorm_nkipy
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)

@pytest.mark.skip
def test_fused_rmsnorm_gemm_nki():
    """Test Fused RMSNorm Gemm NKI implementation with shape (16, 1, 2880) and eps=1e-5"""
    # Test parameters as specified
    batch_size = 16
    seq_len = 1
    hidden_size = 2880
    out_size = 640
    eps = 1e-5
    dtype = bfloat16

    # Generate test data
    x = np.random.normal(
        size=(batch_size, seq_len, hidden_size),
        scale=1.0,
    ).astype(dtype)

    # Gamma/weight shape should be (1, hidden_size) as required by NKI implementation
    weight = np.random.normal(
        size=(1, hidden_size),
        scale=1.0,
    ).astype(dtype)

    y = np.random.normal(
        size=(hidden_size, out_size),
        scale=1.0,
    ).astype(dtype)

    bias = np.random.normal(
        size=(1, out_size),
        scale=1.0,
    ).astype(dtype)

    # Run CPU reference implementation
    output_cpu = (
        rmsnorm_nkipy(
            x=x,
            weight=weight.reshape((hidden_size,)),  # CPU version expects 1D weight
            eps=eps,
            is_neuronpy=False,
        )
        @ y
        + bias
    )
    output_cpu = output_cpu.astype(dtype)

    for kernel in (
        fused_rmsnorm_gemm_v0_dma_transpose,
        fused_rmsnorm_gemm_v0_nc_transpose,
    ):
        nki_result = baremetal_run_traced_kernel(kernel, x=x, weight=weight, y=y, bias=bias, eps=eps)

        print(f"Hidden shape: {x.shape}")
        print(f"Weight shape: {weight.shape}")
        print(f"Dense weight shape:  {y.shape}")
        print(f"Output shape: {nki_result.shape}")

        assert_allclose(
            nki_result,
            output_cpu,
            rtol=1e-2,
            atol=1e-2,
        )

        print(
            f"✓ Fused RMSNorm Gemm NKI test with kernel_name {kernel.__name__} passed!"
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
