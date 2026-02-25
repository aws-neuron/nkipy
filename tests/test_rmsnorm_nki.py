"""
Test module for RMSNorm NKI kernel implementation.
"""

import ml_dtypes
import numpy as np
import pytest

from nkipy.runtime import baremetal_run_traced_kernel
from kernels.rmsnorm import rmsnorm as rmsnorm_nkipy
from kernels.rmsnorm_nki import rmsnorm as rmsnorm_nki
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)


def test_rmsnorm_nki():
    """Test RMSNorm NKI implementation with shape (16, 1, 2880) and eps=1e-5"""
    # Test parameters as specified
    batch_size = 16
    seq_len = 1
    hidden_size = 2880
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
    
    # Run CPU reference implementation
    output_cpu = rmsnorm_nkipy(
        x=x,
        weight=weight.reshape((hidden_size,)),  # CPU version expects 1D weight
        eps=eps,
        is_neuronpy=False,
    )

    # Run NKI kernel on device
    nki_result = baremetal_run_traced_kernel(rmsnorm_nki, x=x, weight=weight, eps=eps)
    
    # Compare results
    print(f"Hidden shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {nki_result.shape}")
    
    assert_allclose(
        nki_result,
        output_cpu,
        rtol=1e-2,
        atol=1e-2,
    )
    
    print("✓ RMSNorm NKI test passed!")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
