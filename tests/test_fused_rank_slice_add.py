"""
Test module for fused_rank_slice_add NKI kernel implementation.
"""

import ml_dtypes
import numpy as np
import pytest

from nkipy.runtime import baremetal_jit
from kernels.fused_rank_slice_nki import fused_rank_slice_add
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)


@pytest.mark.parametrize("test_rank", [0, 1, 2, 3, 15])
def test_fused_rank_slice_add(test_rank):
    """Test fused_rank_slice_add NKI implementation with multiple ranks"""
    # Test parameters
    num_ranks = 16
    batch_size_per_rank = 8
    hidden_size = 2880
    total_batch_size = num_ranks * batch_size_per_rank
    dtype = bfloat16

    # Generate test data
    x = np.random.normal(
        size=(total_batch_size, hidden_size),
        scale=1.0,
    ).astype(dtype)

    residual = np.random.normal(
        size=(total_batch_size, hidden_size),
        scale=1.0,
    ).astype(dtype)

    # Test with specified rank
    rank = np.array([[test_rank]], dtype=np.int32)

    # CPU reference implementation
    x_reshaped = x.reshape((num_ranks, batch_size_per_rank, hidden_size))
    residual_reshaped = residual.reshape((num_ranks, batch_size_per_rank, hidden_size))
    output_cpu = x_reshaped[test_rank] + residual_reshaped[test_rank]

    # Execute with baremetal_jit
    nki_result = baremetal_jit(fused_rank_slice_add)(
        x=x,
        y=residual,
        rank=rank,
        batch_size_per_rank=batch_size_per_rank,
    )

    # Compare results
    print(f"Input shape: {x.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Output shape: {nki_result.shape}")
    print(f"Expected output shape: {output_cpu.shape}")

    assert_allclose(
        nki_result,
        output_cpu,
        rtol=1e-2,
        atol=1e-2,
        use_matrix_rel_err=False,
    )

    print(f"✓ fused_rank_slice_add NKI test passed for rank {test_rank}!")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
