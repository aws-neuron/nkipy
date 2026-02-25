"""
Test module for RMSNorm kernel implementation.
"""

import numpy as np
import pytest
from config import Config
from kernels.rmsnorm import rmsnorm
from utils import (
    assert_allclose,
)
from nkipy.runtime import baremetal_jit


def test_rmsnorm():
    config = Config()
    batch_size = 1
    seq_len = 1024
    hidden_size = config.hidden_size
    eps = config.norm_eps

    x = np.random.normal(
        size=(batch_size, seq_len, hidden_size),
        scale=1.0,
    ).astype(config.dtype)
    weight = np.random.normal(
        size=(hidden_size,),
        scale=1.0,
    ).astype(config.dtype)

    output_cpu = rmsnorm(
        x=x,
        weight=weight,
        eps=eps,
        is_neuronpy=False,
    )

    out_device = baremetal_jit(rmsnorm)(x=x, weight=weight, eps=eps, is_neuronpy=True)
    assert_allclose(output_cpu, out_device)

if __name__ == "__main__":
    pytest.main(["-s", __file__])
