import ml_dtypes
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
import pytest

from config import Config
from nkipy.runtime.decorators import baremetal_jit
from kernels.blockwise_nki import (
    TILE_SIZE,
    load_expert_affinities,
    load_token_indices,
)
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)

T, E = 128, 128

def load_expert_affinities_wrapper(expert_affinities_hbm, token_position_to_id, block_to_expert):
    expert_affinities_out = nl.ndarray(
        (nl.par_dim(TILE_SIZE), 1), dtype=expert_affinities_hbm.dtype, buffer=nl.hbm
    )
    block_idx = 0
    token_indices = load_token_indices(token_position_to_id, block_idx)
    expert = nl.load(block_to_expert[block_idx], dtype=np.int32)
    
    expert_affinities = load_expert_affinities(
        expert_affinities_masked_hbm=expert_affinities_hbm,
        token_indices=token_indices,
        expert=expert,
        compute_dtype=expert_affinities_hbm.dtype,
    )
    nisa.dma_copy(
        dst=expert_affinities_out,
        src=expert_affinities,
    )
    return expert_affinities_out

@pytest.mark.skip
def test_load_expert_affinities():
    expert_affinities_hbm = np.random.random_sample([T, E]).astype(Config.dtype)
    token_position_to_id = np.arange(T, dtype=np.int32).reshape(1, T)
    block_to_expert = np.zeros((1,), dtype=np.int32)
    expert_affinities_out = baremetal_jit(
        load_expert_affinities_wrapper
    )(expert_affinities_hbm, token_position_to_id, block_to_expert)

    assert_allclose(
        expert_affinities_out[:, 0].astype(expert_affinities_hbm.dtype),
        expert_affinities_hbm[:, 0],
    )

if __name__ == "__main__":
    pytest.main(["-s", __file__])
