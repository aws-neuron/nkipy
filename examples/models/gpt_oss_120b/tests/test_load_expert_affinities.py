import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest

from config import Config
from nkipy.runtime.decorators import baremetal_jit
from nkipy.core.nki_op import wrap_nki_kernel
from kernels.blockwise_nki import (
    TILE_SIZE,
    load_expert_affinities,
    load_token_indices,
    _load_block_expert,
)
from utils import assert_allclose

bfloat16 = np.dtype(ml_dtypes.bfloat16)

T, E = 128, 128


@nki.jit
def load_expert_affinities_wrapper(
    expert_affinities_hbm, token_position_to_id, block_to_expert
):
    # beta-3: the migrated helpers are buffered (dst-first, buffer_idx-indexed).
    # Drive a single block (buffer_idx 0, block_idx 0) end to end.
    buffer_idx = 0
    block_idx = 0
    out = nl.ndarray((TILE_SIZE, 1), dtype=expert_affinities_hbm.dtype, buffer=nl.shared_hbm)

    token_indices = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
    load_token_indices(buffer_idx, token_indices, token_position_to_id, block_idx)

    current_expert = nl.zeros((1, 1, 1), dtype=nl.int32, buffer=nl.sbuf)
    _load_block_expert(current_expert, buffer_idx, block_to_expert, block_idx)

    expert_affinities_masked = nl.ndarray(
        (TILE_SIZE, 1), dtype=expert_affinities_hbm.dtype, buffer=nl.sbuf
    )
    load_expert_affinities(
        buffer_idx=buffer_idx,
        expert_affinities_masked=expert_affinities_masked,
        expert_affinities_masked_hbm=expert_affinities_hbm,
        token_indices=token_indices,
        expert=current_expert[:, buffer_idx, :],
        compute_dtype=expert_affinities_hbm.dtype,
    )
    nl.store(out, expert_affinities_masked[0:TILE_SIZE, 0:1])
    return out


def test_load_expert_affinities():
    expert_affinities_hbm = np.random.random_sample([T, E]).astype(bfloat16)
    token_position_to_id = np.arange(T, dtype=np.int32).reshape(1, T)
    block_to_expert = np.zeros((1,), dtype=np.int32)

    operands = [expert_affinities_hbm, token_position_to_id, block_to_expert]
    nki_op = wrap_nki_kernel(
        load_expert_affinities_wrapper,
        operands,
        is_nki_beta_3_version=True,
        platform_target="trn2",
    )

    def call(a, b, c):
        return nki_op(a, b, c)

    expert_affinities_out = baremetal_jit(call)(*operands)

    # block_to_expert[0] == 0, token_position_to_id == arange, so
    # out[p] == expert_affinities_hbm[p, 0].
    assert_allclose(
        expert_affinities_out[:, 0].astype(expert_affinities_hbm.dtype),
        expert_affinities_hbm[:, 0],
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
