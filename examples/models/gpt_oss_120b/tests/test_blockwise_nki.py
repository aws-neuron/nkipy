import math

import ml_dtypes
import numpy as np
import pytest
import nki.language as nl

from config import Config
from nkipy.runtime import baremetal_jit
from nkipy.core.nki_op import wrap_nki_kernel
from kernels.blockwise_nki import blockwise_nki_static, ActFnType, TILE_SIZE
from kernels.blockwise_np import blockwise_np
from utils import assert_allclose
from kernels.blockwise_index import ControlType, BLOCK_SIZE

bfloat16 = np.dtype(ml_dtypes.bfloat16)


def get_blockwise_expert_and_token_mapping(num_blocks, n_experts, seq_len, test_weight_skip: bool):
    block_to_expert = np.full(num_blocks, ControlType.SKIP_BLOCK.value, dtype=np.int8)
    token_position_to_id = np.full((num_blocks, BLOCK_SIZE), ControlType.SKIP_DMA.value, dtype=np.int32)
    block_to_expert[0] = min(n_experts-1, 1) # not use expert 0 to verify this var is loaded
    # test token skip
    block_seq_len = min(seq_len, BLOCK_SIZE)
    token_position_to_id[0, :block_seq_len] = np.arange(block_seq_len)
    # test weight skip
    for i in range(1, num_blocks):
        block_to_expert[i] = ControlType.SKIP_DMA.value if test_weight_skip else i
        token_position_to_id[i] = np.arange(BLOCK_SIZE) + i * BLOCK_SIZE
    return block_to_expert, token_position_to_id

@pytest.mark.parametrize(
    "seq_len, hidden_size, intermediate_size, is_prefill",
    [
        (4096, 2880, 360, True),  # prefill: TP8
        (512, 2880, 360, False),  # tokengen: TP8
        # partial to leverage walrus detecting bugs like uninitialized read/write
        # TODO: fix test below
        (16, 512, 128, True),
        (512, 80, 128, True),
        (512, 512, 80, True),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        bfloat16,
        nl.float8_e5m2,
    ],
)
def test_blockwise_nki(seq_len, hidden_size, intermediate_size, is_prefill, dtype):
    num_blocks = num_static_blocks = math.ceil(seq_len / BLOCK_SIZE)
    activation_function = ActFnType.Swish
    hidden_states = np.random.randn(seq_len, hidden_size).astype(bfloat16)

    expert_affinities_masked_hbm = (
        np.random.rand(seq_len, Config.num_experts).astype(bfloat16) * 0.5
    )  # Keep values reasonable
    gate_up_proj_weight =np.random.normal(
        size=(Config.num_experts, hidden_size, 2, intermediate_size),
        scale=np.sqrt(2 / (hidden_size + 2 * intermediate_size * Config.num_experts)),
    ).astype(dtype)
    down_proj_weight = np.random.normal(
        size=(Config.num_experts, intermediate_size, hidden_size),
        scale=np.sqrt(2 / (hidden_size + intermediate_size * Config.num_experts)),
    ).astype(dtype)
    gate_up_bias_plus1_T = np.random.uniform(
        -0.1, 0.1, size=(Config.num_experts, intermediate_size, 2)
    ).astype(bfloat16)
    # preprocess up bias so no need to +1 for activation at runtime
    gate_up_bias_plus1_T[..., 1] += 1
    down_bias = np.random.uniform(
        -0.1, 0.1, size=(Config.num_experts, hidden_size)
    ).astype(bfloat16)
    down_bias_broadcasted = np.expand_dims(down_bias, axis=[1])
    down_bias_broadcasted = np.broadcast_to(
        down_bias_broadcasted, (Config.num_experts, TILE_SIZE, hidden_size)
    )  # boardcast to reduce the cost of nc_stream_shuffle

    block_to_expert, token_position_to_id = get_blockwise_expert_and_token_mapping(
        num_blocks,
        Config.num_experts,
        seq_len,
        test_weight_skip=is_prefill,
    )

    output_nki = np.zeros((seq_len, hidden_size), dtype=bfloat16)

    BUFFER_DEGREE = 1 if is_prefill else 3

    # beta-3: wrap the kernel, non-tensor args go in kernel_kwargs, then drive it
    # inside a baremetal_jit'd fn (see project_gpt_oss_120b_nki_migration).
    operands = [
        hidden_states,
        output_nki,
        expert_affinities_masked_hbm,
        gate_up_proj_weight,
        gate_up_bias_plus1_T,
        down_proj_weight,
        down_bias_broadcasted,
        token_position_to_id,
        block_to_expert,
    ]
    nki_op = wrap_nki_kernel(
        blockwise_nki_static,
        operands,
        is_nki_beta_3_version=True,
        platform_target="trn2",
        kernel_kwargs={
            "num_static_blocks": num_static_blocks,
            "activation_function": activation_function,
            "compute_dtype": nl.bfloat16,
            "is_tensor_update_accumulating": True,
            "BUFFER_DEGREE": BUFFER_DEGREE,
        },
    )

    def call(hs, out, eam, gup, gub, dpw, dbb, t2i, b2e):
        return nki_op(hs, out, eam, gup, gub, dpw, dbb, t2i, b2e)

    nki_result = baremetal_jit(call)(*operands)

    np_result = blockwise_np(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked_hbm,
        down_proj_weight=down_proj_weight,
        gate_up_proj_weight=gate_up_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        dtype=bfloat16,
        gate_up_bias_plus1_T=gate_up_bias_plus1_T,
        down_bias_broadcasted=down_bias_broadcasted,
        activation_function=activation_function,
    )
    assert_allclose(
        nki_result,
        np_result,
    )

if __name__ == "__main__":
    pytest.main(["-s", __file__])
