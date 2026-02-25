"""
Test module for attention_prefill kernel implementation.
"""

import numpy as np
import pytest
from config import Config
from kernels.router import router
from neuronxcc.nki.language import bfloat16
from utils import assert_allclose
from nkipy.runtime import baremetal_jit


@pytest.mark.parametrize(
    "dtype",
    [
        (np.float32),
        (bfloat16),
    ],
)
def test_cpu_vs_device(dtype):
    config = Config(max_model_len=128 * 1024 // 128)
    hidden_states = np.random.randn(config.max_model_len, config.hidden_size).astype(
        dtype
    )
    router_weight = np.random.normal(
        size=(config.hidden_size, config.num_experts),
        scale=np.sqrt(2 / (config.hidden_size + config.num_experts)),
    ).astype(dtype)
    router_bias = np.random.randn(config.num_experts).astype(dtype)

    top_k_indices_ref, expert_affinities_masked_sharded_ref = router(
        hidden_states_sharded=hidden_states,
        router_weight=router_weight,
        router_bias=router_bias,
        top_k=config.num_experts_per_tok,
        is_prefill=True,
        is_neuronpy=False,
    )

    top_k_indices, expert_affinities_masked_sharded = baremetal_jit(router)(
        hidden_states_sharded=hidden_states,
        router_weight=router_weight,
        router_bias=router_bias,
        top_k=config.num_experts_per_tok,
        is_prefill=True,
        is_neuronpy=True,
    )


    # bf16 cannot match top_k exactly
    if dtype == np.float32:
        np.testing.assert_array_equal(top_k_indices_ref, top_k_indices)
        assert_allclose(
            expert_affinities_masked_sharded_ref,
            expert_affinities_masked_sharded,
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
