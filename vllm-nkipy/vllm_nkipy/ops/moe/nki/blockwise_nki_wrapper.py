# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
import torch
from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType

from torch_to_nkipy.utils.nki import NKIOpRegistry
from vllm_nkipy.compile import local_compile
from vllm_nkipy.ops.moe.nki.blockwise_nki import (
    blockwise_nki_static,
    blockwise_nki_tokengen_one_tile_replicated_hidden_state,
)
from vllm_nkipy.ops.moe.nki.utils import Config


@NKIOpRegistry.register(
    "mylib::blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op",
)
def blockwise_nki_tokengen_one_tile_replicated_hidden_state_wrapper(
    hidden_states,
    expert_affinities_masked_transposed_hbm,  # (E, T)
    gate_up_proj_weight,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight,
    down_bias_broadcasted_hbm,
    token_position_to_id,
    block_to_expert,
    activation_function: int = ActFnType.Swish.value,
    # compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating=True,
    BUFFER_DEGREE=3,
):
    compute_dtype = Config.dtype
    activation_function = ActFnType(activation_function)
    return blockwise_nki_tokengen_one_tile_replicated_hidden_state(
        hidden_states,
        expert_affinities_masked_transposed_hbm,  # (E, T)
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
        activation_function,
        compute_dtype,
        is_tensor_update_accumulating,
        BUFFER_DEGREE,
    )


@torch.library.custom_op(
    "mylib::blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op",
    mutates_args=(),
)
def blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op(
    hidden_states: torch.Tensor,
    expert_affinities_masked_transposed_hbm: torch.Tensor,  # (E, T)
    gate_up_proj_weight: torch.Tensor,
    gate_up_bias_plus1_T_hbm: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_bias_broadcasted_hbm: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    activation_function: int = ActFnType.Swish.value,
    # compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 3,
) -> torch.Tensor:
    output = blockwise_nki_tokengen_one_tile_replicated_hidden_state_wrapper(
        hidden_states,
        expert_affinities_masked_transposed_hbm,  # (E, T)
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
        activation_function,
        # compute_dtype,
        is_tensor_update_accumulating,
        BUFFER_DEGREE,
    )
    return output


@blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op.register_fake
def _(
    hidden_states,
    expert_affinities_masked_transposed_hbm,  # (E, T)
    gate_up_proj_weight,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight,
    down_bias_broadcasted_hbm,
    token_position_to_id,
    block_to_expert,
    activation_function: int = ActFnType.Swish.value,
    # compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating=True,
    BUFFER_DEGREE=3,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


@local_compile(
    backend="nkipy",
    device="nkipy",
    force=True,
    name="blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op_compiled",
    fullgraph=True,
    dynamic=False,
)
def blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op_compiled(
    hidden_states: torch.Tensor,
    expert_affinities_masked_transposed_hbm: torch.Tensor,  # (E, T)
    gate_up_proj_weight: torch.Tensor,
    gate_up_bias_plus1_T_hbm: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_bias_broadcasted_hbm: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    activation_function: int = ActFnType.Swish.value,
    # compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 3,
) -> torch.Tensor:
    return blockwise_nki_tokengen_one_tile_replicated_hidden_state_custom_op(
        hidden_states=hidden_states,
        expert_affinities_masked_transposed_hbm=expert_affinities_masked_transposed_hbm,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm=gate_up_bias_plus1_T_hbm,
        down_proj_weight=down_proj_weight,
        down_bias_broadcasted_hbm=down_bias_broadcasted_hbm,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        activation_function=activation_function,
        # compute_dtype=# compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        BUFFER_DEGREE=BUFFER_DEGREE,
    )


@NKIOpRegistry.register("mylib::blockwise_nki_static_custom_op")
def blockwise_nki_static_wrapper(
    hidden_states,
    # output,
    expert_affinities_masked_hbm,
    gate_up_proj_weight,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight,
    down_bias_broadcasted_hbm,
    token_position_to_id,
    block_to_expert,
    num_static_blocks: int,
    activation_function: int = ActFnType.Swish.value,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 1,
):
    compute_dtype = Config.dtype
    activation_function = ActFnType(activation_function)

    return blockwise_nki_static(
        hidden_states,
        # output,
        expert_affinities_masked_hbm,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
        num_static_blocks,
        activation_function,
        compute_dtype,
        is_tensor_update_accumulating,
        BUFFER_DEGREE,
    )


@torch.library.custom_op("mylib::blockwise_nki_static_custom_op", mutates_args=())
def blockwise_nki_static_custom_op(
    hidden_states: torch.Tensor,
    # output: torch.Tensor,
    expert_affinities_masked_hbm: torch.Tensor,  # (T, E)
    gate_up_proj_weight: torch.Tensor,
    gate_up_bias_plus1_T_hbm: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_bias_broadcasted_hbm: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    num_static_blocks: int,
    activation_function: int = ActFnType.Swish.value,
    # compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 1,
) -> torch.Tensor:
    return blockwise_nki_static_wrapper(
        hidden_states,
        # output,
        expert_affinities_masked_hbm,
        gate_up_proj_weight,
        gate_up_bias_plus1_T_hbm,
        down_proj_weight,
        down_bias_broadcasted_hbm,
        token_position_to_id,
        block_to_expert,
        num_static_blocks,
        activation_function,
        # compute_dtype,
        is_tensor_update_accumulating,
        BUFFER_DEGREE,
    )


@blockwise_nki_static_custom_op.register_fake
def _(
    hidden_states,
    # output,
    expert_affinities_masked_hbm,  # (T, E)
    gate_up_proj_weight,
    gate_up_bias_plus1_T_hbm,
    down_proj_weight,
    down_bias_broadcasted_hbm,
    token_position_to_id,
    block_to_expert,
    num_static_blocks: int,
    activation_function: int = ActFnType.Swish.value,
    # compute_dtype: np.dtype = Config.dtype,
    is_tensor_update_accumulating: bool = True,
    BUFFER_DEGREE: int = 1,
) -> torch.Tensor:
    # Meta path: just shape/dtype propagation
    return torch.empty_like(hidden_states)
