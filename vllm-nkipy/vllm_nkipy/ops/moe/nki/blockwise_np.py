# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""
PyTorch implementation of blockwise MoE computation
"""

import torch
from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType

from vllm_nkipy.ops.moe.nki.utils import BLOCK_SIZE, ControlType


def gelu_apprx_sigmoid(x):
    """Approximate GELU using sigmoid approximation."""
    return x * torch.sigmoid(1.702 * x)


def silu(x):
    """SiLU (Swish) activation function."""
    return x * torch.sigmoid(x)


def blockwise_np(
    hidden_states: torch.Tensor,  # Shape (T, H)
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    gate_up_bias_plus1_T: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_bias_broadcasted: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    activation_function: ActFnType = ActFnType.Swish,
    dtype: torch.dtype = torch.bfloat16,
):
    res_dtype = hidden_states.dtype

    hidden_states = hidden_states.to(dtype)
    output = torch.zeros_like(hidden_states)
    expert_affinities_masked = expert_affinities_masked.to(dtype)
    down_proj_weight = down_proj_weight.to(dtype)
    gate_up_proj_weight = gate_up_proj_weight.to(dtype)

    # Handle optional biases
    if gate_up_bias_plus1_T is not None:
        gate_up_bias_plus1_T = gate_up_bias_plus1_T.to(dtype)
    if down_bias_broadcasted is not None:
        down_bias_broadcasted = down_bias_broadcasted.to(dtype)

    _, intermediate_size, hidden_size = down_proj_weight.shape
    num_blocks = block_to_expert.shape[0]

    expert_idx = None
    for b in range(num_blocks):
        local_token_position_to_id = token_position_to_id[b]
        real_token_idx = local_token_position_to_id != -1  # token skip
        if block_to_expert[b] == ControlType.SKIP_BLOCK.value:
            continue
        elif block_to_expert[b] == ControlType.SKIP_DMA.value:
            assert expert_idx is not None
        else:
            expert_idx = block_to_expert[b]

        local_hidden_states = torch.zeros_like(
            hidden_states[local_token_position_to_id]
        )
        local_hidden_states[real_token_idx] = hidden_states[
            local_token_position_to_id[real_token_idx]
        ]
        local_expert_affinities_masked = torch.zeros_like(
            expert_affinities_masked[local_token_position_to_id, expert_idx]
        )
        local_expert_affinities_masked[real_token_idx] = expert_affinities_masked[
            local_token_position_to_id[real_token_idx], expert_idx
        ]
        gate_up_activation = (
            (
                local_hidden_states
                @ gate_up_proj_weight[expert_idx].reshape(
                    hidden_size, 2 * intermediate_size
                )
            )
            .reshape(BLOCK_SIZE, 2, intermediate_size)
            .to(local_hidden_states.dtype)
        )

        if gate_up_bias_plus1_T is not None:
            # gate_up_bias: (E, I, 2)
            gate_up_activation += gate_up_bias_plus1_T[expert_idx].transpose(0, 1)
        if activation_function == ActFnType.SiLU:
            raise NotImplementedError(
                "gate_up_bias_plus1 optimization for SwiGLU breaks SiLU"
            )
            act_res = silu(gate_up_activation[:, 0])
            multiply_1 = act_res * gate_up_activation[:, 1]
        elif activation_function == ActFnType.Swish:
            x_glu = torch.clamp(gate_up_activation[:, 0], max=7.0)
            x_linear = torch.clamp(gate_up_activation[:, 1], min=-6.0, max=8.0)
            act_res = gelu_apprx_sigmoid(x_glu)
            multiply_1 = act_res * x_linear
        else:
            raise ValueError(f"Activation function {activation_function} not supported")
        down_weights = down_proj_weight[expert_idx]
        down_activation = (multiply_1 @ down_weights).to(multiply_1.dtype)
        if down_bias_broadcasted is not None:
            down_activation += down_bias_broadcasted[expert_idx]
        down_activation = (
            down_activation
            * local_expert_affinities_masked.unsqueeze(-1)
        )
        output[
            local_token_position_to_id[real_token_idx]
        ] += down_activation[real_token_idx]

    return output.to(res_dtype)


def blockwise_np_compile(
    hidden_states: torch.Tensor,  # Shape (T, H)
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    gate_up_bias_plus1_T: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_bias_broadcasted: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    activation_function: ActFnType = ActFnType.Swish,
    dtype: torch.dtype = torch.bfloat16,
):
    res_dtype = hidden_states.dtype

    hidden_states = hidden_states.to(dtype)
    output = torch.zeros_like(hidden_states)
    expert_affinities_masked = expert_affinities_masked.to(dtype)
    down_proj_weight = down_proj_weight.to(dtype)
    gate_up_proj_weight = gate_up_proj_weight.to(dtype)

    # Handle optional biases
    if gate_up_bias_plus1_T is not None:
        gate_up_bias_plus1_T = gate_up_bias_plus1_T.to(dtype)
    if down_bias_broadcasted is not None:
        down_bias_broadcasted = down_bias_broadcasted.to(dtype)

    _, intermediate_size, hidden_size = down_proj_weight.shape
    num_blocks = block_to_expert.shape[0]

    # Optimization: num_valid_tokens is just hidden_states.shape[0]
    # token_position_to_id and block_to_expert have predictable
    # patterns, so we can eliminate indexing
    num_valid_tokens = hidden_states.shape[0]

    # Optimization: block_to_expert[b] == b, so expert_idx is just the block index
    for expert_idx in range(num_blocks):
        # Optimization: Direct slicing instead of complex indexing + masking
        local_hidden_states = torch.zeros(
            BLOCK_SIZE, hidden_size, dtype=dtype, device=hidden_states.device
        )
        local_hidden_states[:num_valid_tokens] = hidden_states

        local_expert_affinities_masked = torch.zeros(
            BLOCK_SIZE, dtype=dtype, device=expert_affinities_masked.device
        )
        local_expert_affinities_masked[:num_valid_tokens] = expert_affinities_masked[
            :, expert_idx
        ]

        gate_up_activation = (
            (
                local_hidden_states
                @ gate_up_proj_weight[expert_idx].reshape(
                    hidden_size, 2 * intermediate_size
                )
            )
            .reshape(BLOCK_SIZE, 2, intermediate_size)
            .to(local_hidden_states.dtype)
        )

        if gate_up_bias_plus1_T is not None:
            # gate_up_bias: (E, I, 2)
            gate_up_activation += gate_up_bias_plus1_T[expert_idx].transpose(0, 1)
        if activation_function == ActFnType.SiLU:
            raise NotImplementedError(
                "gate_up_bias_plus1 optimization for SwiGLU breaks SiLU"
            )
            act_res = silu(gate_up_activation[:, 0])
            multiply_1 = act_res * gate_up_activation[:, 1]
        elif activation_function == ActFnType.Swish:
            x_glu = torch.clamp(gate_up_activation[:, 0], max=7.0)
            x_linear = torch.clamp(gate_up_activation[:, 1], min=-6.0, max=8.0)
            act_res = gelu_apprx_sigmoid(x_glu)
            multiply_1 = act_res * x_linear
        else:
            raise ValueError(f"Activation function {activation_function} not supported")
        down_weights = down_proj_weight[expert_idx]
        down_activation = (multiply_1 @ down_weights).to(multiply_1.dtype)
        if down_bias_broadcasted is not None:
            down_activation += down_bias_broadcasted[expert_idx]
        down_activation = (
            down_activation
            * local_expert_affinities_masked.unsqueeze(-1)
        )

        # Optimization: Direct slicing for output accumulation
        output[:num_valid_tokens] += down_activation[:num_valid_tokens]

    return output.to(res_dtype)
