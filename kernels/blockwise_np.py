
import numpy as np
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import (
    ExpertAffinityScaleMode,
)
from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType
from neuronxcc.starfish.penguin.native_maths import gelu_apprx_sigmoid, silu
from kernels.blockwise_index import ControlType, BLOCK_SIZE
from config import Config


def blockwise_np(
    hidden_states: np.ndarray,  # Shape (T, H)
    expert_affinities_masked: np.ndarray,
    gate_up_proj_weight: np.ndarray,
    gate_up_bias_plus1_T: np.ndarray,
    down_proj_weight: np.ndarray,
    down_bias_broadcasted: np.ndarray,
    token_position_to_id: np.ndarray,
    block_to_expert: np.ndarray,
    activation_function: ActFnType = ActFnType.Swish,
    dtype: np.dtype = Config.dtype,
):
    res_dtype = hidden_states.dtype

    hidden_states = hidden_states.astype(dtype)
    output = np.zeros_like(hidden_states)
    expert_affinities_masked = expert_affinities_masked.astype(dtype)
    down_proj_weight = down_proj_weight.astype(dtype)
    gate_up_proj_weight = gate_up_proj_weight.astype(dtype)

    # Handle optional biases
    if gate_up_bias_plus1_T is not None:
        gate_up_bias_plus1_T = gate_up_bias_plus1_T.astype(dtype)
    if down_bias_broadcasted is not None:
        down_bias_broadcasted = down_bias_broadcasted.astype(dtype)

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
        
        local_hidden_states = np.zeros_like(
            hidden_states[local_token_position_to_id]
        )
        local_hidden_states[real_token_idx] = hidden_states[
            local_token_position_to_id[real_token_idx]
        ]
        local_expert_affinities_masked = np.zeros_like(
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
            .astype(local_hidden_states.dtype)
        )

        if gate_up_bias_plus1_T is not None:
            # gate_up_bias: (E, I, 2)
            gate_up_activation += gate_up_bias_plus1_T[expert_idx].transpose()
        if activation_function == ActFnType.SiLU:
            raise NotImplementedError(
                "gate_up_bias_plus1 optimization for SwiGLU breaks SiLU"
            )
            act_res = silu(gate_up_activation[:, 0])
            multiply_1 = act_res * gate_up_activation[:, 1]
        elif activation_function == ActFnType.Swish:
            x_glu = np.clip(gate_up_activation[:, 0], a_min=None, a_max=7.0)
            x_linear = np.clip(gate_up_activation[:, 1], a_min=-6.0, a_max=8.0)
            act_res = gelu_apprx_sigmoid(x_glu)
            multiply_1 = act_res * x_linear
        else:
            raise ValueError(f"Activation function {activation_function} not supported")
        down_weights = down_proj_weight[expert_idx]
        down_activation = (multiply_1 @ down_weights).astype(multiply_1.dtype)
        if down_bias_broadcasted is not None:
            down_activation += down_bias_broadcasted[expert_idx]
        down_activation = down_activation * local_expert_affinities_masked[:, np.newaxis]
        output[local_token_position_to_id[real_token_idx]] += down_activation[real_token_idx]
        
    return output.astype(res_dtype)

def moe_np(
    expert_masks,
    expert_affinities_masked,
    down_proj_weights,
    gate_and_up_proj_weights,
    hidden_states,
    T,
    H,
    E,
    I_TP,
    output_np,
    quantize=False,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    gate_up_proj_scale=np.empty([]),
    down_proj_scale=np.empty([]),
    dtype=np.float32,
):
  # For each expert
  for expert_idx in range(E):
    local_expert_affinities = expert_affinities_masked[:, expert_idx].reshape(-1, 1).astype(dtype)
    if expert_affinities_scaling_mode in [ExpertAffinityScaleMode.PRE_SCALE, ExpertAffinityScaleMode.PRE_SCALE_DELAYED]:
      scaled_hidden_states = hidden_states * local_expert_affinities
    else:
      scaled_hidden_states = hidden_states
    # Get expert weights
    expert_gate_up = gate_and_up_proj_weights[expert_idx]  # [H, 2, I]
    expert_down = down_proj_weights[expert_idx]        # [I, H]
    
    # Split gate and up weights
    gate_weight = expert_gate_up[:, 0, :]     # [H, I]
    up_weight = expert_gate_up[:, 1, :]       # [H, I]
    
    # Forward computations (needed for backward)
    selected_hidden_states = scaled_hidden_states * expert_masks[:, [expert_idx]]  # [T, H]
    gate_activation = selected_hidden_states @ gate_weight  # [T, I]
    up_activation = selected_hidden_states @ up_weight      # [T, I]
    if quantize:
      # [B, I_TP]
      # Quantize input proj by scaling followed by silu: $f_{act}(data \times scale)$
      silu_activation = silu(gate_activation * gate_up_proj_scale.squeeze()[:I_TP])
      # Quantize gate proj by scaling
      up_activation = up_activation * gate_up_proj_scale.squeeze()[I_TP:]
    else:
      silu_activation = silu(gate_activation)

    first_dot_activation = silu_activation * up_activation  # [T, I]
    
    down_activation = first_dot_activation @ expert_down    # [T, H]
    if quantize:
      # [B, H]
      # Quantize down proj before expert affinities
      down_activation = down_activation * down_proj_scale.squeeze()
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
      scale = down_activation * local_expert_affinities
    else:
      scale = down_activation
    output_np += scale.astype(output_np.dtype)

  return output_np