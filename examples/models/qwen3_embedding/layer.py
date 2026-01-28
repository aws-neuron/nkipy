"""
Qwen3 transformer layer using separate kernels.

This module provides an ALTERNATIVE implementation that uses separate kernels
for attention, RMSNorm, and FFN operations instead of the fused transformer
layer kernel."""

import numpy as np
from config import Qwen3Config
from nkipy.runtime import DeviceKernel, DeviceTensor


class Qwen3Layer:
    """Single transformer layer using separate kernels.

    Each layer performs:
    1. Attention block (with input norm and residual)
    2. FFN block (with post-attention norm and residual)
    """

    def __init__(
        self,
        layer_id: int,
        config: Qwen3Config,
        # Layer-specific weights
        qkv_weight: DeviceTensor,
        q_norm_weight: DeviceTensor,
        k_norm_weight: DeviceTensor,
        o_weight: DeviceTensor,
        input_layernorm_weight: DeviceTensor,
        post_attention_layernorm_weight: DeviceTensor,
        gate_up_weight: DeviceTensor,
        down_weight: DeviceTensor,
        # RoPE tables (shared across layers)
        cos: DeviceTensor,
        sin: DeviceTensor,
        # Shared kernels (compiled once, used by all layers)
        shared_attention_kernel: DeviceKernel,
        shared_rmsnorm_kernel: DeviceKernel,
        shared_ffn_kernel: DeviceKernel,
    ):
        self.layer_id = layer_id
        self.config = config

        # Layer-specific weights
        self.qkv_weight = qkv_weight
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.o_weight = o_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.gate_up_weight = gate_up_weight
        self.down_weight = down_weight

        # RoPE tables
        self.cos = cos
        self.sin = sin

        # Shared kernels
        self.attention_kernel = shared_attention_kernel
        self.rmsnorm_kernel = shared_rmsnorm_kernel
        self.ffn_kernel = shared_ffn_kernel

        # Zero bias tensors (Qwen3 doesn't use bias)
        self.gate_up_bias = DeviceTensor.from_numpy(
            np.zeros(2 * config.intermediate_size, dtype=config.dtype),
            f"gate_up_bias_zero_L{layer_id}",
        )
        self.down_bias = DeviceTensor.from_numpy(
            np.zeros(config.hidden_size, dtype=config.dtype),
            f"down_bias_zero_L{layer_id}",
        )

    def forward(self, hidden_states: DeviceTensor) -> DeviceTensor:
        """Forward pass through the layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        OUTPUT_PREFIX = "output"

        # 1. Attention block (includes input norm and residual)
        attn_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"attn_output_L{self.layer_id}"
        )

        self.attention_kernel(
            inputs={
                "hidden_states": hidden_states,
                "input_layernorm_weight": self.input_layernorm_weight,
                "qkv_weight": self.qkv_weight,
                "o_weight": self.o_weight,
                "q_norm_weight": self.q_norm_weight,
                "k_norm_weight": self.k_norm_weight,
                "cos": self.cos,
                "sin": self.sin,
            },
            outputs={f"{OUTPUT_PREFIX}0": attn_output},
        )

        hidden_states = attn_output

        # 2. FFN block with residual
        residual = hidden_states

        # Post-attention normalization
        normed_hidden_states = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"normed_hidden_L{self.layer_id}"
        )

        self.rmsnorm_kernel(
            inputs={
                "x": hidden_states,
                "weight": self.post_attention_layernorm_weight,
            },
            outputs={f"{OUTPUT_PREFIX}0": normed_hidden_states},
        )

        # FFN
        ffn_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"ffn_output_L{self.layer_id}"
        )

        self.ffn_kernel(
            inputs={
                "x": normed_hidden_states,
                "gate_up_weight": self.gate_up_weight,
                "down_weight": self.down_weight,
                "gate_up_bias": self.gate_up_bias,
                "down_bias": self.down_bias,
            },
            outputs={f"{OUTPUT_PREFIX}0": ffn_output},
        )

        # Add residual connection
        hidden_states = DeviceTensor.from_numpy(
            ffn_output.numpy() + residual.numpy(), f"ffn_residual_L{self.layer_id}"
        )

        return hidden_states
