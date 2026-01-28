"""Combined transformer layer kernel for Qwen3"""

from kernels.attention import qwen3_attention_kernel
from kernels.ffn import feedforward_kernel
from kernels.rmsnorm import rmsnorm


def transformer_layer_kernel(
    hidden_states,
    # Attention weights
    input_layernorm_weight,
    qkv_weight,
    o_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    # FFN weights
    post_attention_layernorm_weight,
    gate_up_weight,
    down_weight,
    gate_up_bias,
    down_bias,
    config,
    compute_dtype,
):
    """
    Combined transformer layer kernel: Attention + FFN with residual connections.

    This kernel calls existing attention and FFN kernels, avoiding code duplication
    while keeping all computation on device.

    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        input_layernorm_weight: [hidden_size]
        qkv_weight: [hidden_size, (n_heads + 2*n_kv_heads) * head_dim]
        o_weight: [(n_heads * head_dim), hidden_size]
        q_norm_weight: [head_dim] - RMSNorm weight for Q
        k_norm_weight: [head_dim] - RMSNorm weight for K
        cos: [max_model_len, head_dim]
        sin: [max_model_len, head_dim]
        post_attention_layernorm_weight: [hidden_size]
        gate_up_weight: [hidden_size, 2 * intermediate_size]
        down_weight: [intermediate_size, hidden_size]
        gate_up_bias: [2 * intermediate_size] or None
        down_bias: [hidden_size] or None
        config: Qwen3Config
        compute_dtype: computation dtype

    Returns:
        output: [batch_size, seq_len, hidden_size]
    """

    hidden_states = qwen3_attention_kernel(
        hidden_states=hidden_states,
        input_layernorm_weight=input_layernorm_weight,
        qkv_weight=qkv_weight,
        o_weight=o_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos=cos,
        sin=sin,
        config=config,
        compute_dtype=compute_dtype,
    )

    ffn_residual = hidden_states

    hidden_states = rmsnorm(
        hidden_states, post_attention_layernorm_weight, config.rms_norm_eps
    )

    ffn_output = feedforward_kernel(
        x=hidden_states,
        gate_up_weight=gate_up_weight,
        down_weight=down_weight,
        gate_up_bias=gate_up_bias,
        down_bias=down_bias,
    )

    output = ffn_output + ffn_residual

    return output
