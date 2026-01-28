"""Simplified attention kernel for Qwen3 without tensor parallelism"""

import numpy as np
from kernels.rmsnorm import rmsnorm
from kernels.rope import rope_qwen3
from kernels.softmax import softmax


def qwen3_attention_kernel(
    hidden_states,
    input_layernorm_weight,
    qkv_weight,
    o_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    config,
    compute_dtype,
):
    """
    Simplified attention for Qwen3 (no TP, no sliding window, no bias, prefill only)

    Args:
        hidden_states: [batch_size,  seq_len, hidden_size]
        input_layernorm_weight: [hidden_size]
        qkv_weight: [hidden_size, (n_heads + 2*n_kv_heads) * head_dim]
        o_weight: [(n_heads * head_dim), hidden_size]
        q_norm_weight: [n_heads * head_dim] - RMSNorm weight for Q
        k_norm_weight: [n_kv_heads * head_dim] - RMSNorm weight for K
        cos: [max_model_len, head_dim]
        sin: [max_model_len, head_dim]
        config: Qwen3Config
        compute_dtype: computation dtype

    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    original_dtype = hidden_states.dtype
    batch_size, seq_len, hidden_size = hidden_states.shape
    assert hidden_size == config.hidden_size, (
        f"Hidden size mismatch: {hidden_size} != {config.hidden_size}"
    )

    hidden_states = hidden_states.astype(compute_dtype)
    qkv_weight = qkv_weight.astype(compute_dtype)
    o_weight = o_weight.astype(compute_dtype)

    # Store original for residual
    residual = hidden_states

    hidden_states = rmsnorm(hidden_states, input_layernorm_weight, config.rms_norm_eps)

    # no bias for Qwen3
    qkv = hidden_states @ qkv_weight

    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    split0 = n_heads * head_dim
    split1 = split0 + n_kv_heads * head_dim
    q, k, v = np.split(qkv, [split0, split1], axis=-1)

    q = q.reshape(batch_size, seq_len, n_heads, head_dim)
    k = k.reshape(batch_size, seq_len, n_kv_heads, head_dim)
    v = v.reshape(batch_size, seq_len, n_kv_heads, head_dim)

    q = rmsnorm(q, q_norm_weight, config.rms_norm_eps)
    k = rmsnorm(k, k_norm_weight, config.rms_norm_eps)

    q, k = rope_qwen3(q, k, cos, sin)

    # Repeat K, V for GQA (n_heads / n_kv_heads times)
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=2)
        v = np.repeat(v, n_rep, axis=2)

    # [batch_size, n_heads, seq_len, head_dim]
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores = scores.astype(compute_dtype)

    causal_mask = np.triu(np.ones((seq_len, seq_len)) * -10000.0, k=1).astype(
        compute_dtype
    )
    scores = scores + causal_mask[None, None, :, :]

    attn_weights = softmax(scores)

    attn_output = (attn_weights @ v).astype(compute_dtype)

    attn_output = attn_output.transpose(0, 2, 1, 3)
    attn_output = attn_output.reshape(batch_size, seq_len, n_heads * head_dim)

    # no bias for Qwen3
    output = attn_output @ o_weight

    output = output + residual

    return output.astype(original_dtype)
