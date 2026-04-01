from typing import Optional

import nkipy.core.typing as nt
import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from nkipy.core import tensor_apis


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def silu(x):
    x_f32 = x.astype(np.float32)
    return x_f32 * sigmoid(x_f32)


def softplus(x):
    return np.log(1.0 + np.exp(x))


def l2norm(x, eps=1e-6):
    inv_norm = 1.0 / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * inv_norm


def rmsnorm_gated(x, weight, gate, eps):
    """RMSNormGated: weight * rmsnorm(x) * silu(gate)

    Used in the GatedDeltaNet output normalization.
    weight is initialized to ones (not the 1+w convention).
    """
    x_f32 = x.astype(np.float32)
    weight_f32 = weight.astype(np.float32)
    variance = np.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    x_normed = x_f32 / np.sqrt(variance + eps)
    x_normed = weight_f32 * x_normed
    x_normed = x_normed * silu(gate)
    return x_normed.astype(x.dtype)


def causal_conv1d_prefill(x, conv_weight, kernel_size):
    """Causal 1D depthwise convolution for prefill.

    Args:
        x: (B, C, S) input (traced tensor)
        conv_weight: (C, kernel_size) depthwise conv weights (traced tensor)
        kernel_size: int (compile-time constant)

    Returns:
        output: (B, C, S) convolved output
        conv_state: (B, C, kernel_size) state for decode
    """
    B, C, S = x.shape
    # Create zero padding as a compile-time constant, then promote to traced
    zeros_pad = tensor_apis.constant(
        np.zeros((B, C, kernel_size - 1), dtype=np.float32).astype(x.dtype)
    )
    padded = np.concatenate([zeros_pad, x], axis=2)

    # Depthwise convolution via shifted sums
    # PyTorch conv1d convention: weight[0] is the most recent timestep
    # With left-padding of K-1 zeros, output[t] = sum_k weight[k] * padded[t + K - 1 - k]
    # Which means shift = k (not kernel_size - 1 - k)
    output = padded[:, :, 0 : S] * np.expand_dims(
        conv_weight[:, 0], axis=(0, 2)
    )
    for k in range(1, kernel_size):
        output = output + padded[:, :, k : S + k] * np.expand_dims(
            conv_weight[:, k], axis=(0, 2)
        )

    # Save conv state (last kernel_size values of padded input)
    conv_state = padded[:, :, -(kernel_size):]

    # SiLU activation
    output = output * sigmoid(output)
    return output, conv_state


def causal_conv1d_decode(x, conv_state, conv_weight, kernel_size):
    """Causal 1D depthwise convolution for single-token decode.

    Args:
        x: (B, C, 1) input token
        conv_state: (B, C, kernel_size) previous state
        conv_weight: (C, kernel_size) depthwise conv weights
        kernel_size: int

    Returns:
        output: (B, C, 1) convolved output
        new_conv_state: (B, C, kernel_size) updated state
    """
    # Shift state left, append new token
    new_conv_state = np.concatenate([conv_state[:, :, 1:], x], axis=2)
    # Dot product along kernel dimension
    output = np.sum(
        new_conv_state * np.expand_dims(conv_weight, axis=0), axis=2, keepdims=True
    )
    # SiLU activation
    output = output * sigmoid(output)
    return output, new_conv_state


def gated_delta_net_kernel(
    x,
    qkv_weight,
    z_weight,
    b_weight,
    a_weight,
    conv_weight,
    dt_bias,
    A_log,
    norm_weight,
    out_weight,
    norm_eps,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_kernel_size,
    conv_state,
    recurrent_state,
    start_pos: Optional[nt.tensor],
):
    """Gated Delta Net (linear attention) kernel for Qwen3.5.

    Implements the recurrent formulation of the gated delta rule.
    """
    is_prefill = start_pos is None
    batch_size, seq_len, _ = x.shape

    ws = dist.get_world_size()
    n_local_k_heads = num_k_heads // ws
    n_local_v_heads = num_v_heads // ws
    key_dim_local = n_local_k_heads * head_k_dim
    value_dim_local = n_local_v_heads * head_v_dim
    v_per_k = n_local_v_heads // n_local_k_heads

    # Projections
    mixed_qkv = np.matmul(x, qkv_weight)  # (B, S, key_dim_local*2 + value_dim_local)
    z = np.matmul(x, z_weight)  # (B, S, value_dim_local)
    b = np.matmul(x, b_weight)  # (B, S, n_local_v_heads)
    a = np.matmul(x, a_weight)  # (B, S, n_local_v_heads)

    # Causal conv1d
    mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # (B, C, S)
    if is_prefill:
        mixed_qkv_t, new_conv_state = causal_conv1d_prefill(
            mixed_qkv_t, conv_weight, conv_kernel_size
        )
    else:
        mixed_qkv_t, new_conv_state = causal_conv1d_decode(
            mixed_qkv_t, conv_state, conv_weight, conv_kernel_size
        )

    # Update conv state
    conv_state[:] = new_conv_state

    mixed_qkv = mixed_qkv_t.transpose(0, 2, 1)  # (B, S, C)

    # Split into Q, K, V
    query, key, value = np.split(
        mixed_qkv,
        [key_dim_local, key_dim_local * 2],
        axis=-1,
    )

    # Reshape to head dims
    query = query.reshape(batch_size, seq_len, n_local_k_heads, head_k_dim)
    key = key.reshape(batch_size, seq_len, n_local_k_heads, head_k_dim)
    value = value.reshape(batch_size, seq_len, n_local_v_heads, head_v_dim)

    # L2 normalize Q and K
    query = l2norm(query.astype(np.float32)).astype(query.dtype)
    key = l2norm(key.astype(np.float32)).astype(key.dtype)

    # Compute gating: g = -exp(A_log) * softplus(a + dt_bias)
    beta = sigmoid(b)  # (B, S, n_local_v_heads)
    g = -np.exp(A_log.astype(np.float32)) * softplus(
        a.astype(np.float32) + dt_bias.astype(np.float32)
    )

    # Repeat K heads to match V heads if needed
    if v_per_k > 1:
        query = np.repeat(query, v_per_k, axis=2)
        key = np.repeat(key, v_per_k, axis=2)

    # Scale query
    scale = 1.0 / np.sqrt(np.float32(head_k_dim))

    # Transpose to (B, heads, S, dim) for recurrence
    query = query.transpose(0, 2, 1, 3).astype(np.float32) * scale
    key = key.transpose(0, 2, 1, 3).astype(np.float32)
    value = value.transpose(0, 2, 1, 3).astype(np.float32)
    beta = beta.transpose(0, 2, 1).astype(np.float32)  # (B, n_v_heads, S)
    g = g.transpose(0, 2, 1).astype(np.float32)  # (B, n_v_heads, S)

    # Recurrent gated delta rule
    # recurrent_state: (B, n_local_v_heads, head_k_dim, head_v_dim)
    state = recurrent_state.astype(np.float32)

    # Collect outputs in a list to avoid assigning traced tensors to numpy slices
    output_steps = []

    for i in range(seq_len):
        q_t = query[:, :, i, :]  # (B, n_v_heads, head_k_dim)
        k_t = key[:, :, i, :]
        v_t = value[:, :, i, :]  # (B, n_v_heads, head_v_dim)
        g_t = np.expand_dims(np.expand_dims(np.exp(g[:, :, i]), -1), -1)
        beta_t = np.expand_dims(beta[:, :, i], -1)  # (B, n_v_heads, 1)

        # Decay state
        state = state * g_t
        # Retrieve from state
        kv_mem = np.sum(state * np.expand_dims(k_t, -1), axis=-2)
        # Delta update
        delta = (v_t - kv_mem) * beta_t
        state = state + np.expand_dims(k_t, -1) * np.expand_dims(delta, -2)
        # Query state -> (B, n_v_heads, head_v_dim)
        step_out = np.sum(state * np.expand_dims(q_t, -1), axis=-2)
        # Add sequence dimension: (B, n_v_heads, 1, head_v_dim)
        output_steps.append(np.expand_dims(step_out, 2))

    # Update recurrent state
    recurrent_state[:] = state.astype(recurrent_state.dtype)

    # Concatenate along sequence dim: (B, n_v_heads, seq_len, head_v_dim)
    core_output = np.concatenate(output_steps, axis=2)

    # Transpose back: (B, heads, S, dim) -> (B, S, heads, dim) -> (B, S, heads*dim)
    core_output = core_output.transpose(0, 2, 1, 3)

    # RMSNormGated: reshape to 2D for norm, then back
    core_flat = core_output.reshape(-1, head_v_dim)
    z_flat = z.reshape(batch_size, seq_len, n_local_v_heads, head_v_dim)
    z_flat = z_flat.reshape(-1, head_v_dim)
    core_flat = rmsnorm_gated(core_flat, norm_weight, z_flat, norm_eps)
    core_output = core_flat.reshape(batch_size, seq_len, -1)

    # Output projection
    output_to_be_reduced = np.matmul(core_output, out_weight)

    # All-reduce (skip for TP=1, evaluated at trace time)
    if dist.get_world_size() > 1:
        output = cc.all_reduce(
            output_to_be_reduced,
            replica_groups=[list(range(dist.get_world_size()))],
            reduce_op=np.add,
        )
    else:
        output = output_to_be_reduced

    # Cast back to input dtype (bfloat16) to match hidden_states
    return output.astype(x.dtype)
