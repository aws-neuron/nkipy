import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from nkipy.core import tensor_apis

from config import Config

from .attention import attention_kernel
from .feedforward import feedforward_kernel, shared_expert_kernel
from .linear_attention import gated_delta_net_kernel
from .rmsnorm import rmsnorm_kernel
from .softmax import softmax_kernel


def _moe_block(
    norm_z,
    z,
    router_weight,
    gate_up_weight,
    down_weight,
    shared_gate_proj_weight,
    shared_up_proj_weight,
    shared_down_proj_weight,
    shared_expert_gate_weight,
    configs: Config,
):
    """MoE block with routed experts + shared expert (common to both layer types)."""
    B, L, D = z.shape
    n_experts = router_weight.shape[-1]
    top_k = configs.num_experts_per_tok

    # Router scores
    router_logits = np.matmul(norm_z, router_weight)

    # Routed expert output
    routed_output = np.empty_like(z)
    for b in range(B):
        for t in range(L):
            token_input = norm_z[b, t, :]
            token_logits = router_logits[b, t]
            token_logits = softmax_kernel(token_logits)
            top_k_logits, top_k_indices = tensor_apis.topk(token_logits, k=top_k)
            top_k_logits /= np.sum(top_k_logits, axis=-1, keepdims=True)

            token_output = tensor_apis.zeros((D), dtype=routed_output.dtype)
            for e in range(top_k):
                expert_idx = top_k_indices[e]
                weight = top_k_logits[e]
                expert_output = feedforward_kernel(
                    token_input, gate_up_weight[expert_idx], down_weight[expert_idx]
                )
                token_output += weight * expert_output
            routed_output[b, t] = token_output

    # Shared expert output
    norm_z_flat = norm_z.reshape(-1, D)
    shared_output = shared_expert_kernel(
        norm_z_flat,
        shared_gate_proj_weight,
        shared_up_proj_weight,
        shared_down_proj_weight,
    )

    # Shared expert gating
    shared_gate = 1.0 / (
        1.0 + np.exp(-(np.matmul(norm_z_flat, shared_expert_gate_weight)).astype(np.float32))
    )
    shared_gate = shared_gate.astype(shared_output.dtype)
    shared_output = shared_gate * shared_output
    shared_output = shared_output.reshape(B, L, D)

    # Combine routed + shared
    output = routed_output + shared_output

    # All-reduce for tensor parallelism (skip for TP=1)
    if dist.get_world_size() > 1:
        output = cc.all_reduce(
            output, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
        )

    final = z + output
    return final.astype(z.dtype)


def transformer_layer_full_attn(
    x,
    start_pos,
    # Attention weights
    qkv_weight,
    o_weight,
    input_weight,
    q_norm_weight,
    k_norm_weight,
    cache_k,
    cache_v,
    # MoE weights
    post_attention_weight,
    router_weight,
    gate_up_weight,
    down_weight,
    shared_gate_proj_weight,
    shared_up_proj_weight,
    shared_down_proj_weight,
    shared_expert_gate_weight,
    configs: Config,
):
    """Transformer layer with full attention (every 4th layer in Qwen3.5)."""
    # Pre-attention RMSNorm
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    # Full attention with output gating and partial RoPE
    h1 = attention_kernel(
        norm_x,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        configs.norm_eps,
        configs.num_heads,
        configs.head_dim,
        configs.num_kv_heads,
        configs.partial_rotary_factor,
        configs.rope_theta,
        cache_k,
        cache_v,
        start_pos=start_pos,
        o_weight=o_weight,
    )

    z = x + h1

    # Pre-MoE RMSNorm
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # MoE with shared expert
    return _moe_block(
        norm_z,
        z,
        router_weight,
        gate_up_weight,
        down_weight,
        shared_gate_proj_weight,
        shared_up_proj_weight,
        shared_down_proj_weight,
        shared_expert_gate_weight,
        configs,
    )


def transformer_layer_linear_attn(
    x,
    start_pos,
    # Linear attention weights
    qkv_weight,
    z_weight,
    b_weight,
    a_weight,
    conv_weight,
    dt_bias,
    A_log,
    linear_norm_weight,
    out_weight,
    input_weight,
    conv_state,
    recurrent_state,
    # MoE weights
    post_attention_weight,
    router_weight,
    gate_up_weight,
    down_weight,
    shared_gate_proj_weight,
    shared_up_proj_weight,
    shared_down_proj_weight,
    shared_expert_gate_weight,
    configs: Config,
):
    """Transformer layer with linear attention (GatedDeltaNet) for Qwen3.5."""
    # Pre-attention RMSNorm
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    # Gated Delta Net (linear attention)
    h1 = gated_delta_net_kernel(
        norm_x,
        qkv_weight,
        z_weight,
        b_weight,
        a_weight,
        conv_weight,
        dt_bias,
        A_log,
        linear_norm_weight,
        out_weight,
        configs.norm_eps,
        configs.linear_num_key_heads,
        configs.linear_num_value_heads,
        configs.linear_key_head_dim,
        configs.linear_value_head_dim,
        configs.linear_conv_kernel_dim,
        conv_state,
        recurrent_state,
        start_pos=start_pos,
    )

    z = x + h1

    # Pre-MoE RMSNorm
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # MoE with shared expert
    return _moe_block(
        norm_z,
        z,
        router_weight,
        gate_up_weight,
        down_weight,
        shared_gate_proj_weight,
        shared_up_proj_weight,
        shared_down_proj_weight,
        shared_expert_gate_weight,
        configs,
    )
