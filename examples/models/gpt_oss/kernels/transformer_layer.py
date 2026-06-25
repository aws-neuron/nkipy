import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist

# Import config from parent directory
from config import Config
from nkipy.core import tensor_apis

# Import kernels from the kernels directory
from .attention import attention_kernel
from .feedforward import feedforward_kernel
from .rmsnorm import rmsnorm_kernel
from .softmax import softmax_kernel


def transformer_layer(
    x,
    start_pos,
    # attention weights
    qkv_weight,
    qkv_bias,
    o_weight,
    o_bias,
    sinks,
    input_weight,
    post_attention_weight,
    # moe weights
    router_weight,
    router_bias,
    gate_up_weight,
    gate_up_bias,
    down_weight,
    down_bias,
    # kv cache
    cache_k,
    cache_v,
    configs: Config,
    sliding_window=None,
):
    """Single gpt-oss transformer layer for prefill and decode.

    When start_pos is None: prefill mode (process full context)
    When start_pos is provided: decode mode (process single token)
    """
    # Apply input RMSNorm
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    # Attention
    h1 = attention_kernel(
        norm_x,
        qkv_weight,
        qkv_bias,
        sinks,
        configs.rope_inv_freq,
        configs.rope_attention_scaling,
        configs.num_heads,
        configs.head_dim,
        configs.num_kv_heads,
        cache_k,
        cache_v,
        start_pos=start_pos,
        o_weight=o_weight,
        o_bias=o_bias,
        sliding_window=sliding_window,
    )

    # Residual connection after attention
    z = x + h1

    # Get shapes
    B, L, D = z.shape
    top_k = configs.num_experts_per_tok

    # Apply RMSNorm before MoE
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # Router logits [B, L, n_experts] (with bias)
    router_logits = np.matmul(norm_z, router_weight) + router_bias

    # Initialize output tensor
    output = np.empty_like(z)

    # Process each batch separately
    for b in range(B):
        # Process each token in the sequence
        for t in range(L):
            # Get token input [D]
            token_input = norm_z[b, t, :]

            # Get token routing logits [n_experts]
            token_logits = router_logits[b, t]

            # gpt-oss routing: pick top-k on raw logits, then softmax over the
            # selected logits (NOT softmax-then-topk).
            top_k_logits, top_k_indices = tensor_apis.topk(token_logits, k=top_k)
            top_k_weights = softmax_kernel(top_k_logits)

            # Process through each selected expert
            token_output = tensor_apis.zeros((D), dtype=output.dtype)

            for e in range(top_k):
                expert_idx = top_k_indices[e]
                weight = top_k_weights[e]

                expert_output = feedforward_kernel(
                    token_input,
                    gate_up_weight[expert_idx],
                    gate_up_bias[expert_idx],
                    down_weight[expert_idx],
                    down_bias[expert_idx],
                    configs.swiglu_alpha,
                    configs.swiglu_limit,
                )

                token_output += weight * expert_output

            output[b, t] = token_output

    # All-reduce for tensor parallelism. Expert weights (gate_up/down) are sharded
    # along the intermediate dimension, so the per-rank partial down-projection
    # outputs sum to the full result. The down_proj bias is replicated and added
    # inside feedforward_kernel; to avoid counting it world_size times after the
    # reduction, weight prep zeroes down_bias on all ranks except rank 0.
    output = cc.all_reduce(
        output, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
    )

    # Add residual connection
    final_output = z + output

    return final_output
