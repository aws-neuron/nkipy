import nkipy.core.typing as nt
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
    # weights
    qkv_weight,
    o_weight,
    input_weight,
    q_norm_weight,
    k_norm_weight,
    # kv cache
    cache_k: nt.mutable_tensor,
    cache_v: nt.mutable_tensor,
    post_attention_weight,
    router_weight,
    gate_up_weight,
    down_weight,
    configs: Config,
):
    """
    Single transformer layer for both context encoding (prefill) and token generation (decode).

    When start_pos is None: prefill mode (process full context)
    When start_pos is provided: decode mode (process single token)
    """
    # Apply input RMSNorm
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    # Attention
    h1, cache_k, cache_v = attention_kernel(
        norm_x,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        configs.norm_eps,
        configs.num_heads,
        configs.head_dim,
        configs.num_kv_heads,
        cache_k,
        cache_v,
        start_pos=start_pos,
        o_weight=o_weight,
    )

    # Residual connection after attention
    z = x + h1

    # Get shapes
    B, L, D = z.shape
    n_experts = router_weight.shape[-1]
    top_k = configs.num_experts_per_tok

    # Apply RMSNorm before MoE
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # Router scores [B, L, n_experts]
    router_logits = np.matmul(norm_z, router_weight)

    # Initialize output tensor
    output = np.empty_like(z)

    # Process each batch separately
    for b in range(B):
        # Process each token in the sequence
        for t in range(L):
            # Get token input [1, 1, D]
            token_input = norm_z[b : b + 1, t : t + 1, :]

            # Get token routing logits [n_experts]
            token_logits = router_logits[b, t]

            # Apply softmax to normalize routing weights
            token_logits = softmax_kernel(token_logits)

            # Get top-k experts for this token
            top_k_logits, top_k_indices = tensor_apis.topk(token_logits, k=top_k)
            top_k_logits /= np.sum(top_k_logits, axis=-1, keepdims=True)

            # Process through each selected expert
            token_output = tensor_apis.zeros((1, 1, D), dtype=output.dtype)

            for e in range(top_k):
                # Get expert index and weight for this token
                expert_idx = top_k_indices[e : e + 1]
                weight = top_k_logits[e]

                # Process through the selected expert
                expert_output = feedforward_kernel(
                    token_input, gate_up_weight[expert_idx], down_weight[expert_idx]
                )

                # Add weighted output to result
                token_output += weight * expert_output

            # Store the result
            output[b, t] = token_output[0, 0]

    # All-reduce for tensor parallelism
    output = cc.all_reduce(
        output, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
    )

    # Add residual connection
    final_output = z + output

    return final_output, cache_k, cache_v
