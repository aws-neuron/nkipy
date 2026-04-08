import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from nkipy.core import tensor_apis

from ..config import Config
from .feedforward import feedforward_kernel
from .rmsnorm import rmsnorm_kernel
from .softmax import softmax_kernel
from .qwen3_attention import qwen3_attention_kernel


def qwen3_transformer_layer(
    x,
    start_pos,
    qkv_weight,
    o_weight,
    input_weight,
    q_norm_weight,
    k_norm_weight,
    cache_k,
    cache_v,
    post_attention_weight,
    router_weight,
    gate_up_weight,
    down_weight,
    configs: Config,
):
    """Single Qwen3 transformer layer (MoE FFN, QK RMSNorm)."""
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    h1 = qwen3_attention_kernel(
        norm_x, qkv_weight, q_norm_weight, k_norm_weight,
        configs.norm_eps, configs.num_heads, configs.head_dim,
        configs.num_kv_heads, cache_k, cache_v, start_pos=start_pos,
        o_weight=o_weight,
    )

    z = x + h1

    B, L, D = z.shape
    n_experts = router_weight.shape[-1]
    top_k = configs.num_experts_per_tok

    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # Router scores [B, L, n_experts]
    router_logits = np.matmul(norm_z, router_weight)

    output = np.empty_like(z)

    for b in range(B):
        for t in range(L):
            token_input = norm_z[b, t, :]
            token_logits = router_logits[b, t]
            token_logits = softmax_kernel(token_logits)

            top_k_logits, top_k_indices = tensor_apis.topk(token_logits, k=top_k)
            top_k_logits /= np.sum(top_k_logits, axis=-1, keepdims=True)

            token_output = tensor_apis.zeros((D), dtype=output.dtype)
            for e in range(top_k):
                expert_idx = top_k_indices[e]
                weight = top_k_logits[e]
                expert_output = feedforward_kernel(
                    token_input, gate_up_weight[expert_idx], down_weight[expert_idx]
                )
                token_output += weight * expert_output

            output[b, t] = token_output

    output = cc.all_reduce(
        output, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
    )

    return z + output
