import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist

from common.config import Config
from common.kernels.feedforward import feedforward_kernel
from common.kernels.rmsnorm import rmsnorm_kernel

from .attention import attention_kernel


def transformer_layer(
    x,
    start_pos,
    # weights
    qkv_weight,
    o_weight,
    input_weight,
    # kv cache
    cache_k,
    cache_v,
    post_attention_weight,
    gate_up_weight,
    down_weight,
    configs: Config,
):
    """
    Single transformer layer for Llama3.

    Compared to Qwen3:
    - No q_norm_weight / k_norm_weight (no QK RMSNorm)
    - No router_weight (dense FFN, not MoE)
    - Dense feedforward instead of expert routing
    """
    # Apply input RMSNorm
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    # Attention (no QK norm)
    h1 = attention_kernel(
        norm_x,
        qkv_weight,
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

    # Apply RMSNorm before FFN
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # Dense feedforward
    ffn_out = feedforward_kernel(norm_z, gate_up_weight, down_weight)

    # All-reduce for tensor parallelism
    ffn_out = cc.all_reduce(
        ffn_out, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
    )

    # Residual connection
    final_output = z + ffn_out

    return final_output
