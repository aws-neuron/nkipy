import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist

from ..config import Config
from .feedforward import feedforward_kernel
from .rmsnorm import rmsnorm_kernel
from .attention import attention_kernel


def transformer_layer(
    x,
    start_pos,
    qkv_weight,
    o_weight,
    input_weight,
    cache_k,
    cache_v,
    post_attention_weight,
    gate_up_weight,
    down_weight,
    configs: Config,
):
    """Single Llama3 transformer layer (dense FFN, no QK norm)."""
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    h1 = attention_kernel(
        norm_x, qkv_weight, configs.num_heads, configs.head_dim,
        configs.num_kv_heads, cache_k, cache_v, start_pos=start_pos,
        o_weight=o_weight,
    )

    z = x + h1
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)
    ffn_out = feedforward_kernel(norm_z, gate_up_weight, down_weight)

    ffn_out = cc.all_reduce(
        ffn_out, replica_groups=[list(range(dist.get_world_size()))], reduce_op=np.add
    )

    return z + ffn_out
