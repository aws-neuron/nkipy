import os

import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist

# Import config from parent directory
from config import Config

# Import kernels from the kernels directory
from .attention import attention_kernel
from .feedforward import (
    moe_batched,
    moe_dense_masked,
    moe_reference,
)
from .rmsnorm import rmsnorm_kernel

# Which MoE implementation the layer uses (all numerically equivalent; see
# feedforward.py). Selectable via env so P-EAGLE can be swept without edits.
#   reference - one gather+GEMV chain per (token, expert); clearest, default
#   batched   - gather top_k experts, batched GEMV      (best for small N)
#   dense     - all experts as one dense GEMM, masked   (best for N >= ~6)
_MOE_KERNELS = {
    "reference": moe_reference,
    "batched": moe_batched,
    "dense": moe_dense_masked,
}
MOE_KERNEL = os.environ.get("GPT_OSS_MOE_KERNEL", "reference")


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

    top_k = configs.num_experts_per_tok

    # Apply RMSNorm before MoE
    norm_z = rmsnorm_kernel(z, post_attention_weight, configs.norm_eps)

    # MoE feed-forward (implementation selected by GPT_OSS_MOE_KERNEL).
    output = _MOE_KERNELS[MOE_KERNEL](
        norm_z,
        router_weight,
        router_bias,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        top_k,
        configs.swiglu_alpha,
        configs.swiglu_limit,
    )

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
