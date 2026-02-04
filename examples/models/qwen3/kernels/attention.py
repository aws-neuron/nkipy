import neuronxcc.nki.language as nl
import nkipy.core.typing as nt
import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from config import Config

from .feedforward import repeat_kv_kernel
from .rmsnorm import rmsnorm_kernel
from .rope import apply_rotary_emb_kernel
from .softmax import softmax_kernel


def attention_kernel(
    x,
    freqs_cos,
    freqs_sin,
    qkv_weight,
    q_norm_weight,
    k_norm_weight,
    norm_eps,
    n_heads,
    head_dim,
    n_kv_heads,
    cache_k,
    cache_v,
    start_pos,
    mask,
    o_weight,
    is_nkipy: bool,
    is_prefill: bool,
):
    # FIXME: L should not show up here for a bucketed kernel
    B, L, _ = x.shape

    n_local_heads = n_heads // dist.get_world_size()
    assert n_local_heads > 0, f"n_local_heads {n_local_heads} is not greater than 0"
    n_local_kv_heads = max(1, n_kv_heads // dist.get_world_size())
    n_rep = n_local_heads // n_local_kv_heads

    # GQA, KV's head dim and Q's are not equal
    split_axis = x.ndim - 1
    split0 = n_local_heads * head_dim
    split1 = split0 + n_local_kv_heads * head_dim
    splits = [split0, split1]
    xq, xk, xv = np.split(np.matmul(x, qkv_weight), splits, axis=split_axis)

    xq = xq.reshape(B, L, n_local_heads, head_dim)
    xk = xk.reshape(B, L, n_local_kv_heads, head_dim)
    xv = xv.reshape(B, L, n_local_kv_heads, head_dim)

    xq = rmsnorm_kernel(xq, q_norm_weight, norm_eps)
    xk = rmsnorm_kernel(xk, k_norm_weight, norm_eps)

    # RoPE #2
    xq, xk = apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin)

    # [:B, start_pos:start_pos + L, :, :]
    if is_nkipy:
        if is_prefill:
            cache_k[:, :L] = xk
            cache_v[:, :L] = xv
        else:
            assert L == 1, "L must be 1 for decode"
            cache_k[:, start_pos] = xk
            cache_v[:, start_pos] = xv
    else:
        cache_k[:, start_pos[0] : start_pos[0] + L] = xk
        cache_v[:, start_pos[0] : start_pos[0] + L] = xv

    # GQA
    xk = repeat_kv_kernel(cache_k, n_rep)
    xv = repeat_kv_kernel(cache_v, n_rep)

    xq = xq.transpose(0, 2, 1, 3)
    xk = xk.transpose(0, 2, 1, 3)
    xv = xv.transpose(0, 2, 1, 3)

    # FIXME: now it has `max_L` as the last dim, ["B, HN, L or 1, max_L"]
    attention = (xq @ xk.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    attention = attention.astype(nl.bfloat16)

    if mask is not None:
        attention = np.add(attention, np.expand_dims(mask[0:L, :], axis=[0, 1]))

    masked_attention = attention

    attention = softmax_kernel(masked_attention)
    output = attention @ xv

    output = np.transpose(output, (0, 2, 1, 3))
    output = np.reshape(output, newshape=(B, L, -1))

    output_to_be_reduced = np.matmul(output, o_weight)

    if is_nkipy and dist.get_world_size() > 1:
        output = cc.all_reduce(
            output_to_be_reduced,
            replica_groups=[list(range(dist.get_world_size()))],
            reduce_op=np.add,
        )
    else:
        output = output_to_be_reduced

    return output, cache_k, cache_v


def layer_wise_attention(
    x,
    start_pos,
    mask,
    # weights
    qkv_weight,
    o_weight,
    input_weight,
    q_norm_weight,
    k_norm_weight,
    # rope
    freqs_cos,
    freqs_sin,
    # kv cache
    cache_k: nt.mutable_tensor,
    cache_v: nt.mutable_tensor,
    configs: Config,
    is_nkipy: bool,
):
    # fix circular import
    norm_x = rmsnorm_kernel(x, input_weight, configs.norm_eps)

    L = x.shape[1]
    if mask is not None:
        # CTE
        freqs_cos = freqs_cos[0:L]
        freqs_sin = freqs_sin[0:L]
    else:
        # TKG
        freqs_cos = freqs_cos[start_pos]
        freqs_sin = freqs_sin[start_pos]

    h1, cache_k, cache_v = attention_kernel(
        norm_x,
        freqs_cos,
        freqs_sin,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        configs.norm_eps,
        configs.n_heads,
        configs.head_dim,
        configs.n_kv_heads,
        cache_k,
        cache_v,
        start_pos,
        mask,
        o_weight,
        is_nkipy,
        is_prefill=True,
    )

    z = x + h1
    return z, cache_k, cache_v
