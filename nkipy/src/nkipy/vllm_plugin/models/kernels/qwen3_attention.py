from typing import Optional

import neuronxcc.nki.language as nl
import nkipy.core.typing as nt
import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from nkipy.core import tensor_apis

from .rmsnorm import rmsnorm_kernel
from .rope import apply_rotary_emb_kernel, compute_cos_sin_cache
from .softmax import softmax_kernel


def repeat_kv_kernel(x, n_rep: int):
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=2)


def qwen3_attention_kernel(
    x,
    qkv_weight,
    q_norm_weight,
    k_norm_weight,
    norm_eps,
    n_heads,
    head_dim,
    n_kv_heads,
    cache_k,
    cache_v,
    start_pos: Optional[nt.tensor],
    o_weight,
):
    """Qwen3 attention: QKV projection, QK RMSNorm, RoPE (base=1000000), GQA."""
    is_prefill = start_pos is None
    batch_size, seq_len, _ = x.shape

    n_local_heads = n_heads // dist.get_world_size()
    assert n_local_heads > 0
    n_local_kv_heads = max(1, n_kv_heads // dist.get_world_size())
    n_rep = n_local_heads // n_local_kv_heads

    split_axis = x.ndim - 1
    split0 = n_local_heads * head_dim
    split1 = split0 + n_local_kv_heads * head_dim
    xq, xk, xv = np.split(np.matmul(x, qkv_weight), [split0, split1], axis=split_axis)

    xq = xq.reshape(batch_size, seq_len, n_local_heads, head_dim)
    xk = xk.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)
    xv = xv.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)

    # QK RMSNorm (Qwen3-specific)
    xq = rmsnorm_kernel(xq, q_norm_weight, norm_eps)
    xk = rmsnorm_kernel(xk, k_norm_weight, norm_eps)

    # RoPE with base=1000000
    max_seq_len = cache_k.shape[1]
    freqs_cos, freqs_sin = compute_cos_sin_cache(
        head_dim, max_seq_len, base=1000000, dtype=nl.bfloat16
    )
    if is_prefill:
        freqs_cos = freqs_cos[0:seq_len]
        freqs_sin = freqs_sin[0:seq_len]
    else:
        freqs_cos = tensor_apis.constant(freqs_cos)
        freqs_sin = tensor_apis.constant(freqs_sin)
        freqs_cos = freqs_cos[start_pos]
        freqs_sin = freqs_sin[start_pos]
    xq, xk = apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin)

    if is_prefill:
        cache_k[:, :seq_len] = xk
        cache_v[:, :seq_len] = xv
    else:
        assert seq_len == 1
        cache_k[:, start_pos] = xk
        cache_v[:, start_pos] = xv

    keys = repeat_kv_kernel(cache_k, n_rep)
    values = repeat_kv_kernel(cache_v, n_rep)

    xq = xq.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    k_seq_len = keys.shape[2]
    scores = (xq @ keys.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    scores = scores.astype(nl.bfloat16)

    causal_mask = np.triu(np.ones((k_seq_len, k_seq_len)) * -100000, k=1).astype(scores.dtype)
    causal_mask = tensor_apis.constant(causal_mask)
    if is_prefill:
        scores = scores + np.expand_dims(causal_mask[:seq_len, :k_seq_len], axis=[0, 1])
    else:
        scores = scores + np.expand_dims(causal_mask[start_pos, :k_seq_len], axis=[0, 1])

    attention_weights = softmax_kernel(scores)
    output = attention_weights @ values

    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(batch_size, seq_len, -1)

    output_to_be_reduced = np.matmul(output, o_weight)
    output = cc.all_reduce(
        output_to_be_reduced,
        replica_groups=[list(range(dist.get_world_size()))],
        reduce_op=np.add,
    )
    return output
