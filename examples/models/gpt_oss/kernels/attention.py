from typing import Optional

import nki.language as nl
import nkipy.core.typing as nt
import nkipy.distributed.collectives as cc
import numpy as np
import torch.distributed as dist
from nkipy.core import tensor_apis

from .rope import apply_rotary_emb_kernel, compute_cos_sin_cache
from .softmax import softmax_kernel


def repeat_kv_kernel(x, n_rep: int):
    """Repeat key-value heads for grouped-query attention."""
    if n_rep == 1:
        return x
    z = np.repeat(x, n_rep, axis=2)
    return z


def attention_kernel(
    x,
    qkv_weight,
    qkv_bias,
    sinks,
    rope_inv_freq,
    rope_attention_scaling,
    n_heads,
    head_dim,
    n_kv_heads,
    cache_k,
    cache_v,
    start_pos: Optional[nt.tensor],
    o_weight,
    o_bias,
    sliding_window: Optional[int],
):
    """Unified attention kernel for gpt-oss.

    Differences from a plain GQA attention:
      * QKV and output projections carry biases.
      * No QK RMSNorm.
      * Per-head attention sinks: a learned logit per head is concatenated to the
        attention scores before softmax, then dropped afterwards.
      * Sliding-window masking on the layers configured for it.
      * YaRN RoPE (cos/sin baked from precomputed inverse frequencies).

    When start_pos is None: prefill mode (process full context).
    When start_pos is provided: decode mode (process single token).
    """
    is_prefill = start_pos is None
    batch_size, seq_len, _ = x.shape

    n_local_heads = n_heads // dist.get_world_size()
    assert n_local_heads > 0, f"n_local_heads {n_local_heads} is not greater than 0"
    n_local_kv_heads = max(1, n_kv_heads // dist.get_world_size())
    n_rep = n_local_heads // n_local_kv_heads

    # QKV projection (+ bias). GQA: KV's head count differs from Q's.
    split_axis = x.ndim - 1
    split0 = n_local_heads * head_dim
    split1 = split0 + n_local_kv_heads * head_dim
    splits = [split0, split1]
    qkv = np.matmul(x, qkv_weight) + qkv_bias
    xq, xk, xv = np.split(qkv, splits, axis=split_axis)

    xq = xq.reshape(batch_size, seq_len, n_local_heads, head_dim)
    xk = xk.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)
    xv = xv.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)

    # RoPE (YaRN)
    max_seq_len = cache_k.shape[1]
    freqs_cos, freqs_sin = compute_cos_sin_cache(
        rope_inv_freq, max_seq_len, rope_attention_scaling, dtype=nl.bfloat16
    )
    if is_prefill:
        freqs_cos = freqs_cos[0:seq_len]
        freqs_sin = freqs_sin[0:seq_len]
    else:
        # Decode (seq_len==1) and speculative verify (seq_len==K+1) both write at a
        # runtime offset. The absolute positions of the query tokens are
        # start_pos + [0, 1, ..., seq_len-1]; promote comptime numpy arrays to
        # runtime tensors so they can be gathered with that runtime index.
        query_pos = start_pos + np.arange(seq_len, dtype=np.int32)
        freqs_cos = tensor_apis.constant(freqs_cos)
        freqs_sin = tensor_apis.constant(freqs_sin)
        freqs_cos = freqs_cos[query_pos]
        freqs_sin = freqs_sin[query_pos]
    xq, xk = apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin)

    # KV cache update
    if is_prefill:
        cache_k[:, :seq_len] = xk
        cache_v[:, :seq_len] = xv
    else:
        # Scatter the seq_len new K/V rows into the cache at their absolute
        # positions (one row for decode, K+1 contiguous rows for verify).
        cache_k[:, query_pos] = xk
        cache_v[:, query_pos] = xv

    # GQA: repeat KV heads
    keys = repeat_kv_kernel(cache_k, n_rep)
    values = repeat_kv_kernel(cache_v, n_rep)

    # Transpose for attention: BSHD -> BHSD
    xq = xq.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # Attention scores: BHSD @ BHDS -> BHSS
    k_seq_len = keys.shape[2]
    scores = (xq @ keys.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    scores = scores.astype(nl.bfloat16)

    # Causal (+ optional sliding-window) mask, computed at compile time.
    NEG = -100000.0
    full_mask = np.triu(np.ones((k_seq_len, k_seq_len)) * NEG, k=1)
    if sliding_window is not None:
        # Disallow attending further back than `sliding_window` tokens: mask
        # positions where (query_pos - key_pos) >= sliding_window.
        window_mask = np.tril(np.ones((k_seq_len, k_seq_len)) * NEG, k=-sliding_window)
        full_mask = full_mask + window_mask
    causal_mask = full_mask.astype(scores.dtype)
    causal_mask = tensor_apis.constant(causal_mask)

    if is_prefill:
        scores = scores + np.expand_dims(causal_mask[:seq_len, :k_seq_len], axis=[0, 1])
    else:
        # Gather one mask row per query token. For verify (seq_len>1) this yields a
        # block-causal mask: query at start_pos+i attends to keys <= start_pos+i
        # (plus the sliding-window limit, already baked into causal_mask).
        scores = scores + np.expand_dims(
            causal_mask[query_pos, :k_seq_len], axis=[0, 1]
        )

    # Attention sinks: concatenate a per-head learned logit as an extra "key"
    # column, softmax over (keys + sink), then drop the sink probability.
    # sinks: [n_local_heads] -> broadcast to [B, H, S, 1]
    sink_col = np.reshape(sinks, (1, n_local_heads, 1, 1))
    sink_col = np.broadcast_to(
        sink_col, (batch_size, n_local_heads, seq_len, 1)
    ).astype(scores.dtype)
    scores = np.concatenate([scores, sink_col], axis=-1)

    attention_weights = softmax_kernel(scores)
    # Drop the sink column before applying to values.
    attention_weights = attention_weights[:, :, :, :k_seq_len]

    # Apply attention to values: BHSS @ BHSD -> BHSD
    output = attention_weights @ values

    # Transpose back: BHSD -> BSHD
    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(batch_size, seq_len, -1)

    # Output projection (+ bias). Bias is replicated across ranks, so apply only
    # on rank 0 to avoid double-counting after the all-reduce.
    output_to_be_reduced = np.matmul(output, o_weight)
    if dist.get_rank() == 0:
        output_to_be_reduced = output_to_be_reduced + o_bias

    # All-reduce for tensor parallelism
    output = cc.all_reduce(
        output_to_be_reduced,
        replica_groups=[list(range(dist.get_world_size()))],
        reduce_op=np.add,
    )

    return output
