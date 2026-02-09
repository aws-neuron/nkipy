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
    """
    Repeat key-value tensors for multi-head attention when using grouped query attention.
    """
    if n_rep == 1:
        return x
    z = np.repeat(x, n_rep, axis=2)
    return z


def attention_kernel(
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
    """
    Unified attention kernel for Qwen3.

    Performs:
    1. QKV projection
    2. QK RMSNorm
    3. RoPE (cos/sin caches are compile-time constants)
    4. KV cache update
    5. GQA (repeat_kv)
    6. Attention scores with causal mask (compile-time constant)
    7. Softmax
    8. Output projection + all-reduce

    Note: Some numpy arrays in this kernel (RoPE frequencies, causal mask) are
    computed from constant arguments rather than runtime tensor inputs. During
    compilation for Trainium, these execute at compile time and become HLO
    constants baked into the compiled graph (similar to Zig's comptime). On CPU,
    they execute as regular numpy.
    """
    is_prefill = start_pos is None
    batch_size, seq_len, _ = x.shape

    n_local_heads = n_heads // dist.get_world_size()
    assert n_local_heads > 0, f"n_local_heads {n_local_heads} is not greater than 0"
    n_local_kv_heads = max(1, n_kv_heads // dist.get_world_size())
    n_rep = n_local_heads // n_local_kv_heads

    # QKV projection
    # GQA: KV's head dim and Q's are not equal
    split_axis = x.ndim - 1
    split0 = n_local_heads * head_dim
    split1 = split0 + n_local_kv_heads * head_dim
    splits = [split0, split1]
    xq, xk, xv = np.split(np.matmul(x, qkv_weight), splits, axis=split_axis)

    xq = xq.reshape(batch_size, seq_len, n_local_heads, head_dim)
    xk = xk.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)
    xv = xv.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)

    # QK RMSNorm
    xq = rmsnorm_kernel(xq, q_norm_weight, norm_eps)
    xk = rmsnorm_kernel(xk, k_norm_weight, norm_eps)

    # RoPE
    max_seq_len = cache_k.shape[1]
    freqs_cos, freqs_sin = compute_cos_sin_cache(
        head_dim, max_seq_len, base=1000000, dtype=nl.bfloat16
    )
    if is_prefill:
        freqs_cos = freqs_cos[0:seq_len]
        freqs_sin = freqs_sin[0:seq_len]
    else:
        # FIXME: start_pos is a tensor, need to handle indexing np with tensor
        freqs_sin = (
            tensor_apis.zeros(
                (freqs_sin.shape[0], freqs_sin.shape[1]), dtype=freqs_sin.dtype
            )
            + freqs_sin
        )
        freqs_cos = (
            tensor_apis.zeros(
                (freqs_cos.shape[0], freqs_cos.shape[1]), dtype=freqs_cos.dtype
            )
            + freqs_cos
        )

        freqs_cos = freqs_cos[start_pos]
        freqs_sin = freqs_sin[start_pos]
    xq, xk = apply_rotary_emb_kernel(xq, xk, freqs_cos, freqs_sin)

    # KV cache update
    if is_prefill:
        cache_k[:, :seq_len] = xk
        cache_v[:, :seq_len] = xv
    else:
        assert seq_len == 1, "seq_len must be 1 for decode"
        cache_k[:, start_pos] = xk
        cache_v[:, start_pos] = xv

    # GQA: repeat KV heads
    keys = repeat_kv_kernel(cache_k, n_rep)
    values = repeat_kv_kernel(cache_v, n_rep)

    # Transpose for attention: BSHD -> BHSD
    xq = xq.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # Compute attention scores: BHSD @ BHDS -> BHSS
    k_seq_len = keys.shape[2]
    scores = (xq @ keys.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(head_dim))
    scores = scores.astype(nl.bfloat16)

    # Comptime: causal mask is a numpy array computed from constants at compile
    # time and baked as an HLO constant. The zeros(...) + array pattern promotes
    # it to a runtime tensor so it can participate in ops with runtime tensors.
    causal_mask = np.triu(np.ones((k_seq_len, k_seq_len)) * -100000, k=1).astype(
        scores.dtype
    )

    # FIXME: need a way to automatically convert constant to tensor
    causal_mask = (
        tensor_apis.zeros((k_seq_len, k_seq_len), dtype=scores.dtype) + causal_mask
    )
    # Apply causal mask
    if is_prefill:
        scores = scores + np.expand_dims(causal_mask[:seq_len, :k_seq_len], axis=[0, 1])
    else:
        scores = scores + np.expand_dims(
            causal_mask[start_pos, :k_seq_len], axis=[0, 1]
        )

    # Softmax
    attention_weights = softmax_kernel(scores)

    # Apply attention to values: BHSS @ BHSD -> BHSD
    output = attention_weights @ values  # [8, 128]

    # Transpose back: BHSD -> BSHD
    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(batch_size, seq_len, -1)

    # Output projection
    output_to_be_reduced = np.matmul(output, o_weight)

    # All-reduce for tensor parallelism
    output = cc.all_reduce(
        output_to_be_reduced,
        replica_groups=[list(range(dist.get_world_size()))],
        reduce_op=np.add,
    )

    return output, cache_k, cache_v
