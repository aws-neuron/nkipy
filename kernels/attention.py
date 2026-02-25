from typing import Optional

import neuronxcc.nki.typing as nt
import numpy as np
from collective import all_gather, all_reduce, reduce_scatter
from config import Config
from nkipy.core import tensor_apis
from nkipy.core.nki_op import wrap_nki_kernel
from parallel_state import get_tp_group, get_tp_size, get_world_group

from .attention_nki.prefill_kernel import flash_attn_prefill, attention_prefill_sw128
from .attention_nki.decode_kernel import flash_attn_decode
from .rmsnorm import rmsnorm
from .rope import rope_yarn
from .softmax import softmax
from .fused_rmsnorm_gemm_nki import fused_rmsnorm_gemm


def repeat_kv_kernel(x: nt.tensor, n_rep: int):
    """
    Repeat key-value tensors for multi-head attention when using grouped query attention.
    """
    if n_rep == 1:
        return x
    z = np.repeat(x, n_rep, axis=2)
    return z


def attention_module(
    hidden_states: nt.tensor,
    input_weight: nt.tensor,
    qkv_weight: nt.tensor,
    cache_k: nt.tensor[nt.mutable],
    cache_v: nt.tensor[nt.mutable],
    o_weight: nt.tensor,
    sliding_window: int,
    sink: nt.tensor,
    qkv_bias: nt.tensor,
    o_bias: nt.tensor,
    start_pos: Optional[nt.tensor],
    cos: nt.tensor,
    sin: nt.tensor,
    config: Config,
    compute_dtype: np.dtype,
    is_neuronpy: bool,
):
    # TODO: split into prefill and decode?
    qkv, cache_k, cache_v = pre_attention(
        hidden_states=hidden_states,
        input_weight=input_weight,
        qkv_weight=qkv_weight,
        cache_k=cache_k,
        cache_v=cache_v,
        qkv_bias=qkv_bias,
        start_pos=start_pos,
        cos=cos,
        sin=sin,
        config=config,
        compute_dtype=compute_dtype,
        is_neuronpy=is_neuronpy,
    )
    is_prefill = start_pos is None

    if is_neuronpy:
        sink = sink.astype(compute_dtype)
        if is_prefill:
            q, k, v = qkv
            b, n_local_kv_heads, _, k_seq_len = k.shape
            if sliding_window == 0:
                causal_mask = np.tril(np.ones((k_seq_len, k_seq_len)), k=0).astype(np.uint8)
                if sliding_window != 0:
                    causal_mask = np.triu(
                        np.tril(np.ones((k_seq_len, k_seq_len)), k=0), k=-(sliding_window - 1)
                    ).astype(np.uint8)
                causal_mask = (
                    tensor_apis.zeros((k_seq_len, k_seq_len), dtype=np.uint8) + causal_mask
                )
                attn_nki = wrap_nki_kernel(
                    flash_attn_prefill,
                    [q, k, v, sink, causal_mask],
                    grid=[b, n_local_kv_heads],
                )
                attn_out = attn_nki(q, k, v, sink, causal_mask)
            else:
                assert n_local_kv_heads == 1
                attn_nki = wrap_nki_kernel(
                    attention_prefill_sw128,
                    [q, k, v, sink, sliding_window],
                    grid=[b, n_local_kv_heads],
                )
                attn_out = attn_nki(q, k, v, sink)
            # Transpose back: BHSD -> BSHD
            attn_out = attn_out.transpose(0, 2, 1, 3)
        else:
            # decode
            b, n_local_kv_heads, d, _ = cache_k.shape
            decode_attn_nki = wrap_nki_kernel(
                flash_attn_decode,
                [qkv, cos, sin, cache_k, cache_v, sink, start_pos, sliding_window],
                grid=[n_local_kv_heads],
            )
            attn_out, cache_k, cache_v = decode_attn_nki(
                qkv, cos, sin, cache_k, cache_v, sink, start_pos
            )
            n_local_heads = qkv.shape[1] - 2 * n_local_kv_heads
            attn_out = attn_out.reshape((b, 1, n_local_heads, d))
    else:
        q, k, v = qkv
        # Neuronpy decode attention or Numpy attention
        attn_out = attention_native(
            q=q,
            k=k,
            v=v,
            sink=sink,
            sliding_window=sliding_window,
            start_pos=start_pos,
            config=config,
            compute_dtype=compute_dtype,
            is_neuronpy=is_neuronpy,
        )
        # Transpose back: BHSD -> BSHD
        attn_out = attn_out.transpose(0, 2, 1, 3)

    out_dtype = qkv.dtype if not isinstance(qkv, tuple) else qkv[0].dtype
    output = post_attention(
        output=attn_out,
        o_weight=o_weight,
        o_bias=o_bias,
        residual=hidden_states,
        compute_dtype=compute_dtype,
        output_dtype=out_dtype,
        is_prefill=is_prefill,
        is_neuronpy=is_neuronpy,
    )
    return output, cache_k, cache_v


def pre_attention(
    *,
    hidden_states: nt.tensor,
    input_weight: nt.tensor,
    qkv_weight: nt.tensor,
    cache_k: nt.tensor[nt.mutable],
    cache_v: nt.tensor[nt.mutable],
    qkv_bias: nt.tensor,
    start_pos: Optional[nt.tensor],
    cos: nt.tensor,
    sin: nt.tensor,
    config: Config,
    compute_dtype: np.dtype,
    is_neuronpy: bool,
):
    """
    Performs:
    1. RMSNorm, sequence parallel enabled for prefill
    2. All Gather if sequence parallel prefill
    3. QKV linear
    4. ROPE
    5. QKV Layout transformation for attention
    """
    hidden_states = hidden_states.astype(compute_dtype)
    qkv_weight = qkv_weight.astype(compute_dtype)
    qkv_bias = qkv_bias.astype(compute_dtype)

    is_prefill = start_pos is None

    if is_prefill:
        # # RMSNORM NKI Version: worse performance due to IO tensor on HBM
        # rmsnorm_nki_op = wrap_nki_kernel(
        #     rmsnorm_nki,
        #     [
        #         np.empty(hidden_states.shape, dtype=hidden_states.dtype),
        #         np.empty(input_weight.shape, dtype=input_weight.dtype),
        #         config.norm_eps,
        #     ],
        # )
        # hidden_states_normed = rmsnorm_nki_op(hidden_states, input_weight)
        hidden_states_normed = rmsnorm(
            hidden_states,
            input_weight,
            config.norm_eps,
            is_neuronpy=is_neuronpy,
        )
        # prefill in seq parallel
        hidden_states_normed = all_gather(
            hidden_states_normed,
            all_gather_dim=0,
            replica_groups=get_tp_group(),
            is_neuronpy=is_neuronpy,
        )
        hidden_states_normed = hidden_states_normed.reshape(
            (config.max_batch_size_per_dp, config.max_model_len, config.hidden_size)
        )
        # seq_len is bucketed
        batch_size, seq_len, _ = hidden_states_normed.shape
        qkv_output = (hidden_states_normed @ qkv_weight).astype(hidden_states_normed.dtype)
        qkv_output = qkv_output + qkv_bias
    else:
        qkv_output = fused_rmsnorm_gemm(
            hidden_states,
            input_weight,
            qkv_weight,
            qkv_bias,
            config.norm_eps,
        )
        # seq_len is bucketed
        batch_size, seq_len, _ = hidden_states.shape

    n_local_heads = config.n_heads // get_tp_size()
    assert n_local_heads > 0, f"n_local_heads {n_local_heads} is not greater than 0"
    n_local_kv_heads = max(1, config.n_kv_heads // get_tp_size())

    if is_neuronpy and not is_prefill:
        # nkipy decode
        qkv = qkv_output.reshape(batch_size, n_local_heads + n_local_kv_heads * 2, config.head_dim)
    else:
        # nkipy prefill or numpy prefill/decode

        # GQA, KV's head dim and Q's are not equal
        split_axis = hidden_states_normed.ndim - 1
        split0 = n_local_heads * config.head_dim
        split1 = split0 + n_local_kv_heads * config.head_dim
        splits = [split0, split1]
        q, k, v = np.split(qkv_output, splits, axis=split_axis)
        q = q.reshape(batch_size, seq_len, n_local_heads, config.head_dim)
        k = k.reshape(batch_size, seq_len, n_local_kv_heads, config.head_dim)
        v = v.reshape(batch_size, seq_len, n_local_kv_heads, config.head_dim)

        q, k = rope_yarn(
            q,
            k,
            cos=cos,
            sin=sin,
            start_pos=start_pos,
        )

        n_rep = n_local_heads // n_local_kv_heads
        # Update cache based on mode
        if is_prefill:
            if is_neuronpy:
                # prepare input layout for kernel
                # BSHD -> BHDS
                k = k.transpose(0, 2, 3, 1)

                cache_k[:, :, :, :seq_len] = k.astype(cache_k.dtype)
                cache_v[:, :seq_len] = v.astype(cache_v.dtype)
                # BSHD -> BDHS
                q = q.transpose(0, 3, 2, 1)
                # BSHD -> BHDS
                v = v.transpose(0, 2, 3, 1)
            else:
                # Numpy code path
                cache_k[:, :, :, :seq_len] = k.transpose(0, 2, 3, 1).astype(cache_k.dtype)
                cache_v[:, :seq_len] = v.astype(cache_v.dtype)
                k = repeat_kv_kernel(k, n_rep)
                v = repeat_kv_kernel(v, n_rep)
                q = q.transpose(0, 2, 1, 3)
                k = k.transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)
        else:
            # Numpy token generation mode: update single position
            np.put_along_axis(
                cache_k,
                start_pos.reshape(batch_size, 1, 1, 1),
                k.astype(cache_k.dtype).reshape(batch_size, n_local_kv_heads, config.head_dim, 1),
                axis=3,
            )
            np.put_along_axis(
                cache_v,
                start_pos.reshape(batch_size, 1, 1, 1),
                v.astype(cache_v.dtype),
                axis=1,
            )
            # BHDS -> BSHD
            k = repeat_kv_kernel(cache_k.transpose(0, 3, 1, 2).astype(compute_dtype), n_rep)
            v = repeat_kv_kernel(cache_v.astype(compute_dtype), n_rep)

            # CPU path: Numpy implementation for reference and testing
            # BSHD -> BHSD
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

        qkv = (q, k, v)

    return qkv, cache_k, cache_v


def post_attention(
    *,
    output: nt.tensor,
    o_weight: nt.tensor,
    o_bias: nt.tensor,
    residual: nt.tensor,
    compute_dtype: np.dtype,
    output_dtype: np.dtype,
    is_prefill: bool,
    is_neuronpy: bool,
):
    """
    Performs:
    1. Output projection
    2. ReduceScatter (prefill) or AllReduce (decode)
    3. Residual connection
    """
    o_weight = o_weight.astype(compute_dtype)
    o_bias = o_bias.astype(compute_dtype)
    original_dtype = residual.dtype

    batch_size, seq_len, _, _ = output.shape

    output = output.reshape(batch_size, seq_len, -1)
    output = (output @ o_weight).astype(output_dtype)
    output = output + o_bias

    # Ensure output maintains input dtype
    assert output.dtype == output_dtype

    # For token generation (is_prefill=False), use all_reduce instead of reduce_scatter
    if is_prefill:
        # prefill in seq parallel
        output = reduce_scatter(
            output.reshape(batch_size * seq_len, -1),
            reduce_scatter_dim=0,
            replica_groups=get_tp_group(),
            is_neuronpy=is_neuronpy,
        )
    else:
        output = all_reduce(
            output,
            replica_groups=get_tp_group(),
            is_neuronpy=is_neuronpy,
        )

    # Add residual connection
    output = residual + output

    if not is_prefill:
        output = all_gather(
            output,
            all_gather_dim=0,
            # replica_groups=get_dp_group(),
            # FIXME: this changes improve the CC performance by avoiding bad topology [0, 8, ...]
            # This introduces duplicated data which needs to be explicitly filtered out
            replica_groups=get_world_group(),
            is_neuronpy=is_neuronpy,
        )

        # attn_hidden_states = attn_hidden_states.reshape(batch_size, hidden_size)
        # the duplicated degree is world size // dp size == tp size
        duplicated_size = get_tp_size()

        # Create base pattern [0, 1, 2, ..., batch_size-1]
        base = np.arange(batch_size)
        
        # Create offsets [0, batch_size*duplicated_size, 2*batch_size*duplicated_size, ...]
        interval = batch_size * duplicated_size
        offsets = np.arange(0, output.shape[0], interval).reshape(-1, 1)
        
        # Add offsets to base pattern using broadcasting
        indices = offsets + base
        indices = indices.reshape(-1).astype(np.int32)

        output = output[indices, :, :]

    return output.astype(original_dtype)


def attention_native(
    *,
    q: nt.tensor,
    k: nt.tensor,
    v: nt.tensor,
    sink: nt.tensor,
    sliding_window: int,
    start_pos: Optional[nt.tensor],
    config: Config,
    compute_dtype: np.dtype,
    is_neuronpy: bool,
):
    """
    Attention implemented using framework-native (neuronpy or numpy) interfaces
    """
    is_prefill = start_pos is None
    batch_size, n_local_heads, seq_len, _ = q.shape
    k_seq_len = k.shape[2]
    sink = sink.astype(compute_dtype)

    # Compute attention scores: BHSD@BHDS -> BH[S|1]S
    scores = (q @ k.transpose(0, 1, 3, 2) / np.sqrt(config.head_dim)).astype(q.dtype)

    causal_mask = np.triu(np.ones((k_seq_len, k_seq_len)) * -100000, k=1).astype(
        q.dtype
    )  # convert type because *np.inf will cast to fp32
    if is_neuronpy:
        causal_mask = (
            tensor_apis.zeros((k_seq_len, k_seq_len), dtype=q.dtype) + causal_mask
        )

    # Apply sliding window mask if specified
    if sliding_window > 0:
        window_mask = np.tril(
            np.ones((k_seq_len, k_seq_len)) * -100000, k=-sliding_window
        ).astype(q.dtype)
        causal_mask = causal_mask + window_mask

    if is_prefill:
        # For NeuronPy to compile
        # scores = scores + causal_mask[None, None, :, :]
        scores += np.expand_dims(causal_mask, axis=[0, 1])
    else:
        scores += np.expand_dims(causal_mask[start_pos], axis=[1, 2])

    sink_expanded = sink.reshape(1, n_local_heads, 1, 1)
    sink_expanded = np.broadcast_to(
        sink_expanded, (batch_size, n_local_heads, seq_len, 1)
    )
    scores = np.concatenate([scores, sink_expanded], axis=-1)

    # Apply softmax
    attention_weights = softmax(scores, is_neuronpy=is_neuronpy)

    # Remove sink dimension after softmax
    attention_weights = attention_weights[:, :, :, :-1]

    # Apply attention to values
    output = (attention_weights @ v).astype(q.dtype)

    # layout: BHSD
    return output
