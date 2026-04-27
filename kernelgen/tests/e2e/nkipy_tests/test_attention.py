# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/attention_dynamo.py

Full attention layer (prefill) with rotary position embeddings and KV cache,
generated from torch dynamo graph.

Operations: transpose, reshape, matmul, expand_dims, broadcast_to, multiply,
            negative, concatenate, copy, dynamic indexing, softmax, divide.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


@pytest.mark.xfail(reason="np.copy, np.concatenate, dynamic index assignment, multiple unsupported ops")
def test_attention_prefill():
    seq_len = 7
    hidden = 256
    n_heads = 4
    n_kv_heads = 2
    head_dim = 64
    max_seq = 16

    @trace(input_specs=[
        ((1, seq_len, hidden), "f32"),                        # input
        ((hidden, hidden), "f32"),                             # q_weight
        ((n_kv_heads * head_dim, hidden), "f32"),              # k_weight
        ((n_kv_heads * head_dim, hidden), "f32"),              # v_weight
        ((1, seq_len, head_dim), "f32"),                       # rotary_cos
        ((1, seq_len, head_dim), "f32"),                       # rotary_sin
        ((seq_len,), "i32"),                                   # position_ids
        ((1, n_kv_heads, max_seq, head_dim), "f32"),           # k_cache
        ((1, n_kv_heads, max_seq, head_dim), "f32"),           # v_cache
        ((1, 1, seq_len, max_seq), "f32"),                     # attn_mask
        ((hidden, hidden), "f32"),                             # o_weight
    ])
    def kernel(x, q_w, k_w, v_w, cos_emb, sin_emb, pos_ids,
               k_cache, v_cache, attn_mask, o_w):
        # Q projection
        q_wt = np.transpose(q_w, [1, 0])
        x_2d = np.reshape(x, [seq_len, hidden])
        q = np.matmul(x_2d, q_wt)
        q = np.reshape(q, [1, seq_len, hidden])
        q = np.reshape(q, [1, seq_len, -1, head_dim])
        q = np.transpose(q, [0, 2, 1, 3])

        # K projection
        k_wt = np.transpose(k_w, [1, 0])
        k = np.matmul(x_2d, k_wt)
        k = np.reshape(k, [1, seq_len, n_kv_heads * head_dim])
        k = np.reshape(k, [1, seq_len, -1, head_dim])
        k = np.transpose(k, [0, 2, 1, 3])

        # V projection
        v_wt = np.transpose(v_w, [1, 0])
        v = np.matmul(x_2d, v_wt)
        v = np.reshape(v, [1, seq_len, n_kv_heads * head_dim])
        v = np.reshape(v, [1, seq_len, -1, head_dim])
        v = np.transpose(v, [0, 2, 1, 3])

        # Apply rotary embeddings to Q
        cos_unsq = np.expand_dims(cos_emb, 1)
        sin_unsq = np.expand_dims(sin_emb, 1)
        q_rot = np.multiply(q, cos_unsq)
        q_half1 = q[:, :, :, 0:head_dim // 2]
        q_half2 = q[:, :, :, head_dim // 2:]
        q_neg = np.negative(q_half2)
        q_cat = np.concatenate([q_neg, q_half1], -1)
        q_rot = np.add(q_rot, np.multiply(q_cat, sin_unsq))

        # Apply rotary embeddings to K
        k_rot = np.multiply(k, cos_unsq)
        k_half1 = k[:, :, :, 0:head_dim // 2]
        k_half2 = k[:, :, :, head_dim // 2:]
        k_neg = np.negative(k_half2)
        k_cat = np.concatenate([k_neg, k_half1], -1)
        k_rot = np.add(k_rot, np.multiply(k_cat, sin_unsq))

        # Update KV cache
        new_k_cache = np.copy(k_cache)
        new_k_cache[:, :, pos_ids] = k_rot
        new_v_cache = np.copy(v_cache)
        new_v_cache[:, :, pos_ids] = v

        # GQA expand
        kv_slice = new_k_cache[0:, 0:]
        kv_unsq = np.expand_dims(kv_slice, 2)
        kv_exp = np.broadcast_to(kv_unsq[:, :, :, :, 0:],
                                 [1, n_kv_heads, n_heads // n_kv_heads, max_seq, head_dim])
        k_full = np.reshape(np.copy(kv_exp), [1, n_heads, max_seq, head_dim])

        v_slice = new_v_cache[0:, 0:]
        v_unsq = np.expand_dims(v_slice, 2)
        v_exp = np.broadcast_to(v_unsq[:, :, :, :, 0:],
                                [1, n_kv_heads, n_heads // n_kv_heads, max_seq, head_dim])
        v_full = np.reshape(np.copy(v_exp), [1, n_heads, max_seq, head_dim])

        # Attention scores
        k_t = np.transpose(k_full, [0, 1, 3, 2])
        q_3d = np.reshape(np.broadcast_to(q_rot, [1, n_heads, seq_len, head_dim]),
                          [n_heads, seq_len, head_dim])
        k_3d = np.reshape(np.broadcast_to(k_t, [1, n_heads, head_dim, max_seq]),
                          [n_heads, head_dim, max_seq])
        scores = np.matmul(q_3d, k_3d)
        scores = np.reshape(scores, [1, n_heads, seq_len, max_seq])
        scores = np.multiply(scores, 1.0 / np.sqrt(head_dim).item())

        # Add mask and softmax
        scores = np.add(scores, attn_mask[0:, 0:, 0:])
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(np.subtract(scores, scores_max))
        attn_weights = np.divide(scores_exp, np.sum(scores_exp, axis=-1, keepdims=True))

        # Weighted sum of values
        aw_3d = np.reshape(np.broadcast_to(np.copy(attn_weights),
                           [1, n_heads, seq_len, max_seq]),
                           [n_heads, seq_len, max_seq])
        v_3d = np.reshape(np.broadcast_to(v_full, [1, n_heads, max_seq, head_dim]),
                          [n_heads, max_seq, head_dim])
        context = np.matmul(aw_3d, v_3d)
        context = np.reshape(context, [1, n_heads, seq_len, head_dim])
        context = np.transpose(context, [0, 2, 1, 3])
        context = np.reshape(np.copy(context), [1, seq_len, -1])

        # Output projection
        o_wt = np.transpose(o_w, [1, 0])
        out_2d = np.reshape(context, [seq_len, hidden])
        out = np.matmul(out_2d, o_wt)
        return np.reshape(out, [1, seq_len, hidden])

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )
