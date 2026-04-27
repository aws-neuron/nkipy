# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ported from nkipy/tests/kernels/llama_decoder_dynamo.py

Complete Llama transformer decoder block with self-attention,
rotary position embeddings, and SwiGLU MLP.

This is the most complex test kernel, combining:
- Embedding lookup (np.take)
- RMSNorm (power, mean, rsqrt)
- Rotary position embeddings (cos, sin, concatenate)
- Multi-head attention with GQA and KV cache
- SwiGLU MLP (sigmoid gating)
- Residual connections
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from harness import run_kernel_test, Mode


@pytest.mark.xfail(reason="np.copy, np.concatenate, np.power, np.cos, np.sin, "
                          "dynamic index assignment, int dtype inputs not yet supported")
def test_llama_decoder():
    seq_len = 7
    hidden = 256
    n_heads = 4
    n_kv_heads = 2
    head_dim = 64
    intermediate = 512
    max_seq = 16
    vocab_size = 1024

    @trace(input_specs=[
        ((1, seq_len), "i32"),                     # token_ids
        ((vocab_size, hidden), "f32"),              # embed_table
        ((seq_len,), "i32"),                        # position_indices
        ((1, seq_len), "i32"),                      # rotary_positions
        ((1, n_kv_heads, max_seq, head_dim), "f32"),# kv_cache
        ((1, 1, seq_len, max_seq), "f32"),          # attn_mask
        ((head_dim // 2,), "f32"),                  # rotary_base
        ((hidden,), "f32"),                         # ln1_weight
        ((hidden, hidden), "f32"),                  # q_weight
        ((n_kv_heads * head_dim, hidden), "f32"),   # k_weight
        ((n_kv_heads * head_dim, hidden), "f32"),   # v_weight
        ((1, n_kv_heads, max_seq, head_dim), "f32"),# v_cache
        ((hidden, hidden), "f32"),                  # attn_out_weight
        ((hidden,), "f32"),                         # ln2_weight
        ((intermediate, hidden), "f32"),            # mlp_up_weight
        ((intermediate, hidden), "f32"),            # mlp_gate_weight
        ((hidden, intermediate), "f32"),            # mlp_down_weight
        ((hidden,), "f32"),                         # final_ln_weight
    ])
    def kernel(token_ids, embed_table, pos_indices, rotary_pos,
               k_cache, attn_mask, rotary_base, ln1_w,
               q_w, k_w, v_w, v_cache, attn_out_w,
               ln2_w, mlp_up_w, mlp_gate_w, mlp_down_w, final_ln_w):
        # Embedding
        embedding = np.take(embed_table, token_ids, axis=0)

        # RoPE frequency computation
        unsqueeze = np.expand_dims(rotary_base, 0)
        unsqueeze_1 = np.expand_dims(unsqueeze[:, 0:], 2)
        expand = np.broadcast_to(unsqueeze_1, [1, head_dim // 2, 1])
        slice_2 = rotary_pos[0:]
        unsqueeze_2 = np.expand_dims(slice_2, 1)
        to_float = unsqueeze_2[:, :, 0:].astype(np.float32)
        view = np.reshape(np.broadcast_to(expand, [1, head_dim // 2, 1]), [1, head_dim // 2, 1])
        view_1 = np.reshape(np.broadcast_to(to_float, [1, 1, seq_len]), [1, 1, seq_len])
        bmm = np.matmul(view, view_1)
        permute = np.transpose(np.reshape(bmm, [1, head_dim // 2, seq_len]), [0, 2, 1])
        cat = np.concatenate([permute, permute], -1)
        cos = np.multiply(np.cos(cat), 1.0)
        sin = np.multiply(np.sin(cat), 1.0)

        # RMSNorm 1
        pow_1 = np.power(embedding, 2)
        mean = np.divide(np.sum(pow_1, axis=(-1,), keepdims=True), pow_1.shape[-1])
        rsqrt = np.divide(1, np.sqrt(np.add(mean, 1e-05)))
        normed = np.multiply(ln1_w, np.multiply(embedding, rsqrt))

        # Q/K/V projections
        normed_2d = np.reshape(normed, [seq_len, hidden])
        q = np.reshape(np.reshape(np.matmul(normed_2d, np.transpose(q_w, [1, 0])),
                       [1, seq_len, hidden]), [1, seq_len, -1, head_dim])
        q = np.transpose(q, [0, 2, 1, 3])
        k = np.transpose(np.reshape(np.reshape(np.matmul(normed_2d, np.transpose(k_w, [1, 0])),
                         [1, seq_len, n_kv_heads * head_dim]), [1, seq_len, -1, head_dim]), [0, 2, 1, 3])
        v = np.transpose(np.reshape(np.reshape(np.matmul(normed_2d, np.transpose(v_w, [1, 0])),
                         [1, seq_len, n_kv_heads * head_dim]), [1, seq_len, -1, head_dim]), [0, 2, 1, 3])

        # Apply RoPE
        cos_unsq = np.expand_dims(cos, 1)
        sin_unsq = np.expand_dims(sin, 1)
        half = head_dim // 2
        q_rot = np.add(np.multiply(q, cos_unsq),
                       np.multiply(np.concatenate([np.negative(q[:, :, :, half:]),
                                                   q[:, :, :, 0:half]], -1), sin_unsq))
        k_rot = np.add(np.multiply(k, cos_unsq),
                       np.multiply(np.concatenate([np.negative(k[:, :, :, half:]),
                                                   k[:, :, :, 0:half]], -1), sin_unsq))

        # Update KV cache
        new_k = np.copy(k_cache)
        new_k[:, :, pos_indices] = k_rot
        new_v = np.copy(v_cache)
        new_v[:, :, pos_indices] = v

        # GQA expand keys
        k_exp = np.reshape(np.copy(np.broadcast_to(
            np.expand_dims(new_k[0:, 0:], 2)[:, :, :, 0:, 0:],
            [1, n_kv_heads, n_heads // n_kv_heads, max_seq, head_dim])),
            [1, n_heads, max_seq, head_dim])
        v_exp = np.reshape(np.copy(np.broadcast_to(
            np.expand_dims(new_v[0:, 0:], 2)[:, :, :, 0:, 0:],
            [1, n_kv_heads, n_heads // n_kv_heads, max_seq, head_dim])),
            [1, n_heads, max_seq, head_dim])

        # Attention
        k_t = np.transpose(k_exp, [0, 1, 3, 2])
        q_3d = np.reshape(np.broadcast_to(q_rot, [1, n_heads, seq_len, head_dim]),
                          [n_heads, seq_len, head_dim])
        k_3d = np.reshape(np.broadcast_to(k_t, [1, n_heads, head_dim, max_seq]),
                          [n_heads, head_dim, max_seq])
        scores = np.multiply(np.reshape(np.matmul(q_3d, k_3d),
                             [1, n_heads, seq_len, max_seq]), 0.125)
        scores = np.add(scores, attn_mask[0:, 0:, 0:])
        scores_max = np.max(scores, axis=-1, keepdims=True)
        softmax = np.divide(np.exp(np.subtract(scores, scores_max)),
                            np.sum(np.exp(np.subtract(scores, scores_max)), axis=-1, keepdims=True))
        attn_out = np.matmul(
            np.reshape(np.broadcast_to(np.copy(softmax), [1, n_heads, seq_len, max_seq]),
                       [n_heads, seq_len, max_seq]),
            np.reshape(np.broadcast_to(v_exp, [1, n_heads, max_seq, head_dim]),
                       [n_heads, max_seq, head_dim]))
        attn_out = np.reshape(np.copy(np.transpose(
            np.reshape(attn_out, [1, n_heads, seq_len, head_dim]), [0, 2, 1, 3])),
            [1, seq_len, -1])

        # Output projection
        out = np.reshape(np.matmul(np.reshape(attn_out, [seq_len, hidden]),
                         np.transpose(attn_out_w, [1, 0])), [1, seq_len, hidden])

        # Residual
        residual = np.add(embedding, out)

        # RMSNorm 2
        pow_2 = np.power(residual, 2)
        mean_2 = np.divide(np.sum(pow_2, axis=(-1,), keepdims=True), pow_2.shape[-1])
        rsqrt_2 = np.divide(1, np.sqrt(np.add(mean_2, 1e-05)))
        normed_2 = np.multiply(ln2_w, np.multiply(residual, rsqrt_2))

        # SwiGLU MLP
        normed_2_2d = np.reshape(normed_2, [seq_len, hidden])
        gate = np.reshape(np.matmul(normed_2_2d, np.transpose(mlp_up_w, [1, 0])),
                          [1, seq_len, intermediate])
        gate_sig = np.divide(1, np.add(1, np.exp(np.negative(gate))))
        gate_out = np.multiply(gate, gate_sig)
        up = np.reshape(np.matmul(normed_2_2d, np.transpose(mlp_gate_w, [1, 0])),
                        [1, seq_len, intermediate])
        mlp_mid = np.multiply(gate_out, up)
        mlp_out = np.reshape(np.matmul(np.reshape(mlp_mid, [seq_len, intermediate]),
                             np.transpose(mlp_down_w, [1, 0])), [1, seq_len, hidden])

        # Residual + final norm
        final_residual = np.add(residual, mlp_out)
        pow_3 = np.power(final_residual, 2)
        mean_3 = np.divide(np.sum(pow_3, axis=(-1,), keepdims=True), pow_3.shape[-1])
        rsqrt_3 = np.divide(1, np.sqrt(np.add(mean_3, 1e-05)))
        final_normed = np.multiply(final_ln_w, np.multiply(final_residual, rsqrt_3))

        # LM head
        lm_out = np.reshape(
            np.matmul(np.reshape(final_normed[:, final_normed.shape[1] - 1:, 0:], [1, hidden]),
                      np.transpose(embed_table, [1, 0])),
            [1, 1, vocab_size])
        return lm_out

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )
