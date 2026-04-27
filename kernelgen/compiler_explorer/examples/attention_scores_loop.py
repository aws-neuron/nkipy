import numpy as np
from nkipy_kernelgen import trace, knob
from nkipy_kernelgen.apis import fori_loop

# Hardcoded dimensions
batch = 2
n_heads = 4
seq_len = 256
head_dim = 256
tile_size = [128, 128]

scale = 1.0 / np.sqrt(head_dim).item()

@trace(input_specs=[
    ((batch * n_heads, seq_len, head_dim), "f32"),
    ((batch * n_heads, head_dim, seq_len), "f32"),
])
def attention_kernel_loop(q, k_transposed):
    init_result = np.empty((batch * n_heads, seq_len, seq_len), dtype=np.float32)

    def body(i, acc):
        q_i = q[i]
        k_i = k_transposed[i]

        # Score computation (K is pre-transposed to avoid np.transpose)
        scores = np.matmul(q_i, k_i) * scale
        knob.knob(scores, mem_space="Sbuf", tile_size=tile_size, reduction_tile=[128])

        # Softmax
        scores_fp32 = scores.astype(np.float32)

        scores_max = np.max(scores_fp32, axis=-1, keepdims=True)
        knob.knob(scores_max, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])

        shifted = scores_fp32 - scores_max
        knob.knob(shifted, mem_space="Sbuf", tile_size=tile_size)

        exp_s = np.exp(shifted)
        knob.knob(exp_s, mem_space="Sbuf", tile_size=tile_size)

        sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
        knob.knob(sum_exp, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])

        softmax_out = exp_s / sum_exp
        knob.knob(softmax_out, mem_space="SharedHbm", tile_size=tile_size)

        acc[i] = softmax_out
        return acc

    results = fori_loop(0, batch * n_heads, body, init_result)
    return results
