import numpy as np
from nkipy_kernelgen import trace, knob

# Hardcoded dimensions
batch = 2
n_heads = 4
seq_len = 256
head_dim = 256
tile_size = [1, 128, 128]

scale = 1.0 / np.sqrt(head_dim).item()

@trace(input_specs=[
    ((batch * n_heads, seq_len, head_dim), "f32"),
    ((batch * n_heads, head_dim, seq_len), "f32"),
])
def attention_kernel(q, k_transposed):
    # Score computation (K is pre-transposed to avoid np.transpose)
    bmm_result = np.matmul(q, k_transposed)
    # knob.knob(bmm_result, mem_space="SharedHbm", tile_size=tile_size, reduction_tile=[128])
    
    # When placed to SBUF, the M-dim is the partition dimension
    knob.knob(bmm_result, mem_space="Sbuf", tile_size=tile_size, reduction_tile=[128])

    scores = bmm_result * scale
    knob.knob(scores, mem_space="Sbuf", tile_size=tile_size, partition_dim=1)

    # Softmax
    scores_fp32 = scores.astype(np.float32)

    scores_max = np.max(scores_fp32, axis=-1, keepdims=True)
    knob.knob(scores_max, mem_space="Sbuf", tile_size=[1, 128], reduction_tile=[128], partition_dim=1)

    shifted = scores_fp32 - scores_max
    knob.knob(shifted, mem_space="Sbuf", tile_size=tile_size, partition_dim=1)

    exp_s = np.exp(shifted)
    knob.knob(exp_s, mem_space="Sbuf", tile_size=tile_size, partition_dim=1)

    sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
    knob.knob(sum_exp, mem_space="Sbuf", tile_size=[1, 128], reduction_tile=[128], partition_dim=1)

    result = exp_s / sum_exp
    knob.knob(result, mem_space="SharedHbm", tile_size=tile_size, partition_dim=1)
    return result
