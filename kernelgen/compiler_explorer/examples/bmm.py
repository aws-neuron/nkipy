import numpy as np
from nkipy_kernelgen import trace, knob

batch = 8
M, N, K = 256, 256, 256

@trace(input_specs=[
    ((batch, M, K), "f32"),
    ((batch, K, N), "f32"),
])
def bmm_kernel(a, b):
    result = np.matmul(a, b)
    knob.knob(result, mem_space="SharedHbm", tile_size=[16, 128, 128], reduction_tile=[128])
    return result
