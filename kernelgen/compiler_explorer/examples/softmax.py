import numpy as np
from nkipy_kernelgen import trace, knob

# Hardcoded dimensions
M = 256
N = 256
tile_size = [128, 128]

@trace(input_specs=[((M, N), "f32")])
def softmax_kernel(x):
    x_fp32 = x.astype(np.float32)
    
    x_max = np.max(x_fp32, axis=-1, keepdims=True)
    knob.knob(x_max, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])
    
    shifted = x_fp32 - x_max
    knob.knob(shifted, mem_space="Sbuf", tile_size=tile_size)
    
    exp_x = np.exp(shifted)
    knob.knob(exp_x, mem_space="Sbuf", tile_size=tile_size)
    
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    knob.knob(sum_exp, mem_space="Sbuf", tile_size=[128], reduction_tile=[128])
    
    result = exp_x / sum_exp
    knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
    return result