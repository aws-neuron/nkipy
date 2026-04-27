"""
Example NKIPy kernel: Basic element-wise operations

This kernel computes: result = (a + b) * c - d
"""
import numpy as np
from nkipy_kernelgen import trace, knob

M, N, K = 256, 256, 256
add_tile = [128, 128]          # TILE_M, TILE_N
    
@trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
def matmul_add_kernel(a, b, bias):
    # Matmul outputs to SBUF for reuse in the add
    c = np.matmul(a, b)
    knob.knob(c, mem_space="Sbuf", tile_size=[128, 128], reduction_tile=[128])
    
    # Add outputs to SharedHbm (returned from kernel)
    result = c + bias
    knob.knob(result, mem_space="SharedHbm", tile_size=add_tile)
    
    return result