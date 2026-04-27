"""
Example NKIPy kernel: Transpose operations.

Uncomment one (input_shape, axes) pair at a time to try different cases.
"""
import numpy as np
from nkipy_kernelgen import trace
from nkipy_kernelgen.knob import knob

# PASS
# -- 2D transpose: (128, 256) with axes [1, 0] --
# Output shape: (256, 128), tile_size: [128, 128]
input_shape = (128, 256)
axes = [1, 0]
tile_size = [128, 128]

# PASS
# -- 3D transpose (swap last two dims): (2, 128, 256) with axes [0, 2, 1] --
# Output shape: (2, 256, 128), tile_size: [1, 128, 128] (batch dim tiled to 1)
input_shape = (2, 128, 256)
axes = [0, 2, 1]
tile_size = [1, 128, 128]

# -- 3D transpose (rotate dims): (2, 128, 256) with axes [1, 2, 0] --
# Output shape: (128, 256, 2), tile_size: [128, 128, 1]
# (Non-unit dims keep order → effectively a reshape, emits copy not transpose)
# input_shape = (2, 128, 256)
# axes = [1, 2, 0]
# tile_size = [128, 128, 1]

# PASS
# -- 3D transpose (reverse dims): (2, 128, 256) with axes [2, 1, 0] --
# Output shape: (256, 128, 2), tile_size: [128, 128, 1]
input_shape = (2, 128, 256)
axes = [2, 1, 0]
tile_size = [128, 128, 1]


@trace(input_specs=[(input_shape, "f32")])
def kernel(x):
    result = np.transpose(x, axes)
    return knob(result, tile_size=tile_size)
