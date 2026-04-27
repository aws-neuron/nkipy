"""
Example NKIPy kernel: Reshape operations (contiguous dim merge/split).

Uncomment one (input_shape, output_shape) pair at a time to try different cases.
"""
import numpy as np
from nkipy_kernelgen import trace

# PASS
# -- Merge dims: (2, 128, 256) -> (256, 256) --
input_shape = (2, 128, 256)
output_shape = (256, 256)

# PASS
# -- Split dim: (256, 256) -> (2, 128, 256) --
input_shape = (256, 256)
output_shape = (2, 128, 256)

# PASS
# -- Insert unit dim: (128, 256) -> (128, 1, 256) --
input_shape = (128, 256)
output_shape = (128, 1, 256)

# PASS
# -- Remove unit dim (squeeze): (128, 1, 256) -> (128, 256) --
input_shape = (128, 1, 256)
output_shape = (128, 256)

# PASS
# -- Infer dim with -1: (2, 128, 256) -> (-1, 256) --
input_shape = (2, 128, 256)
output_shape = (-1, 256)

# PASS
# -- Identity reshape (no-op): (128, 256) -> (128, 256) --
input_shape = (128, 256)
output_shape = (128, 256)


@trace(input_specs=[(input_shape, "f32")])
def kernel(x):
    return np.reshape(x, output_shape)
