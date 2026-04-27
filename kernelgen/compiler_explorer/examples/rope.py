import numpy as np
from nkipy_kernelgen import trace, knob

# Hardcoded dimensions
batch = 2
seq_len = 128
n_heads = 4
head_dim = 128
half_h = head_dim // 2
bs = batch * seq_len
tile_size = [128, 1, 64]

@trace(input_specs=[
    ((bs, n_heads, head_dim), "f32"),
    ((bs, half_h), "f32"),
    ((bs, half_h), "f32"),
])
def rope_kernel(x, freqs_cos, freqs_sin):
    # Broadcast cos/sin to (bs, 1, half_h)
    cos = np.expand_dims(freqs_cos, axis=1)
    sin = np.expand_dims(freqs_sin, axis=1)
    
    knob.knob(cos, mem_space="Sbuf")
    knob.knob(sin, mem_space="Sbuf")

    # Split input into two halves along head_dim
    x0 = x[:, :, :half_h]
    x1 = x[:, :, half_h:]

    # Apply rotation
    out_0 = x0 * cos - x1 * sin
    knob.knob(out_0, mem_space="Sbuf", tile_size=tile_size)

    out_1 = x0 * sin + x1 * cos
    knob.knob(out_1, mem_space="Sbuf", tile_size=tile_size)

    # Concatenate back along head_dim axis
    result = np.concatenate([out_0, out_1], axis=-1)
    knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)
    
    return result
