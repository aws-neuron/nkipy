from nkipy_kernelgen import trace, knob
import numpy as np


M, N = 256, 256
tile_size = [128, 128]
reduction_tile_size = [128]


@trace(input_specs=[((M, N), "f32")])
def kernel(x):
    sq = np.square(x.astype(np.float32))
    knob.knob(sq, mem_space="Sbuf", tile_size=tile_size)

    # np.mean(sq, axis=-1) == np.sum(sq, axis=-1) * (1/N)
    sm = np.sum(sq, axis=-1, keepdims=True)
    knob.knob(
        sm,
        mem_space="SharedHbm",
        tile_size=[128],
        reduction_tile=[128],
    )

    result = sm * np.float32(1.0 / N)
    knob.knob(
        result,
        mem_space="SharedHbm",
        tile_size=[128, 1],
    )
    return result
