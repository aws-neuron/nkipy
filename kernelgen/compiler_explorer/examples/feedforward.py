"""
Example NKIPy kernel: Feedforward Network (SwiGLU)

This kernel implements a feedforward layer with SwiGLU activation:
1. Gate+Up projection: x @ gate_up_weight -> split into gate and up
2. SwiGLU activation: SiLU(gate) * up
3. Down projection: result @ down_weight
"""
import numpy as np
from nkipy_kernelgen import trace, knob

# Hardcoded dimensions
batch_size = 256
hidden_size = 256
intermediate_size = 256

# Tile sizes
matmul_tile = [128, 128, 128]  # TILE_M, TILE_N, TILE_K
elementwise_tile = [128, 128]  # TILE_M, TILE_N

@trace(input_specs=[
    ((batch_size, hidden_size), "f32"),                      # x
    ((hidden_size, 2 * intermediate_size), "f32"),           # gate_up_weight
    ((intermediate_size, hidden_size), "f32"),               # down_weight
])
def feedforward_kernel(x, gate_up_weight, down_weight):
    """Feedforward network: Gate+Up projection -> SwiGLU -> Down projection"""
    # Gate and Up projection
    mm_gup = np.matmul(x, gate_up_weight)
    knob.knob(mm_gup, mem_space="Sbuf", tile_size=matmul_tile)

    # Split into gate and up components
    split_axis = mm_gup.ndim - 1
    gate, up = np.split(mm_gup, 2, axis=split_axis)

    # Apply SiLU activation to gate: sigmoid(gate) * gate
    # Break down sigmoid into individual ops so each can be tiled
    neg_gate = -gate
    knob.knob(neg_gate, mem_space="Sbuf", tile_size=elementwise_tile)

    exp_neg_gate = np.exp(neg_gate)
    knob.knob(exp_neg_gate, mem_space="Sbuf", tile_size=elementwise_tile)

    one_plus_exp = exp_neg_gate + 1.0
    knob.knob(one_plus_exp, mem_space="Sbuf", tile_size=elementwise_tile)

    sigmoid_gate = 1.0 / one_plus_exp
    knob.knob(sigmoid_gate, mem_space="Sbuf", tile_size=elementwise_tile)

    swish_gate = gate * sigmoid_gate
    knob.knob(swish_gate, mem_space="Sbuf", tile_size=elementwise_tile)

    # Element-wise multiplication (gating)
    gated = swish_gate * up
    knob.knob(gated, mem_space="Sbuf", tile_size=elementwise_tile)

    # Down projection
    output = np.matmul(gated, down_weight)
    knob.knob(output, mem_space="SharedHbm", tile_size=matmul_tile)

    return output
