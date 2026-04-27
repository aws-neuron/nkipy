# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MLP with gated activation (SwiGLU).

test_mlp_swiglu: compiler-friendly version (passes)
test_mlp_swiglu_original: original dynamo-traced version (xfail, see comments)
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


def test_mlp_swiglu():
    batch = 256
    hidden = 256
    intermediate = 256

    matmul_tile = [128, 128]
    matmul_reduction = [128]
    elementwise_tile = [128, 128]

    @trace(input_specs=[
        ((batch, hidden), "f32"),                      # x
        ((hidden, 2 * intermediate), "f32"),           # gate_up_weight
        ((intermediate, hidden), "f32"),               # down_weight
    ])
    def kernel(x, gate_up_weight, down_weight):
        # Combined gate+up projection
        mm_gup = np.matmul(x, gate_up_weight)
        knob.knob(mm_gup, mem_space="Sbuf", tile_size=matmul_tile, reduction_tile=matmul_reduction)

        # Split into gate and up
        split_axis = mm_gup.ndim - 1
        gate, up = np.split(mm_gup, 2, axis=split_axis)

        # SiLU(gate) = gate * sigmoid(gate)
        neg_gate = -gate
        knob.knob(neg_gate, mem_space="Sbuf", tile_size=elementwise_tile)

        exp_neg = np.exp(neg_gate)
        knob.knob(exp_neg, mem_space="Sbuf", tile_size=elementwise_tile)

        one_plus_exp = exp_neg + 1.0
        knob.knob(one_plus_exp, mem_space="Sbuf", tile_size=elementwise_tile)

        sigmoid = 1.0 / one_plus_exp
        knob.knob(sigmoid, mem_space="Sbuf", tile_size=elementwise_tile)

        swish_gate = gate * sigmoid
        knob.knob(swish_gate, mem_space="Sbuf", tile_size=elementwise_tile)

        # Gating
        gated = swish_gate * up
        knob.knob(gated, mem_space="Sbuf", tile_size=elementwise_tile)

        # Down projection
        output = np.matmul(gated, down_weight)
        knob.knob(output, mem_space="SharedHbm", tile_size=matmul_tile, reduction_tile=matmul_reduction)

        return output

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )



@pytest.mark.xfail(reason="1D input + 3D intermediates: DMA partition mismatch and expand_shape legalization")
def test_mlp_swiglu_original():
    """Original dynamo-traced SwiGLU — 1D input, separate matmuls, 3D reshapes.

    Remaining issues after AnnotateMemorySpace resolveViewConflicts fix:
    - 1D input reshape(x, [1, hidden]) creates partition dim = 1, but DMA
      requires partition >= 128 for HBM->SBUF transpose
    - 3D expand_shape of SBUF allocs can't be legalized consistently
      (expand_shape assumes contiguous layout, legalization interleaves)
    """
    hidden = 256
    intermediate = 256

    @trace(input_specs=[
        ((intermediate, hidden), "f32"),  # gate weight
        ((hidden,), "f32"),               # input vector
        ((intermediate, hidden), "f32"),  # up weight
        ((hidden, intermediate), "f32"),  # down weight
    ])
    def kernel(gate_w, x, up_w, down_w):
        gate_wt = np.transpose(gate_w, [1, 0])
        view = np.reshape(x, [1, hidden])
        mm = np.matmul(view, gate_wt)
        view_1 = np.reshape(mm, [1, 1, intermediate])
        sigmoid = 1 / (1 + np.exp(-view_1))
        mul = np.multiply(view_1, sigmoid)

        up_wt = np.transpose(up_w, [1, 0])
        view_2 = np.reshape(x, [1, hidden])
        mm_1 = np.matmul(view_2, up_wt)
        view_3 = np.reshape(mm_1, [1, 1, intermediate])
        mul_1 = np.multiply(mul, view_3)

        down_wt = np.transpose(down_w, [1, 0])
        view_4 = np.reshape(mul_1, [1, intermediate])
        mm_2 = np.matmul(view_4, down_wt)
        view_5 = np.reshape(mm_2, [1, 1, hidden])
        return view_5

    run_kernel_test(
        kernel,
        modes=Mode.BIR_SIM | Mode.HW,
        rtol=1e-3,
        atol=1e-3,
    )
