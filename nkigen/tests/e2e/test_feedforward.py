"""
End-to-end tests for feedforward network kernel without bias.

The feedforward kernel implements:
1. Gate+Up projection: x @ gate_up_weight -> split into gate and up
2. SwiGLU activation: SiLU(gate) * up
3. Down projection: result @ down_weight

This test runs the full pipeline:
1. Trace Python code to MLIR
2. Run passes through nkipy-opt (assign-linalg-op-ids, knob-driven-tiling, etc.)
3. Convert to NISA dialect (linalg-to-nisa, prepare-for-nki)
4. Simulate with neuron-cc

Run with: pytest tests/e2e/test_feedforward.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.parametrize("batch_size, hidden_size, intermediate_size, matmul_tile, matmul_reduction_tile, elementwise_tile", [
    (256, 256, 256, [128, 128], [128], [128, 128]),
])
def test_feedforward_sbuf(batch_size, hidden_size, intermediate_size,
                          matmul_tile, matmul_reduction_tile, elementwise_tile):
    """
    Test feedforward network kernel with SBUF intermediates.
    SiLU broken down into individual ops, each annotated with a knob.
    """
    @trace(input_specs=[
        ((batch_size, hidden_size), "f32"),
        ((hidden_size, 2 * intermediate_size), "f32"),
        ((intermediate_size, hidden_size), "f32"),
    ])
    def feedforward_kernel(x, gate_up_weight, down_weight):
        """Feedforward network: Gate+Up projection -> SwiGLU -> Down projection"""
        # Gate and Up projection
        mm_gup = np.matmul(x, gate_up_weight)
        knob.knob(mm_gup).tile_op(tile_size=matmul_tile + matmul_reduction_tile).layout(mem_space="Sbuf")

        # Split into gate and up components
        split_axis = mm_gup.ndim - 1
        gate, up = np.split(mm_gup, 2, axis=split_axis)

        # Apply SiLU activation to gate: sigmoid(gate) * gate
        # Break down sigmoid into individual ops so each can be tiled
        neg_gate = -gate
        knob.knob(neg_gate).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        exp_neg_gate = np.exp(neg_gate)
        knob.knob(exp_neg_gate).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        one_plus_exp = exp_neg_gate + 1.0
        knob.knob(one_plus_exp).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        sigmoid_gate = 1.0 / one_plus_exp
        knob.knob(sigmoid_gate).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        swish_gate = gate * sigmoid_gate
        knob.knob(swish_gate).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        # Element-wise multiplication (gating)
        gated = swish_gate * up
        knob.knob(gated).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        # Down projection
        output = np.matmul(gated, down_weight)
        knob.knob(output).tile_op(tile_size=matmul_tile + matmul_reduction_tile).layout(mem_space="SharedHbm")

        return output

    run_kernel_test(
        feedforward_kernel,
        check_ir_contains=[
            "nisa.alloc", "nisa.matmul", "nisa.target",
        ],
        check_ir_not_contains=["transform.named_sequence"],
        rtol=1e-3,  # Relaxed due to accumulated errors across many ops
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK,
    )


@pytest.mark.parametrize("batch_size, hidden_size, intermediate_size, matmul_tile, matmul_reduction_tile, elementwise_tile", [
    (256, 256, 256, [128, 128], [128], [128, 128]),
])
def test_feedforward_sbuf_compact_silu(batch_size, hidden_size, intermediate_size,
                                       matmul_tile, matmul_reduction_tile, elementwise_tile):
    """
    Test feedforward network kernel with compact SiLU expression.

    Same computation as test_feedforward_sbuf but with SiLU written
    as a single expression instead of broken-down ops with per-op knobs.
    """
    @trace(input_specs=[
        ((batch_size, hidden_size), "f32"),
        ((hidden_size, 2 * intermediate_size), "f32"),
        ((intermediate_size, hidden_size), "f32"),
    ])
    def feedforward_kernel(x, gate_up_weight, down_weight):
        """Feedforward network: Gate+Up projection -> SwiGLU -> Down projection"""
        # Gate and Up projection
        mm_gup = np.matmul(x, gate_up_weight)
        knob.knob(mm_gup).tile_op(tile_size=matmul_tile + matmul_reduction_tile).layout(mem_space="Sbuf")

        # Split into gate and up components
        split_axis = mm_gup.ndim - 1
        gate, up = np.split(mm_gup, 2, axis=split_axis)

        # SwiGLU: SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
        gated = gate / (1.0 + np.exp(-gate)) * up
        knob.knob(gated).tile_op(tile_size=elementwise_tile).layout(mem_space="Sbuf")

        # Down projection
        output = np.matmul(gated, down_weight)
        knob.knob(output).tile_op(tile_size=matmul_tile + matmul_reduction_tile).layout(mem_space="SharedHbm")

        return output

    run_kernel_test(
        feedforward_kernel,
        check_ir_contains=[
            "nisa.alloc", "nisa.matmul", "nisa.target",
        ],
        check_ir_not_contains=["transform.named_sequence"],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
