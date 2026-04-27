"""
End-to-end tests for 3D elementwise operations.

These tests exercise the generalized rank-R pipeline with 3D tensors:
1. Trace 3D Python ops to MLIR
2. Tile with 3D tile_size (middle dims must be 1)
3. Legalize layout: 3D -> 5D physical -> 2D collapse
4. Lower to NISA dialect
5. Simulate and validate against NumPy

Run with: pytest tests/e2e/test_3d_elementwise.py -v
"""

import pytest

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Test Cases
# ============================================================================

def test_3d_add_chain():
    """
    Test 3D add chain: result = (a + b) + c

    Shape: (256, 2, 256) with tile [128, 1, 128]
    - Intermediate (a+b) stored in SBUF (triggers 5D physical layout)
    - Result stored in SharedHbm

    This exercises:
    - 3D -> 5D SBUF legalization: [128, 2, 2, 2, 128]
    - 3-level tiled copy loops (HBM <-> SBUF)
    - Collapse to 2D for NISA compute ops
    - Named op reconstruction (linalg.add with 2D operands)
    """
    B, M, N = 256, 2, 256
    tile_size = [128, 1, 128]

    @trace(input_specs=[((B, M, N), "f32"), ((B, M, N), "f32"), ((B, M, N), "f32")])
    def add_chain_3d(a, b, c):
        intermediate = a + b
        knob.knob(intermediate, mem_space="Sbuf", tile_size=tile_size)

        result = intermediate + c
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)

        return result

    run_kernel_test(
        add_chain_3d,
        check_ir_contains=[
            "nisa.alloc", "nisa.tensor_tensor_arith", "nisa.target",
        ],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


def test_3d_add_hbm_only():
    """
    Test 3D add with HBM intermediate (no SBUF legalization needed).

    Shape: (256, 2, 256) with tile [128, 1, 128]
    - Both intermediate and result go to SharedHbm
    - No SBUF layout transformation — serves as a simpler 3D baseline
    """
    B, M, N = 256, 2, 256
    tile_size = [128, 1, 128]

    @trace(input_specs=[((B, M, N), "f32"), ((B, M, N), "f32"), ((B, M, N), "f32")])
    def add_chain_3d_hbm(a, b, c):
        intermediate = a + b
        knob.knob(intermediate, mem_space="SharedHbm", tile_size=tile_size)

        result = intermediate + c
        knob.knob(result, mem_space="SharedHbm", tile_size=tile_size)

        return result

    run_kernel_test(
        add_chain_3d_hbm,
        check_ir_contains=["nisa.alloc", "nisa.target"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
