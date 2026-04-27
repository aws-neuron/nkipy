"""
End-to-end tests for matmul + add using the complete C++ pass pipeline.

This test runs the full pipeline:
1. Trace Python code to MLIR
2. Run passes through nkipy-opt (assign-linalg-op-ids, knob-driven-tiling, etc.)
3. Convert to NISA dialect (linalg-to-nisa, prepare-for-nki)
4. Simulate with neuron-cc

Run with: pytest tests/e2e/test_matmul_add.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Parameterized shapes and tile sizes
# ============================================================================

# (M, N, K, matmul_tile, matmul_reduction_tile, add_tile)
# SBUF budget per partition (trn1): 180,224 bytes.
# Peak SBUF during matmul = result_buf + lhs_buf + rhs_buf, where:
#   result_buf = (M/tileM) * (N/tileN) * tileN * 4   (full MxN in SBUF)
#   lhs_buf    = (K/tileK) * 2 * tileK * 4            (full K for one BLOCK_M)
#   rhs_buf    = (K/tileK) * 2 * tileK * 4            (full K for one BLOCK_N)
# With tile=128: result = M*N/32, each operand = K*8. Total <= 180,224.
MATMUL_ADD_CONFIGS = [
    # Small matmul: blocking degenerates to 1 (tile == dim for M and N)
    (128, 128, 128, [128, 128], [128], [128, 128]),
    # Standard cases with block size 2
    (256, 256, 256, [128, 128], [128], [128, 128]),
    (1024, 1024, 1024, [128, 128], [128], [128, 128]),
    (2048, 2048, 2048, [128, 128], [128], [128, 128]),
    (4096, 1024, 1024, [128, 128], [128], [128, 128]),
    (1024, 4096, 1024, [128, 128], [128], [128, 128]),
    (2048, 1024, 2048, [128, 128], [128], [128, 128]),
    (1024, 2048, 2048, [128, 128], [128], [128, 128]),
    (2048, 2048, 1024, [128, 128], [128], [128, 128]),
]


def _config_id(val):
    M, N, K, mt, rt, at = val
    return f"{M}x{N}x{K}_mt{'x'.join(map(str, mt))}_rt{rt[0]}_at{'x'.join(map(str, at))}"


# ============================================================================
# Test Cases
# ============================================================================

@pytest.mark.parametrize(
    "M, N, K, matmul_tile, matmul_reduction_tile, add_tile",
    MATMUL_ADD_CONFIGS,
    ids=[_config_id(c) for c in MATMUL_ADD_CONFIGS],
)
def test_matmul_sbuf_add_hbm(M, N, K, matmul_tile, matmul_reduction_tile, add_tile):
    """
    Test matmul + add with SBUF intermediate.

    Pattern: result = matmul(A, B) + bias
    - matmul output: SBUF intermediate
    - final result: HBM output
    """
    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        c = np.matmul(a, b)
        knob.knob(c, mem_space="Sbuf", tile_size=matmul_tile, reduction_tile=matmul_reduction_tile)

        # Add outputs to SharedHbm (returned from kernel)
        result = c + bias
        knob.knob(result, mem_space="SharedHbm", tile_size=add_tile)

        return result

    run_kernel_test(
        matmul_add_kernel,
        check_ir_contains=[
            "nisa.alloc", "nisa.matmul", "nisa.tensor_tensor_arith",
            "nisa.dma_transpose", "nisa.dma_copy", "nisa.target",
        ],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


@pytest.mark.parametrize(
    "M, N, K, matmul_tile, matmul_reduction_tile, add_tile",
    MATMUL_ADD_CONFIGS,
    ids=[_config_id(c) for c in MATMUL_ADD_CONFIGS],
)
def test_matmul_hbm_add_hbm(M, N, K, matmul_tile, matmul_reduction_tile, add_tile):
    """
    Test matmul + add with HBM intermediate (no SBUF buffer reuse).
    """
    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel_hbm(a, b, bias):
        c = np.matmul(a, b)
        knob.knob(c, mem_space="SharedHbm", tile_size=matmul_tile, reduction_tile=matmul_reduction_tile)
        result = c + bias
        knob.knob(result, mem_space="SharedHbm", tile_size=add_tile)
        return result

    run_kernel_test(
        matmul_add_kernel_hbm,
        check_ir_contains=["nisa.alloc", "nisa.matmul", "nisa.target"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK | Mode.HW,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
