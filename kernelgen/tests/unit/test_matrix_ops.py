"""
Tests for matrix operations: matmul, matmul chains, and batched matmul.

These tests verify that MLIR/LLVM execution matches NumPy CPU execution
for matrix operations.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import nkipy_kernelgen_test, run_kernel_test, Mode


# ============================================================================
# Matmul
# ============================================================================

@pytest.mark.parametrize("shape_a,shape_b", [
    ((128, 64), (64, 256)),    # rectangular
    ((256, 256), (256, 256)),  # square
    ((64, 128), (128, 64)),    # different rectangular
])
def test_matmul(shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(A, B):
        return np.matmul(A, B)

    run_kernel_test(kernel, stop_after="trace", modes=Mode.LLVM)


# ============================================================================
# Chained ops
# ============================================================================

@nkipy_kernelgen_test(
    input_specs=[((128, 64), "f32"), ((64, 256), "f32"), ((128, 256), "f32")],
    stop_after="trace",
    modes=Mode.LLVM,
)
def test_matmul_then_add(A, B, C):
    """Matrix multiplication followed by addition."""
    temp = np.matmul(A, B)
    return np.add(temp, C)


@nkipy_kernelgen_test(
    input_specs=[((128, 64), "f32"), ((128, 64), "f32"), ((64, 256), "f32")],
    stop_after="trace",
    modes=Mode.LLVM,
)
def test_add_then_matmul(A, B, C):
    """Addition followed by matrix multiplication."""
    temp = np.add(A, B)
    return np.matmul(temp, C)


@nkipy_kernelgen_test(
    input_specs=[((128, 64), "f32"), ((64, 256), "f32"), ((256, 128), "f32")],
    stop_after="trace",
    modes=Mode.LLVM,
)
def test_matmul_chain(A, B, C):
    """Chained matrix multiplications."""
    temp = np.matmul(A, B)
    return np.matmul(temp, C)


# ============================================================================
# Batched matmul
# ============================================================================

@pytest.mark.parametrize("shape_a,shape_b", [
    ((4, 128, 64), (4, 64, 256)),        # 3D batched
    ((2, 4, 128, 64), (2, 4, 64, 256)),  # 4D batched
])
def test_batched_matmul(shape_a, shape_b):
    @trace(input_specs=[(shape_a, "f32"), (shape_b, "f32")])
    def kernel(A, B):
        return np.matmul(A, B)

    run_kernel_test(kernel, stop_after="trace", modes=Mode.LLVM)


# ============================================================================
# Batched matmul (end-to-end with knobs)
# ============================================================================

BMM_CONFIGS = [
    (2, 256, 256, 256, [1, 128, 128], [128]),
    (8, 256, 256, 256, [1, 128, 128], [128]),
]


def _bmm_config_id(val):
    B, M, N, K, ts, rt = val
    return f"b{B}_{M}x{N}x{K}_tile{ts[-1]}"


@pytest.mark.parametrize(
    "B, M, N, K, tile_size, reduction_tile",
    BMM_CONFIGS,
    ids=[_bmm_config_id(c) for c in BMM_CONFIGS],
)
def test_bmm_e2e(B, M, N, K, tile_size, reduction_tile):
    """End-to-end batched matmul: trace → knob-driven tiling → NISA → BIR sim."""
    @trace(input_specs=[((B, M, K), "f32"), ((B, K, N), "f32")])
    def bmm_kernel(a, b):
        result = np.matmul(a, b)
        knob.knob(result, mem_space="SharedHbm",
                  tile_size=tile_size, reduction_tile=reduction_tile)
        return result

    run_kernel_test(
        bmm_kernel,
        stop_after="apply-and-strip-transforms",
        modes=Mode.LLVM,
    )
    run_kernel_test(
        bmm_kernel,
        check_ir_contains=["nisa.alloc", "nisa.matmul", "nisa.dma_transpose",
                           "nisa.dma_copy", "nisa.target"],
        modes=Mode.BIR_SIM | Mode.STRING_CHECK,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
