"""
End-to-end tests for multi-output kernels.

Verifies that kernels returning tuples of tensors compile and produce
correct results through LLVM JIT, BIR simulation, and hardware execution.
"""

from nkipy_kernelgen.trace import trace
from harness import run_kernel_test, Mode


# ============================================================================
# Multi-output kernel definitions
# ============================================================================


@trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
def add_and_sub(a, b):
    """Return both sum and difference."""
    return a + b, a - b


@trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
def add_and_mul(a, b):
    """Return both sum and product."""
    s = a + b
    p = a * b
    return s, p


@trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
def three_outputs(a, b):
    """Return three outputs: sum, difference, product."""
    return a + b, a - b, a * b


# ============================================================================
# Tests: Tracing
# ============================================================================


def test_multi_output_traces():
    """Verify that multi-output kernels trace to MLIR with correct func signature."""
    module = add_and_sub.to_mlir()
    mlir_str = str(module)
    # Function should have two result types
    assert "-> (tensor<256x256xf32>, tensor<256x256xf32>)" in mlir_str


def test_three_output_traces():
    """Verify three-output kernel traces correctly."""
    module = three_outputs.to_mlir()
    mlir_str = str(module)
    assert (
        "-> (tensor<256x256xf32>, tensor<256x256xf32>, tensor<256x256xf32>)" in mlir_str
    )


# ============================================================================
# Tests: LLVM JIT verification
# ============================================================================


def test_add_and_sub_llvm(request):
    """Two-output kernel: sum and difference, verified via LLVM JIT."""
    run_kernel_test(
        add_and_sub,
        stop_after="trace",
        modes=Mode.LLVM,
        request=request,
    )


def test_add_and_mul_llvm(request):
    """Two-output kernel: sum and product, verified via LLVM JIT."""
    run_kernel_test(
        add_and_mul,
        stop_after="trace",
        modes=Mode.LLVM,
        request=request,
    )


def test_three_outputs_llvm(request):
    """Three-output kernel verified via LLVM JIT."""
    run_kernel_test(
        three_outputs,
        stop_after="trace",
        modes=Mode.LLVM,
        request=request,
    )


# ============================================================================
# Tests: Full pipeline (BIR_SIM)
# ============================================================================


def test_add_and_sub_bir_sim(request):
    """Two-output kernel through full pipeline + BIR simulation."""
    run_kernel_test(
        add_and_sub,
        modes=Mode.BIR_SIM,
        request=request,
    )


def test_add_and_mul_bir_sim(request):
    """Two-output kernel through full pipeline + BIR simulation."""
    run_kernel_test(
        add_and_mul,
        modes=Mode.BIR_SIM,
        request=request,
    )
