"""
Tests for elementwise operations: binary, unary, scalar, and chained.

These tests verify that MLIR/LLVM execution matches NumPy CPU execution
and that the tracer emits the correct linalg ops.
Tests with knobs also verify KnobDrivenTiling + linalg-to-nisa produces
correct NISA dialect ops.
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from harness import nkipy_kernelgen_test, run_kernel_test, Mode


# ============================================================================
# Binary ops (identical shapes) → linalg.<op>
# ============================================================================

@pytest.mark.parametrize("op,ir_op,shape,dtype,tile_size", [
    (np.add, "linalg.add", (128, 256), "f32", [64, 128]),
    (np.add, "linalg.add", (256, 512), "f32", [128, 256]),
    (np.add, "linalg.add", (128, 128), "f16", [64, 64]),
    (np.subtract, "linalg.sub", (128, 256), "f32", [64, 128]),
    (np.subtract, "linalg.sub", (256, 512), "f32", [128, 256]),
    (np.multiply, "linalg.mul", (128, 256), "f32", [64, 128]),
    (np.multiply, "linalg.mul", (256, 512), "f32", [128, 256]),
])
def test_binary_op(op, ir_op, shape, dtype, tile_size):
    rtol, atol = (0.01, 0.01) if dtype == "f16" else (1e-5, 1e-8)

    @trace(input_specs=[(shape, dtype), (shape, dtype)])
    def kernel(a, b):
        result = op(a, b)
        knob.knob(result, tile_size=tile_size)
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=[ir_op],
        rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32", ir_op],
        rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="linalg-to-nisa",
        check_ir_contains=["nisa.tensor_tensor_arith"],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Divide (custom inputs to avoid division by near-zero)
# ============================================================================

@pytest.mark.parametrize("shape,dtype,tile_size", [
    ((128, 256), "f32", [64, 128]),
    ((256, 512), "f32", [128, 256]),
    ((128, 128), "f16", [64, 64]),
])
def test_divide(shape, dtype, tile_size):
    rtol, atol = (0.01, 0.01) if dtype == "f16" else (1e-5, 1e-8)
    np_dtype = np.float16 if dtype == "f16" else np.float32

    @trace(input_specs=[(shape, dtype), (shape, dtype)])
    def kernel(a, b):
        result = np.divide(a, b)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.random.randn(*shape).astype(np_dtype)
    B = (np.abs(np.random.randn(*shape)) + 0.5).astype(np_dtype)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.div"],
        inputs=[A, B], rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        inputs=[A, B], rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# Scalar ops
# ============================================================================

@pytest.mark.parametrize("op,scalar,tile_size", [
    (np.add, 2.5, [64, 128]),
    (np.multiply, 3.0, [64, 128]),
    (np.subtract, 1.0, [64, 128]),
])
def test_scalar_op(op, scalar, tile_size):
    @trace(input_specs=[((128, 256), "f32")])
    def kernel(a):
        result = op(a, scalar)
        knob.knob(result, tile_size=tile_size)
        return result

    run_kernel_test(kernel, stop_after="trace", modes=Mode.LLVM)

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="linalg-to-nisa",
        check_ir_contains=["nisa.tensor_scalar_arith"],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Unary ops → linalg.<op>
# ============================================================================

@pytest.mark.parametrize("op,ir_op,nisa_op,shape,dtype,tile_size", [
    (np.square, "linalg.square", "nisa.activation", (128, 256), "f32", [64, 128]),
    (np.square, "linalg.square", "nisa.activation", (256, 512), "f32", [128, 256]),
    (np.square, "linalg.square", "nisa.activation", (128, 128), "f16", [64, 64]),
    (np.abs, "linalg.abs", None, (128, 256), "f32", [64, 128]),
    (np.abs, "linalg.abs", None, (128, 128), "f16", [64, 64]),
])
def test_unary_op(op, ir_op, nisa_op, shape, dtype, tile_size):
    rtol, atol = (0.01, 0.01) if dtype == "f16" else (1e-5, 1e-8)

    @trace(input_specs=[(shape, dtype)])
    def kernel(a):
        result = op(a)
        knob.knob(result, tile_size=tile_size)
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=[ir_op],
        rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32", ir_op],
        rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    if nisa_op:
        run_kernel_test(
            kernel, stop_after="linalg-to-nisa",
            check_ir_contains=[nisa_op],
            modes=Mode.STRING_CHECK,
        )


# ============================================================================
# Negative (implemented as 0 - x, emits linalg.generic)
# ============================================================================

@nkipy_kernelgen_test(
    input_specs=[((128, 256), "f32")],
    stop_after="trace",
    modes=Mode.LLVM,
)
def test_negative(A):
    return np.negative(A)


# ============================================================================
# Sqrt (needs positive inputs)
# ============================================================================

@pytest.mark.parametrize("shape,dtype,tile_size", [
    ((128, 256), "f32", [64, 128]),
    ((256, 512), "f32", [128, 256]),
    ((128, 128), "f16", [64, 64]),
])
def test_sqrt(shape, dtype, tile_size):
    rtol, atol = (0.01, 0.01) if dtype == "f16" else (1e-5, 1e-8)
    np_dtype = np.float16 if dtype == "f16" else np.float32

    @trace(input_specs=[(shape, dtype)])
    def kernel(a):
        result = np.sqrt(a)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = np.abs(np.random.randn(*shape)).astype(np_dtype)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.sqrt"],
        inputs=[A], rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.sqrt"],
        inputs=[A], rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


# ============================================================================
# Exp (needs small inputs to avoid overflow)
# ============================================================================

@pytest.mark.parametrize("shape,dtype,tile_size", [
    ((128, 256), "f32", [64, 128]),
    ((256, 512), "f32", [128, 256]),
    ((128, 128), "f16", [64, 64]),
])
def test_exp(shape, dtype, tile_size):
    rtol, atol = (0.01, 0.01) if dtype == "f16" else (1e-5, 1e-8)
    np_dtype = np.float16 if dtype == "f16" else np.float32

    @trace(input_specs=[(shape, dtype)])
    def kernel(a):
        result = np.exp(a)
        knob.knob(result, tile_size=tile_size)
        return result

    np.random.seed(42)
    A = (np.random.randn(*shape) * 0.5).astype(np_dtype)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.exp"],
        inputs=[A], rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32", "linalg.exp"],
        inputs=[A], rtol=rtol, atol=atol,
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="linalg-to-nisa",
        check_ir_contains=["nisa.activation"],
        inputs=[A],
        modes=Mode.STRING_CHECK,
    )


# ============================================================================
# Chained expressions
# ============================================================================

def test_add_then_multiply():
    @trace(input_specs=[((128, 256), "f32"), ((128, 256), "f32")])
    def kernel(A, B):
        temp = np.add(A, B)
        knob.knob(temp, tile_size=[64, 128])
        return np.multiply(temp, 2.0)

    run_kernel_test(
        kernel, stop_after="trace",
        modes=Mode.LLVM,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_add_then_square():
    @trace(input_specs=[((128, 256), "f32"), ((128, 256), "f32")])
    def kernel(A, B):
        result = np.square(np.add(A, B))
        knob.knob(result, tile_size=[64, 128])
        return result

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.add", "linalg.square"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_square_then_divide():
    @trace(input_specs=[((128, 256), "f32"), ((128, 256), "f32")])
    def kernel(A, B):
        squared = np.square(A)
        knob.knob(squared, tile_size=[64, 128])
        return np.divide(squared, B)

    np.random.seed(42)
    A = np.random.randn(128, 256).astype(np.float32)
    B = np.random.randn(128, 256).astype(np.float32) + 1.0

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.square", "linalg.div"],
        inputs=[A, B],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        inputs=[A, B],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_complex_expression():
    @trace(input_specs=[
        ((256, 256), "f32"), ((256, 256), "f32"), ((256, 256), "f32")
    ])
    def kernel(A, B, C):
        squared = np.square(A)
        knob.knob(squared, tile_size=[128, 128])
        sum_result = np.add(squared, B)
        return np.divide(sum_result, C)

    np.random.seed(42)
    A = np.random.randn(256, 256).astype(np.float32)
    B = np.random.randn(256, 256).astype(np.float32)
    C = np.random.randn(256, 256).astype(np.float32) + 1.0

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.square", "linalg.add", "linalg.div"],
        inputs=[A, B, C],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        inputs=[A, B, C],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_square_in_expression():
    @trace(input_specs=[((128, 256), "f32")])
    def kernel(A):
        squared = np.square(A)
        knob.knob(squared, tile_size=[64, 128])
        return np.add(squared, 1.0)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.square"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_exp_in_expression():
    @trace(input_specs=[((128, 128), "f32")])
    def kernel(A):
        squared = np.square(A)
        knob.knob(squared, tile_size=[64, 64])
        return np.exp(squared * 0.1)

    np.random.seed(42)
    A = np.random.randn(128, 128).astype(np.float32)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.exp", "linalg.square"],
        inputs=[A],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        inputs=[A],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


def test_sqrt_in_expression():
    @trace(input_specs=[((128, 128), "f32")])
    def kernel(A):
        squared = np.square(A)
        knob.knob(squared, tile_size=[64, 64])
        return np.sqrt(squared)

    np.random.seed(42)
    A = np.abs(np.random.randn(128, 128)).astype(np.float32)

    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=["linalg.sqrt", "linalg.square"],
        inputs=[A],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=["scf.for", "memory_space = 3 : i32"],
        inputs=[A],
        modes=Mode.LLVM | Mode.STRING_CHECK,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
