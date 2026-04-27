"""
Unit tests for the CustomOp Python module.

Tests CustomOp creation, tracing integration (func.call emission),
declaration generation, and NISA body stashing.

Run with: python -m pytest tests/unit/test_custom_op.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace
from nkipy_kernelgen.custom_op import (
    CustomOp,
    emit_custom_op_declaration,
)


# ============================================================================
# CustomOp construction
# ============================================================================


def test_custom_op_init():
    """Test CustomOp constructor."""
    op = CustomOp(
        nisa_mlir="module {}",
        func_name="test_func",
        input_names=["x"],
        output_names=["output"],
        input_shapes=[(128, 128)],
        output_shapes=[(128, 128)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )
    assert op.func_name == "__custom_op__test_func"
    assert op.input_shapes == [(128, 128)]
    assert op.output_shapes == [(128, 128)]


def test_custom_op_name_prefixing():
    """Test that func_name gets __custom_op__ prefix."""
    op = CustomOp(
        nisa_mlir="module {}",
        func_name="my_kernel",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(64, 64)],
        output_shapes=[(64, 64)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )
    assert op.func_name == "__custom_op__my_kernel"


# ============================================================================
# Reference function fallback
# ============================================================================


def test_reference_fn_called_outside_tracing():
    """When called with numpy arrays (not TracedArrays), use reference_fn."""
    ref_fn = lambda x: x * 2.0

    op = CustomOp(
        nisa_mlir="module {}",
        func_name="double",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(4, 4)],
        output_shapes=[(4, 4)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
        reference_fn=ref_fn,
    )
    x = np.ones((4, 4), dtype=np.float32)
    result = op(x)
    np.testing.assert_allclose(result, x * 2.0)


def test_no_reference_fn_raises_outside_tracing():
    """Error when called outside tracing without reference_fn."""
    op = CustomOp(
        nisa_mlir="module {}",
        func_name="no_ref",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(4, 4)],
        output_shapes=[(4, 4)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )
    with pytest.raises(RuntimeError, match="without a reference_fn"):
        op(np.ones((4, 4), dtype=np.float32))


# ============================================================================
# Tracing integration
# ============================================================================


def test_custom_op_emits_func_call_during_tracing():
    """Verify that calling a CustomOp during tracing emits func.call in IR."""
    custom_identity = CustomOp(
        nisa_mlir="module {}",
        func_name="identity_128x128_128x128",
        input_names=["x"],
        output_names=["output"],
        input_shapes=[(128, 128)],
        output_shapes=[(128, 128)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        return custom_identity(x)

    module = kernel.to_mlir()
    mlir_str = str(module)

    # Check that func.call is emitted
    assert "call @__custom_op__identity_128x128_128x128" in mlir_str
    # Check that declaration is emitted
    assert "nkipy.custom_op" in mlir_str
    # Check that NISA body is stashed
    assert "nkipy.custom_op_bodies" in mlir_str


def test_custom_op_shape_mismatch_raises():
    """Error when shape doesn't match during tracing."""
    custom_op = CustomOp(
        nisa_mlir="module {}",
        func_name="expects_64x64",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(64, 64)],
        output_shapes=[(64, 64)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        return custom_op(x)

    with pytest.raises(ValueError, match="Shape mismatch"):
        kernel.to_mlir()


def test_custom_op_wrong_arg_count_raises():
    """Error when wrong number of args during tracing."""
    custom_op = CustomOp(
        nisa_mlir="module {}",
        func_name="expects_two",
        input_names=["x", "y"],
        output_names=["z"],
        input_shapes=[(64, 64), (64, 64)],
        output_shapes=[(64, 64)],
        input_dtypes=["f32", "f32"],
        output_dtypes=["f32"],
    )

    @trace(input_specs=[((64, 64), "f32")])
    def kernel(x):
        return custom_op(x)

    with pytest.raises(ValueError, match="expects 2 inputs, got 1"):
        kernel.to_mlir()


# ============================================================================
# Registry
# ============================================================================


def test_registry_deduplicates_by_func_name():
    """Same CustomOp called twice should only register once in module bodies."""
    custom_op = CustomOp(
        nisa_mlir="module {}",
        func_name="dedup_test",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(128, 128)],
        output_shapes=[(128, 128)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        r1 = custom_op(x)
        r2 = custom_op(r1)
        return r2

    module = kernel.to_mlir()
    mlir_str = str(module)

    # The custom op body should be stashed exactly once despite two call sites
    assert mlir_str.count("__custom_op__dedup_test") >= 2, (
        "expected at least 2 call sites"
    )
    # Only one private declaration should exist
    func_decl_count = mlir_str.count("func.func private @__custom_op__dedup_test")
    assert func_decl_count == 1, f"expected 1 declaration, got {func_decl_count}"


# ============================================================================
# emit_custom_op_declaration
# ============================================================================


def test_emit_custom_op_declaration():
    """Verify emitted func.func has correct signature and attributes."""
    from mlir import ir

    op = CustomOp(
        nisa_mlir="module {}",
        func_name="my_activation_128x128_128x128",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(128, 128)],
        output_shapes=[(128, 128)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = emit_custom_op_declaration(op)

        mlir_str = str(module)
        # Private visibility
        assert "private" in mlir_str
        # Correct name with prefix
        assert "@__custom_op__my_activation_128x128_128x128" in mlir_str
        # nkipy.custom_op marker
        assert "nkipy.custom_op" in mlir_str
        # Input and output types
        assert "128x128xf32" in mlir_str


def test_emit_custom_op_declaration_multi_io():
    """Verify declaration for a multi-input multi-output custom op."""
    from mlir import ir

    op = CustomOp(
        nisa_mlir="module {}",
        func_name="multi_io",
        input_names=["a", "b"],
        output_names=["x", "y"],
        input_shapes=[(64, 64), (32, 32)],
        output_shapes=[(64, 64), (32, 32)],
        input_dtypes=["f32", "f16"],
        output_dtypes=["f32", "f16"],
    )

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = emit_custom_op_declaration(op)

        mlir_str = str(module)
        assert "64x64xf32" in mlir_str
        assert "32x32xf16" in mlir_str


# ============================================================================
# Multi-output custom op tracing
# ============================================================================


def test_multi_output_custom_op_tracing():
    """Verify that a multi-output CustomOp returns a tuple of TracedArrays."""
    custom_split = CustomOp(
        nisa_mlir="module {}",
        func_name="split_128x64_128x64",
        input_names=["x"],
        output_names=["left", "right"],
        input_shapes=[(128, 128)],
        output_shapes=[(128, 64), (128, 64)],
        input_dtypes=["f32"],
        output_dtypes=["f32", "f32"],
    )

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        left, right = custom_split(x)
        return left

    module = kernel.to_mlir()
    mlir_str = str(module)

    assert "call @__custom_op__split_128x64_128x64" in mlir_str
    # Two result types in the call
    assert "128x64xf32" in mlir_str


# ============================================================================
# from_kernel_builder
# ============================================================================


def test_from_kernel_builder():
    """Verify from_kernel_builder produces a CustomOp with real NISA MLIR."""
    import nki.compiler.kernel_builder as nb

    def relu_kernel(x_hbm, out_hbm):
        """ReLU activation: load from HBM, activate in SBUF, store back."""
        x_sbuf = nb.ndarray((128, 128), x_hbm.dtype, nb.sbuf)
        nb.isa.dma_copy(dst=x_sbuf, src=x_hbm[0:128, 0:128])

        out_sbuf = nb.ndarray((128, 128), x_hbm.dtype, nb.sbuf)
        bias = nb.ndarray((128, 1), x_hbm.dtype, nb.sbuf)
        nb.isa.memset(dst=bias, value=0.0)
        scale = nb.ndarray((128, 1), x_hbm.dtype, nb.sbuf)
        nb.isa.memset(dst=scale, value=1.0)

        nb.isa.activation(
            dst=out_sbuf,
            src=x_sbuf,
            bias=bias,
            scale=scale,
            op=nb.isa.activation_function.relu,
        )
        nb.isa.dma_copy(dst=out_hbm[0:128, 0:128], src=out_sbuf)

    op = CustomOp.from_kernel_builder(
        kernel_func=relu_kernel,
        input_specs={"x_hbm": nb.Tensor((128, 128), nb.float32, nb.shared_hbm)},
        output_specs={"out_hbm": nb.Tensor((128, 128), nb.float32, nb.shared_hbm)},
        reference_fn=lambda x: np.maximum(x, 0),
    )

    assert op.func_name.startswith("__custom_op__relu_kernel_")
    assert op.input_shapes == [(128, 128)]
    assert op.output_shapes == [(128, 128)]
    # NISA MLIR should contain real ops from the kernel (generic form)
    for op_name in ["nisa.dma_copy", "nisa.activation", "nisa.memset"]:
        assert op_name in op.nisa_mlir or f'"{op_name}"' in op.nisa_mlir, (
            f"expected {op_name} in NISA MLIR"
        )
    # Reference fn should work
    x = np.array([[1, -2], [-3, 4]], dtype=np.float32)
    np.testing.assert_allclose(op(x), np.array([[1, 0], [0, 4]], dtype=np.float32))


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
