"""
End-to-end tests for custom op integration.

Tests the full flow: tracing with CustomOp -> pipeline passes -> resolve-custom-ops.
The custom op replaces the activation function in a kernel with a
pre-compiled NISA function body (either hand-written or built via kernel_builder).

Run with: pytest tests/e2e/test_custom_op.py -v
"""

import pytest
import numpy as np

from nkipy_kernelgen import trace, knob
from nkipy_kernelgen.custom_op import CustomOp
from harness import run_kernel_test, Mode

import nki.compiler.kernel_builder as nb


def _make_silu_custom_op(M, N, tile_p=128, tile_f=128):
    """Create a CustomOp with real NISA MLIR compiled via kernel_builder.

    Uses kernel_builder to compile a real SiLU activation, then extracts the
    MLIR string and passes it to the direct CustomOp() constructor.  This tests
    the raw-constructor path with genuine NISA ops (dma_copy, activation, etc.)
    rather than a hand-written stub.
    """
    shape = (M, N)

    def silu_kernel(x_hbm, out_hbm):
        import nki.language as nl

        n_row_tiles = M // tile_p
        n_col_tiles = N // tile_f
        for r in nl.affine_range(n_row_tiles):
            for t in nl.affine_range(n_col_tiles):
                x_sbuf = nb.ndarray((tile_p, tile_f), x_hbm.dtype, nb.sbuf)
                nb.isa.dma_copy(
                    dst=x_sbuf,
                    src=x_hbm[
                        r * tile_p : (r + 1) * tile_p, t * tile_f : (t + 1) * tile_f
                    ],
                )

                out_sbuf = nb.ndarray((tile_p, tile_f), x_hbm.dtype, nb.sbuf)

                bias = nb.ndarray((tile_p, 1), x_hbm.dtype, nb.sbuf)
                nb.isa.memset(dst=bias, value=0.0)
                scale = nb.ndarray((tile_p, 1), x_hbm.dtype, nb.sbuf)
                nb.isa.memset(dst=scale, value=1.0)

                nb.isa.activation(
                    dst=out_sbuf,
                    src=x_sbuf,
                    bias=bias,
                    scale=scale,
                    op=nb.isa.activation_function.silu,
                )

                nb.isa.dma_copy(
                    dst=out_hbm[
                        r * tile_p : (r + 1) * tile_p, t * tile_f : (t + 1) * tile_f
                    ],
                    src=out_sbuf,
                )

    # Compile via kernel_builder and extract MLIR string
    module = nb.build_kernel(
        silu_kernel,
        input_specs={"x_hbm": nb.Tensor(shape, nb.float32, nb.shared_hbm)},
        output_specs={"out_hbm": nb.Tensor(shape, nb.float32, nb.shared_hbm)},
    )
    nisa_mlir = module.operation.get_asm(print_generic_op_form=True)

    def silu_reference(x):
        return x / (1.0 + np.exp(-x))

    return CustomOp(
        nisa_mlir=nisa_mlir,
        func_name=f"silu_{M}x{N}_{M}x{N}",
        input_names=["x_hbm"],
        output_names=["out_hbm"],
        input_shapes=[shape],
        output_shapes=[shape],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
        reference_fn=silu_reference,
    )


# ============================================================================
# Test: custom op tracing produces correct IR structure
# ============================================================================


def test_custom_op_trace_ir_structure():
    """
    Verify that tracing with a CustomOp produces the expected IR:
    - func.call to the custom op
    - func.func private declaration with nkipy.custom_op
    - nkipy.custom_op_bodies stashed on the module
    """
    custom_silu = _make_silu_custom_op(256, 256)

    @trace(input_specs=[((256, 256), "f32")])
    def kernel(x):
        return custom_silu(x)

    module = kernel.to_mlir()
    mlir_str = str(module)

    # Verify call site
    assert "call @__custom_op__silu_256x256_256x256" in mlir_str
    # Verify declaration
    assert "nkipy.custom_op" in mlir_str
    # Verify body stashing
    assert "nkipy.custom_op_bodies" in mlir_str
    # Verify the NISA body string is stashed
    assert "nisa.target" in mlir_str


# ============================================================================
# Test: custom op in feedforward kernel (full pipeline, STRING_CHECK)
# ============================================================================


def test_matmul_custom_activation_string_check():
    """
    Simple kernel: matmul followed by a CustomOp activation.

    The pipeline should:
    1. Trace the kernel with func.call to the custom op
    2. Run all passes (tiling, bufferize, annotate, legalize, linalg-to-nisa)
       - The custom op declaration passes through as a bodyless func.func
    3. resolve-custom-ops links the NISA body and rewrites call sites
    4. prepare-for-nki strips nkipy.* attrs and adds nisa.target

    We verify the final IR contains the resolved custom op.
    """
    custom_silu = _make_silu_custom_op(256, 256)

    @trace(
        input_specs=[
            ((256, 256), "f32"),  # x
            ((256, 256), "f32"),  # weight
        ]
    )
    def matmul_activation_kernel(x, weight):
        # Matrix multiply
        mm_out = np.matmul(x, weight)
        knob.knob(
            mm_out, mem_space="SharedHbm", tile_size=[128, 128], reduction_tile=[128]
        )

        # Custom SiLU activation on result (input/output on HBM)
        output = custom_silu(mm_out)

        return output

    run_kernel_test(
        matmul_activation_kernel,
        check_ir_contains=[
            # NISA ops from the main kernel
            "nisa.matmul",
            "nisa.target",
            # NISA ops from the inlined SiLU custom op
            "nisa.activation",
            "nisa.dma_copy",
            "nisa.memset",
        ],
        check_ir_not_contains=[
            # These should be stripped by prepare-for-nki
            "nkipy.custom_op_bodies",
            "nkipy.custom_op",
            "transform.named_sequence",
            # Function should be inlined, not linked
            "__custom_op__silu_256x256_256x256",
        ],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.BIR_SIM | Mode.STRING_CHECK,
    )


# ============================================================================
# Test: custom op reference_fn works for numpy execution
# ============================================================================


def test_custom_op_reference_fn_numpy():
    """
    Verify that the custom op's reference_fn produces correct numpy results
    when called outside of tracing (for test validation).
    """
    custom_silu = _make_silu_custom_op(4, 4)

    x = np.random.randn(4, 4).astype(np.float32)
    result = custom_silu(x)
    expected = x / (1.0 + np.exp(-x))
    np.testing.assert_allclose(result, expected, rtol=1e-6)


# ============================================================================
# Test: kernel_builder SiLU custom op (full pipeline, STRING_CHECK)
# ============================================================================


def _make_silu_kernel_builder_op(M, N, tile_p=128, tile_f=128):
    """Create a CustomOp using kernel_builder to compile a real SiLU activation.

    The kernel tiles internally: processes (tile_p x tile_f) chunks of the
    (M x N) input, one column-tile at a time.  This keeps SBUF usage to
    tile_p*tile_f elements (fitting in one SBUF partition row).
    """

    def silu_kernel(x_hbm, out_hbm):
        import nki.language as nl

        n_row_tiles = M // tile_p
        n_col_tiles = N // tile_f
        for r in nl.affine_range(n_row_tiles):
            for t in nl.affine_range(n_col_tiles):
                x_sbuf = nb.ndarray((tile_p, tile_f), x_hbm.dtype, nb.sbuf)
                nb.isa.dma_copy(
                    dst=x_sbuf,
                    src=x_hbm[
                        r * tile_p : (r + 1) * tile_p, t * tile_f : (t + 1) * tile_f
                    ],
                )

                out_sbuf = nb.ndarray((tile_p, tile_f), x_hbm.dtype, nb.sbuf)

                bias = nb.ndarray((tile_p, 1), x_hbm.dtype, nb.sbuf)
                nb.isa.memset(dst=bias, value=0.0)
                scale = nb.ndarray((tile_p, 1), x_hbm.dtype, nb.sbuf)
                nb.isa.memset(dst=scale, value=1.0)

                nb.isa.activation(
                    dst=out_sbuf,
                    src=x_sbuf,
                    bias=bias,
                    scale=scale,
                    op=nb.isa.activation_function.silu,
                )

                nb.isa.dma_copy(
                    dst=out_hbm[
                        r * tile_p : (r + 1) * tile_p, t * tile_f : (t + 1) * tile_f
                    ],
                    src=out_sbuf,
                )

    def silu_reference(x):
        return x / (1.0 + np.exp(-x))

    return CustomOp.from_kernel_builder(
        kernel_func=silu_kernel,
        input_specs={"x_hbm": nb.Tensor((M, N), nb.float32, nb.shared_hbm)},
        output_specs={"out_hbm": nb.Tensor((M, N), nb.float32, nb.shared_hbm)},
        reference_fn=silu_reference,
    )


def test_kernel_builder_silu():
    """
    Matmul + SiLU activation where SiLU is compiled via kernel_builder.

    This tests the from_kernel_builder() path which produces real NISA ops
    (dma_copy, activation, memset) rather than a hand-written stub.

    Uses 128x128 tiles because SBUF partition dim max is 128.
    Verifies both IR structure (STRING_CHECK) and numerical correctness (BIR_SIM).
    """
    # Custom op processes 256x256 HBM buffer, tiling to 128x128 internally
    custom_silu = _make_silu_kernel_builder_op(256, 256, tile_p=128, tile_f=128)

    @trace(
        input_specs=[
            ((256, 256), "f32"),  # x
            ((256, 256), "f32"),  # weight
        ]
    )
    def matmul_silu_kernel(x, weight):
        mm_out = np.matmul(x, weight)
        knob.knob(
            mm_out, mem_space="SharedHbm", tile_size=[128, 128], reduction_tile=[128]
        )

        output = custom_silu(mm_out)
        return output

    run_kernel_test(
        matmul_silu_kernel,
        check_ir_contains=[
            # Custom op body is inlined — check for NISA ops from both
            # the main kernel (matmul) and the inlined SiLU activation
            "nisa.activation",
            "nisa.dma_copy",
            "nisa.matmul",
            "nisa.target",
            "nisa.memset",  # from SiLU bias/scale initialization
        ],
        check_ir_not_contains=[
            "nkipy.custom_op_bodies",
            "nkipy.custom_op",
            # Function should be inlined, not linked as separate func
            "__custom_op__silu_kernel",
        ],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.BIR_SIM | Mode.STRING_CHECK,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
