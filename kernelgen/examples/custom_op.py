# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Custom Op Integration — using pre-compiled NISA kernels in NKIPy.

Demonstrates how to wrap a kernel_builder-compiled NISA function as a CustomOp
and call it from an @trace-decorated kernel. This lets you mix high-level NumPy
tracing (matmul, elementwise) with hand-optimized NISA code (activations, custom
DMA patterns).

The flow:
  1. Write a NISA kernel using kernel_builder (nki.compiler.kernel_builder).
  2. Wrap it as a CustomOp via CustomOp.from_kernel_builder().
  3. Call the CustomOp inside an @trace kernel — it emits a func.call during
     tracing and falls back to a reference NumPy function outside of tracing.
  4. The compiler pipeline inlines the NISA body at each call site via the
     resolve-custom-ops pass.

Key concepts shown:
  - kernel_builder API for writing NISA kernels (nb.ndarray, nb.isa.*)
  - CustomOp.from_kernel_builder() for wrapping kernel_builder output
  - Mixing custom ops with traced NumPy operations
  - reference_fn for NumPy-level testing without hardware

Usage:
    # Compile and dump intermediate IR:
    python examples/custom_op.py

    # Or run the full test suite:
    python -m pytest tests/e2e/test_custom_op.py -v
"""

import numpy as np

from nkipy_kernelgen import trace, knob
from nkipy_kernelgen.custom_op import CustomOp
from nkipy_kernelgen.transforms.nkipy_opt import apply_complete_knob_pipeline

import nki.compiler.kernel_builder as nb


# ---------------------------------------------------------------------------
# Step 1: Define a NISA kernel using kernel_builder
# ---------------------------------------------------------------------------

def make_silu_custom_op(M, N, tile_p=128, tile_f=128):
    """Create a CustomOp that computes SiLU activation using real NISA ops.

    The kernel tiles the (M, N) input into (tile_p, tile_f) chunks and
    processes each tile: DMA in -> activation -> DMA out.

    Args:
        M, N: Input/output tensor dimensions.
        tile_p: Partition tile size (max 128 for SBUF).
        tile_f: Free-dimension tile size.

    Returns:
        A CustomOp callable in @trace kernels and in plain NumPy.
    """

    def silu_kernel(x_hbm, out_hbm):
        """NISA implementation of SiLU, written with kernel_builder."""
        import nki.language as nl

        n_row_tiles = M // tile_p
        n_col_tiles = N // tile_f
        for r in nl.affine_range(n_row_tiles):
            for t in nl.affine_range(n_col_tiles):
                # Load a tile from HBM to SBUF
                x_sbuf = nb.ndarray((tile_p, tile_f), x_hbm.dtype, nb.sbuf)
                nb.isa.dma_copy(
                    dst=x_sbuf,
                    src=x_hbm[
                        r * tile_p : (r + 1) * tile_p,
                        t * tile_f : (t + 1) * tile_f,
                    ],
                )

                # Allocate output tile and bias/scale for activation
                out_sbuf = nb.ndarray((tile_p, tile_f), x_hbm.dtype, nb.sbuf)
                bias = nb.ndarray((tile_p, 1), x_hbm.dtype, nb.sbuf)
                nb.isa.memset(dst=bias, value=0.0)
                scale = nb.ndarray((tile_p, 1), x_hbm.dtype, nb.sbuf)
                nb.isa.memset(dst=scale, value=1.0)

                # Apply SiLU activation via NISA hardware instruction
                nb.isa.activation(
                    dst=out_sbuf,
                    src=x_sbuf,
                    bias=bias,
                    scale=scale,
                    op=nb.isa.activation_function.silu,
                )

                # Store result from SBUF back to HBM
                nb.isa.dma_copy(
                    dst=out_hbm[
                        r * tile_p : (r + 1) * tile_p,
                        t * tile_f : (t + 1) * tile_f,
                    ],
                    src=out_sbuf,
                )

    def silu_reference(x):
        """NumPy reference for SiLU: x * sigmoid(x)."""
        return x / (1.0 + np.exp(-x))

    # Compile the kernel_builder function and wrap as CustomOp
    return CustomOp.from_kernel_builder(
        kernel_func=silu_kernel,
        input_specs={"x_hbm": nb.Tensor((M, N), nb.float32, nb.shared_hbm)},
        output_specs={"out_hbm": nb.Tensor((M, N), nb.float32, nb.shared_hbm)},
        reference_fn=silu_reference,
    )


# ---------------------------------------------------------------------------
# Step 2: Create the custom op instance
# ---------------------------------------------------------------------------

# Build a SiLU custom op for 256x256 tensors with 128x128 internal tiling.
# SBUF partition dim max is 128, so tile_p must be <= 128.
custom_silu = make_silu_custom_op(256, 256, tile_p=128, tile_f=128)


# ---------------------------------------------------------------------------
# Step 3: Use the custom op in a traced kernel
# ---------------------------------------------------------------------------

@trace(
    input_specs=[
        ((256, 256), "f32"),  # x
        ((256, 256), "f32"),  # weight
    ]
)
def matmul_silu_kernel(x, weight):
    """Matrix multiply followed by custom SiLU activation.

    The matmul is compiled by the NKIPy pipeline (tiling, bufferization,
    NISA lowering), while the SiLU is a pre-compiled NISA function that
    gets inlined at the call site by resolve-custom-ops.
    """
    mm_out = np.matmul(x, weight)
    knob.knob(
        mm_out, mem_space="SharedHbm", tile_size=[128, 128], reduction_tile=[128]
    )

    # Call the custom op — during tracing this emits a func.call;
    # during NumPy execution it falls back to silu_reference.
    output = custom_silu(mm_out)

    return output


# ---------------------------------------------------------------------------
# Compile, print IR, and verify correctness when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import tempfile

    # Trace and compile
    module = matmul_silu_kernel.to_mlir()
    traced_ir = str(module)

    dump_dir = tempfile.mkdtemp(prefix="custom_op_")
    compiled_ir = apply_complete_knob_pipeline(traced_ir, dump_dir=dump_dir)
    print(compiled_ir)

    # Verify numerical correctness via BIR simulation
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from harness import simulate_mlir, generate_inputs, compute_reference

    inputs = generate_inputs(matmul_silu_kernel.input_specs)
    reference = compute_reference(matmul_silu_kernel, inputs)

    success, max_diff, artifacts = simulate_mlir(
        compiled_ir,
        func_name="matmul_silu_kernel",
        test_inputs=inputs,
        expected_output=reference,
        rtol=1e-3,
        atol=1e-3,
    )
    print(f"\nBIR simulation: {'PASS' if success else 'FAIL'} (max_diff={max_diff:.2e})")
