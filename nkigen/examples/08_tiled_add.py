"""
Example: Full compilation pipeline for tiled add operation

This demonstrates end-to-end compilation with the C++ pipeline for element-wise addition:
  NumPy -> Traced MLIR -> nkipy-opt pipeline -> BIR -> Simulation

Usage:
    python 08_tiled_add.py
"""

import os
import tempfile
import numpy as np
from nkigen import trace, knob
from nkigen.transforms import apply_complete_knob_pipeline
from nki.compiler.baremetal_compilation import CompileOptions, simulate_from_mlir
from nki.compiler._internal import ir, register_all_dialects

# Configuration
M, N = 256, 256  # Matrix dimensions for element-wise addition: a(MxN) + b(MxN)
TARGET = "core_v2"  # Target: core_v2, core_v3, or core_v4


@trace(input_specs=[((M, N), "f32"), ((M, N), "f32")])
def tiled_add(a, b):
    """Compute element-wise addition: a + b"""
    result = np.add(a, b)
    knob.knob(result).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
    return result


def main():
    print(f"Running tiled add pipeline: M={M}, N={N}, Target={TARGET}\n")

    # Step 1: Trace to MLIR
    print("Step 1: Tracing NumPy to MLIR...")
    mlir_module = tiled_add.to_mlir()
    print("\n" + "="*80)
    print("Initial MLIR (after tracing):")
    print("="*80)
    print(mlir_module)
    print("="*80 + "\n")

    # Step 2: Apply full C++ pipeline (tiling + bufferization + NISA lowering)
    print("Step 2: Applying nkipy-opt pipeline...")
    debug_dir = tempfile.mkdtemp(prefix="tiled_add_")
    nisa_module = apply_complete_knob_pipeline(str(mlir_module), dump_dir=debug_dir)
    print("\n" + "="*80)
    print(f"Final NISA module (target={TARGET}):")
    print("="*80)
    print(nisa_module)
    print("="*80 + "\n")

    # Step 3: Compile to BIR and simulate
    print("Step 3: Compiling and simulating...")
    test_inputs = {
        'a': np.random.randn(M, N).astype(np.float32),
        'b': np.random.randn(M, N).astype(np.float32),
    }
    expected = test_inputs['a'] + test_inputs['b']

    print(f"\nDebug artifacts directory: {debug_dir}")

    opts = CompileOptions(
        target="trn2",
        verbose=False,
        output_path=os.path.join(debug_dir, "kernel.neff"),
        neuronx_cc_args=("--lnc=1",),
        artifacts_dir=debug_dir,
    )

    # Parse and simulate
    ctx = ir.Context()
    register_all_dialects(ctx)
    try:
        with ctx:
            mlir = ir.Module.parse(nisa_module, ctx)
            result = simulate_from_mlir(
                mlir, "tiled_add",
                [test_inputs['a'], test_inputs['b']],
                ["in_tensor_0", "in_tensor_1"],
                compile_opts=opts,
            )
    except Exception as e:
        print(f"\n✗ Compilation failed!")
        print(f"  Error: {e}")
        print(f"  Debug artifacts saved to: {debug_dir}")
        print(f"  Check log file: {debug_dir}/log-neuron-cc.txt")
        raise

    # Validate
    max_diff = np.max(np.abs(result - expected))
    matches = np.allclose(result, expected, rtol=1e-4, atol=1e-4)

    print(f"\nResults:")
    print(f"  Output shape: {result.shape}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Match: {matches}")
    print(f"  Status: {'✓ SUCCESS' if matches else '✗ FAILED'}")
    print(f"  Artifacts: {debug_dir}")

    return matches


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
