"""
Example: Full compilation pipeline from NumPy to NISA and BIR simulation with tiling

This demonstrates end-to-end compilation with tiling support:
  NumPy -> Traced MLIR -> Remove Redundant Fills -> Tiling ->
  Fix Loop-Carried Vars -> Bufferization -> Canonicalize ->
  NISA (with optimized tiled matmul) -> BIR -> Simulation

The pipeline includes:
1. Optimization pass that removes redundant linalg.fill operations with zero
   constants before matmul (Neuron hardware auto-zeros PSUM buffers)
2. Tiling pass using transform dialect to create scf.for loops
3. Loop-carried variable fixing pass (required for bufferization to work)
4. Bufferization (tensor -> memref conversion)
5. NISA conversion with pattern recognition for tiled matmul:
   - Detects triple-nested loop structure
   - Allocates PSUM outside k-loop for hardware accumulation
   - Generates optimized code matching the pattern in optimized.mlir

Usage:
    LD_LIBRARY_PATH=/home/ubuntu/llvm-install/lib:$LD_LIBRARY_PATH python 07_full_pipeline.py
"""

import os
import tempfile
import numpy as np
from nkigen import trace, apply_passes, knob
from nki.compiler.baremetal_compilation import CompileOptions, simulate_from_mlir
from nki.compiler._internal import ir, register_all_dialects

# Configuration
M, N, K = 256, 256, 256  # Matrix dimensions: a(MxK) @ b(KxN) + c(MxN)
#M, N, K = 128, 128, 128  # Matrix dimensions: a(MxK) @ b(KxN) + c(MxN)
#M, N, K = 512, 512, 512  # Matrix dimensions: a(MxK) @ b(KxN) + c(MxN)
TARGET = "core_v2"        # Target: core_v2, core_v3, or core_v4

# NOTE: Tiled pattern generates correct NISA structure
# Workaround uses offset 0,0 for all DMA operations (numerically wrong but structurally correct)


@trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
def matmul_add(a, b, c):
    """Compute matmul(a, b) + c"""
    tmp = np.matmul(a, b)
    knob.knob(tmp).tile_op(tile_size=[128, 128], reduction_tile=[128]).layout(mem_space="Sbuf", partition_dim=0)
    res = np.add(tmp, c)
    knob.knob(res).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
    return res


def main():
    print(f"Running pipeline: M={M}, N={N}, K={K}, Target={TARGET}\n")

    # Step 1: Trace to MLIR
    print("Step 1: Tracing NumPy to MLIR...")
    mlir_module = matmul_add.to_mlir()
    print("\n" + "="*80)
    print("Initial MLIR (after tracing):")
    print("="*80)
    print(mlir_module)
    print("="*80 + "\n")

    # Step 2: Apply transformations (tiling + bufferization + linalg-to-NISA)
    print("Step 2: Applying transformations...")

    # Apply complete knob pipeline (all passes through nkipy-opt)
    # This includes remove-redundant-zero-fill as the first C++ pass
    print("Applying complete knob pipeline...")

    from nkigen.transforms import apply_complete_knob_pipeline
    final_module = apply_complete_knob_pipeline(mlir_module, target=TARGET, print_ir_after_all=True)
    
    print("\n" + "="*80)
    print("After complete knob pipeline:")
    print("="*80)
    print(final_module)
    print("="*80 + "\n")

    print(f"Generated NISA module with target: {TARGET}")

    # Step 3: Compile to BIR and simulate
    print("Step 3: Compiling and simulating...")
    test_inputs = {
        'a': np.random.randn(M, K).astype(np.float32),
        'b': np.random.randn(K, N).astype(np.float32),
        'c': np.random.randn(M, N).astype(np.float32),
    }
    expected = np.matmul(test_inputs['a'], test_inputs['b']) + test_inputs['c']

    # Setup and run simulation
    debug_dir = tempfile.mkdtemp(prefix="bir_sim_")
    print(f"\nDebug artifacts directory: {debug_dir}")

    opts = CompileOptions(
        target="trn2",
        verbose=False,
        output_path=os.path.join(debug_dir, "kernel.neff"),
        neuronx_cc_args=("--lnc=1",),
        artifacts_dir=debug_dir,
    )

    # Parse and simulate (transform module stripped and nisa.target added by nkipy-opt)
    ctx = ir.Context()
    register_all_dialects(ctx)
    try:
        with ctx:
            mlir = ir.Module.parse(str(final_module), ctx)
            result = simulate_from_mlir(
                mlir, "matmul_add",
                [test_inputs['a'], test_inputs['b'], test_inputs['c']],
                ["in_tensor_0", "in_tensor_1", "in_tensor_2"],
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
