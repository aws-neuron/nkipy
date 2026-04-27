#!/usr/bin/env python3
"""
Run BIRSim on a pre-compiled NISA MLIR file and compare against numpy reference.

Usage: python3 run_sim.py <mlir_file>

Expects a kernel.py in the same directory as <mlir_file> containing a function
with the same name as the MLIR func. After BIRSim succeeds, runs the numpy
reference with the same inputs and compares outputs.
"""

import sys
import os
import shutil
import importlib.util
import numpy as np

from nki.compiler.ncc_driver import CompileOptions, compile_mlir_to_neff
from nki.compiler._internal import ir, register_all_dialects


MLIR_DTYPE_TO_NP = {
    "f32": np.float32,
    "f16": np.float16,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
}


def get_memref_info(memref_type):
    """Extract shape and numpy dtype from an MLIR MemRefType."""
    shape = tuple(memref_type.shape)
    etype = memref_type.element_type
    np_dtype = MLIR_DTYPE_TO_NP.get(str(etype), np.float32)
    return shape, np_dtype


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_sim.py <mlir_file>", file=sys.stderr)
        sys.exit(1)

    mlir_file = sys.argv[1]
    if not os.path.exists(mlir_file):
        print(f"Error: File not found: {mlir_file}", file=sys.stderr)
        sys.exit(1)

    with open(mlir_file, "r") as f:
        mlir_str = f.read()

    # Parse MLIR and extract function signature
    ctx = ir.Context()
    register_all_dialects(ctx)

    with ctx:
        module = ir.Module.parse(mlir_str, ctx)

        # Find the func.func operation
        func_op = None
        for op in module.body.operations:
            if "function_type" in op.attributes:
                func_op = op
                break

        if func_op is None:
            print("Error: No function found in module", file=sys.stderr)
            sys.exit(1)

        func_name = func_op.attributes["sym_name"].value
        func_type = func_op.attributes["function_type"].value

        # Extract input types
        input_specs = []
        for arg_type in func_type.inputs:
            shape, dtype = get_memref_info(ir.MemRefType(arg_type))
            input_specs.append((shape, dtype))

        # Extract output type
        results = list(func_type.results)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        out_type = results[0]
        out_shape, out_dtype = get_memref_info(ir.MemRefType(out_type))

        print(f"Function: {func_name}")
        for i, (shape, dtype) in enumerate(input_specs):
            print(f"  arg{i}: {shape} {dtype}")
        print(f"  returns: {out_shape} {out_dtype}")

        # Generate small random inputs ([-1, 1] range for numerical stability)
        np.random.seed(42)
        test_inputs = []
        for shape, dtype in input_specs:
            test_inputs.append(np.random.uniform(-1, 1, size=shape).astype(dtype))

        output_placeholder = np.zeros(out_shape, dtype=out_dtype)

        # Artifacts dir
        base = os.path.splitext(os.path.basename(mlir_file))[0]
        artifacts_dir = os.path.join(
            os.path.dirname(os.path.abspath(mlir_file)), f"artifacts_{base}"
        )
        if os.path.exists(artifacts_dir):
            shutil.rmtree(artifacts_dir)
        os.makedirs(artifacts_dir)

        input_names = [f"in_tensor_{i}" for i in range(len(test_inputs))]

        # Extract output name from nki.output_names attribute if present
        output_name = "out_tensor_0"
        if "nki.output_names" in func_op.attributes:
            names_attr = ir.ArrayAttr(func_op.attributes["nki.output_names"])
            output_name = str(ir.StringAttr(names_attr[0])).strip('"')

        opts = CompileOptions(
            target="trn2",
            verbose=True,
            output_path=os.path.join(artifacts_dir, "kernel.neff"),
            neuronx_cc_args=("--lnc=1",),
            artifacts_dir=artifacts_dir,
            enable_simulation=True,
        )

        print(f"\nArtifacts: {artifacts_dir}")
        print("=" * 60)

        compile_result = compile_mlir_to_neff(
            module,
            func_name,
            list(test_inputs) + [output_placeholder],
            input_names + [output_name],
            [output_name],
            opts,
        )

    print("=" * 60)

    if compile_result.neuronx_cc_error:
        print(f"\nFAILED: neuronx-cc error: {compile_result.neuronx_cc_error}")
        log_path = os.path.join(artifacts_dir, "log-neuron-cc.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()
            print(f"\nErrors from log:")
            for line in lines:
                if "ERROR" in line or "ISIM" in line:
                    print(f"  {line.rstrip()}")
        sys.exit(1)

    if compile_result.birsim_outputs is None:
        print("\nFAILED: BIRSim produced no outputs")
        sys.exit(1)

    result = compile_result.birsim_outputs[0]
    print(f"\nBIRSim output: shape={result.shape}, dtype={result.dtype}")
    print(f"  range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  mean:  {result.mean():.4f}")
    print("\nBIRSim PASSED")

    # --- NumPy reference comparison ---
    mlir_dir = os.path.dirname(os.path.abspath(mlir_file))
    kernel_py = os.path.join(mlir_dir, "kernel.py")
    if not os.path.exists(kernel_py):
        print(f"\nNo kernel.py found in {mlir_dir}, skipping numpy comparison")
        return

    print(f"\n--- Running numpy reference from kernel.py ---")
    spec = importlib.util.spec_from_file_location("kernel", kernel_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ref_fn = getattr(mod, func_name, None)
    if ref_fn is None:
        print(f"  Warning: function '{func_name}' not found in kernel.py, skipping")
        return

    ref_output = ref_fn(*test_inputs)
    diff = np.abs(result.astype(np.float64) - ref_output.astype(np.float64))
    max_diff = diff.max()
    mean_diff = diff.mean()
    match = np.allclose(result, ref_output, atol=1e-2, rtol=1e-2)

    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Match: {match}")

    if match:
        print("\nSIMULATION PASSED")
    else:
        print(f"\nSIMULATION FAILED (max_diff={max_diff:.2e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
