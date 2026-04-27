#!/usr/bin/env python3
"""
NKIPy Compiler wrapper for Compiler Explorer.

This script takes Python source code with @trace decorated functions,
traces them to MLIR, and runs the full compilation pipeline.

Usage:
    python nkipy_compiler.py <input.py> [options]

Options:
    --stop=N              Stop after pass N (0 = trace only, 1-24 = after that pass)
    --sim                 Run simulation (BIR sim for full pipeline, LLVM JIT with --stop)
    --hw                  Compile to NEFF and execute on Trainium hardware
    --target=<target>     Target hardware: trn1, trn2, trn3 (default: trn2)
"""

import sys
import os
import argparse
import tempfile
import re
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nkipy_kernelgen import trace, apply_passes
from nkipy_kernelgen.transforms.nkipy_opt import apply_complete_knob_pipeline


def add_loc_comments(mlir_text: str) -> str:
    """
    Convert MLIR location attributes to .loc comments for Compiler Explorer.

    MLIR locations look like:
      - Inline: `linalg.add ... loc("file.py":6:5)`
      - Aliased: `#loc1 = loc("file.py":6:5)` at bottom, referenced as `loc(#loc1)`
      - Unknown: `loc(unknown)` or `#loc1 = loc(unknown)`

    CE expects:
      - `.file 1 "file.py"` at the top
      - `.loc 1 6 5` comment before each operation

    Args:
        mlir_text: MLIR text with location attributes (from --mlir-print-debuginfo)

    Returns:
        MLIR text with .loc comments inserted
    """
    # Parse location aliases from the bottom of the file
    # Format: #loc1 = loc("file.py":line:col)
    loc_alias_pattern = re.compile(r'^(#loc\d*)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)', re.MULTILINE)
    # Also match unknown locations: #loc1 = loc(unknown)
    unknown_loc_pattern = re.compile(r'^(#loc\d*)\s*=\s*loc\(unknown\)', re.MULTILINE)

    loc_aliases = {}
    unknown_aliases = set()

    for match in loc_alias_pattern.finditer(mlir_text):
        alias, filename, line, col = match.groups()
        loc_aliases[alias] = (filename, int(line), int(col))

    for match in unknown_loc_pattern.finditer(mlir_text):
        unknown_aliases.add(match.group(1))

    # Build file number mapping
    files = {}
    file_counter = 1
    for alias, (filename, _, _) in loc_aliases.items():
        if filename not in files:
            files[filename] = file_counter
            file_counter += 1

    # Also scan for inline locations to build file mapping
    inline_loc_pattern = re.compile(r'loc\("([^"]+)":(\d+):(\d+)\)')
    for match in inline_loc_pattern.finditer(mlir_text):
        filename = match.group(1)
        if filename not in files:
            files[filename] = file_counter
            file_counter += 1

    if not files:
        # No locations found, return unchanged but strip loc attributes
        lines = mlir_text.split('\n')
        result = []
        for line in lines:
            # Skip unknown location alias lines
            if unknown_loc_pattern.match(line.strip()):
                continue
            # Remove loc(unknown) and loc(#locN) references
            clean = re.sub(r'\s*loc\(unknown\)', '', line)
            clean = re.sub(r'\s*loc\(#loc\d*\)', '', clean)
            result.append(clean)
        return '\n'.join(result)

    # Build .file directives
    file_directives = []
    for filename, num in sorted(files.items(), key=lambda x: x[1]):
        file_directives.append(f'.file {num} "{filename}"')

    # Process each line and insert .loc comments
    lines = mlir_text.split('\n')
    result_lines = []

    # Add file directives at the top (after any initial comments)
    file_header_added = False

    for line in lines:
        # Skip location alias definitions at the bottom (both known and unknown)
        if loc_alias_pattern.match(line.strip()) or unknown_loc_pattern.match(line.strip()):
            continue

        # Add file header before first non-empty, non-comment line
        if not file_header_added and line.strip() and not line.strip().startswith('//'):
            result_lines.extend(file_directives)
            result_lines.append('')  # blank line after file directives
            file_header_added = True

        # Check for location on this line
        loc_info = None

        # Try aliased location first: loc(#loc1)
        alias_ref_match = re.search(r'loc\((#loc\d*)\)', line)
        if alias_ref_match:
            alias = alias_ref_match.group(1)
            if alias in loc_aliases:
                loc_info = loc_aliases[alias]

        # Try inline location: loc("file":line:col)
        if not loc_info:
            inline_match = re.search(r'loc\("([^"]+)":(\d+):(\d+)\)', line)
            if inline_match:
                filename, line_num, col = inline_match.groups()
                loc_info = (filename, int(line_num), int(col))

        # Remove all loc(...) from the line for cleaner output
        clean_line = re.sub(r'\s*loc\([^)]+\)', '', line)

        # Insert .loc comment before the line if we found location info
        if loc_info:
            filename, line_num, col = loc_info
            file_num = files.get(filename, 1)
            result_lines.append(f'.loc {file_num} {line_num} {col}')
            result_lines.append(clean_line)
        else:
            # Still use clean_line to remove any remaining loc(...) references
            result_lines.append(clean_line)

    return '\n'.join(result_lines)


def find_traced_function(module):
    """Find the first @trace decorated function in a module."""
    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, 'to_mlir') and callable(getattr(obj, 'to_mlir')):
            print(f"DEBUG: Found traced function: {name}", file=sys.stderr)
            return obj
    print(f"DEBUG: No traced function found. Checking for callable objects...", file=sys.stderr)
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith('_'):
            print(f"DEBUG: Callable '{name}' has attrs: {[a for a in dir(obj) if not a.startswith('_')][:10]}", file=sys.stderr)
    return None


def load_source_file(filepath):
    """Dynamically load a Python source file as a module."""
    # Read the source code
    with open(filepath, 'r') as f:
        source_code = f.read()

    # Debug: show what we're compiling
    print(f"DEBUG: Compiling file: {filepath}", file=sys.stderr)
    print(f"DEBUG: Source code ({len(source_code)} chars):", file=sys.stderr)
    print(source_code[:500], file=sys.stderr)
    if len(source_code) > 500:
        print("...(truncated)", file=sys.stderr)

    # Create a module and execute the code in it
    import types
    module = types.ModuleType("user_kernel")
    module.__file__ = filepath

    # Compile with filepath so inspect.getsourcefile works for source locations
    code = compile(source_code, filepath, 'exec')
    exec(code, module.__dict__)

    # Debug: show what's in the module
    print(f"DEBUG: Module contents: {[n for n in dir(module) if not n.startswith('_')]}", file=sys.stderr)

    return module


def compile_to_mlir(source_path, stop_after=None, target="trn2", raw=False):
    """
    Compile Python source to MLIR.

    Args:
        source_path: Path to Python file with @trace decorated function
        stop_after: Pass number to stop after (0 = trace only, 1-19 = after that pass, None = all)
        target: Target hardware
        raw: If True, omit source location debug info for cleaner CLI output

    Returns:
        Tuple of (ce_output, raw_mlir, traced_func):
        - ce_output: MLIR with .loc comments for Compiler Explorer
        - raw_mlir: Valid MLIR string (usable for simulation)
        - traced_func: The traced function object
    """
    # Load and execute the source file
    module = load_source_file(source_path)

    # Find the traced function
    traced_func = find_traced_function(module)
    if traced_func is None:
        print("Error: No @trace decorated function found in input file", file=sys.stderr)
        sys.exit(1)

    # Step 1: Trace to MLIR
    mlir_module = traced_func.to_mlir()

    # Include debug info for CE source-line mapping; skip for --raw CLI output
    include_debuginfo = not raw
    current_mlir = mlir_module.operation.get_asm(enable_debug_info=include_debuginfo)

    if stop_after == 0:
        return add_loc_comments(current_mlir), current_mlir, traced_func

    # Run the compilation pipeline (optionally stopping after a specific pass)
    result = apply_complete_knob_pipeline(
        current_mlir, target=target, stop_after=stop_after, print_debuginfo=include_debuginfo
    )
    return add_loc_comments(result), result, traced_func


DTYPE_MAP = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "f64": "float64",
    "i32": "int32",
    "i64": "int64",
}


def run_simulation(raw_mlir, traced_func):
    """Run BIR simulation on compiled MLIR and verify against NumPy reference."""
    import numpy as np

    tests_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"
    )
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    from harness import simulate_mlir

    # Generate test inputs from input_specs
    np.random.seed(42)
    test_inputs = []
    for shape, dtype_str in traced_func.input_specs:
        np_dtype = DTYPE_MAP.get(dtype_str, dtype_str)
        test_inputs.append(np.random.uniform(-1, 1, size=shape).astype(np_dtype))

    # Compute NumPy reference using the original unwrapped function
    expected = traced_func.__wrapped__(*test_inputs)
    print(f"NumPy reference: shape={expected.shape}, dtype={expected.dtype}", file=sys.stderr)

    func_name = traced_func.__wrapped__.__name__

    # Run BIR simulation and compare.
    # Redirect stdout to stderr so harness diagnostics appear in the CE log
    # instead of polluting the MLIR output pane.
    import io
    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        success, max_diff, artifacts = simulate_mlir(
            mlir_str=raw_mlir,
            func_name=func_name,
            test_inputs=test_inputs,
            expected_output=expected,
            rtol=1e-3,
            atol=1e-3,
            verbose=True,
            keep_artifacts=True,
        )
    except (RuntimeError, AssertionError) as e:
        sys.stdout = old_stdout
        harness_output = captured.getvalue()
        if harness_output:
            print(harness_output, file=sys.stderr)
        print(f"SIMULATION FAILED: {e}", file=sys.stderr)
        return False
    finally:
        sys.stdout = old_stdout

    # Print harness output (including any error details) to stderr
    harness_output = captured.getvalue()
    if harness_output:
        print(harness_output, file=sys.stderr)

    if success:
        print(f"SIMULATION PASSED (max_diff={max_diff:.2e})")
    else:
        print(f"SIMULATION FAILED (max_diff={max_diff:.2e})")
        if artifacts:
            print(f"Artifacts: {artifacts}")
            # Dump neuronx-cc log so the actual error is visible
            ncc_log = os.path.join(artifacts, "log-neuron-cc.txt")
            if os.path.exists(ncc_log):
                with open(ncc_log, 'r') as f:
                    log_lines = f.readlines()
                # Print lines containing errors/warnings, plus last 20 lines for context
                error_lines = [l.rstrip() for l in log_lines
                               if any(k in l for k in ('ERROR', 'error', 'FAIL', 'fail', 'Exception'))]
                if error_lines:
                    print("neuronx-cc errors:")
                    for line in error_lines[:30]:
                        print(f"  {line}")
                tail = [l.rstrip() for l in log_lines[-20:]]
                print(f"neuronx-cc log (last 20 lines):")
                for line in tail:
                    print(f"  {line}")

    return success


def run_llvm_simulation(raw_mlir, traced_func):
    """Run LLVM JIT simulation on intermediate MLIR and verify against NumPy reference."""
    tests_passes_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "passes"
    )
    if tests_passes_path not in sys.path:
        sys.path.insert(0, tests_passes_path)
    from pass_utils import verify_tiled_mlir_with_numpy

    func_name = traced_func.__wrapped__.__name__
    verify_tiled_mlir_with_numpy(
        raw_mlir, traced_func, rtol=1e-4, atol=1e-4, func_name=func_name
    )


def run_hw_execution(raw_mlir, traced_func, target):
    """Compile MLIR to NEFF and execute on Trainium hardware, verify against NumPy reference."""
    import numpy as np
    import tempfile

    from nki.compiler.ncc_driver import CompileOptions, compile_mlir_to_neff
    from nki.compiler._internal import ir, register_all_dialects
    from nki.runtime import SpikeModel, SpikeTensor

    func_name = traced_func.__wrapped__.__name__

    # Generate test inputs from input_specs
    np.random.seed(42)
    test_inputs = []
    for shape, dtype_str in traced_func.input_specs:
        np_dtype = DTYPE_MAP.get(dtype_str, dtype_str)
        test_inputs.append(np.random.uniform(-1, 1, size=shape).astype(np_dtype))

    # Compute NumPy reference
    expected = traced_func.__wrapped__(*test_inputs)
    print(f"NumPy reference: shape={expected.shape}, dtype={expected.dtype}", file=sys.stderr)

    # Compile MLIR to NEFF
    debug_dir = tempfile.mkdtemp(prefix="hw_exec_")
    opts = CompileOptions(
        target=target,
        verbose=False,
        output_path=os.path.join(debug_dir, "kernel.neff"),
        neuronx_cc_args=("--lnc=1",),
        artifacts_dir=debug_dir,
        enable_simulation=False,
    )

    ctx = ir.Context()
    register_all_dialects(ctx)
    with ctx:
        mlir = ir.Module.parse(raw_mlir, ctx)

        input_names = [f"in_tensor_{i}" for i in range(len(test_inputs))]
        output_name = "out_tensor"
        output_placeholder = np.zeros_like(expected)

        all_arrays = list(test_inputs) + [output_placeholder]
        argument_names = input_names + [output_name]
        output_arg_names = [output_name]

        compile_result = compile_mlir_to_neff(
            mlir,
            func_name,
            all_arrays,
            argument_names,
            output_arg_names,
            opts,
        )

    neff_path = compile_result.output_path
    print(f"NEFF compiled: {neff_path}", file=sys.stderr)

    # Load and execute on hardware
    model = SpikeModel.load_from_neff(neff_path)

    # Use the model's actual tensor names (NEFF compiler may rename them, e.g. _0 suffix)
    neff_input_names = list(model.input_tensors_info.keys())
    neff_output_names = list(model.output_tensors_info.keys())
    print(f"NEFF inputs: {neff_input_names}, outputs: {neff_output_names}", file=sys.stderr)

    # Map compile-time names to arrays, then look up by NEFF name
    # (NEFF preserves names but may reorder them)
    compile_input_map = dict(zip(input_names, test_inputs))
    spike_inputs = {
        name: SpikeTensor.from_numpy(compile_input_map[name], name=name)
        for name in neff_input_names
    }

    # Let the model auto-allocate output tensors with correct names
    spike_outputs = model(inputs=spike_inputs, outputs=None)

    # Read back the first output
    result_tensor = list(spike_outputs.values())[0]
    result = np.frombuffer(
        result_tensor.numpy(), dtype=expected.dtype
    ).reshape(expected.shape)

    # Compare against NumPy reference
    max_diff = np.max(np.abs(result - expected))
    success = np.allclose(result, expected, rtol=1e-3, atol=1e-3)

    if success:
        print(f"HW EXECUTION PASSED (max_diff={max_diff:.2e})")
    else:
        print(f"HW EXECUTION FAILED (max_diff={max_diff:.2e})")
        print(f"Artifacts: {debug_dir}")

    return success


def main():
    # Debug: log all arguments received from Compiler Explorer
    print(f"DEBUG: sys.argv = {sys.argv}", file=sys.stderr)

    # Handle version request early (Compiler Explorer uses this during setup)
    if "--version" in sys.argv or "-v" in sys.argv:
        print("NKIPy MLIR Compiler 0.1.0")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="NKIPy Compiler for Compiler Explorer"
    )
    parser.add_argument("input", nargs='?', help="Input Python file with @trace function")
    parser.add_argument(
        "--stop",
        dest="stop_after",
        type=int,
        default=None,
        help="Stop after pass N (0 = trace only, 1-24 = after that pass, omit for all passes)"
    )
    parser.add_argument(
        "--target",
        choices=["trn1", "trn2", "trn3"],
        default="trn2",
        help="Target hardware"
    )
    # Compiler Explorer typically passes these
    parser.add_argument("-o", "--outputfile", help="Output file (ignored, we write to stdout)")
    parser.add_argument("-S", action="store_true", help="Compile to assembly (CE flag, ignored)")
    parser.add_argument(
        "--sim", action="store_true",
        help="Run simulation and verify against NumPy (BIR sim for full pipeline, LLVM JIT when used with --stop)"
    )
    parser.add_argument(
        "--hw", action="store_true",
        help="Compile to NEFF and execute on Trainium hardware (requires neuron device, uses --target)"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Output clean MLIR without .loc/.file annotations (for CLI use; CE UI needs them)"
    )

    # Use parse_known_args to ignore Compiler Explorer's extra flags (-I, etc.)
    args, unknown = parser.parse_known_args()

    print(f"DEBUG: parsed args = {args}", file=sys.stderr)
    print(f"DEBUG: unknown args = {unknown}", file=sys.stderr)

    # Find the input file - CE may pass it as a positional arg or as last unknown arg
    # CE uses "<source>" as placeholder when source is passed via stdin
    input_file = args.input
    if input_file == "<source>":
        # CE passes source via stdin - read it to a temp file
        source_code = sys.stdin.read()
        print(f"DEBUG: Read {len(source_code)} chars from stdin", file=sys.stderr)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(source_code)
            input_file = f.name
        print(f"DEBUG: Wrote source to temp file: {input_file}", file=sys.stderr)
    elif not input_file or not os.path.exists(input_file):
        # Try to find a .py file in unknown args
        for arg in unknown:
            if arg.endswith('.py') and os.path.exists(arg):
                input_file = arg
                break
        # Also check if any unknown arg is an existing file
        if not input_file:
            for arg in unknown:
                if os.path.exists(arg) and not arg.endswith('.s'):
                    input_file = arg
                    break

    if not input_file or not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        print(f"DEBUG: Searched in args.input={args.input} and unknown={unknown}", file=sys.stderr)
        sys.exit(1)

    SEPARATOR = "=" * 60

    try:
        # ---- Compilation ----
        pipeline_desc = f"stop={args.stop_after}" if args.stop_after is not None else "full pipeline"
        print(f"\n{SEPARATOR}", file=sys.stderr)
        print(f"  Compilation started ({pipeline_desc}, target={args.target})", file=sys.stderr)
        print(SEPARATOR, file=sys.stderr)

        t0 = time.time()
        ce_output, raw_mlir, traced_func = compile_to_mlir(
            input_file,
            stop_after=args.stop_after,
            target=args.target,
            raw=args.raw,
        )
        elapsed = time.time() - t0

        print(f"\n{SEPARATOR}", file=sys.stderr)
        print(f"  Compilation finished  --  {elapsed:.2f}s", file=sys.stderr)
        print(f"{SEPARATOR}\n", file=sys.stderr)

        # --raw: print clean MLIR (no .loc annotations); default: CE-style with .loc
        output = raw_mlir if args.raw else ce_output

        # CE expects output in the file specified by -o
        if args.outputfile:
            with open(args.outputfile, 'w') as f:
                f.write(output)
            print(f"DEBUG: Wrote output to {args.outputfile}", file=sys.stderr)
        else:
            print(output)

        # Run simulation based on --sim and --stop flags:
        #   --sim only:         full pipeline + BIR simulation
        #   --stop only:        print IR, no simulation
        #   --sim + --stop:     print IR + LLVM JIT execution of intermediate IR
        if args.sim:
            if args.stop_after is None:
                # ---- BIR simulation ----
                print(f"\n{SEPARATOR}", file=sys.stderr)
                print(f"  BIR simulation started", file=sys.stderr)
                print(SEPARATOR, file=sys.stderr)

                t0 = time.time()
                success = run_simulation(raw_mlir, traced_func)
                elapsed = time.time() - t0
                status = "PASSED" if success else "FAILED"

                print(f"\n{SEPARATOR}", file=sys.stderr)
                print(f"  BIR simulation {status}  --  {elapsed:.2f}s", file=sys.stderr)
                print(f"{SEPARATOR}\n", file=sys.stderr)
                if not success:
                    sys.exit(1)
            else:
                print(raw_mlir)
                # ---- LLVM JIT simulation ----
                print(f"\n{SEPARATOR}", file=sys.stderr)
                print(f"  LLVM JIT simulation started", file=sys.stderr)
                print(SEPARATOR, file=sys.stderr)

                t0 = time.time()
                run_llvm_simulation(raw_mlir, traced_func)
                elapsed = time.time() - t0

                print(f"\n{SEPARATOR}", file=sys.stderr)
                print(f"  LLVM JIT simulation finished  --  {elapsed:.2f}s", file=sys.stderr)
                print(f"{SEPARATOR}\n", file=sys.stderr)

        # Run on Trainium hardware (requires full pipeline, i.e. no --stop)
        if args.hw:
            if args.stop_after is not None:
                print("Error: --hw requires full pipeline (cannot be used with --stop)", file=sys.stderr)
                sys.exit(1)

            # ---- HW execution ----
            print(f"\n{SEPARATOR}", file=sys.stderr)
            print(f"  HW execution started (target={args.target})", file=sys.stderr)
            print(SEPARATOR, file=sys.stderr)

            t0 = time.time()
            success = run_hw_execution(raw_mlir, traced_func, target=args.target)
            elapsed = time.time() - t0
            status = "PASSED" if success else "FAILED"

            print(f"\n{SEPARATOR}", file=sys.stderr)
            print(f"  HW execution {status}  --  {elapsed:.2f}s", file=sys.stderr)
            print(f"{SEPARATOR}\n", file=sys.stderr)
            if not success:
                sys.exit(1)
            if not success:
                sys.exit(1)
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
