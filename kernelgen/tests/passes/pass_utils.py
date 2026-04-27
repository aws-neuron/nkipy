"""
Common utilities for testing MLIR passes.

Includes FileCheck support, MLIR compilation helpers, and test infrastructure.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from nkipy_kernelgen import apply_passes
from nkipy_kernelgen.transforms.nkipy_opt import (
    run_nkipy_opt_passes,
    apply_complete_knob_pipeline,
)
from nkipy_kernelgen.llvm import LLVMModule, extract_and_clean_func_from_module


# ============================================================================
# MLIR Pass Execution
# ============================================================================


def run_passes(
    mlir_module: str, passes: List[str], print_ir_after_all: bool = False
) -> str:
    """
    Run a list of passes on an MLIR module.

    Args:
        mlir_module: Input MLIR module as string
        passes: List of pass names to run
        print_ir_after_all: If True, print IR after each pass for debugging

    Returns:
        Transformed MLIR module as string
    """
    return run_nkipy_opt_passes(mlir_module, passes, print_ir_after_all)


def trace_to_mlir_with_preprocessing(traced_func) -> str:
    """
    Convert a traced function to MLIR string.

    Args:
        traced_func: A traced function with .to_mlir() method

    Returns:
        MLIR module as string
    """
    mlir_module = traced_func.to_mlir()
    return str(mlir_module)


def compile_through_passes(
    traced_func,
    passes: List[str],
    dump_dir: Optional[str] = None,
    preprocessing: bool = True,
) -> str:
    """
    Compile a traced function through a specified pass pipeline.

    Args:
        traced_func: A traced function with .to_mlir() method
        passes: List of pass names to run
        dump_dir: Optional directory to save intermediate MLIR files
        preprocessing: Whether to apply preprocessing (currently a no-op, kept for API compat)

    Returns:
        Final MLIR module as string
    """
    # Get MLIR from traced function
    if preprocessing:
        mlir_str = trace_to_mlir_with_preprocessing(traced_func)
    else:
        mlir_str = str(traced_func.to_mlir())

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        save_mlir_to_file(mlir_str, os.path.join(dump_dir, "00_input.mlir"))

    # Run each pass individually and save output
    current_mlir = mlir_str
    for i, pass_name in enumerate(passes):
        try:
            current_mlir = run_nkipy_opt_passes(
                current_mlir, [pass_name], print_stderr=True
            )
            if dump_dir:
                output_filename = f"{i + 1:02d}_{pass_name.replace('-', '_')}.mlir"
                save_mlir_to_file(current_mlir, os.path.join(dump_dir, output_filename))
        except RuntimeError as e:
            # On error, save what we have and re-raise
            if dump_dir:
                error_file = os.path.join(dump_dir, f"ERROR_{pass_name}.txt")
                with open(error_file, "w") as f:
                    f.write(str(e))
            raise

    return current_mlir


# ============================================================================
# File Utilities
# ============================================================================


def save_mlir_to_file(mlir_text: str, output_path: str) -> None:
    """
    Save MLIR text to a file.

    Args:
        mlir_text: The MLIR module text to save
        output_path: Path to the output file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(mlir_text)
    print(f"Saved MLIR to: {output_path}")


def get_test_output_dir(test_file: str) -> str:
    """Get the output directory for a specific test file."""
    test_dir = Path(test_file).parent
    output_dir = test_dir / "outputs"
    os.makedirs(output_dir, exist_ok=True)
    return str(output_dir)


# ============================================================================
# FileCheck Support
# ============================================================================


def get_filecheck_path() -> str:
    """Get the path to the FileCheck executable."""
    # Check in LLVM build directory
    package_dir = Path(__file__).parent.parent.parent
    llvm_build = package_dir.parent / "llvm-project" / "build" / "bin" / "FileCheck"

    if llvm_build.exists():
        return str(llvm_build)

    # Check in PATH
    import shutil

    filecheck = shutil.which("FileCheck")
    if filecheck:
        return filecheck

    raise FileNotFoundError(
        "FileCheck not found. Please build LLVM or add FileCheck to PATH.\n"
        f"Looked in: {llvm_build}"
    )


def assert_ir_unchanged(
    before_file: str, after_file: str, pass_name: str = "pass"
) -> None:
    """
    Assert that two MLIR files are identical.

    This is useful for testing that a pass doesn't modify IR when there's
    nothing to transform (e.g., no SBUF outputs for legalize-sbuf-outputs).

    Args:
        before_file: Path to the MLIR file before the pass
        after_file: Path to the MLIR file after the pass
        pass_name: Name of the pass (for error messages)

    Raises:
        AssertionError: If the files don't exist
        pytest.fail: If the files differ
    """
    import pytest

    assert os.path.exists(before_file), f"Before file not found: {before_file}"
    assert os.path.exists(after_file), f"After file not found: {after_file}"

    # Use diff to compare the files
    result = subprocess.run(
        ["diff", "-q", before_file, after_file], capture_output=True, text=True
    )

    if result.returncode != 0:
        # Files differ - show the diff
        diff_result = subprocess.run(
            ["diff", "-u", before_file, after_file], capture_output=True, text=True
        )
        pytest.fail(
            f"{pass_name} pass modified IR when it should not have!\n"
            f"Diff:\n{diff_result.stdout}"
        )


def run_filecheck(mlir_output: str, check_patterns: str) -> None:
    """
    Run FileCheck to verify MLIR output against check patterns.

    Args:
        mlir_output: The MLIR text to verify
        check_patterns: String containing CHECK patterns (e.g., "CHECK: scf.for\\nCHECK: linalg.add")

    Raises:
        AssertionError: If FileCheck fails, with detailed error message

    Example:
        check_patterns = '''
        CHECK: func.func
        CHECK: scf.for
        CHECK-NOT: linalg.fill
        CHECK: linalg.matmul
        '''
        run_filecheck(mlir_output, check_patterns)

    FileCheck Regex Notes:
        - Use {{.*}} to match any text (FileCheck regex)
        - In Python f-strings, use {{{{.*}}}} to get {{.*}}
        - Use %c0{{.*}} to match %c0, %c0_1, %c0_3, etc.
    """
    filecheck = get_filecheck_path()

    # Write check patterns to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".check", delete=False) as f:
        f.write(check_patterns)
        check_file = f.name

    try:
        # Run FileCheck: reads patterns from check_file, reads input from stdin
        result = subprocess.run(
            [filecheck, check_file], input=mlir_output, capture_output=True, text=True
        )

        if result.returncode != 0:
            # FileCheck failed - provide helpful error message
            error_msg = "FileCheck verification failed!\n"
            error_msg += f"FileCheck stderr:\n{result.stderr}\n"
            error_msg += f"\n--- Check Patterns ---\n{check_patterns}\n"
            error_msg += (
                f"\n--- MLIR Output (first 3000 chars) ---\n{mlir_output[:3000]}\n"
            )
            raise AssertionError(error_msg)

    finally:
        if os.path.exists(check_file):
            os.unlink(check_file)


# ============================================================================
# Knob Pipeline
# ============================================================================


def compile_knob_pipeline(traced_func, stop_after=None, dump_dir=None, **kwargs):
    """
    Trace a function and run it through the knob compilation pipeline.

    Args:
        traced_func: A traced function with .to_mlir() method
        stop_after: Pass name (str) or index (int) to stop after, or None for all passes
        dump_dir: Optional directory to save intermediate MLIR files
        **kwargs: Additional arguments passed to apply_complete_knob_pipeline

    Returns:
        Transformed MLIR module as string
    """
    mlir_str = trace_to_mlir_with_preprocessing(traced_func)
    return apply_complete_knob_pipeline(
        mlir_str, stop_after=stop_after, dump_dir=dump_dir, **kwargs
    )


# ============================================================================
# LLVM CPU Execution Verification
# ============================================================================


def verify_tiled_mlir_with_numpy(
    tiled_mlir: str,
    traced_func,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    func_name: str = "top",
) -> None:
    """
    Verify that compiled MLIR produces the same results as the original function.

    This function:
    1. Extracts input specs from the traced function
    2. Generates random test inputs based on those specs
    3. Runs the original function (via __wrapped__) to get expected output
    4. Compiles and executes the MLIR using LLVM JIT
    5. Compares the MLIR output with the original function output

    Args:
        tiled_mlir: The transformed MLIR code
        traced_func: A traced function (decorated with @trace) with __wrapped__ attribute
        rtol: Relative tolerance for np.allclose comparison
        atol: Absolute tolerance for np.allclose comparison
        func_name: Name of the top-level function in the MLIR module

    Raises:
        AssertionError: If MLIR output doesn't match original function output
    """
    original_func = traced_func.__wrapped__
    input_specs = traced_func.input_specs

    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "f16": np.float16,
        "i32": np.int32,
        "i64": np.int64,
    }

    inputs = []
    for shape, dtype_str in input_specs:
        if dtype_str not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}"
            )
        np_dtype = dtype_map[dtype_str]
        if np_dtype in [np.float32, np.float64, np.float16]:
            arr = np.random.rand(*shape).astype(np_dtype)
        else:
            arr = np.random.randint(0, 100, size=shape).astype(np_dtype)
        inputs.append(arr)

    numpy_result = original_func(*inputs)

    clean_mlir, actual_func_name = extract_and_clean_func_from_module(tiled_mlir)
    runner = LLVMModule(clean_mlir, actual_func_name)
    mlir_result = runner(*[inp.copy() for inp in inputs])

    # Normalize both to lists for uniform comparison
    if not isinstance(mlir_result, list):
        mlir_result = [mlir_result]
    if isinstance(numpy_result, tuple):
        numpy_result = list(numpy_result)
    elif not isinstance(numpy_result, list):
        numpy_result = [numpy_result]

    assert len(mlir_result) == len(numpy_result), (
        f"Output count mismatch: MLIR returned {len(mlir_result)}, "
        f"NumPy returned {len(numpy_result)}"
    )

    for i, (mr, nr) in enumerate(zip(mlir_result, numpy_result)):
        if not np.allclose(mr, nr, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(mr - nr))
            raise AssertionError(
                f"MLIR result does not match original function output (output {i}).\n"
                f"Max difference: {max_diff}\n"
                f"Relative tolerance: {rtol}, Absolute tolerance: {atol}"
            )
