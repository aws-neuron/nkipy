"""
Unified declarative test harness for NKIPyKernelGen.

Provides:
  - Mode enum for declaring test verification modes
  - run_kernel_test() for programmatic test invocation
  - @nkipy_kernelgen_test decorator for declarative test definitions
  - Auto input generation from traced function's input_specs
  - Per-mode default tolerances
  - Parameter validation with clear error messages

Usage:
    from harness import nkipy_kernelgen_test, run_kernel_test, Mode

    # Decorator form (non-parametrized tests):
    @nkipy_kernelgen_test(
        input_specs=[((256, 256), "f32"), ((256, 256), "f32")],
        stop_after="apply-and-strip-transforms",
        check_patterns="CHECK: scf.for\\nCHECK: linalg.matmul",
        modes=Mode.LLVM | Mode.FILECHECK,
    )
    def test_matmul_tiling(a, b):
        result = np.matmul(a, b)
        knob.knob(result, tile_size=[128, 128, 128])
        return result

    # Function form (parametrized tests):
    @pytest.mark.parametrize("shape,tile_size", [...])
    def test_add_shapes(shape, tile_size, request):
        @trace(input_specs=[(shape, "f32"), (shape, "f32")])
        def kernel(a, b):
            ...
        run_kernel_test(kernel, stop_after="apply-and-strip-transforms",
                        modes=Mode.LLVM | Mode.FILECHECK, request=request)
"""

import enum
import os
import tempfile
from typing import List, Optional, Tuple

import pytest

import numpy as np

# Set by conftest.py pytest_configure when --dump-ir is passed.
_DUMP_IR_ENABLED = False


# ============================================================================
# Mode Enum
# ============================================================================


class Mode(enum.Flag):
    """Test execution modes (combinable via |).

    LLVM and BIR_SIM/HW are mutually exclusive (different pipeline stages).
    STRING_CHECK and FILECHECK can combine with any execution mode.
    """

    LLVM = enum.auto()  # LLVM JIT execution, compare to NumPy (requires stop_after)
    BIR_SIM = (
        enum.auto()
    )  # BIR simulation via neuron-cc (full pipeline, stop_after=None)
    HW = enum.auto()  # Trainium hardware execution (full pipeline, stop_after=None)
    STRING_CHECK = (
        enum.auto()
    )  # check_ir_contains / check_ir_not_contains on compiled IR
    FILECHECK = enum.auto()  # check_patterns via external FileCheck tool on compiled IR


# ============================================================================
# Constants
# ============================================================================

# Default tolerances per execution mode
DEFAULT_TOLERANCES = {
    Mode.LLVM: {"rtol": 1e-5, "atol": 1e-8},
    Mode.BIR_SIM: {"rtol": 1e-4, "atol": 1e-4},
    Mode.HW: {"rtol": 1e-3, "atol": 1e-3},
}

# dtype string -> numpy dtype mapping
DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "bf16": np.dtype("bfloat16") if hasattr(np, "bfloat16") else "bfloat16",
    "i32": np.int32,
    "i64": np.int64,
}

# Sentinel for "trace only, no passes"
TRACE = "trace"


def _default_target() -> str:
    """Auto-detect hardware target, defaulting to trn2 on trn1 machines.

    trn1 has restrictive limitations (e.g. scalar bias in activations)
    and is no longer the preferred target. On trn1 instances we warn and
    fall back to trn2, which BIR simulation supports via cross-target
    compilation. trn2+ targets are returned as-is.
    """
    from nki.compiler.target import resolve_target

    detected = resolve_target()
    if detected == "trn1":
        import warnings
        warnings.warn(
            f"Detected {detected} instance — defaulting to trn2 target "
            "for compilation (trn1 has known NISA limitations)",
            stacklevel=2,
        )
        return "trn2"
    return detected


# ============================================================================
# Parameter Validation
# ============================================================================


def _validate_params(
    modes: Mode,
    stop_after,
    check_patterns: Optional[str],
    check_ir_contains: Optional[List[str]],
    check_ir_not_contains: Optional[List[str]],
) -> None:
    """Validate parameter combinations and raise ValueError for misconfigurations."""

    has_llvm = Mode.LLVM in modes
    has_bir_sim = Mode.BIR_SIM in modes
    has_hw = Mode.HW in modes
    has_filecheck = Mode.FILECHECK in modes
    has_string_check = Mode.STRING_CHECK in modes

    # LLVM requires stop_after (LLVM JIT cannot execute NISA IR)
    if has_llvm and stop_after is None:
        raise ValueError(
            "Mode.LLVM requires stop_after (LLVM JIT cannot execute NISA IR "
            "from full pipeline). Use stop_after='trace' for raw traced MLIR "
            "or stop_after='<pass-name>' for intermediate IR."
        )

    # BIR_SIM / HW require full pipeline (no stop_after)
    if has_bir_sim and stop_after is not None:
        raise ValueError(
            "Mode.BIR_SIM requires full pipeline (stop_after must not be specified)."
        )
    if has_hw and stop_after is not None:
        raise ValueError(
            "Mode.HW requires full pipeline (stop_after must not be specified)."
        )

    # LLVM is mutually exclusive with BIR_SIM / HW
    if has_llvm and (has_bir_sim or has_hw):
        raise ValueError(
            "Mode.LLVM cannot be combined with Mode.BIR_SIM/HW "
            "(different pipeline stages). Use separate tests."
        )

    # FILECHECK requires check_patterns
    if has_filecheck and not check_patterns:
        raise ValueError("Mode.FILECHECK requires check_patterns string.")

    # STRING_CHECK requires check_ir_contains or check_ir_not_contains
    if has_string_check and not check_ir_contains and not check_ir_not_contains:
        raise ValueError(
            "Mode.STRING_CHECK requires check_ir_contains or check_ir_not_contains."
        )


# ============================================================================
# Input Generation
# ============================================================================


def generate_inputs(input_specs, seed: int = 42) -> List[np.ndarray]:
    """Generate random test inputs from input_specs.

    Args:
        input_specs: List of (shape, dtype_str) tuples, e.g. [((256, 256), "f32")]
        seed: Random seed for reproducibility

    Returns:
        List of numpy arrays matching the specs
    """
    np.random.seed(seed)
    inputs = []
    for shape, dtype_str in input_specs:
        if dtype_str not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype_str}. Supported: {list(DTYPE_MAP.keys())}"
            )
        np_dtype = DTYPE_MAP[dtype_str]
        if np_dtype in (np.float32, np.float64, np.float16):
            arr = np.random.rand(*shape).astype(np_dtype)
        else:
            arr = np.random.randint(0, 100, size=shape).astype(np_dtype)
        inputs.append(arr)
    return inputs


def compute_reference(traced_func, inputs: List[np.ndarray]) -> List[np.ndarray]:
    """Compute reference output by calling the original (unwrapped) function.

    Args:
        traced_func: A traced function decorated with @trace
        inputs: List of numpy input arrays

    Returns:
        List of NumPy reference outputs (single-element list for single-output kernels)
    """
    original_func = traced_func.__wrapped__
    result = original_func(*inputs)
    if isinstance(result, tuple):
        return list(result)
    return [result]


# ============================================================================
# Hardware Detection
# ============================================================================


def is_hw_available() -> bool:
    """Check if Trainium hardware is available."""
    try:
        from nki.runtime import SpikeModel  # noqa: F401

        # Check if a neuron device is actually present
        import subprocess

        result = subprocess.run(
            ["neuron-ls"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "NEURON" in result.stdout
    except Exception:
        return False


def _can_run_hw(target: Optional[str]) -> bool:
    """Check if Mode.HW can run for the given target.

    Mode.HW requires trn2+. Even if a Neuron device is detected, trn1
    cannot execute HW mode. The detected device must also match the
    requested target (e.g. a trn2 test cannot run on trn1 hardware).
    """
    if not is_hw_available():
        return False
    from nki.compiler.target import resolve_target
    detected = resolve_target()
    # trn1 does not support Mode.HW
    if detected == "trn1":
        return False
    # If the test requests a specific target, the device must match
    if target is not None and detected != target:
        return False
    return True


# ============================================================================
# Compilation Pipeline
# ============================================================================


def _compile_pipeline(
    traced_func,
    stop_after,
    target: Optional[str] = None,
    dump_dir=None,
    print_generic: bool = False,
) -> str:
    """Compile a traced function through the pipeline.

    Args:
        traced_func: A traced function with .to_mlir() method
        stop_after: "trace" for trace-only, pass name for intermediate, None for full pipeline
        target: Hardware target (trn1, trn2, trn3)
        dump_dir: Optional directory to save intermediate MLIR

    Returns:
        Compiled MLIR/NISA IR as string
    """
    if stop_after == TRACE:
        # Trace only -- no pass pipeline
        from pass_utils import trace_to_mlir_with_preprocessing

        mlir_str = trace_to_mlir_with_preprocessing(traced_func)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, "00_traced.mlir"), "w") as f:
                f.write(mlir_str)
        return mlir_str

    elif stop_after is not None:
        # Intermediate pass -- use compile_knob_pipeline with stop_after
        from pass_utils import compile_knob_pipeline

        return compile_knob_pipeline(
            traced_func, stop_after=stop_after, dump_dir=dump_dir
        )

    else:
        # Full pipeline to NISA
        if target is None:
            target = _default_target()
        from pass_utils import trace_to_mlir_with_preprocessing
        from nkipy_kernelgen.transforms.nkipy_opt import apply_complete_knob_pipeline

        mlir_str = trace_to_mlir_with_preprocessing(traced_func)
        return apply_complete_knob_pipeline(
            mlir_str, target=target, dump_dir=dump_dir, print_generic=print_generic
        )


# ============================================================================
# Per-Mode Verification
# ============================================================================


def _run_string_check(compiled_ir: str, check_ir_contains, check_ir_not_contains):
    """Run simple string containment checks on the compiled IR."""
    if check_ir_contains:
        for pattern in check_ir_contains:
            assert pattern in compiled_ir, (
                f"Expected pattern not found in IR: '{pattern}'\n"
                f"IR (first 2000 chars):\n{compiled_ir[:2000]}"
            )
    if check_ir_not_contains:
        for pattern in check_ir_not_contains:
            assert pattern not in compiled_ir, (
                f"Unexpected pattern found in IR: '{pattern}'\n"
                f"IR (first 2000 chars):\n{compiled_ir[:2000]}"
            )


def _run_filecheck(compiled_ir: str, check_patterns: str):
    """Run FileCheck verification on the compiled IR."""
    from pass_utils import run_filecheck

    run_filecheck(compiled_ir, check_patterns)


def _run_llvm_verification(compiled_ir: str, traced_func, rtol: float, atol: float):
    """Run LLVM JIT verification against NumPy reference."""
    from pass_utils import verify_tiled_mlir_with_numpy

    verify_tiled_mlir_with_numpy(compiled_ir, traced_func, rtol=rtol, atol=atol)


def simulate_mlir(
    mlir_str: str,
    func_name: str,
    test_inputs: List[np.ndarray],
    expected_output: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    verbose: bool = False,
    keep_artifacts: bool = False,
    artifacts_dir: Optional[str] = None,
) -> Tuple[bool, float, Optional[str]]:
    """
    Run simulation on an MLIR string.

    Parses the MLIR, compiles to NEFF with simulation enabled, and compares
    the simulation output against expected_output.

    Args:
        mlir_str: MLIR module as a string
        func_name: Name of the function to simulate
        test_inputs: List of input tensors (in function argument order)
        expected_output: Expected output for validation
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        verbose: Print detailed output
        keep_artifacts: Keep debug artifacts on success
        artifacts_dir: Custom path for artifacts (created if doesn't exist).
                       If None, uses a temp directory.

    Returns:
        Tuple of (success, max_diff, artifacts_dir or None)
    """
    import tempfile
    from nki.compiler.ncc_driver import CompileOptions, compile_mlir_to_neff
    from nki.compiler._internal import ir, register_all_dialects

    # Setup artifacts directory
    if artifacts_dir:
        os.makedirs(artifacts_dir, exist_ok=True)
        debug_dir = artifacts_dir
    else:
        debug_dir = tempfile.mkdtemp(prefix="e2e_sim_")

    if verbose:
        print("MLIR for simulation:")
        print(mlir_str)

    opts = CompileOptions(
        target="trn2",
        verbose=False,
        output_path=os.path.join(debug_dir, "kernel.neff"),
        neuronx_cc_args=("--lnc=1",),
        artifacts_dir=debug_dir,
        enable_simulation=True,
        kernel_json_filename="kernel.json",
    )

    # Parse and simulate
    ctx = ir.Context()
    register_all_dialects(ctx)

    try:
        with ctx:
            mlir = ir.Module.parse(mlir_str, ctx)

            # Extract nki.output_names from the target function so that
            # output_arg_names match what the backend pipeline uses for BIRSim.
            output_arg_names = None
            for op in mlir.body.operations:
                if hasattr(op, "name") and hasattr(op, "attributes"):
                    op_name = None
                    try:
                        op_name = op.attributes["sym_name"]
                    except (KeyError, IndexError):
                        pass
                    if op_name and str(op_name).strip('"') == func_name:
                        try:
                            names_attr = op.attributes["nki.output_names"]
                            output_arg_names = [
                                str(names_attr[i]).strip('"')
                                for i in range(len(names_attr))
                            ]
                        except (KeyError, IndexError):
                            pass
            if output_arg_names is None:
                output_arg_names = ["out_tensor"]

            input_names = [f"in_tensor_{i}" for i in range(len(test_inputs))]

            # Extract output names from nki.output_names attribute set by
            # prepare-for-nki pass; fall back to "output_0" if not present.
            output_names = ["output_0"]
            for op in mlir.body.operations:
                if (
                    "function_type" in op.attributes
                    and "nki.output_names" in op.attributes
                ):
                    names_attr = ir.ArrayAttr(op.attributes["nki.output_names"])
                    output_names = [
                        str(ir.StringAttr(names_attr[i])).strip('"')
                        for i in range(len(names_attr))
                    ]
                    break

            # Normalize expected_output to list
            if isinstance(expected_output, (list, tuple)):
                expected_outputs = list(expected_output)
            else:
                expected_outputs = [expected_output]

            output_placeholders = [np.zeros_like(eo) for eo in expected_outputs]
            all_arrays = list(test_inputs) + output_placeholders
            argument_names = input_names + output_names
            output_arg_names = output_names

            compile_result = compile_mlir_to_neff(
                mlir,
                func_name,
                all_arrays,
                argument_names,
                output_arg_names,
                opts,
            )

            if compile_result.birsim_outputs is None:
                err = getattr(compile_result, 'neuronx_cc_error', None)
                raise RuntimeError(
                    f"BIRSim produced no outputs. "
                    f"neuronx-cc error: {err}, artifacts: {debug_dir}"
                )
            results = [
                compile_result.birsim_outputs[i] for i in range(len(expected_outputs))
            ]
    except Exception as e:
        import traceback

        print(f"Parsing/simulation failed: {e}")
        print(f"Artifacts: {debug_dir}")
        if verbose:
            traceback.print_exc()
        return False, float("inf"), debug_dir

    # Validate all outputs
    max_diff = 0.0
    matches = True
    for i, (result, expected) in enumerate(zip(results, expected_outputs)):
        diff = np.max(np.abs(result - expected))
        max_diff = max(max_diff, diff)
        if not np.allclose(result, expected, rtol=rtol, atol=atol):
            matches = False

    if verbose:
        print(f"Max difference: {max_diff:.2e}")
        print(f"Match: {matches}")

    # Cleanup if success and not keeping artifacts (only for temp dirs)
    artifacts_path = debug_dir
    if matches and not keep_artifacts and not artifacts_dir:
        import shutil

        try:
            shutil.rmtree(debug_dir)
            artifacts_path = None
        except:
            pass

    return matches, max_diff, artifacts_path


def _compile_nisa_to_neff(
    compiled_ir: str,
    traced_func,
    inputs: List[np.ndarray],
    reference_output: np.ndarray,
    target: Optional[str],
    dump_dir: Optional[str],
    enable_simulation: bool = True,
):
    """Compile NISA IR to NEFF (and optionally run BIR simulation).

    This is the shared compilation step for both BIR_SIM and HW modes.
    When enable_simulation=True, the result includes birsim_outputs.
    The NEFF is always produced and can be used for HW execution.

    Returns:
        Tuple of (compile_result, input_names, debug_dir)
    """
    import tempfile
    from nki.compiler.ncc_driver import CompileOptions, compile_mlir_to_neff
    from nki.compiler._internal import ir, register_all_dialects

    if target is None:
        target = _default_target()

    func_name = traced_func.__wrapped__.__name__
    debug_dir = dump_dir or tempfile.mkdtemp(prefix="e2e_compile_")
    os.makedirs(debug_dir, exist_ok=True)

    opts = CompileOptions(
        target=target,
        verbose=False,
        output_path=os.path.join(debug_dir, "kernel.neff"),
        neuronx_cc_args=("--lnc=1",),
        artifacts_dir=debug_dir,
        enable_simulation=enable_simulation,
        kernel_json_filename="kernel.json",
    )

    ctx = ir.Context()
    register_all_dialects(ctx)
    with ctx:
        mlir = ir.Module.parse(compiled_ir, ctx)

        input_names = [f"in_tensor_{i}" for i in range(len(inputs))]

        output_names = ["output_0"]
        for op in mlir.body.operations:
            if "function_type" in op.attributes and "nki.output_names" in op.attributes:
                names_attr = ir.ArrayAttr(op.attributes["nki.output_names"])
                output_names = [
                    str(ir.StringAttr(names_attr[i])).strip('"')
                    for i in range(len(names_attr))
                ]
                break

        output_placeholders = [np.zeros_like(ro) for ro in reference_output]
        all_arrays = list(inputs) + output_placeholders
        argument_names = input_names + output_names
        output_arg_names = output_names

        compile_result = compile_mlir_to_neff(
            mlir,
            func_name,
            all_arrays,
            argument_names,
            output_arg_names,
            opts,
        )

    return compile_result, input_names, debug_dir


def _run_bir_sim(
    compiled_ir: str,
    traced_func,
    inputs: List[np.ndarray],
    reference_output: List[np.ndarray],
    rtol: float,
    atol: float,
    dump_dir: Optional[str],
    compile_result=None,
    target: Optional[str] = None,
):
    """Run BIR simulation and compare to reference.

    If compile_result is provided, uses the pre-compiled birsim_outputs.
    Otherwise compiles from scratch.
    """
    if compile_result is None:
        compile_result, _, _ = _compile_nisa_to_neff(
            compiled_ir,
            traced_func,
            inputs,
            reference_output,
            target=target,
            dump_dir=dump_dir,
            enable_simulation=True,
        )

    if compile_result.birsim_outputs is None:
        err = getattr(compile_result, 'neuronx_cc_error', None) or "unknown"
        artifacts = getattr(compile_result, 'artifacts_dir', None) or "N/A"
        raise AssertionError(
            f"BIRSim produced no outputs (birsim_outputs is None).\n"
            f"neuronx-cc error: {err}\n"
            f"Artifacts dir: {artifacts}"
        )

    for i, expected in enumerate(reference_output):
        result = compile_result.birsim_outputs[i]
        max_diff = np.max(np.abs(result - expected))
        success = np.allclose(result, expected, rtol=rtol, atol=atol)

        print(f"\nOutput {i} max difference: {max_diff:.2e}")
        print(f"Output {i} match: {success}")

        assert success, (
            f"BIR simulation failed on output {i} with max_diff={max_diff:.2e} "
            f"(rtol={rtol}, atol={atol})"
        )


def _run_hw_execution(
    compiled_ir: str,
    traced_func,
    inputs: List[np.ndarray],
    reference_output: List[np.ndarray],
    rtol: float,
    atol: float,
    target: Optional[str],
    dump_dir: Optional[str],
    compile_result=None,
    input_names=None,
    compile_debug_dir=None,
):
    """Run hardware execution and compare to reference.

    If compile_result is provided, reuses the NEFF from a previous compilation.
    Otherwise compiles from scratch.
    """
    import pytest

    if not is_hw_available():
        pytest.skip("No Trainium device detected -- skipping Mode.HW")

    from nki.runtime import SpikeModel, SpikeTensor

    if compile_result is None:
        compile_result, input_names, compile_debug_dir = _compile_nisa_to_neff(
            compiled_ir,
            traced_func,
            inputs,
            reference_output,
            target=target,
            dump_dir=dump_dir,
            enable_simulation=False,
        )
    if input_names is None:
        input_names = [f"in_tensor_{i}" for i in range(len(inputs))]

    neff_path = compile_result.neff_path
    model = SpikeModel.load_from_neff(neff_path)

    neff_input_names = list(model.input_tensors_info.keys())
    compile_input_map = dict(zip(input_names, inputs))
    spike_inputs = {
        name: SpikeTensor.from_numpy(compile_input_map[name], name=name)
        for name in neff_input_names
    }

    spike_outputs = model(inputs=spike_inputs, outputs=None)
    artifacts = compile_debug_dir or dump_dir or "unknown"

    # Look up outputs by name rather than relying on dict iteration order,
    # which may not match the expected order.
    neff_output_names = sorted(model.output_tensors_info.keys())

    for i, expected in enumerate(reference_output):
        name = neff_output_names[i] if i < len(neff_output_names) else None
        result_tensor = spike_outputs.get(name) if name else list(spike_outputs.values())[i]
        raw = result_tensor.numpy()
        # SpikeTensor.numpy() may return void dtype (V4). Interpret using
        # the expected element size to determine the correct float type.
        if raw.dtype.kind == 'V':
            elem_size = raw.dtype.itemsize
            float_dtype = {2: np.float16, 4: np.float32, 8: np.float64}.get(elem_size, np.float32)
            result = raw.view(float_dtype)
        else:
            result = raw
        # Cast expected to match the HW output dtype (e.g., bool -> f32 for
        # comparison ops, since NISA always produces float results).
        if result.dtype != expected.dtype:
            expected = expected.astype(result.dtype)

        max_diff = np.max(np.abs(result - expected))
        success = np.allclose(result, expected, rtol=rtol, atol=atol)
        assert success, (
            f"HW execution failed on output {i} with max_diff={max_diff:.2e} "
            f"(rtol={rtol}, atol={atol}). Artifacts: {artifacts}"
        )


# ============================================================================
# Dump-IR Helpers
# ============================================================================


def _print_dump_dir_listing(dump_dir: str) -> None:
    """Print the list of IR files saved in dump_dir."""
    if not dump_dir or not os.path.isdir(dump_dir):
        return
    files = sorted(f for f in os.listdir(dump_dir) if f.endswith((".mlir", ".txt")))
    if not files:
        print(f"[dump-ir] No IR files in {dump_dir}")
        return
    print(f"[dump-ir] {len(files)} IR files saved to: {dump_dir}")
    for f in files:
        size = os.path.getsize(os.path.join(dump_dir, f))
        print(f"  {f}  ({size:,} bytes)")


# ============================================================================
# Main Entry Point
# ============================================================================


def run_kernel_test(
    traced_func,
    *,
    stop_after=None,
    target: Optional[str] = None,
    check_patterns: Optional[str] = None,
    check_ir_contains: Optional[List[str]] = None,
    check_ir_not_contains: Optional[List[str]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    seed: int = 42,
    inputs: Optional[List[np.ndarray]] = None,
    reference_output: Optional[np.ndarray] = None,
    modes: Mode = Mode.LLVM,
    dump_dir: Optional[str] = None,
    request=None,
):
    """Unified test runner for NKIPyKernelGen kernels.

    Compiles the traced function through the pipeline and runs all requested
    verification modes.

    Args:
        traced_func: A @trace-decorated function
        stop_after: Pipeline stop point:
            - "trace": trace to MLIR only, no passes
            - "<pass-name>": stop after a specific pass
            - None: run all passes (full pipeline to NISA)
        target: Hardware target (trn1, trn2, trn3)
        check_patterns: FileCheck patterns string (required for Mode.FILECHECK)
        check_ir_contains: List of strings that must appear in compiled IR
        check_ir_not_contains: List of strings that must NOT appear in compiled IR
        rtol: Relative tolerance override (None = use per-mode defaults)
        atol: Absolute tolerance override (None = use per-mode defaults)
        seed: Random seed for input generation
        inputs: Override auto-generated inputs
        reference_output: Override auto-computed reference output
        modes: Verification modes to run (combined with |)
        dump_dir: Directory for compilation artifacts. Auto-created from
            request.node.name if request is provided and dump_dir is None.
        request: pytest request fixture (for auto-naming artifacts)

    Raises:
        ValueError: If parameter validation fails
        AssertionError: If any verification mode fails
    """
    # 1. Validate parameters
    _validate_params(
        modes, stop_after, check_patterns, check_ir_contains, check_ir_not_contains
    )

    # 1b. Strip Mode.HW if hardware cannot run it (trn1, no device, or
    #     device/target mismatch). Other modes still execute normally.
    if Mode.HW in modes and not _can_run_hw(target):
        modes = modes & ~Mode.HW
        if not modes:
            pytest.skip("Test requires Mode.HW but no compatible device detected")

    # 2. Resolve artifact directory
    #    --dump-ir: always create a dump directory so intermediate IR is saved.
    #    Without --dump-ir: only create when request is provided (backward compat).
    if dump_dir is None and request is not None:
        dump_dir = os.path.join(
            os.path.dirname(os.path.abspath(request.fspath)),
            "outputs",
            request.node.name,
        )
    elif dump_dir is None and _DUMP_IR_ENABLED:
        # --dump-ir without request fixture: use a temp directory
        dump_dir = tempfile.mkdtemp(prefix="nkipy_dump_ir_")

    if dump_dir and _DUMP_IR_ENABLED:
        print(f"\n[dump-ir] IR will be saved to: {dump_dir}")

    # 3. Generate inputs if not provided
    if inputs is None:
        inputs = generate_inputs(traced_func.input_specs, seed=seed)

    # 4. Compute reference output if not provided, and normalize to list
    if reference_output is None:
        reference_output = compute_reference(traced_func, inputs)
    elif not isinstance(reference_output, list):
        if isinstance(reference_output, tuple):
            reference_output = list(reference_output)
        else:
            reference_output = [reference_output]

    # 5. Compile through pipeline
    #    On failure: if no dump_dir was set, re-run pass-by-pass into a temp
    #    directory so the user gets intermediate IR for debugging.
    try:
        compiled_ir = _compile_pipeline(
            traced_func, stop_after, target=target, dump_dir=dump_dir
        )
    except Exception as exc:
        if dump_dir:
            # dump_dir was already set — IR files are already there
            _print_dump_dir_listing(dump_dir)
            raise
        # No dump_dir: re-run pass-by-pass to capture intermediate IR
        fallback_dir = tempfile.mkdtemp(prefix="nkipy_fail_dump_")
        print(f"\n[dump-ir] Compilation failed — dumping intermediate IR to: {fallback_dir}")
        try:
            _compile_pipeline(
                traced_func, stop_after, target=target, dump_dir=fallback_dir
            )
        except Exception:
            pass  # expected to fail again; we just want the IR files
        _print_dump_dir_listing(fallback_dir)
        raise exc

    # 6. Run each verification mode
    if Mode.STRING_CHECK in modes:
        _run_string_check(compiled_ir, check_ir_contains, check_ir_not_contains)

    if Mode.FILECHECK in modes:
        _run_filecheck(compiled_ir, check_patterns)

    if Mode.LLVM in modes:
        tol = _resolve_tolerances(Mode.LLVM, rtol, atol)
        _run_llvm_verification(compiled_ir, traced_func, **tol)

    # BIR_SIM and HW both need NISA→NEFF compilation. Compile once if both
    # are requested, then share the result.
    need_bir = Mode.BIR_SIM in modes
    need_hw = Mode.HW in modes

    if need_bir or need_hw:
        try:
            compile_result, input_names, compile_dir = _compile_nisa_to_neff(
                compiled_ir,
                traced_func,
                inputs,
                reference_output,
                target=target,
                dump_dir=dump_dir,
                enable_simulation=need_bir,
            )
        except Exception:
            # Non-round-trippable custom assembly (e.g. view(...) syntax)
            # can't be parsed. Recompile with generic MLIR form.
            compiled_ir = _compile_pipeline(
                traced_func,
                stop_after,
                target=target,
                dump_dir=dump_dir,
                print_generic=True,
            )
            compile_result, input_names, compile_dir = _compile_nisa_to_neff(
                compiled_ir,
                traced_func,
                inputs,
                reference_output,
                target=target,
                dump_dir=dump_dir,
                enable_simulation=need_bir,
            )

    if need_bir:
        tol = _resolve_tolerances(Mode.BIR_SIM, rtol, atol)
        _run_bir_sim(
            compiled_ir,
            traced_func,
            inputs,
            reference_output,
            dump_dir=dump_dir,
            compile_result=compile_result,
            target=target,
            **tol,
        )

    if need_hw:
        tol = _resolve_tolerances(Mode.HW, rtol, atol)
        _run_hw_execution(
            compiled_ir,
            traced_func,
            inputs,
            reference_output,
            target=target,
            dump_dir=dump_dir,
            compile_result=compile_result,
            input_names=input_names,
            compile_debug_dir=compile_dir,
            **tol,
        )

    # Print dump directory listing on success when --dump-ir is active
    if dump_dir and _DUMP_IR_ENABLED:
        _print_dump_dir_listing(dump_dir)


def _resolve_tolerances(mode: Mode, rtol: Optional[float], atol: Optional[float]):
    """Get tolerances: use overrides if provided, else per-mode defaults."""
    defaults = DEFAULT_TOLERANCES.get(mode, {"rtol": 1e-5, "atol": 1e-6})
    return {
        "rtol": rtol if rtol is not None else defaults["rtol"],
        "atol": atol if atol is not None else defaults["atol"],
    }


# ============================================================================
# Decorator
# ============================================================================


def nkipy_kernelgen_test(
    input_specs,
    *,
    modes: Mode = Mode.LLVM,
    stop_after=None,
    **kwargs,
):
    """Decorator that turns a kernel function into a pytest test.

    The decorated function is the kernel body. It will be traced with @trace
    using the given input_specs, then compiled and verified according to the
    specified modes.

    Args:
        input_specs: Input specifications for @trace, e.g. [((256, 256), "f32")]
        modes: Verification modes (combined with |)
        stop_after: Pipeline stop point ("trace", "<pass-name>", or None)
        **kwargs: Additional arguments passed to run_kernel_test()

    Returns:
        A pytest-compatible test function

    Example:
        @nkipy_kernelgen_test(
            input_specs=[((256, 256), "f32"), ((256, 256), "f32")],
            stop_after="apply-and-strip-transforms",
            check_patterns="CHECK: scf.for",
            modes=Mode.LLVM | Mode.FILECHECK,
        )
        def test_matmul_tiling(a, b):
            result = np.matmul(a, b)
            knob.knob(result, tile_size=[128, 128, 128])
            return result
    """
    # Validate early (at decoration time) so misconfigured tests fail on import
    check_patterns = kwargs.get("check_patterns")
    check_ir_contains = kwargs.get("check_ir_contains")
    check_ir_not_contains = kwargs.get("check_ir_not_contains")
    _validate_params(
        modes, stop_after, check_patterns, check_ir_contains, check_ir_not_contains
    )

    def decorator(func):
        def test_wrapper():
            from nkipy_kernelgen import trace as nkipy_trace

            traced = nkipy_trace(input_specs=input_specs)(func)
            run_kernel_test(
                traced,
                modes=modes,
                stop_after=stop_after,
                **kwargs,
            )

        # Preserve name/qualname for pytest discovery (do NOT use functools.wraps
        # here -- it copies __wrapped__/__signature__ from func, which makes pytest
        # think test_wrapper has func's kernel parameters and try to resolve them
        # as fixtures)
        test_wrapper.__name__ = func.__name__
        test_wrapper.__qualname__ = func.__qualname__
        test_wrapper.__doc__ = func.__doc__
        test_wrapper.__module__ = func.__module__
        return test_wrapper

    return decorator
