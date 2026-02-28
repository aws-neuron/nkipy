# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuous kernel sweep: generate and test diverse numpy kernels overnight."""

import json
import random
import signal
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from nkipy.tools.kernel_agent.generator import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    load_prompt,
)


def get_kernel_prompts() -> List[str]:
    """Load kernel prompts from prompts/kernel_prompts.txt."""
    text = load_prompt("kernel_prompts.txt")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


@dataclass
class SweepRecord:
    """Record for a single sweep iteration."""

    iteration: int
    timestamp: str
    prompt: str
    model_id: str
    kernel_name: str = ""
    generated_code: str = ""
    input_specs: Dict[str, Any] = field(default_factory=dict)
    numpy_success: Optional[bool] = None
    numpy_error: Optional[str] = None
    compile_success: Optional[bool] = None
    compile_error: Optional[str] = None
    hardware_success: Optional[bool] = None
    hardware_error: Optional[str] = None
    max_diff: Optional[float] = None
    allclose: Optional[bool] = None
    shapes_match: Optional[bool] = None
    passed: bool = False
    failure_stage: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    rerun_source: Optional[str] = None


# ANSI color helpers
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"

# Global shutdown flag set by signal handler
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\nShutdown requested, finishing current iteration...")


class _IterationTimeout(Exception):
    pass


def _handle_alarm(signum, frame):
    raise _IterationTimeout()


@contextmanager
def _sweep_signals():
    """Set up and tear down signal handlers for graceful shutdown."""
    global _shutdown
    _shutdown = False
    prev_sigint = signal.signal(signal.SIGINT, _handle_signal)
    prev_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    prev_sigalrm = signal.signal(signal.SIGALRM, _handle_alarm)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGALRM, prev_sigalrm)


def _log_record(f, record: SweepRecord) -> None:
    """Append a JSON line to the log file and flush immediately."""
    f.write(json.dumps(asdict(record), default=str) + "\n")
    f.flush()


def _input_specs_from_arrays(inputs: Dict) -> Dict[str, Any]:
    """Extract shape/dtype specs from numpy arrays for logging."""
    import numpy as np

    specs = {}
    for name, arr in inputs.items():
        if isinstance(arr, np.ndarray):
            specs[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return specs


def _run_and_record(
    record: SweepRecord,
    kernel_fn,
    inputs: Dict,
    run_hardware: bool,
) -> bool:
    """Execute a kernel and record results into *record*.

    Returns True if the kernel passed all stages, False otherwise.
    """
    from nkipy.tools.kernel_agent.executor import compare_outputs, run_kernel

    result = run_kernel(kernel_fn, inputs, run_hardware=run_hardware)

    # Record per-stage results
    if result.numpy:
        record.numpy_success = result.numpy.success
        if result.numpy.error:
            record.numpy_error = result.numpy.error
    if result.compile:
        record.compile_success = result.compile.success
        if result.compile.error:
            record.compile_error = result.compile.error
    if result.hardware:
        record.hardware_success = result.hardware.success
        if result.hardware.error:
            record.hardware_error = result.hardware.error

    # Determine failure stage
    if result.numpy and not result.numpy.success:
        record.failure_stage = "numpy"
        record.error_message = result.numpy.error
    elif result.compile and not result.compile.success:
        record.failure_stage = "compile"
        record.error_message = result.compile.error
    elif result.hardware and not result.hardware.success:
        record.failure_stage = "hardware"
        record.error_message = result.hardware.error

    # Compare outputs if all stages passed
    if result.passed:
        comp = compare_outputs(result)
        record.max_diff = comp.get("max_diff")
        record.allclose = comp.get("allclose")
        record.shapes_match = comp.get("shapes_match")
        record.passed = True

    return record.passed


def _execute_iteration(record, body_fn, f, fail_counts, timeout, *, propagate=()):
    """Run one iteration with timeout, error handling, and centralized accounting.

    Args:
        record: SweepRecord to populate.
        body_fn: Callable returning True if iteration passed.
        f: Open file handle for JSONL logging.
        fail_counts: Dict tracking failure counts by stage.
        timeout: Per-iteration timeout in seconds (0 to disable).
        propagate: Exception types to re-raise after logging.

    Returns:
        True if the iteration passed, False otherwise.
    """
    iter_start = time.time()
    if timeout:
        signal.alarm(timeout)
    passed = False
    try:
        passed = body_fn()
    except _IterationTimeout:
        record.failure_stage = "timeout"
        record.error_message = f"Exceeded {timeout}s timeout"
    except Exception as e:
        if propagate and isinstance(e, propagate):
            if not record.failure_stage:
                record.failure_stage = "unexpected"
                record.error_message = str(e)
            raise  # finally still runs (record logged), then exception propagates
        if not record.failure_stage:
            record.failure_stage = "unexpected"
            record.error_message = str(e)
    finally:
        signal.alarm(0)
        if record.failure_stage:
            fail_counts[record.failure_stage] = (
                fail_counts.get(record.failure_stage, 0) + 1
            )
        record.duration_seconds = time.time() - iter_start
        _log_record(f, record)
        if record.passed:
            status = f"{_GREEN}PASS{_RESET}"
        else:
            status = f"{_RED}FAIL {record.failure_stage}{_RESET}"
        print(f"  [{record.iteration}] {status:30} {record.duration_seconds:5.1f}s")
    return passed


def _print_summary(
    counts: Dict[str, int], total: int, passed: int, start_time: float
) -> None:
    """Print a summary of sweep progress."""
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\n--- Summary after {total} iterations ({elapsed:.0f}s, {rate:.2f}/s) ---")
    pct = 100 * passed / total if total > 0 else 0.0
    print(f"  {_GREEN}Passed:{_RESET}     {passed}/{total} ({pct:.1f}%)")
    for stage, count in sorted(counts.items()):
        print(f"  {_RED}Failed at {stage:12}: {count}{_RESET}")
    print()


def run_sweep(
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
    run_hardware: bool = True,
    max_iterations: Optional[int] = None,
    output_dir: str = "sweep_results",
    delay: float = 0,
    summary_interval: int = 10,
    timeout: int = 120,
) -> Path:
    """Run continuous kernel generation and testing sweep.

    Returns the path to the JSONL log file.
    """
    import botocore.exceptions

    from nkipy.tools.kernel_agent.generator import compile_code, generate_kernel

    prompts = get_kernel_prompts()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = out_path / f"sweep_{ts}.jsonl"

    iteration = 0
    passed_count = 0
    fail_counts: Dict[str, int] = {}
    start_time = time.time()

    with _sweep_signals():
        with open(log_path, "a") as f:
            print(f"Sweep started, logging to {log_path}")
            print(f"Model: {model_id}, hardware: {run_hardware}")
            if timeout:
                print(f"Per-iteration timeout: {timeout}s")
            if max_iterations:
                print(f"Max iterations: {max_iterations}")
            print()

            try:
                while not _shutdown:
                    if max_iterations is not None and iteration >= max_iterations:
                        break

                    iteration += 1
                    prompt = random.choice(prompts)

                    record = SweepRecord(
                        iteration=iteration,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        prompt=prompt,
                        model_id=model_id,
                    )

                    print(
                        f"  [{iteration}] RUN    {prompt[:60]}",
                        flush=True,
                    )

                    def body(rec=record):
                        try:
                            name, code, inputs = generate_kernel(
                                rec.prompt, model_id, region
                            )
                        except Exception as e:
                            rec.failure_stage = "generation"
                            rec.error_message = str(e)
                            raise

                        rec.kernel_name = name
                        rec.generated_code = code
                        rec.input_specs = _input_specs_from_arrays(inputs)

                        kernel_fn = compile_code(code)
                        if kernel_fn is None:
                            rec.failure_stage = "compile_code"
                            rec.error_message = "compile_code returned None"
                            return False

                        return _run_and_record(rec, kernel_fn, inputs, run_hardware)

                    passed = _execute_iteration(
                        record,
                        body,
                        f,
                        fail_counts,
                        timeout,
                        propagate=(
                            botocore.exceptions.BotoCoreError,
                            botocore.exceptions.ClientError,
                        ),
                    )
                    if passed:
                        passed_count += 1

                    if summary_interval > 0 and iteration % summary_interval == 0:
                        _print_summary(
                            fail_counts,
                            iteration,
                            passed_count,
                            start_time,
                        )

                    if delay > 0:
                        time.sleep(delay)

            finally:
                if iteration > 0:
                    _print_summary(fail_counts, iteration, passed_count, start_time)
                    print(f"Results saved to {log_path}")

    return log_path


def run_rerun(
    source_path: str,
    run_hardware: bool = True,
    output_dir: str = "sweep_results",
    timeout: int = 120,
    summary_interval: int = 10,
) -> Path:
    """Re-execute kernels from a previous sweep JSONL file.

    Args:
        source_path: Path to the source JSONL file from a previous sweep.
        run_hardware: Whether to run hardware tests.
        output_dir: Directory for output JSONL file.
        timeout: Per-iteration timeout in seconds.
        summary_interval: Print summary every N iterations.

    Returns:
        Path to the rerun output JSONL file.
    """
    from nkipy.tools.kernel_agent.generator import build_inputs, compile_code

    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if src.suffix != ".jsonl":
        raise ValueError(f"Source file must be a .jsonl file, got: {src.suffix!r}")

    # Read and filter source records
    source_records = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            source_records.append(json.loads(line))

    eligible = [rec for rec in source_records if rec.get("generated_code")]

    if not eligible:
        raise ValueError(f"No eligible records to rerun in {source_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = out_path / f"rerun_{ts}.jsonl"

    iteration = 0
    passed_count = 0
    fail_counts: Dict[str, int] = {}
    start_time = time.time()

    with _sweep_signals():
        with open(log_path, "a") as f:
            print(f"Rerun started from {src}, logging to {log_path}")
            print(f"Eligible: {len(eligible)}, hardware: {run_hardware}")
            if timeout:
                print(f"Per-iteration timeout: {timeout}s")
            print()

            try:
                for src_rec in eligible:
                    if _shutdown:
                        break

                    iteration += 1

                    record = SweepRecord(
                        iteration=iteration,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        prompt=src_rec.get("prompt", ""),
                        model_id=src_rec.get("model_id", ""),
                        kernel_name=src_rec.get("kernel_name", ""),
                        generated_code=src_rec["generated_code"],
                        input_specs=src_rec.get("input_specs", {}),
                        rerun_source=str(src),
                    )

                    print(
                        f"  [{iteration}] RERUN  {record.prompt[:60]}",
                        flush=True,
                    )

                    def body(rec=record):
                        kernel_fn = compile_code(rec.generated_code)
                        if kernel_fn is None:
                            rec.failure_stage = "compile_code"
                            rec.error_message = "compile_code returned None"
                            return False

                        inputs = build_inputs(rec.input_specs)
                        return _run_and_record(rec, kernel_fn, inputs, run_hardware)

                    passed = _execute_iteration(record, body, f, fail_counts, timeout)
                    if passed:
                        passed_count += 1

                    if summary_interval > 0 and iteration % summary_interval == 0:
                        _print_summary(
                            fail_counts,
                            iteration,
                            passed_count,
                            start_time,
                        )

            finally:
                if iteration > 0:
                    _print_summary(fail_counts, iteration, passed_count, start_time)
                    print(f"Results saved to {log_path}")

    return log_path
