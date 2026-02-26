# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Continuous kernel sweep: generate and test diverse numpy kernels overnight."""

import json
import random
import signal
import time
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
    from nkipy.tools.kernel_agent.executor import compare_outputs, run_kernel
    from nkipy.tools.kernel_agent.generator import compile_code, generate_kernel

    global _shutdown
    _shutdown = False

    prompts = get_kernel_prompts()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = out_path / f"sweep_{ts}.jsonl"

    # Install signal handlers for graceful shutdown
    prev_sigint = signal.signal(signal.SIGINT, _handle_signal)
    prev_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    prev_sigalrm = signal.signal(signal.SIGALRM, _handle_alarm)

    iteration = 0
    passed_count = 0
    fail_counts: Dict[str, int] = {}
    start_time = time.time()

    try:
        with open(log_path, "a") as f:
            print(f"Sweep started, logging to {log_path}")
            print(f"Model: {model_id}, hardware: {run_hardware}")
            if timeout:
                print(f"Per-iteration timeout: {timeout}s")
            if max_iterations:
                print(f"Max iterations: {max_iterations}")
            print()

            while not _shutdown:
                if max_iterations is not None and iteration >= max_iterations:
                    break

                iteration += 1
                prompt = random.choice(prompts)
                iter_start = time.time()

                record = SweepRecord(
                    iteration=iteration,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    prompt=prompt,
                    model_id=model_id,
                )

                # Print what we're about to run
                print(
                    f"  [{iteration}] RUN    {prompt[:60]}",
                    flush=True,
                )

                # Arm the per-iteration timeout
                if timeout:
                    signal.alarm(timeout)

                try:
                    # Stage: LLM generation
                    try:
                        name, code, inputs = generate_kernel(prompt, model_id, region)
                    except Exception as e:
                        record.failure_stage = "generation"
                        record.error_message = str(e)
                        fail_counts["generation"] = fail_counts.get("generation", 0) + 1
                        continue
                    record.kernel_name = name
                    record.generated_code = code
                    record.input_specs = _input_specs_from_arrays(inputs)

                    # Stage: compile generated code to callable
                    kernel_fn = compile_code(code)
                    if kernel_fn is None:
                        record.failure_stage = "compile_code"
                        record.error_message = "compile_code returned None"
                        fail_counts["compile_code"] = (
                            fail_counts.get("compile_code", 0) + 1
                        )
                        continue

                    # Stage: execution pipeline
                    result = run_kernel(
                        kernel_fn,
                        inputs,
                        run_hardware=run_hardware,
                    )

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

                    # Determine failure stage from execution result
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
                        passed_count += 1

                    if record.failure_stage:
                        fail_counts[record.failure_stage] = (
                            fail_counts.get(record.failure_stage, 0) + 1
                        )

                except _IterationTimeout:
                    record.failure_stage = "timeout"
                    record.error_message = f"Exceeded {timeout}s timeout"
                    fail_counts["timeout"] = fail_counts.get("timeout", 0) + 1

                finally:
                    # Always cancel pending alarm
                    signal.alarm(0)

                record.duration_seconds = time.time() - iter_start
                _log_record(f, record)

                # Print result
                if record.passed:
                    status = f"{_GREEN}PASS{_RESET}"
                else:
                    status = f"{_RED}FAIL {record.failure_stage}{_RESET}"
                print(f"  [{iteration}] {status:30} {record.duration_seconds:5.1f}s")

                # Periodic summary
                if iteration % summary_interval == 0:
                    _print_summary(
                        fail_counts,
                        iteration,
                        passed_count,
                        start_time,
                    )

                if delay > 0:
                    time.sleep(delay)

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGALRM, prev_sigalrm)

    # Final summary
    _print_summary(fail_counts, iteration, passed_count, start_time)
    print(f"Results saved to {log_path}")
    return log_path
