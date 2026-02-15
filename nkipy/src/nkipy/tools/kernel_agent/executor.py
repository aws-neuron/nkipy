# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel execution: numpy → compile → hardware pipeline."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class StageResult:
    """Result of a single execution stage."""

    success: bool
    output: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Complete execution result."""

    numpy: Optional[StageResult] = None
    compile: Optional[StageResult] = None
    hardware: Optional[StageResult] = None

    @property
    def passed(self) -> bool:
        return all(
            s is not None and s.success
            for s in [self.numpy, self.compile, self.hardware]
        )

    @property
    def summary(self) -> str:
        if self.passed:
            return "pass"
        for stage in ["hardware", "compile", "numpy"]:
            s = getattr(self, stage)
            if s and not s.success:
                return (
                    f"fail: {stage} - {s.error[:50]}" if s.error else f"fail: {stage}"
                )
        return "fail"


def run_kernel(
    kernel_fn: Callable,
    inputs: Dict[str, np.ndarray],
    run_hardware: bool = True,
    artifacts_dir: Optional[str] = None,
) -> ExecutionResult:
    """Run kernel through numpy → compile → hardware pipeline."""
    result = ExecutionResult()

    # Get args from function signature
    sig = inspect.signature(kernel_fn)
    args = [inputs[p] for p in sig.parameters]

    # Stage 1: Pure NumPy
    try:
        out = kernel_fn(*args)
        result.numpy = StageResult(success=True, output=np.asarray(out))
    except Exception as e:
        result.numpy = StageResult(success=False, error=str(e))
        return result

    # Stage 2: Trace + simulate
    try:
        from nkipy.core.trace import NKIPyKernel
        from nkipy.runtime.execute import simulate_traced_kernel

        traced = NKIPyKernel.trace(kernel_fn)
        out = simulate_traced_kernel(traced, *args)
        result.compile = StageResult(success=True, output=np.asarray(out))
    except Exception as e:
        result.compile = StageResult(success=False, error=str(e))
        return result

    # Stage 3: Hardware
    if run_hardware:
        try:
            from nkipy.core.trace import NKIPyKernel
            from nkipy.runtime.execute import baremetal_run_traced_kernel

            traced = NKIPyKernel.trace(kernel_fn)
            out = baremetal_run_traced_kernel(
                traced, *args, artifacts_dir=artifacts_dir
            )
            result.hardware = StageResult(success=True, output=np.asarray(out))
        except Exception as e:
            result.hardware = StageResult(success=False, error=str(e))
    else:
        result.hardware = StageResult(success=True, output=result.compile.output)

    return result


def compare_outputs(result: ExecutionResult) -> Dict[str, Any]:
    """Compare numpy vs hardware outputs."""
    if not result.numpy or not result.hardware:
        return {}
    if not result.numpy.success or not result.hardware.success:
        return {}

    np_out = result.numpy.output
    hw_out = result.hardware.output

    return {
        "shapes_match": np_out.shape == hw_out.shape,
        "max_diff": float(np.max(np.abs(np_out - hw_out))),
        "allclose": bool(np.allclose(np_out, hw_out, rtol=1e-4, atol=1e-4)),
    }
