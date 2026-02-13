# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities for NKIPy tests"""

import os
import shutil
import tempfile
from functools import partial

import numpy as np
import pytest
from nkipy.core import compile
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import is_neuron_compatible
from nkipy.runtime.execute import baremetal_run_traced_kernel

# Constants for tolerance levels
CPU_RTOL = 1e-4
CPU_ATOL = 1e-4
BAREMETAL_RTOL = 1e-2
BAREMETAL_ATOL = 1e-2

# Pre-configured assert functions
cpu_assert_allclose = partial(np.testing.assert_allclose, rtol=CPU_RTOL, atol=CPU_ATOL)
baremetal_assert_allclose = partial(
    np.testing.assert_allclose, rtol=BAREMETAL_RTOL, atol=BAREMETAL_ATOL
)

# Hardware availability check
NEURON_AVAILABLE = is_neuron_compatible()


# Trace mode fixture - tests will run with HLO tracing
@pytest.fixture(params=["hlo"])
def trace_mode(request):
    """Fixture to run tests with HLO tracing mode"""
    return request.param


def trace_and_compile(kernel_fn, trace_mode, *args, **kwargs):
    """
    Validate kernel is traceable to HLO and compilable to NEFF.

    Traces the kernel to HLO IR and compiles it using the Neuron compiler,
    but does not execute on device.

    Args:
        kernel_fn: The kernel function to test
        trace_mode: "hlo" or other supported tracing mode
        *args: Input arrays for the kernel
        **kwargs: Additional arguments
    """
    if trace_mode == "hlo":
        traced_kernel = NKIPyKernel.trace(kernel_fn, backend="hlo")
    else:
        raise ValueError(f"Unknown trace mode: {trace_mode}")

    # Trace to HLO
    traced_kernel.specialize(*args, **kwargs)

    # Compile to NEFF
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    artifacts_dir = os.path.join(tempfile.gettempdir(), f"nkipy_artifacts_{worker_id}")
    if os.path.exists(artifacts_dir):
        shutil.rmtree(artifacts_dir)

    additional_compiler_args = compile.nkipy_compiler_args
    compile.compile_to_neff(
        traced_kernel,
        artifacts_dir,
        additional_compiler_args=additional_compiler_args,
    )


def on_device_test(kernel_fn, trace_mode, *args, artifacts_dir=None, **kwargs):
    """
    Unified baremetal execution function that traces and runs on device.

    Args:
        kernel_fn: The kernel function to execute
        trace_mode: "hlo" or other supported tracing mode
        *args: Input arrays for the kernel
        artifacts_dir: Directory for compilation artifacts (for parallel test isolation)
        **kwargs: Additional arguments

    Returns:
        Device execution output
    """
    # Auto-generate worker-specific artifacts_dir if not provided
    if artifacts_dir is None:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        artifacts_dir = os.path.join(
            tempfile.gettempdir(), f"nkipy_artifacts_{worker_id}"
        )

    if trace_mode == "hlo":
        traced_kernel = NKIPyKernel.trace(kernel_fn, backend="hlo")
    else:
        raise ValueError(f"Unknown trace mode: {trace_mode}")

    return baremetal_run_traced_kernel(
        traced_kernel, *args, artifacts_dir=artifacts_dir, **kwargs
    )
