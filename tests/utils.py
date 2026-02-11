# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities for NKIPy tests"""

import os
import tempfile
from functools import partial

import numpy as np
import pytest
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


def trace_and_run(kernel_fn, trace_mode, *args, **kwargs):
    """
    Validate kernel is traceable to HLO, then run on CPU.

    Args:
        kernel_fn: The kernel function to test
        trace_mode: "hlo" or other supported tracing mode
        *args: Input arrays for the kernel
        **kwargs: Additional arguments

    Returns:
        CPU execution output
    """
    if trace_mode == "hlo":
        traced_kernel = NKIPyKernel.trace(kernel_fn, backend="hlo")
    else:
        raise ValueError(f"Unknown trace mode: {trace_mode}")

    # Validate HLO generation
    traced_kernel.specialize(*args, **kwargs)

    # Run on CPU (default backend)
    return kernel_fn(*args, **kwargs)


def baremetal_run_kernel_unified(
    kernel_fn, trace_mode, *args, artifacts_dir=None, **kwargs
):
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
