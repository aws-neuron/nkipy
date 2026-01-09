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
from nkipy.runtime.execute import baremetal_run_traced_kernel, simulate_traced_kernel

# Constants for tolerance levels
SIMULATE_RTOL = 1e-4
SIMULATE_ATOL = 1e-4
BAREMETAL_RTOL = 1e-2
BAREMETAL_ATOL = 1e-2

# Pre-configured assert functions
simulate_assert_allclose = partial(
    np.testing.assert_allclose, rtol=SIMULATE_RTOL, atol=SIMULATE_ATOL
)
baremetal_assert_allclose = partial(
    np.testing.assert_allclose, rtol=BAREMETAL_RTOL, atol=BAREMETAL_ATOL
)

# Hardware availability check
NEURON_AVAILABLE = is_neuron_compatible()


# Simulation mode fixture - tests will run with both IR and HLO simulation
@pytest.fixture(params=["hlo"])
def sim_mode(request):
    """Fixture to run tests with both IR and HLO simulation modes"""
    return request.param


def simulate_kernel_unified(kernel_fn, sim_mode, *args, **kwargs):
    """
    Unified simulation function that dispatches to IR or HLO simulation

    Args:
        kernel_fn: The kernel function to simulate
        sim_mode: "hlo" or other supported mode
        *args: Input arrays for the kernel
        **kwargs: Additional arguments

    Returns:
        Simulation output
    """
    if sim_mode == "hlo":
        traced_kernel = NKIPyKernel.trace(kernel_fn, backend="hlo")
    else:
        raise ValueError(f"Unknown simulation mode: {sim_mode}")

    return simulate_traced_kernel(traced_kernel, *args, **kwargs)


def baremetal_run_kernel_unified(
    kernel_fn, sim_mode, *args, artifacts_dir=None, **kwargs
):
    """
    Unified baremetal execution function that dispatches to IR or HLO simulation

    Args:
        kernel_fn: The kernel function to simulate
        sim_mode: "hlo" or other supported mode
        *args: Input arrays for the kernel
        artifacts_dir: Directory for compilation artifacts (for parallel test isolation)
        **kwargs: Additional arguments

    Returns:
        Simulation output
    """
    # Auto-generate worker-specific artifacts_dir if not provided
    if artifacts_dir is None:
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        artifacts_dir = os.path.join(
            tempfile.gettempdir(), f"nkipy_artifacts_{worker_id}"
        )

    if sim_mode == "hlo":
        traced_kernel = NKIPyKernel.trace(kernel_fn, backend="hlo")
    else:
        raise ValueError(f"Unknown simulation mode: {sim_mode}")

    return baremetal_run_traced_kernel(
        traced_kernel, *args, artifacts_dir=artifacts_dir, **kwargs
    )
