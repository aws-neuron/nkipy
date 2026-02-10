# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for spiky test suite.

All tests require real Neuron hardware. The hw_backend fixture is session-scoped
because the C++ runtime segfaults if you close and re-init the device within
a single process.
"""

import os
import logging

# Set NEURON_RT_ROOT_COMM_ID before any NRT initialization.
# The Neuron runtime reads this env var at init time to set up the
# global communicator for collective operations.
if os.environ.get("NEURON_RT_ROOT_COMM_ID") is None:
    _master = os.environ.get("MASTER_ADDR", "localhost")
    _port = os.environ.get("NEURON_RT_PORT", "61234")
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{_master}:{_port}"

import pytest
import torch


def has_neuron_hardware() -> bool:
    """Check if Neuron hardware is available."""
    try:
        import spiky
        return spiky.device_count() > 0
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast smoke tests (shape checks only)")
    config.addinivalue_line("markers", "requires_neuron: requires Neuron hardware")


requires_neuron = pytest.mark.skipif(
    not has_neuron_hardware(), reason="Requires Neuron hardware"
)


@pytest.fixture(scope="session")
def hw_backend(tmp_path_factory):
    """Session-scoped backend initialization on real hardware.

    Inits the device once for the entire pytest session. Session scope is
    required because the C++ runtime segfaults on close + re-init.

    Yields the cache directory path.
    """
    from spiky.torch.backend import init_nkipy_backend

    cache_dir = str(tmp_path_factory.mktemp("nkipy_cache"))

    init_nkipy_backend(
        nkipy_cache=cache_dir,
        log_level=logging.INFO,
        additional_compiler_args="-O1 --model-type=transformer --lnc=1",
    )

    yield cache_dir

    # Do NOT call reset_nkipy_backend / nkipy_close here.
    # The C++ runtime segfaults on close + re-init within a process.


@pytest.fixture
def ntff_output_dir(tmp_path):
    """Per-test temp directory for NTFF profiling output."""
    return tmp_path / "ntff_output"


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo before each test to avoid stale graph caches."""
    torch._dynamo.reset()
