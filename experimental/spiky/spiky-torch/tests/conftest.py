# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for spiky.torch tests."""

import gc
import os
import re

import pytest

# Set NEURON_RT_VISIBLE_CORES before importing spiky/spiky.torch
if "NEURON_RT_VISIBLE_CORES" not in os.environ:
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session", autouse=True)
def _spiky_runtime_session():
    import spiky
    import spiky.torch  # noqa: F401

    try:
        spiky.init(0)
    except RuntimeError as e:
        msg = str(e)
        if re.search(r"Failed to initialize NRT", msg):
            pytest.skip(f"NRT not available in this environment: {msg}")
        raise

    yield

    try:
        spiky.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    gc.collect()


@pytest.fixture
def nkipy_device():
    import torch

    return torch.device("nkipy")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
