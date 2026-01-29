# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for spike-torch tests."""

import gc
import os

import pytest

# Set NEURON_RT_VISIBLE_CORES before importing spike_torch
if "NEURON_RT_VISIBLE_CORES" not in os.environ:
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"


def pytest_configure(config):
    """Configure pytest markers and initialize NRT."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

    # Import spike_torch to trigger NRT initialization
    import spike_torch  # noqa: F401


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test to free memory."""
    yield
    gc.collect()


@pytest.fixture
def nkipy_device():
    """Get a nkipy device for testing."""
    import torch

    return torch.device("nkipy")


def pytest_collection_modifyitems(config, items):
    """Mark performance tests as slow."""
    for item in items:
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
