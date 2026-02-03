# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device initialization and lifecycle management."""

import os
import warnings

import spike_torch
from spike import get_spike_singleton

_nkipy_initialized = False


def nkipy_init(visible_core: int):
    """Initialize Neuron Runtime for the given core.

    Args:
        visible_core: The core index to use (rank + core_offset from backend init)

    Sets up visible cores and root comm ID for collectives.
    """
    global _nkipy_initialized

    # Set root comm ID for collectives
    if os.environ.get("NEURON_RT_ROOT_COMM_ID", None) is None:
        root_addr = os.environ.get("MASTER_ADDR", "localhost")
        root_port = os.environ.get("NEURON_RT_PORT", "61234")
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"

    # Reset and reconfigure spike with visible cores for this core.
    # spike_torch auto-initializes on import with default cores, so we need
    # to reset and reconfigure for the specific core.
    # Suppress the expected warning about invalidated objects since this is
    # initialization and no user objects exist yet.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="spike.reset\\(\\) called")
        spike_torch.reset()
    spike_torch.configure(visible_cores=[visible_core])

    # Re-initialize the spike singleton with the new configuration.
    # This is needed because spike_torch.reset() closed the previous runtime.
    get_spike_singleton()
    _nkipy_initialized = True


def is_nkipy_device_initialized() -> bool:
    """Check if nkipy device is properly initialized."""
    return _nkipy_initialized


def nkipy_close():
    """Close Neuron Runtime and release resources."""
    global _nkipy_initialized
    spike_torch.reset()
    _nkipy_initialized = False
