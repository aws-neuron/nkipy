# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device initialization and lifecycle management."""

from torch_to_nkipy.device.runtime_backend import get_backend, reset_backend


def nkipy_init(visible_core: int):
    """Initialize Neuron Runtime for the given core.

    Args:
        visible_core: The core index to use (rank + core_offset from backend init)

    Sets up visible cores and root comm ID for collectives.
    """
    get_backend().init(visible_core)


def is_nkipy_device_initialized() -> bool:
    """Check if nkipy device is properly initialized."""
    return get_backend().is_initialized()


def nkipy_close():
    """Close Neuron Runtime and release resources."""
    get_backend().close()
    reset_backend()
