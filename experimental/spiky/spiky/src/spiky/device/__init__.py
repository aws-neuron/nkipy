# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device management for NKIPy on AWS Neuron hardware."""

# Re-export from submodules
from spiky.device.init import (
    is_nkipy_device_initialized,
    nkipy_close,
    nkipy_init,
)
from spiky.device.profiling import nkipy_profile

__all__ = [
    # Initialization
    "nkipy_init",
    "nkipy_close",
    "is_nkipy_device_initialized",
    # Profiling
    "nkipy_profile",
]


def _register_device():
    """Register nkipy device with PyTorch (called on init_nkipy_backend)."""
    from spiky.backend import get_backend
    get_backend().register_torch_device()
    # Import distributed for side effects (backend registration)
    from spiky.device import distributed  # noqa: F401
