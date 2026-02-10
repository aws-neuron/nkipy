# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for spiky.

Importing this module registers the 'nkipy' backend with torch.compile.
"""

from spiky.torch.config import (
    NKIPyBackendConfig,
    get_nkipy_backend_config,
    reset_nkipy_backend_config,
    set_nkipy_backend_config,
)

# Import backend module to register 'nkipy' backend with torch.compile
# This import triggers register_backend() call
from spiky.torch.backend import (
    init_nkipy_backend,
    is_nkipy_backend_initialized,
    reset_nkipy_backend,
)

# Device utilities - lazy import to avoid circular deps
def current_device() -> int:
    """Return the current Neuron device index."""
    from spiky.device.module import current_device as _current_device
    return _current_device()


def set_device(device: int) -> None:
    """Set the current Neuron device."""
    from spiky.device.module import set_device as _set_device
    _set_device(device)


def device_count() -> int:
    """Return the number of Neuron devices available."""
    from spiky.device.module import device_count as _device_count
    return _device_count()


__all__ = [
    # Backend initialization
    "init_nkipy_backend",
    "is_nkipy_backend_initialized",
    "reset_nkipy_backend",
    # Configuration
    "NKIPyBackendConfig",
    "get_nkipy_backend_config",
    "set_nkipy_backend_config",
    "reset_nkipy_backend_config",
    # Device utilities
    "current_device",
    "set_device",
    "device_count",
]
