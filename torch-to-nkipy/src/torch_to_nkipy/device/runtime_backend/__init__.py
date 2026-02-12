# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend abstraction."""

from typing import Optional

from .protocol import LoadedModel, RuntimeBackend

_current_backend: Optional[RuntimeBackend] = None


def get_backend() -> RuntimeBackend:
    """Get the singleton runtime backend instance.

    Returns:
        RuntimeBackend instance (SpikeBackend)
    """
    global _current_backend

    if _current_backend is None:
        from .spike_backend import SpikeBackend

        _current_backend = SpikeBackend()

    return _current_backend


def reset_backend() -> None:
    """Reset the backend instance.

    Called during nkipy_close() to allow re-initialization.
    """
    global _current_backend
    _current_backend = None


__all__ = [
    "LoadedModel",
    "RuntimeBackend",
    "get_backend",
    "reset_backend",
]
