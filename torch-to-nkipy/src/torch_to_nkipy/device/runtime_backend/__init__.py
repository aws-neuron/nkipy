# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend abstraction for spike/spiky switching."""

from typing import Optional

from .protocol import LoadedModel, RuntimeBackend

_current_backend: Optional[RuntimeBackend] = None
_use_spiky: bool = False


def set_runtime_type(use_spiky: bool) -> None:
    """Set the runtime type. Must be called before get_backend().

    Args:
        use_spiky: If True, use spiky runtime; otherwise use spike (default)
    """
    global _use_spiky, _current_backend
    _use_spiky = use_spiky
    # Reset backend if runtime type changes
    _current_backend = None


def get_runtime_type() -> bool:
    """Get the configured runtime type.

    Returns:
        True if spiky is configured, False for spike
    """
    return _use_spiky


def get_backend() -> RuntimeBackend:
    """Get the singleton runtime backend instance.

    Returns:
        RuntimeBackend instance (SpikeBackend or SpikyBackend)
    """
    global _current_backend

    if _current_backend is None:
        if _use_spiky:
            raise RuntimeError(
                "Spiky runtime is not available in torch-to-nkipy. "
                "Use spiky package directly: from spiky.backend import get_backend"
            )
        else:
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
    "get_runtime_type",
    "set_runtime_type",
    "reset_backend",
]
