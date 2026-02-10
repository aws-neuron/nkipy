# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend for spiky."""

from typing import Optional

from spiky.backend.protocol import RuntimeBackend

_current_backend: Optional[RuntimeBackend] = None


def get_backend() -> RuntimeBackend:
    """Get the singleton runtime backend instance.

    Returns:
        RuntimeBackend instance (SpikyBackend)
    """
    global _current_backend

    if _current_backend is None:
        from spiky.backend.runtime import SpikyBackend

        _current_backend = SpikyBackend()

    return _current_backend


def reset_backend() -> None:
    """Reset the backend instance.

    Called during nkipy_close() to allow re-initialization.
    """
    global _current_backend
    _current_backend = None


__all__ = [
    "RuntimeBackend",
    "get_backend",
    "reset_backend",
]
