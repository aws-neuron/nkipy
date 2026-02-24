# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend module for NKIPy core functionality.

Public API:

- ``tracing(ctx)``           — context manager to activate/deactivate a trace context
- ``get_backend()``          — ``"cpu"`` or ``ctx.backend_name``
- ``set_source_location()``  — delegates to the active context
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple

# Package-private active context — shared with submodules (e.g. hlo.py).
_active_ctx = None


@contextmanager
def tracing(ctx):
    """Activate *ctx* as the current trace context.

    Args:
        ctx: A trace context object (e.g. ``HLOTraceContext``).
             Must expose a ``backend_name`` attribute.

    Yields:
        The active context.
    """
    global _active_ctx
    _active_ctx = ctx
    try:
        yield ctx
    finally:
        _active_ctx = None


def get_backend() -> str:
    """Return the name of the active backend.

    Returns ``ctx.backend_name`` when a trace context is active,
    ``"cpu"`` otherwise.
    """
    if _active_ctx is not None:
        return _active_ctx.backend_name
    return "cpu"


def set_source_location(location: Optional[Tuple[str, int]]) -> None:
    """Set the current source location in the trace context.

    Args:
        location: Tuple of (filename, line_number) or None to clear.
    """
    if _active_ctx is not None:
        _active_ctx.set_source_location(location)
