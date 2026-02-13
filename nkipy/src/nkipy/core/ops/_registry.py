# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple operation registry and dispatcher for NKIPy.

This module provides:
- Backend management (set_backend, get_backend)
- Context management (set_context, get_context)
- Operation dispatcher (Op class)

Backends:
- 'cpu': CPU eager execution backend (default, no context required)
- 'hlo': HLO tracing backend (requires HLOTraceContext)
- Future: 'mlir', etc.
"""

from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from nkipy.core.backend import TraceContext

# =============================================================================
# Global State
# =============================================================================

# Current backend name ('hlo', 'cpu', etc.)
_current_backend: Optional[str] = None

# Current trace context (may be None for eager backends like 'cpu')
_current_context: Optional["TraceContext"] = None


# =============================================================================
# Backend Management
# =============================================================================


def set_backend(
    backend: Optional[str], context: Optional["TraceContext"] = None
) -> None:
    """Set the current backend and optional trace context.

    Called by tracing infrastructure when a kernel is being traced.

    Args:
        backend: Backend name ('hlo', 'cpu', or future backends), or None to clear.
        context: Trace context for tracing backends (required for 'hlo').
    """
    global _current_backend, _current_context
    _current_backend = backend
    _current_context = context
    if backend == "hlo" and context is None:
        raise ValueError("Trace context is required for 'hlo' backend")


def get_backend() -> str:
    """Get the current backend name.

    Returns 'cpu' by default when no backend has been explicitly set.
    During tracing, the backend is set to 'hlo' by the tracing infrastructure.

    Returns:
        Current backend name.
    """
    if _current_backend is None:
        return "cpu"
    return _current_backend


# =============================================================================
# Context Management
# =============================================================================


def set_context(context: Optional["TraceContext"]) -> None:
    """Set the current trace context.

    This is typically called together with set_backend, but can be used
    separately if needed.

    Args:
        context: Trace context or None to clear.
    """
    global _current_context
    _current_context = context


def get_context() -> Optional["TraceContext"]:
    """Get the current trace context.

    Returns:
        Current trace context, or None for eager backends (like 'cpu').
    """
    return _current_context


def get_source_location() -> Optional[Tuple[str, int]]:
    """Get the current source location from the trace context.

    This is a convenience function that retrieves the source location
    from the current context if available.

    Returns:
        Tuple of (filename, line_number) or None if no context or location set.
    """
    if _current_context is not None:
        return _current_context.get_source_location()
    return None


def set_source_location(location: Optional[Tuple[str, int]]) -> None:
    """Set the current source location in the trace context.

    This is a convenience function that sets the source location
    in the current context if available.

    Args:
        location: Tuple of (filename, line_number) or None to clear.
    """
    if _current_context is not None:
        _current_context.set_source_location(location)


# =============================================================================
# Operation Dispatcher
# =============================================================================


class Op:
    """Simple operation dispatcher.

    Usage::

        zeros = Op('zeros')

        @zeros.impl('hlo')
        def _zeros_hlo(shape, dtype):
            # HLO implementation
            ...

        @zeros.impl('cpu')
        def _zeros_cpu(shape, dtype):
            # CPU implementation
            ...

        # Later, during tracing:
        result = zeros(shape, dtype)  # Dispatches based on current backend
    """

    def __init__(self, name: str):
        self.name = name
        self._impls: Dict[str, Callable] = {}

    def impl(self, backend: str) -> Callable:
        """Decorator to register a backend implementation.

        Args:
            backend: Backend name ('hlo', 'cpu', or other supported).

        Returns:
            Decorator function.
        """

        def decorator(fn: Callable) -> Callable:
            self._impls[backend] = fn
            return fn

        return decorator

    def __call__(self, *args, **kwargs):
        """Dispatch to the appropriate backend implementation."""
        backend = get_backend()
        if backend not in self._impls:
            available = ", ".join(self._impls.keys()) if self._impls else "none"
            raise NotImplementedError(
                f"Operation '{self.name}' not implemented for backend '{backend}'."
                f" Available backends: {available}"
            )
        return self._impls[backend](*args, **kwargs)

    def __repr__(self) -> str:
        backends = ", ".join(self._impls.keys()) if self._impls else "none"
        return f"Op('{self.name}', backends=[{backends}])"
