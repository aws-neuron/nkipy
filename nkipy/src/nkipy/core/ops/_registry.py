# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operation dispatcher for NKIPy.

This module provides the ``Op`` class which dispatches operations to the
appropriate backend implementation based on the current tracing state.
"""

from typing import Callable, Dict, Optional


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

    Composed ops (built from other dispatched ops) use ``composed_impl``::

        floor_divide = Op('floor_divide')

        @floor_divide.composed_impl
        def _floor_divide(x, y):
            return floor(divide(x, y))

    The composed fallback is used when no backend-specific implementation
    is registered. Since it calls other ops via dispatch, it works on any
    backend that has the underlying primitives registered.
    """

    def __init__(self, name: str):
        self.name = name
        self._impls: Dict[str, Callable] = {}
        self._composed: Optional[Callable] = None

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

    @property
    def composed_impl(self):
        """Decorator to register a composed (backend-agnostic) fallback.

        A composed implementation is built entirely from calls to other
        dispatched ops, so it works on any backend that has those primitives.
        It is used as a fallback when no backend-specific impl is registered.
        """

        def decorator(fn: Callable) -> Callable:
            self._composed = fn
            return fn

        return decorator

    def __call__(self, *args, **kwargs):
        """Dispatch to the appropriate backend implementation."""
        from nkipy.core.backend import get_backend

        backend = get_backend()
        if backend in self._impls:
            return self._impls[backend](*args, **kwargs)
        if self._composed is not None:
            return self._composed(*args, **kwargs)
        available = ", ".join(self._impls.keys()) if self._impls else "none"
        raise NotImplementedError(
            f"Operation '{self.name}' not implemented for backend '{backend}'."
            f" Available backends: {available}"
        )

    def __repr__(self) -> str:
        backends = ", ".join(self._impls.keys()) if self._impls else "none"
        composed = ", composed" if self._composed else ""
        return f"Op('{self.name}', backends=[{backends}]{composed})"
