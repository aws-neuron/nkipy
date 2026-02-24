# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operation dispatcher for NKIPy.

This module provides the ``Op`` class which dispatches operations to the
appropriate backend implementation based on the current tracing state.
"""

from typing import Callable, Dict


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
        from nkipy.core.backend import get_backend

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
