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
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Shared IR data types
# ---------------------------------------------------------------------------


@dataclass
class TensorPlaceholder:
    """Lightweight tensor metadata used by the execution pipeline.

    Attributes:
        name: Identifier used to key this tensor in input/output dicts at runtime.
        shape: Static shape of the tensor.
        dtype: NumPy dtype of the tensor elements.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class AliasInfo:
    """One input-output alias pair.

    Attributes:
        output_index: Position of this alias in the IR outputs list.
        param_index: Position of the aliased parameter in the IR inputs list.
        param_name: Name of the aliased input parameter.
        is_user_returned: True when the user's kernel explicitly returns this
            tensor.  False when the framework auto-appended it as an output
            solely to write back an in-place mutation.
    """

    output_index: int
    param_index: int
    param_name: str
    is_user_returned: bool


# ---------------------------------------------------------------------------
# IR Protocol — the interface that every backend IR must satisfy
# ---------------------------------------------------------------------------


@runtime_checkable
class ComputationIR(Protocol):
    """Protocol satisfied by both ``HLOModule`` and ``KernelGenIR``."""

    @property
    def inputs(self) -> List[TensorPlaceholder]: ...

    @property
    def outputs(self) -> List[TensorPlaceholder]: ...

    @property
    def aliases(self) -> List[AliasInfo]:
        """Input-output alias pairs for in-place mutations."""
        ...

    @property
    def auto_aliased_indices(self) -> set[int]:
        """Output indices implicitly appended for write-back, not user-returned."""
        ...

    def resolve_input_arrays(
        self, original_inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Map parameter names to backend-specific input names."""
        ...

    def get_alias_input_name(self, alias: AliasInfo) -> str:
        """Return the backend input name for an aliased parameter."""
        ...

    def content_hash(self, compiler_args: str) -> str:
        """Deterministic hash of IR content and compiler flags for caching."""
        ...


# ---------------------------------------------------------------------------
# Package-private active context — shared with submodules (e.g. hlo.py).
# ---------------------------------------------------------------------------

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
