# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend module for NKIPy core functionality.

This module provides abstract base classes for trace contexts that can be
implemented by different backends (HLO, MLIR, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class TraceContext(ABC):
    """Abstract base class for trace contexts across backends.

    A trace context manages the state during kernel tracing, including
    source location tracking for debugging and profiling.

    Backends that support tracing (HLO, MLIR) should implement this interface.
    Eager backends (CPU) may not have a context (None).
    """

    @abstractmethod
    def set_source_location(self, location: Optional[Tuple[str, int]]) -> None:
        """Set the current source location for operations.

        Args:
            location: Tuple of (filename, line_number) or None to clear.
        """
        pass

    @abstractmethod
    def get_source_location(self) -> Optional[Tuple[str, int]]:
        """Get the current source location.

        Returns:
            Tuple of (filename, line_number) or None if not set.
        """
        pass
