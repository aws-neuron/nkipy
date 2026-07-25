"""Runtime hooks for NKI integration side effects."""

from __future__ import annotations

import importlib
from functools import cache


@cache
def ensure_nki_bridge() -> None:
    """Install nkipy's GenericKernel bridge exactly once per process."""
    importlib.import_module("nkipy.core.nki_op")
