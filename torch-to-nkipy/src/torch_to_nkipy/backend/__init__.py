# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPy backend for PyTorch Dynamo.

Note: Heavy imports (nkipy_backend_fn, etc.) are deferred to avoid circular imports.
Use `from torch_to_nkipy.backend.nkipy_backend import ...` for those.
"""

from torch_to_nkipy.backend.nkipy_backend_config import (
    NKIPyBackendConfig,
    get_nkipy_backend_config,
    reset_nkipy_backend_config,
    set_nkipy_backend_config,
)


def __getattr__(name):
    """Lazy import for functions with heavy dependencies."""
    if name in ("init_nkipy_backend", "is_nkipy_backend_initialized",
                "reset_nkipy_backend", "nkipy_backend_fn"):
        from torch_to_nkipy.backend import nkipy_backend
        return getattr(nkipy_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Backend functions (lazy loaded)
    "init_nkipy_backend",
    "is_nkipy_backend_initialized",
    "reset_nkipy_backend",
    "nkipy_backend_fn",  # torch.compile backend callable for Dynamo
    # Config (eager loaded)
    "NKIPyBackendConfig",
    "get_nkipy_backend_config",
    "set_nkipy_backend_config",
    "reset_nkipy_backend_config",
]
