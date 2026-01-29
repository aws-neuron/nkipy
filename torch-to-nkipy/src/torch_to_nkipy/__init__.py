# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch_to_nkipy import device  # noqa: F401
from torch_to_nkipy.backend.nkipy_backend import (
    init_nkipy_backend,
    is_nkipy_backend_initialized,
    reset_nkipy_backend,
)
from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config
from torch_to_nkipy.utils.nki import NKIOpRegistry
from torch_to_nkipy.utils.ops import mark_subgraph_identity

__all__ = [
    # Backend
    "init_nkipy_backend",
    "is_nkipy_backend_initialized",
    "reset_nkipy_backend",
    "get_nkipy_backend_config",
    "mark_subgraph_identity",
    "NKIOpRegistry",
]
