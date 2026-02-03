# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for NKIPy - torch.compile backend for AWS Neuron."""

# Import device module for side effects (registers "nkipy" device)
from torch_to_nkipy import device  # noqa: F401

# Re-export from submodules
from torch_to_nkipy.backend import (
    get_nkipy_backend_config,
    init_nkipy_backend,
    is_nkipy_backend_initialized,
    reset_nkipy_backend,
)
from torch_to_nkipy.utils import NKIOpRegistry, mark_subgraph_identity

__all__ = [
    # Backend initialization
    "init_nkipy_backend",
    "is_nkipy_backend_initialized",
    "reset_nkipy_backend",
    "get_nkipy_backend_config",
    # Utilities
    "mark_subgraph_identity",
    "NKIOpRegistry",
]
