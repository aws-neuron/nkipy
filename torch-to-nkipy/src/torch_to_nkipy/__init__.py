# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch-to-nkipy: IR building for NKIPy.

This package provides tools for converting PyTorch FX graphs to NKIPy IR.
For runtime, device management, and PyTorch integration, use `spiky.torch`.

Primary exports:
- NKIPyBuilder: Builds NKIPy IR from FX graphs
- NKIPyKernel: Manages kernel compilation and execution
- NKIPyAST, InputNode, OutputNode, ComputationNode: AST types
"""

# IR building exports (primary functionality of this package)
from torch_to_nkipy.nkipy_builder import (
    NKIPyBuilder,
    NKIPyKernel,
    NKIPyAST,
    InputNode,
    OutputNode,
    ComputationNode,
)
from torch_to_nkipy.utils import NKIOpRegistry, mark_subgraph_identity
from torch_to_nkipy.utils.graph import (
    gm_split_and_wrap,
    load_func_from_file,
    save_string_to_file,
)


__all__ = [
    # IR Building (primary exports)
    "NKIPyBuilder",
    "NKIPyKernel",
    "NKIPyAST",
    "InputNode",
    "OutputNode",
    "ComputationNode",
    # Utilities
    "mark_subgraph_identity",
    "NKIOpRegistry",
    "gm_split_and_wrap",
    "load_func_from_file",
    "save_string_to_file",
]
