# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPy builder for generating NKIPy functions from FX graphs.

Note: Heavy imports are deferred to avoid circular imports.
"""

from torch_to_nkipy.nkipy_builder.nkipy_ast import (
    ComputationNode,
    InputNode,
    NKIPyAST,
    OutputNode,
)


def __getattr__(name):
    """Lazy import for classes with heavy dependencies."""
    if name == "NKIPyBuilder":
        from torch_to_nkipy.nkipy_builder.nkipy_builder import NKIPyBuilder
        return NKIPyBuilder
    if name == "NKIPyKernel":
        from torch_to_nkipy.nkipy_builder.nkipy_kernel import NKIPyKernel
        return NKIPyKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NKIPyBuilder",
    "NKIPyKernel",
    "NKIPyAST",
    "InputNode",
    "OutputNode",
    "ComputationNode",
]
