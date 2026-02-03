# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for torch-to-nkipy."""

from torch_to_nkipy.utils.dtype import (
    convert_numpy_arrays_to_tensors,
    meta_tensor_to_numpy,
    numpy_to_tensor,
    numpy_to_torch_dtype,
    tensor_to_numpy,
    torch_to_numpy_dtype_str,
)
from torch_to_nkipy.utils.nki import NKIOpRegistry
from torch_to_nkipy.utils.ops import mark_subgraph_identity

__all__ = [
    # NKI
    "NKIOpRegistry",
    # Ops
    "mark_subgraph_identity",
    # Dtype conversions
    "numpy_to_torch_dtype",
    "torch_to_numpy_dtype_str",
    "meta_tensor_to_numpy",
    "tensor_to_numpy",
    "numpy_to_tensor",
    "convert_numpy_arrays_to_tensors",
]
