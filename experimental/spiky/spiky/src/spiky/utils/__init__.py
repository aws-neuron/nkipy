# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility modules for spiky dynamic shape handling and runtime support."""

# Runtime utilities (migrated from torch-to-nkipy)
from .dtype import (
    DTYPE_MAPPINGS,
    NUMPY_TO_TORCH_DTYPE_MAP,
    TORCH_TO_NUMPY_DTYPE_MAP,
    TORCH_TO_NUMPY_DTYPE_STR_MAP,
    convert_numpy_arrays_to_tensors,
    meta_tensor_to_numpy,
    numpy_to_tensor,
    numpy_to_torch_dtype,
    tensor_to_numpy,
    torch_to_numpy_dtype_str,
)
from .dynamic_shapes import (
    DynamicSpec,
    discover_dynamic_specs,
    infer_buckets,
    select_bucket,
)
from .names import (
    ALIAS_MAP_FILE,
    ARG_SHAPE_DTYPE_FILE,
    CC_PKG,
    COMPILER_DIR,
    IO_SPECS_FILE,
    NEFF_FILE,
    NKIPY_DEBUG_FUNC_FILE,
    NKIPY_DEBUG_FUNC_NAME,
    NKIPY_FUNC_FILE,
    NKIPY_FUNC_NAME,
    NONE_IDX_LIST,
    NUMPY_PKG,
)
from .ntff_meta import NtffMeta
from .padding import pad_inputs, unpad_outputs
from .target import (
    SUPPORTED_TYPES,
    CompilationTarget,
    get_platform_target,
)
from .tensor_metadata import (
    PaddingMetadata,
    attach_metadata,
    get_metadata,
    has_metadata,
)

__all__ = [
    # Dynamic shapes
    "DynamicSpec",
    "discover_dynamic_specs",
    "infer_buckets",
    "select_bucket",
    # Padding
    "pad_inputs",
    "unpad_outputs",
    # Tensor metadata
    "PaddingMetadata",
    "attach_metadata",
    "get_metadata",
    "has_metadata",
    # Dtype utilities
    "DTYPE_MAPPINGS",
    "NUMPY_TO_TORCH_DTYPE_MAP",
    "TORCH_TO_NUMPY_DTYPE_MAP",
    "TORCH_TO_NUMPY_DTYPE_STR_MAP",
    "numpy_to_torch_dtype",
    "torch_to_numpy_dtype_str",
    "meta_tensor_to_numpy",
    "tensor_to_numpy",
    "numpy_to_tensor",
    "convert_numpy_arrays_to_tensors",
    # Target utilities
    "CompilationTarget",
    "SUPPORTED_TYPES",
    "get_platform_target",
    # Name constants
    "NKIPY_FUNC_NAME",
    "NKIPY_FUNC_FILE",
    "NKIPY_DEBUG_FUNC_NAME",
    "NKIPY_DEBUG_FUNC_FILE",
    "ALIAS_MAP_FILE",
    "NONE_IDX_LIST",
    "ARG_SHAPE_DTYPE_FILE",
    "NUMPY_PKG",
    "CC_PKG",
    "COMPILER_DIR",
    "NEFF_FILE",
    "IO_SPECS_FILE",
    # Profiling metadata
    "NtffMeta",
]
