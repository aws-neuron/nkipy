# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime compilation and execution for spiky."""

from spiky.runtime.cache import (
    get_compile_dir_and_neff_path,
    get_kernel_hash_from_path,
    hashes_to_kernel_dirs,
)
from spiky.runtime.compile import compile_model, compile_model_wrapped
from spiky.runtime.parallel import (
    in_parallel_compile_context,
    parallel_compile_context,
    parallel_compile_model,
)
from spiky.runtime.specs import IOSpecs, TensorSpec

__all__ = [
    # Specs
    "TensorSpec",
    "IOSpecs",
    # Compilation
    "compile_model",
    "compile_model_wrapped",
    # Caching
    "hashes_to_kernel_dirs",
    "get_kernel_hash_from_path",
    "get_compile_dir_and_neff_path",
    # Parallel
    "parallel_compile_model",
    "parallel_compile_context",
    "in_parallel_compile_context",
]
