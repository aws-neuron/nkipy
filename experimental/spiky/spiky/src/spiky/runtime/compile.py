# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model compilation for Neuron hardware."""

import contextlib
import os
import pickle
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

from nkipy.core.compile import compile_to_neff, trace
from spiky.runtime.cache import get_kernel_hash_from_path, hashes_to_kernel_dirs
from spiky.runtime.specs import IOSpecs, TensorSpec
from spiky.torch.config import get_nkipy_backend_config
from spiky.utils.dtype import meta_tensor_to_numpy, numpy_to_torch_dtype
from spiky.utils.names import (
    ARG_SHAPE_DTYPE_FILE,
    COMPILER_DIR,
    IO_SPECS_FILE,
    NEFF_FILE,
    NKIPY_FUNC_FILE,
    NKIPY_FUNC_NAME,
)
from spiky.utils.target import get_platform_target

# Import graph utilities from torch-to-nkipy for IR building
from torch_to_nkipy.utils.graph import load_func_from_file


@contextlib.contextmanager
def in_directory(target_dir: Path):
    """Context manager for temporarily changing the working directory."""
    original_dir = Path.cwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)


def compile_model(
    nkipy_func: Callable,
    args: Tuple[torch.Tensor, ...],
    kernel_dir: Path,
    numpy_args: Tuple[np.ndarray] = None,
    use_numpy_args: bool = False,
) -> Tuple[Path, IOSpecs]:
    """
    Compile a Python function to a Neuron executable (NEFF).

    This function handles the multi-step process of:
    1. Tracing the function to capture its execution graph
    2. Specializing the graph for specific input shapes and types
    3. Compiling the specialized graph to a Neuron executable
    4. Extracting and saving metadata about inputs and outputs

    Args:
        nkipy_func: The Python function to compile
        args: Input tensors used to determine shapes and types for specialization
        kernel_dir: Directory for storing compiled artifacts
        numpy_args: Optional, can directly specify numpy args and skip the torch
            to numpy conversion if needed.
        use_numpy_args: Set to True to use numpy_args.

    Returns:
        Tuple of (path to compiled NEFF file, I/O specifications)
    """
    # First check if there's a hash hit (i.e., kernel compiled by other ranks)
    kernel_hash = get_kernel_hash_from_path(kernel_dir)
    if kernel_hash in hashes_to_kernel_dirs:
        kernel_dir = hashes_to_kernel_dirs[kernel_hash]
    kernel_compile_dir = kernel_dir / COMPILER_DIR
    neff_path = kernel_compile_dir / NEFF_FILE
    io_specs_path = kernel_compile_dir / IO_SPECS_FILE

    # Check if compilation artifacts already exist
    if neff_path.exists() and io_specs_path.exists():
        # Load existing I/O specifications from cache
        with open(io_specs_path, "rb") as f:
            io_specs = pickle.load(f)
        return neff_path, io_specs

    # Ensure the output directory exists
    kernel_compile_dir.mkdir(parents=True, exist_ok=True)

    # Change to kernel_dir so that constant tensor .npy files can be found
    # by the numpy.load() calls in the nkipy function
    with in_directory(kernel_dir):
        # Step 1: Trace the function to capture its computation graph
        traced_kernel = trace(nkipy_func, backend="hlo")

        # Step 2: Convert meta tensors to numpy arrays for specialization
        if use_numpy_args:
            args_numpy = numpy_args
        else:
            args_numpy = [meta_tensor_to_numpy(arg) for arg in args]

        # Step 3: Specialize the traced kernel with concrete shapes and types
        traced_kernel.specialize(*args_numpy)

    # Step 4: Compile the specialized kernel to a Neuron executable (NEFF)
    nkipy_config = get_nkipy_backend_config()
    additional_args = nkipy_config.additional_compiler_args if nkipy_config else ""
    neff_path = compile_to_neff(
        trace_kernel=traced_kernel,
        target=get_platform_target(compiler_args=additional_args),
        output_dir=kernel_compile_dir,
        save_artifacts=True,  # Save additional artifacts for debugging
        additional_compiler_args=additional_args,
    )
    # Step 5: Extract and save I/O specifications for later use
    with open(io_specs_path, "wb") as f:
        # Convert NKIPY tensor specs to our TensorSpec format
        input_specs = [
            TensorSpec(name=t.name, shape=t.shape, dtype=numpy_to_torch_dtype(t.dtype))
            for t in traced_kernel._code.inputs
        ]
        output_specs = [
            TensorSpec(name=t.name, shape=t.shape, dtype=numpy_to_torch_dtype(t.dtype))
            for t in traced_kernel._code.outputs
        ]

        io_specs = IOSpecs(input_specs=input_specs, output_specs=output_specs)
        pickle.dump(io_specs, f)

    return Path(neff_path), io_specs


def compile_model_wrapped(kernel_dir: Path):
    """
    Wrapper around the compile_model function for parallel compilation.
    """
    kernel_dir = Path(kernel_dir)
    nkipy_file_path = kernel_dir / NKIPY_FUNC_FILE
    nkipy_func = load_func_from_file(nkipy_file_path, NKIPY_FUNC_NAME)
    arg_dtype_and_shape_file = kernel_dir / ARG_SHAPE_DTYPE_FILE
    with open(arg_dtype_and_shape_file, "rb") as f:
        arg_dtype_and_shape = pickle.load(f)
    numpy_args = [np.empty(t[0], t[1]) for t in arg_dtype_and_shape]
    neff_path, io_specs = compile_model(
        nkipy_func=nkipy_func,
        args=None,
        kernel_dir=kernel_dir,
        numpy_args=numpy_args,
        use_numpy_args=True,
    )
    return neff_path, io_specs
