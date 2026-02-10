# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime module for compiling and executing models on AWS Neuron hardware.

This module implements the compilation pipeline and execution framework that:
1. Converts PyTorch functions to Neuron-compatible representations
2. Compiles them to Neuron executables (NEFF files)
3. Manages caching of compiled models for efficient reuse
4. Handles tensor allocation and data transfer between CPU and Neuron devices
5. Executes the compiled models with proper memory management

The workflow typically follows: trace -> compile -> load -> execute -> retrieve results.
"""

import atexit
import contextlib
import contextvars
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch

from nkipy.core.compile import compile_to_neff, trace
from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config
from torch_to_nkipy.device import load_spike_model, nkipy_profile, spike_execute
from torch_to_nkipy.device.runtime_backend import LoadedModel
from torch_to_nkipy.utils.dtype import meta_tensor_to_numpy, numpy_to_torch_dtype
from torch_to_nkipy.utils.graph import load_func_from_file
from torch_to_nkipy.utils.name import (
    ARG_SHAPE_DTYPE_FILE,
    COMPILER_DIR,
    IO_SPECS_FILE,
    NEFF_FILE,
    NKIPY_FUNC_FILE,
    NKIPY_FUNC_NAME,
)
from torch_to_nkipy.utils.ntff_meta import NtffMeta
from torch_to_nkipy.utils.target import get_platform_target

logger = logging.getLogger(__name__)


@dataclass
class TensorSpec:
    """
    Metadata specification for a tensor without storing actual data.

    This class captures the essential properties needed to allocate and
    manage tensors during model execution, including name (for NEFF binding),
    shape, and data type.
    """

    name: str
    shape: tuple  # Tuple of dimensions (d1, d2, ..., dn)
    dtype: torch.dtype  # PyTorch data type


@dataclass
class IOSpecs:
    """
    Container for input and output tensor specifications of a compiled model.

    This class maintains the structural information about a model's expected
    inputs and outputs, ensuring proper tensor allocation and binding during
    execution without requiring recompilation or inference of shapes.
    """

    input_specs: List[TensorSpec]  # Specifications for all input tensors
    output_specs: List[TensorSpec]  # Specifications for all output tensors


@dataclass
class NeuronExecutable:
    """
    A compiled and loaded Neuron model ready for execution.

    This class represents the runtime state of a compiled model, linking
    the loaded Neuron Runtime (NRT) model with its input/output specifications.
    It contains everything needed to execute the model and interpret its results.
    """

    nkipy_model: LoadedModel  # Reference to the loaded model
    io_specs: IOSpecs  # Input/output specifications for tensor allocation and binding
    neff_path: str


# Global model cache indexed by kernel hash to avoid redundant compilation/loading
loaded_models: Dict[str, NeuronExecutable] = {}

# Another global cache indexed by kernel hash to guarantee kernels with the same
# hash are only compiled once across multiple arnks
hashes_to_kernel_dirs: Dict[str, Path] = {}


def _cleanup_loaded_models():
    """Clean up loaded models at exit to prevent nanobind leak warnings."""
    loaded_models.clear()
    hashes_to_kernel_dirs.clear()


atexit.register(_cleanup_loaded_models)


@contextlib.contextmanager
def in_directory(target_dir: Path):
    """Context manager for temporarily changing the working directory."""
    original_dir = Path.cwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)


def get_compile_dir_and_neff_path(kernel_dir: Path):
    kernel_compile_dir = kernel_dir / COMPILER_DIR
    neff_path = kernel_compile_dir / NEFF_FILE
    return kernel_compile_dir, neff_path


def get_kernel_hash_from_path(kernel_dir: Union[str, Path]):
    return str(kernel_dir)[-8:]


def compile_load_execute(
    nkipy_func: Callable,
    kernel_hash: str,
    args: Tuple[torch.Tensor, ...],
    alias_map: Dict[int, int],
    none_idx_list: List[int],
    kernel_dir: Path,
    ntff_meta: NtffMeta,
    rank: int = 0,
    world_size: int = 1,
    additional_compiler_args: str = "",
) -> List[torch.Tensor]:
    """
    End-to-end function to compile, load, and execute a model on Neuron hardware.

    This is the main entry point for Neuron execution, handling the entire pipeline
    from a Python function to executed results. It leverages caching to avoid
    redundant compilation for previously seen functions.

    Args:
        nkipy_func: The Python function to compile for Neuron execution
        kernel_hash: Unique identifier for the compiled kernel (used for caching)
        args: Input tensors to use for execution
        alias_map: Maps output indices to input indices for in-place operations
                  (key=output_idx, value=input_idx)
        kernel_dir: Base directory for storing compilation artifacts

    Returns:
        List of output tensors containing the results of model execution
    """
    # First phase: Ensure the model is compiled and loaded (with profiling)
    with torch.profiler.record_function("nkipy_load_model"):
        neuron_executable = load_model(
            nkipy_func=nkipy_func,
            kernel_hash=kernel_hash,
            args=args,
            kernel_dir=kernel_dir,
            rank=rank,
            world_size=world_size,
            additional_compiler_args=additional_compiler_args,
        )

    # Second phase: Execute the loaded model with provided inputs (with profiling)
    with torch.profiler.record_function(f"run_neff_model {kernel_hash}"):
        output_tensors = run_neff_model(
            nkipy_model=neuron_executable.nkipy_model,
            io_specs=neuron_executable.io_specs,
            neff_path=neuron_executable.neff_path,
            alias_map=alias_map,
            none_idx_list=none_idx_list,
            args=args,
            ntff_meta=ntff_meta,
            rank=rank,
        )

    return output_tensors


def load_model(
    nkipy_func: Callable,
    kernel_hash: str,
    args: Tuple[torch.Tensor, ...],
    kernel_dir: Path,
    rank: int = 0,
    world_size: int = 1,
    additional_compiler_args: str = "",
) -> NeuronExecutable:
    """
    Load a compiled Neuron model, compiling it first if not already cached.

    This function implements a simple caching strategy to avoid recompiling
    models that have already been processed. It uses kernel_hash as the cache key.

    Args:
        nkipy_func: The function to compile if needed
        kernel_hash: Unique identifier for the kernel (cache key)
        args: Input tensors for shape/type specialization if compilation is needed
        kernel_dir: Directory for storing compiled artifacts

    Returns:
        NeuronExecutable containing the loaded model and its I/O specifications
    """
    # Check if the model is already in the cache
    if kernel_hash not in loaded_models:
        # Not cached - compile the model and get its specs
        neff_path, io_specs = compile_model(
            nkipy_func=nkipy_func,
            args=args,
            kernel_dir=kernel_dir,
            additional_compiler_args=additional_compiler_args,
        )

        # Load the compiled model into the Neuron runtime
        logger.info(f"Rank {rank}: loading from {neff_path}...")
        nkipy_model = load_spike_model(
            neff_file=str(neff_path),
            # Enable collective communication for distributed execution
            cc_enabled=world_size > 1,
            device_id=rank,
            device_count=world_size,
        )

        # Create and cache the executable
        neuron_executable = NeuronExecutable(nkipy_model, io_specs, neff_path)
        loaded_models[kernel_hash] = neuron_executable

    return loaded_models[kernel_hash]


def compile_model(
    nkipy_func: Callable,
    args: Tuple[torch.Tensor, ...],
    kernel_dir: Path,
    numpy_args: Tuple[np.ndarray] = None,
    use_numpy_args: bool = False,
    additional_compiler_args: str = "",
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
    kernel_compile_dir, neff_path = get_compile_dir_and_neff_path(kernel_dir)
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
    neff_path = compile_to_neff(
        trace_kernel=traced_kernel,
        target=get_platform_target(),
        output_dir=kernel_compile_dir,
        save_artifacts=True,  # Save additional artifacts for debugging
        additional_compiler_args=additional_compiler_args,
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


def parallel_compile_model(
    nkipy_cache_dir: Union[Path, List[Path]], num_workers: int, is_master: bool
):
    # Reset the compiled kernel set
    hashes_to_kernel_dirs.clear()
    # Get all kernels
    if not isinstance(nkipy_cache_dir, list):
        nkipy_cache_dir = [nkipy_cache_dir]

    for cache_dir in nkipy_cache_dir:
        kernels = [p for p in os.listdir(cache_dir) if p.startswith("kernel_")]
        hashes = [get_kernel_hash_from_path(p) for p in kernels]
        for h, kernel_path in zip(hashes, kernels):
            if h not in hashes_to_kernel_dirs:
                hashes_to_kernel_dirs[h] = cache_dir / kernel_path

    # Use only unique kernels (one per hash)
    unique_kernels = [p for p in hashes_to_kernel_dirs.values()]

    # Only the master rank performs the compilation. Others can finish after
    # retrieving the unique kernels.
    if not is_master:
        return

    # Get parallel workers
    assert num_workers > 0, (
        f"Must have a valid number of parallel workers, getting {num_workers}!"
    )

    # Compile
    logger.info(
        f"Parallel compiling {len(unique_kernels)} unique kernels with {num_workers} workers..."  # noqa
    )
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for kernel_dir in unique_kernels:
            logger.info(f"Compiling {kernel_dir}...")
            futures.append(executor.submit(compile_model_wrapped, kernel_dir))

        all_results = [future.result() for future in futures]  # noqa


_in_parallel_compile_context = contextvars.ContextVar(
    "in_parallel_compile_context", default=False
)


@contextlib.contextmanager
def parallel_compile_context(num_workers: int = 1):
    """
    Context manager for parallel model compilation. User is expected to do a
    dummy-run of the model inside this context to trigger compilation of all
    NEFFs in parallel. A real execution is needed later to get actual output.

    Example usage:

    # Parallel compile
    with parallel_compile_context(num_workers=2):
        dummy_output = model(*args, **kwargs)

    # Actual execution
    real_output = model(*args, **kwargs)

    Arguments:
    - num_workers: Number of workers used for parallel compilation. Default is 1.
    """
    token = _in_parallel_compile_context.set(True)
    try:
        yield
    finally:
        is_dist_initialized = torch.distributed.is_initialized()
        rank = 0
        world_size = 1
        if is_dist_initialized:
            torch.distributed.barrier()
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            logger.warning(
                """Torch distributed not initialized, assuming """
                """single-process execution. Spawning parallel """
                """workers from every process..."""
            )
        nkipy_cache_prefix = get_nkipy_backend_config().nkipy_cache_prefix
        all_cache_dirs = [
            Path(f"{nkipy_cache_prefix}/rank_{r}") for r in range(world_size)
        ]
        parallel_compile_model(all_cache_dirs, num_workers, rank == 0)
        if is_dist_initialized:
            torch.distributed.barrier()
        _in_parallel_compile_context.reset(token)


def in_parallel_compile_context():
    """Check if currently in parallel_compile_context."""
    return _in_parallel_compile_context.get()


def run_neff_model(
    nkipy_model: LoadedModel,
    args: Tuple[torch.Tensor, ...],
    alias_map: Dict[int, int],
    none_idx_list: List[int],
    io_specs: IOSpecs,
    neff_path: str,
    ntff_meta: NtffMeta,
    rank: int = 0,
) -> List[torch.Tensor]:
    """
    Execute a compiled NEFF model on Neuron hardware with the given inputs.

    This function handles the complex task of:
    1. Mapping input tensors to their Neuron runtime counterparts
    2. Allocating memory for output tensors or setting up in-place operations
    3. Binding inputs and outputs to the model by name
    4. Triggering execution on the Neuron device
    5. Cleaning up temporary resources

    Args:
        nkipy_model: Reference to the loaded model (LoadedModel)
        args: Input tensors for model execution
        alias_map: Mapping from output indices to input indices for in-place operations
        io_specs: Input/output specifications that define tensor shapes and types
        neff_path: Path to the NEFF file (for profiling)
        ntff_meta: Metadata for NTFF profiling

    Returns:
        List of output tensors containing the results of execution
    """
    neff_input_tensors = args
    neff_output_tensors = []
    dynamo_output_tensors = []

    # Set up output tensors - either new allocations or aliases to inputs
    for idx, output_spec in enumerate(io_specs.output_specs):
        if idx in alias_map:
            # For in-place operations: create an alias to the corresponding input tensor
            input_idx = alias_map[idx]
            output_tensor = neff_input_tensors[input_idx]
        else:
            # For new outputs: allocate new meta tensor and corresponding Neuron memory
            output_tensor = torch.empty(
                size=output_spec.shape, dtype=output_spec.dtype, device="nkipy"
            )
            dynamo_output_tensors.append(output_tensor)

        # Add to the list of all output tensors for binding
        neff_output_tensors.append(output_tensor)

    # Steps 4-5: Prepare bindings and execute model
    neff_input_names = [input_spec.name for input_spec in io_specs.input_specs]
    neff_output_names = [output_spec.name for output_spec in io_specs.output_specs]

    neff_input_dict = dict(zip(neff_input_names, neff_input_tensors))
    neff_output_dict = dict(zip(neff_output_names, neff_output_tensors))

    with nkipy_profile(ntff_meta, neff_path, rank=rank) as (save_trace, ntff_name):
        # Execute the model on Neuron hardware
        spike_execute(
            model=nkipy_model,
            inputs=neff_input_dict,
            outputs=neff_output_dict,
            save_trace=save_trace,
            ntff_name=ntff_name,
        )

    for idx in none_idx_list:
        dynamo_output_tensors.insert(idx, None)

    # Return only the newly allocated output tensors
    return dynamo_output_tensors


# Backward compatibility alias
execute_model = run_neff_model
