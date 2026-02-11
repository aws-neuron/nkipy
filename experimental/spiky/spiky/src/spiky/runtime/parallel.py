# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel compilation support for Neuron models."""

import contextlib
import contextvars
import logging
import os
from pathlib import Path
from typing import List, Union

import torch

from spiky.runtime.cache import get_kernel_hash_from_path, hashes_to_kernel_dirs

logger = logging.getLogger(__name__)

_in_parallel_compile_context = contextvars.ContextVar(
    "in_parallel_compile_context", default=False
)


def parallel_compile_model(
    nkipy_cache_dir: Union[Path, List[Path]], num_workers: int, is_master: bool
):
    # Reset the compiled kernel set
    hashes_to_kernel_dirs.clear()
    # Get all kernels
    if not isinstance(nkipy_cache_dir, list):
        nkipy_cache_dir = [nkipy_cache_dir]

    for cache_dir in nkipy_cache_dir:
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            continue
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

    from spiky.runtime.compile import compile_model_wrapped

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for kernel_dir in unique_kernels:
            logger.info(f"Compiling {kernel_dir}...")
            futures.append(executor.submit(compile_model_wrapped, kernel_dir))

        all_results = [future.result() for future in futures]  # noqa


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
        # Reset context var FIRST, before any operation that might throw,
        # so the ContextVar is always restored even if cleanup fails.
        _in_parallel_compile_context.reset(token)

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
        from spiky.torch.config import get_nkipy_backend_config

        nkipy_config = get_nkipy_backend_config()
        nkipy_cache_prefix = (
            nkipy_config.nkipy_cache_prefix if nkipy_config else "./nkipy_cache"
        )
        all_cache_dirs = [
            Path(f"{nkipy_cache_prefix}/rank_{r}") for r in range(world_size)
        ]
        parallel_compile_model(all_cache_dirs, num_workers, rank == 0)
        if is_dist_initialized:
            torch.distributed.barrier()


def in_parallel_compile_context():
    """Check if currently in parallel_compile_context."""
    return _in_parallel_compile_context.get()
