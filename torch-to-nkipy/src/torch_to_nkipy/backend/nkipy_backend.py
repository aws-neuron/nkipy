# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPy backend implementation for PyTorch Dynamo."""

import builtins
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
import torch._dynamo
import torch._functorch._aot_autograd.runtime_wrappers as runtime_wrappers
import torch.distributed as dist
import torch.fx as fx
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.registry import register_backend
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.utils import InputType

from torch_to_nkipy.backend.nkipy_backend_config import (
    NKIPyBackendConfig,
    get_nkipy_backend_config,
    reset_nkipy_backend_config,
    set_nkipy_backend_config,
)
from torch_to_nkipy.device import nkipy_close, nkipy_init
from torch_to_nkipy.nkipy_builder.nkipy_kernel import NKIPyKernel
from torch_to_nkipy.utils.graph import _count_subgraph_markers as count_subgraph_markers
from torch_to_nkipy.utils.graph import gm_split_and_wrap

# FIXME replace the AliasOfInputHandler in aot_module_simplified
runtime_wrappers.AliasOfInputHandler.__call__ = (
    runtime_wrappers.NoopAliasHandler.__call__
)

logger = logging.getLogger(__name__)


def _get_rank(explicit_rank: Optional[int] = None) -> int:
    """Get rank with fallback chain.

    Priority:
    1. Explicit parameter - if rank is passed, use it
    2. torch.distributed - if process group initialized, use dist.get_rank()
    3. LOCAL_RANK env - for torchrun/other launchers
    4. RANK env - fallback for some launchers
    5. Default 0 - single process case
    """
    if explicit_rank is not None:
        return explicit_rank

    # Try torch.distributed first
    if dist.is_initialized():
        return dist.get_rank()

    # Try environment variables
    for env_var in ["LOCAL_RANK", "RANK"]:
        if env_var in os.environ:
            return int(os.environ[env_var])

    return 0  # Default for single process


def _get_world_size(explicit_world_size: Optional[int] = None) -> int:
    """Get world size with fallback chain.

    Priority:
    1. Explicit parameter - if world_size is passed, use it
    2. torch.distributed - if process group initialized, use dist.get_world_size()
    3. WORLD_SIZE env - for torchrun/other launchers
    4. Default 1 - single process case
    """
    if explicit_world_size is not None:
        return explicit_world_size

    if dist.is_initialized():
        return dist.get_world_size()

    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    return 1  # Default for single process


def is_nkipy_backend_initialized() -> bool:
    """Check if NKIPy backend is initialized."""
    return get_nkipy_backend_config() is not None


def init_nkipy_backend(
    nkipy_cache: str = "./nkipy_cache",
    log_level: int = logging.INFO,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    core_offset: int = 0,
    additional_compiler_args: str = "",
    use_spiky: Optional[bool] = None,
) -> None:
    """Initialize the NKIPy backend.

    Args:
        nkipy_cache: Directory path for cache storage
        log_level: Logging level
        rank: Process rank. Auto-detected if None:
              1. From torch.distributed if initialized
              2. From LOCAL_RANK environment variable
              3. From RANK environment variable
              4. Defaults to 0
        world_size: Total processes. Auto-detected if None:
              1. From torch.distributed if initialized
              2. From WORLD_SIZE environment variable
              3. Defaults to 1
        core_offset: Offset for visible_cores assignment (default 0).
              visible_cores = [rank + core_offset]
              Useful for multi-node setups where cores start at different indices.
        additional_compiler_args: Custom compiler flags
        use_spiky: Use experimental spiky runtime instead of spike.
              If None, checks NKIPY_USE_SPIKY environment variable.
              Defaults to False (use spike).

    Raises:
        RuntimeError: If backend is already initialized
    """
    if is_nkipy_backend_initialized():
        raise RuntimeError("NKIPy backend has already been initialized.")

    # Resolve use_spiky: explicit arg > env var > default False
    if use_spiky is None:
        use_spiky = os.environ.get("NKIPY_USE_SPIKY", "0").lower() in (
            "1",
            "true",
            "yes",
        )

    # Configure runtime backend type BEFORE any initialization
    from torch_to_nkipy.device.runtime_backend import set_runtime_type

    set_runtime_type(use_spiky)

    # Auto-detect rank and world_size
    rank = _get_rank(rank)
    world_size = _get_world_size(world_size)

    # FIXME Currently each process (TP rank) should have its own nkipy_cache
    nkipy_cache = str(Path(nkipy_cache).resolve())
    os.makedirs(nkipy_cache, exist_ok=True)

    logging.basicConfig(level=log_level)

    # Calculate visible core with offset
    visible_core = rank + core_offset
    nkipy_init(visible_core)

    nkipy_backend_config = NKIPyBackendConfig(
        nkipy_cache_prefix=nkipy_cache,
        log_level=log_level,
        rank=rank,
        world_size=world_size,
        additional_compiler_args=additional_compiler_args,
        use_spiky=use_spiky,
    )
    set_nkipy_backend_config(nkipy_backend_config)

    logger.debug(f"NKIPy backend initialized with config: {nkipy_backend_config}")


def reset_nkipy_backend():
    reset_nkipy_backend_config()
    nkipy_close()


class CompiledWrapper(torch.nn.Module):
    def __init__(
        self,
        gm: fx.GraphModule,
        options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    ):
        super().__init__()
        self._gm = gm
        self._handle = None
        self._options = options
        config = get_nkipy_backend_config()
        self._cache_dir = config.nkipy_cache
        self._additional_compiler_args = config.additional_compiler_args
        self._rank = config.rank
        self._world_size = config.world_size

    def forward(self, *args, **kwargs):
        if self._handle is None:
            self._handle = NKIPyKernel(
                self._gm, args, self._options,
                cache_dir=self._cache_dir,
                additional_compiler_args=self._additional_compiler_args,
                rank=self._rank,
                world_size=self._world_size,
            )
        return self._handle(*args, **kwargs)


def nkipy_backend_fn_decomposed(
    gm: fx.GraphModule,
    example_inputs: Sequence[InputType],
    options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
) -> Callable:
    """Decompose the graph for NKIPy backend.

    Args:
        gm: The FX graph module to decompose
        example_inputs: Example inputs for the graph
        options: Backend options

    Returns:
        Callable: The decomposed graph
    """
    # Default path (spike backend or static shapes)
    if count_subgraph_markers(gm) == 0:
        compiled_fn = CompiledWrapper(gm, options)
    else:
        compiled_fn = gm_split_and_wrap(gm, CompiledWrapper, options)

    return make_boxed_func(compiled_fn)


def nkipy_backend_fn(
    gm: fx.GraphModule,
    example_inputs: Sequence[InputType],
    options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
) -> Callable:
    """Main backend function for NKIPy.

    Args:
        graph: The FX graph module to compile
        example_inputs: Example inputs for the graph

    Returns:
        Callable: The compiled function
    """
    # FIXME Kernel-centric Execution is not supported yet
    # FIXME We are still using torch's aot_module_simplified

    def fw_compiler_with_options(gm, example_inputs):
        return nkipy_backend_fn_decomposed(gm, example_inputs, options=options)

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw_compiler_with_options,
        decompositions=core_aten_decompositions(),
        keep_inference_input_mutations=True,
    )


register_backend(name="nkipy", compiler_fn=nkipy_backend_fn)
