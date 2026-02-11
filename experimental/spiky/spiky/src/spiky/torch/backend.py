# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPy backend implementation for PyTorch Dynamo."""

from __future__ import annotations

import builtins
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

if TYPE_CHECKING:
    from spiky.callable import NKIPyCallable

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

from spiky.device.init import nkipy_close, nkipy_init
from spiky.runtime.parallel import in_parallel_compile_context
from spiky.torch.config import (
    NKIPyBackendConfig,
    get_nkipy_backend_config,
    reset_nkipy_backend_config,
    set_nkipy_backend_config,
)

# Import from torch-to-nkipy for IR building
from torch_to_nkipy.nkipy_builder.nkipy_kernel import NKIPyKernel
from torch_to_nkipy.utils.graph import _count_subgraph_markers as count_subgraph_markers
from torch_to_nkipy.utils.graph import gm_split_and_wrap

logger = logging.getLogger(__name__)

_ALIAS_HANDLER_PATCHED = False

# Ops that produce incorrect results when inputs are zero-padded
_PAD_UNSAFE_OPS = frozenset({
    "softmax", "_softmax", "mean", "layer_norm", "group_norm",
    "batch_norm", "var", "std", "var_mean",
})


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
    pad_on_device: bool = True,
    keep_outputs_on_device: bool = False,
    pipelined: bool = True,
    input_layout: str = "auto",
    output_layout: str = "unpad",
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
        pad_on_device: Whether to pad inputs on device (default True)
        keep_outputs_on_device: Whether to keep outputs on device (default False)
        pipelined: Whether to use pipelined execution (default True)
        input_layout: Input layout strategy: "auto" or "padded" (default "auto")
        output_layout: Output layout strategy: "unpad" or "padded" (default "unpad")

    Raises:
        RuntimeError: If backend is already initialized
    """
    if is_nkipy_backend_initialized():
        raise RuntimeError("NKIPy backend has already been initialized.")

    # Patch AliasOfInputHandler once (scoped to init, not module import)
    global _ALIAS_HANDLER_PATCHED
    if not _ALIAS_HANDLER_PATCHED:
        runtime_wrappers.AliasOfInputHandler.__call__ = (
            runtime_wrappers.NoopAliasHandler.__call__
        )
        _ALIAS_HANDLER_PATCHED = True

    # Register torch device
    from spiky.device import _register_device

    _register_device()

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
        pad_on_device=pad_on_device,
        keep_outputs_on_device=keep_outputs_on_device,
        pipelined=pipelined,
        input_layout=input_layout,
        output_layout=output_layout,
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
        self._pipelined = (options or {}).get("pipelined", config.pipelined)
        self._keep_outputs_on_device = (options or {}).get(
            "keep_outputs_on_device", config.keep_outputs_on_device
        )
        self._callable: Optional["NKIPyCallable"] = None

    def forward(self, *args, **kwargs):
        if self._handle is None:
            self._handle = NKIPyKernel(
                self._gm,
                args,
                self._options,
                cache_dir=self._cache_dir,
                additional_compiler_args=self._additional_compiler_args,
                rank=self._rank,
                world_size=self._world_size,
            )
        kernel = self._handle
        if kernel.on_device:
            if in_parallel_compile_context():
                kernel._save_arg_shape_dtype(args)
                return kernel._generate_dummy_outputs(args[0].device)
            if self._callable is None:
                self._create_static_callable(kernel, args)
            return self._callable(*args)
        else:
            return kernel._execute_on_host(*args)

    def _create_static_callable(self, kernel, args):
        from spiky.callable import CallableConfig, NKIPyCallable
        from spiky.runtime.compile import compile_model

        _nkipy_func = kernel.nkipy_func
        _kernel_dir = kernel.kernel_paths["kernel_dir"]
        _alias_map = kernel.alias_map
        _none_idx_list = kernel.none_idx_list
        _args = args

        def static_compiler_fn(bucket_size: int):
            neff_path, io_specs = compile_model(
                nkipy_func=_nkipy_func,
                args=_args,
                kernel_dir=_kernel_dir,
            )
            return str(neff_path), _alias_map, _none_idx_list

        self._callable = NKIPyCallable(
            config=CallableConfig(
                cache_dir=Path(self._cache_dir),
                buckets=[1],
                dynamic_specs={},
                jit_enabled=False,
                pipelined=self._pipelined,
                pad_on_device=False,
                keep_outputs_on_device=self._keep_outputs_on_device,
                unpad_outputs=False,
                cc_enabled=self._world_size > 1,
                rank_id=self._rank,
                world_size=self._world_size,
                ntff_meta=kernel.ntff_meta,
            ),
            compiler_fn=static_compiler_fn,
        )


def _has_dynamic_dims(example_inputs: Sequence) -> bool:
    """Check if any input has symbolic/dynamic dimensions.

    Detects dynamic dimensions via:
    - torch.SymInt in tensor shapes
    - _dynamo_dynamic_indices attribute
    - _dynamo_weak_dynamic_indices attribute
    """
    for inp in example_inputs:
        if not hasattr(inp, "shape"):
            continue
        # Check for SymInt in shape
        for dim in inp.shape:
            if isinstance(dim, torch.SymInt):
                return True
        # Check for dynamo attributes
        if getattr(inp, "_dynamo_dynamic_indices", None):
            return True
        if getattr(inp, "_dynamo_weak_dynamic_indices", None):
            return True
    return False


def _create_spiky_callable(
    gm: fx.GraphModule,
    example_inputs: Sequence[InputType],
    options: Optional[dict] = None,
):
    """Create spiky-backed callable for dynamic shape execution.

    This function routes dynamic shape execution to spiky.callable.NKIPyCallable,
    which handles bucket selection, JIT compilation, and padded execution.

    Args:
        gm: FX GraphModule to compile
        example_inputs: Example inputs with dynamic dimensions marked
        options: Backend options (buckets, jit, etc.)

    Returns:
        NKIPyCallable instance, or None if routing should fall back to default path
    """
    try:
        from spiky.callable import CallableConfig, NKIPyCallable
        from spiky.utils.dynamic_shapes import discover_dynamic_specs, infer_buckets
    except ImportError:
        logger.warning("spiky not available, falling back to default path")
        return None

    config = get_nkipy_backend_config()
    if config is None:
        return None

    # Discover dynamic dimensions
    dynamic_specs = discover_dynamic_specs(gm, example_inputs)
    if not dynamic_specs:
        return None  # No dynamic dims, fall back to default path

    # Warn about ops that are not invariant to zero-padding
    unsafe_found = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_str = getattr(node.target, "__name__", str(node.target))
            op_name = target_str.rsplit(".", 1)[-1]
            if op_name in _PAD_UNSAFE_OPS:
                unsafe_found.append(op_name)
    if unsafe_found:
        logger.warning(
            "Dynamic shapes with zero-padding may produce incorrect results for: %s. "
            "Consider static shapes or masking for these operations.",
            unsafe_found,
        )

    # Get or infer bucket sizes
    buckets = options.get("buckets") if options else None
    if buckets is None:
        buckets = infer_buckets(
            dynamic_specs,
            min_size=32,
            max_size=getattr(config, "max_bucket_size", 2048),
            strategy=getattr(config, "bucket_strategy", "power_of_2"),
        )

    # Capture input metadata (concrete shapes, dtypes) before SymInts disappear.
    # SymInt entries get type "symint"; regular scalars preserve their value.
    input_metadata = []
    symint_indices = []
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, torch.Tensor):
            shape = []
            for dim_size in inp.shape:
                shape.append(int(dim_size))
            input_metadata.append(
                {
                    "shape": tuple(shape),
                    "dtype": inp.dtype,
                    "is_floating_point": inp.is_floating_point(),
                }
            )
        elif isinstance(inp, torch.SymInt):
            input_metadata.append({"type": "symint"})
            symint_indices.append(i)
        else:
            # Regular scalar (int, float, bool) — preserve concrete value
            input_metadata.append({"type": "scalar", "value": inp})

    # Build mapping from arg_idx to dynamic dim
    dynamic_arg_to_dim = {spec.arg_idx: spec.dim_idx for spec in dynamic_specs.values()}

    # Create compiler callback that uses make_fx + NKIPyKernel
    def compiler_fn(bucket_size: int):
        """Compile graph for a specific bucket size.

        Following the neuron_vm reference pattern:
        1. Create concrete inputs (replace SymInt dims with bucket_size)
        2. Re-trace with make_fx to get a graph with concrete shapes
        3. Remove SymInt placeholder nodes from the graph
        4. Pass only tensor inputs to NKIPyKernel
        5. Force NEFF compilation
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        from spiky.runtime.compile import compile_model

        # Create concrete inputs from metadata
        concrete_inputs = []
        symint_indices = []
        for i, meta in enumerate(input_metadata):
            is_symint = meta is None or (
                isinstance(meta, dict) and meta.get("type") == "symint"
            )
            if is_symint:
                # SymInt entry — provide bucket_size as concrete value
                concrete_inputs.append(bucket_size)
                symint_indices.append(i)
                continue

            if isinstance(meta, dict) and meta.get("type") == "scalar":
                # Regular scalar — preserve original value
                concrete_inputs.append(meta["value"])
                continue

            shape = list(meta["shape"])
            dtype = meta["dtype"]

            if i in dynamic_arg_to_dim:
                dim = dynamic_arg_to_dim[i]
                if len(shape) > dim:
                    shape[dim] = bucket_size

            if meta["is_floating_point"]:
                concrete_inputs.append(torch.randn(*shape, dtype=dtype))
            elif dtype == torch.bool:
                concrete_inputs.append(torch.zeros(*shape, dtype=torch.bool))
            else:
                concrete_inputs.append(torch.randint(0, 100, shape, dtype=dtype))

        # Re-trace with make_fx to get a graph with concrete shapes
        with torch.no_grad():
            concrete_gm = make_fx(gm, decomposition_table=core_aten_decompositions())(
                *concrete_inputs
            )

        # Remove SymInt placeholder nodes — NKIPyKernel only takes tensors
        graph = concrete_gm.graph
        placeholders_to_remove = []
        tensor_inputs = []
        input_idx = 0

        for node in list(graph.nodes):
            if node.op == "placeholder":
                if input_idx < len(concrete_inputs):
                    inp = concrete_inputs[input_idx]
                    if not isinstance(inp, torch.Tensor):
                        for user in list(node.users.keys()):
                            user.replace_input_with(node, inp)
                        placeholders_to_remove.append(node)
                    else:
                        tensor_inputs.append(inp)
                    input_idx += 1

        for node in placeholders_to_remove:
            graph.erase_node(node)

        if placeholders_to_remove:
            graph.lint()
            concrete_gm.recompile()

        # Build IR via NKIPyKernel (tensor inputs only)
        kernel = NKIPyKernel(
            concrete_gm,
            tensor_inputs,
            options,
            cache_dir=config.nkipy_cache,
            additional_compiler_args=config.additional_compiler_args,
            rank=config.rank,
            world_size=config.world_size,
        )

        # Force NEFF compilation
        neff_path, io_specs = compile_model(
            nkipy_func=kernel.nkipy_func,
            args=tensor_inputs,
            kernel_dir=kernel.kernel_paths["kernel_dir"],
        )
        return str(neff_path), kernel.alias_map, kernel.none_idx_list

    # Determine output layout and derive unpad_outputs consistently.
    # When output_layout is "padded", the engine must NOT unpad so that
    # the caller receives padded tensors with correct PaddingMetadata.
    output_layout = (
        options.get("output_layout", config.output_layout)
        if options
        else config.output_layout
    )
    if options and "unpad_outputs" in options:
        unpad_outputs = options["unpad_outputs"]
    else:
        unpad_outputs = output_layout != "padded"

    # Create callable config
    callable_config = CallableConfig(
        cache_dir=Path(config.nkipy_cache),
        buckets=buckets,
        dynamic_specs=dynamic_specs,
        symint_indices=symint_indices,
        jit_enabled=options.get("jit", True) if options else True,
        pipelined=options.get("pipelined", config.pipelined)
        if options
        else config.pipelined,
        unpad_outputs=unpad_outputs,
        pad_on_device=options.get("pad_on_device", config.pad_on_device)
        if options
        else config.pad_on_device,
        keep_outputs_on_device=options.get(
            "keep_outputs_on_device", config.keep_outputs_on_device
        )
        if options
        else config.keep_outputs_on_device,
        input_layout=options.get("input_layout", config.input_layout)
        if options
        else config.input_layout,
        output_layout=output_layout,
        cc_enabled=config.world_size > 1,
        rank_id=config.rank,
        world_size=config.world_size,
    )

    return NKIPyCallable(config=callable_config, compiler_fn=compiler_fn)


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
    config = get_nkipy_backend_config()

    # Route to spiky for dynamic shapes
    if config and _has_dynamic_dims(example_inputs):
        callable = _create_spiky_callable(gm, example_inputs, options)
        if callable is not None:
            return make_boxed_func(callable)

    # Default path (static shapes)
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
