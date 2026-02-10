# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.fx as fx
from torch._inductor.utils import InputType

from torch_to_nkipy.nkipy_builder.nkipy_builder import NKIPyBuilder
from torch_to_nkipy.runtime.runtime import (
    compile_load_execute,
    in_parallel_compile_context,
)
from torch_to_nkipy.utils.dtype import (
    convert_numpy_arrays_to_tensors,
    meta_tensor_to_numpy,
    tensor_to_numpy,
)
from torch_to_nkipy.utils.graph import (
    get_dtype_from_fx_node,
    get_shape_from_fx_node,
    hash_gm_with_tensors,
    load_func_from_file,
    save_gm_to_file,
)
from torch_to_nkipy.utils.name import (
    ALIAS_MAP_FILE,
    ARG_SHAPE_DTYPE_FILE,
    NKIPY_DEBUG_FUNC_FILE,
    NKIPY_FUNC_FILE,
    NKIPY_FUNC_NAME,
    NONE_IDX_LIST,
)
from torch_to_nkipy.utils.nki import NKIOpRegistry
from torch_to_nkipy.utils.ntff_meta import NtffMeta

logger = logging.getLogger(__name__)


class NKIPyKernel:
    """
    Represents a compiled kernel for Neuron execution.

    This class handles kernel compilation, caching, and execution.
    It manages the creation, loading, and execution of NKIPy functions
    that implement the computation defined by the provided FX graph.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        example_inputs: Sequence[InputType],
        options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
        cache_dir: str = "",
        additional_compiler_args: str = "",
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize a NKIPyKernel.

        Args:
            gm: The FX GraphModule to compile
            example_inputs: Example inputs for tracing and optimization
            options: Backend options
            cache_dir: Directory for kernel cache artifacts
            additional_compiler_args: Extra compiler flags (included in kernel hash)
            rank: Process rank for distributed execution
            world_size: Total number of processes
        """
        self._cache_dir = cache_dir
        self._additional_compiler_args = additional_compiler_args
        self._rank = rank
        self._world_size = world_size
        self.kernel_hash = hash_gm_with_tensors(
            gm, example_inputs,
            additional_compiler_args=additional_compiler_args,
        )
        self.kernel_paths = self._setup_paths()
        self.nkipy_func = None
        self.alias_map: Dict[int, int] = {}
        self.none_idx_list: List[int] = []
        self.gm = gm

        # FIXME fix to a default value
        self.on_device = True

        # Only build if not cached, then load (either from cache or newly built)
        if not self._is_kernel_cached():
            logger.info(f"Building new kernel with hash {self.kernel_hash}")
            self._build_new_kernel(gm, example_inputs)

        logger.info(f"Loading kernel with hash {self.kernel_hash}")
        self._load_cached_kernel()

        self.ntff_meta = NtffMeta.from_options_and_kernel_hash(
            options, self.kernel_hash
        )

    def _setup_paths(self) -> Dict[str, Path]:
        """
        Set up paths for kernel artifacts.

        Creates a dictionary of paths for various kernel-related files
        based on the kernel hash.

        Returns:
            Dictionary mapping artifact names to their file paths
        """
        kernel_dir = Path(self._cache_dir) / f"kernel_{self.kernel_hash}"
        return {
            "kernel_dir": kernel_dir,
            "func_file": kernel_dir / NKIPY_FUNC_FILE,
            "alias_file": kernel_dir / ALIAS_MAP_FILE,
            "none_output_idx_file": kernel_dir / NONE_IDX_LIST,
            "debug_file": kernel_dir / NKIPY_DEBUG_FUNC_FILE,
            "arg_shape_dtype_file": kernel_dir / ARG_SHAPE_DTYPE_FILE,
        }

    def _is_kernel_cached(self) -> bool:
        """
        Check if the kernel is already cached.

        Returns:
            True if all required kernel files exist, False otherwise
        """
        return (
            self.kernel_paths["func_file"].exists()
            and self.kernel_paths["alias_file"].exists()
            and self.kernel_paths["none_output_idx_file"].exists()
        )

    def _load_cached_kernel(self) -> None:
        """
        Load a cached kernel from disk.

        Raises:
            ValueError: If loading fails due to missing or corrupt files
        """
        try:
            self.nkipy_func = load_func_from_file(
                self.kernel_paths["func_file"], NKIPY_FUNC_NAME
            )
            with open(self.kernel_paths["alias_file"], "rb") as f:
                self.alias_map = pickle.load(f)
            with open(self.kernel_paths["none_output_idx_file"], "rb") as f:
                self.none_idx_list = pickle.load(f)
        except (IOError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load cached kernel: {e}")
            raise ValueError(f"Failed to load cached kernel: {e}") from e

    def _build_new_kernel(
        self, gm: fx.GraphModule, example_inputs: Sequence[InputType]
    ) -> None:
        """
        Build a new kernel from the given GraphModule.

        Creates necessary directories, saves the graph module for reference,
        and builds the NKIPy function.

        Args:
            gm: The GraphModule to compile
            example_inputs: Example inputs for tracing
        """
        # Create kernel directory if it doesn't exist
        self.kernel_paths["kernel_dir"].mkdir(parents=True, exist_ok=True)

        # Save the original graph module for reference
        save_gm_to_file(gm, self.kernel_paths["kernel_dir"], "dynamo_aten")

        # Clear the NKI kernel cache
        NKIOpRegistry.reset_procesed_kernels()

        # Build kernel and save artifacts
        builder = NKIPyBuilder(gm, example_inputs, self.kernel_paths["kernel_dir"])
        builder.save_nkipy_func_and_meta(
            self.kernel_paths["func_file"],
            self.kernel_paths["alias_file"],
            self.kernel_paths["none_output_idx_file"],
        )
        builder.save_debug_file(self.kernel_paths["debug_file"])

    def _handle_inplace_update(self, inputs, outputs):
        """
        Performs in-place updates on input objects using corresponding output objects
        based on an alias mapping.

        Args:
            inputs: List of input objects (numpy arrays or PyTorch tensors)
            outputs: List of output objects (numpy arrays or PyTorch tensors)

        Returns:
            List of outputs excluding those used for in-place updates
        """
        alias_map = self.alias_map

        if not alias_map:
            return outputs

        for output_index, input_index in alias_map.items():
            output_obj = outputs[output_index]
            input_obj = inputs[input_index]

            if isinstance(input_obj, np.ndarray):
                np.copyto(input_obj, output_obj)
            elif isinstance(input_obj, torch.Tensor):
                input_obj.copy_(output_obj)
            else:
                raise TypeError(
                    f"In-place update not supported for type "
                    f"{type(input_obj).__name__}. "
                    f"Supported types are numpy.ndarray and torch.Tensor."
                )

        # Filter out outputs that were used for in-place updates
        outputs_filtered = [
            outputs[i] for i in range(len(outputs)) if i not in alias_map
        ]
        return outputs_filtered

    def _handle_none_output(self, outputs):
        outputs = list(outputs)
        for idx in self.none_idx_list:
            outputs.insert(idx, None)
        return outputs

    def _execute_on_host(self, *args: torch.Tensor):
        """
        Execute the compiled kernel with the given inputs on the host.

        Converts PyTorch tensors to NumPy arrays, runs the NKIPy function,
        and converts the results back to PyTorch tensors.

        Args:
            *args: Input tensors (PyTorch)

        Returns:
            Output tensor(s) from the compiled function
        """
        if self.nkipy_func is None:
            raise RuntimeError("Kernel not initialized - nkipy_func is None")

        # Convert PyTorch tensors to NumPy arrays
        args_numpy = [tensor_to_numpy(arg) for arg in args]

        # Execute the nkipy function
        out_numpy = self.nkipy_func(*args_numpy)

        # Handle any inplace updates
        out_numpy = self._handle_inplace_update(args_numpy, out_numpy)

        out_numpy = self._handle_none_output(out_numpy)

        # Convert back to PyTorch tensors
        out_tensor = convert_numpy_arrays_to_tensors(out_numpy)
        return out_tensor

    def _execute_on_device(self, *args: torch.Tensor):
        """
        Execute the compiled kernel with the given inputs on the device.

        Args:
            *args: Input tensors (PyTorch)

        Returns:
            Output tensor(s) from the compiled function
        """
        return compile_load_execute(
            nkipy_func=self.nkipy_func,
            kernel_hash=self.kernel_hash,
            args=args,
            alias_map=self.alias_map,
            none_idx_list=self.none_idx_list,
            kernel_dir=self.kernel_paths["kernel_dir"],
            ntff_meta=self.ntff_meta,
            rank=self._rank,
            world_size=self._world_size,
            additional_compiler_args=self._additional_compiler_args,
        )

    def _save_arg_shape_dtype(self, args):
        """
        Save the input shapes and dtypes to a file for later use.
        """
        args_numpy = [meta_tensor_to_numpy(arg) for arg in args]
        args_shapes_dtypes = [(arg.shape, arg.dtype) for arg in args_numpy]
        with open(self.kernel_paths["arg_shape_dtype_file"], "wb") as f:
            pickle.dump(args_shapes_dtypes, f)

    def _generate_dummy_outputs(self, device):
        """
        Generate dummy outputs with the same dtype, shape, and device as real outputs.
        """
        output_node = self.gm.graph.output_node()
        dummy_outputs = []
        for arg_node in output_node.args[0]:
            dummy_o = torch.empty(
                get_shape_from_fx_node(arg_node),
                dtype=get_dtype_from_fx_node(arg_node),
                device=device,
            )
            dummy_outputs.append(dummy_o)
        return tuple(dummy_outputs)

    @torch._dynamo.disable
    def __call__(self, *args: torch.Tensor):
        """
        Execute the compiled kernel with the given inputs.

        Routes execution to either host or device based on configuration.

        Args:
            *args: Input tensors (PyTorch)

        Returns:
            Output tensor(s) from the compiled function
        """
        if self.on_device:
            # If parallel compilation is enabled, we will always return some
            # empty values if the kernel is called without being compiled.
            # User will see warnings about this behavior.
            if in_parallel_compile_context():
                logger.warning(
                    f"""Executing kernel {self.kernel_hash} in parallel compile """
                    f"""context. Dummy values will be returned. Make sure to run """
                    f"""the kernel outside of the parallel compile context to get"""
                    f"""actual outputs. """
                )
                # We write the input shapes and data types into the compile cache
                # directory for later use
                self._save_arg_shape_dtype(args)
                return self._generate_dummy_outputs(args[0].device)
            else:
                return self._execute_on_device(*args)
        else:
            return self._execute_on_host(*args)
