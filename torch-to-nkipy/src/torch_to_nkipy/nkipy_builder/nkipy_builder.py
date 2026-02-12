# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.fx as fx
from torch._inductor.utils import InputType
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import _get_qualified_name

from nkipy.core.language import bfloat16
from torch_to_nkipy.nkipy_builder.aten_op_registry import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import (
    ComputationNode,
    InputNode,
    NKIPyAST,
    OutputNode,
)
from torch_to_nkipy.utils.graph import save_string_to_file, stringify_fx_node
from torch_to_nkipy.utils.name import NKIPY_DEBUG_FUNC_NAME, NKIPY_FUNC_NAME

logger = logging.getLogger(__name__)


class NKIPyBuilder:
    """
    Builder for generating NKIPy functions from FX GraphModules.

    This class processes an FX GraphModule to create a NKIPy function
    that implements the same computation, represented as an AST that can
    be converted to Python code.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        example_inputs: Sequence[InputType],
        kernel_dir: Path,
    ):
        """
        Initialize a NKIPyBuilder.

        Args:
            gm: The FX GraphModule to compile
            example_inputs: Example inputs for tracing and optimization
        """
        self.gm = gm
        self.example_inputs = example_inputs
        self.kernel_dir = kernel_dir

        # Create the AST representation for the NKIPy function
        self.nkipy_func_ast = NKIPyAST()

        # Process each node in the graph to build the function
        logger.info(
            f"Building NKIPyAST from GraphModule with "
            f"{len(list(self.gm.graph.nodes))} nodes"
        )
        for node in self.gm.graph.nodes:
            self._process_node(node)

    def save_nkipy_func_and_meta(
        self, func_file: Path, alias_file: Path, none_idx_file: Path
    ) -> None:
        """
        Save the generated NKIPy function and alias map to files.

        Args:
            func_file: Path where the function code will be saved
            alias_file: Path where the alias map will be saved

        Raises:
            IOError: If writing to either file fails
        """
        # Generate function code as a string
        raw_code = self.nkipy_func_ast.generate_function_code(NKIPY_FUNC_NAME)

        # Ensure parent directory exists
        func_file.parent.mkdir(parents=True, exist_ok=True)

        # Save function code
        logger.debug(f"Saving NKIPy function to {func_file}")
        save_string_to_file(raw_code, func_file)

        # Save alias map
        logger.debug(f"Saving alias map to {alias_file}")
        with open(alias_file, "wb") as f:
            pickle.dump(self.nkipy_func_ast.alias_map, f)

        with open(none_idx_file, "wb") as f:
            pickle.dump(self.nkipy_func_ast.non_tensor_outputs, f)

    def save_debug_file(self, debug_file: Path) -> None:
        """
        Save a debug file containing both the original model and the generated
        NKIPy function.

        Args:
            debug_file: Path where the debug file will be saved

        Raises:
            IOError: If writing to the file fails
        """
        # Generate import statements
        raw_imports = [
            "import torch",
            "from nkipy.core.compile import trace",
            "from nkipy.runtime.execute import simulate_traced_kernel "
            "as simulate_kernel",
            "from nkipy.runtime.execute import baremetal_run_traced_kernel "
            "as baremetal_run",
            "from nkipy.core import tensor_apis",
            "from neuronxcc.nki.language import nc as VNC",
        ]

        # Generate NKIPy function code
        nkipy_raw_code = self.nkipy_func_ast.generate_function_code(NKIPY_FUNC_NAME)

        # Generate modified FX graph module code
        gm_raw_code = self.gm.code
        gm_raw_code = gm_raw_code.replace(
            "\n\n\ndef forward(self, ", f"def {NKIPY_DEBUG_FUNC_NAME}("
        )

        # Generate input tensors creation code
        raw_torch_inputs = []
        torch_inputs = []
        raw_numpy_inputs = []
        numpy_inputs = []

        for i, example_input in enumerate(self.example_inputs):
            if not hasattr(example_input, "shape"):
                # Non-tensor input (e.g. SymInt from dynamic shapes)
                raw_torch_inputs.append(f"arg_torch_{i} = {repr(example_input)}")
                torch_inputs.append(f"arg_torch_{i}")
                raw_numpy_inputs.append(f"arg_np_{i} = arg_torch_{i}")
                numpy_inputs.append(f"arg_np_{i}")
                continue
            shape = example_input.shape
            dtype = example_input.dtype
            tensor_size_dtype_str = f"size={tuple(shape)}, dtype={dtype}"
            if torch.is_floating_point(example_input):
                raw_torch_inputs.append(
                    f"arg_torch_{i} = torch.randn({tensor_size_dtype_str})"
                )
            else:
                raw_torch_inputs.append(
                    f"arg_torch_{i} = torch.randint(0, 100, {tensor_size_dtype_str})"
                )
            # Use clone to avoid inplace update
            torch_inputs.append(f"arg_torch_{i}.clone()")

            # FIXME torch.bfloat16 is not supported in numpy
            raw_numpy_inputs.append(f"arg_np_{i} = arg_torch_{i}.numpy()")
            # Use copy to avoid inplace update
            numpy_inputs.append(f"arg_np_{i}.copy()")

        torch_inputs_str = ", ".join(torch_inputs)
        numpy_inputs_str = ", ".join(numpy_inputs)

        # Generate testing code with descriptive comments
        raw_testing = [
            "# Run original PyTorch model",
            f"out_torch = {NKIPY_DEBUG_FUNC_NAME}({torch_inputs_str})",
            "",
            "# Run NKIPy function directly",
            f"out_numpy = {NKIPY_FUNC_NAME}({numpy_inputs_str})",
            "",
            "# Trace and specialize NKIPy function",
            f"traced_kernel = trace({NKIPY_FUNC_NAME}, kernel_return=True)",
            f"traced_kernel.specialize({numpy_inputs_str})",
            "",
            "# Simulate the kernel",
            f"out_sim = simulate_kernel(traced_kernel, {numpy_inputs_str})",
            "",
            "# Run with baremetal executor",
            f"out_baremetal = baremetal_run(traced_kernel, {numpy_inputs_str})",
            "",
            "# Checking",
            "num_out_torch = len(out_torch)",
            "num_out_numpy = len(out_numpy)",
            "offset = num_out_numpy - num_out_torch",
            "atol = 1e-5",
            "for i in range(num_out_torch):",
            "    torch_np = out_torch[i].detach().numpy()",
            "    numpy_np = out_numpy[i + offset]",
            "    is_close = np.allclose(torch_np, numpy_np, atol=atol)",
            "    print(f'PyTorch/Numpy [out {i}] allclose: {is_close}')",
            "for i in range(num_out_numpy):",
            "    sim_close = np.allclose(out_numpy[i], out_sim[i], atol=atol)",
            "    print(f'Numpy/Simulation [out {i}] allclose: {sim_close}')",
            "    bare_close = np.allclose(out_numpy[i], out_baremetal[i], atol=atol)",
            "    print(f'Numpy/Baremetal [out {i}] allclose: {bare_close}')",
        ]

        # Combine all sections with clear section headers
        raw_code = [
            "# ===============================================",
            "# Auto-generated NKIPy Debug File",
            "# ===============================================",
            "",
            "# ----- Imports -----",
            "\n".join(raw_imports),
            "",
            "# ----- NKIPy Implementation -----",
            nkipy_raw_code,
            "",
            "# ----- Original PyTorch Implementation -----",
            gm_raw_code,
            "",
            "# ----- Test Inputs -----",
            "torch.manual_seed(0)",
            "\n".join(raw_torch_inputs),
            "",
            "\n".join(raw_numpy_inputs),
            "",
            "# ----- Interactive Debug Point -----",
            "# import pdb; pdb.set_trace()",
            "",
            "# ----- Test Execution Paths -----",
            "\n".join(raw_testing),
        ]

        combined_code = "\n".join(raw_code)

        # Ensure parent directory exists
        debug_file.parent.mkdir(parents=True, exist_ok=True)

        # Save function code
        logger.debug(f"Saving NKIPy debug file to {debug_file}")
        save_string_to_file(combined_code, debug_file)

    def _process_node(self, node: fx.Node) -> None:
        """
        Process a single node from the FX graph.

        Based on the node's operation type, delegates to specialized
        processing methods.

        Args:
            node: The FX node to process

        Raises:
            NotImplementedError: If node type is not supported
        """
        # Handle different node types
        if node.op == "placeholder":
            self._process_placeholder(node)
        elif node.op == "call_function":
            self._process_call_function(node)
        elif node.op == "output":
            self._process_output(node)
        elif node.op == "get_attr":
            self._process_get_attr(node)
        else:
            raise NotImplementedError(
                f"Unsupported node type: {node.op} for node {node.name}"
            )

    def _process_placeholder(self, node: fx.Node) -> None:
        """
        Process an input placeholder node.

        Creates an InputNode and adds it to the AST.

        Args:
            node: The placeholder FX node
        """

        # create and add an input node
        input_node = InputNode(
            name=node.name,
            fx_node_str=stringify_fx_node(node),
        )
        self.nkipy_func_ast.add_input(input_node)
        logger.debug(f"Added input node: {node.name}")

    def _process_call_function(self, node: fx.Node) -> None:
        """
        Process a function call node.

        Finds the appropriate handler for the operation and uses it to
        create a computation node.

        Args:
            node: The call_function FX node

        Raises:
            NotImplementedError: If no handler is found for the operation
        """
        # Get the fully qualified name of the operation
        op_name = _get_qualified_name(node.target)

        # Find a handler for this operation
        handler = AtenOpRegistry.get_handler(op_name)
        if not handler:
            logger.error(f"No handler for operation: {op_name}")
            raise NotImplementedError(f"Operation '{op_name}' is not supported")

        # Use the handler to create a computation node
        comp_node = handler(node)
        self.nkipy_func_ast.add_computation(comp_node)
        logger.debug(f"Added computation node for {op_name}: {node.name}")

    def _process_output(self, node: fx.Node) -> None:
        """
        Process an output node.

        Creates OutputNodes for each output and adds them to the AST.
        Skips outputs that were already added as aliased outputs by
        _add_aliased_output (e.g. from copy_ nodes), to avoid
        double-counting when aot_autograd includes mutations as outputs.

        When the same tensor appears multiple times in the output tuple
        (e.g. aot_autograd saving an intermediate for backward), inserts
        an explicit copy so neuronx-cc sees distinct output tensors.

        Args:
            node: The output FX node
        """
        # Collect names already present from aliased outputs
        existing_output_names = {o.name for o in self.nkipy_func_ast.outputs}

        # Track names seen in this loop to detect duplicates
        seen_in_loop = {}

        # For each result in the output tuple
        for arg_idx, arg_node in enumerate(node.args[0]):
            if arg_node is None:
                self.nkipy_func_ast.add_non_tensor_output(arg_idx, None)
                continue
            # aot_autograd may include non-Node values (e.g. int for tensor
            # sizes) in the output tuple for dynamic-shape backward graphs.
            if not isinstance(arg_node, fx.Node):
                self.nkipy_func_ast.add_non_tensor_output(arg_idx, arg_node)
                continue
            # Skip if already added as an aliased output (e.g. from copy_)
            if arg_node.name in existing_output_names:
                logger.debug(
                    f"Skipping duplicate output {arg_idx}: {arg_node.name} "
                    f"(already present as aliased output)"
                )
                continue

            if arg_node.name in seen_in_loop:
                # Duplicate output — insert a copy so neuronx-cc sees
                # distinct tensors (inputs and outputs occupy separate
                # memory regions on device).
                copy_name = f"{arg_node.name}_copy_{arg_idx}"
                copy_node = ComputationNode(name=copy_name)
                copy_node.ast_code_block.add_numpy_call_assignment(
                    target=copy_name,
                    func_name="copy",
                    args=[arg_node.name],
                )
                self.nkipy_func_ast.add_computation(copy_node)
                output_node = OutputNode(name=copy_name)
                logger.debug(
                    f"Inserted copy for duplicate output {arg_idx}: "
                    f"{arg_node.name} -> {copy_name}"
                )
            else:
                seen_in_loop[arg_node.name] = arg_idx
                output_node = OutputNode(name=arg_node.name)
                logger.debug(f"Added output node {arg_idx}: {arg_node.name}")

            self.nkipy_func_ast.add_output(output_node)

    def _process_get_attr(self, node: fx.Node) -> None:
        """
        Process a get_attr node representing a constant tensor.

        Extracts the constant tensor from the GraphModule, converts it to
        a NumPy array, saves it to disk, and creates a node that will load
        this tensor in the generated code.

        Args:
            node: The get_attr FX node

        Raises:
            ValueError: If the attribute is not a tensor
        """
        attr_str = node.target
        const_tensor = getattr(self.gm, attr_str)

        # Ensure we're dealing with a tensor
        if not isinstance(const_tensor, torch.Tensor):
            raise ValueError(
                f"Expected tensor attribute but got {type(const_tensor)} for {attr_str}"
            )

        logger.debug(
            f"Processing constant tensor: {node.name} with shape {const_tensor.shape} "
            f"and dtype {const_tensor.dtype}"
        )

        # Ensure kernel directory exists
        self.kernel_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensor to numpy array with appropriate dtype handling
        with unset_fake_temporarily():
            if const_tensor.dtype == torch.bfloat16:
                # Convert bfloat16 to float32 then to numpy, then cast back to bfloat16
                const_numpy = const_tensor.float().numpy().astype(bfloat16)
            else:
                const_numpy = const_tensor.numpy()

        # Save tensor to file
        tensor_path = self.kernel_dir / f"{node.name}.npy"
        np.save(tensor_path, const_numpy)
        logger.debug(f"Saved constant tensor to {tensor_path}")

        # Create computation node
        comp_node = ComputationNode(
            name=node.name,
            fx_node_str=stringify_fx_node(node),
            fx_node_stack_trace=node.stack_trace,
        )

        # Add code to load the tensor from file
        # Using quoted string for file path to ensure it's treated as a string literal
        comp_node.ast_code_block.add_numpy_call_assignment(
            target=node.name, func_name="load", args=[f'"{node.name}.npy"']
        )

        self.nkipy_func_ast.add_computation(comp_node)
        logger.debug(f"Added get_attr computation node: {node.name}")
