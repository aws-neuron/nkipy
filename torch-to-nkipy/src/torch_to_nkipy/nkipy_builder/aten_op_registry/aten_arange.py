# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str


@AtenOpRegistry.register("torch.ops.aten.arange.start_step")
def arange_start_step(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle arange operation with start and step parameters.

    Maps PyTorch's torch.arange(start, end, step) to NumPy's arange function.

    Args:
        node: The FX node representing the arange operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    args = list(node.args)

    # Check for step in kwargs and add it to args if present
    if "step" in node.kwargs:
        args.append(node.kwargs["step"])

    kwargs = {}
    if "dtype" in node.kwargs:
        kwargs["dtype"] = torch_to_numpy_dtype_str(node.kwargs["dtype"])

    # Convert args to strings for the AST builder
    arg_strings = [str(arg) for arg in args]

    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="arange", args=arg_strings, kwargs=kwargs
    )
