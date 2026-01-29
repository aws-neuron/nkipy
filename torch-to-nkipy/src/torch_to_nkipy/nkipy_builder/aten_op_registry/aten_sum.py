# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.sum.dim_IntList")
def sum_dim_intlist(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle sum operation along specified dimensions.

    Maps PyTorch's torch.sum(input, dim, keepdim) to NumPy's sum function.

    PyTorch signature: torch.sum(input, dim, keepdim=False)
    NumPy equivalent: np.sum(a, axis=dim, keepdims=keepdim)

    Args:
        node: The FX node representing the sum operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0].name
    dim = node.args[1]  # This could be an int or a list of ints

    # Extract keepdim with default value (False in PyTorch)
    keepdim = False
    if len(node.args) > 2:
        keepdim = node.args[2]

    if isinstance(dim, list):
        if len(dim) == 0:
            dim_arg = "None"
        elif len(dim) == 1:
            dim_arg = str(dim[0])
        else:
            dim_arg = str(tuple(dim))
    else:
        raise NotImplementedError(
            f"Unhandled Node {node} with args {node.args} and kwargs {node.kwargs}"
        )

    # Add the sum operation
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="sum",
        args=[input_tensor],
        kwargs={"axis": dim_arg, "keepdims": str(keepdim)},
    )
