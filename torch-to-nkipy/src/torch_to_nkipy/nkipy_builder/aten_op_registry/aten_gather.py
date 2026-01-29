# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.gather.default")
def gather_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle gather operation.

    Maps PyTorch's torch.gather to NumPy's take_along_axis function.
    Gathers values along an axis according to indices.

    PyTorch signature: torch.gather(input, dim, index) -> Tensor
    NumPy equivalent: np.take_along_axis(input, indices, axis=dim)

    Args:
        node: The FX node representing the gather operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 3:
        raise ValueError(
            "gather.default requires input tensor, dimension, and index arguments"
        )

    # Extract arguments
    input_tensor = node.args[0].name
    dim = node.args[1]
    index = node.args[2].name

    # Add the gather operation using take_along_axis
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="take_along_axis",
        args=[input_tensor, index],
        kwargs={"axis": str(dim)},
    )
