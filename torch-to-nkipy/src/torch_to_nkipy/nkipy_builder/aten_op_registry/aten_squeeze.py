# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.squeeze.dims")
def squeeze_dims(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle squeeze operation with specified dimensions using reshape.

    Maps PyTorch's torch.squeeze with dims parameter to reshape with a statically
    determined shape.

    Example:
        PyTorch:
        ```
        # Remove dimension 0 which has size 1, converting from shape [1, 768, 2048]
        # to shape [768, 2048]
        result = torch.ops.aten.squeeze.dims(tensor, [0])
        ```

        Our NumPy implementation using reshape:
        ```
        # Static shape calculation at compile time
        result = tensor.reshape(768, 2048)
        ```

    Args:
        node: The FX node representing the squeeze operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor_node = node.args[0]  # The tensor node to squeeze
    input_tensor_name = input_tensor_node.name  # The tensor name
    dims = node.args[1]  # The dimensions to squeeze

    # Get the shape statically at compile time
    orig_shape = get_shape_from_fx_node(input_tensor_node)

    # Convert dims to list if it's not already
    dims_list = dims if isinstance(dims, (list, tuple)) else [dims]
    # Handle negative dim
    dims_list = [dim + len(orig_shape) if dim < 0 else dim for dim in dims_list]

    # Calculate new shape by removing dimensions with size 1 at specified positions
    new_shape = [s for i, s in enumerate(orig_shape) if i not in dims_list or s != 1]

    # Use reshape with statically determined shape
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="reshape",
        args=[input_tensor_name, str(tuple(new_shape))],
    )
