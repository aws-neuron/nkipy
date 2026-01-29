# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.index_select.default")
def index_select_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle index_select operation.

    Maps PyTorch's torch.index_select to NumPy's take function with expand_dims
    to maintain correct dimensionality.

    Example:
        PyTorch:
        ```
        # Select elements from tensor along dimension 0 using indices in select_118
        result = torch.ops.aten.index_select.default(tensor, 0, select_118)
        # For example, if tensor has shape [5, 768, 2048] and select_118 contains [3],
        # result will have shape [1, 768, 2048]
        ```

        Our NumPy implementation:
        ```
        # Take elements from tensor along axis 0 using indices in select_118
        temp = np.take(tensor, select_118, axis=0)

        # If take returns a tensor without the indices dimension, add it back
        if select_118.shape[0] == 1:
            result = np.expand_dims(temp, axis=0)
        else:
            result = temp
        ```

    Args:
        node: The FX node representing the index_select operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0]  # The tensor to select from
    dim = node.args[1]  # The dimension to select along
    indices = node.args[2]  # The indices tensor

    # Use temp var generator for intermediate results
    temp_vars = TempVarGenerator(node.name)
    take_result = temp_vars.next()

    # First perform the take operation
    ast_block.add_numpy_call_assignment(
        target=take_result,
        func_name="take",
        args=[input_tensor.name, indices.name],
        kwargs={"axis": str(dim)},
    )

    # Then add the condition to check if we need to add the dimension back
    condition = len(get_shape_from_fx_node(indices)) == 0

    # Create an if statement block for the condition
    if condition:
        # Inside the if block: result = np.expand_dims(take_result, axis=dim)
        ast_block.add_numpy_call_assignment(
            target=node.name,
            func_name="expand_dims",
            args=[take_result],
            kwargs={"axis": str(dim)},
        )
    else:
        # Inside the else block: result = take_result
        ast_block.add_assignment(
            CodeGenerator.name_store(node.name), CodeGenerator.name_load(take_result)
        )
