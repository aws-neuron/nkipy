# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.slice_scatter.default")
def slice_scatter_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle slice_scatter operation.

    Maps PyTorch's slice_scatter to NumPy operations. This operation copies the source
    tensor and then updates a slice of it with new values.

    PyTorch signature: torch.slice_scatter(input, src, dim=0, start=None, end=None,
    step=1)
    NumPy equivalent: result = input.copy(); result[slice_indices] = src

    Args:
        node: The FX node representing the slice_scatter operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0].name
    src = node.args[1].name
    dim = node.args[2]
    start = node.args[3]
    end = node.args[4]
    step = node.args[5] if len(node.args) > 5 else 1

    # Use temp var generator for the copy operation
    temp_vars = TempVarGenerator(node.name)
    copy_var = temp_vars.next()

    # Step 1: Create a copy of the input tensor
    ast_block.add_numpy_call_assignment(
        target=copy_var, func_name="copy", args=[input_tensor]
    )

    # Create index strings for each dimension
    indices = [""] * dim  # Empty strings for dimensions before our target

    # Create the slice string for our target dimension
    if start == -1:
        start_expr = f"{input_tensor}.shape[{dim}] - 1"
    else:
        start_expr = str(start)

    if end == sys.maxsize:
        end_expr = ""
    else:
        end_expr = str(end)

    # Add step if it's not 1
    step_expr = f":{step}" if step != 1 else ""

    # Combine the slice expression
    slice_str = f"{start_expr}:{end_expr}{step_expr}"
    indices.append(slice_str)

    # Get input tensor shape to ensure all dimensions are included
    input_shape = get_shape_from_fx_node(node.args[0])
    if input_shape is not None:
        indices.extend([""] * (len(input_shape) - len(indices)))

    # Step 2: Update the slice with source values
    ast_block.add_subscript_assignment(
        lhs_name=copy_var, indices=indices, rhs_name=src, lhs_is_indexed=True
    )

    # Step 3: Assign the result
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(copy_var)
    )
