# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node

# FIXME cumsum is not supported in NKIPy yet
# @AtenOpRegistry.register("torch.ops.aten.cumsum.default")
# def cumsum_default(node: fx.Node, computation_node: ComputationNode) -> None:
#    """
#    Handle cumulative sum operation.
#
#    Maps PyTorch's torch.cumsum to NumPy's cumsum function.
#    Computes the cumulative sum of elements along a given dimension.
#
#    PyTorch signature: torch.cumsum(input, dim) -> Tensor
#    NumPy equivalent: np.cumsum(input, axis=dim)
#
#    Args:
#        node: The FX node representing the cumsum operation
#        computation_node: The ComputationNode to add code to
#
#    Raises:
#        ValueError: If required arguments are missing
#    """
#    ast_block = computation_node.ast_code_block
#
#    # Validate inputs
#    if len(node.args) < 2:
#        raise ValueError("cumsum.default requires input tensor and dimension args")
#
#    # Extract arguments
#    input_tensor = node.args[0].name
#    dim = node.args[1]
#
#    # Add the cumsum operation
#    ast_block.add_numpy_call_assignment(
#        target=node.name,
#        func_name="cumsum",
#        args=[input_tensor],
#        kwargs={"axis": str(dim)},
#    )


@AtenOpRegistry.register("torch.ops.aten.cumsum.default")
def cumsum_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle cumulative sum operation by unrolling it into a static sequence of additions.

    Maps PyTorch's torch.cumsum to a series of NumPy take, add, expand_dims,
    and concatenate operations. This avoids using np.cumsum directly.

    The logic is as follows:
    1. Iteratively slice the input tensor along the specified dimension.
    2. Maintain an accumulator that holds the running sum.
    3. In each step, add the current slice to the accumulator.
    4. Store each intermediate cumulative sum.
    5. After iterating through all slices, concatenate the results to form the
       final output tensor.

    This entire sequence is generated statically at compile time.

    Args:
        node: The FX node representing the cumsum operation.
        computation_node: The ComputationNode to add generated code to.

    Raises:
        ValueError: If required arguments are missing or the input tensor shape
                    cannot be determined.
    """
    ast_block = computation_node.ast_code_block

    # Validate and extract inputs
    if len(node.args) < 2:
        raise ValueError("cumsum.default requires input tensor and dimension arguments")

    input_node = node.args[0]
    input_tensor_name = input_node.name
    dim = node.args[1]

    # Get the shape to determine the number of iterations
    input_shape = get_shape_from_fx_node(input_node)
    if input_shape is None:
        raise ValueError(
            f"Cannot determine shape for input tensor {input_tensor_name} in cumsum op."
        )

    # Handle negative dimension
    if dim < 0:
        dim += len(input_shape)

    dim_size = input_shape[dim]

    if dim_size == 0:
        # If the dimension is empty, the result is an empty tensor of the same shape.
        # This is essentially a copy.
        ast_block.add_numpy_call_assignment(
            target=node.name, func_name="copy", args=[input_tensor_name]
        )
        return

    temp_vars = TempVarGenerator(node.name)
    cumulative_slices = []

    # --- Step 1: Handle the first slice (initial accumulator) ---
    # Take the first slice
    first_slice_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=first_slice_var,
        func_name="take",
        args=[input_tensor_name, "0"],
        kwargs={"axis": str(dim)},
    )

    # The first slice is the initial value of our accumulator
    accumulator_var = first_slice_var

    # Expand its dimension to prepare for concatenation
    expanded_first_slice_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=expanded_first_slice_var,
        func_name="expand_dims",
        args=[accumulator_var],
        kwargs={"axis": str(dim)},
    )
    cumulative_slices.append(expanded_first_slice_var)

    # --- Step 2: Statically unroll the loop for the remaining slices ---
    for i in range(1, dim_size):
        # Take the current slice
        current_slice_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=current_slice_var,
            func_name="take",
            args=[input_tensor_name, str(i)],
            kwargs={"axis": str(dim)},
        )

        # Add it to the accumulator
        new_sum_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=new_sum_var,
            func_name="add",
            args=[accumulator_var, current_slice_var],
        )

        # Update the accumulator for the next iteration
        accumulator_var = new_sum_var

        # Expand dimension of the new sum and add to our list
        expanded_sum_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=expanded_sum_var,
            func_name="expand_dims",
            args=[accumulator_var],
            kwargs={"axis": str(dim)},
        )
        cumulative_slices.append(expanded_sum_var)

    # --- Step 3: Concatenate all the expanded slices ---
    # If there's only one slice, the result is just that slice itself, already expanded.
    if len(cumulative_slices) == 1:
        ast_block.add_assignment(
            CodeGenerator.name_store(node.name),
            CodeGenerator.name_load(cumulative_slices[0]),
        )
    else:
        ast_block.add_numpy_call_assignment(
            target=node.name,
            func_name="concatenate",
            args=[cumulative_slices],
            kwargs={"axis": str(dim)},
        )
