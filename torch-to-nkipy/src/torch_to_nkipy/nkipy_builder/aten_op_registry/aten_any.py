# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.any.default")
def any_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    torch.any(input) → bool (0-D)
    """
    ast_block = computation_node.ast_code_block
    x = node.args[0].name
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="any",
        args=[x],
    )


# FIXME The numpy.any has not been implemented in NKIPy yet
# we use a workaround solution for now
# @AtenOpRegistry.register("torch.ops.aten.any.dim")
# def any_dim(node: fx.Node, computation_node: ComputationNode) -> None:
#    """
#    Handle any operation along a specific dimension.
#
#    Maps PyTorch's torch.any(input, dim, keepdim) to NumPy's any() function.
#
#    PyTorch signature: torch.any(input, dim, keepdim=False) -> Tensor
#    NumPy equivalent: np.any(a, axis=dim, keepdims=keepdim)
#
#    Args:
#        node: The FX node representing the any operation
#        computation_node: The ComputationNode to add code to
#
#    Raises:
#        ValueError: If the required arguments are missing
#    """
#    ast_block = computation_node.ast_code_block
#
#    # Validate inputs
#    if len(node.args) < 2:
#        raise ValueError(f"any.dim requires at least 2 args, got {len(node.args)}")
#
#    # Extract arguments
#    source = node.args[0].name
#    dim = node.args[1]
#
#    # Extract keepdim with default value (False in PyTorch)
#    keepdim = False
#    if len(node.args) > 2:
#        keepdim = node.args[2]
#
#    # Add the any operation
#    # Map PyTorch's 'dim' to NumPy's 'axis' and 'keepdim' to 'keepdims'
#    ast_block.add_numpy_call_assignment(
#        target=node.name,
#        func_name="any",
#        args=[source],
#        kwargs={"axis": str(dim), "keepdims": str(keepdim)},
#    )


@AtenOpRegistry.register("torch.ops.aten.any.dim")
def any_dim(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle any operation along a specific dimension.

    Maps PyTorch's torch.any(input, dim, keepdim) to a sequence of NumPy operations:
    1. Calculate not_equal(input, 0) to get boolean values
    2. Sum the boolean values along the specified axis
    3. Convert back to boolean by checking not_equal(sum, 0)

    This is functionally equivalent to np.any(input, axis=dim, keepdims=keepdim).

    Args:
        node: The FX node representing the any operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 2:
        raise ValueError(f"any.dim requires at least 2 arguments, got {len(node.args)}")

    # Extract arguments
    source = node.args[0].name
    dim = node.args[1]

    # Extract keepdim with default value (False in PyTorch)
    keepdim = False
    if len(node.args) > 2:
        keepdim = node.args[2]

    # Create temp var generator for intermediate results
    temp_vars = TempVarGenerator(node.name)
    tmp_bool = temp_vars.next()
    tmp_sum = temp_vars.next()

    # Step 1: Calculate not_equal(input, 0) to get boolean values
    ast_block.add_numpy_call_assignment(
        target=tmp_bool, func_name="not_equal", args=[source, "0"]
    )

    # Step 2: Sum the boolean values along the specified axis
    ast_block.add_numpy_call_assignment(
        target=tmp_sum,
        func_name="sum",
        args=[tmp_bool],
        kwargs={"axis": str(dim), "keepdims": str(keepdim)},
    )

    # Step 3: Convert back to boolean by checking not_equal(sum, 0)
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="not_equal", args=[tmp_sum, "0"]
    )
