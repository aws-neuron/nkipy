# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.name import NUMPY_PKG


def _normalize_scalar_constant(value):
    """
    Normalize scalar constants for NumPy operations.

    Handles special constants like infinity values that need to be
    represented as NumPy constants in the generated code.

    Args:
        value: The scalar value to normalize

    Returns:
        String representation of the normalized value
    """
    if value == float("inf"):
        return f"{NUMPY_PKG}.inf"
    elif value == float("-inf"):
        return f"{NUMPY_PKG}.NINF"
    else:
        return str(value)


def _create_comparison_handler(
    numpy_func_name: str, is_scalar: bool = False
) -> Callable:
    """
    Factory function to create comparison operation handlers.

    Creates handlers for both tensor-tensor and tensor-scalar comparisons
    with consistent patterns and special constant handling.

    Args:
        numpy_func_name: The NumPy function name (e.g., "greater", "equal")
        is_scalar: Whether this handles scalar comparisons (affects argument processing)

    Returns:
        A handler function for the comparison operation
    """

    def handler(node: fx.Node, computation_node: ComputationNode) -> None:
        ast_block = computation_node.ast_code_block

        # Extract arguments
        left = node.args[0].name
        right = node.args[1]

        if is_scalar:
            # For scalar operations, normalize the right operand
            right_arg = _normalize_scalar_constant(right)
        else:
            # For tensor operations, right is another tensor
            right_arg = right.name

        # Add the comparison operation
        ast_block.add_numpy_call_assignment(
            target=node.name, func_name=numpy_func_name, args=[left, right_arg]
        )

    return handler
