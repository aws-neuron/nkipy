# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.aten_op_registry.helper_functions import (
    _normalize_scalar_constant,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.full_like.default")
def full_like_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle full_like operation.

    Maps PyTorch's torch.full_like to NumPy's full_like. This creates a tensor
    with the same size and dtype as the input, filled with a specified value.

    Expected PyTorch format: torch.full_like(input, fill_value)
    NumPy equivalent: np.full_like(input, fill_value)

    Args:
        node: The FX node representing the full_like operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing or of incorrect types
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 2:
        raise ValueError(
            f"full_like requires at least 2 arguments, got {len(node.args)}"
        )

    # Extract arguments
    tensor = node.args[0].name
    value = node.args[1]

    # Special handling for constants
    value = _normalize_scalar_constant(value)

    # Add the full_like operation
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="full_like", args=[tensor, value]
    )
