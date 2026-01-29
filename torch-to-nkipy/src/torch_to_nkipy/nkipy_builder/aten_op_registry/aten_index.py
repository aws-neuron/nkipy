# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.index.Tensor")
def index_tensor(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor indexing operation.

    Maps PyTorch's tensor indexing to NumPy's advanced indexing.
    Handles multiple index tensors for advanced indexing.

    PyTorch signature: torch.index(input, indices) -> Tensor
    NumPy equivalent: input[indices]

    Args:
        node: The FX node representing the index operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 2:
        raise ValueError("index.Tensor requires input tensor and indices arguments")

    # Extract arguments
    input_tensor = node.args[0].name
    indices = node.args[1]

    # Process indices
    index_strs = []
    for idx in indices:
        if idx is None:
            index_strs.append("")  # None becomes empty string for slice
        elif isinstance(idx, fx.Node):
            index_strs.append(idx.name)
        else:
            index_strs.append(str(idx))

    # Add the indexing operation
    ast_block.add_subscript_assignment(
        lhs_name=node.name,
        rhs_name=input_tensor,
        indices=index_strs,
        lhs_is_indexed=False,
    )
