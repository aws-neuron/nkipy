# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.split_with_sizes.default")
def split_with_sizes_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle split_with_sizes operation.

    Maps PyTorch's torch.split(tensor, split_sizes, dim) to NumPy's split function.
    Splits a tensor into chunks of specified sizes along a given dimension.

    Only supports static split_sizes provided as a list.

    Args:
        node: The FX node representing the split_with_sizes operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    tensor = node.args[0].name
    split_sizes = node.args[1]

    # Default dimension is 0 if not specified
    dim = 0
    if len(node.args) > 2:
        dim = node.args[2]

    # Calculate split indices directly at compile time
    # For example, split_sizes [32, 32] becomes split_indices [32]
    split_indices = np.cumsum(split_sizes)[:-1].tolist()

    # Add the split operation directly
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="split",
        args=[tensor, str(split_indices)],
        kwargs={"axis": str(dim)},
    )
