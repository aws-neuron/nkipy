# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("_operator.getitem")
def getitem_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle _operator.getitem operation to access elements by index.

    Maps operations like `result = container[index]` to proper indexing
    in generated code. Commonly used to extract elements from tuples
    returned by operations.

    Args:
        node: The FX node representing the getitem operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract the container and index
    container = node.args[0].name
    index = node.args[1]

    # Create the indexing operation
    ast_block.add_subscript_assignment(
        lhs_name=node.name,
        rhs_name=container,
        indices=[str(index)],
        lhs_is_indexed=False,
    )
