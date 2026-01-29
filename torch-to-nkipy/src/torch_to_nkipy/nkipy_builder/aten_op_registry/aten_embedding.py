# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.embedding.default")
def embedding_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle embedding lookup operation.

    Maps PyTorch's torch.embedding to NumPy's take operation along axis 0.
    This performs lookup in an embedding table using input indices.

    Args:
        node: The FX node representing the embedding operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    weight = node.args[0].name
    input_ids = node.args[1].name

    # Add the take operation
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="take",
        args=[weight, input_ids],
        kwargs={"axis": "0"},
    )
