# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.select.int")
def select_int(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor selection at a specific index along a dimension.

    Maps PyTorch's torch.select(tensor, dim, index) to NumPy's take function.

    PyTorch signature: torch.select(tensor, dim, index)
    NumPy equivalent: np.take(tensor, indices=index, axis=dim)

    Args:
        node: The FX node representing the select operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    tensor = node.args[0].name
    dim = node.args[1]
    index = node.args[2]

    # Add the take operation
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="take",
        args=[tensor, str(index)],
        kwargs={"axis": str(dim)},
    )
