# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.relu.default")
def relu_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle ReLU (Rectified Linear Unit) activation.

    Implements the ReLU function: ReLU(x) = max(0, x)
    Maps PyTorch's torch.relu to NumPy's maximum operation.

    Args:
        node: The FX node representing the ReLU operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input
    x = node.args[0].name

    # Add the maximum operation
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="maximum", args=["0", x]
    )
