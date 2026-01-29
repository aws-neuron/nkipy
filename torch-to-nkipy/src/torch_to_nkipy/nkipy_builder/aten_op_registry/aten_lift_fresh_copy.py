# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.lift_fresh_copy.default")
def lift_fresh_copy_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle lift_fresh_copy operation.

    Maps PyTorch's lift_fresh_copy to NumPy's copy operation.
    This creates a new copy of the input tensor.

    Args:
        node: The FX node representing the lift_fresh_copy operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input tensor
    input_tensor = node.args[0].name

    # Add the copy operation
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="copy", args=[input_tensor]
    )
