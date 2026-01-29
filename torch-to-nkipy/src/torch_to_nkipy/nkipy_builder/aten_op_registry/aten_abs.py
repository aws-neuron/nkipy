# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.abs.default")
def abs_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle absolute value operation.

    Maps PyTorch's torch.abs to NumPy's abs function.
    Computes the absolute value of each element in the input tensor.

    PyTorch signature: torch.abs(input) -> Tensor
    NumPy equivalent: np.abs(input)

    Args:
        node: The FX node representing the abs operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If input tensor argument is missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if not node.args:
        raise ValueError("abs.default requires input tensor argument")

    # Extract input tensor
    input_tensor = node.args[0].name

    # Add the absolute value operation
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="abs", args=[input_tensor]
    )
