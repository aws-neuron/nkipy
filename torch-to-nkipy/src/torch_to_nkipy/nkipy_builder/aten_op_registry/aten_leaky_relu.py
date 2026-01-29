# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.leaky_relu.default")
def leaky_relu_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle Leaky ReLU activation.

    Implements the Leaky ReLU function: LeakyReLU(x) = max(negative_slope * x, x)
    Maps PyTorch's torch.leaky_relu to NumPy's maximum operation.

    Args:
        node: The FX node representing the Leaky ReLU operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input
    x = node.args[0].name

    # Extract negative slope (default to 0.01 if not provided)
    if len(node.args) > 1:
        negative_slope = node.args[1]
    else:
        negative_slope = 0.01

    # Create the scaled input expression
    scaled_x = f"{negative_slope} * {x}"

    # Add the maximum operation: max(negative_slope * x, x)
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="maximum", args=[scaled_x, x]
    )
