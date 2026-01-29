# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.hardtanh.default")
def hardtanh_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle hardtanh.default operation.

    Implements the hardtanh activation function: f(x) = clip(x, min_val, max_val)
    Maps PyTorch's hardtanh to NumPy's minimum/maximum functions.

    Args:
        node: The FX node representing the hardtanh operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0].name

    # Get optional arguments with their default values
    min_val = node.args[1] if len(node.args) > 1 else -1.0
    max_val = node.args[2] if len(node.args) > 2 else 1.0

    # Convert arguments to strings for the AST
    min_val_str = str(min_val)
    max_val_str = str(max_val)

    # Use temp var generator for intermediate steps
    temp_vars = TempVarGenerator(node.name)
    temp_result = temp_vars.next()

    # Apply hardtanh using numpy minimum/maximum
    # Step 1: Apply minimum bound using maximum: max(x, min_val)
    ast_block.add_numpy_call_assignment(
        target=temp_result, func_name="maximum", args=[input_tensor, min_val_str]
    )

    # Step 2: Apply maximum bound using minimum: min(result, max_val)
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="minimum", args=[temp_result, max_val_str]
    )
