# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.log1p.default")
def log1p_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle log1p operation.

    Implements the log1p function: log1p(x) = log(1 + x)
    Maps PyTorch's torch.log1p to a sequence of NumPy operations.

    Args:
        node: The FX node representing the log1p operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract the input argument
    x = node.args[0].name

    # Use temp var generator for intermediate steps
    temp_vars = TempVarGenerator(node.name)

    # Step 1: Add 1 to the input
    sum_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(target=sum_var, func_name="add", args=[x, "1"])

    # Step 2: Apply logarithm to the result
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="log", args=[sum_var]
    )
