# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.sigmoid.default")
def sigmoid_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle sigmoid activation.

    Implements the sigmoid function: σ(x) = 1 / (1 + exp(-x))
    Maps PyTorch's torch.sigmoid to a sequence of NumPy operations.

    This is an example of a complex operation that can't be mapped
    directly to a single NumPy function, so it's decomposed into
    multiple steps using the TempVarGenerator.

    Args:
        node: The FX node representing the sigmoid operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract the input node name
    x = node.args[0].name

    # Use temp var generator to create variables for each step
    temp_vars = TempVarGenerator(node.name)

    # Step 1: Compute -x
    neg_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(target=neg_var, func_name="negative", args=[x])

    # Step 2: Compute exp(-x)
    exp_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(target=exp_var, func_name="exp", args=[neg_var])

    # Step 3: Compute 1 + exp(-x)
    add_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=add_var, func_name="add", args=["1", exp_var]
    )

    # Step 4: Compute 1 / (1 + exp(-x))
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="divide", args=["1", add_var]
    )
