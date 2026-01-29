# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.rsqrt.default")
def rsqrt_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle reciprocal square root operation.

    Implements torch.rsqrt(x) as a two-step operation:
    1. Calculate square root of x
    2. Take reciprocal (1/sqrt(x))

    Args:
        node: The FX node representing the rsqrt operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input argument
    x = node.args[0].name

    # Use temp var generator for intermediate calculation
    temp_vars = TempVarGenerator(node.name)
    sqrt_var = temp_vars.next()

    # Step 1: Calculate square root
    ast_block.add_numpy_call_assignment(target=sqrt_var, func_name="sqrt", args=[x])

    # Step 2: Calculate reciprocal (1 / sqrt(x))
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="divide", args=["1", sqrt_var]
    )
