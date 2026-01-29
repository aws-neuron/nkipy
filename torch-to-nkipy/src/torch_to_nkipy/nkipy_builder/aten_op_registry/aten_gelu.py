# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.gelu.default")
def gelu_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle GELU (Gaussian Error Linear Unit) activation.

    Implements GELU: GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal
    distribution.

    Args:
        node: The FX node representing the GELU operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input
    x = node.args[0].name

    # Use temp vars for intermediate calculations
    temp_vars = TempVarGenerator(node.name)

    # Calculate x^3
    cube_term = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=cube_term, func_name="power", args=[x, "3"]
    )

    # Calculate 0.044715 * x^3
    scaled_cube = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=scaled_cube, func_name="multiply", args=["0.044715", cube_term]
    )

    # Calculate x + 0.044715 * x^3
    inner_sum = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=inner_sum, func_name="add", args=[x, scaled_cube]
    )

    # Calculate 0.7978845608028654 * (x + 0.044715 * x^3)
    tanh_input = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=tanh_input, func_name="multiply", args=["0.7978845608028654", inner_sum]
    )

    # Calculate tanh term
    tanh_result = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=tanh_result, func_name="tanh", args=[tanh_input]
    )

    # Calculate final result
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="multiply", args=[x, f"(0.5 * (1 + {tanh_result}))"]
    )
