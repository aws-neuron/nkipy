# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.clamp.default")
def clamp_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle clamp operation to restrict values to [min, max] range.
    """
    ast_block = computation_node.ast_code_block

    # Extract inputs
    x = node.args[0].name
    min_val = node.args[1]  # Could be None or a value
    max_val = node.args[2] if len(node.args) > 2 else None

    # Generate variable names
    temp_vars = TempVarGenerator(node.name)
    result = node.name

    # Handle clamping based on provided min and max values
    if min_val is not None and max_val is not None:
        # Both min and max are specified
        temp_result = temp_vars.next()

        # First apply the minimum bound
        ast_block.add_numpy_call_assignment(
            target=temp_result, func_name="maximum", args=[x, str(min_val)]
        )

        # Then apply the maximum bound
        ast_block.add_numpy_call_assignment(
            target=result, func_name="minimum", args=[temp_result, str(max_val)]
        )
    elif min_val is not None:
        # Only min is specified
        ast_block.add_numpy_call_assignment(
            target=result, func_name="maximum", args=[x, str(min_val)]
        )
    elif max_val is not None:
        # Only max is specified
        ast_block.add_numpy_call_assignment(
            target=result, func_name="minimum", args=[x, str(max_val)]
        )
    else:
        raise ValueError("clamp.default requires a min or max input")
