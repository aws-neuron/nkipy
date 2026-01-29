# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.log.default")
def log_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle log operation with safe handling of zeros.
    """
    ast_block = computation_node.ast_code_block

    # Extract input
    x = node.args[0].name

    # Use temp vars for safe computation
    temp_vars = TempVarGenerator(node.name)

    # Add small epsilon to prevent log(0)
    safe_input = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=safe_input,
        func_name="maximum",
        args=[x, "1e-7"],  # Small epsilon value
    )

    # Compute log safely
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="log", args=[safe_input]
    )
