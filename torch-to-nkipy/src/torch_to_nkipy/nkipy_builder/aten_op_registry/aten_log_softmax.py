# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten._log_softmax.default")
def _log_softmax_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle log-softmax operation.

    Implements the log-softmax function:
    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    Maps PyTorch's torch.log_softmax to a sequence of NumPy operations.

    Args:
        node: The FX node representing the log_softmax operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    x = node.args[0].name
    dim = str(node.args[1])  # dimension along which to compute log-softmax

    if node.args[2] is True:
        raise NotImplementedError("log_softmax half_to_float has not been implemented")

    # Use temp var generator to create variables for intermediate steps
    temp_vars = TempVarGenerator(node.name)

    # Step 1: Compute max along specified dimension (keeping dims for broadcasting)
    max_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=max_var,
        func_name="max",
        args=[x],
        kwargs={"axis": dim, "keepdims": "True"},
    )

    # Step 2: Subtract max from input (for numerical stability)
    shifted_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=shifted_var, func_name="subtract", args=[x, max_var]
    )

    # Step 3: Compute exp(x - max)
    exp_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=exp_var, func_name="exp", args=[shifted_var]
    )

    # Step 4: Compute sum of exp values along specified dimension
    sum_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=sum_var,
        func_name="sum",
        args=[exp_var],
        kwargs={"axis": dim, "keepdims": "True"},
    )

    # Step 5: Compute log of the sum
    log_sum_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=log_sum_var, func_name="log", args=[sum_var]
    )

    # Step 6: Compute final log-softmax by subtracting log(sum(exp)) from shifted input
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="subtract", args=[shifted_var, log_sum_var]
    )
