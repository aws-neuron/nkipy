# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.topk.default")
def topk_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle topk operation.

    Maps PyTorch's torch.topk to NKIPy's tensor_apis.topk function.
    PyTorch signature: torch.topk(input, k, dim=-1, largest=True, sorted=True)

    Args:
        node: The FX node representing the topk operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0]
    k = node.args[1]
    axis = len(get_shape_from_fx_node(input_tensor)) - 1

    # Set axis to last dimension
    kwargs = {"k": str(k), "axis": str(axis)}

    # Check for extra positional arguments and warn if present
    if len(node.args) > 2:
        user_axis = node.args[2]
        if user_axis != -1 and user_axis != axis:
            raise NotImplementedError(f"last_axis={axis}, user_axis={user_axis}")

    if len(node.args) > 3:
        largest = node.args[3]
        kwargs["is_ascend"] = str(not (largest))

    if len(node.args) > 4:
        sorted = node.args[4]
        raise NotImplementedError(f"sorted={sorted} not implemented")

    if node.kwargs:
        raise NotImplementedError(f"kwargs={node.kwargs} not implemented")

    # Add the topk operation with k and axis as keyword arguments
    ast_block.add_call_assignment(
        target=node.name,
        pkg_or_obj="tensor_apis",
        func="topk",
        args=[input_tensor.name],
        kwargs=kwargs,
    )
