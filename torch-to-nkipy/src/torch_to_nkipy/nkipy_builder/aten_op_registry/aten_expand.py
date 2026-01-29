# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import resolve_shape_placeholder_expand


@AtenOpRegistry.register("torch.ops.aten.expand.default")
def expand_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor expansion operation.

    Maps PyTorch's tensor.expand() to NumPy's broadcast_to function.
    Uses infer_from_implicit_shape to handle shape inference.

    Args:
        node: The FX node representing the expand operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Get input tensor and infer shape
    input_tensor = node.args[0].name
    shape = resolve_shape_placeholder_expand(node.args[0], node.args[1])

    # Add the broadcast_to operation
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="broadcast_to", args=[input_tensor, str(shape)]
    )
