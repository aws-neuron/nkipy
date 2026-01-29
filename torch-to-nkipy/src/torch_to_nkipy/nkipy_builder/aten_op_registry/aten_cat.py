# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.cat.default")
def cat_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor concatenation operation.

    Maps PyTorch's torch.cat to NumPy's concatenate.

    Expected PyTorch format: torch.cat(tensors, dim=0)
    NumPy equivalent: np.concatenate(tensors, axis=dim)

    Args:
        node: The FX node representing the concatenation operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If the first argument is not a list of tensors
    """
    ast_block = computation_node.ast_code_block

    # Ensure we have tensor list as first argument
    if not node.args or not isinstance(node.args[0], (list, tuple)):
        raise ValueError(
            f"Expected first argument to be a list of tensors, got: {node.args}"
        )

    # Extract tensor list
    tensors = node.args[0]
    tensor_names = [tensor.name for tensor in tensors]

    # Determine concatenation dimension
    if len(node.args) > 1:
        dim = node.args[1]
    else:
        dim = 0  # Default dimension in PyTorch

    # Map PyTorch's dim to NumPy's axis
    kwargs = {"axis": str(dim)}

    # Add the concatenate operation
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="concatenate",
        args=[tensor_names],  # Pass tensor names as a list
        kwargs=kwargs,
    )
