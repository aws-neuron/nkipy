# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.constant_pad_nd.default")
def constant_pad_nd_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle constant padding operation.

    Lowers PyTorch's constant padding operation to numpy. Since NKIPy does not
    support padding for now, we allocate a larger tensor and store the original
    tensor into its space.

    Args:
        node: The FX node representing the constant_pad_nd operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input tensor and padding information
    input_tensor = node.args[0].name
    pad_sizes = node.args[1]
    # Handle pad_val from either kwargs or args, default to 0
    pad_val = node.kwargs.get("pad_val", node.args[2] if len(node.args) > 2 else 0)
    dtype = torch_to_numpy_dtype_str(node.args[0].meta["val"].dtype)

    # Since NKIPy does not support np.pad yet, we create a constant tensor
    # of the target shape and put the original tensor into it

    # Compute target shape and create index strings for slicing
    input_shape = get_shape_from_fx_node(node.args[0])
    input_dim = len(input_shape)
    target_shape = [None] * input_dim
    slicing_indices = []
    for dim in range(len(pad_sizes) // 2):
        total_pad_size_dim = pad_sizes[dim * 2] + pad_sizes[dim * 2 + 1]
        target_shape[input_dim - dim - 1] = (
            input_shape[input_dim - dim - 1] + total_pad_size_dim
        )
        if total_pad_size_dim == 0:
            slicing_indices.append("")
        else:
            start = pad_sizes[dim * 2]
            end = start + input_shape[input_dim - dim - 1]
            slicing_indices.append(f"{start}:{end}")
    slicing_indices.reverse()

    # Create constant tensor
    padded_constant_tensor = node.name + "_full"
    ast_block.add_call_assignment(
        target=padded_constant_tensor,
        pkg_or_obj="tensor_apis",
        func="full",
        args=[str(tuple(target_shape)), str(pad_val)],
        kwargs={"dtype": dtype},
    )

    # Create slicing
    # Generate the subscript assignment
    # target_tensor[indices] = source_tensor
    ast_block.add_subscript_assignment(
        lhs_name=padded_constant_tensor,
        rhs_name=input_tensor,
        indices=slicing_indices,
        lhs_is_indexed=True,
    )

    # Need to add another statement to make sure the rest of the graph can be
    # properly processed
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name),
        CodeGenerator.name_load(padded_constant_tensor),
    )
