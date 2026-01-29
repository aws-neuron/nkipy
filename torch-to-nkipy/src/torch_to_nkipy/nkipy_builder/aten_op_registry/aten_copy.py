# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node, get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.copy_.default")
def copy__default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle in-place copy operation.

    Maps PyTorch's tensor.copy_(source) to an aliasing relationship.
    Instead of generating code, this sets up alias information to be
    handled by the NKIPy runtime.

    Args:
        node: The FX node representing the copy operation
        computation_node: The ComputationNode to update with alias info
    """
    # Extract arguments
    source = node.args[1].name
    target = node.args[0].name

    # Set alias information instead of generating code
    computation_node.set_alias_info(source, target)

    ast_block = computation_node.ast_code_block
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(source)
    )


@AtenOpRegistry.register("torch.ops.aten.copy.default")
def copy_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle copy operation between tensors.

    Maps PyTorch's torch.copy(destination, source) to NumPy operations.
    Returns a tensor with destination's metadata but source's data.

    Handles:
    - Different shapes through broadcasting (only when needed)
    - Different data types through type conversion (only when needed)
    - Preserves destination's size/shape

    Example:
        PyTorch:
        ```
        # If select_110 contains [1,2,3] and select_106 contains [4,5,6]
        result = torch.ops.aten.copy.default(select_110, select_106)
        # result now contains [4,5,6]
        ```

    Args:
        node: The FX node representing the copy operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    dest = node.args[0]
    source = node.args[1]

    # Get shapes and dtypes
    dest_shape = get_shape_from_fx_node(dest)
    source_shape = get_shape_from_fx_node(source)
    dest_dtype = get_dtype_from_fx_node(dest)
    source_dtype = get_dtype_from_fx_node(source)

    # Variable to track current tensor in the processing chain
    current_var = source.name
    temp_vars = TempVarGenerator(node.name)

    # Step 1: Handle broadcasting if shapes differ
    if dest_shape != source_shape:
        broadcast_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=broadcast_var,
            func_name="broadcast_to",
            args=[current_var, str(tuple(dest_shape))],
        )
        current_var = broadcast_var

    # Step 2: Handle dtype conversion if needed
    if dest_dtype != source_dtype:
        dtype_var = temp_vars.next()
        ast_block.add_call_assignment(
            target=dtype_var,
            pkg_or_obj=current_var,
            func="astype",
            args=[torch_to_numpy_dtype_str(dest_dtype)],
        )
        current_var = dtype_var

    # Final step: Assign the result
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(current_var)
    )
