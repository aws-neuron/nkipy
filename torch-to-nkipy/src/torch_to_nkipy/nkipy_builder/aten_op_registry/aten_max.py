# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.max.dim")
def max_dim(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle max operation along a dimension.

    Maps PyTorch's torch.max(input, dim) to tensor_apis.topk function with k=1.
    Returns both the maximum values and their indices along the specified dimension.

    PyTorch signature: torch.max(input, dim, keepdim=False) -> (Tensor, LongTensor)

    Args:
        node: The FX node representing the max operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 2:
        raise ValueError("max.dim requires input tensor and dimension arguments")

    # Extract arguments
    input_tensor = node.args[0].name
    dim = node.args[1]

    # Handle keepdim argument (default is False)
    keepdim = False
    if len(node.args) > 2:
        keepdim = node.args[2]

    # Use temp var generator to create variables for intermediate steps
    temp_vars = TempVarGenerator(node.name)

    # Get original input shape and calculate actual dimension index
    input_shape = get_shape_from_fx_node(node.args[0])
    ndim = len(input_shape)
    actual_dim = dim if dim >= 0 else ndim + dim
    last_dim = ndim - 1

    # Step 1: Move target dimension to last position if needed
    need_transpose = actual_dim != last_dim

    if need_transpose:
        # Create forward permutation: move actual_dim to the end
        forward_perm_var = temp_vars.next()

        # Build forward permutation list:
        # [0,1,2,...,actual_dim-1,actual_dim+1,...,ndim-1,actual_dim]
        forward_perm_elements = []
        for i in range(ndim):
            if i != actual_dim:
                forward_perm_elements.append(ast.Constant(value=i))
        forward_perm_elements.append(ast.Constant(value=actual_dim))

        ast_block.add_assignment(
            CodeGenerator.name_store(forward_perm_var),
            ast.Tuple(elts=forward_perm_elements, ctx=ast.Load()),
        )

        # Create inverse permutation for later use
        inverse_perm_var = temp_vars.next()

        # Build inverse permutation: where each dimension should go back to
        inverse_perm_elements = [ast.Constant(value=0)] * ndim
        for new_pos, old_pos in enumerate(forward_perm_elements):
            if isinstance(old_pos, ast.Constant):
                inverse_perm_elements[old_pos.value] = ast.Constant(value=new_pos)

        ast_block.add_assignment(
            CodeGenerator.name_store(inverse_perm_var),
            ast.Tuple(elts=inverse_perm_elements, ctx=ast.Load()),
        )

        # Transpose input to move target dimension to last position
        transposed_input_var = temp_vars.next()

        ast_block.add_numpy_call_assignment(
            target=transposed_input_var,
            func_name="transpose",
            args=[input_tensor, forward_perm_var],
        )

        topk_input = transposed_input_var
        topk_axis = last_dim  # Now we always use the last dimension
    else:
        topk_input = input_tensor
        topk_axis = actual_dim

    # Step 2: Use topk with k=1 to get max values and indices
    topk_result_var = temp_vars.next()

    # Set up kwargs for topk (always use last dimension after potential transpose)
    topk_kwargs = {
        "k": "1",
        "axis": str(topk_axis),
        "is_ascend": "False",  # We want the largest value (max)
    }

    ast_block.add_call_assignment(
        target=topk_result_var,
        pkg_or_obj="tensor_apis",
        func="topk",
        args=[topk_input],
        kwargs=topk_kwargs,
    )

    # Step 3: Extract values and indices from topk result
    max_values_var = temp_vars.next()
    max_indices_var = temp_vars.next()

    # topk returns (values, indices), so extract each
    ast_block.add_assignment(
        CodeGenerator.name_store(max_values_var),
        CodeGenerator.name_load(f"{topk_result_var}[0]"),
    )

    ast_block.add_assignment(
        CodeGenerator.name_store(max_indices_var),
        CodeGenerator.name_load(f"{topk_result_var}[1]"),
    )

    # Step 4: Move dimensions back to original positions if we transposed
    if need_transpose:
        # Transpose back using inverse permutation
        detransposed_values_var = temp_vars.next()
        detransposed_indices_var = temp_vars.next()

        ast_block.add_numpy_call_assignment(
            target=detransposed_values_var,
            func_name="transpose",
            args=[max_values_var, inverse_perm_var],
        )

        ast_block.add_numpy_call_assignment(
            target=detransposed_indices_var,
            func_name="transpose",
            args=[max_indices_var, inverse_perm_var],
        )

        max_values_var = detransposed_values_var
        max_indices_var = detransposed_indices_var

    # Step 5: Handle keepdim=False by reshaping to remove the k dimension
    if not keepdim:
        # Create new shape tuple excluding the specified dimension
        new_shape_elements = []
        for i, size in enumerate(input_shape):
            if i != actual_dim:
                new_shape_elements.append(ast.Constant(value=size))

        new_shape_var = temp_vars.next()
        ast_block.add_assignment(
            CodeGenerator.name_store(new_shape_var),
            ast.Tuple(elts=new_shape_elements, ctx=ast.Load()),
        )

        # Reshape instead of squeeze
        reshaped_values_var = temp_vars.next()
        reshaped_indices_var = temp_vars.next()

        ast_block.add_numpy_call_assignment(
            target=reshaped_values_var,
            func_name="reshape",
            args=[max_values_var, new_shape_var],
        )

        ast_block.add_numpy_call_assignment(
            target=reshaped_indices_var,
            func_name="reshape",
            args=[max_indices_var, new_shape_var],
        )

        final_values_var = reshaped_values_var
        final_indices_var = reshaped_indices_var
    else:
        final_values_var = max_values_var
        final_indices_var = max_indices_var

    # Step 6: Create tuple result using AST
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name),
        ast.Call(
            func=ast.Name(id="tuple", ctx=ast.Load()),
            args=[
                ast.List(
                    elts=[
                        ast.Name(id=final_values_var, ctx=ast.Load()),
                        ast.Name(id=final_indices_var, ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        ),
    )
