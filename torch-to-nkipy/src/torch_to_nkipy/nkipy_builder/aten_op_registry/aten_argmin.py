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


@AtenOpRegistry.register("torch.ops.aten.argmin.default")
def argmin_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle argmin operation along a dimension.

    Maps PyTorch's torch.argmin(input, dim, keepdim) to tensor_apis.topk function
    with k=1.
    Uses negation (-1 multiplication) with is_ascend=False to find minimum values.
    Returns the indices of minimum values along the specified dimension.

    PyTorch signature: torch.argmin(input, dim=None, keepdim=False) -> LongTensor

    Args:
        node: The FX node representing the argmin operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 1:
        raise ValueError("argmin.default requires input tensor argument")

    # Extract arguments
    input_tensor = node.args[0].name
    dim = node.args[1] if len(node.args) > 1 else None
    keepdim = node.args[2] if len(node.args) > 2 else False

    # Use temp var generator to create variables for intermediate steps
    temp_vars = TempVarGenerator(node.name)

    # Handle case where dim is None (flatten and find global argmin)
    if dim is None:
        # Get input shape to calculate total number of elements
        input_shape = get_shape_from_fx_node(node.args[0])

        # Calculate total number of elements
        total_elements = 1
        for size in input_shape:
            total_elements *= size

        # Create flattened shape tuple (-1,) or (total_elements,)
        flattened_shape_var = temp_vars.next()
        ast_block.add_assignment(
            CodeGenerator.name_store(flattened_shape_var),
            ast.Tuple(elts=[ast.Constant(value=total_elements)], ctx=ast.Load()),
        )

        # Reshape the tensor to 1D instead of using flatten
        flattened_var = temp_vars.next()

        ast_block.add_numpy_call_assignment(
            target=flattened_var,
            func_name="reshape",
            args=[input_tensor, flattened_shape_var],
        )

        # Negate the flattened tensor to convert min to max problem
        negated_var = temp_vars.next()

        ast_block.add_assignment(
            CodeGenerator.name_store(negated_var),
            ast.BinOp(
                left=ast.Name(id=flattened_var, ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Constant(value=-1),
            ),
        )

        # Use topk on the negated tensor (dimension 0 after reshape to 1D)
        topk_result_var = temp_vars.next()

        topk_kwargs = {
            "k": "1",
            "axis": "0",  # Only dimension after reshape to 1D
            "is_ascend": "False",  # We want the largest value
            # (which corresponds to smallest original)
        }

        ast_block.add_call_assignment(
            target=topk_result_var,
            pkg_or_obj="tensor_apis",
            func="topk",
            args=[negated_var],
            kwargs=topk_kwargs,
        )

        # Extract indices from topk result (we don't need values for argmin)
        indices_var = temp_vars.next()

        ast_block.add_assignment(
            CodeGenerator.name_store(indices_var),
            CodeGenerator.name_load(f"{topk_result_var}[1]"),
        )

        # For global argmin, reshape to scalar (remove the k and axis dimensions)
        scalar_shape_var = temp_vars.next()
        ast_block.add_assignment(
            CodeGenerator.name_store(scalar_shape_var),
            ast.Tuple(elts=[], ctx=ast.Load()),  # Empty tuple for scalar shape
        )

        final_result_var = temp_vars.next()

        ast_block.add_numpy_call_assignment(
            target=final_result_var,
            func_name="reshape",
            args=[indices_var, scalar_shape_var],
        )

        final_indices_var = final_result_var

    else:
        # Handle case where dim is specified - similar to min.dim logic
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

            topk_input_pre_negate = transposed_input_var
            topk_axis = last_dim  # Now we always use the last dimension
        else:
            topk_input_pre_negate = input_tensor
            topk_axis = actual_dim

        # Step 1.5: Negate the tensor to convert min to max problem
        negated_input_var = temp_vars.next()

        ast_block.add_assignment(
            CodeGenerator.name_store(negated_input_var),
            ast.BinOp(
                left=ast.Name(id=topk_input_pre_negate, ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Constant(value=-1),
            ),
        )

        topk_input = negated_input_var

        # Step 2: Use topk with k=1 to get argmin indices
        topk_result_var = temp_vars.next()

        # Set up kwargs for topk (always use last dimension after potential transpose)
        topk_kwargs = {
            "k": "1",
            "axis": str(topk_axis),
            "is_ascend": "False",  # We want the largest value
            # (which corresponds to smallest original)
        }

        ast_block.add_call_assignment(
            target=topk_result_var,
            pkg_or_obj="tensor_apis",
            func="topk",
            args=[topk_input],
            kwargs=topk_kwargs,
        )

        # Step 3: Extract indices from topk result (we only need indices for argmin)
        argmin_indices_var = temp_vars.next()

        # topk returns (values, indices), so extract indices
        ast_block.add_assignment(
            CodeGenerator.name_store(argmin_indices_var),
            CodeGenerator.name_load(f"{topk_result_var}[1]"),
        )

        # Step 4: Move dimensions back to original positions if we transposed
        if need_transpose:
            # Transpose back using inverse permutation
            detransposed_indices_var = temp_vars.next()

            ast_block.add_numpy_call_assignment(
                target=detransposed_indices_var,
                func_name="transpose",
                args=[argmin_indices_var, inverse_perm_var],
            )

            argmin_indices_var = detransposed_indices_var

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

            # Reshape to remove the reduced dimension
            reshaped_indices_var = temp_vars.next()

            ast_block.add_numpy_call_assignment(
                target=reshaped_indices_var,
                func_name="reshape",
                args=[argmin_indices_var, new_shape_var],
            )

            final_indices_var = reshaped_indices_var
        else:
            final_indices_var = argmin_indices_var

    # Final step: Assign result (only indices, unlike min.dim which returns tuple)
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name),
        CodeGenerator.name_load(final_indices_var),
    )
