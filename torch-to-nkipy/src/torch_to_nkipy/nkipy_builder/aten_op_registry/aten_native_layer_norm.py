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


@AtenOpRegistry.register("torch.ops.aten.native_layer_norm.default")
def native_layer_norm_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle layer normalization operation.
    Returns tuple of (normalized_output, mean, inv_std)
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0].name
    normalized_shape = node.args[1]
    weight = node.args[2].name if node.args[2] is not None else None
    bias = node.args[3].name if node.args[3] is not None else None
    eps = node.args[4]

    # Use temp vars for intermediate calculations
    temp_vars = TempVarGenerator(node.name)

    # Step 1: Calculate mean along normalized dimensions
    # Instead of using mean directly, use sum and divide
    sum_result = temp_vars.next()
    norm_axes = tuple(range(-len(normalized_shape), 0))
    ast_block.add_numpy_call_assignment(
        target=sum_result,
        func_name="sum",
        args=[input_tensor],
        kwargs={"axis": str(norm_axes), "keepdims": "True"},
    )

    # Calculate number of elements
    num_elements = normalized_shape[0]  # Assuming last dimension size

    mean = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=mean, func_name="divide", args=[sum_result, str(num_elements)]
    )

    # Step 2: Calculate variance
    centered = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=centered, func_name="subtract", args=[input_tensor, mean]
    )

    # Calculate squared values
    squared = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=squared, func_name="power", args=[centered, "2"]
    )

    # Calculate variance using sum instead of mean
    sum_squared = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=sum_squared,
        func_name="sum",
        args=[squared],
        kwargs={"axis": str(norm_axes), "keepdims": "True"},
    )

    variance = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=variance, func_name="divide", args=[sum_squared, str(num_elements)]
    )

    # Calculate variance + eps
    variance_eps = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=variance_eps, func_name="add", args=[variance, str(eps)]
    )

    # Calculate sqrt(variance + eps)
    std = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=std, func_name="sqrt", args=[variance_eps]
    )

    # Calculate inv_std = 1/sqrt(variance + eps)
    inv_std = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=inv_std, func_name="divide", args=["1", std]
    )

    # Step 3: Normalize
    normalized = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=normalized, func_name="multiply", args=[centered, inv_std]
    )

    # Step 4: Apply affine transformation if weight and bias are provided
    result = temp_vars.next()
    if weight and bias:
        weighted = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=weighted, func_name="multiply", args=[normalized, weight]
        )
        ast_block.add_numpy_call_assignment(
            target=result, func_name="add", args=[weighted, bias]
        )
    elif weight:
        ast_block.add_numpy_call_assignment(
            target=result, func_name="multiply", args=[normalized, weight]
        )
    elif bias:
        ast_block.add_numpy_call_assignment(
            target=result, func_name="add", args=[normalized, bias]
        )
    else:
        ast_block.add_assignment(
            CodeGenerator.name_store(result), CodeGenerator.name_load(normalized)
        )

    # Get input shape for reshaping
    input_shape = get_shape_from_fx_node(node.args[0])

    # Create temporary variable for the normalized output
    normalized_output = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=normalized_output,
        func_name="reshape",
        args=[result, str(tuple(input_shape))],
    )

    # Create tuple using standard Python tuple
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name),
        ast.Call(
            func=ast.Name(id="tuple", ctx=ast.Load()),
            args=[
                ast.List(
                    elts=[
                        ast.Name(id=normalized_output, ctx=ast.Load()),
                        ast.Name(id=mean, ctx=ast.Load()),
                        ast.Name(id=inv_std, ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        ),
    )
