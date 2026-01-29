# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.mean.default")
def mean_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle mean.default operation.

    Computes the mean over all elements in a tensor.
    Maps PyTorch's mean() to NumPy's mean function.

    Args:
        node: The FX node representing the mean operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    x = node.args[0].name

    # Get optional dtype argument from kwargs
    dtype_arg = node.kwargs.get("dtype", None)

    # Get shape information at compile time
    input_shape = get_shape_from_fx_node(node.args[0])

    # Use temp var generator for intermediate steps
    temp_vars = TempVarGenerator(node.name)

    # Calculate total size from shape (for flattening)
    total_size = 1
    for dim_size in input_shape:
        total_size *= dim_size

    # Flatten the tensor
    flattened_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=flattened_var, func_name="reshape", args=[x, str((total_size,))]
    )

    # Compute mean over all elements
    sum_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=sum_var,
        func_name="sum",
        args=[flattened_var],
        kwargs={"axis": "0", "keepdims": "True"},
    )

    mean_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=mean_var, func_name="divide", args=[sum_var, str(total_size)]
    )

    # Ensure output is scalar (empty shape tuple)
    out_name = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=out_name,
        func_name="reshape",
        args=[mean_var, "()"],  # Scalar output
    )

    # Handle dtype casting if specified
    if dtype_arg is not None:
        numpy_dtype = torch_to_numpy_dtype_str(dtype_arg)

        # Add the astype operation
        ast_block.add_call_assignment(
            target=node.name,
            pkg_or_obj=out_name,
            func="astype",
            args=[numpy_dtype],
        )
    else:
        # No dtype conversion needed
        ast_block.add_assignment(
            CodeGenerator.name_store(node.name), CodeGenerator.name_load(out_name)
        )


@AtenOpRegistry.register("torch.ops.aten.mean.dim")
def mean_dim(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle mean operation along specified dimensions.

    Implements torch.mean(x, dim, keepdim) as a two-step operation:
    1. Calculate sum along dimensions
    2. Divide by the size of those dimensions

    Args:
        node: The FX node representing the mean operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input arguments
    x = node.args[0].name
    dims = node.args[1]

    # Extract keepdim with default value (False in PyTorch)
    keepdims = False
    if len(node.args) > 2:
        keepdims = node.args[2]

    # Use temp var generator for intermediate operations
    temp_vars = TempVarGenerator(node.name)
    sum_var = temp_vars.next()

    # Step 1: Calculate sum along specified dimensions
    ast_block.add_numpy_call_assignment(
        target=sum_var,
        func_name="sum",
        args=[x],
        kwargs={"axis": str(tuple(dims)), "keepdims": str(keepdims)},
    )

    # Step 2: Divide by the size of those dimensions
    # For each dimension, we need the corresponding size from the shape
    if isinstance(dims, (list, tuple)) and len(dims) == 1:
        # Special case for single dimension
        shape_dim = f"{x}.shape[{dims[0]}]"
    else:
        # For multiple dimensions, calculate the product of the sizes
        shape_dim = " * ".join([f"{x}.shape[{d}]" for d in dims])

    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="divide", args=[sum_var, shape_dim]
    )
