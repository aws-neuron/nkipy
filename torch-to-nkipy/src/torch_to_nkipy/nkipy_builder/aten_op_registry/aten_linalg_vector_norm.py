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


@AtenOpRegistry.register("torch.ops.aten.linalg_vector_norm.default")
def linalg_vector_norm_default(
    node: fx.Node, computation_node: ComputationNode
) -> None:
    """
    Handle linalg_vector_norm operation.

    Implements vector p-norm: ||x||_p
    Maps PyTorch's linalg_vector_norm to a sequence of NumPy operations.

    Args:
        node: The FX node representing the linalg_vector_norm operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    x = node.args[0].name

    # Get optional arguments with their default values
    ord_arg = node.args[1] if len(node.args) > 1 else 2
    dim_arg = node.args[2] if len(node.args) > 2 else None
    keepdim_arg = node.args[3] if len(node.args) > 3 else False

    # Get shape information at compile time
    input_shape = get_shape_from_fx_node(node.args[0])

    # Convert arguments to strings for the AST
    ord_str = str(ord_arg)
    dim_str = str(dim_arg) if dim_arg is not None else "None"
    keepdim_str = str(keepdim_arg)

    # Use temp var generator for intermediate steps
    temp_vars = TempVarGenerator(node.name)
    out_name = temp_vars.next()

    # For handling None dim - we'll add different logic paths
    output_shape = ()
    if dim_arg is None:
        # When dim is None, we flatten the tensor using shape information
        flattened_var = temp_vars.next()

        # Calculate total size from shape
        total_size = 1
        for dim_size in input_shape:
            total_size *= dim_size

        # Create the target shape tuple
        reshape_tuple = (total_size,)

        # Reshape to a 1D tensor
        ast_block.add_numpy_call_assignment(
            target=flattened_var, func_name="reshape", args=[x, str(reshape_tuple)]
        )

        input_tensor = flattened_var
        dim_str = "0"  # After flattening, we use axis 0
        keepdim_str = "True"

    else:
        input_tensor = x
        # If dim is a single integer wrapped in a list (common case like [-1]),
        # we can unwrap it for cleaner code if list has only one element
        if isinstance(dim_arg, list) and len(dim_arg) == 1:
            dim_str = str(dim_arg[0])
        else:
            dim_str = str(tuple(dim_arg))

        # Calculate output shape based on reduction dimensions
        output_shape = list(input_shape)

        # Convert dim_arg to a list of dimensions
        if isinstance(dim_arg, int):
            dims_to_reduce = [dim_arg]
        elif isinstance(dim_arg, (list, tuple)):
            dims_to_reduce = list(dim_arg)
        else:
            # Handle other cases (could be a tensor)
            dims_to_reduce = []

        # Handle negative indices
        ndim = len(input_shape)
        dims_to_reduce = [(d if d >= 0 else d + ndim) for d in dims_to_reduce]

        if keepdim_arg:
            # Keep dimensions but set reduced dimensions to 1
            for d in dims_to_reduce:
                output_shape[d] = 1
        else:
            # Remove reduced dimensions (from highest to lowest to avoid index shifts)
            for d in sorted(dims_to_reduce, reverse=True):
                output_shape.pop(d)

        output_shape = tuple(output_shape)

    # Special handling for different norm orders
    if ord_arg == 2:  # Most common case: L2 norm (Euclidean)
        # Step 1: Square the values
        squared_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=squared_var, func_name="square", args=[input_tensor]
        )

        # Step 2: Sum the squares along specified dimension
        sum_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=sum_var,
            func_name="sum",
            args=[squared_var],
            kwargs={"axis": dim_str, "keepdims": keepdim_str},
        )

        # Step 3: Take the square root
        result_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=result_var, func_name="sqrt", args=[sum_var]
        )

        # NOTE: returning np.array(scalar) can cause error in NKIPy if scalar is not
        # actual number
        ast_block.add_numpy_call_assignment(
            target=out_name, func_name="reshape", args=[result_var, str(output_shape)]
        )

    elif ord_arg == 1:  # L1 norm (sum of absolute values)
        # Step 1: Take absolute values
        abs_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=abs_var, func_name="abs", args=[input_tensor]
        )

        # Step 2: Sum along specified dimension
        result_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=result_var,
            func_name="sum",
            args=[abs_var],
            kwargs={"axis": dim_str, "keepdims": keepdim_str},
        )

        ast_block.add_numpy_call_assignment(
            target=out_name, func_name="reshape", args=[result_var, str(output_shape)]
        )

    elif ord_arg == float("inf") or ord_str == "float('inf')":
        # Infinity norm (maximum absolute value)
        abs_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=abs_var, func_name="abs", args=[input_tensor]
        )

        result_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=result_var,
            func_name="max",
            args=[abs_var],
            kwargs={"axis": dim_str, "keepdims": keepdim_str},
        )

        ast_block.add_numpy_call_assignment(
            target=out_name, func_name="reshape", args=[result_var, str(output_shape)]
        )

    elif ord_arg == float("-inf") or ord_str == "float('-inf')":
        # -Infinity norm (minimum absolute value)
        abs_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=abs_var, func_name="abs", args=[input_tensor]
        )

        result_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=result_var,
            func_name="min",
            args=[abs_var],
            kwargs={"axis": dim_str, "keepdims": keepdim_str},
        )

        ast_block.add_numpy_call_assignment(
            target=out_name, func_name="reshape", args=[result_var, str(output_shape)]
        )

    else:
        # General p-norm
        # Step 1: Take absolute values
        abs_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=abs_var, func_name="abs", args=[input_tensor]
        )

        # Step 2: Raise to power 'ord'
        power_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=power_var,
            func_name="power",
            args=[abs_var, ord_str],
        )

        # Step 3: Sum along specified dimension
        sum_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=sum_var,
            func_name="sum",
            args=[power_var],
            kwargs={"axis": dim_str, "keepdims": keepdim_str},
        )

        # Step 4: Take the 1/ord power
        power_inv = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=power_inv, func_name="divide", args=["1", ord_str]
        )

        result_var = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=result_var,
            func_name="power",
            args=[sum_var, power_inv],
        )

        ast_block.add_numpy_call_assignment(
            target=out_name, func_name="reshape", args=[result_var, str(output_shape)]
        )

    # Handle dtype casting if specified
    if "dtype" in node.kwargs:
        torch_dtype = node.kwargs["dtype"]
        numpy_dtype = torch_to_numpy_dtype_str(torch_dtype)

        # Add the astype operation
        ast_block.add_call_assignment(
            target=node.name,
            pkg_or_obj=out_name,
            func="astype",
            args=[numpy_dtype],
        )
    else:
        ast_block.add_assignment(
            CodeGenerator.name_store(node.name), CodeGenerator.name_load(out_name)
        )
