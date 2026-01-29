# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.slice.Tensor")
def slice_tensor(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor slicing operation.
    Maps PyTorch's tensor slice operation to NumPy slicing syntax.
    For step != 1, it uses np.take with a generated index array from
    np.arange to remain compatible with compilers that do not support
    advanced slicing `[::step]`.
    PyTorch signature: torch.ops.aten.slice.Tensor(tensor, dim, start, end, step=1)
    NumPy equivalent: result = tensor[slices]
    Args:
        node: The FX node representing the slice operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block
    # Extract arguments
    source_node = node.args[0]
    source_name = source_node.name
    dim = node.args[1]
    start_orig = node.args[2]
    end_orig = node.args[3]
    step = node.args[4] if len(node.args) > 4 else 1

    source_shape = get_shape_from_fx_node(source_node)
    if source_shape is None:
        raise ValueError(
            f"Cannot determine shape for input tensor {source_name} in slice op."
        )

    # Generate clean slice strings by omitting defaults.
    if start_orig == 0:
        start_expr = ""
    else:
        # Convert negative start to positive.  Needs fix.
        if start_orig < 0:
            start_expr = f"{source_name}.shape[{dim}] + {start_orig}"
        else:
            start_expr = str(start_orig)

    if end_orig == sys.maxsize:
        end_expr = ""
    else:
        # Convert negative end to positive
        if end_orig < 0:
            end_expr = f"{source_name}.shape[{dim}] + {end_orig}"
        else:
            end_expr = str(end_orig)

    temp_vars = TempVarGenerator(node.name)

    # Base slice string (always step=1 for the first operation)
    slice_str_step1 = f"{start_expr}:{end_expr}"

    indices_step1 = [":"] * len(source_shape)
    indices_step1[dim] = slice_str_step1

    # First, get the intermediate slice with step=1. This creates a view.
    intermediate_slice_view = temp_vars.next()
    ast_block.add_subscript_assignment(
        lhs_name=intermediate_slice_view,
        indices=indices_step1,
        rhs_name=source_name,
        lhs_is_indexed=False,
    )

    # Immediately create a copy, as PyTorch's slice is not a view.
    intermediate_slice_copy = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=intermediate_slice_copy, func_name="copy", args=[intermediate_slice_view]
    )

    # If step is 1, the copied slice is the final result.
    if step == 1:
        ast_block.add_assignment(
            CodeGenerator.name_store(node.name),
            CodeGenerator.name_load(intermediate_slice_copy),
        )
    else:
        # If step is not 1, we operate on the *copied* intermediate slice.
        # Calculate the length of the intermediate slice's dimension statically
        dim_size = source_shape[dim]
        # Normalize start/end to calculate slice length accurately
        start, end = start_orig, end_orig
        if start < 0:
            start += dim_size
        if end == sys.maxsize or end > dim_size:
            end = dim_size
        if end < 0:
            end += dim_size
        slice_len = len(range(start, end, 1))

        # Generate an array of indices, e.g., [0, 2, 4, ...]
        indices_arr_name = temp_vars.next()
        ast_block.add_numpy_call_assignment(
            target=indices_arr_name,
            func_name="arange",
            args=["0", str(slice_len), str(step)],
        )

        # Use take to select elements from the copied slice.
        ast_block.add_numpy_call_assignment(
            target=node.name,
            func_name="take",
            args=[intermediate_slice_copy, indices_arr_name],
            kwargs={"axis": str(dim)},
        )
