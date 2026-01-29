# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.aten_op_registry.helper_functions import (
    _normalize_scalar_constant,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode


@AtenOpRegistry.register("torch.ops.aten.scatter.value")
def scatter_value(node: fx.Node, computation_node: ComputationNode) -> None:
    # FIXME: the current scatter is implemented using np.put_along_axis
    """
    Handle scatter.value operation using np.put_along_axis.

    Maps PyTorch's torch.scatter(input, dim, index, value) to NumPy's
    put_along_axis function, which avoids multi-dimensional advanced indexing.

    Example:
        PyTorch: result = torch.scatter(input, 1, index, -1.0)
        NumPy:   result = np.copy(input)
                 np.put_along_axis(result, index, -1.0, axis=1)

    Args:
        node: The FX node representing the scatter operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    input_tensor = node.args[0].name
    dim = node.args[1]
    index = node.args[2].name

    # Handle scalar value (normalize for special constants like inf)
    value_expr = _normalize_scalar_constant(node.args[3])

    # Create a copy of the input tensor
    temp_vars = TempVarGenerator(node.name)
    result_var = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=result_var, func_name="copy", args=[input_tensor]
    )

    # Use put_along_axis to update values
    put_along_axis_args = [result_var, index, value_expr]
    put_along_axis_kwargs = {"axis": str(dim)}

    # Add the put_along_axis operation
    ast_block.add_numpy_call_assignment(
        target=temp_vars.next(),
        func_name="put_along_axis",
        args=put_along_axis_args,
        kwargs=put_along_axis_kwargs,
    )

    # Assign the result to the output name
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(result_var)
    )
