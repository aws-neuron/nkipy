# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.name import NUMPY_PKG


@AtenOpRegistry.register("torch.ops.aten.ne.Scalar")
def ne_scalar(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle not equal operation with scalar.

    Maps PyTorch's tensor != scalar to NumPy's not_equal function.
    Performs element-wise comparison between a tensor and a scalar value.

    PyTorch signature: torch.ne(input, other) -> Tensor
    NumPy equivalent: np.not_equal(input, scalar)

    Args:
        node: The FX node representing the not equal operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 2:
        raise ValueError("ne.Scalar requires input tensor and scalar value arguments")

    # Extract arguments
    input_tensor = node.args[0].name
    scalar_value = node.args[1]

    # Special handling for infinity values
    if scalar_value == float("inf"):
        scalar_value = f"{NUMPY_PKG}.inf"
    elif scalar_value == float("-inf"):
        scalar_value = f"{NUMPY_PKG}.NINF"
    else:
        scalar_value = str(scalar_value)

    # Add the not equal operation
    temp_vars = TempVarGenerator(node.name)
    tmp_int = temp_vars.next()
    ast_block.add_numpy_call_assignment(
        target=tmp_int, func_name="not_equal", args=[input_tensor, scalar_value]
    )

    # Add the astype operation
    numpy_dtype = torch_to_numpy_dtype_str(torch.uint8)
    ast_block.add_call_assignment(
        target=node.name,
        pkg_or_obj=tmp_int,
        func="astype",
        args=[numpy_dtype],
    )
