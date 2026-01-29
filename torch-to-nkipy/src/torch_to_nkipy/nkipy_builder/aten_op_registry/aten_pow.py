# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.aten_op_registry.helper_functions import (
    _normalize_scalar_constant,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.pow.Scalar")
def pow_scalar(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle power operation with scalar base and tensor exponent.

    Maps PyTorch's torch.pow(scalar, exponent_tensor) to NumPy's power function.
    This computes scalar raised to the power of each element in the tensor.

    Example:
        PyTorch:
        ```
        # Compute 2 raised to each element in tensor
        result = torch.ops.aten.pow.Scalar(2.0, tensor)
        # If tensor = [1, 2, 3], result = [2.0, 4.0, 8.0]
        ```

        NumPy equivalent:
        ```
        result = np.power(2.0, tensor)
        ```

    Args:
        node: The FX node representing the power operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    base_scalar = node.args[0]  # Scalar value
    exponent = node.args[1].name  # Tensor

    # Special handling for special constants (inf, -inf, etc.)
    base_str = _normalize_scalar_constant(base_scalar)

    # Get the expected dtype
    dtype = get_dtype_from_fx_node(node)
    numpy_dtype = torch_to_numpy_dtype_str(dtype)

    # Add the power operation
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="power",
        args=[base_str, exponent],
        kwargs={"dtype": numpy_dtype},
    )


@AtenOpRegistry.register("torch.ops.aten.pow.Tensor_Tensor")
def pow_tensor_tensor(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle element-wise power operation between two tensors.

    Maps PyTorch's torch.pow(input, exponent) to NumPy's power function.

    Args:
        node: The FX node representing the power operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    base = node.args[0].name
    exponent = node.args[1].name

    # Get the expected dtype
    dtype = get_dtype_from_fx_node(node)
    numpy_dtype = torch_to_numpy_dtype_str(dtype)

    # Add the power operation
    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="power",
        args=[base, exponent],
        kwargs={"dtype": numpy_dtype},
    )
