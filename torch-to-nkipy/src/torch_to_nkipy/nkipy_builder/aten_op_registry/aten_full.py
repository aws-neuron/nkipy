# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.name import NUMPY_PKG


@AtenOpRegistry.register("torch.ops.aten.full.default")
def full_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle full tensor creation operation.

    Maps PyTorch's torch.full to NumPy's full function.
    Creates a tensor of given size filled with specified value.

    PyTorch signature: torch.full(size, fill_value, *, dtype=None) -> Tensor
    NumPy equivalent: np.full(shape, fill_value, dtype=dtype)

    Args:
        node: The FX node representing the full operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If required arguments are missing
    """
    ast_block = computation_node.ast_code_block

    # Validate inputs
    if len(node.args) < 2:
        raise ValueError("full.default requires size and fill_value arguments")

    # Extract arguments
    size = node.args[0]
    fill_value = node.args[1]

    # Handle dtype if provided
    dtype_arg = ""
    if "dtype" in node.kwargs:
        dtype = torch_to_numpy_dtype_str(node.kwargs["dtype"])
        dtype_arg = f", dtype={dtype}"

    # Special handling for infinity values
    if fill_value == float("inf"):
        fill_value = f"{NUMPY_PKG}.inf"
    elif fill_value == float("-inf"):
        fill_value = f"{NUMPY_PKG}.NINF"
    else:
        fill_value = str(fill_value)

    # Add the full operation
    ast_block.add_call_assignment(
        target=node.name,
        pkg_or_obj="tensor_apis",
        func="full",
        args=[str(tuple(size)), fill_value],
        kwargs={} if not dtype_arg else {"dtype": dtype},
    )
