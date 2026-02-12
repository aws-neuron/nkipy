# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.aten_op_registry.helper_functions import (
    _normalize_scalar_constant,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str


@AtenOpRegistry.register("torch.ops.aten.scalar_tensor.default")
def scalar_tensor_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle scalar tensor creation.

    Maps PyTorch's torch.scalar_tensor() to NumPy's array() or direct type conversion
    depending on the dtype.

    Args:
        node: The FX node representing the scalar_tensor operation
        computation_node: The ComputationNode to add code to

    Raises:
        ValueError: If dtype is not provided in kwargs
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments and dtype
    value = node.args[0]

    torch_dtype = node.kwargs.get("dtype")
    if torch_dtype is None:
        # Infer dtype from the node's output metadata (e.g. backward graphs
        # may emit scalar_tensor without an explicit dtype kwarg).
        meta_val = node.meta.get("val")
        if meta_val is not None:
            torch_dtype = meta_val.dtype
    if torch_dtype is None:
        raise ValueError("scalar_tensor_default: no torch dtype found")
    numpy_dtype = torch_to_numpy_dtype_str(torch_dtype)

    # Special handling for constants
    value = _normalize_scalar_constant(value)

    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="array",
        args=[value],
        kwargs={"dtype": numpy_dtype},
    )
