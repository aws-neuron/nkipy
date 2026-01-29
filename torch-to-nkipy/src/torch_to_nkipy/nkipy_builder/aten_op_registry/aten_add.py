# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.add.Tensor")
def add_tensor(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle element-wise addition between two tensors with dtype enforcement.
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    left = str(node.args[0])
    right = str(node.args[1])

    # Get the expected dtype and convert to NumPy dtype
    dtype = get_dtype_from_fx_node(node)
    numpy_dtype = torch_to_numpy_dtype_str(dtype)

    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="add",
        args=[left, right],
        kwargs={"dtype": numpy_dtype},
    )
