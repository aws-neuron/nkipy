# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import (
    AtenOpRegistry,
    TempVarGenerator,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode
from torch_to_nkipy.utils.dtype import torch_to_numpy_dtype_str


@AtenOpRegistry.register("torch.ops.aten.empty_permuted.default")
def empty_permuted_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle empty_permuted operation.

    Maps torch.empty_permuted to NumPy's empty followed by transpose if needed.
    This creates an empty tensor with the specified shape and memory layout according to
    the dimension permutation.

    Example:
        PyTorch:
        ```
        # Creates a tensor with shape [2, 3, 4] but memory layout optimized for
        # dimension access in order [2, 0, 1]
        x = torch.ops.aten.empty_permuted.default([2, 3, 4], [2, 0, 1],
                                                 dtype=torch.float32)
        ```

        Our NumPy implementation:
        ```
        # First create empty array
        temp = np.zeros([2, 3, 4], dtype=np.float32)
        # Then transpose to get the right memory layout
        x = np.transpose(temp, [2, 0, 1])
        ```

    Args:
        node: The FX node representing the empty_permuted operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    shape = node.args[0]
    permutation = node.args[1]

    # Get dtype from kwargs
    torch_dtype = node.kwargs.get("dtype")
    numpy_dtype = torch_to_numpy_dtype_str(torch_dtype)

    # Create temp variable generator for intermediate results if needed
    temp_vars = TempVarGenerator(node.name)
    empty_var = temp_vars.next()

    # Create the empty tensor with the specified shape and dtype
    ast_block.add_numpy_call_assignment(
        target=empty_var,
        func_name="zeros",
        args=[str(shape)],
        kwargs={"dtype": numpy_dtype},
    )

    # Check if permutation is not the identity permutation
    # For example, for 3D tensor, identity would be [0, 1, 2]
    ndim = len(shape)
    identity_perm = list(range(ndim))

    if permutation == identity_perm:
        # If it's the identity permutation, just use the empty tensor
        ast_block.add_assignment(
            CodeGenerator.name_store(node.name), CodeGenerator.name_load(empty_var)
        )
    else:
        # Otherwise, transpose to get the correct memory layout
        ast_block.add_numpy_call_assignment(
            target=node.name, func_name="transpose", args=[empty_var, str(permutation)]
        )
