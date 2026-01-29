# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.reciprocal.default")
def reciprocal_default(node: fx.Node, computation_node: ComputationNode) -> None:
    # FIXME np.reciprocal is not implemented in NKIPy
    """
    Handle reciprocal operation.

    Implements torch.reciprocal(x) = 1/x for each element in the input tensor.
    Maps PyTorch's torch.reciprocal to NumPy's divide operation.

    Example:
        PyTorch:
        ```
        # Compute 1/x for each element
        result = torch.reciprocal(tensor)
        # If tensor = [2.0, 4.0, 0.5], result = [0.5, 0.25, 2.0]
        ```

        NumPy equivalent:
        ```
        result = np.divide(1, tensor)
        ```

    Args:
        node: The FX node representing the reciprocal operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract input argument
    x = node.args[0].name

    # Calculate reciprocal (1 / x)
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="divide", args=["1", x]
    )
