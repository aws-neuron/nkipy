# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode


@AtenOpRegistry.register("torch.ops.aten.clone.default")
def clone_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle tensor clone operation.

    Maps PyTorch's torch.clone to NumPy's copy.

    Note:
        Raises a warning if memory_format is specified since nkipy
        cannot trace np.ascontiguousarray. This will not affect the NEFF layouts,
        we only handle ascontiguousarray in the NUMPY CPU mode.

    Args:
        node: The FX node representing the clone operation
        computation_node: The ComputationNode to add code to
    """

    ast_block = computation_node.ast_code_block

    # Extract input arguments
    inputs = [arg.name for arg in node.args]

    # Add the copy operation
    ast_block.add_numpy_call_assignment(target=node.name, func_name="copy", args=inputs)
