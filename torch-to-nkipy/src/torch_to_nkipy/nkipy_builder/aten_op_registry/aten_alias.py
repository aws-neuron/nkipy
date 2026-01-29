# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import CodeGenerator, ComputationNode


@AtenOpRegistry.register("torch.ops.aten.alias.default")
def alias_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Args:
        node: The FX node representing the copy operation
        computation_node: The ComputationNode to update with alias info
    """
    # We simply do an assignment since the inplace update has been handled by
    # torch compile pass
    input_tensor = node.args[0].name

    ast_block = computation_node.ast_code_block
    ast_block.add_assignment(
        CodeGenerator.name_store(node.name), CodeGenerator.name_load(input_tensor)
    )
