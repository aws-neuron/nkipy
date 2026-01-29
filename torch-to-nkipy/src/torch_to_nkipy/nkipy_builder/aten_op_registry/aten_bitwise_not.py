# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.bitwise_not.default")
def bitwise_not_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Unary bitwise not that:
      • if dtype == torch.bool → np.logical_not
      • else → np.bitwise_not
    """
    ast_block = computation_node.ast_code_block
    x_n = node.args[0]
    x_name = x_n.name
    dt = get_dtype_from_fx_node(x_n)

    func = "logical_not" if dt == torch.bool else "bitwise_not"
    ast_block.add_numpy_call_assignment(target=node.name, func_name=func, args=[x_name])
