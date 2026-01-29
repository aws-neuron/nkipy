# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node, get_shape_from_fx_node
from torch_to_nkipy.utils.name import NUMPY_PKG


@AtenOpRegistry.register("torch.ops.aten.where.self")
def where_self(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    torch.where(cond, x, y) → numpy.where
    • If cond.shape[1:] are all 1 (the NKIPy fast path), cast cond→uint8.
    • Otherwise just emit np.where unchanged.
    """
    ast_block = computation_node.ast_code_block
    cond_n, x_n, y_n = node.args  # fx.Node objects
    cond, x, y = cond_n.name, x_n.name, y_n.name

    # TODO: fix it
    # to make nkipy work for some shape
    cond_shape = get_shape_from_fx_node(cond_n)
    if cond_shape and len(cond_shape) > 1 and all(d == 1 for d in cond_shape[1:]):
        if get_dtype_from_fx_node(cond_n) != "torch.uint8":
            tmp = f"{node.name}_pred_uint8"
            ast_block.add_call_assignment(
                target=tmp,
                pkg_or_obj=cond,
                func="astype",
                args=[f"{NUMPY_PKG}.uint8"],
            )
            cond = tmp

    ast_block.add_numpy_call_assignment(
        target=node.name,
        func_name="where",
        args=[cond, x, y],
    )
