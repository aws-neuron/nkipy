# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.aten_op_registry.helper_functions import (
    _create_comparison_handler,
)
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_dtype_from_fx_node

# Register basic element-wise operations
# Use the batch registration utility for simple mappings
AtenOpRegistry.batch_register(
    {
        "torch.ops.aten.add.Scalar": "add",
        "torch.ops.aten.logical_and.default": "logical_and",
        "torch.ops.aten.logical_or.default": "logical_or",
        "torch.ops.aten.logical_xor.default": "logical_xor",
        "torch.ops.aten.bmm.default": "matmul",
        "torch.ops.aten.cos.default": "cos",
        "torch.ops.aten.div.Tensor": "divide",
        "torch.ops.aten.logical_not.default": "logical_not",
        "torch.ops.aten.mm.default": "matmul",
        "torch.ops.aten.mul.Scalar": "multiply",
        "torch.ops.aten.mul.Tensor": "multiply",
        "torch.ops.aten.neg.default": "negative",
        "torch.ops.aten.permute.default": "transpose",
        "torch.ops.aten.pow.Tensor_Scalar": "power",
        "torch.ops.aten.sin.default": "sin",
        "torch.ops.aten.tanh.default": "tanh",
        "torch.ops.aten.log.default": "log",
        "torch.ops.aten.minimum.default": "minimum",
        "torch.ops.aten.sub.Tensor": "subtract",
        "torch.ops.aten.unsqueeze.default": "expand_dims",
        "torch.ops.aten.view.default": "reshape",
        "torch.ops.aten.exp.default": "exp",
        "torch.ops.aten.floor.default": "floor",
        "torch.ops.aten.sqrt.default": "sqrt",
    }
)


# ============================================================================
# COMPARISON OPERATIONS
# ============================================================================
# Unified implementation of PyTorch comparison operations mapped to NumPy.
# Covers both tensor-tensor and tensor-scalar comparisons with consistent
# handling of special constants (inf, -inf, nan).

# Register tensor-tensor comparison operations (simple direct mappings)
AtenOpRegistry.batch_register(
    {
        "torch.ops.aten.eq.Tensor": "equal",
        "torch.ops.aten.ne.Tensor": "not_equal",
        "torch.ops.aten.gt.Tensor": "greater",
        "torch.ops.aten.ge.Tensor": "greater_equal",
        "torch.ops.aten.lt.Tensor": "less",
        "torch.ops.aten.le.Tensor": "less_equal",
    }
)

# Register tensor-scalar comparison operations (need special constant handling)
_scalar_comparison_ops = {
    "torch.ops.aten.eq.Scalar": "equal",
    "torch.ops.aten.ne.Scalar": "not_equal",
    "torch.ops.aten.gt.Scalar": "greater",
    "torch.ops.aten.ge.Scalar": "greater_equal",
    "torch.ops.aten.lt.Scalar": "less",
    "torch.ops.aten.le.Scalar": "less_equal",
}

for op_name, numpy_func in _scalar_comparison_ops.items():
    handler = _create_comparison_handler(numpy_func, is_scalar=True)
    AtenOpRegistry.register(op_name)(handler)


def _register_bitwise_binary(aten_name: str, numpy_bitwise: str, numpy_logical: str):
    """
    Create a handler for binary bitwise ops that:
      • asserts both inputs have identical dtype
      • if dtype == torch.bool, lowers to the logical op
      • otherwise uses the numpy bitwise op
    """

    @AtenOpRegistry.register(aten_name)
    def _handler(node: fx.Node, computation_node: ComputationNode) -> None:
        ast_block = computation_node.ast_code_block

        # Expect exactly two tensor args
        a_n, b_n = node.args[0], node.args[1]
        a_name, b_name = a_n.name, b_n.name

        dt_a = get_dtype_from_fx_node(a_n)
        dt_b = get_dtype_from_fx_node(b_n)

        if dt_a != dt_b:
            raise NotImplementedError(f"{aten_name}: dtype mismatch: {dt_a} vs {dt_b}")

        func = numpy_logical if dt_a == torch.bool else numpy_bitwise
        ast_block.add_numpy_call_assignment(
            target=node.name, func_name=func, args=[a_name, b_name]
        )

    return _handler


_register_bitwise_binary(
    "torch.ops.aten.bitwise_and.Tensor", "bitwise_and", "logical_and"
)
_register_bitwise_binary("torch.ops.aten.bitwise_or.Tensor", "bitwise_or", "logical_or")
_register_bitwise_binary(
    "torch.ops.aten.bitwise_xor.Tensor", "bitwise_xor", "logical_xor"
)
