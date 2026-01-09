# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary operations: add, subtract, multiply, divide, etc."""

import numpy as np

from nkipy.core.backend.hlo import (
    as_hlo_tensor,
    broadcast_operands_hlo,
    find_common_type_hlo,
    get_hlo_context,
)
from nkipy.core.ops._registry import Op
from nkipy.core.tensor import NKIPyTensorRef

# =============================================================================
# HLO Implementation Helpers
# =============================================================================


def _build_binary_hlo(x, y, np_op, out=None, dtype=None):
    """Build a binary HLO operation with broadcasting

    Args:
        x: First operand
        y: Second operand
        np_op: NumPy operation
        out: Unused (for API compatibility with IR)
        dtype: Optional output dtype to cast result to
    """
    ctx = get_hlo_context()

    promoted_dtype = find_common_type_hlo(x, y)

    # If dtype parameter is provided, use it as the output dtype
    if dtype is not None:
        output_dtype = np.dtype(dtype)
    else:
        output_dtype = promoted_dtype

    # Convert to HLOTensor
    x = (
        x.backend_tensor
        if isinstance(x, NKIPyTensorRef)
        else as_hlo_tensor(ctx, x, promoted_dtype)
    )
    y = (
        y.backend_tensor
        if isinstance(y, NKIPyTensorRef)
        else as_hlo_tensor(ctx, y, promoted_dtype)
    )

    # Map numpy ops to HLO opcodes
    op_map = {
        np.add: "add",
        np.subtract: "subtract",
        np.multiply: "multiply",
        np.divide: "divide",
        np.power: "power",
        np.maximum: "maximum",
        np.minimum: "minimum",
        np.bitwise_and: "and",
        np.bitwise_or: "or",
        np.bitwise_xor: "xor",
        np.logical_and: "and",
        np.logical_or: "or",
        np.logical_xor: "xor",
    }

    hlo_op = op_map.get(
        np_op, np_op.__name__ if hasattr(np_op, "__name__") else str(np_op)
    )

    # Broadcast operands to compatible shapes
    x_broadcast, y_broadcast = broadcast_operands_hlo(ctx, x, y)

    # Explicit type promotion to match IR behavior
    # This ensures operands have the correct dtype before the operation
    if output_dtype != x_broadcast.dtype:
        x_broadcast = ctx.build_op(
            "convert", [x_broadcast], x_broadcast.shape, output_dtype
        )
    if output_dtype != y_broadcast.dtype:
        y_broadcast = ctx.build_op(
            "convert", [y_broadcast], y_broadcast.shape, output_dtype
        )

    result_tensor = ctx.build_op(
        hlo_op, [x_broadcast, y_broadcast], x_broadcast.shape, output_dtype
    )

    return NKIPyTensorRef(result_tensor)


def _build_comparison_hlo(x, y, np_op, out=None, dtype=None):
    """Build a comparison HLO operation"""
    ctx = get_hlo_context()

    promoted_dtype = find_common_type_hlo(x, y)

    # Convert to HLOTensor
    x = (
        x.backend_tensor
        if isinstance(x, NKIPyTensorRef)
        else as_hlo_tensor(ctx, x, promoted_dtype)
    )
    y = (
        y.backend_tensor
        if isinstance(y, NKIPyTensorRef)
        else as_hlo_tensor(ctx, y, promoted_dtype)
    )

    # Type promotion: convert both tensors to the promoted dtype if needed
    if x.dtype != promoted_dtype:
        x = ctx.build_op("convert", [x], x.shape, promoted_dtype)
    if y.dtype != promoted_dtype:
        y = ctx.build_op("convert", [y], y.shape, promoted_dtype)

    # Map numpy ops to HLO comparison directions
    comp_map = {
        np.equal: "EQ",
        np.not_equal: "NE",
        np.less: "LT",
        np.less_equal: "LE",
        np.greater: "GT",
        np.greater_equal: "GE",
    }

    comp_dir = comp_map.get(np_op, "EQ")

    # Broadcast operands to compatible shapes
    x_broadcast, y_broadcast = broadcast_operands_hlo(ctx, x, y)

    result_tensor = ctx.build_op(
        "compare",
        [x_broadcast, y_broadcast],
        x_broadcast.shape,
        np.bool_,
        {"comparison_direction": comp_dir},
    )

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# Factory function for simple binary ops
# -----------------------------------------------------------------------------
def _make_binary_op(name: str, np_op) -> Op:
    """Create a binary Op with IR and HLO implementations."""
    op = Op(name)

    @op.impl("hlo")
    def _impl_hlo(x, y, out=None, dtype=None):
        return _build_binary_hlo(x, y, np_op, out=out, dtype=dtype)

    return op


def _make_comparison_op(name: str, np_op) -> Op:
    """Create a comparison Op with IR and HLO implementations."""
    op = Op(name)

    @op.impl("hlo")
    def _impl_hlo(x, y, out=None, dtype=None):
        return _build_comparison_hlo(x, y, np_op, out=out, dtype=dtype)

    return op


def _build_logical_hlo(x, y, hlo_op_name, out=None, dtype=None):
    """Build a logical HLO operation.

    Logical operations first convert inputs to boolean (non-zero check),
    then apply the logical operation.

    Args:
        x: First operand
        y: Second operand
        hlo_op_name: HLO operation name ('and', 'or', 'xor')
        out: Unused (for API compatibility)
        dtype: Unused (for API compatibility)
    """
    ctx = get_hlo_context()

    # Find common type for scalars
    promoted_dtype = find_common_type_hlo(x, y)

    # Convert to HLOTensor
    x = (
        x.backend_tensor
        if isinstance(x, NKIPyTensorRef)
        else as_hlo_tensor(ctx, x, promoted_dtype)
    )
    y = (
        y.backend_tensor
        if isinstance(y, NKIPyTensorRef)
        else as_hlo_tensor(ctx, y, promoted_dtype)
    )

    # Broadcast operands to compatible shapes
    x_broadcast, y_broadcast = broadcast_operands_hlo(ctx, x, y)

    # Convert to boolean: x != 0
    zero_x = as_hlo_tensor(ctx, 0, x_broadcast.dtype)
    if x_broadcast.shape:
        zero_x = ctx.build_op(
            "broadcast",
            [zero_x],
            x_broadcast.shape,
            x_broadcast.dtype,
            {"broadcast_dimensions": []},
        )
    x_bool = ctx.build_op(
        "compare",
        [x_broadcast, zero_x],
        x_broadcast.shape,
        np.bool_,
        {"comparison_direction": "NE"},
    )

    zero_y = as_hlo_tensor(ctx, 0, y_broadcast.dtype)
    if y_broadcast.shape:
        zero_y = ctx.build_op(
            "broadcast",
            [zero_y],
            y_broadcast.shape,
            y_broadcast.dtype,
            {"broadcast_dimensions": []},
        )
    y_bool = ctx.build_op(
        "compare",
        [y_broadcast, zero_y],
        y_broadcast.shape,
        np.bool_,
        {"comparison_direction": "NE"},
    )

    # Apply logical operation on boolean values
    result_tensor = ctx.build_op(
        hlo_op_name, [x_bool, y_bool], x_broadcast.shape, np.bool_
    )

    return NKIPyTensorRef(result_tensor)


def _make_logical_op(name: str, hlo_op_name: str) -> Op:
    """Create a logical Op with HLO implementation.

    Logical operations convert inputs to boolean first, then apply the operation.
    """
    op = Op(name)

    @op.impl("hlo")
    def _impl_hlo(x, y, out=None, dtype=None):
        return _build_logical_hlo(x, y, hlo_op_name, out=out, dtype=dtype)

    return op


# -----------------------------------------------------------------------------
# Arithmetic operations
# -----------------------------------------------------------------------------
add = _make_binary_op("add", np.add)
subtract = _make_binary_op("subtract", np.subtract)
multiply = _make_binary_op("multiply", np.multiply)
divide = _make_binary_op("divide", np.divide)
power = _make_binary_op("power", np.power)
maximum = _make_binary_op("maximum", np.maximum)
minimum = _make_binary_op("minimum", np.minimum)

# -----------------------------------------------------------------------------
# Bitwise operations
# -----------------------------------------------------------------------------
bitwise_and = _make_binary_op("bitwise_and", np.bitwise_and)
bitwise_or = _make_binary_op("bitwise_or", np.bitwise_or)
bitwise_xor = _make_binary_op("bitwise_xor", np.bitwise_xor)

# -----------------------------------------------------------------------------
# Comparison operations
# -----------------------------------------------------------------------------
equal = _make_comparison_op("equal", np.equal)
not_equal = _make_comparison_op("not_equal", np.not_equal)
greater = _make_comparison_op("greater", np.greater)
greater_equal = _make_comparison_op("greater_equal", np.greater_equal)
less = _make_comparison_op("less", np.less)
less_equal = _make_comparison_op("less_equal", np.less_equal)

# -----------------------------------------------------------------------------
# Logical operations
# -----------------------------------------------------------------------------
logical_and = _make_logical_op("logical_and", "and")
logical_or = _make_logical_op("logical_or", "or")
logical_xor = _make_logical_op("logical_xor", "xor")
