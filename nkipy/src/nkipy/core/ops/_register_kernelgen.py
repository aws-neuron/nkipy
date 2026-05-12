# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register kernelgen backend implementations for all ops.

Called lazily the first time the kernelgen backend is activated, so MLIR
imports only happen when needed.

Composed ops (floor_divide, tan, rint, etc.) use ``composed_impl`` on the
Op itself and need no per-backend registration — they dispatch through
other ops that have kernelgen primitives registered.
"""

_registered = False


def register_all_kernelgen_impls():
    global _registered
    if _registered:
        return
    _registered = True

    from nkipy.core.ops import _kernelgen_impls as kernelgen_impls

    # --- Binary ops (primitives) ---
    from nkipy.core.ops.binary import (
        add, subtract, multiply, divide, power, maximum, minimum,
        equal, not_equal, greater, greater_equal, less, less_equal,
        bitwise_and, bitwise_or, bitwise_xor,
    )
    add.impl("kernelgen")(kernelgen_impls.add)
    subtract.impl("kernelgen")(kernelgen_impls.subtract)
    multiply.impl("kernelgen")(kernelgen_impls.multiply)
    divide.impl("kernelgen")(kernelgen_impls.divide)
    power.impl("kernelgen")(kernelgen_impls.power)
    maximum.impl("kernelgen")(kernelgen_impls.maximum)
    minimum.impl("kernelgen")(kernelgen_impls.minimum)
    equal.impl("kernelgen")(kernelgen_impls.equal)
    not_equal.impl("kernelgen")(kernelgen_impls.not_equal)
    greater.impl("kernelgen")(kernelgen_impls.greater)
    greater_equal.impl("kernelgen")(kernelgen_impls.greater_equal)
    less.impl("kernelgen")(kernelgen_impls.less)
    less_equal.impl("kernelgen")(kernelgen_impls.less_equal)
    bitwise_and.impl("kernelgen")(kernelgen_impls.bitwise_and)
    bitwise_or.impl("kernelgen")(kernelgen_impls.bitwise_or)
    bitwise_xor.impl("kernelgen")(kernelgen_impls.bitwise_xor)

    # --- Unary ops (primitives) ---
    from nkipy.core.ops.unary import (
        abs, exp, log, sqrt, sin, cos, tanh, ceil, floor, sign,
        negative, reciprocal, square, logical_not,
    )
    exp.impl("kernelgen")(kernelgen_impls.exp)
    log.impl("kernelgen")(kernelgen_impls.log)
    sqrt.impl("kernelgen")(kernelgen_impls.sqrt)
    tanh.impl("kernelgen")(kernelgen_impls.tanh)
    sin.impl("kernelgen")(kernelgen_impls.sin)
    cos.impl("kernelgen")(kernelgen_impls.cos)
    sign.impl("kernelgen")(kernelgen_impls.sign)
    abs.impl("kernelgen")(kernelgen_impls.abs)
    ceil.impl("kernelgen")(kernelgen_impls.ceil)
    floor.impl("kernelgen")(kernelgen_impls.floor)
    negative.impl("kernelgen")(kernelgen_impls.negative)
    reciprocal.impl("kernelgen")(kernelgen_impls.reciprocal)
    square.impl("kernelgen")(kernelgen_impls.square)
    logical_not.impl("kernelgen")(kernelgen_impls.logical_not)

    # --- Linalg ops ---
    from nkipy.core.ops.linalg import matmul
    matmul.impl("kernelgen")(kernelgen_impls.matmul)

    # --- Reduction ops ---
    from nkipy.core.ops.reduce import sum, prod, max, min, mean, std, var
    sum.impl("kernelgen")(kernelgen_impls.reduce_sum)
    prod.impl("kernelgen")(kernelgen_impls.reduce_prod)
    max.impl("kernelgen")(kernelgen_impls.reduce_max)
    min.impl("kernelgen")(kernelgen_impls.reduce_min)
    mean.impl("kernelgen")(kernelgen_impls.reduce_mean)
    std.impl("kernelgen")(kernelgen_impls.reduce_std)
    var.impl("kernelgen")(kernelgen_impls.reduce_var)

    # --- Creation ops ---
    from nkipy.core.ops.creation import (
        zeros as zeros_op, full as full_op,
        zeros_like, ones_like, empty_like, full_like,
    )
    zeros_op.impl("kernelgen")(kernelgen_impls.zeros)
    full_op.impl("kernelgen")(kernelgen_impls.full)
    zeros_like.impl("kernelgen")(kernelgen_impls.zeros_like)
    ones_like.impl("kernelgen")(kernelgen_impls.ones_like)
    empty_like.impl("kernelgen")(kernelgen_impls.empty_like)
    full_like.impl("kernelgen")(kernelgen_impls.full_like)

    # --- Transform ops ---
    from nkipy.core.ops.transform import (
        transpose, reshape, expand_dims, concatenate,
        split, copy, broadcast_to, astype, squeeze, swapaxes, stack,
    )
    transpose.impl("kernelgen")(kernelgen_impls.transpose)
    reshape.impl("kernelgen")(kernelgen_impls.reshape)
    expand_dims.impl("kernelgen")(kernelgen_impls.expand_dims)
    concatenate.impl("kernelgen")(kernelgen_impls.concatenate)
    split.impl("kernelgen")(kernelgen_impls.split)
    copy.impl("kernelgen")(kernelgen_impls.copy)
    broadcast_to.impl("kernelgen")(kernelgen_impls.broadcast_to)
    astype.impl("kernelgen")(kernelgen_impls.astype)
    squeeze.impl("kernelgen")(kernelgen_impls.squeeze)
    swapaxes.impl("kernelgen")(kernelgen_impls.swapaxes)
    stack.impl("kernelgen")(kernelgen_impls.stack)

    # --- Indexing ops ---
    from nkipy.core.ops.indexing import (
        where as where_op, take as take_op,
        static_slice, dynamic_update_slice,
    )
    where_op.impl("kernelgen")(kernelgen_impls.where)
    take_op.impl("kernelgen")(kernelgen_impls.take)
    static_slice.impl("kernelgen")(kernelgen_impls.static_slice)
    dynamic_update_slice.impl("kernelgen")(kernelgen_impls.dynamic_update_slice)
