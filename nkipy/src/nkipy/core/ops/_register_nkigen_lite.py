# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register nkigen-lite backend implementations for all ops.

Called lazily the first time the nkigen-lite backend is activated, so
nkigen_lite imports only happen when needed.

Composed ops (floor_divide, tan, rint, etc.) use ``composed_impl`` on the
Op itself and need no per-backend registration — they dispatch through
other ops that have nkigen-lite primitives registered.
"""

_registered = False


def register_all_nkigen_lite_impls():
    global _registered
    if _registered:
        return
    _registered = True

    from nkipy.core.ops import _nkigen_lite_impls as lite_impls

    # --- Binary ops (primitives) ---
    from nkipy.core.ops.binary import (
        add, subtract, multiply, divide, power, maximum, minimum,
        equal, not_equal, greater, greater_equal, less, less_equal,
        bitwise_and, bitwise_or, bitwise_xor,
    )
    add.impl("nkigen-lite")(lite_impls.add)
    subtract.impl("nkigen-lite")(lite_impls.subtract)
    multiply.impl("nkigen-lite")(lite_impls.multiply)
    divide.impl("nkigen-lite")(lite_impls.divide)
    power.impl("nkigen-lite")(lite_impls.power)
    maximum.impl("nkigen-lite")(lite_impls.maximum)
    minimum.impl("nkigen-lite")(lite_impls.minimum)
    equal.impl("nkigen-lite")(lite_impls.equal)
    not_equal.impl("nkigen-lite")(lite_impls.not_equal)
    greater.impl("nkigen-lite")(lite_impls.greater)
    greater_equal.impl("nkigen-lite")(lite_impls.greater_equal)
    less.impl("nkigen-lite")(lite_impls.less)
    less_equal.impl("nkigen-lite")(lite_impls.less_equal)
    bitwise_and.impl("nkigen-lite")(lite_impls.bitwise_and)
    bitwise_or.impl("nkigen-lite")(lite_impls.bitwise_or)
    bitwise_xor.impl("nkigen-lite")(lite_impls.bitwise_xor)

    # --- Unary ops (primitives) ---
    from nkipy.core.ops.unary import (
        abs, exp, log, sqrt, sin, cos, arctan, tanh, ceil, floor, sign,
        negative, reciprocal, square, logical_not,
    )
    exp.impl("nkigen-lite")(lite_impls.exp)
    log.impl("nkigen-lite")(lite_impls.log)
    sqrt.impl("nkigen-lite")(lite_impls.sqrt)
    tanh.impl("nkigen-lite")(lite_impls.tanh)
    sin.impl("nkigen-lite")(lite_impls.sin)
    cos.impl("nkigen-lite")(lite_impls.cos)
    arctan.impl("nkigen-lite")(lite_impls.arctan)
    sign.impl("nkigen-lite")(lite_impls.sign)
    abs.impl("nkigen-lite")(lite_impls.abs_)
    ceil.impl("nkigen-lite")(lite_impls.ceil)
    floor.impl("nkigen-lite")(lite_impls.floor)
    negative.impl("nkigen-lite")(lite_impls.negative)
    reciprocal.impl("nkigen-lite")(lite_impls.reciprocal)
    square.impl("nkigen-lite")(lite_impls.square)
    logical_not.impl("nkigen-lite")(lite_impls.logical_not)

    # --- Linalg ops ---
    from nkipy.core.ops.linalg import matmul
    matmul.impl("nkigen-lite")(lite_impls.matmul)

    # --- Reduction ops ---
    from nkipy.core.ops.reduce import sum, prod, max, min, mean, std, var
    sum.impl("nkigen-lite")(lite_impls.reduce_sum)
    prod.impl("nkigen-lite")(lite_impls.reduce_prod)
    max.impl("nkigen-lite")(lite_impls.reduce_max)
    min.impl("nkigen-lite")(lite_impls.reduce_min)
    mean.impl("nkigen-lite")(lite_impls.reduce_mean)
    std.impl("nkigen-lite")(lite_impls.reduce_std)
    var.impl("nkigen-lite")(lite_impls.reduce_var)

    # --- Creation ops ---
    from nkipy.core.ops.creation import (
        zeros as zeros_op, full as full_op, constant as constant_op,
        zeros_like, ones_like, empty_like, full_like,
    )
    zeros_op.impl("nkigen-lite")(lite_impls.zeros)
    full_op.impl("nkigen-lite")(lite_impls.full)
    constant_op.impl("nkigen-lite")(lite_impls.constant)
    zeros_like.impl("nkigen-lite")(lite_impls.zeros_like)
    ones_like.impl("nkigen-lite")(lite_impls.ones_like)
    empty_like.impl("nkigen-lite")(lite_impls.empty_like)
    full_like.impl("nkigen-lite")(lite_impls.full_like)

    # --- Transform ops ---
    from nkipy.core.ops.transform import (
        transpose, reshape, expand_dims, concatenate,
        split, copy, broadcast_to, astype, squeeze, swapaxes, stack,
    )
    transpose.impl("nkigen-lite")(lite_impls.transpose)
    reshape.impl("nkigen-lite")(lite_impls.reshape)
    expand_dims.impl("nkigen-lite")(lite_impls.expand_dims)
    concatenate.impl("nkigen-lite")(lite_impls.concatenate)
    split.impl("nkigen-lite")(lite_impls.split)
    copy.impl("nkigen-lite")(lite_impls.copy)
    broadcast_to.impl("nkigen-lite")(lite_impls.broadcast_to)
    astype.impl("nkigen-lite")(lite_impls.astype)
    squeeze.impl("nkigen-lite")(lite_impls.squeeze)
    swapaxes.impl("nkigen-lite")(lite_impls.swapaxes)
    stack.impl("nkigen-lite")(lite_impls.stack)

    # --- Indexing ops ---
    from nkipy.core.ops.indexing import (
        where as where_op, take as take_op,
        static_slice, dynamic_update_slice,
    )
    where_op.impl("nkigen-lite")(lite_impls.where)
    take_op.impl("nkigen-lite")(lite_impls.take)
    static_slice.impl("nkigen-lite")(lite_impls.static_slice)
    dynamic_update_slice.impl("nkigen-lite")(lite_impls.dynamic_update_slice)
