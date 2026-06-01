# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register nkigen backend implementations for all ops.

Called lazily the first time the nkigen backend is activated, so MLIR
imports only happen when needed.

Composed ops (floor_divide, tan, rint, etc.) use ``composed_impl`` on the
Op itself and need no per-backend registration — they dispatch through
other ops that have nkigen primitives registered.
"""

_registered = False


def register_all_nkigen_impls():
    global _registered
    if _registered:
        return
    _registered = True

    from nkipy.core.ops import _nkigen_impls as nkigen_impls

    # --- Binary ops (primitives) ---
    from nkipy.core.ops.binary import (
        add, subtract, multiply, divide, power, maximum, minimum,
        equal, not_equal, greater, greater_equal, less, less_equal,
        bitwise_and, bitwise_or, bitwise_xor,
    )
    add.impl("nkigen")(nkigen_impls.add)
    subtract.impl("nkigen")(nkigen_impls.subtract)
    multiply.impl("nkigen")(nkigen_impls.multiply)
    divide.impl("nkigen")(nkigen_impls.divide)
    power.impl("nkigen")(nkigen_impls.power)
    maximum.impl("nkigen")(nkigen_impls.maximum)
    minimum.impl("nkigen")(nkigen_impls.minimum)
    equal.impl("nkigen")(nkigen_impls.equal)
    not_equal.impl("nkigen")(nkigen_impls.not_equal)
    greater.impl("nkigen")(nkigen_impls.greater)
    greater_equal.impl("nkigen")(nkigen_impls.greater_equal)
    less.impl("nkigen")(nkigen_impls.less)
    less_equal.impl("nkigen")(nkigen_impls.less_equal)
    bitwise_and.impl("nkigen")(nkigen_impls.bitwise_and)
    bitwise_or.impl("nkigen")(nkigen_impls.bitwise_or)
    bitwise_xor.impl("nkigen")(nkigen_impls.bitwise_xor)

    # --- Unary ops (primitives) ---
    from nkipy.core.ops.unary import (
        abs, exp, log, sqrt, sin, cos, tanh, ceil, floor, sign,
        negative, reciprocal, square, logical_not,
    )
    exp.impl("nkigen")(nkigen_impls.exp)
    log.impl("nkigen")(nkigen_impls.log)
    sqrt.impl("nkigen")(nkigen_impls.sqrt)
    tanh.impl("nkigen")(nkigen_impls.tanh)
    sin.impl("nkigen")(nkigen_impls.sin)
    cos.impl("nkigen")(nkigen_impls.cos)
    sign.impl("nkigen")(nkigen_impls.sign)
    abs.impl("nkigen")(nkigen_impls.abs)
    ceil.impl("nkigen")(nkigen_impls.ceil)
    floor.impl("nkigen")(nkigen_impls.floor)
    negative.impl("nkigen")(nkigen_impls.negative)
    reciprocal.impl("nkigen")(nkigen_impls.reciprocal)
    square.impl("nkigen")(nkigen_impls.square)
    logical_not.impl("nkigen")(nkigen_impls.logical_not)

    # --- Linalg ops ---
    from nkipy.core.ops.linalg import matmul
    matmul.impl("nkigen")(nkigen_impls.matmul)

    # --- Reduction ops ---
    from nkipy.core.ops.reduce import sum, prod, max, min, mean, std, var
    sum.impl("nkigen")(nkigen_impls.reduce_sum)
    prod.impl("nkigen")(nkigen_impls.reduce_prod)
    max.impl("nkigen")(nkigen_impls.reduce_max)
    min.impl("nkigen")(nkigen_impls.reduce_min)
    mean.impl("nkigen")(nkigen_impls.reduce_mean)
    std.impl("nkigen")(nkigen_impls.reduce_std)
    var.impl("nkigen")(nkigen_impls.reduce_var)

    # --- Creation ops ---
    from nkipy.core.ops.creation import (
        zeros as zeros_op, full as full_op,
        zeros_like, ones_like, empty_like, full_like,
    )
    zeros_op.impl("nkigen")(nkigen_impls.zeros)
    full_op.impl("nkigen")(nkigen_impls.full)
    zeros_like.impl("nkigen")(nkigen_impls.zeros_like)
    ones_like.impl("nkigen")(nkigen_impls.ones_like)
    empty_like.impl("nkigen")(nkigen_impls.empty_like)
    full_like.impl("nkigen")(nkigen_impls.full_like)

    # --- Transform ops ---
    from nkipy.core.ops.transform import (
        transpose, reshape, expand_dims, concatenate,
        split, copy, broadcast_to, astype, squeeze, swapaxes, stack,
    )
    transpose.impl("nkigen")(nkigen_impls.transpose)
    reshape.impl("nkigen")(nkigen_impls.reshape)
    expand_dims.impl("nkigen")(nkigen_impls.expand_dims)
    concatenate.impl("nkigen")(nkigen_impls.concatenate)
    split.impl("nkigen")(nkigen_impls.split)
    copy.impl("nkigen")(nkigen_impls.copy)
    broadcast_to.impl("nkigen")(nkigen_impls.broadcast_to)
    astype.impl("nkigen")(nkigen_impls.astype)
    squeeze.impl("nkigen")(nkigen_impls.squeeze)
    swapaxes.impl("nkigen")(nkigen_impls.swapaxes)
    stack.impl("nkigen")(nkigen_impls.stack)

    # --- Indexing ops ---
    from nkipy.core.ops.indexing import (
        where as where_op, take as take_op,
        static_slice, dynamic_update_slice,
    )
    where_op.impl("nkigen")(nkigen_impls.where)
    take_op.impl("nkigen")(nkigen_impls.take)
    static_slice.impl("nkigen")(nkigen_impls.static_slice)
    dynamic_update_slice.impl("nkigen")(nkigen_impls.dynamic_update_slice)
