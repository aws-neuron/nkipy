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
        floor_divide, remainder,
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
    floor_divide.impl("nkigen-lite")(lite_impls.floor_divide)
    remainder.impl("nkigen-lite")(lite_impls.remainder)
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
    from nkipy.core.ops.linalg import matmul, trace
    matmul.impl("nkigen-lite")(lite_impls.matmul)
    trace.impl("nkigen-lite")(lite_impls.trace)

    # --- Reduction ops ---
    from nkipy.core.ops.reduce import (
        sum, prod, max, min, mean, std, var, argmax, argmin, cumsum,
    )
    sum.impl("nkigen-lite")(lite_impls.reduce_sum)
    prod.impl("nkigen-lite")(lite_impls.reduce_prod)
    max.impl("nkigen-lite")(lite_impls.reduce_max)
    min.impl("nkigen-lite")(lite_impls.reduce_min)
    mean.impl("nkigen-lite")(lite_impls.reduce_mean)
    std.impl("nkigen-lite")(lite_impls.reduce_std)
    var.impl("nkigen-lite")(lite_impls.reduce_var)
    argmax.impl("nkigen-lite")(lite_impls.argmax)
    argmin.impl("nkigen-lite")(lite_impls.argmin)
    cumsum.impl("nkigen-lite")(lite_impls.cumsum)

    # --- Creation ops ---
    from nkipy.core.ops.creation import (
        zeros as zeros_op, full as full_op, constant as constant_op,
        zeros_like, ones_like, empty_like, full_like,
        tril, triu, diag,
    )
    zeros_op.impl("nkigen-lite")(lite_impls.zeros)
    full_op.impl("nkigen-lite")(lite_impls.full)
    constant_op.impl("nkigen-lite")(lite_impls.constant)
    zeros_like.impl("nkigen-lite")(lite_impls.zeros_like)
    ones_like.impl("nkigen-lite")(lite_impls.ones_like)
    empty_like.impl("nkigen-lite")(lite_impls.empty_like)
    full_like.impl("nkigen-lite")(lite_impls.full_like)
    tril.impl("nkigen-lite")(lite_impls.tril)
    triu.impl("nkigen-lite")(lite_impls.triu)
    diag.impl("nkigen-lite")(lite_impls.diag)

    # --- Transform ops ---
    from nkipy.core.ops.transform import (
        transpose, reshape, expand_dims, concatenate,
        split, copy, broadcast_to, astype, squeeze, swapaxes, stack,
        pad, diff, flip, tile, roll,
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
    pad.impl("nkigen-lite")(lite_impls.pad)
    diff.impl("nkigen-lite")(lite_impls.diff)
    flip.impl("nkigen-lite")(lite_impls.flip)
    tile.impl("nkigen-lite")(lite_impls.tile)
    roll.impl("nkigen-lite")(lite_impls.roll)

    # --- Indexing ops ---
    from nkipy.core.ops.indexing import (
        where as where_op, take as take_op,
        static_slice, dynamic_update_slice,
    )
    where_op.impl("nkigen-lite")(lite_impls.where)
    take_op.impl("nkigen-lite")(lite_impls.take)
    static_slice.impl("nkigen-lite")(lite_impls.static_slice)
    dynamic_update_slice.impl("nkigen-lite")(lite_impls.dynamic_update_slice)

    # --- Collective ops ---
    from nkipy.core.ops.collectives import (
        all_gather, all_reduce, reduce_scatter, all_to_all,
    )
    all_gather.impl("nkigen-lite")(lite_impls.all_gather)
    all_reduce.impl("nkigen-lite")(lite_impls.all_reduce)
    reduce_scatter.impl("nkigen-lite")(lite_impls.reduce_scatter)
    all_to_all.impl("nkigen-lite")(lite_impls.all_to_all)
