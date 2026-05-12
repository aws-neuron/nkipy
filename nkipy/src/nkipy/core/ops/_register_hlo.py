# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register HLO backend implementations for all ops.

Called lazily the first time the HLO backend is activated, so HLO-specific
imports only happen when needed.

Composed ops (floor_divide, tan, rint, etc.) are registered as
``composed_impl`` on the Op itself — they need no per-backend registration
since they dispatch through other ops.
"""

_registered = False


def register_all_hlo_impls():
    global _registered
    if _registered:
        return
    _registered = True

    from nkipy.core.ops import _hlo_impls as hlo_impls

    # --- Binary ops (primitives) ---
    from nkipy.core.ops.binary import (
        add, subtract, multiply, divide, power, maximum, minimum,
        bitwise_and, bitwise_or, bitwise_xor,
        equal, not_equal, greater, greater_equal, less, less_equal,
        logical_and, logical_or, logical_xor,
    )
    add.impl("hlo")(hlo_impls.add)
    subtract.impl("hlo")(hlo_impls.subtract)
    multiply.impl("hlo")(hlo_impls.multiply)
    divide.impl("hlo")(hlo_impls.divide)
    power.impl("hlo")(hlo_impls.power)
    maximum.impl("hlo")(hlo_impls.maximum)
    minimum.impl("hlo")(hlo_impls.minimum)
    bitwise_and.impl("hlo")(hlo_impls.bitwise_and)
    bitwise_or.impl("hlo")(hlo_impls.bitwise_or)
    bitwise_xor.impl("hlo")(hlo_impls.bitwise_xor)
    equal.impl("hlo")(hlo_impls.equal)
    not_equal.impl("hlo")(hlo_impls.not_equal)
    greater.impl("hlo")(hlo_impls.greater)
    greater_equal.impl("hlo")(hlo_impls.greater_equal)
    less.impl("hlo")(hlo_impls.less)
    less_equal.impl("hlo")(hlo_impls.less_equal)
    logical_and.impl("hlo")(hlo_impls.logical_and)
    logical_or.impl("hlo")(hlo_impls.logical_or)
    logical_xor.impl("hlo")(hlo_impls.logical_xor)

    # --- Unary ops (primitives) ---
    # Note: reciprocal, square, logical_not use composed_impl (dispatch through
    # other ops) so they don't need backend-specific registration.
    from nkipy.core.ops.unary import (
        abs, exp, log, sqrt, sin, cos, tanh, ceil, floor, sign,
        negative, arctan, invert, bitwise_not,
    )
    abs.impl("hlo")(hlo_impls.abs)
    exp.impl("hlo")(hlo_impls.exp)
    log.impl("hlo")(hlo_impls.log)
    sqrt.impl("hlo")(hlo_impls.sqrt)
    sin.impl("hlo")(hlo_impls.sin)
    cos.impl("hlo")(hlo_impls.cos)
    tanh.impl("hlo")(hlo_impls.tanh)
    ceil.impl("hlo")(hlo_impls.ceil)
    floor.impl("hlo")(hlo_impls.floor)
    sign.impl("hlo")(hlo_impls.sign)
    negative.impl("hlo")(hlo_impls.negative)
    arctan.impl("hlo")(hlo_impls.arctan)
    invert.impl("hlo")(hlo_impls.invert)
    bitwise_not.impl("hlo")(hlo_impls.bitwise_not)

    # --- Reduction ops (primitives) ---
    from nkipy.core.ops.reduce import sum, prod, max, min, argmax, argmin
    sum.impl("hlo")(hlo_impls.reduce_sum)
    prod.impl("hlo")(hlo_impls.reduce_prod)
    max.impl("hlo")(hlo_impls.reduce_max)
    min.impl("hlo")(hlo_impls.reduce_min)
    argmax.impl("hlo")(hlo_impls.argmax)
    argmin.impl("hlo")(hlo_impls.argmin)

    # --- Linalg ops ---
    from nkipy.core.ops.linalg import matmul, dot, trace
    matmul.impl("hlo")(hlo_impls.matmul)
    dot.impl("hlo")(hlo_impls.dot)
    trace.impl("hlo")(hlo_impls.trace)

    # --- Creation ops ---
    from nkipy.core.ops.creation import (
        zeros as zeros_op, full as full_op, constant,
        zeros_like, ones_like, empty_like, full_like,
        tril, triu, diag,
    )
    zeros_op.impl("hlo")(hlo_impls.zeros)
    full_op.impl("hlo")(hlo_impls.full)
    constant.impl("hlo")(hlo_impls.constant)
    zeros_like.impl("hlo")(hlo_impls.zeros_like)
    ones_like.impl("hlo")(hlo_impls.ones_like)
    empty_like.impl("hlo")(hlo_impls.empty_like)
    full_like.impl("hlo")(hlo_impls.full_like)
    tril.impl("hlo")(hlo_impls.tril)
    triu.impl("hlo")(hlo_impls.triu)
    diag.impl("hlo")(hlo_impls.diag)

    # --- Transform ops ---
    from nkipy.core.ops.transform import (
        reshape, transpose, expand_dims, concatenate, split,
        copy, repeat, broadcast_to, astype, squeeze, pad,
        swapaxes, stack, diff, flip, tile, roll,
    )
    reshape.impl("hlo")(hlo_impls.reshape)
    transpose.impl("hlo")(hlo_impls.transpose)
    expand_dims.impl("hlo")(hlo_impls.expand_dims)
    concatenate.impl("hlo")(hlo_impls.concatenate)
    split.impl("hlo")(hlo_impls.split)
    copy.impl("hlo")(hlo_impls.copy)
    repeat.impl("hlo")(hlo_impls.repeat)
    broadcast_to.impl("hlo")(hlo_impls.broadcast_to)
    astype.impl("hlo")(hlo_impls.astype)
    squeeze.impl("hlo")(hlo_impls.squeeze)
    pad.impl("hlo")(hlo_impls.pad)
    swapaxes.impl("hlo")(hlo_impls.swapaxes)
    stack.impl("hlo")(hlo_impls.stack)
    diff.impl("hlo")(hlo_impls.diff)
    flip.impl("hlo")(hlo_impls.flip)
    tile.impl("hlo")(hlo_impls.tile)
    roll.impl("hlo")(hlo_impls.roll)

    # --- Indexing ops ---
    from nkipy.core.ops.indexing import (
        where as where_op, take as take_op,
        take_along_axis, put_along_axis, scatter_along_axis,
        static_slice, dynamic_update_slice, scatter_strided,
    )
    where_op.impl("hlo")(hlo_impls.where)
    take_op.impl("hlo")(hlo_impls.take)
    take_along_axis.impl("hlo")(hlo_impls.take_along_axis)
    put_along_axis.impl("hlo")(hlo_impls.put_along_axis)
    scatter_along_axis.impl("hlo")(hlo_impls.scatter_along_axis)
    static_slice.impl("hlo")(hlo_impls.static_slice)
    dynamic_update_slice.impl("hlo")(hlo_impls.dynamic_update_slice)
    scatter_strided.impl("hlo")(hlo_impls.scatter_strided)

    # --- NN ops ---
    from nkipy.core.ops.nn import topk
    topk.impl("hlo")(hlo_impls.topk)

    # --- Collective ops ---
    from nkipy.core.ops.collectives import (
        all_gather, all_reduce, reduce_scatter, all_to_all,
    )
    all_gather.impl("hlo")(hlo_impls.all_gather)
    all_reduce.impl("hlo")(hlo_impls.all_reduce)
    reduce_scatter.impl("hlo")(hlo_impls.reduce_scatter)
    all_to_all.impl("hlo")(hlo_impls.all_to_all)

    # --- Conv ops ---
    from nkipy.core.ops.conv import conv2d, conv3d
    conv2d.impl("hlo")(hlo_impls.conv2d)
    conv3d.impl("hlo")(hlo_impls.conv3d)
