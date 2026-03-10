# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy Operations Module

This module provides a unified interface for tensor operations that dispatch
to the appropriate backend (IR or HLO) based on the current tracing context.
"""

from nkipy.core.backend import (
    get_backend,
    set_source_location,
)
from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.binary import (
    add,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    divide,
    equal,
    floor_divide,
    greater,
    greater_equal,
    less,
    less_equal,
    logaddexp,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    not_equal,
    power,
    remainder,
    subtract,
)

# -----------------------------------------------------------------------------
# Collective operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    reduce_scatter,
)

# -----------------------------------------------------------------------------
# Convolution operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.conv import (
    conv2d,
    conv3d,
)

# -----------------------------------------------------------------------------
# Creation operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.creation import (
    constant,
    diag,
    empty_like,
    full,
    full_like,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

# -----------------------------------------------------------------------------
# Indexing operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.indexing import (
    dynamic_update_slice,
    put_along_axis,
    scatter_along_axis,
    scatter_strided,
    static_slice,
    take,
    take_along_axis,
    where,
)

# -----------------------------------------------------------------------------
# Linear algebra operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.linalg import dot, matmul, norm, outer, trace

# -----------------------------------------------------------------------------
# Neural network operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.nn import (
    rms_norm,
    softmax,
    topk,
)

# -----------------------------------------------------------------------------
# Reduction operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.reduce import (
    any,
    argmax,
    argmin,
    count_nonzero,
    cumsum,
    max,
    mean,
    min,
    prod,
    std,
    sum,
    var,
)

# -----------------------------------------------------------------------------
# Transform operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.transform import (
    astype,
    broadcast_to,
    concatenate,
    copy,
    copyto,
    diff,
    expand_dims,
    flip,
    pad,
    repeat,
    reshape,
    roll,
    split,
    squeeze,
    stack,
    swapaxes,
    tile,
    transpose,
)

# -----------------------------------------------------------------------------
# Unary operations
# -----------------------------------------------------------------------------
from nkipy.core.ops.unary import (
    abs,
    arctan,
    bitwise_not,
    ceil,
    clip,
    cos,
    exp,
    expm1,
    floor,
    invert,
    isfinite,
    isnan,
    log,
    log1p,
    log2,
    logical_not,
    negative,
    reciprocal,
    rint,
    round_,
    sign,
    sin,
    sqrt,
    square,
    tan,
    tanh,
    trunc,
)

__all__ = [
    # Registry & backend
    "Op",
    "get_backend",
    "set_source_location",
    # Binary
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "maximum",
    "minimum",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logaddexp",
    "remainder",
    "floor_divide",
    # Unary
    "abs",
    "exp",
    "log",
    "sqrt",
    "square",
    "negative",
    "reciprocal",
    "sin",
    "cos",
    "tan",
    "arctan",
    "tanh",
    "ceil",
    "floor",
    "rint",
    "trunc",
    "sign",
    "invert",
    "bitwise_not",
    "logical_not",
    "clip",
    "log1p",
    "log2",
    "expm1",
    "round_",
    "isnan",
    "isfinite",
    # Creation
    "constant",
    "zeros",
    "full",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "tril",
    "triu",
    "diag",
    # Linalg
    "dot",
    "matmul",
    "norm",
    "outer",
    "trace",
    # Reduction
    "sum",
    "max",
    "min",
    "mean",
    "any",
    "var",
    "std",
    "argmax",
    "argmin",
    "cumsum",
    "prod",
    "count_nonzero",
    # Transform
    "reshape",
    "transpose",
    "expand_dims",
    "concatenate",
    "split",
    "copy",
    "repeat",
    "broadcast_to",
    "copyto",
    "astype",
    "squeeze",
    "pad",
    "swapaxes",
    "stack",
    "diff",
    "flip",
    "tile",
    "roll",
    # NN
    "softmax",
    "topk",
    "rms_norm",
    # Conv
    "conv2d",
    "conv3d",
    # Indexing
    "where",
    "take",
    "take_along_axis",
    "put_along_axis",
    "scatter_along_axis",
    "static_slice",
    "dynamic_update_slice",
    "scatter_strided",
    # Collectives
    "all_gather",
    "all_reduce",
    "reduce_scatter",
    "all_to_all",
]
