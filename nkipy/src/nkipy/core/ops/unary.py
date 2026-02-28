# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary operations: abs, exp, log, sqrt, sin, cos, etc."""

import numpy as np

from nkipy.core.ops._registry import Op

# =============================================================================
# HLO Implementation
# =============================================================================


def _build_unary_hlo(x, np_op, out=None, dtype=None):
    """Build a unary HLO operation."""
    from nkipy.core.backend.hlo import as_hlo_tensor, get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef

    ctx = get_hlo_context()

    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor

    # Special handling for arctan: use atan2(x, 1)
    if np_op == np.arctan:
        one_tensor = as_hlo_tensor(ctx, 1.0, x.dtype)
        if x.shape:
            one_tensor = ctx.build_op(
                "broadcast",
                [one_tensor],
                x.shape,
                x.dtype,
                {"broadcast_dimensions": []},
            )
        result_tensor = ctx.build_op("atan2", [x, one_tensor], x.shape, x.dtype)
        return NKIPyTensorRef(result_tensor)

    # Map numpy ops to HLO opcodes
    op_map = {
        np.abs: "abs",
        np.exp: "exponential",
        np.log: "log",
        np.sqrt: "sqrt",
        np.sin: "sine",
        np.cos: "cosine",
        np.tanh: "tanh",
        np.negative: "negate",
        np.ceil: "ceil",
        np.floor: "floor",
        np.sign: "sign",
        np.bitwise_not: "not",
        np.invert: "not",
    }

    hlo_op = op_map.get(
        np_op, np_op.__name__ if hasattr(np_op, "__name__") else str(np_op)
    )
    result_tensor = ctx.build_op(hlo_op, [x], x.shape, x.dtype)

    return NKIPyTensorRef(result_tensor)


# -----------------------------------------------------------------------------
# Factory function for simple unary ops
# -----------------------------------------------------------------------------
def _make_unary_op(name: str, np_op) -> Op:
    """Create a unary Op with IR and HLO implementations."""
    op = Op(name)

    @op.impl("hlo")
    def _impl_hlo(x, out=None, dtype=None):
        return _build_unary_hlo(x, np_op, out=out, dtype=dtype)

    return op


# -----------------------------------------------------------------------------
# Math operations
# -----------------------------------------------------------------------------
abs = _make_unary_op("abs", np.abs)
exp = _make_unary_op("exp", np.exp)
log = _make_unary_op("log", np.log)
sqrt = _make_unary_op("sqrt", np.sqrt)
negative = _make_unary_op("negative", np.negative)

reciprocal = Op("reciprocal")


@reciprocal.impl("hlo")
def _reciprocal_hlo(x, out=None, dtype=None):
    from nkipy.core.ops.binary import divide

    return divide(1.0, x)


square = Op("square")


@square.impl("hlo")
def _square_hlo(x, out=None, dtype=None):
    from nkipy.core.ops.binary import multiply

    return multiply(x, x)


# -----------------------------------------------------------------------------
# Trigonometric operations
# -----------------------------------------------------------------------------
sin = _make_unary_op("sin", np.sin)
cos = _make_unary_op("cos", np.cos)

tan = Op("tan")


@tan.impl("hlo")
def _tan_hlo(x, out=None, dtype=None):
    from nkipy.core.ops.binary import divide

    return divide(sin(x), cos(x))


arctan = _make_unary_op("arctan", np.arctan)
tanh = _make_unary_op("tanh", np.tanh)

# -----------------------------------------------------------------------------
# Rounding operations
# -----------------------------------------------------------------------------
ceil = _make_unary_op("ceil", np.ceil)
floor = _make_unary_op("floor", np.floor)
sign = _make_unary_op("sign", np.sign)

# rint: round to nearest even (banker's rounding) - simplified using ops
rint = Op("rint")


@rint.impl("hlo")
def _rint_hlo(x, out=None, dtype=None):
    """Round to nearest even integer (banker's rounding).

    Logic:
    - frac = x - floor(x)
    - If frac > 0.5: round up
    - If frac == 0.5 AND floor is odd: round up (to make even)
    - Otherwise: use floor
    """
    from nkipy.core.ops.binary import (
        add,
        divide,
        equal,
        greater,
        logical_and,
        multiply,
        not_equal,
        subtract,
    )
    from nkipy.core.ops.indexing import where

    floor_val = floor(x)
    frac = subtract(x, floor_val)
    ceil_val = add(floor_val, 1.0)

    # Check if frac > 0.5
    frac_gt_half = greater(frac, 0.5)

    # Check if floor is odd: floor % 2 != 0
    # floor % 2 = floor - 2 * floor(floor / 2)
    floor_div_two = floor(divide(floor_val, 2.0))
    floor_mod_two = subtract(floor_val, multiply(floor_div_two, 2.0))
    is_odd = not_equal(floor_mod_two, 0.0)

    # Check if frac == 0.5 AND floor is odd
    frac_eq_half = equal(frac, 0.5)
    should_round_up_for_even = logical_and(frac_eq_half, is_odd)

    # First select: if frac > 0.5, use ceil, else use floor
    result = where(frac_gt_half, ceil_val, floor_val)

    # Second select: if frac == 0.5 AND odd, use ceil (to make even)
    return where(should_round_up_for_even, ceil_val, result)


trunc = Op("trunc")


@trunc.impl("hlo")
def _trunc_hlo(x, out=None, dtype=None):
    from nkipy.core.ops.binary import greater_equal
    from nkipy.core.ops.indexing import where

    return where(greater_equal(x, 0), floor(x), ceil(x))


# -----------------------------------------------------------------------------
# Bitwise/logical operations
# -----------------------------------------------------------------------------
invert = _make_unary_op("invert", np.invert)
bitwise_not = _make_unary_op("bitwise_not", np.bitwise_not)

logical_not = Op("logical_not")


@logical_not.impl("hlo")
def _logical_not_hlo(x, out=None, dtype=None):
    from nkipy.core.ops.binary import equal

    return equal(x, 0)


# -----------------------------------------------------------------------------
# clip: clamp values between a_min and a_max
# -----------------------------------------------------------------------------
clip = Op("clip")


@clip.impl("hlo")
def _clip_hlo(x, a_min=None, a_max=None, out=None):
    from nkipy.core.ops.binary import maximum, minimum

    result = x
    if a_min is not None:
        result = maximum(result, a_min)
    if a_max is not None:
        result = minimum(result, a_max)
    return result


# -----------------------------------------------------------------------------
# log1p: log(1 + x)
# -----------------------------------------------------------------------------
log1p = Op("log1p")


@log1p.impl("hlo")
def _log1p_hlo(x, out=None, dtype=None):
    """log(1+x). Note: uses log(1+x) decomposition; loses precision for |x| << 1."""
    from nkipy.core.ops.binary import add

    return log(add(x, 1.0))
