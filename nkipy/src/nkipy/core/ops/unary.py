# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary operations: abs, exp, log, sqrt, sin, cos, etc."""

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Math operations (primitives)
# -----------------------------------------------------------------------------
abs = Op("abs")
exp = Op("exp")
log = Op("log")
sqrt = Op("sqrt")
negative = Op("negative")
sin = Op("sin")
cos = Op("cos")
arctan = Op("arctan")
tanh = Op("tanh")
ceil = Op("ceil")
floor = Op("floor")
sign = Op("sign")

# -----------------------------------------------------------------------------
# Bitwise/logical operations (primitives)
# -----------------------------------------------------------------------------
invert = Op("invert")
bitwise_not = Op("bitwise_not")

# -----------------------------------------------------------------------------
# Composed unary ops — built from other dispatched ops
# -----------------------------------------------------------------------------

reciprocal = Op("reciprocal")


@reciprocal.composed_impl
def _reciprocal(x, out=None, dtype=None):
    from nkipy.core.ops.binary import divide

    return divide(1.0, x)


square = Op("square")


@square.composed_impl
def _square(x, out=None, dtype=None):
    from nkipy.core.ops.binary import multiply

    return multiply(x, x)


tan = Op("tan")


@tan.composed_impl
def _tan(x, out=None, dtype=None):
    from nkipy.core.ops.binary import divide

    return divide(sin(x), cos(x))


logical_not = Op("logical_not")


@logical_not.composed_impl
def _logical_not(x, out=None, dtype=None):
    from nkipy.core.ops.binary import equal

    return equal(x, 0)


rint = Op("rint")


@rint.composed_impl
def _rint(x, out=None, dtype=None):
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

    frac_gt_half = greater(frac, 0.5)

    floor_div_two = floor(divide(floor_val, 2.0))
    floor_mod_two = subtract(floor_val, multiply(floor_div_two, 2.0))
    is_odd = not_equal(floor_mod_two, 0.0)

    frac_eq_half = equal(frac, 0.5)
    should_round_up_for_even = logical_and(frac_eq_half, is_odd)

    result = where(frac_gt_half, ceil_val, floor_val)

    return where(should_round_up_for_even, ceil_val, result)


trunc = Op("trunc")


@trunc.composed_impl
def _trunc(x, out=None, dtype=None):
    from nkipy.core.ops.binary import greater_equal
    from nkipy.core.ops.indexing import where

    return where(greater_equal(x, 0), floor(x), ceil(x))


clip = Op("clip")


@clip.composed_impl
def _clip(x, a_min=None, a_max=None, out=None):
    from nkipy.core.ops.binary import maximum, minimum

    result = x
    if a_min is not None:
        result = maximum(result, a_min)
    if a_max is not None:
        result = minimum(result, a_max)
    return result


log1p = Op("log1p")


@log1p.composed_impl
def _log1p(x, out=None, dtype=None):
    from nkipy.core.ops.binary import add

    return log(add(x, 1.0))


log2 = Op("log2")


@log2.composed_impl
def _log2(x, out=None, dtype=None):
    import numpy as np

    from nkipy.core.ops.binary import divide

    return divide(log(x), float(np.log(2.0)))


expm1 = Op("expm1")


@expm1.composed_impl
def _expm1(x, out=None, dtype=None):
    from nkipy.core.ops.binary import subtract

    return subtract(exp(x), 1.0)


round_ = Op("round")


@round_.composed_impl
def _round(x, decimals=0, out=None):
    if decimals == 0:
        return rint(x)
    from nkipy.core.ops.binary import divide, multiply

    scale = float(10**decimals)
    return divide(rint(multiply(x, scale)), scale)


isnan = Op("isnan")


@isnan.composed_impl
def _isnan(x, out=None, dtype=None):
    from nkipy.core.ops.binary import not_equal

    return not_equal(x, x)


isfinite = Op("isfinite")


@isfinite.composed_impl
def _isfinite(x, out=None, dtype=None):
    from nkipy.core.ops.binary import equal, subtract

    return equal(subtract(x, x), 0.0)
