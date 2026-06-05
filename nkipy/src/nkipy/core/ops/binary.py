# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary operations: add, subtract, multiply, divide, etc."""

from nkipy.core.ops._registry import Op

# -----------------------------------------------------------------------------
# Arithmetic operations
# -----------------------------------------------------------------------------
add = Op("add")
subtract = Op("subtract")
multiply = Op("multiply")
divide = Op("divide")
power = Op("power")
maximum = Op("maximum")
minimum = Op("minimum")

# -----------------------------------------------------------------------------
# Bitwise operations
# -----------------------------------------------------------------------------
bitwise_and = Op("bitwise_and")
bitwise_or = Op("bitwise_or")
bitwise_xor = Op("bitwise_xor")

# -----------------------------------------------------------------------------
# Comparison operations
# -----------------------------------------------------------------------------
equal = Op("equal")
not_equal = Op("not_equal")
greater = Op("greater")
greater_equal = Op("greater_equal")
less = Op("less")
less_equal = Op("less_equal")

# -----------------------------------------------------------------------------
# Logical operations
# -----------------------------------------------------------------------------
logical_and = Op("logical_and")
logical_or = Op("logical_or")
logical_xor = Op("logical_xor")


# -----------------------------------------------------------------------------
# Composed logical operations
#
# Backends without a native logical_and (e.g. nkigen-lite) fall back to these.
# Inputs are reduced to 0/1 truthiness, so the result matches numpy semantics
# for arbitrary numeric inputs, not just boolean-like ones.
# -----------------------------------------------------------------------------


@logical_and.composed_impl
def _logical_and(x, y, out=None, dtype=None):
    return multiply(not_equal(x, 0), not_equal(y, 0))


@logical_or.composed_impl
def _logical_or(x, y, out=None, dtype=None):
    # OR(a, b) = (a != 0) OR (b != 0); max of the two 0/1 predicates.
    return maximum(not_equal(x, 0), not_equal(y, 0))


@logical_xor.composed_impl
def _logical_xor(x, y, out=None, dtype=None):
    return not_equal(not_equal(x, 0), not_equal(y, 0))


# -----------------------------------------------------------------------------
# Composed binary operations
# -----------------------------------------------------------------------------

logaddexp = Op("logaddexp")


@logaddexp.composed_impl
def _logaddexp(x, y):
    from nkipy.core.ops.unary import exp, log

    m = maximum(x, y)
    return add(m, log(add(exp(subtract(x, m)), exp(subtract(y, m)))))


remainder = Op("remainder")


@remainder.composed_impl
def _remainder(x, y):
    from nkipy.core.ops.unary import floor

    return subtract(x, multiply(y, floor(divide(x, y))))


floor_divide = Op("floor_divide")


@floor_divide.composed_impl
def _floor_divide(x, y):
    from nkipy.core.ops.unary import floor

    return floor(divide(x, y))
