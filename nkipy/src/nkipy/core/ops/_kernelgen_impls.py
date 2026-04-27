# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""KernelGen backend implementations for NKIPy ops.

Trivial ops use _unary/_binary factories that delegate to the builder.
Non-trivial ops with custom logic are explicit functions below.
"""

from __future__ import annotations

import numpy as np

from nkipy.core.tensor import NKIPyTensorRef
from nkipy.core.backend.kernelgen import KernelGenTensor

_builder_module = None


def _builder():
    global _builder_module
    if _builder_module is None:
        import nkipy_kernelgen.builder as _mod
        _builder_module = _mod
    return _builder_module


def _unwrap(x):
    if isinstance(x, NKIPyTensorRef):
        return x.backend_tensor.handle
    return x


def _wrap(handle):
    kt = KernelGenTensor(handle, handle.shape, handle.dtype)
    return NKIPyTensorRef(kt)


# ---------------------------------------------------------------------------
# Factories for trivial delegation to builder
# ---------------------------------------------------------------------------

def _unary(method):
    def impl(x, out=None, dtype=None):
        return _wrap(getattr(_builder(), method)(_unwrap(x)))
    return impl


def _binary(method):
    def impl(x, y, out=None, dtype=None):
        return _wrap(getattr(_builder(), method)(_unwrap(x), _unwrap(y)))
    return impl


def _reduce(method):
    def impl(x, axis=None, keepdims=False, **kwargs):
        return _wrap(getattr(_builder(), method)(_unwrap(x), axis=axis, keepdims=keepdims))
    return impl


# Binary ops
add = _binary("add")
subtract = _binary("subtract")
multiply = _binary("multiply")
divide = _binary("divide")
power = _binary("power")
maximum = _binary("maximum")
minimum = _binary("minimum")
equal = _binary("equal")
not_equal = _binary("not_equal")
greater = _binary("greater")
greater_equal = _binary("greater_equal")
less = _binary("less")
less_equal = _binary("less_equal")
bitwise_and = _binary("bitwise_and")
bitwise_or = _binary("bitwise_or")
bitwise_xor = _binary("bitwise_xor")
matmul = _binary("matmul")

# Unary ops
exp = _unary("exp")
log = _unary("log")
sqrt = _unary("sqrt")
tanh = _unary("tanh")
sin = _unary("sin")
cos = _unary("cos")
sign = _unary("sign")
abs = _unary("abs_")
ceil = _unary("ceil_")
floor = _unary("floor_")

# Reductions
reduce_sum = _reduce("reduce_sum")
reduce_prod = _reduce("reduce_prod")
reduce_max = _reduce("reduce_max")
reduce_min = _reduce("reduce_min")
reduce_mean = _reduce("reduce_mean")
reduce_std = _reduce("reduce_std")
reduce_var = _reduce("reduce_var")


# ---------------------------------------------------------------------------
# Composed unary ops
# ---------------------------------------------------------------------------

def negative(x, out=None, dtype=None):
    return _wrap(_builder().subtract(_unwrap(0), _unwrap(x)))


def reciprocal(x, out=None, dtype=None):
    return _wrap(_builder().divide(_unwrap(1.0), _unwrap(x)))


def square(x, out=None, dtype=None):
    h = _unwrap(x)
    return _wrap(_builder().multiply(h, h))


def logical_not(x, out=None, dtype=None):
    return _wrap(_builder().subtract(_unwrap(1), _unwrap(x)))


# ---------------------------------------------------------------------------
# Transform ops with custom signatures
# ---------------------------------------------------------------------------

def transpose(x, axes=None):
    return _wrap(_builder().transpose(_unwrap(x), axes=axes))


def reshape(x, newshape, order='C'):
    return _wrap(_builder().reshape(_unwrap(x), newshape))


def expand_dims(x, axis):
    return _wrap(_builder().expand_dims(_unwrap(x), axis))


def copy(x, order='K', subok=True):
    return _wrap(_builder().copy_(_unwrap(x)))


def broadcast_to(x, shape):
    return _wrap(_builder().broadcast_to(_unwrap(x), tuple(shape)))


def astype(x, dtype):
    return _wrap(_builder().astype(_unwrap(x), dtype))


def concatenate(arrays, axis=0, out=None, dtype=None):
    handles = [_unwrap(a) for a in arrays]
    return _wrap(_builder().concatenate(handles, axis=axis))


def where(condition, x, y):
    return _wrap(_builder().where(_unwrap(condition), _unwrap(x), _unwrap(y)))


def take(a, indices, axis=0):
    return _wrap(_builder().take(_unwrap(a), _unwrap(indices), axis=axis))


# ---------------------------------------------------------------------------
# Creation ops
# ---------------------------------------------------------------------------

def zeros(shape, dtype=np.float32):
    return _wrap(_builder().zeros(tuple(shape), dtype))


def full(shape, fill_value, dtype=np.float32):
    return _wrap(_builder().full(tuple(shape), fill_value, dtype))


def zeros_like(x, dtype=None):
    h = _unwrap(x)
    dt = dtype if dtype is not None else h.dtype
    return zeros(h.shape, dt)


def ones_like(x, dtype=None):
    h = _unwrap(x)
    dt = dtype if dtype is not None else h.dtype
    return full(h.shape, 1.0, dt)


def empty_like(x, dtype=None):
    h = _unwrap(x)
    dt = dtype if dtype is not None else h.dtype
    return _wrap(_builder().empty(h.shape, dt))


def full_like(x, fill_value, dtype=None):
    h = _unwrap(x)
    dt = dtype if dtype is not None else h.dtype
    return full(h.shape, fill_value, dt)


# ---------------------------------------------------------------------------
# Squeeze / swapaxes / stack / split
# ---------------------------------------------------------------------------

def squeeze(x, axis=None):
    h = _unwrap(x)
    shape = h.shape
    if axis is None:
        new_shape = tuple(d for d in shape if d != 1)
    else:
        if isinstance(axis, int):
            axis = (axis,)
        new_shape = tuple(d for i, d in enumerate(shape) if i not in axis)
    if new_shape == shape:
        return x
    return reshape(x, new_shape)


def swapaxes(x, axis1, axis2):
    h = _unwrap(x)
    rank = len(h.shape)
    perm = list(range(rank))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return transpose(x, axes=perm)


def stack(arrays, axis=0):
    expanded = [expand_dims(a, axis) for a in arrays]
    return concatenate(expanded, axis=axis)


def split(x, indices_or_sections, axis=0):
    h = _unwrap(x)
    shape = h.shape
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        size = shape[axis]
        section_size = size // sections
        results = []
        for i in range(sections):
            start = [0] * len(shape)
            start[axis] = i * section_size
            limit = list(shape)
            limit[axis] = (i + 1) * section_size
            strides = [1] * len(shape)
            results.append(static_slice(x, start, limit, strides, []))
        return tuple(results)
    raise NotImplementedError("split with explicit indices not yet implemented")


# ---------------------------------------------------------------------------
# Static slicing
# ---------------------------------------------------------------------------

def static_slice(x, start_indices, limit_indices, strides, squeeze_dims):
    return _wrap(_builder().static_slice(
        _unwrap(x), start_indices, limit_indices, strides, squeeze_dims,
    ))


# ---------------------------------------------------------------------------
# Slice assignment (dynamic_update_slice)
# ---------------------------------------------------------------------------

def dynamic_update_slice(x, value, start_indices, update_shape):
    b = _builder()
    x_h = _unwrap(x)
    if isinstance(value, NKIPyTensorRef):
        value_h = _unwrap(value)
    elif isinstance(value, (int, float)):
        value_h = b.full(tuple(update_shape), value, x_h.dtype)
    elif isinstance(value, np.ndarray):
        raise NotImplementedError(
            "Assigning a raw np.ndarray constant in kernelgen is not supported. "
            "Use a traced tensor expression instead."
        )
    else:
        value_h = value

    if value_h.shape != tuple(update_shape):
        value_h = b.reshape(value_h, tuple(update_shape))

    sizes = list(update_shape)
    strides = [1] * len(start_indices)
    result_h = b.static_insert_slice(x_h, value_h, start_indices, sizes, strides)
    return _wrap(result_h)
