# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NkiGen-Lite backend implementations for NKIPy ops.

Delegates to the nkigen_lite.tensor_ir.Builder API. Scalar operands are
promoted to constant tensors matching the other operand's shape/dtype.
"""

from __future__ import annotations

import numpy as np

from nkipy.core.tensor import NKIPyTensorRef
from nkipy.core.backend.nkigen_lite import (
    NkiGenLiteTensor,
    get_nkigen_lite_context,
    np_dtype_to_lite,
)


def _ctx():
    return get_nkigen_lite_context()


def _builder():
    return _ctx().builder


def _unwrap(x):
    """Unwrap NKIPyTensorRef to get the nkigen_lite Value handle."""
    if isinstance(x, NKIPyTensorRef):
        return x.backend_tensor.handle
    return x


def _wrap(value):
    """Wrap a nkigen_lite Value into a NKIPyTensorRef."""
    from nkigen_lite.core import to_np_dtype
    shape = value.type.shape
    dtype = to_np_dtype(value.type.dtype)
    kt = NkiGenLiteTensor(value, shape, dtype)
    return NKIPyTensorRef(kt)


def _ensure_value(x, ref_value):
    """Ensure x is a nkigen_lite Value. If scalar, broadcast to match ref_value."""
    from nkigen_lite.core import Value
    if isinstance(x, Value):
        return x
    if isinstance(x, NKIPyTensorRef):
        return x.backend_tensor.handle
    b = _builder()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return b.constant(float(x.item()), ref_value.type.shape, ref_value.type.dtype)
        flat = x.ravel()
        if np.all(flat == flat[0]):
            return b.constant(float(flat[0]), ref_value.type.shape, ref_value.type.dtype)
        raise NotImplementedError(
            f"Non-uniform numpy array constants not yet supported in nkigen-lite"
        )
    shape = ref_value.type.shape
    dtype = ref_value.type.dtype
    return b.constant(float(x), shape, dtype)


def _broadcast_pair(a_val, b_val):
    """Ensure a pair of Values have compatible shapes via broadcast."""
    a_shape = a_val.type.shape
    b_shape = b_val.type.shape
    if a_shape == b_shape:
        return a_val, b_val
    out_shape = np.broadcast_shapes(a_shape, b_shape)
    b = _builder()
    if a_shape != out_shape:
        a_val = b.broadcast_to(a_val, out_shape)
    if b_shape != out_shape:
        b_val = b.broadcast_to(b_val, out_shape)
    return a_val, b_val


def _cast_if_needed(val, target_dtype):
    """Cast val to target_dtype if they differ."""
    if val.type.dtype == target_dtype:
        return val
    return _builder().cast(val, target_dtype)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

def _binary_op(method_name, x, y, out=None, dtype=None):
    b = _builder()
    x_val = _unwrap(x)
    y_val = _unwrap(y)

    # Handle scalars
    from nkigen_lite.core import Value
    if not isinstance(x_val, Value):
        x_val = _ensure_value(x_val, y_val)
    if not isinstance(y_val, Value):
        y_val = _ensure_value(y_val, x_val)

    # Type promotion
    if x_val.type.dtype != y_val.type.dtype:
        target = x_val.type.dtype
        y_val = _cast_if_needed(y_val, target)

    # Broadcast shapes
    x_val, y_val = _broadcast_pair(x_val, y_val)

    result = getattr(b, method_name)(x_val, y_val)
    return _wrap(result)


def add(x, y, out=None, dtype=None):
    return _binary_op("add", x, y, out, dtype)


def subtract(x, y, out=None, dtype=None):
    return _binary_op("sub", x, y, out, dtype)


def multiply(x, y, out=None, dtype=None):
    return _binary_op("mul", x, y, out, dtype)


def divide(x, y, out=None, dtype=None):
    return _binary_op("div", x, y, out, dtype)


def power(x, y, out=None, dtype=None):
    return _binary_op("power", x, y, out, dtype)


def maximum(x, y, out=None, dtype=None):
    return _binary_op("maximum", x, y, out, dtype)


def minimum(x, y, out=None, dtype=None):
    return _binary_op("minimum", x, y, out, dtype)


# Comparison ops
def _compare_op(method_name, x, y, out=None, dtype=None):
    b = _builder()
    x_val = _unwrap(x)
    y_val = _unwrap(y)
    from nkigen_lite.core import Value
    if not isinstance(x_val, Value):
        x_val = _ensure_value(x_val, y_val)
    if not isinstance(y_val, Value):
        y_val = _ensure_value(y_val, x_val)
    if x_val.type.dtype != y_val.type.dtype:
        y_val = _cast_if_needed(y_val, x_val.type.dtype)
    x_val, y_val = _broadcast_pair(x_val, y_val)
    result = getattr(b, method_name)(x_val, y_val)
    return _wrap(result)


def equal(x, y, out=None, dtype=None):
    return _compare_op("equal", x, y, out, dtype)


def not_equal(x, y, out=None, dtype=None):
    return _compare_op("not_equal", x, y, out, dtype)


def greater(x, y, out=None, dtype=None):
    return _compare_op("greater", x, y, out, dtype)


def greater_equal(x, y, out=None, dtype=None):
    return _compare_op("greater_equal", x, y, out, dtype)


def less(x, y, out=None, dtype=None):
    return _compare_op("less", x, y, out, dtype)


def less_equal(x, y, out=None, dtype=None):
    return _compare_op("less_equal", x, y, out, dtype)


# Bitwise ops — implemented as comparison + select patterns for nkigen-lite
def bitwise_and(x, y, out=None, dtype=None):
    return _binary_op("bitwise_and", x, y, out, dtype)


def bitwise_or(x, y, out=None, dtype=None):
    return _binary_op("bitwise_or", x, y, out, dtype)


def bitwise_xor(x, y, out=None, dtype=None):
    return _binary_op("bitwise_xor", x, y, out, dtype)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------

def _unary_op(method_name, x, out=None, dtype=None):
    b = _builder()
    return _wrap(getattr(b, method_name)(_unwrap(x)))


def exp(x, out=None, dtype=None):
    return _unary_op("exp", x, out, dtype)


def log(x, out=None, dtype=None):
    return _unary_op("log", x, out, dtype)


def sqrt(x, out=None, dtype=None):
    return _unary_op("sqrt", x, out, dtype)


def tanh(x, out=None, dtype=None):
    return _unary_op("tanh", x, out, dtype)


def sin(x, out=None, dtype=None):
    return _unary_op("sin", x, out, dtype)


def cos(x, out=None, dtype=None):
    return _unary_op("cos", x, out, dtype)


def arctan(x, out=None, dtype=None):
    return _unary_op("arctan", x, out, dtype)


def sign(x, out=None, dtype=None):
    return _unary_op("sign", x, out, dtype)


def abs_(x, out=None, dtype=None):
    return _unary_op("abs", x, out, dtype)


def ceil(x, out=None, dtype=None):
    return _unary_op("ceil", x, out, dtype)


def floor(x, out=None, dtype=None):
    return _unary_op("floor", x, out, dtype)


def negative(x, out=None, dtype=None):
    return _unary_op("neg", x, out, dtype)


def reciprocal(x, out=None, dtype=None):
    return _unary_op("reciprocal", x, out, dtype)


def square(x, out=None, dtype=None):
    b = _builder()
    x_val = _unwrap(x)
    return _wrap(b.mul(x_val, x_val))


def logical_not(x, out=None, dtype=None):
    b = _builder()
    x_val = _unwrap(x)
    from nkigen_lite.core import DType
    zero = b.constant(0.0, x_val.type.shape, x_val.type.dtype)
    return _wrap(b.equal(x_val, zero))


# ---------------------------------------------------------------------------
# Linalg ops
# ---------------------------------------------------------------------------

def matmul(x, y, out=None, dtype=None):
    b = _builder()
    x_val = _unwrap(x)
    y_val = _unwrap(y)
    from nkigen_lite.core import Value
    if not isinstance(x_val, Value):
        x_val = _ensure_value(x_val, y_val)
    if not isinstance(y_val, Value):
        y_val = _ensure_value(y_val, x_val)
    if x_val.type.dtype != y_val.type.dtype:
        y_val = _cast_if_needed(y_val, x_val.type.dtype)
    # 1D promotion following NumPy matmul semantics
    squeeze_lhs = False
    squeeze_rhs = False
    if len(x_val.type.shape) == 1:
        x_val = b.reshape(x_val, (1, x_val.type.shape[0]))
        squeeze_lhs = True
    if len(y_val.type.shape) == 1:
        y_val = b.reshape(y_val, (y_val.type.shape[0], 1))
        squeeze_rhs = True
    k1 = x_val.type.shape[-1]
    k2 = y_val.type.shape[-2]
    assert k1 == k2, f"Incompatible shapes for matmul: {x_val.type.shape} @ {y_val.type.shape}"
    result = b.matmul(x_val, y_val)
    if squeeze_lhs and squeeze_rhs:
        result = b.reshape(result, ())
    elif squeeze_lhs:
        new_shape = result.type.shape[:-2] + result.type.shape[-1:]
        result = b.reshape(result, new_shape)
    elif squeeze_rhs:
        new_shape = result.type.shape[:-1]
        result = b.reshape(result, new_shape)
    return _wrap(result)


# ---------------------------------------------------------------------------
# Collective communication ops
# ---------------------------------------------------------------------------

def _reduce_op_to_str(reduce_op):
    """Map a numpy reduce ufunc to the nkigen_lite collective reduce-op name."""
    mapping = {np.add: "add", np.maximum: "max", np.minimum: "min",
               np.multiply: "multiply"}
    return mapping.get(reduce_op, "add")


def all_reduce(data, replica_groups, reduce_op=np.add, **kwargs):
    b = _builder()
    x_val = _unwrap(data)
    return _wrap(b.all_reduce(x_val, replica_groups, _reduce_op_to_str(reduce_op)))


def _move_dim_to_last(b, x_val, dim):
    """Transpose so `dim` becomes the last axis; return (xt, inverse_perm).

    The NKI collective KB API only operates on the last (free) axis, so
    collectives along other axes are staged via transpose. The returned
    inverse permutation restores the original axis order afterwards.
    """
    rank = len(x_val.type.shape)
    dim = dim % rank
    if dim == rank - 1:
        return x_val, None
    perm = [i for i in range(rank) if i != dim] + [dim]
    inverse = [perm.index(i) for i in range(rank)]
    return b.transpose(x_val, tuple(perm)), tuple(inverse)


def all_gather(data, all_gather_dim, replica_groups, **kwargs):
    b = _builder()
    x_val = _unwrap(data)
    rank = len(x_val.type.shape)
    xt, inverse = _move_dim_to_last(b, x_val, all_gather_dim)
    gathered = b.all_gather(xt, len(xt.type.shape) - 1, replica_groups)
    if inverse is not None:
        gathered = b.transpose(gathered, inverse)
    return _wrap(gathered)


def reduce_scatter(data, reduce_scatter_dim, replica_groups, reduce_op=np.add, **kwargs):
    b = _builder()
    x_val = _unwrap(data)
    xt, inverse = _move_dim_to_last(b, x_val, reduce_scatter_dim)
    scattered = b.reduce_scatter(
        xt, len(xt.type.shape) - 1, replica_groups, _reduce_op_to_str(reduce_op)
    )
    if inverse is not None:
        scattered = b.transpose(scattered, inverse)
    return _wrap(scattered)


def all_to_all(data, split_dimension, concat_dimension, replica_groups, **kwargs):
    b = _builder()
    x_val = _unwrap(data)
    return _wrap(
        b.all_to_all(x_val, split_dimension, concat_dimension, replica_groups)
    )


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------

def _reduce_op(kind, x, axis=None, keepdims=False, **kwargs):
    b = _builder()
    x_val = _unwrap(x)
    if axis is None:
        axis = tuple(range(len(x_val.type.shape)))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    return _wrap(b.reduce(x_val, axis=axis, kind=kind, keepdims=keepdims))


def reduce_sum(x, axis=None, keepdims=False, **kwargs):
    return _reduce_op("sum", x, axis, keepdims)


def reduce_prod(x, axis=None, keepdims=False, **kwargs):
    # nkigen_lite doesn't have prod reduction; use log-sum-exp pattern
    # prod(x) = exp(sum(log(x)))
    b = _builder()
    x_val = _unwrap(x)
    if axis is None:
        axis = tuple(range(len(x_val.type.shape)))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    log_x = b.log(x_val)
    sum_log = b.reduce(log_x, axis=axis, kind="sum", keepdims=keepdims)
    return _wrap(b.exp(sum_log))


def reduce_max(x, axis=None, keepdims=False, **kwargs):
    return _reduce_op("max", x, axis, keepdims)


def reduce_min(x, axis=None, keepdims=False, **kwargs):
    return _reduce_op("min", x, axis, keepdims)


def reduce_mean(x, axis=None, keepdims=False, **kwargs):
    return _reduce_op("mean", x, axis, keepdims)


def reduce_std(x, axis=None, keepdims=False, **kwargs):
    # std = sqrt(var)
    b = _builder()
    x_val = _unwrap(x)
    if axis is None:
        axis = tuple(range(len(x_val.type.shape)))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    mean_val = b.reduce(x_val, axis=axis, kind="mean", keepdims=True)
    # Broadcast mean back
    diff = b.sub(x_val, b.broadcast_to(mean_val, x_val.type.shape))
    sq = b.mul(diff, diff)
    var_val = b.reduce(sq, axis=axis, kind="mean", keepdims=keepdims)
    return _wrap(b.sqrt(var_val))


def reduce_var(x, axis=None, keepdims=False, **kwargs):
    b = _builder()
    x_val = _unwrap(x)
    if axis is None:
        axis = tuple(range(len(x_val.type.shape)))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    mean_val = b.reduce(x_val, axis=axis, kind="mean", keepdims=True)
    diff = b.sub(x_val, b.broadcast_to(mean_val, x_val.type.shape))
    sq = b.mul(diff, diff)
    return _wrap(b.reduce(sq, axis=axis, kind="mean", keepdims=keepdims))


# ---------------------------------------------------------------------------
# Creation ops
# ---------------------------------------------------------------------------

def zeros(shape, dtype=np.float32):
    b = _builder()
    lite_dtype = np_dtype_to_lite(np.dtype(dtype))
    if isinstance(shape, int):
        shape = (shape,)
    return _wrap(b.zeros(tuple(shape), lite_dtype))


def full(shape, fill_value, dtype=np.float32):
    b = _builder()
    lite_dtype = np_dtype_to_lite(np.dtype(dtype))
    if isinstance(shape, int):
        shape = (shape,)
    return _wrap(b.full(tuple(shape), float(fill_value), lite_dtype))


def constant(value, dtype=None):
    # Passthrough for already-traced tensors, optionally casting dtype.
    if isinstance(value, NKIPyTensorRef):
        if dtype is not None and value.dtype != np.dtype(dtype):
            return astype(value, dtype)
        return value

    # Resolve target dtype following numpy's scalar conventions.
    if dtype is not None:
        target_dtype = np.dtype(dtype)
    elif hasattr(value, "dtype"):
        target_dtype = np.dtype(value.dtype)
    elif isinstance(value, bool):
        target_dtype = np.dtype(np.bool_)
    elif isinstance(value, int):
        target_dtype = np.dtype(np.int32)
    elif isinstance(value, float):
        target_dtype = np.dtype(np.float32)
    else:
        target_dtype = np.dtype(np.asarray(value).dtype)

    b = _builder()
    lite_dtype = np_dtype_to_lite(target_dtype)
    arr = np.asarray(value, dtype=target_dtype)
    # The lite builder can only represent uniform-valued constants (fill).
    flat = arr.ravel()
    if flat.size > 0 and not np.all(flat == flat[0]):
        raise NotImplementedError(
            "Non-uniform array constants are not yet supported in nkigen-lite"
        )
    fill = float(flat[0]) if flat.size > 0 else 0.0
    return _wrap(b.constant(fill, tuple(arr.shape), lite_dtype))


def zeros_like(x, dtype=None):
    h = _unwrap(x)
    dt = np_dtype_to_lite(np.dtype(dtype)) if dtype is not None else h.type.dtype
    return _wrap(_builder().zeros(h.type.shape, dt))


def ones_like(x, dtype=None):
    h = _unwrap(x)
    dt = np_dtype_to_lite(np.dtype(dtype)) if dtype is not None else h.type.dtype
    return _wrap(_builder().full(h.type.shape, 1.0, dt))


def empty_like(x, dtype=None):
    h = _unwrap(x)
    dt = np_dtype_to_lite(np.dtype(dtype)) if dtype is not None else h.type.dtype
    return _wrap(_builder().zeros(h.type.shape, dt))


def full_like(x, fill_value, dtype=None):
    h = _unwrap(x)
    dt = np_dtype_to_lite(np.dtype(dtype)) if dtype is not None else h.type.dtype
    return _wrap(_builder().full(h.type.shape, float(fill_value), dt))


# ---------------------------------------------------------------------------
# Transform ops
# ---------------------------------------------------------------------------

def transpose(x, axes=None):
    b = _builder()
    x_val = _unwrap(x)
    if axes is None:
        axes = tuple(reversed(range(len(x_val.type.shape))))
    return _wrap(b.transpose(x_val, tuple(axes)))


def reshape(x, newshape, order='C'):
    b = _builder()
    x_val = _unwrap(x)
    if isinstance(newshape, int):
        newshape = [newshape]
    else:
        newshape = list(newshape)
    # Resolve -1 dimension
    if -1 in newshape:
        from math import prod
        total = prod(x_val.type.shape)
        known = prod(s for s in newshape if s != -1)
        idx = newshape.index(-1)
        newshape[idx] = total // known
    return _wrap(b.reshape(x_val, tuple(newshape)))


def expand_dims(x, axis):
    b = _builder()
    x_val = _unwrap(x)
    ndim = len(x_val.type.shape)
    if isinstance(axis, (list, tuple)):
        out_ndim = ndim + len(axis)
        norm_axes = []
        for a in axis:
            if a < -out_ndim or a >= out_ndim:
                raise ValueError(
                    f"axis {a} is out of bounds for array of dimension {out_ndim}"
                )
            norm_axes.append(a % out_ndim)
        if len(set(norm_axes)) != len(norm_axes):
            raise ValueError(f"repeated axis in expand_dims")
        result = x_val
        for ax in sorted(norm_axes):
            result = b.expand_dims(result, ax)
        return _wrap(result)
    out_ndim = ndim + 1
    if axis < -out_ndim or axis >= out_ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {out_ndim}"
        )
    return _wrap(b.expand_dims(x_val, axis))


def copy(x, order='K', subok=True):
    # In SSA-based IR, every op produces a new value — identity is copy
    b = _builder()
    x_val = _unwrap(x)
    zero = b.constant(0.0, x_val.type.shape, x_val.type.dtype)
    return _wrap(b.add(x_val, zero))


def broadcast_to(x, shape):
    b = _builder()
    x_val = _unwrap(x)
    return _wrap(b.broadcast_to(x_val, tuple(shape)))


def astype(x, dtype):
    b = _builder()
    x_val = _unwrap(x)
    lite_dtype = np_dtype_to_lite(np.dtype(dtype))
    if x_val.type.dtype == lite_dtype:
        return x if isinstance(x, NKIPyTensorRef) else _wrap(x_val)
    return _wrap(b.cast(x_val, lite_dtype))


def concatenate(arrays, axis=0, out=None, dtype=None):
    b = _builder()
    if len(arrays) == 0:
        raise ValueError("Need at least one tensor to concatenate")
    values = [_unwrap(a) for a in arrays]
    if len(values) == 1:
        return arrays[0] if isinstance(arrays[0], NKIPyTensorRef) else _wrap(values[0])
    rank = len(values[0].type.shape)
    if axis < -rank or axis >= rank:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {rank}"
        )
    return _wrap(b.concat(values, axis=axis))


def where(condition, x, y):
    b = _builder()
    c_val = _unwrap(condition)
    x_val = _unwrap(x)
    y_val = _unwrap(y)
    from nkigen_lite.core import Value, DType
    if not isinstance(c_val, Value):
        ref = x_val if isinstance(x_val, Value) else y_val
        c_val = _ensure_value(c_val, ref)
    if not isinstance(x_val, Value):
        x_val = _ensure_value(x_val, y_val)
    if not isinstance(y_val, Value):
        y_val = _ensure_value(y_val, x_val)
    if x_val.type.dtype != y_val.type.dtype:
        y_val = _cast_if_needed(y_val, x_val.type.dtype)
    # Ensure condition is float (1.0/0.0) matching x/y dtype
    if c_val.type.dtype != x_val.type.dtype:
        zero = b.constant(0.0, c_val.type.shape, c_val.type.dtype)
        c_val = b.not_equal(c_val, zero)
        if c_val.type.dtype != x_val.type.dtype:
            c_val = b.cast(c_val, x_val.type.dtype)
    # Broadcast all to common shape
    out_shape = np.broadcast_shapes(c_val.type.shape, x_val.type.shape, y_val.type.shape)
    if c_val.type.shape != out_shape:
        c_val = b.broadcast_to(c_val, out_shape)
    if x_val.type.shape != out_shape:
        x_val = b.broadcast_to(x_val, out_shape)
    if y_val.type.shape != out_shape:
        y_val = b.broadcast_to(y_val, out_shape)
    return _wrap(b.where(c_val, x_val, y_val))


def take(a, indices, axis=None):
    """np.take with static (trace-time) integer indices.

    Implemented as a slice-based gather: each requested index becomes a
    width-1 slice along ``axis``; the slices are concatenated and reshaped
    so the gathered axis is replaced by ``indices.shape``. This matches
    numpy semantics:

        out.shape == a.shape[:axis] + indices.shape + a.shape[axis + 1:]

    A scalar index removes the axis entirely (``indices.shape == ()``).
    """
    b = _builder()
    a_val = _unwrap(a)

    # Dynamic (traced) indices need a hardware gather, which isn't supported.
    if isinstance(indices, NKIPyTensorRef):
        raise NotImplementedError(
            "Dynamic tensor indexing is not yet supported in nkigen-lite. "
            "Use static numpy array indices instead."
        )

    idx_arr = np.asarray(indices)

    # axis=None flattens the input and gathers from the flat vector.
    if axis is None:
        total = int(np.prod(a_val.type.shape)) if a_val.type.shape else 1
        a_val = b.reshape(a_val, (total,))
        axis = 0

    in_shape = a_val.type.shape
    rank = len(in_shape)
    axis = axis % rank
    axis_dim = in_shape[axis]

    # Gather each flat index as a width-1 slice along `axis`, then concat.
    flat_idx = idx_arr.flatten()
    slices = []
    for raw in flat_idx:
        i = int(raw) % axis_dim  # normalize negatives like numpy
        starts = tuple(0 if d != axis else i for d in range(rank))
        stops = tuple(in_shape[d] if d != axis else i + 1 for d in range(rank))
        slices.append(b.slice(a_val, starts, stops))

    gathered = slices[0] if len(slices) == 1 else b.concat(slices, axis=axis)

    # gathered currently has `axis` size == len(flat_idx); reshape so that
    # axis is replaced by indices.shape (dropped entirely for scalar index).
    out_shape = in_shape[:axis] + tuple(idx_arr.shape) + in_shape[axis + 1:]
    if gathered.type.shape != out_shape:
        gathered = b.reshape(gathered, out_shape)
    return _wrap(gathered)


# ---------------------------------------------------------------------------
# Squeeze / swapaxes / stack / split
# ---------------------------------------------------------------------------

def squeeze(x, axis=None):
    b = _builder()
    x_val = _unwrap(x)
    shape = x_val.type.shape
    rank = len(shape)
    if axis is None:
        new_shape = tuple(d for d in shape if d != 1)
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(a % rank for a in axis)
        for a in axis:
            if shape[a] != 1:
                raise ValueError(
                    f"cannot select an axis to squeeze out which has size "
                    f"!= 1 (got {shape[a]} for axis {a})"
                )
        new_shape = tuple(d for i, d in enumerate(shape) if i not in axis)
    if new_shape == shape:
        return x if isinstance(x, NKIPyTensorRef) else _wrap(x_val)
    return _wrap(b.reshape(x_val, new_shape))


def swapaxes(x, axis1, axis2):
    x_val = _unwrap(x)
    rank = len(x_val.type.shape)
    perm = list(range(rank))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return transpose(x, axes=perm)


def stack(arrays, axis=0):
    expanded = [expand_dims(a, axis) for a in arrays]
    return concatenate(expanded, axis=axis)


def split(x, indices_or_sections, axis=0):
    b = _builder()
    x_val = _unwrap(x)
    shape = x_val.type.shape
    rank = len(shape)
    if axis < -rank or axis >= rank:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {rank}"
        )
    axis = axis % rank
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if sections <= 0:
            raise ValueError("number of sections must be larger than 0")
        size = shape[axis]
        if size % sections != 0:
            raise ValueError(
                f"array split does not result in an equal division: "
                f"shape {shape} axis {axis} sections {sections}"
            )
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
    b = _builder()
    x_val = _unwrap(x)
    starts = tuple(start_indices)
    stops = tuple(limit_indices)
    strs = tuple(strides) if strides else None
    result = b.slice(x_val, starts, stops, strs)
    if squeeze_dims:
        new_shape = tuple(
            s for i, s in enumerate(result.type.shape) if i not in squeeze_dims
        )
        if new_shape != result.type.shape:
            result = b.reshape(result, new_shape)
    return _wrap(result)


# ---------------------------------------------------------------------------
# Slice assignment (dynamic_update_slice)
# ---------------------------------------------------------------------------

def dynamic_update_slice(x, value, start_indices, update_shape):
    # In nkigen-lite SSA IR, we can't do in-place update.
    # We need to produce a new tensor. The lowering pipeline will handle
    # the actual memory management.
    # Strategy: slice the original into prefix/update/suffix and concat back.
    # For simplicity, use the "scatter" pattern via slice + concat.
    b = _builder()
    x_val = _unwrap(x)

    if isinstance(value, NKIPyTensorRef):
        value_val = _unwrap(value)
    elif isinstance(value, (int, float)):
        lite_dtype = x_val.type.dtype
        value_val = b.full(tuple(update_shape), float(value), lite_dtype)
    elif isinstance(value, np.ndarray):
        flat = value.ravel()
        if np.all(flat == flat[0]):
            value_val = b.full(tuple(update_shape), float(flat[0]), x_val.type.dtype)
        else:
            raise NotImplementedError(
                "Non-uniform numpy array in dynamic_update_slice not supported"
            )
    else:
        value_val = value

    # Ensure value matches update_shape
    if value_val.type.shape != tuple(update_shape):
        value_val = b.reshape(value_val, tuple(update_shape))

    # For a simple 1D/2D slice update along a single axis, we can decompose
    # into concat(prefix, value, suffix). For multi-axis updates, this is
    # more complex. We'll handle the common case.
    rank = len(x_val.type.shape)

    # Find the axis with non-zero start (the update axis)
    # For multi-axis updates, we need a more general approach
    # Use slice-based reconstruction
    # General approach: for each dimension, if start > 0 or end < dim_size,
    # we need to preserve the surrounding data.

    # Simple approach: build the full tensor via slice decomposition
    # This works for contiguous updates along any combination of axes.
    # We produce: for each axis, slice before + update + slice after, nested.
    # But this gets complex for multi-axis. Use a simpler recursive approach.

    # Find first axis where update is partial (not full extent)
    update_axis = None
    for i in range(rank):
        if start_indices[i] != 0 or update_shape[i] != x_val.type.shape[i]:
            update_axis = i
            break

    if update_axis is None:
        # Full replacement
        return _wrap(value_val)

    # Split along update_axis: prefix + update_region + suffix
    axis = update_axis
    start = start_indices[axis]
    end = start + update_shape[axis]
    dim_size = x_val.type.shape[axis]

    parts = []
    if start > 0:
        pre_starts = tuple(0 if i != axis else 0 for i in range(rank))
        pre_stops = tuple(x_val.type.shape[i] if i != axis else start for i in range(rank))
        parts.append(b.slice(x_val, pre_starts, pre_stops))

    # For the middle part, if there are further partial axes, recurse
    remaining_axes_partial = any(
        start_indices[i] != 0 or update_shape[i] != x_val.type.shape[i]
        for i in range(axis + 1, rank)
    )

    if remaining_axes_partial:
        # Extract the middle slice from x, then recursively update within it
        mid_starts = tuple(0 if i != axis else start for i in range(rank))
        mid_stops = tuple(x_val.type.shape[i] if i != axis else end for i in range(rank))
        mid_slice = b.slice(x_val, mid_starts, mid_stops)

        # Recursive update on the sub-slice
        sub_start = [start_indices[i] if i != axis else 0 for i in range(rank)]
        sub_shape = list(update_shape)
        sub_shape[axis] = update_shape[axis]
        sub_result = _dynamic_update_inner(mid_slice, value_val, sub_start, sub_shape, axis + 1)
        parts.append(sub_result)
    else:
        parts.append(value_val)

    if end < dim_size:
        suf_starts = tuple(0 if i != axis else end for i in range(rank))
        suf_stops = tuple(x_val.type.shape[i] for i in range(rank))
        parts.append(b.slice(x_val, suf_starts, suf_stops))

    if len(parts) == 1:
        return _wrap(parts[0])
    result = b.concat(parts, axis=axis)
    return _wrap(result)


def _dynamic_update_inner(x_val, value_val, start_indices, update_shape, from_axis):
    """Recursively handle multi-axis updates."""
    b = _builder()
    rank = len(x_val.type.shape)

    update_axis = None
    for i in range(from_axis, rank):
        if start_indices[i] != 0 or update_shape[i] != x_val.type.shape[i]:
            update_axis = i
            break

    if update_axis is None:
        return value_val

    axis = update_axis
    start = start_indices[axis]
    end = start + update_shape[axis]
    dim_size = x_val.type.shape[axis]

    parts = []
    if start > 0:
        pre_starts = tuple(0 for _ in range(rank))
        pre_stops = tuple(x_val.type.shape[i] if i != axis else start for i in range(rank))
        parts.append(b.slice(x_val, pre_starts, pre_stops))

    remaining = any(
        start_indices[i] != 0 or update_shape[i] != x_val.type.shape[i]
        for i in range(axis + 1, rank)
    )
    if remaining:
        mid_starts = tuple(0 if i != axis else start for i in range(rank))
        mid_stops = tuple(x_val.type.shape[i] if i != axis else end for i in range(rank))
        mid_slice = b.slice(x_val, mid_starts, mid_stops)
        sub_start = list(start_indices)
        sub_start[axis] = 0
        sub_result = _dynamic_update_inner(mid_slice, value_val, sub_start, update_shape, axis + 1)
        parts.append(sub_result)
    else:
        parts.append(value_val)

    if end < dim_size:
        suf_starts = tuple(0 if i != axis else end for i in range(rank))
        suf_stops = tuple(x_val.type.shape[i] for i in range(rank))
        parts.append(b.slice(x_val, suf_starts, suf_stops))

    if len(parts) == 1:
        return parts[0]
    return b.concat(parts, axis=axis)
