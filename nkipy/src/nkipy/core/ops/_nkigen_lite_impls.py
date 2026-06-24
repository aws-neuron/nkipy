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
        # Non-uniform array: materialize it via the general constant path
        # (run-length fills + concat) rather than a single fill.
        return _unwrap(constant(x))
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


def floor_divide(x, y, out=None, dtype=None):
    # Route to the native floor_divide opcode so the decompose pass'
    # divide-then-verify-and-correct lowering fires. The default composed
    # impl (floor(divide(x, y))) is wrong at exact-integer quotients because
    # the reciprocal-based divide undershoots.
    return _binary_op("floor_divide", x, y, out, dtype)


def remainder(x, y, out=None, dtype=None):
    # Route to the native mod opcode (decomposed as a - b*floor_divide(a, b)
    # using the corrected floor_divide).
    return _binary_op("mod", x, y, out, dtype)


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


def cumsum(x, axis=None, dtype=None):
    """Cumulative sum via matmul with an upper-triangular ones matrix.

    out = x_2d @ U, where U[i, j] = 1 if i <= j else 0 (so column j sums all
    rows 0..j).  U is built with iota + compare rather than a non-uniform
    constant, which keeps the flattened (axis=None) case tractable.
    """
    b = _builder()
    x_val = _unwrap(x)
    x_shape = x_val.type.shape
    ndim = len(x_shape)

    if axis is None:
        total = int(np.prod(x_shape)) if x_shape else 1
        x_val = b.reshape(x_val, (total,))
        x_shape = (total,)
        ndim = 1
        axis = 0
    elif axis < 0:
        axis = ndim + axis

    N = x_shape[axis]
    work_dtype = x_val.type.dtype

    # U[i, j] = 1.0 if i <= j else 0.0  (row index <= col index)
    row = b.iota((N, N), dim=0, dtype=np_dtype_to_lite(np.dtype(np.int32)))
    col = b.iota((N, N), dim=1, dtype=np_dtype_to_lite(np.dtype(np.int32)))
    mask = b.less_equal(row, col)
    ones = b.constant(1.0, (N, N), work_dtype)
    zeros = b.constant(0.0, (N, N), work_dtype)
    tri = _unwrap(where(_wrap(mask), _wrap(ones), _wrap(zeros)))

    def _cumsum_last_axis(x2d_val):
        return _unwrap(matmul(_wrap(x2d_val), _wrap(tri)))

    if ndim == 1:
        x_2d = b.reshape(x_val, (1, N))
        result = b.reshape(_cumsum_last_axis(x_2d), (N,))
    elif axis == ndim - 1:
        batch = int(np.prod(x_shape[:-1]))
        x_2d = b.reshape(x_val, (batch, N))
        result = b.reshape(_cumsum_last_axis(x_2d), x_shape)
    else:
        perm = list(range(ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x_t = b.transpose(x_val, tuple(perm))
        x_t_shape = tuple(x_shape[p] for p in perm)
        batch = int(np.prod(x_t_shape[:-1]))
        x_2d = b.reshape(x_t, (batch, N))
        result_t = b.reshape(_cumsum_last_axis(x_2d), x_t_shape)
        result = b.transpose(result_t, tuple(perm))

    if dtype is not None:
        result = _cast_if_needed(result, np_dtype_to_lite(np.dtype(dtype)))
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


def all_gather(data, all_gather_dim, replica_groups, **kwargs):
    b = _builder()
    x_val = _unwrap(data)
    return _wrap(b.all_gather(x_val, all_gather_dim, replica_groups))


def reduce_scatter(data, reduce_scatter_dim, replica_groups, reduce_op=np.add, **kwargs):
    b = _builder()
    x_val = _unwrap(data)
    return _wrap(
        b.reduce_scatter(x_val, reduce_scatter_dim, replica_groups,
                         _reduce_op_to_str(reduce_op))
    )


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


def _argreduce(kind, x, axis=None, keepdims=False):
    """argmax/argmin via index masking.

    Find the extreme value along ``axis``, mark every position equal to it
    with its index (an iota ramp) and all others with a large sentinel, then
    min-reduce the indices — yielding the *first* index that attains the
    extreme, matching numpy.
    """
    b = _builder()
    x_val = _unwrap(x)
    orig_shape = x_val.type.shape
    orig_axis = axis

    if axis is None:
        total = int(np.prod(orig_shape)) if orig_shape else 1
        x_val = b.reshape(x_val, (total,))
        axis = 0

    ndim = len(x_val.type.shape)
    axis = axis % ndim

    # The whole computation runs in float32 (matching HLO). min/max reductions
    # init with +/-inf, which cannot be memset into an integer tile, so an
    # integer input (or index ramp) would fail to compile. Cast to int32 only
    # at the very end.
    f32 = np_dtype_to_lite(np.dtype(np.float32))
    x_val = _cast_if_needed(x_val, f32)

    # Extreme value along axis, broadcast back for the equality mask.
    extreme = b.reduce(x_val, axis=(axis,), kind=kind, keepdims=True)
    mask = b.equal(x_val, b.broadcast_to(extreme, x_val.type.shape))

    idx = b.iota(x_val.type.shape, dim=axis, dtype=f32)
    sentinel = float(x_val.type.shape[axis] + 1)
    masked = where(_wrap(mask), _wrap(idx), sentinel)

    result = b.reduce(_unwrap(masked), axis=(axis,), kind="min", keepdims=False)
    result = b.cast(result, np_dtype_to_lite(np.dtype(np.int32)))

    if keepdims:
        if orig_axis is not None:
            new_shape = list(orig_shape)
            new_shape[orig_axis % len(orig_shape)] = 1
        else:
            new_shape = [1] * len(orig_shape)
        result = b.reshape(result, tuple(new_shape))

    return _wrap(result)


def argmax(x, axis=None, out=None, keepdims=False):
    return _argreduce("max", x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, out=None, keepdims=False):
    return _argreduce("min", x, axis=axis, keepdims=keepdims)


def topk(x, k, axis=0, is_ascend=False, out=None, dtype=None):
    """Top-k values and indices along ``axis`` (descending; ascending if
    ``is_ascend``), matching torch.topk.

    Delegates to the hardware ``topk`` op (canonical max8 + match_replace8
    scan): the topk axis is moved to the free dim, leading dims flattened to
    the partition dim, and the (P, F) tile reduced.  Supports any k <= axis
    size (ceil(k/8) hardware folds).
    """
    b = _builder()
    x_val = _unwrap(x)
    ndim = len(x_val.type.shape)
    axis = axis % ndim
    n = x_val.type.shape[axis]
    if k > n:
        raise ValueError(f"topk: k={k} exceeds axis {axis} size {n}")

    f32 = np_dtype_to_lite(np.dtype(np.float32))
    work = _cast_if_needed(x_val, f32)
    if is_ascend:
        work = b.neg(work)

    # Move topk axis to last, flatten leading dims to a single partition dim.
    if axis != ndim - 1:
        perm = [d for d in range(ndim) if d != axis] + [axis]
        work = b.transpose(work, tuple(perm))
    else:
        perm = list(range(ndim))
    lead_shape = work.type.shape[:-1]
    P = int(np.prod(lead_shape)) if lead_shape else 1
    F = work.type.shape[-1]
    work2d = b.reshape(work, (P, F))

    vals_k, idx_k = b.topk(work2d, k)             # (P, k), (P, k) int32

    # Reshape (P, k) back to transposed layout, then undo the transpose.
    out_t_shape = tuple(lead_shape) + (k,)
    vals_t = b.reshape(vals_k, out_t_shape)
    idx_t = b.reshape(idx_k, out_t_shape)
    if axis != ndim - 1:
        inv = [0] * ndim
        for new_pos, old in enumerate(perm):
            inv[old] = new_pos
        vals_t = b.transpose(vals_t, tuple(inv))
        idx_t = b.transpose(idx_t, tuple(inv))

    if is_ascend:
        vals_t = b.neg(vals_t)
    idx_out = b.cast(idx_t, np_dtype_to_lite(np.dtype(np.uint32)))
    return _wrap(vals_t), _wrap(idx_out)


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
    flat = arr.ravel()

    # Uniform array (or scalar): a single fill.
    if flat.size <= 1 or np.all(flat == flat[0]):
        fill = float(flat[0]) if flat.size > 0 else 0.0
        return _wrap(b.constant(fill, tuple(arr.shape), lite_dtype))

    # Non-uniform: the builder only emits uniform fills, so materialize the
    # data as a flat sequence of run-length fills, concatenate, and reshape.
    # Cheap for structured/small arrays; worst case (all-distinct) is one fill
    # per element, so cap to keep tracing bounded.
    MAX_RUNS = 4096
    # Run-length encode the flat array.
    change = np.nonzero(np.diff(flat))[0] + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    if len(starts) > MAX_RUNS:
        raise NotImplementedError(
            f"Non-uniform constant with {len(starts)} runs exceeds the "
            f"nkigen-lite limit of {MAX_RUNS}; provide it as a kernel input"
        )

    pieces = [
        b.constant(float(flat[s]), (int(n),), lite_dtype)
        for s, n in zip(starts, lengths)
    ]
    flat_val = pieces[0] if len(pieces) == 1 else b.concat(pieces, axis=0)
    if arr.shape != (flat.size,):
        flat_val = b.reshape(flat_val, tuple(arr.shape))
    return _wrap(flat_val)


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
# Triangular / diagonal ops (built from iota index masks)
# ---------------------------------------------------------------------------

def _i32():
    from nkigen_lite.core import DType
    return DType.I32


def _iota(shape, dim):
    """int32 index ramp along ``dim``, broadcast over the other axes."""
    return _builder().iota(tuple(shape), dim=dim, dtype=_i32())


def _shift(idx_val, k):
    """idx_val - k as an int32 tensor (no-op when k == 0)."""
    if k == 0:
        return idx_val
    b = _builder()
    k_const = b.constant(float(k), idx_val.type.shape, _i32())
    return b.sub(idx_val, k_const)


def _triangular(x, k, keep_lower):
    """Zero out the upper (tril) or lower (triu) triangle.

    tril keeps row >= col - k; triu keeps row <= col - k.  The mask is built
    from row/col iotas over the last two axes (broadcast over any batch dims).
    """
    b = _builder()
    x_val = _unwrap(x)
    shape = x_val.type.shape
    ndim = len(shape)
    if ndim < 2:
        raise ValueError(f"input must be at least 2-D, got {ndim}-D")

    row = _iota(shape, ndim - 2)
    col = _shift(_iota(shape, ndim - 1), k)
    mask = b.greater_equal(row, col) if keep_lower else b.less_equal(row, col)
    return where(_wrap(mask), _wrap(x_val), 0.0)


def tril(x, k=0):
    return _triangular(x, k, keep_lower=True)


def triu(x, k=0):
    return _triangular(x, k, keep_lower=False)


def diag(v, k=0):
    b = _builder()
    v_val = _unwrap(v)
    shape = v_val.type.shape
    ndim = len(shape)

    if ndim == 1:
        # Build an (N, N) matrix with v on the k-th diagonal.  Dynamic gather
        # is unsupported on nkigen-lite, so instead extend v to length N (pad
        # with zeros on the side away from the diagonal), broadcast it across
        # columns, and keep only the diagonal entries (col == row + k).
        n = shape[0]
        N = n + abs(k)
        if k > 0:
            v_ext = b.concat([v_val, b.zeros((k,), v_val.type.dtype)], axis=0)
        elif k < 0:
            v_ext = b.concat([b.zeros((-k,), v_val.type.dtype), v_val], axis=0)
        else:
            v_ext = v_val

        rows = b.broadcast_to(b.reshape(v_ext, (N, 1)), (N, N))
        row_idx = _shift(_iota((N, N), 0), -k)  # row + k
        col_idx = _iota((N, N), 1)
        mask = b.equal(col_idx, row_idx)
        return where(_wrap(mask), _wrap(rows), 0.0)

    elif ndim == 2:
        # Extract the k-th diagonal of a 2-D matrix.
        rows, cols = shape
        if k >= 0:
            diag_len = min(rows, cols - k)
        else:
            diag_len = min(rows + k, cols)
        if diag_len <= 0:
            return _wrap(b.zeros((0,), v_val.type.dtype))

        row_idx = _iota(shape, 0)
        col_idx = _shift(_iota(shape, 1), k)
        mask = b.equal(row_idx, col_idx)
        masked = where(_wrap(mask), _wrap(v_val), 0.0)
        # Each diagonal element survives in exactly one row (k>=0) or column
        # (k<0); summing that axis collapses the mask to the diagonal vector.
        summed = reduce_sum(masked, axis=1 if k >= 0 else 0)
        s_val = _unwrap(summed)
        if s_val.type.shape[0] != diag_len:
            s_val = b.slice(s_val, (0,), (diag_len,))
        return _wrap(s_val)

    raise ValueError(f"Input must be 1-D or 2-D, got {ndim}-D")


def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    b = _builder()
    a_val = _unwrap(a)
    shape = a_val.type.shape
    ndim = len(shape)
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim

    row = _iota(shape, axis1)
    col = _shift(_iota(shape, axis2), offset)
    mask = b.equal(row, col)
    masked = where(_wrap(mask), _wrap(a_val), 0.0)

    result = masked
    for ax in sorted([axis1, axis2], reverse=True):
        result = reduce_sum(result, axis=ax)
    if dtype is not None:
        result = astype(result, np.dtype(dtype))
    return result


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

    # Explicit split indices: numpy semantics — boundaries at the given
    # indices, producing len(indices)+1 sub-arrays (clamped to the axis size,
    # and possibly empty if indices repeat or exceed the size).
    boundaries = [int(i) for i in indices_or_sections]
    axis_size = shape[axis]
    edges = [0] + [min(max(i, 0), axis_size) for i in boundaries] + [axis_size]
    results = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            # numpy yields an empty sub-array here, but the lite IR has no
            # representation for a zero-size tensor (slice rejects it).
            raise NotImplementedError(
                "split producing an empty sub-array (repeated or out-of-range "
                "index) is not supported in nkigen-lite"
            )
        start = [0] * len(shape)
        start[axis] = lo
        limit = list(shape)
        limit[axis] = hi
        strides = [1] * len(shape)
        results.append(static_slice(x, start, limit, strides, []))
    return tuple(results)


def repeat(x, repeats, axis=None):
    """np.repeat with a scalar integer ``repeats``.

    Insert a size-1 axis after ``axis``, broadcast it to ``repeats``, then
    reshape to fold it back in — so each element is duplicated in place.
    """
    b = _builder()
    x_val = _unwrap(x)

    if axis is None:
        total = int(np.prod(x_val.type.shape)) if x_val.type.shape else 1
        x_val = b.reshape(x_val, (total,))
        axis = 0

    ndim = len(x_val.type.shape)
    axis = axis % ndim

    if not isinstance(repeats, (int, np.integer)):
        raise TypeError(
            "Only compile-time integer repeats are supported in nkigen-lite, "
            f"got {type(repeats).__name__}"
        )
    repeats = int(repeats)

    shape = x_val.type.shape
    expanded = b.expand_dims(x_val, axis + 1)            # (..., d, 1, ...)
    bshape = list(expanded.type.shape)
    bshape[axis + 1] = repeats
    broadcast = b.broadcast_to(expanded, tuple(bshape))   # (..., d, r, ...)
    new_shape = list(shape)
    new_shape[axis] = shape[axis] * repeats
    return _wrap(b.reshape(broadcast, tuple(new_shape)))


def _axis_slice(x_val, axis, start, stop):
    """Slice [start:stop] along ``axis``, full extent on every other axis."""
    rank = len(x_val.type.shape)
    starts = tuple(start if d == axis else 0 for d in range(rank))
    stops = tuple(
        stop if d == axis else x_val.type.shape[d] for d in range(rank)
    )
    return _builder().slice(x_val, starts, stops)


def flip(x, axis=None):
    b = _builder()
    x_val = _unwrap(x)
    ndim = len(x_val.type.shape)
    if axis is None:
        axes = list(range(ndim))
    elif isinstance(axis, int):
        axes = [axis % ndim]
    else:
        axes = [a % ndim for a in axis]

    result = x_val
    for ax in axes:
        n = result.type.shape[ax]
        # Reverse by concatenating width-1 slices in descending index order.
        parts = [_axis_slice(result, ax, i, i + 1) for i in range(n - 1, -1, -1)]
        result = b.concat(parts, axis=ax) if len(parts) > 1 else parts[0]
    return _wrap(result)


def tile(x, reps):
    b = _builder()
    x_val = _unwrap(x)
    if isinstance(reps, int):
        reps = (reps,)
    reps = tuple(int(r) for r in reps)

    x_shape = x_val.type.shape
    ndim = len(x_shape)
    if len(reps) < ndim:
        reps = (1,) * (ndim - len(reps)) + reps
    elif len(reps) > ndim:
        x_val = b.reshape(x_val, (1,) * (len(reps) - ndim) + tuple(x_shape))
        x_shape = x_val.type.shape
        ndim = len(x_shape)

    # Repeat each axis by concatenating ``r`` copies of the running result.
    result = x_val
    for ax, r in enumerate(reps):
        if r == 1:
            continue
        result = b.concat([result] * r, axis=ax)
    if result is x_val:
        # all reps == 1: return a fresh value (copy semantics)
        return copy(x)
    return _wrap(result)


def roll(x, shift, axis=None):
    b = _builder()
    x_val = _unwrap(x)
    x_shape = x_val.type.shape
    ndim = len(x_shape)

    if axis is None:
        # Flatten, roll the single axis, restore shape.
        total = int(np.prod(x_shape)) if x_shape else 1
        flat = b.reshape(x_val, (total,))
        rolled = _roll_axis(flat, shift, 0)
        return _wrap(b.reshape(rolled, x_shape))

    if isinstance(shift, (list, tuple)):
        if not isinstance(axis, (list, tuple)):
            raise ValueError("If shift is a tuple, axis must also be a tuple")
        result = x_val
        for s, a in zip(shift, axis):
            result = _roll_axis(result, s, a % ndim)
        return _wrap(result)

    return _wrap(_roll_axis(x_val, shift, axis % ndim))


def _roll_axis(x_val, shift, axis):
    """Cyclic shift along ``axis`` via split + swapped concat."""
    b = _builder()
    n = x_val.type.shape[axis]
    shift = shift % n
    if shift == 0:
        return x_val
    split = n - shift
    tail = _axis_slice(x_val, axis, split, n)   # wraps to the front
    head = _axis_slice(x_val, axis, 0, split)
    return b.concat([tail, head], axis=axis)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    b = _builder()
    a_val = _unwrap(a)
    ndim = len(a_val.type.shape)
    if prepend is not None or append is not None:
        raise NotImplementedError(
            "diff prepend/append not yet supported in nkigen-lite"
        )
    axis = axis % ndim
    result = a_val
    for _ in range(n):
        size = result.type.shape[axis]
        upper = _axis_slice(result, axis, 1, size)      # x[1:]
        lower = _axis_slice(result, axis, 0, size - 1)  # x[:-1]
        result = b.sub(upper, lower)
    return _wrap(result)


def pad(x, pad_width, mode="constant", constant_values=0, **kwargs):
    b = _builder()
    x_val = _unwrap(x)
    shape = x_val.type.shape
    ndim = len(shape)
    dtype = x_val.type.dtype

    # Normalize pad_width to a per-axis [(before, after), ...] list.
    pad_arr = np.asarray(pad_width)
    if pad_arr.ndim == 0:
        pad_list = [(int(pad_arr), int(pad_arr))] * ndim
    elif pad_arr.ndim == 1 and pad_arr.size == 2:
        pad_list = [(int(pad_arr[0]), int(pad_arr[1]))] * ndim
    elif pad_arr.ndim == 2:
        if len(pad_arr) == 1:
            pad_arr = np.broadcast_to(pad_arr, (ndim, 2))
        pad_list = [(int(pad_arr[i, 0]), int(pad_arr[i, 1])) for i in range(ndim)]
    else:
        raise ValueError(f"unsupported pad_width: {pad_width!r}")

    if mode == "constant":
        result = x_val
        for ax, (before, after) in enumerate(pad_list):
            parts = []
            if before > 0:
                p_shape = tuple(
                    before if d == ax else result.type.shape[d] for d in range(ndim)
                )
                parts.append(b.full(p_shape, float(constant_values), dtype))
            parts.append(result)
            if after > 0:
                p_shape = tuple(
                    after if d == ax else result.type.shape[d] for d in range(ndim)
                )
                parts.append(b.full(p_shape, float(constant_values), dtype))
            if len(parts) > 1:
                result = b.concat(parts, axis=ax)
        return _wrap(result)

    elif mode == "edge":
        result = x_val
        for ax, (before, after) in enumerate(pad_list):
            parts = []
            if before > 0:
                edge = _axis_slice(result, ax, 0, 1)         # first slab
                parts.extend([edge] * before)
            parts.append(result)
            if after > 0:
                last = result.type.shape[ax]
                edge = _axis_slice(result, ax, last - 1, last)  # last slab
                parts.extend([edge] * after)
            if len(parts) > 1:
                result = b.concat(parts, axis=ax)
        return _wrap(result)

    raise NotImplementedError(
        f"pad mode {mode!r} is not supported; only 'constant' and 'edge'"
    )


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


# ---------------------------------------------------------------------------
# Convolution (im2col-free: sum of per-kernel-position channel matmuls)
# ---------------------------------------------------------------------------

def _conv_pad_spatial(b, x_val, pads):
    """Zero-pad only the trailing spatial axes by ``pads`` = [(lo, hi), ...]."""
    ndim = len(x_val.type.shape)
    n_spatial = len(pads)
    result = x_val
    for j, (lo, hi) in enumerate(pads):
        if lo == 0 and hi == 0:
            continue
        ax = ndim - n_spatial + j
        slabs = []
        if lo > 0:
            sh = tuple(lo if d == ax else result.type.shape[d] for d in range(ndim))
            slabs.append(b.constant(0.0, sh, result.type.dtype))
        slabs.append(result)
        if hi > 0:
            sh = tuple(hi if d == ax else result.type.shape[d] for d in range(ndim))
            slabs.append(b.constant(0.0, sh, result.type.dtype))
        if len(slabs) > 1:
            result = b.concat(slabs, axis=ax)
    return result


def _conv_nd(input, weight, bias, stride, padding, dilation, groups, n):
    """N-D convolution (n spatial dims) via im2col + a single matmul.

    out[n, co, *p] = sum over ci, *k of
        in_padded[n, ci, *(p*stride + k*dilation)] * weight[co, ci, *k]

    Each kernel position's strided window is reshaped to (N, Ci, out_pts) and
    concatenated along the channel axis into a column tensor
    (N, Ci*prod(K), out_pts); the weight flattens to (Co, Ci*prod(K)); one
    batched matmul produces (N, Co, out_pts).  A single fused matmul compiles
    ~35% faster than accumulating prod(K) separate matmuls.
    """
    b = _builder()
    x = _unwrap(input)
    w = _unwrap(weight)

    if groups != 1:
        raise NotImplementedError(
            f"conv groups != 1 not supported in nkigen-lite, got {groups}"
        )

    x_shape = x.type.shape
    w_shape = w.type.shape
    batch, in_ch = x_shape[0], x_shape[1]
    out_ch, w_in_ch = w_shape[0], w_shape[1]
    ksize = w_shape[2:]
    if w_in_ch != in_ch:
        raise ValueError(
            f"conv: weight in-channels {w_in_ch} != input channels {in_ch}"
        )

    # Pad the spatial dims, then compute output spatial extents.
    x = _conv_pad_spatial(b, x, [(p, p) for p in padding])
    padded_spatial = x.type.shape[2:]
    out_spatial = [
        (padded_spatial[j] - dilation[j] * (ksize[j] - 1) - 1) // stride[j] + 1
        for j in range(n)
    ]
    out_pts = int(np.prod(out_spatial)) if out_spatial else 1

    dtype = x.type.dtype
    w = _cast_if_needed(w, dtype)

    # im2col: gather every kernel position's strided window as a
    # (N, Ci, out_pts) column block, then concat along the channel axis.
    # Iterate kernel offsets in row-major order so the column order matches
    # the weight's (Ci, *K) C-order flattening below.
    cols = []
    for flat_k in range(int(np.prod(ksize)) if ksize else 1):
        koff = []
        rem = flat_k
        for d in reversed(ksize):
            koff.append(rem % d)
            rem //= d
        koff = list(reversed(koff))

        starts = [0, 0]
        stops = [batch, in_ch]
        strides = [1, 1]
        for j in range(n):
            s0 = koff[j] * dilation[j]
            starts.append(s0)
            stops.append(s0 + (out_spatial[j] - 1) * stride[j] + 1)
            strides.append(stride[j])
        window = b.slice(x, tuple(starts), tuple(stops), tuple(strides))
        cols.append(b.reshape(window, (batch, in_ch, out_pts)))

    # Column order is [k0_ci0..ci_{Ci-1}, k1_ci0..]; to match the weight's
    # (Co, Ci, *K) -> (Co, Ci*prod(K)) flattening (C-order: ci outer, k inner)
    # we transpose the weight to (Co, *K, Ci) before flattening, and likewise
    # the column blocks are ordered by kernel position then channel — which is
    # exactly (prod(K), Ci). Concatenate on channel axis to get that order.
    col = cols[0] if len(cols) == 1 else b.concat(cols, axis=1)  # (N, Ci*prodK, P)

    # Weight: (Co, Ci, *K) -> (Co, *K, Ci) -> (Co, prod(K)*Ci) to match the
    # column ordering (kernel-position outer, channel inner).
    perm = (0,) + tuple(range(2, 2 + n)) + (1,)
    w_t = b.transpose(w, perm)
    w_flat = b.reshape(w_t, (out_ch, int(np.prod(ksize)) * in_ch if ksize else in_ch))

    # (Co, K*Ci) @ (N, K*Ci, P) -> (N, Co, P)
    out = _unwrap(matmul(_wrap(w_flat), _wrap(col)))
    out = b.reshape(out, (batch, out_ch, *out_spatial))

    if bias is not None:
        bias_val = _cast_if_needed(_unwrap(bias), dtype)
        bias_val = b.reshape(bias_val, (1, out_ch) + (1,) * n)
        out = b.add(out, b.broadcast_to(bias_val, out.type.shape))

    return _wrap(out)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1, out=None, dtype=None):
    from nkipy.core.ops.conv import _normalize_tuple_2d
    return _conv_nd(
        input, weight, bias,
        _normalize_tuple_2d(stride, "stride"),
        _normalize_tuple_2d(padding, "padding"),
        _normalize_tuple_2d(dilation, "dilation"),
        groups, n=2,
    )


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1, out=None, dtype=None):
    from nkipy.core.ops.conv import _normalize_tuple_3d
    return _conv_nd(
        input, weight, bias,
        _normalize_tuple_3d(stride, "stride"),
        _normalize_tuple_3d(padding, "padding"),
        _normalize_tuple_3d(dilation, "dilation"),
        groups, n=3,
    )
