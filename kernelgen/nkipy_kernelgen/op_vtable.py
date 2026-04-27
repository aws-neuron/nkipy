"""
VTable (Virtual Table) for mapping NumPy operations to MLIR operations.

This module contains the mapping between NumPy ufuncs/functions and their
corresponding MLIR implementations. It delegates to builder.py for IR
construction via TracedArray↔TensorHandle conversion.
"""

from collections.abc import Callable

import numpy as np
from mlir import ir

from . import builder
from .builder import TensorHandle


# ---------------------------------------------------------------------------
# TracedArray ↔ TensorHandle bridge
# ---------------------------------------------------------------------------


def _to_handle(a):
    """Convert a TracedArray to a TensorHandle for builder.py calls.

    Scalars pass through unchanged — builder's binary dispatch handles them.
    """
    from .traced_array import TracedArray

    if isinstance(a, TracedArray):
        return TensorHandle(a.value, a.shape, a.dtype, a.elem_ty)
    return a


def _from_handle(h: TensorHandle, source_file: str = "unknown"):
    """Convert a TensorHandle back to a TracedArray."""
    from .traced_array import TracedArray

    return TracedArray(h._value, h.shape, h._elem_ty, source_file)


def _source(a, b=None) -> str:
    """Extract source_file from TracedArray operands."""
    from .traced_array import TracedArray

    if isinstance(a, TracedArray):
        return a.source_file
    if b is not None and isinstance(b, TracedArray):
        return b.source_file
    return "unknown"


# ---------------------------------------------------------------------------
# Ufunc operations (unary and binary)
# ---------------------------------------------------------------------------


def _add_op(a, b, loc):
    return _from_handle(builder.add(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _subtract_op(a, b, loc):
    return _from_handle(builder.subtract(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _multiply_op(a, b, loc):
    return _from_handle(builder.multiply(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _divide_op(a, b, loc):
    return _from_handle(builder.divide(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _maximum_op(a, b, loc):
    return _from_handle(builder.maximum(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _minimum_op(a, b, loc):
    return _from_handle(builder.minimum(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _power_op(a, b, loc):
    return _from_handle(builder.power(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _mod_op(a, b, loc):
    return _from_handle(builder.mod(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _square_op(a, loc):
    return _from_handle(builder.square(_to_handle(a), loc=loc), _source(a))


def _sqrt_op(a, loc):
    return _from_handle(builder.sqrt(_to_handle(a), loc=loc), _source(a))


def _exp_op(a, loc):
    return _from_handle(builder.exp(_to_handle(a), loc=loc), _source(a))


def _log_op(a, loc):
    return _from_handle(builder.log(_to_handle(a), loc=loc), _source(a))


def _tanh_op(a, loc):
    return _from_handle(builder.tanh(_to_handle(a), loc=loc), _source(a))


def _ceil_op(a, loc):
    return _from_handle(builder.ceil_(_to_handle(a), loc=loc), _source(a))


def _floor_op(a, loc):
    return _from_handle(builder.floor_(_to_handle(a), loc=loc), _source(a))


def _sin_op(a, loc):
    return _from_handle(builder.sin(_to_handle(a), loc=loc), _source(a))


def _cos_op(a, loc):
    return _from_handle(builder.cos(_to_handle(a), loc=loc), _source(a))


def _sign_op(a, loc):
    return _from_handle(builder.sign(_to_handle(a), loc=loc), _source(a))


def _abs_op(a, loc):
    return _from_handle(builder.abs_(_to_handle(a), loc=loc), _source(a))


def _negative_op(a, loc):
    return _from_handle(builder.negative(_to_handle(a), loc=loc), _source(a))


def _reciprocal_op(a, loc):
    return _from_handle(builder.reciprocal(_to_handle(a), loc=loc), _source(a))


def _greater_equal_op(a, b, loc):
    return _from_handle(builder.greater_equal(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _less_op(a, b, loc):
    return _from_handle(builder.less(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _equal_op(a, b, loc):
    return _from_handle(builder.equal(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _bitwise_and_op(a, b, loc):
    return _from_handle(builder.bitwise_and(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _bitwise_or_op(a, b, loc):
    return _from_handle(builder.bitwise_or(_to_handle(a), _to_handle(b), loc=loc), _source(a, b))


def _logical_not_op(a, loc):
    return _from_handle(builder.logical_not(_to_handle(a), loc=loc), _source(a))


# VTable for NumPy ufuncs
NUMPY_UFUNC_VTABLE: dict[str, Callable] = {
    "add": _add_op,
    "subtract": _subtract_op,
    "multiply": _multiply_op,
    "divide": _divide_op,
    "square": _square_op,
    "sqrt": _sqrt_op,
    "exp": _exp_op,
    "negative": _negative_op,
    "reciprocal": _reciprocal_op,
    "absolute": _abs_op,
    "abs": _abs_op,
    "log": _log_op,
    "sin": _sin_op,
    "cos": _cos_op,
    "tanh": _tanh_op,
    "ceil": _ceil_op,
    "floor": _floor_op,
    "sign": _sign_op,
    "maximum": _maximum_op,
    "minimum": _minimum_op,
    "power": _power_op,
    "remainder": _mod_op,
    "mod": _mod_op,
    "greater_equal": _greater_equal_op,
    "less": _less_op,
    "equal": _equal_op,
    "bitwise_and": _bitwise_and_op,
    "bitwise_or": _bitwise_or_op,
    "logical_not": _logical_not_op,
    "matmul": lambda a, b, loc: _matmul_op((a, b), {}, loc),
}


# ---------------------------------------------------------------------------
# NumPy function vtable operations
# ---------------------------------------------------------------------------


def _matmul_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A, B = args
    if not isinstance(A, TracedArray) or not isinstance(B, TracedArray):
        raise TypeError("matmul requires TracedArray inputs")
    h = builder.matmul(_to_handle(A), _to_handle(B), loc=loc)
    return _from_handle(h, A.source_file)


def _transpose_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("transpose requires TracedArray input")
    axes = args[1] if len(args) > 1 else kwargs.get("axes", None)
    h = builder.transpose(_to_handle(A), axes=axes, loc=loc)
    return _from_handle(h, A.source_file)


def _reshape_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A, newshape = args
    if not isinstance(A, TracedArray):
        raise TypeError("reshape requires TracedArray input")
    h = builder.reshape(_to_handle(A), newshape, loc=loc)
    return _from_handle(h, A.source_file)


def _sum_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("sum requires TracedArray input")
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    h = builder.reduce_sum(_to_handle(A), axis=axis, keepdims=keepdims, loc=loc)
    return _from_handle(h, A.source_file)


def _prod_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("prod requires TracedArray input")
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    h = builder.reduce_prod(_to_handle(A), axis=axis, keepdims=keepdims, loc=loc)
    return _from_handle(h, A.source_file)


def _max_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("max requires TracedArray input")
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    h = builder.reduce_max(_to_handle(A), axis=axis, keepdims=keepdims, loc=loc)
    return _from_handle(h, A.source_file)


def _min_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("min requires TracedArray input")
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    h = builder.reduce_min(_to_handle(A), axis=axis, keepdims=keepdims, loc=loc)
    return _from_handle(h, A.source_file)


def _mean_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("mean requires TracedArray input")
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    h = builder.reduce_mean(_to_handle(A), axis=axis, keepdims=keepdims, loc=loc)
    return _from_handle(h, A.source_file)


def _std_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("std requires TracedArray input")
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    h = builder.reduce_std(_to_handle(A), axis=axis, keepdims=keepdims, loc=loc)
    return _from_handle(h, A.source_file)


def _concatenate_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    arrays = args[0]
    axis = kwargs.get("axis", 0)
    if not all(isinstance(a, TracedArray) for a in arrays):
        raise TypeError("concatenate requires all inputs to be TracedArray")
    handles = [_to_handle(a) for a in arrays]
    h = builder.concatenate(handles, axis=axis, loc=loc)
    return _from_handle(h, arrays[0].source_file)


def _split_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    indices_or_sections = args[1]
    axis = kwargs.get("axis", args[2] if len(args) > 2 else 0)
    if not isinstance(A, TracedArray):
        raise TypeError("split requires TracedArray input")

    if isinstance(indices_or_sections, int):
        handles = builder.split(_to_handle(A), indices_or_sections, axis=axis, loc=loc)
        return tuple(_from_handle(h, A.source_file) for h in handles)
    else:
        raise NotImplementedError("split with explicit indices not yet implemented")


def _expand_dims_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    axis = args[1] if len(args) > 1 else kwargs.get("axis")
    if not isinstance(A, TracedArray):
        raise TypeError("expand_dims requires TracedArray input")
    h = builder.expand_dims(_to_handle(A), axis, loc=loc)
    return _from_handle(h, A.source_file)


def _broadcast_to_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    target_shape = tuple(args[1])
    if not isinstance(A, TracedArray):
        raise TypeError("broadcast_to requires TracedArray input")
    h = builder.broadcast_to(_to_handle(A), target_shape, loc=loc)
    return _from_handle(h, A.source_file)


def _copy_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    A = args[0]
    if not isinstance(A, TracedArray):
        raise TypeError("copy requires TracedArray input")
    h = builder.copy_(_to_handle(A), loc=loc)
    return _from_handle(h, A.source_file)


def _take_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    a = args[0]
    indices = args[1]
    axis = kwargs.get("axis", args[2] if len(args) > 2 else 0)
    if isinstance(indices, TracedArray) and isinstance(indices.elem_ty, ir.FloatType):
        indices_h = builder.astype(_to_handle(indices), np.int32, loc=loc)
        indices = _from_handle(indices_h, indices.source_file)
    h = builder.take(_to_handle(a), _to_handle(indices), axis=axis, loc=loc)
    return _from_handle(h, a.source_file)


def _where_op(args: tuple, kwargs: dict, loc):
    from .traced_array import TracedArray

    condition = args[0]
    x = args[1]
    y = args[2]
    h = builder.where(_to_handle(condition), _to_handle(x), _to_handle(y), loc=loc)
    sf = _source(condition, x if isinstance(x, TracedArray) else y)
    return _from_handle(h, sf)


# VTable for NumPy array functions
NUMPY_FUNCTION_VTABLE: dict[Callable, Callable] = {
    np.matmul: _matmul_op,
    np.transpose: _transpose_op,
    np.reshape: _reshape_op,
    np.sum: _sum_op,
    np.prod: _prod_op,
    np.max: _max_op,
    np.min: _min_op,
    np.mean: _mean_op,
    np.std: _std_op,
    np.concatenate: _concatenate_op,
    np.split: _split_op,
    np.expand_dims: _expand_dims_op,
    np.broadcast_to: _broadcast_to_op,
    np.copy: _copy_op,
    np.take: _take_op,
    np.where: _where_op,
}
