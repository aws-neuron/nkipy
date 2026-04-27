"""
Control flow operations for traced execution.
"""

from typing import Callable, Union

from . import builder
from .builder import LoopIndexHandle
from .traced_array import TracedArray


class LoopIndex:
    """Wrapper for loop index that supports arithmetic operations and tracks constants."""

    def __init__(self, value_or_handle, mul_factor: int = 1, add_offset: int = 0):
        if isinstance(value_or_handle, LoopIndexHandle):
            self._handle = value_or_handle
        else:
            self._handle = LoopIndexHandle(value_or_handle, mul_factor, add_offset)

    @property
    def value(self):
        return self._handle._value

    @property
    def mul_factor(self):
        return self._handle.mul_factor

    @property
    def add_offset(self):
        return self._handle.add_offset

    def __mul__(self, other):
        if isinstance(other, int):
            return LoopIndex(builder.loop_index_mul(self._handle, other))
        raise TypeError(
            f"LoopIndex * {type(other).__name__} is not supported; only integer constants are allowed"
        )

    def __add__(self, other):
        if isinstance(other, int):
            return LoopIndex(builder.loop_index_add(self._handle, other))
        elif isinstance(other, LoopIndex):
            return LoopIndex(builder.loop_index_add_loop_index(self._handle, other._handle))
        raise TypeError(
            f"LoopIndex + {type(other).__name__} is not supported; use int or LoopIndex"
        )

    def __sub__(self, other):
        if isinstance(other, int):
            return LoopIndex(builder.loop_index_add(self._handle, -other))
        raise TypeError(
            f"LoopIndex - {type(other).__name__} is not supported; only integer constants are allowed"
        )

    def __rsub__(self, other):
        if isinstance(other, int):
            neg = LoopIndex(builder.loop_index_mul(self._handle, -1))
            return LoopIndex(builder.loop_index_add(neg._handle, other))
        raise TypeError(
            f"{type(other).__name__} - LoopIndex is not supported; only integer constants are allowed"
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


def _to_handle(val):
    """Convert a TracedArray, scalar, or value to a TensorHandle for builder calls."""
    from .op_vtable import _to_handle as vtable_to_handle

    if isinstance(val, TracedArray):
        return vtable_to_handle(val)
    if isinstance(val, (int, float)):
        return builder.lift_scalar_to_tensor(val)
    raise TypeError(f"init_val must be TracedArray, int, or float, got {type(val)}")


def _from_handle(h, source_file):
    """Convert a TensorHandle back to a TracedArray."""
    from .op_vtable import _from_handle as vtable_from_handle

    return vtable_from_handle(h, source_file)


def fori_loop(
    lower_bound: Union[int, TracedArray],
    upper_bound: Union[int, TracedArray],
    body_fn: Callable,
    init_val: Union[TracedArray, float, int, tuple],
) -> Union[TracedArray, tuple]:
    is_tuple = isinstance(init_val, tuple)
    if is_tuple:
        is_tracing = any(isinstance(v, TracedArray) for v in init_val)
    else:
        is_tracing = isinstance(init_val, TracedArray)

    if not is_tracing:
        acc = init_val
        for i in range(lower_bound, upper_bound):
            acc = body_fn(i, acc)
        return acc

    if is_tuple:
        init_handles = [_to_handle(v) for v in init_val]
        source_files = [
            v.source_file if isinstance(v, TracedArray) else "unknown"
            for v in init_val
        ]
    else:
        init_handles = [_to_handle(init_val)]
        source_files = [
            init_val.source_file if isinstance(init_val, TracedArray) else "unknown"
        ]

    if not isinstance(lower_bound, int):
        raise TypeError("Dynamic lower bounds not yet supported")
    if not isinstance(upper_bound, int):
        raise TypeError("Dynamic upper bounds not yet supported")

    from .op_vtable import _to_handle as vtable_to_handle

    def wrapped_body(loop_idx_handle, acc_handles):
        loop_idx = LoopIndex(loop_idx_handle)

        if is_tuple:
            accs = tuple(
                _from_handle(ah, sf) for ah, sf in zip(acc_handles, source_files)
            )
            result = body_fn(loop_idx, accs)
            if not isinstance(result, tuple):
                raise TypeError(
                    f"body_fn must return a tuple of {len(init_handles)} elements, "
                    f"got {type(result).__name__}"
                )
            if len(result) != len(init_handles):
                raise TypeError(
                    f"body_fn must return a tuple of {len(init_handles)} elements, "
                    f"got tuple of {len(result)}"
                )
            return [vtable_to_handle(r) for r in result]
        else:
            acc = _from_handle(acc_handles[0], source_files[0])
            result = body_fn(loop_idx, acc)
            if not isinstance(result, TracedArray):
                raise TypeError(
                    f"body_fn must return a TracedArray, got {type(result).__name__}"
                )
            return [vtable_to_handle(result)]

    result_handles = builder.fori_loop(
        lower_bound, upper_bound, wrapped_body, init_handles
    )

    if is_tuple:
        return tuple(
            _from_handle(rh, sf) for rh, sf in zip(result_handles, source_files)
        )
    else:
        return _from_handle(result_handles[0], source_files[0])
