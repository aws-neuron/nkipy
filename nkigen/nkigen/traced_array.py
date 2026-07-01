"""
TracedArray class for intercepting NumPy operations.
"""

import inspect
import os
from typing import Tuple, Optional
import numpy as np
from mlir import ir

from .op_vtable import NUMPY_UFUNC_VTABLE, NUMPY_FUNCTION_VTABLE


class TracedArray:
    """Represents a traced array that builds MLIR operations."""
    
    # Enable NumPy's __array_function__ protocol
    __array_priority__ = 1000
    
    def __init__(self, value: ir.Value, shape: Tuple[int, ...], elem_ty: ir.Type, source_file: Optional[str] = None):
        self.value = value
        self.shape = tuple(shape)
        self.elem_ty = elem_ty
        self.source_file = source_file or "unknown"
    
    @property
    def dtype(self):
        """Return the numpy-compatible dtype of the TracedArray."""
        # Map MLIR type to numpy dtype
        if isinstance(self.elem_ty, ir.FloatType):
            if self.elem_ty.width == 32:
                return np.float32
            elif self.elem_ty.width == 64:
                return np.float64
            elif self.elem_ty.width == 16:
                return np.float16
        elif isinstance(self.elem_ty, ir.IntegerType):
            if self.elem_ty.width == 32:
                return np.int32 if self.elem_ty.is_signed else np.uint32
            elif self.elem_ty.width == 64:
                return np.int64 if self.elem_ty.is_signed else np.uint64
            elif self.elem_ty.width == 16:
                return np.int16 if self.elem_ty.is_signed else np.uint16
            elif self.elem_ty.width == 8:
                return np.int8 if self.elem_ty.is_signed else np.uint8
        return self.elem_ty  # Fallback to MLIR type

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return len(self.shape)

    def __repr__(self) -> str:
        """Return a detailed string representation of the TracedArray."""
        return f"TracedArray(shape={self.shape}, dtype={self.elem_ty}, source={self.source_file})"

    def __str__(self) -> str:
        """Return a user-friendly string representation of the TracedArray."""
        return f"TracedArray{self.shape} of type {self.elem_ty}"
    
    def _get_caller_location(self) -> ir.Location:
        """Get MLIR location from Python call stack."""
        # Walk up the stack to find the user's code (skip internal frames)
        frame = inspect.currentframe()
        try:
            # Walk through all frames and find the first one that matches our source file
            while frame is not None:
                frame = frame.f_back
                if frame is None:
                    break
                    
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                
                if self.source_file != "unknown":
                    source_basename = os.path.basename(self.source_file)
                    frame_basename = os.path.basename(filename)
                    
                    # If the basenames match, or if the source_file is contained in filename
                    if source_basename == frame_basename or self.source_file in filename:
                        ctx = ir.Context.current
                        return ir.Location.file(self.source_file, lineno, 0, context=ctx)
        finally:
            del frame  # Avoid reference cycles
        
        # Fallback to the value's location
        return self.value.location
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept NumPy ufuncs like np.add, np.multiply, np.square, etc."""
        if method != "__call__":
            raise NotImplementedError(f"Method {method} not supported")
        
        # Get location from Python call stack
        loc = self._get_caller_location()
        
        ufunc_name = ufunc.__name__
        
        # Check if this ufunc is supported
        if ufunc_name not in NUMPY_UFUNC_VTABLE:
            raise NotImplementedError(f"ufunc {ufunc_name} not supported")
        
        # Check if this is a unary operation
        if len(inputs) == 1:
            A = inputs[0]
            if not isinstance(A, TracedArray):
                A = lift_constant(A, self.elem_ty, self.shape, loc, self.source_file)
            return NUMPY_UFUNC_VTABLE[ufunc_name](A, loc)
        
        # Check if this is a binary operation
        elif len(inputs) == 2:
            # Pass inputs directly — scalar constants are handled by each op
            A, B = inputs[0], inputs[1]
            return NUMPY_UFUNC_VTABLE[ufunc_name](A, B, loc)
        
        else:
            raise NotImplementedError(f"ufunc {ufunc_name} with {len(inputs)} inputs not supported")
    
    def __array_function__(self, func, types, args, kwargs):
        """Intercept NumPy array functions like matmul, transpose, etc."""
        if func not in NUMPY_FUNCTION_VTABLE:
            return NotImplemented
        
        # Get location from Python call stack
        loc = self._get_caller_location()
        return NUMPY_FUNCTION_VTABLE[func](args, kwargs, loc)
    
    def __getitem__(self, key):
        """Support indexing/slicing using tensor.extract_slice or gather."""
        from .control_flow import LoopIndex
        from .op_vtable import _to_handle, _from_handle
        from . import builder

        if not isinstance(key, tuple):
            key = (key,)

        # Gather path: first dim is a TracedArray
        if len(key) > 0 and isinstance(key[0], TracedArray):
            for i, slice_spec in enumerate(key[1:], start=1):
                if not isinstance(slice_spec, slice):
                    raise TypeError("Gather with TracedArray indices only supports full slices on remaining dimensions")
                if slice_spec != slice(None, None, None):
                    raise NotImplementedError("Partial slicing with gather not yet supported")
            result_h = builder.take(_to_handle(self), _to_handle(key[0]), axis=0)
            return _from_handle(result_h, self.source_file)

        # Check for dynamic indices
        has_dynamic = any(
            isinstance(s, LoopIndex) or (
                isinstance(s, slice) and (
                    isinstance(s.start, LoopIndex) or isinstance(s.stop, LoopIndex)
                )
            )
            for s in key
        )

        self_h = _to_handle(self)
        if has_dynamic:
            indices = self._convert_key_to_builder_indices(key, LoopIndex)
            result_h = builder.dynamic_slice(self_h, indices)
        else:
            start_indices, limit_indices, strides_list, squeeze_dims = \
                self._convert_key_to_static(key)
            result_h = builder.static_slice(
                self_h, start_indices, limit_indices, strides_list, squeeze_dims
            )
        return _from_handle(result_h, self.source_file)

    def _convert_key_to_builder_indices(self, key, LoopIndex):
        """Convert __getitem__ key to builder.dynamic_slice indices tuple."""
        from .builder import LoopIndexHandle
        indices = []
        for slice_spec in key:
            if isinstance(slice_spec, LoopIndex):
                indices.append(LoopIndexHandle(
                    slice_spec.value, slice_spec.mul_factor, slice_spec.add_offset
                ))
            elif isinstance(slice_spec, slice):
                start = slice_spec.start
                stop = slice_spec.stop
                step = slice_spec.step
                new_start = start
                new_stop = stop
                if isinstance(start, LoopIndex):
                    new_start = LoopIndexHandle(
                        start.value, start.mul_factor, start.add_offset
                    )
                if isinstance(stop, LoopIndex):
                    new_stop = LoopIndexHandle(
                        stop.value, stop.mul_factor, stop.add_offset
                    )
                indices.append(slice(new_start, new_stop, step))
            else:
                indices.append(slice_spec)
        return tuple(indices)

    def _convert_key_to_static(self, key):
        """Convert __getitem__ key to static_slice parameters."""
        start_indices = []
        limit_indices = []
        strides_list = []
        squeeze_dims = []

        for dim_idx, (slice_spec, dim_size) in enumerate(zip(key, self.shape)):
            if isinstance(slice_spec, slice):
                start = slice_spec.start if slice_spec.start is not None else 0
                stop = slice_spec.stop if slice_spec.stop is not None else dim_size
                step = slice_spec.step if slice_spec.step is not None else 1
                start_indices.append(int(start))
                limit_indices.append(int(stop))
                strides_list.append(int(step))
            elif isinstance(slice_spec, int):
                start_indices.append(int(slice_spec))
                limit_indices.append(int(slice_spec) + 1)
                strides_list.append(1)
                squeeze_dims.append(dim_idx)
            else:
                raise TypeError(f"Unsupported index type: {type(slice_spec)}")

        for dim_idx in range(len(key), len(self.shape)):
            start_indices.append(0)
            limit_indices.append(self.shape[dim_idx])
            strides_list.append(1)

        return start_indices, limit_indices, strides_list, squeeze_dims
    
    def __setitem__(self, key, value):
        """Support item assignment using tensor.insert_slice.

        Since MLIR tensors are SSA values (immutable), this updates self.value
        to a new tensor with the slice inserted.
        """
        from .control_flow import LoopIndex
        from .op_vtable import _to_handle, _from_handle
        from . import builder

        if not isinstance(value, TracedArray):
            raise TypeError(f"Can only assign TracedArray values, got {type(value)}")

        if not isinstance(key, tuple):
            key = (key,)

        # Expand value for any single-index dims that were collapsed.
        insert_val = value
        for dim_idx, slice_spec in enumerate(key):
            if isinstance(slice_spec, (int, LoopIndex)):
                insert_val = np.expand_dims(insert_val, axis=dim_idx)

        has_dynamic = any(
            isinstance(s, LoopIndex) or (
                isinstance(s, slice) and (
                    isinstance(s.start, LoopIndex) or isinstance(s.stop, LoopIndex)
                )
            )
            for s in key
        )

        self_h = _to_handle(self)
        insert_h = _to_handle(insert_val)

        if has_dynamic:
            indices = self._convert_key_to_builder_indices(key, LoopIndex)
            result_h = builder.dynamic_insert_slice(self_h, insert_h, indices)
        else:
            offsets, sizes, strides_list = [], [], []

            for slice_spec, dim_size in zip(key, self.shape):
                if isinstance(slice_spec, slice):
                    start = slice_spec.start if slice_spec.start is not None else 0
                    stop = slice_spec.stop if slice_spec.stop is not None else dim_size
                    step = slice_spec.step if slice_spec.step is not None else 1
                    offsets.append(int(start))
                    sizes.append(int(stop - start))
                    strides_list.append(int(step))
                elif isinstance(slice_spec, int):
                    offsets.append(int(slice_spec))
                    sizes.append(1)
                    strides_list.append(1)
                else:
                    raise TypeError(f"Unsupported index type: {type(slice_spec)}")

            for dim_idx in range(len(key), len(self.shape)):
                offsets.append(0)
                sizes.append(self.shape[dim_idx])
                strides_list.append(1)

            result_h = builder.static_insert_slice(
                self_h, insert_h, offsets, sizes, strides_list
            )

        self.value = result_h._value

    def astype(self, dtype, **kwargs):
        """Cast array to specified dtype."""
        from .op_vtable import _to_handle, _from_handle
        from . import builder

        result_h = builder.astype(_to_handle(self), dtype)
        return _from_handle(result_h, self.source_file)

    def reshape(self, *shape):
        """Reshape the array using np.reshape."""
        # Flatten shape if it's passed as a tuple
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        # Handle -1 in shape (infer dimension)
        if -1 in shape:
            # Compute the inferred dimension
            known_size = 1
            unknown_idx = None
            for i, dim in enumerate(shape):
                if dim == -1:
                    if unknown_idx is not None:
                        raise ValueError("can only specify one unknown dimension")
                    unknown_idx = i
                else:
                    known_size *= dim
            total_size = np.prod(self.shape)
            inferred_dim = total_size // known_size
            shape = list(shape)
            shape[unknown_idx] = inferred_dim
            shape = tuple(shape)
        return np.reshape(self, shape)

    # Python operator overloads to trigger NumPy ufuncs
    def __add__(self, other):        return np.add(self, other)
    def __sub__(self, other):        return np.subtract(self, other)
    def __mul__(self, other):        return np.multiply(self, other)
    def __truediv__(self, other):    return np.divide(self, other)
    def __radd__(self, other):       return np.add(other, self)
    def __rsub__(self, other):       return np.subtract(other, self)
    def __rmul__(self, other):       return np.multiply(other, self)
    def __rtruediv__(self, other):   return np.divide(other, self)
    def __neg__(self):               return np.negative(self)
    def __matmul__(self, other):     return np.matmul(self, other)
    def __rmatmul__(self, other):    return np.matmul(other, self)


def lift_constant(val: float, elem_ty: ir.Type, shape: Tuple[int, ...], loc: ir.Location, source_file: str = "unknown") -> TracedArray:
    """Convert a scalar constant to a TracedArray by filling a tensor.

    The result is annotated with CONSTANT memory space (mem_space=5) to indicate
    that this tensor represents a broadcasted scalar constant. This allows later
    passes to optimize tensor-scalar operations (e.g., use nisa.tensor_scalar_arith
    instead of nisa.tensor_tensor_arith).
    """
    from . import builder

    h = builder.constant_tensor(val, shape, elem_ty)
    return TracedArray(h._value, shape, elem_ty, source_file)
