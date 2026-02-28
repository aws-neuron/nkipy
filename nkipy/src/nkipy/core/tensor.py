# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor representation for NKIPy kernels

This module provides the NKIPyTensorRef class which is the main tensor type
used during tracing. It is backend-agnostic and delegates operations to
the ops module which dispatches to the appropriate backend.
"""

import inspect
import math
from typing import Tuple, Union

import numpy as np


def find_source_loc() -> Union[None, Tuple[str, int]]:
    """Find the source location of the calling code.

    This walks up the call stack to find the first frame that's not
    in the nkipy/core modules.

    Returns:
        Tuple of (filename, line_number) or None if extraction fails.
    """
    # FIXME: this a very opportunistic way, need to find a more robust way
    try:
        # first get out of this helper function, and __array_xxx__ dispatcher
        frame = inspect.currentframe().f_back.f_back
        # find the last frame that's not in our codebase
        while "nkipy/core" in frame.f_code.co_filename:
            frame = frame.f_back
        # hopefully we land on the actual source code
        source_info = (frame.f_code.co_filename, frame.f_lineno)
        return source_info
    except Exception as e:
        print(f"line extraction failed with exception {e}")
        return None


def _set_source_location(location):
    """Set the source location in the current context.

    This is a lazy import to avoid circular dependencies.
    """
    from nkipy.core.backend import set_source_location

    set_source_location(location)


class TensorArithmeticMixin:
    # Basic arithmetic operations
    # +
    def __add__(self, other):
        return np.add(self, other)

    # -
    def __sub__(self, other):
        return np.subtract(self, other)

    # *
    def __mul__(self, other):
        return np.multiply(self, other)

    # /
    def __truediv__(self, other):
        return np.divide(self, other)

    # //
    def __floordiv__(self, other):
        return np.floor_divide(self, other)

    # %
    def __mod__(self, other):
        return np.mod(self, other)

    # **
    def __pow__(self, other):
        return np.power(self, other)

    # @
    def __matmul__(self, other):
        return np.matmul(self, other)

    # Reverse operations
    # other + self
    def __radd__(self, other):
        return np.add(other, self)

    # other - self
    def __rsub__(self, other):
        return np.subtract(other, self)

    # other * self
    def __rmul__(self, other):
        return np.multiply(other, self)

    # other / self
    def __rtruediv__(self, other):
        return np.divide(other, self)

    # other // self
    def __rfloordiv__(self, other):
        return np.floor_divide(other, self)

    # other % self
    def __rmod__(self, other):
        return np.mod(other, self)

    # other ** self
    def __rpow__(self, other):
        return np.power(other, self)

    # other @ self
    def __rmatmul__(self, other):
        return np.matmul(other, self)

    # Unary operations
    # abs(self)
    def __abs__(self):
        return np.abs(self)

    # -self
    def __neg__(self):
        return np.negative(self)

    # +self
    def __pos__(self):
        # NB: equivalent to np.copy
        return np.copy(self)

    # In-place operations (reusing regular operations)
    # +=
    def __iadd__(self, other):
        return self.__add__(other)

    # -=
    def __isub__(self, other):
        return self.__sub__(other)

    # *=
    def __imul__(self, other):
        return self.__mul__(other)

    # /=
    def __itruediv__(self, other):
        return self.__truediv__(other)

    # //=
    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    # %=
    def __imod__(self, other):
        return self.__mod__(other)

    # **=
    def __ipow__(self, other):
        return self.__pow__(other)

    # @=
    def __imatmul__(self, other):
        return self.__matmul__(other)

    # Comparison operations
    # <
    def __lt__(self, other):
        return np.less(self, other)

    # <=
    def __le__(self, other):
        return np.less_equal(self, other)

    # >
    def __gt__(self, other):
        return np.greater(self, other)

    # >=
    def __ge__(self, other):
        return np.greater_equal(self, other)

    # ==
    def __eq__(self, other):
        return np.equal(self, other)

    # !=
    def __ne__(self, other):
        return np.not_equal(self, other)


class TensorOperationMixin:
    def transpose(self, *axes):
        if len(axes) == 0:
            return np.transpose(self, axes=None)
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            return np.transpose(self, axes[0])  # type: ignore
        else:
            return np.transpose(self, axes)  # type: ignore

    def reshape(self, *shape):
        if len(shape) == 0:
            raise TypeError("reshape() takes at least 1 argument (0 given)")
        elif len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return np.reshape(self, shape[0])  # type: ignore
        else:
            return np.reshape(self, shape)  # type: ignore

    def astype(self, dtype):
        from nkipy.core import ops as nkipy_ops

        return nkipy_ops.astype(self, dtype=dtype)


def _expand_ellipsis(indices: tuple, ndim: int) -> tuple:
    """Expand ``...`` into the correct number of ``slice(None)``."""
    ellipsis_count = sum(1 for idx in indices if idx is Ellipsis)
    if ellipsis_count == 0:
        return indices
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis (...)")
    num_real = sum(1 for idx in indices if idx is not None and idx is not Ellipsis)
    num_expand = ndim - num_real
    if num_expand < 0:
        raise IndexError(
            f"too many indices for tensor: tensor is {ndim}-dimensional, "
            f"but {num_real} non-ellipsis indices were given"
        )
    new_indices = ()
    for idx in indices:
        if idx is Ellipsis:
            new_indices += (slice(None),) * num_expand
        else:
            new_indices += (idx,)
    return new_indices


def _strip_newaxis(indices: tuple) -> tuple:
    """Remove None (np.newaxis) from indices; return cleaned indices and expand axes."""
    cleaned = []
    newaxis_axes = []
    output_pos = 0
    for idx in indices:
        if idx is None:
            newaxis_axes.append(output_pos)
            output_pos += 1
        elif isinstance(idx, int):
            cleaned.append(idx)  # int consumes a dim but produces no output dim
        else:
            cleaned.append(idx)  # slice / tensor / list → produces an output dim
            output_pos += 1
    return tuple(cleaned), newaxis_axes


class NKIPyTensorRef(TensorArithmeticMixin, TensorOperationMixin):
    """
    NKIPy Tensor Reference

    This class uses mixins to provide arithmetic operations and implements
    the numpy array protocol to intercept operations during tracing.

    The tensor is backend-agnostic - it holds a reference to a backend tensor
    (e.g., HLOTensor for HLO backend) and delegates operations to the ops
    module which dispatches to the appropriate backend implementation.
    """

    __slots__ = (
        "backend_tensor",
        "_shape",
        "_dtype",
        "_name",
        "_original_parameter",
        "_is_mutated",
    )
    _tensor_apis = {}

    def __init__(self, backend_tensor, name: str = None):
        """
        Initialize tensor reference

        Args:
            backend_tensor: the target backend tensor, e.g., HLOTensor for HLO backend
            name: Optional name for the tensor
        """
        self.backend_tensor = backend_tensor
        self._shape = backend_tensor.shape
        self._dtype = backend_tensor.dtype
        self._name = name
        # Kept for debugging: stores the original HLO parameter before mutations
        self._original_parameter = None
        self._is_mutated = False

    @property
    def name(self):
        return self._name

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def T(self):
        return self.transpose()

    @property
    def size(self) -> int:
        return math.prod(self._shape)

    def __repr__(self):
        return (
            f"NKIPyTensorRef(name={self._name}, shape={self.shape}, dtype={self.dtype})"
        )

    def astype(self, dtype):
        """Convert tensor to a different dtype"""
        # Set source location in context
        _set_source_location(find_source_loc())
        try:
            from nkipy.core import ops as nkipy_ops

            return nkipy_ops.astype(self, dtype)
        finally:
            _set_source_location(None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle numpy universal functions

        This is called when numpy ufuncs are applied to NKIPyTensorRef objects.
        We dispatch to registered implementations.
        """
        if method != "__call__":
            return NotImplemented

        _set_source_location(find_source_loc())
        try:
            return self.__array_function__(ufunc, (), inputs, kwargs)
        finally:
            _set_source_location(None)

    def __array_function__(self, func, types, args, kwargs):
        """
        Handle numpy array functions

        This is called when numpy functions are applied to NKIPyTensorRef objects.
        We dispatch to registered implementations.
        """
        try:
            impl = self._tensor_apis[func]
        except KeyError:
            raise NotImplementedError(
                f"NumPy function '{func.__name__}' has not been implemented in NKIPy."
            )

        # Set source location in context for direct numpy function calls (not ufuncs)
        # Ufuncs go through __array_ufunc__ which already sets the location
        _set_source_location(find_source_loc())
        try:
            return impl(*args, **kwargs)
        finally:
            _set_source_location(None)

    def __getitem__(self, indices):
        """
        Handle indexing operations like a[:, t, :]

        This method translates Python indexing syntax into backend operations.
        It supports:
        - Integer indexing: a[0]
        - Slice indexing: a[0:5]
        - Tensor indexing: a[:, t, :] where t is a tensor
        - Mixed indexing: a[0:2, t, :] - static slice then dynamic index
        """
        from nkipy.core import ops as nkipy_ops

        _set_source_location(find_source_loc())
        try:
            # Normalize indices to a tuple
            if not isinstance(indices, tuple):
                indices = (indices,)

            indices = _expand_ellipsis(indices, len(self.shape))
            indices, newaxis_axes = _strip_newaxis(indices)

            # Pad with full slices if needed
            while len(indices) < len(self.shape):
                indices = indices + (slice(None),)

            # Check if we have any tensor indices (dynamic indexing)
            # This includes NKIPyTensorRef, np.ndarray, and Python lists
            has_tensor_index = any(
                isinstance(idx, (NKIPyTensorRef, np.ndarray, list)) for idx in indices
            )

            # Check for boolean indexing (not supported)
            for idx in indices:
                if (
                    isinstance(idx, (NKIPyTensorRef, np.ndarray))
                    and idx.dtype == np.bool_
                ):
                    raise NotImplementedError(
                        "Boolean indexing is not supported. "
                        "Boolean masks result in variable-length "
                        "outputs which cannot be statically determined at compile time."
                    )

            if has_tensor_index:
                # Mixed static/dynamic indexing:
                #   apply static slicing first, then dynamic indexing
                # For example: a[0:2, seq_indices, :] should:
                # 1. Slice to get a[0:2, :, :]
                # 2. Then apply dynamic indexing on the result

                # Separate static and dynamic indices
                static_indices = []
                dynamic_index_dim = None
                dynamic_index_value = None

                for dim, idx in enumerate(indices):
                    if isinstance(idx, (NKIPyTensorRef, np.ndarray, list)):
                        if dynamic_index_dim is not None:
                            raise ValueError(
                                "Only one tensor index is currently supported"
                            )
                        dynamic_index_dim = dim
                        dynamic_index_value = idx
                        # For the static slice, use full slice for this dimension
                        static_indices.append(slice(None))
                    else:
                        static_indices.append(idx)

                # Apply static slicing first if there are any non-full slices
                has_static_slice = any(
                    not (isinstance(idx, slice) and idx == slice(None))
                    for i, idx in enumerate(static_indices)
                    if i != dynamic_index_dim
                )

                if has_static_slice:
                    # Apply static slicing first
                    sliced_tensor = self._do_static_slice(tuple(static_indices))
                    # Now apply dynamic indexing on the sliced result
                    # Need to adjust the dimension index after slicing
                    # Count how many dimensions were removed before dynamic_index_dim
                    dims_removed_before = sum(
                        1
                        for i, idx in enumerate(static_indices[:dynamic_index_dim])
                        if isinstance(idx, int)
                    )
                    adjusted_dim = dynamic_index_dim - dims_removed_before

                    # Use np.take for dynamic indexing (dispatches to ops.take)
                    result = nkipy_ops.take(
                        sliced_tensor, dynamic_index_value, axis=adjusted_dim
                    )
                else:
                    # No static slicing needed, just dynamic indexing
                    result = nkipy_ops.take(
                        self, dynamic_index_value, axis=dynamic_index_dim
                    )
            else:
                # Pure static indexing
                result = self._do_static_slice(indices)

            if newaxis_axes:
                result = nkipy_ops.expand_dims(result, axis=newaxis_axes)
            return result
        finally:
            _set_source_location(None)

    def __setitem__(self, indices, value):
        """
        Handle slice assignment operations like a[:, t, :] = b

        IMPORTANT: In tracing backends (HLO), operations create a new tensor rather than
        mutating in place. This means we need to create a new tensor with the
        updated values.

        However, Python's __setitem__ doesn't return a value. To work around this,
        we update self.backend_tensor to point to the new tensor. This maintains
        the illusion of in-place mutation while actually creating a new tensor
        in the computation graph.

        Note on view aliasing: Mutations through views are NOT tracked. If you
        write ``b = a[0]; b[x] = y``, the mutation to ``b`` will not propagate
        back to ``a`` because ``__getitem__`` creates a new NKIPyTensorRef with
        no parent link. Use ``a[0, x] = y`` instead.
        """
        _set_source_location(find_source_loc())
        try:
            # Normalize indices to a tuple
            if not isinstance(indices, tuple):
                indices = (indices,)

            if any(idx is None for idx in indices):
                raise IndexError(
                    "cannot use a newaxis (None) in a setitem expression. "
                    "Use np.expand_dims() on the value instead."
                )

            indices = _expand_ellipsis(indices, len(self.shape))

            # Pad with full slices if needed
            while len(indices) < len(self.shape):
                indices = indices + (slice(None),)

            # Check if we have any tensor indices (dynamic indexing)
            has_tensor_index = any(isinstance(idx, NKIPyTensorRef) for idx in indices)

            if has_tensor_index:
                self._is_mutated = True
                # Use scatter for dynamic indexing via put_along_axis
                self._do_scatter_indexing(indices, value)
            else:
                self._is_mutated = True
                # Use static slice assignment
                self._do_static_slice_assignment(indices, value)
        finally:
            _set_source_location(None)

    def _do_static_slice(self, indices):
        """Perform static slicing using the static_slice op.

        Args:
            indices: Tuple of slice/int indices

        Returns:
            Sliced tensor
        """
        from nkipy.core import ops as nkipy_ops

        start_indices = []
        limit_indices = []
        strides = []
        squeeze_dims = []  # Dimensions to squeeze (from integer indexing)

        for dim, idx in enumerate(indices):
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.shape[dim]
                step = idx.step if idx.step is not None else 1

                # Normalize negative indices
                if start < 0:
                    start = self.shape[dim] + start
                if stop < 0:
                    stop = self.shape[dim] + stop

                start_indices.append(start)
                limit_indices.append(stop)
                strides.append(step)
            elif isinstance(idx, int):
                # Integer index - this dimension will be squeezed
                if idx < 0:
                    idx = self.shape[dim] + idx
                start_indices.append(idx)
                limit_indices.append(idx + 1)
                strides.append(1)
                squeeze_dims.append(dim)
            else:
                raise ValueError(
                    f"Unsupported index type in static slicing: {type(idx)}"
                )

        return nkipy_ops.static_slice(
            self, start_indices, limit_indices, strides, squeeze_dims
        )

    def _do_scatter_indexing(self, indices, value):
        """Handle dynamic slice assignment using scatter operation.

        This updates self.backend_tensor to point to the new scattered tensor.
        """
        # Find the tensor index
        tensor_idx_dim = None
        tensor_idx_value = None

        for dim, idx in enumerate(indices):
            if isinstance(idx, NKIPyTensorRef):
                if tensor_idx_dim is not None:
                    raise ValueError("Only one tensor index is currently supported")
                tensor_idx_dim = dim
                tensor_idx_value = idx

        # Use scatter_along_axis (window-level scatter with 1D indices)
        from nkipy.core import ops as nkipy_ops

        result = nkipy_ops.scatter_along_axis(
            self, tensor_idx_value, value, axis=tensor_idx_dim
        )

        # Update self to point to the new tensor
        if result is not None:
            self.backend_tensor = result.backend_tensor
            self._shape = result.shape
            self._dtype = result.dtype

    def _do_static_slice_assignment(self, indices, value):
        """Handle static slice assignment.

        For contiguous slices (step=1), uses dynamic_update_slice.
        For strided slices (step>1), uses scatter_strided.
        """
        # Check if any dimension has step > 1
        has_stride = any(
            isinstance(idx, slice) and (idx.step is not None and idx.step > 1)
            for idx in indices
        )

        if has_stride:
            # Use scatter_strided for strided assignment
            self._do_scatter_strided_assignment(indices, value)
        else:
            # Use dynamic_update_slice for contiguous assignment
            self._do_dynamic_update_slice_assignment(indices, value)

    def _do_dynamic_update_slice_assignment(self, indices, value):
        """Handle contiguous slice assignment using dynamic_update_slice op.

        For a[0:3, :4] = b, we need:
        - operand: a
        - update: b (must match slice shape, including size-1 dims for integer indices)
        - start_indices: [0, 0]
        """
        from nkipy.core import ops as nkipy_ops

        # Calculate update shape and start indices
        update_shape = []
        start_indices = []

        for dim, idx in enumerate(indices):
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.shape[dim]
                step = idx.step if idx.step is not None else 1
                if start < 0:
                    start = self.shape[dim] + start
                if stop < 0:
                    stop = self.shape[dim] + stop
                size = (stop - start + step - 1) // step
                update_shape.append(size)
                start_indices.append(start)
            elif isinstance(idx, int):
                # Integer index means this dimension should have size 1 in the update
                if idx < 0:
                    idx = self.shape[dim] + idx
                update_shape.append(1)
                start_indices.append(idx)

        update_shape = tuple(update_shape)

        # Call the op
        result = nkipy_ops.dynamic_update_slice(
            self, value, start_indices, update_shape
        )

        # Update self to point to the new tensor
        self.backend_tensor = result.backend_tensor
        self._shape = result.shape
        self._dtype = result.dtype

    def _do_scatter_strided_assignment(self, indices, value):
        """Handle strided slice assignment using scatter_strided op.

        For a[::2, ::2] = b, scatter values to strided positions.
        """
        from nkipy.core import ops as nkipy_ops

        # Generate scatter indices for strided positions
        scatter_indices_per_dim = []
        for dim, idx in enumerate(indices):
            if isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.shape[dim]
                step = idx.step if idx.step is not None else 1
                if start < 0:
                    start = self.shape[dim] + start
                if stop < 0:
                    stop = self.shape[dim] + stop

                # Generate indices: start, start+step, start+2*step, ...
                dim_indices = list(range(start, stop, step))
                scatter_indices_per_dim.append(dim_indices)
            elif isinstance(idx, int):
                if idx < 0:
                    idx = self.shape[dim] + idx
                scatter_indices_per_dim.append([idx])

        # Call the op
        result = nkipy_ops.scatter_strided(self, value, scatter_indices_per_dim)

        # Update self to point to the new tensor
        self.backend_tensor = result.backend_tensor
        self._shape = result.shape
        self._dtype = result.dtype
