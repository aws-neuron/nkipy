"""
Trace decorator for converting Python functions with NumPy operations to MLIR.
"""

import contextlib
import functools
import inspect
from typing import Callable, Optional

import numpy as np
from mlir import ir

from .builder import IRBuilder
from . import builder
from .traced_array import TracedArray
from .op_vtable import _to_handle, _from_handle
from .custom_op import _get_registry, _clear_registry


def _normalize_shape(shape):
    if isinstance(shape, TracedArray):
        return tuple(shape.shape)
    if isinstance(shape, (list, tuple)):
        return tuple(int(d) for d in shape)
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


@contextlib.contextmanager
def _numpy_constructor_patch(source_file: str):
    """Patch NumPy constructors during tracing so they emit MLIR and return TracedArray."""
    originals = {}

    def _patch(name, fn):
        originals[name] = getattr(np, name)
        setattr(np, name, fn)

    def _make_loc():
        return ir.Location.file(source_file, 0, 0, context=ir.Context.current)

    def ones(shape, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.full(shp, 1.0, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def zeros(shape, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.zeros(shp, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def full(shape, fill_value, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.full(shp, fill_value, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def empty(shape, dtype=None, **kw):
        shp = _normalize_shape(shape)
        h = builder.empty(shp, dtype or np.float32, loc=_make_loc())
        return _from_handle(h, source_file)

    def array(obj, dtype=None, **kw):
        if isinstance(obj, TracedArray):
            return obj
        return originals["array"](obj, dtype=dtype, **kw)

    def asarray(obj, dtype=None, **kw):
        return array(obj, dtype=dtype, **kw)

    try:
        for name, fn in (
            ("ones", ones),
            ("zeros", zeros),
            ("empty", empty),
            ("full", full),
            ("array", array),
            ("asarray", asarray),
        ):
            _patch(name, fn)
        yield
    finally:
        for k, v in originals.items():
            setattr(np, k, v)


def trace(
    func_to_trace: Optional[Callable] = None,
    *,
    input_specs: Optional[list] = None,
    name: Optional[str] = None,
) -> Callable:
    """Decorator to trace a Python function with NumPy APIs into MLIR."""

    def decorator(f: Callable) -> Callable:
        func_name = name or f.__name__
        source_file = inspect.getsourcefile(f) or "unknown"

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        def to_mlir(specs: Optional[list] = None, debug: bool = False):
            """Generate MLIR module from the traced function."""
            nonlocal input_specs
            specs = specs or input_specs
            if not specs:
                raise ValueError(
                    "input_specs must be provided either in decorator or to_mlir()"
                )

            _clear_registry()

            b = IRBuilder(source_file=source_file)
            arg_shapes = [s for s, _ in specs]
            arg_dtypes = [d for _, d in specs]
            handles = b.begin_function(func_name, arg_shapes, arg_dtypes)

            traced_args = [
                TracedArray(h._value, h.shape, h._elem_ty, source_file=source_file)
                for h in handles
            ]

            try:
                with _numpy_constructor_patch(source_file):
                    result = f(*traced_args)

                if isinstance(result, tuple):
                    results = list(result)
                elif isinstance(result, TracedArray):
                    results = [result]
                else:
                    raise TypeError(
                        f"Result must be a TracedArray or tuple of TracedArrays, got {type(result)}"
                    )

                for i, r in enumerate(results):
                    if not isinstance(r, TracedArray):
                        raise TypeError(
                            f"Result element {i} must be a TracedArray, got {type(r)}"
                        )

                result_handles = [_to_handle(r) for r in results]
                b.finish_function(result_handles)

                custom_ops = _get_registry()
                b.emit_custom_op_declarations(custom_ops)

                b.run_canonicalize()
                module = b.module
                return module
            finally:
                _clear_registry()
                b.cleanup()

        wrapper.to_mlir = to_mlir
        wrapper.__traced__ = True
        wrapper.input_specs = input_specs

        return wrapper

    if func_to_trace is None:
        return decorator
    else:
        return decorator(func_to_trace)
