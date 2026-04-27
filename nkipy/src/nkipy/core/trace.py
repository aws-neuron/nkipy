# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace wrappers to lower NKIPy kernels"""

import inspect
import warnings

import numpy as np

from nkipy.core._numpy_dispatch import register_all_numpy_apis
from nkipy.core.backend import AliasInfo, tracing
from nkipy.core.backend.hlo import (
    HLOModule,
    HLOTraceContext,
    get_hlo_context,
)
from nkipy.core.tensor import NKIPyTensorRef

# Register numpy APIs to use ops module implementations
# This replaces the tensor_apis registration
register_all_numpy_apis()

# Dtypes unsupported by Neuron hardware that should be auto-downcast
_DTYPE_DOWNCAST = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
}


def _sanitize_array_dtype(arr: np.ndarray, name: str = "") -> np.ndarray:
    """Downcast arrays with unsupported dtypes (float64, int64, uint64).

    Returns the array unchanged if its dtype is already supported, or a
    downcasted copy with a user-visible warning otherwise.
    """
    target = _DTYPE_DOWNCAST.get(arr.dtype)
    if target is None:
        return arr
    label = f" '{name}'" if name else ""
    warnings.warn(
        f"Input tensor{label} has dtype {arr.dtype} which is not supported by "
        f"NeuronCore hardware. Automatically casting to {target}.",
        stacklevel=3,
    )
    return arr.astype(target)


def _convert_args(sig, boundargs, convert_arg):
    """Convert bound arguments to traced tensor refs.

    Shared by both HLO and kernelgen specialization paths.

    Each argument is passed through *convert_arg* which replaces ndarrays
    with backend-specific tensor refs and returns non-tensor values unchanged.
    VAR_POSITIONAL and VAR_KEYWORD arguments are expanded so each element is
    converted individually.

    Args:
        sig: The function's inspect.Signature.
        boundargs: The BoundArguments (already defaulted).
        convert_arg: ``(name, arg) -> converted_value``.

    Returns:
        ``(converted_args, converted_kwargs)`` ready to call the kernel.
    """
    converted_args = []
    converted_kwargs = {}

    for name, arg in boundargs.arguments.items():
        param = sig.parameters[name]

        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            converted_args.append(convert_arg(name, arg))
        elif param.kind == param.KEYWORD_ONLY:
            converted_kwargs[name] = convert_arg(name, arg)
        elif param.kind == param.VAR_POSITIONAL:
            for item in arg:
                converted_args.append(convert_arg(name, item))
        elif param.kind == param.VAR_KEYWORD:
            for k, v in arg.items():
                converted_kwargs[k] = convert_arg(k, v)

    return converted_args, converted_kwargs


class NKIPyKernel:
    """Simplified kernel wrapper for NKIPy tracing"""

    def __init__(self, func, backend):
        self.func = func
        self.backend = backend
        self._code = None

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return f"<NKIPyKernel '{self.func.__name__}' backend='{self.backend}'>"

    def specialize(self, *args, **kwargs):
        if self.backend == "hlo":
            return self._specialize_hlo(*args, **kwargs)
        elif self.backend == "kernelgen":
            return self._specialize_kernelgen(*args, **kwargs)
        elif self.backend == "cpu":
            warnings.warn(
                "CPU backend does not require specialization", stacklevel=2
            )
            return
        else:
            raise ValueError(f"Unknown backend {self.backend}")

    def _create_parameter_hlo(self, shape, dtype, name=""):
        """Create an HLO parameter tensor."""
        ctx = get_hlo_context()
        hlo_tensor = ctx.module.add_parameter(shape, dtype, name=name)
        return NKIPyTensorRef(hlo_tensor, name=name)

    def _specialize_hlo(self, *args, **kwargs):
        """Trace the kernel with specific arguments."""

        code = HLOModule(name=self.func.__name__)

        with tracing(HLOTraceContext(code)):
            sig = inspect.signature(self.func)
            boundargs = sig.bind(*args, **kwargs)
            boundargs.apply_defaults()

            param_tensor_refs = []

            def _make_hlo_ref(name, arg):
                if isinstance(arg, np.ndarray):
                    arg = _sanitize_array_dtype(arg, name)
                    tensor_ref = self._create_parameter_hlo(arg.shape, arg.dtype, name)
                    tensor_ref._original_parameter = tensor_ref.backend_tensor
                    param_index = tensor_ref.backend_tensor.parameter_id
                    param_tensor_refs.append((name, param_index, tensor_ref))
                    return tensor_ref
                return arg

            converted_args, converted_kwargs = _convert_args(
                sig, boundargs, _make_hlo_ref
            )

            ret = self.func(*converted_args, **converted_kwargs)

            self._mark_hlo_outputs(code, ret, param_tensor_refs)
            self._code = code

        return code

    @staticmethod
    def _detect_mutations(ret, param_tensor_refs):
        """Detect mutated parameters and auto-append them to the return list.

        Checks which parameter tensor refs were mutated (had __setitem__
        called on them) during kernel execution.  Mutated parameters that
        the user did not return are appended automatically so the backend
        can compile them as outputs.

        Note: Only direct mutations on the original parameter tensor refs are
        detected. View aliasing (e.g. ``b = a[0]; b[x] = y``) is not tracked
        because ``__getitem__`` creates a new NKIPyTensorRef with no parent link.

        Args:
            ret: The return value(s) from the kernel function.
            param_tensor_refs: List of (param_name, param_index, tensor_ref).

        Returns:
            ``(ret, user_return_len, alias_map)`` where *ret* is the
            (possibly extended) list of outputs, *user_return_len* is the
            original count before auto-appending, and *alias_map* is
            ``{output_index: (param_name, param_index)}``.
        """
        if ret is None:
            ret = []
        elif not isinstance(ret, (list, tuple)):
            ret = [ret]
        ret = list(ret)
        user_return_len = len(ret)

        alias_map = {}
        for name, pidx, tr in param_tensor_refs:
            if not tr._is_mutated:
                continue
            found_at = None
            for i, r in enumerate(ret):
                if isinstance(r, NKIPyTensorRef) and r is tr:
                    found_at = i
                    break
            if found_at is not None:
                alias_map[found_at] = (name, pidx)
            else:
                ret.append(tr)
                alias_map[len(ret) - 1] = (name, pidx)

        return ret, user_return_len, alias_map

    def _mark_hlo_outputs(self, code: HLOModule, ret, param_tensor_refs):
        """Mark HLO outputs using mutation tracking.

        Args:
            code: The HLOModule being built
            ret: The return value(s) from the kernel function
            param_tensor_refs: List of (param_name, param_index, tensor_ref)
        """
        ret, user_return_len, alias_map = self._detect_mutations(
            ret, param_tensor_refs
        )

        ctx = get_hlo_context()

        # Rename mutated HLO parameters for compiler convention
        for _, (param_name, _) in alias_map.items():
            for hlo_param in code.parameters:
                if hlo_param.name == param_name:
                    hlo_param.name = f"{param_name}.must_alias_input"
                    break

        # Insert explicit copy for unmutated pass-through outputs.
        # The Neuron compiler cannot handle outputs that are raw parameter
        # references because inputs and outputs occupy separate memory regions.
        for i, r in enumerate(ret):
            if not isinstance(r, NKIPyTensorRef):
                continue
            if i in alias_map:
                continue
            bt = r.backend_tensor
            if bt.is_parameter:
                copy_tensor = ctx.build_op("copy", [bt], bt.shape, bt.dtype)
                ret[i] = NKIPyTensorRef(copy_tensor, name="")

        # Assign output names and build AliasInfo list
        for idx, r in enumerate(ret):
            if not isinstance(r, NKIPyTensorRef):
                raise RuntimeError(f"Unexpected return value type: {type(r)}")

            if idx in alias_map:
                param_name, param_index = alias_map[idx]
                code.aliases.append(
                    AliasInfo(
                        output_index=idx,
                        param_index=param_index,
                        param_name=param_name,
                        is_user_returned=idx < user_return_len,
                    )
                )
                r.backend_tensor.name = param_name

            # N.B.: the name "output{idx}" is specific
            # it avoids variable folding in HLO lowering in Neuron Compiler
            if not r.backend_tensor.name:
                r.backend_tensor.name = f"output{idx}"

        result_tensors = [r.backend_tensor for r in ret]
        code.set_results(result_tensors)

    def _specialize_kernelgen(self, *args, **kwargs):
        """Trace the kernel to MLIR linalg/tensor IR via the kernelgen backend."""
        from nkipy.core.backend.kernelgen import KernelGenTraceContext
        from nkipy.core.ops._register_kernelgen import register_all_kernelgen_impls

        register_all_kernelgen_impls()

        kctx = KernelGenTraceContext()

        sig = inspect.signature(self.func)
        boundargs = sig.bind(*args, **kwargs)
        boundargs.apply_defaults()

        arg_shapes = []
        arg_dtypes = []
        arg_names = []

        def _collect_array(name, arg):
            arg = _sanitize_array_dtype(arg, name)
            arg_shapes.append(arg.shape)
            arg_dtypes.append(arg.dtype)
            arg_names.append(name)
            return arg

        for name, arg in boundargs.arguments.items():
            param = sig.parameters[name]
            if param.kind == param.VAR_POSITIONAL:
                sanitized = []
                for item in arg:
                    sanitized.append(
                        _collect_array(name, item)
                        if isinstance(item, np.ndarray)
                        else item
                    )
                boundargs.arguments[name] = tuple(sanitized)
            elif param.kind == param.VAR_KEYWORD:
                for k, v in arg.items():
                    if isinstance(v, np.ndarray):
                        arg[k] = _collect_array(k, v)
            elif isinstance(arg, np.ndarray):
                arg = _collect_array(name, arg)
                boundargs.arguments[name] = arg

        param_tensors = kctx._begin_function(self.func.__name__, arg_shapes, arg_dtypes)
        for pt, name in zip(param_tensors, arg_names):
            pt.name = name

        param_tensor_refs = []

        with tracing(kctx):
            param_idx = 0

            def _make_kg_ref(name, arg):
                nonlocal param_idx
                if isinstance(arg, np.ndarray):
                    ref = NKIPyTensorRef(param_tensors[param_idx], name=name)
                    param_tensor_refs.append((name, param_idx, ref))
                    param_idx += 1
                    return ref
                return arg

            converted_args, converted_kwargs = _convert_args(
                sig, boundargs, _make_kg_ref
            )

            raw_ret = self.func(*converted_args, **converted_kwargs)

            ret, user_return_len, alias_map = self._detect_mutations(
                raw_ret, param_tensor_refs
            )

            result_kg_tensors = []
            for r in ret:
                if isinstance(r, NKIPyTensorRef):
                    result_kg_tensors.append(r.backend_tensor)
                else:
                    raise RuntimeError(f"Unexpected return type: {type(r)}")

            kctx._finish_function(result_kg_tensors)

        kctx._run_canonicalize()

        mlir_text = kctx._get_ir_text()
        kctx._cleanup()

        # Use NEFF-compatible names: the kernelgen NEFF uses "in_tensor_N"
        # for inputs and "output" / "output_N" for outputs (determined by the
        # NKI compiler C++ pipeline from unnamed MLIR block arguments and the
        # nki.output_names attribute set by the linalg-to-nisa pass).
        num_outputs = len(result_kg_tensors)
        input_info = [
            (f"in_tensor_{i}", shape, dtype)
            for i, (shape, dtype) in enumerate(zip(arg_shapes, arg_dtypes))
        ]
        output_info = [
            (
                "output" if num_outputs == 1 else f"output_{i}",
                t.shape,
                t.dtype,
            )
            for i, t in enumerate(result_kg_tensors)
        ]

        # Map NEFF input names back to original parameter names so
        # resolve_input_arrays can look up the right numpy arrays.
        param_name_by_neff = {
            f"in_tensor_{i}": name
            for i, name in enumerate(arg_names)
        }

        from nkipy.core.backend.kernelgen import KernelGenIR

        self._code = KernelGenIR(
            mlir_text=mlir_text,
            func_name=self.func.__name__,
            input_specs=input_info,
            output_specs=output_info,
            alias_map=alias_map,
            user_return_len=user_return_len,
            param_name_by_neff=param_name_by_neff,
        )
        return self._code

    @classmethod
    def trace(cls, func=None, backend="hlo"):
        """Decorator to create traced kernel."""
        if func is None:
            return lambda f: cls(f, backend)
        return cls(func, backend)
