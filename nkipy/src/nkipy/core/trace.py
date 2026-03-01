# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace wrappers to lower NKIPy kernels"""

import inspect
import warnings

import numpy as np

from nkipy.core._numpy_dispatch import register_all_numpy_apis
from nkipy.core.backend import tracing
from nkipy.core.backend.hlo import (
    AliasInfo,
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


class NKIPyKernel:
    """Simplified kernel wrapper for NKIPy tracing"""

    def __init__(self, func, backend, **kwargs):
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
        elif self.backend == "cpu":
            print("CPU backend does not require specialization")
            return
        else:
            raise ValueError(f"Unknown backend {self.backend}")

    def _create_parameter_hlo(self, shape, dtype, name=""):
        """Create an HLO parameter tensor"""
        ctx = get_hlo_context()
        hlo_tensor = ctx.module.add_parameter(shape, dtype, name=name)
        return NKIPyTensorRef(hlo_tensor, name=name)

    def _specialize_hlo(self, *args, **kwargs):
        """Trace the kernel with specific arguments"""

        code = HLOModule(name=self.func.__name__)

        with tracing(HLOTraceContext(code)):
            # Bind arguments
            sig = inspect.signature(self.func)
            boundargs = sig.bind(*args, **kwargs)
            boundargs.apply_defaults()

            # Convert numpy arrays to tensor references
            converted_args = []
            converted_kwargs = {}
            # Track parameter tensor refs: list of (param_name, tensor_ref) for arrays
            param_tensor_refs = []

            for name, arg in boundargs.arguments.items():
                param = sig.parameters[name]

                if isinstance(arg, np.ndarray):
                    arg = _sanitize_array_dtype(arg, name)
                    tensor_ref = self._create_parameter_hlo(arg.shape, arg.dtype, name)
                    tensor_ref._original_parameter = tensor_ref.backend_tensor
                    converted_value = tensor_ref
                    param_tensor_refs.append((name, tensor_ref))
                else:
                    converted_value = arg

                # Determine if this should be positional or keyword
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                    converted_args.append(converted_value)
                elif param.kind == param.KEYWORD_ONLY:
                    converted_kwargs[name] = converted_value
                elif param.kind == param.VAR_POSITIONAL:
                    if isinstance(arg, (list, tuple)):
                        for item in arg:
                            if isinstance(item, np.ndarray):
                                item = _sanitize_array_dtype(item, f"{name}_item")
                                converted_args.append(
                                    self._create_parameter_hlo(
                                        item.shape, item.dtype, f"{name}_item"
                                    )
                                )
                            else:
                                converted_args.append(item)
                    else:
                        converted_args.append(converted_value)
                elif param.kind == param.VAR_KEYWORD:
                    if isinstance(arg, dict):
                        for k, v in arg.items():
                            if isinstance(v, np.ndarray):
                                v = _sanitize_array_dtype(v, k)
                                converted_kwargs[k] = self._create_parameter_hlo(
                                    v.shape, v.dtype, k
                                )
                            else:
                                converted_kwargs[k] = v

            # Execute function
            ret = self.func(*converted_args, **converted_kwargs)

            # Mark outputs
            self._mark_hlo_outputs(code, ret, param_tensor_refs)
            self._code = code

        return code

    def _mark_hlo_outputs(self, code: HLOModule, ret, param_tensor_refs):
        """Mark HLO outputs using mutation tracking.

        Detects aliasing by checking which parameter tensor refs were mutated
        (had __setitem__ called on them) during kernel execution.

        Note: Only direct mutations on the original parameter tensor refs are
        detected. View aliasing (e.g. ``b = a[0]; b[x] = y``) is not tracked
        because ``__getitem__`` creates a new tensor ref with no parent link.

        Args:
            code: The HLOModule being built
            ret: The return value(s) from the kernel function
            param_tensor_refs: List of (param_name, tensor_ref) for array parameters
        """
        # Normalize return value to a list (may be None for mutation-only kernels)
        if ret is None:
            ret = []
        elif not isinstance(ret, (list, tuple)):
            ret = [ret]
        ret = list(ret)
        user_return_len = len(ret)

        ctx = get_hlo_context()

        # Step 1: For each mutated param, rename HLO parameter.
        # Check if user returned it; if not, auto-append to output list.
        aliased_return_positions = {}  # output_index -> (param_name, param_index)
        for name, tr in param_tensor_refs:
            if not tr._is_mutated:
                continue

            # Rename HLO parameter for compiler convention
            param_index = None
            for hlo_param in code.parameters:
                if hlo_param.name == name:
                    hlo_param.name = f"{name}.must_alias_input"
                    param_index = hlo_param.parameter_id
                    break

            if param_index is None:
                raise RuntimeError(
                    f"Mutated parameter '{name}' not found in HLO parameters"
                )

            # Check if this mutated param is in the user's return values (identity check)
            found_at = None
            for i, r in enumerate(ret):
                if isinstance(r, NKIPyTensorRef) and r is tr:
                    found_at = i
                    break

            if found_at is not None:
                aliased_return_positions[found_at] = (name, param_index)
            else:
                # Auto-append to output list
                ret.append(tr)
                aliased_return_positions[len(ret) - 1] = (name, param_index)

        # Step 2: Insert explicit copy for unmutated pass-through outputs.
        # The Neuron compiler cannot handle outputs that are raw parameter
        # references because inputs and outputs occupy separate memory regions.
        for i, r in enumerate(ret):
            if not isinstance(r, NKIPyTensorRef):
                continue
            if i in aliased_return_positions:
                continue
            bt = r.backend_tensor
            if bt.is_parameter:
                copy_tensor = ctx.build_op("copy", [bt], bt.shape, bt.dtype)
                ret[i] = NKIPyTensorRef(copy_tensor, name="")

        # Step 3: Assign output names and build AliasInfo list
        for idx, r in enumerate(ret):
            if not isinstance(r, NKIPyTensorRef):
                raise RuntimeError(f"Unexpected return value type: {type(r)}")

            if idx in aliased_return_positions:
                param_name, param_index = aliased_return_positions[idx]
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

    @classmethod
    def trace(cls, func=None, backend="hlo", **kwargs):
        """Decorator to create traced kernel"""
        if func is None:
            return lambda f: cls(f, backend, **kwargs)
        return cls(func, backend, **kwargs)
