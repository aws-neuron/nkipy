# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace wrappers to lower NKIPy kernels"""

import ast
import inspect
import textwrap

import numpy as np

import nkipy.core.typing as nt
from nkipy.core._numpy_dispatch import register_all_numpy_apis
from nkipy.core.backend.hlo import HLOModule, HLOTraceContext
from nkipy.core.ops._registry import set_backend
from nkipy.core.tensor import NKIPyTensorRef

# Register numpy APIs to use ops module implementations
# This replaces the tensor_apis registration
register_all_numpy_apis()


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
        ctx = HLOTraceContext._global_ctx
        hlo_tensor = ctx.module.add_parameter(shape, dtype, name=name)
        return NKIPyTensorRef(hlo_tensor, name=name)

    def _specialize_hlo(self, *args, **kwargs):
        """Trace the kernel with specific arguments"""

        code = HLOModule(name=self.func.__name__)
        ctx = HLOTraceContext(code)
        HLOTraceContext._global_ctx = ctx
        set_backend("hlo", ctx)

        # 4. Bind arguments
        sig = inspect.signature(self.func)
        boundargs = sig.bind(*args, **kwargs)
        boundargs.apply_defaults()

        # 5. Convert numpy arrays to tensor references
        converted_args = []
        converted_kwargs = {}

        for name, arg in boundargs.arguments.items():
            param = sig.parameters[name]

            if isinstance(arg, np.ndarray):
                tensor_ref = self._create_parameter_hlo(arg.shape, arg.dtype, name)
                converted_value = tensor_ref
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
                            converted_kwargs[k] = self._create_parameter_hlo(
                                v.shape, v.dtype, k
                            )
                        else:
                            converted_kwargs[k] = v

        # 6. Execute function
        ret = self.func(*converted_args, **converted_kwargs)

        # 7. Mark outputs
        self._mark_hlo_outputs(code, ret)
        self._code = code
        set_backend(None)

        return code

    def _mark_hlo_outputs(self, code: HLOModule, ret):
        """Mark HLO outputs"""

        if ret is not None:
            if not isinstance(ret, (list, tuple)):
                ret = [ret]

            # Detect mutable_tensor parameters by inspecting function signature
            sig = inspect.signature(self.func)
            mutable_params = {}
            for param_name, param in sig.parameters.items():
                # Check if the annotation has the mutable modifier
                # This handles both nt.mutable_tensor and tensor[mutable, dtype, shape]
                is_mutable = False
                if (
                    hasattr(param.annotation, "modifier")
                    and param.annotation.modifier is nt.mutable
                ):
                    # Check for modifier attribute (works for tensor[nt.mutable])
                    is_mutable = True
                elif param.annotation == nt.mutable_tensor:
                    # Fallback for direct mutable_tensor annotation
                    is_mutable = True

                if is_mutable:
                    # Find the corresponding parameter in code.parameters
                    for hlo_param in code.parameters:
                        if hlo_param.name == param_name:
                            mutable_params[param_name] = (
                                hlo_param.parameter_id,
                                hlo_param,
                            )
                            # Rename the parameter to include .must_alias_input suffix
                            hlo_param.name = f"{param_name}.must_alias_input"
                            break

            # Extract returned variable names from function source using AST
            returned_var_names = []
            try:
                source = inspect.getsource(self.func)
                tree = ast.parse(textwrap.dedent(source))

                # Find return statement(s) and extract variable names
                for node in ast.walk(tree):
                    if isinstance(node, ast.Return) and node.value:
                        if isinstance(node.value, ast.Name):
                            # Single return: return x
                            returned_var_names.append(node.value.id)
                        elif isinstance(node.value, ast.Tuple):
                            # Multiple return: return x, y, z
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Name):
                                    returned_var_names.append(elt.id)
            except Exception as e:
                # If AST parsing fails, fall back to empty list
                # This means no aliasing will be detected
                print(f"Failed to parse function source for aliasing detection: {e}")
                returned_var_names = []

            # Insert explicit copy for pass-through outputs (outputs that are
            # unmodified input parameters). The Neuron compiler cannot handle
            # outputs that are raw parameter references because inputs and outputs
            # occupy separate memory regions on device.
            ret = list(ret)
            ctx = HLOTraceContext._global_ctx
            for i, r in enumerate(ret):
                if not isinstance(r, NKIPyTensorRef):
                    continue
                bt = r.backend_tensor
                if bt.is_parameter:
                    # Skip mutable aliases — those are handled via input_output_alias
                    var_name = (
                        returned_var_names[i] if i < len(returned_var_names) else None
                    )
                    if var_name not in mutable_params:
                        copy_tensor = ctx.build_op("copy", [bt], bt.shape, bt.dtype)
                        ret[i] = NKIPyTensorRef(copy_tensor, name="")

            idx = 0
            for r in ret:
                if not isinstance(r, NKIPyTensorRef):
                    raise RuntimeError(f"Unexpected return value type: {type(r)}")

                # Check if this output should alias with a mutable input parameter
                # Use the variable name from the return statement
                if idx < len(returned_var_names):
                    var_name = returned_var_names[idx]
                    if var_name in mutable_params:
                        param_id, _ = mutable_params[var_name]
                        # Record aliasing: output index -> input parameter number
                        code.input_output_alias[idx] = param_id

                        # FIXME: Also override the name
                        r.backend_tensor.name = var_name

                # N.B.: the name "output{idx}" is specific
                # it avoids variable folding in HLO lowering in Neuron Compiler
                if not r.backend_tensor.name:
                    r.backend_tensor.name = f"output{idx}"

                idx += 1
            result_tensors = [r.backend_tensor for r in ret]

            # If there are multiple results, create a tuple result
            if len(result_tensors) > 1:
                # For multiple outputs, we need to create a tuple in HLO
                # This is done by setting multiple results
                code.set_results(result_tensors)
            else:
                # Single output
                code.set_results(result_tensors)

    @classmethod
    def trace(cls, func=None, backend="hlo", **kwargs):
        """Decorator to create traced kernel"""
        if func is None:
            return lambda f: cls(f, backend, **kwargs)
        return cls(func, backend, **kwargs)
