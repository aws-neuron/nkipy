# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""KernelGen backend for NKIPy.

This module provides the kernelgen backend by delegating to
``nkipy_kernelgen.builder`` for all MLIR construction.  No MLIR types
are imported or exposed — the builder API is the sole interface.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from nkipy.core.backend import AliasInfo, TensorPlaceholder


# ---------------------------------------------------------------------------
# KernelGenTensor -- analogue of HLOTensor
# ---------------------------------------------------------------------------

class KernelGenTensor:
    """Backend tensor for the kernelgen backend.

    Wraps an opaque ``TensorHandle`` from ``nkipy_kernelgen.builder``
    with the metadata that ``NKIPyTensorRef`` expects.
    """

    __slots__ = ("handle", "shape", "dtype", "is_parameter", "parameter_id", "name", "id")

    _next_id = 0

    def __init__(self, handle, shape, dtype, *, is_parameter=False, parameter_id=None, name=""):
        self.handle = handle
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        self.is_parameter = is_parameter
        self.parameter_id = parameter_id
        self.name = name
        self.id = KernelGenTensor._next_id
        KernelGenTensor._next_id += 1


# ---------------------------------------------------------------------------
# KernelGenTraceContext
# ---------------------------------------------------------------------------

class KernelGenTraceContext:
    """Trace context that delegates to ``nkipy_kernelgen.builder.IRBuilder``."""

    backend_name = "kernelgen"

    def __init__(self):
        from nkipy_kernelgen.builder import IRBuilder
        self._builder = IRBuilder()
        self._parameters: List[KernelGenTensor] = []
        self.current_source_location = None

    @property
    def module(self):
        """Return the underlying MLIR module from the builder."""
        return self._builder.module

    def set_source_location(self, location):
        """Set the current source location for diagnostic tracking."""
        self.current_source_location = location

    def _begin_function(self, name, arg_shapes, arg_dtypes):
        """Start an MLIR function and return parameter tensors."""
        handles = self._builder.begin_function(name, arg_shapes, arg_dtypes)
        tensors = []
        for i, (h, (shape, dtype)) in enumerate(
            zip(handles, zip(arg_shapes, arg_dtypes))
        ):
            kt = KernelGenTensor(
                h, shape, dtype,
                is_parameter=True, parameter_id=i, name=f"arg{i}"
            )
            self._parameters.append(kt)
            tensors.append(kt)
        return tensors

    def _finish_function(self, result_tensors):
        """Finalize the MLIR function with the given result tensors."""
        self._builder.finish_function([t.handle for t in result_tensors])

    def _run_canonicalize(self):
        """Run MLIR canonicalization passes on the module."""
        self._builder.run_canonicalize()

    def _get_ir_text(self):
        """Export the MLIR module as a text string."""
        return self._builder.get_ir_text()

    def _cleanup(self):
        """Release builder resources."""
        self._builder.cleanup()


# ---------------------------------------------------------------------------
# Module-level context accessor
# ---------------------------------------------------------------------------

def get_kernelgen_context() -> KernelGenTraceContext:
    """Return the active ``KernelGenTraceContext``, or raise if none is active."""
    from nkipy.core.backend import _active_ctx
    if _active_ctx is None or _active_ctx.backend_name != "kernelgen":
        raise RuntimeError("No active kernelgen trace context")
    return _active_ctx


# ---------------------------------------------------------------------------
# KernelGenIR -- make MLIR IR compatible with execution pipeline
# ---------------------------------------------------------------------------


class KernelGenIR:
    """Adapter that makes an MLIR module compatible with the execution pipeline.

    Provides the same interface as ``HLOModule`` (``.inputs``, ``.outputs``,
    ``.aliases``, ``.auto_aliased_indices``) so that ``compile.py`` and
    ``execute.py`` can handle both backends uniformly.
    """

    def __init__(self, mlir_text, func_name, input_specs, output_specs,
                 alias_map=None, user_return_len=None, param_name_by_neff=None):
        self._mlir_text = mlir_text
        self._func_name = func_name
        self._input_specs = input_specs    # [(name, shape, dtype), ...]
        self._output_specs = output_specs  # [(name, shape, dtype), ...]
        # alias_map: {output_index: (param_name, param_index)}
        self._alias_map = alias_map or {}
        self._user_return_len = user_return_len if user_return_len is not None else len(output_specs)
        # Maps NEFF input names ("in_tensor_0") to original param names ("A")
        self._param_name_by_neff = param_name_by_neff or {}

    @property
    def inputs(self):
        """Return input tensor metadata as ``TensorPlaceholder`` list."""
        return [TensorPlaceholder(n, tuple(s), np.dtype(d)) for n, s, d in self._input_specs]

    @property
    def outputs(self):
        """Return output tensor metadata as ``TensorPlaceholder`` list."""
        return [TensorPlaceholder(n, tuple(s), np.dtype(d)) for n, s, d in self._output_specs]

    @property
    def aliases(self):
        """Return input-output alias pairs as ``AliasInfo`` list."""
        return [
            AliasInfo(
                output_index=out_idx,
                param_index=pidx,
                param_name=pname,
                is_user_returned=out_idx < self._user_return_len,
            )
            for out_idx, (pname, pidx) in self._alias_map.items()
        ]

    @property
    def auto_aliased_indices(self):
        """Output indices that were auto-added (not user-returned)."""
        return {
            out_idx for out_idx in self._alias_map
            if out_idx >= self._user_return_len
        }

    def resolve_input_arrays(self, original_inputs):
        """Map NEFF input names to numpy arrays.

        NEFF inputs use ``in_tensor_N`` names.  *original_inputs* is keyed
        by bare parameter names (``A``, ``B``).  ``_param_name_by_neff``
        bridges the two.
        """
        if len(original_inputs) != len(self._input_specs):
            raise RuntimeError(
                f"Expected {len(self._input_specs)} tensor arguments, "
                f"got {len(original_inputs)}"
            )
        mapping = {}
        for intensor in self.inputs:
            param_name = self._param_name_by_neff.get(intensor.name, intensor.name)
            mapping[intensor.name] = original_inputs[param_name]
        return mapping

    def get_alias_input_name(self, alias):
        """Return the NEFF input name for an aliased parameter."""
        for neff_name, param_name in self._param_name_by_neff.items():
            if param_name == alias.param_name:
                return neff_name
        return alias.param_name

    def content_hash(self, compiler_args: str) -> str:
        """Compute a content hash from the MLIR text and compiler args."""
        h = hashlib.sha256()
        h.update(self._mlir_text.encode("utf-8"))
        h.update(compiler_args.encode("utf-8"))
        return h.hexdigest()[:12]

