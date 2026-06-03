# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NkiGen-Lite backend for NKIPy.

This module provides the nkigen-lite backend by delegating to
``nkigen_lite.tensor_ir.Builder`` for all IR construction.  The resulting
graph is lowered through nkigen_lite's pass pipeline (canonicalize →
decompose → layout_solver → lower_to_nki) and compiled via the NKI
kernel_builder API.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from nkipy.core.backend import AliasInfo, TensorPlaceholder


# ---------------------------------------------------------------------------
# Numpy dtype ↔ nkigen_lite DType conversion
# ---------------------------------------------------------------------------

_NP_TO_LITE_DTYPE = None


def _get_np_to_lite_dtype():
    global _NP_TO_LITE_DTYPE
    if _NP_TO_LITE_DTYPE is None:
        from nkigen_lite.core import DType, _DTYPE_TO_NP
        # Build reverse mapping; for np.float32 prefer F32 over TF32
        # since TF32 is a hardware-internal format that may not be
        # supported in all compilation paths.
        _NP_TO_LITE_DTYPE = {}
        for k, v in _DTYPE_TO_NP.items():
            nd = np.dtype(v)
            if nd not in _NP_TO_LITE_DTYPE or k == DType.F32:
                _NP_TO_LITE_DTYPE[nd] = k
    return _NP_TO_LITE_DTYPE


def np_dtype_to_lite(dtype: np.dtype):
    """Convert a numpy dtype to nkigen_lite DType."""
    mapping = _get_np_to_lite_dtype()
    dtype = np.dtype(dtype)
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for nkigen-lite: {dtype}")
    return mapping[dtype]


def lite_dtype_to_np(lite_dtype) -> np.dtype:
    """Convert a nkigen_lite DType to numpy dtype."""
    from nkigen_lite.core import to_np_dtype
    return to_np_dtype(lite_dtype)


# ---------------------------------------------------------------------------
# NkiGenLiteTensor -- analogue of HLOTensor / NkiGenTensor
# ---------------------------------------------------------------------------

class NkiGenLiteTensor:
    """Backend tensor for the nkigen-lite backend.

    Wraps a ``nkigen_lite.core.Value`` with the metadata that
    ``NKIPyTensorRef`` expects.
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
        self.id = NkiGenLiteTensor._next_id
        NkiGenLiteTensor._next_id += 1


# ---------------------------------------------------------------------------
# NkiGenLiteTraceContext
# ---------------------------------------------------------------------------

class NkiGenLiteTraceContext:
    """Trace context that delegates to ``nkigen_lite.tensor_ir.Builder``."""

    backend_name = "nkigen-lite"

    def __init__(self, name: str = "main"):
        from nkigen_lite.tensor_ir import Builder
        self._builder = Builder(name)
        self._parameters: List[NkiGenLiteTensor] = []
        self.current_source_location = None

    @property
    def builder(self):
        """Return the underlying nkigen_lite Builder."""
        return self._builder

    def set_source_location(self, location):
        """Set the current source location for diagnostic tracking."""
        self.current_source_location = location

    def add_parameter(self, shape, dtype, name=""):
        """Add a graph input parameter and return a NkiGenLiteTensor."""
        lite_dtype = np_dtype_to_lite(dtype)
        value = self._builder.add_input(name, tuple(shape), lite_dtype)
        param_id = len(self._parameters)
        tensor = NkiGenLiteTensor(
            value, shape, dtype,
            is_parameter=True, parameter_id=param_id, name=name,
        )
        self._parameters.append(tensor)
        return tensor

    def set_outputs(self, output_values: dict):
        """Finalize the graph with the given named outputs."""
        self._builder.set_outputs(output_values)

    @property
    def graph(self):
        """Return the constructed graph."""
        return self._builder.graph


# ---------------------------------------------------------------------------
# Module-level context accessor
# ---------------------------------------------------------------------------

def get_nkigen_lite_context() -> NkiGenLiteTraceContext:
    """Return the active ``NkiGenLiteTraceContext``, or raise if none is active."""
    from nkipy.core.backend import _active_ctx
    if _active_ctx is None or _active_ctx.backend_name != "nkigen-lite":
        raise RuntimeError("No active nkigen-lite trace context")
    return _active_ctx


# ---------------------------------------------------------------------------
# NkiGenLiteIR -- make tensor_ir Graph compatible with execution pipeline
# ---------------------------------------------------------------------------

class NkiGenLiteIR:
    """Adapter that makes a nkigen_lite tensor_ir Graph compatible with
    the execution pipeline.

    Provides the same interface as ``HLOModule`` and ``NkiGenIR``
    (``.inputs``, ``.outputs``, ``.aliases``, ``.auto_aliased_indices``)
    so that ``compile.py`` and ``execute.py`` can handle all backends
    uniformly.
    """

    def __init__(self, graph, func_name, input_specs, output_specs,
                 alias_map=None, user_return_len=None, original_param_names=None):
        self._graph = graph
        self._func_name = func_name
        self._input_specs = input_specs    # [(name, shape, dtype), ...]
        self._output_specs = output_specs  # [(name, shape, dtype), ...]
        # alias_map: {output_index: (param_name, param_index)}
        self._alias_map = alias_map or {}
        self._user_return_len = user_return_len if user_return_len is not None else len(output_specs)
        self._original_param_names = original_param_names or []

    @property
    def inputs(self):
        """Return input tensor metadata as ``TensorPlaceholder`` list."""
        return [
            TensorPlaceholder(n, tuple(s), np.dtype(d), original_name=self._original_param_names[i])
            for i, (n, s, d) in enumerate(self._input_specs)
        ]

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

    def content_hash(self, compiler_args: str) -> str:
        """Compute a content hash from the graph dump and compiler args."""
        h = hashlib.sha256()
        h.update(self._graph.dump().encode("utf-8"))
        h.update(compiler_args.encode("utf-8"))
        return h.hexdigest()[:12]
