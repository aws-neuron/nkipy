# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime module for compiling and executing models on AWS Neuron hardware.

Note: Heavy imports are deferred to avoid circular imports.
"""


def __getattr__(name):
    """Lazy import for runtime functions."""
    _runtime_exports = {
        "IOSpecs", "NeuronExecutable", "TensorSpec",
        "compile_load_execute", "compile_model", "execute_model",
        "in_parallel_compile_context", "load_model",
        "parallel_compile_context", "parallel_compile_model", "run_neff_model",
    }
    if name in _runtime_exports:
        from torch_to_nkipy.runtime import runtime
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main functions
    "compile_load_execute",
    "compile_model",
    "load_model",
    "run_neff_model",
    "execute_model",  # backward compat
    # Parallel compilation
    "parallel_compile_context",
    "parallel_compile_model",
    "in_parallel_compile_context",
    # Data classes
    "TensorSpec",
    "IOSpecs",
    "NeuronExecutable",
]
