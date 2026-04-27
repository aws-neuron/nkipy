# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution wrappers for NKIPy kernels"""

from __future__ import annotations

import inspect
import os
import shutil

import numpy as np

from nkipy.core import compile
from nkipy.core.backend import ComputationIR

try:
    from nkipy.runtime.device_kernel import DeviceKernel
    from nkipy.runtime.device_tensor import DeviceTensor

    _RUNTIME_AVAILABLE = True
except ImportError:
    _RUNTIME_AVAILABLE = False


def _compile_kernel(
    kernel,
    *args,
    artifacts_dir=None,
    additional_compiler_args="",
    target=compile.CompilationTarget.DEFAULT,
    **kwargs,
):
    """Specialize and compile a traced kernel to NEFF.

    Returns (neff_path, kernel_name, ir, original_inputs).
    """
    kernel.specialize(*args, **kwargs)
    ir = kernel._code

    sig = inspect.signature(kernel.func)
    boundargs = sig.bind(*args, **kwargs)
    boundargs.apply_defaults()

    original_inputs = {
        name: arr
        for name, arr in boundargs.arguments.items()
        if isinstance(arr, np.ndarray)
    }

    name = kernel.__name__

    build_dir = artifacts_dir if artifacts_dir else f"{compile._get_build_dir()}/{name}"
    if isinstance(kernel, compile.NKIPyKernel):
        additional_compiler_args = (
            compile.nkipy_compiler_args + " " + additional_compiler_args
        )
    else:
        additional_compiler_args = (
            compile.DEFAULT_ADDITIONAL_COMPILER_ARGS + " " + additional_compiler_args
        )

    # always clean the build dir in baremetal mode
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    neff = compile.compile_to_neff(
        trace_kernel=kernel,
        output_dir=build_dir,
        neff_name=f"{name}.neff",
        save_artifacts=True,
        additional_compiler_args=additional_compiler_args,
        target=target,
    )

    return neff, name, ir, original_inputs


def _execute_neff(neff, name, ir: ComputationIR, original_inputs, save_trace=False):
    """Load a compiled NEFF and run it on hardware.

    Returns output numpy array(s), with auto-aliased outputs filtered out.
    Aliased output data is written back to the original input arrays.
    """
    if not _RUNTIME_AVAILABLE:
        raise RuntimeError(
            "Runtime is not available. Please install Spike to use this function."
        )

    device_kernel = DeviceKernel.load_from_neff(neff, name)

    # Build alias lookup: output_index -> AliasInfo
    alias_by_output = {a.output_index: a for a in ir.aliases}

    input_arrays = ir.resolve_input_arrays(original_inputs)
    device_inputs = {
        input_name: DeviceTensor.from_numpy(arr)
        for input_name, arr in input_arrays.items()
    }

    device_outputs = {}
    for i, outtensor in enumerate(ir.outputs):
        if i in alias_by_output:
            alias = alias_by_output[i]
            input_name = ir.get_alias_input_name(alias)
            device_outputs[outtensor.name] = device_inputs[input_name]
        else:
            np_output = np.zeros(outtensor.shape, dtype=outtensor.dtype)
            device_outputs[outtensor.name] = DeviceTensor.from_numpy(np_output)

    device_kernel(inputs=device_inputs, outputs=device_outputs, save_trace=save_trace)

    output_arrays = {}
    for i, outtensor in enumerate(ir.outputs):
        result = device_outputs[outtensor.name].numpy()
        if i in alias_by_output:
            alias = alias_by_output[i]
            np.copyto(dst=original_inputs[alias.param_name], src=result)
        output_arrays[outtensor.name] = result

    # Filter out auto-aliased outputs (not user-returned)
    auto_indices = ir.auto_aliased_indices
    user_outputs = [
        output_arrays[out.name]
        for i, out in enumerate(ir.outputs)
        if i not in auto_indices
    ]

    if len(user_outputs) == 1:
        return user_outputs[0]
    elif len(user_outputs) > 1:
        return tuple(user_outputs)
    return None


def baremetal_run_traced_kernel(
    kernel,
    *args,
    artifacts_dir=None,
    save_trace=False,
    additional_compiler_args="",
    target=compile.CompilationTarget.DEFAULT,
    **kwargs,
):
    """Compile and run a traced kernel on hardware."""
    neff, name, ir, original_inputs = _compile_kernel(
        kernel,
        *args,
        artifacts_dir=artifacts_dir,
        additional_compiler_args=additional_compiler_args,
        target=target,
        **kwargs,
    )
    return _execute_neff(
        neff, name, ir, original_inputs, save_trace=save_trace
    )
