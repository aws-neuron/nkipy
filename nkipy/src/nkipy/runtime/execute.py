# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution wrappers for NKIPy kernels"""

import os
import shutil
from typing import Optional

import numpy as np

from nkipy.core import compile
from nkipy.core.compile import CompilerConfig
from nkipy.core.ops._registry import set_backend
from nkipy.core.trace import NKIPyKernel

try:
    from nkipy.runtime.device_kernel import DeviceKernel
    from nkipy.runtime.device_tensor import DeviceTensor

    _RUNTIME_AVAILABLE = True
except ImportError:
    _RUNTIME_AVAILABLE = False


def cpu_simulate_traced_kernel(traced_kernel, *args, **kwargs):
    """
    CPU-based simulation using pure NumPy operations.

    This simulation mode executes the kernel function directly with NumPy arrays,
    using the CPU backend implementations of all operations.

    Limitations (TODO):
        - Mutable tensor semantics not supported (outputs are not written back)
        - NKI kernels are not supported (only NKIPy kernels with HLO backend)
    """
    if not isinstance(traced_kernel, NKIPyKernel):
        raise TypeError(
            f"Expected NKIPyKernel for simulation, got {type(traced_kernel)}"
        )

    set_backend("cpu")
    try:
        result = traced_kernel.func(*args, **kwargs)
        return result
    finally:
        set_backend(None)


def simulate_traced_kernel(traced_kernel, *args, **kwargs):
    # Step 1: make sure HLO can be generated
    # Trace the kernel with the provided arguments
    traced_kernel.specialize(*args, **kwargs)

    # Step 2: simulate the HLO
    # TODO: the HLO simulation is path now disabled, we use CPU simulation for now
    return cpu_simulate_traced_kernel(traced_kernel, *args, **kwargs)


def baremetal_run_traced_kernel(
    kernel,
    *args,
    artifacts_dir=None,
    save_trace=False,
    compiler_config: Optional[CompilerConfig] = None,
    additional_compiler_args="",
    target=compile.CompilationTarget.DEFAULT,
    **kwargs,
):
    """Execute a traced kernel on Trainium hardware.

    Args:
        kernel: The traced kernel to execute.
        *args: Positional arguments for the kernel.
        artifacts_dir: Directory to save compilation artifacts.
        save_trace: Whether to save execution trace.
        compiler_config: Structured compiler configuration (recommended).
            If not provided, auto-detects kernel type and uses appropriate preset.
        additional_compiler_args: Additional arguments to append (legacy).
            If both compiler_config and additional_compiler_args are provided,
            additional_compiler_args will be appended.
        target: Compilation target (default: auto-detect).
        **kwargs: Keyword arguments for the kernel.

    Returns:
        The kernel output(s).
    """
    if not _RUNTIME_AVAILABLE:
        raise RuntimeError(
            "Runtime is not available. Please install Spike to use this function."
        )

    # Trace the kernel with the provided arguments
    kernel.specialize(*args, **kwargs)
    ir = kernel._code

    # Bind arguments for input/output mapping
    import inspect

    sig = inspect.signature(kernel.func)
    boundargs = sig.bind(*args, **kwargs)
    boundargs.apply_defaults()

    # Allocate output tensors based on IR outputs
    for outtensor in ir.outputs:
        output_array = np.empty(outtensor.shape, dtype=outtensor.dtype)
        boundargs.arguments[outtensor.name] = output_array

    name = kernel.__name__

    build_dir = artifacts_dir if artifacts_dir else f"{compile._get_build_dir()}/{name}"

    # Resolve compiler args: compiler_config takes precedence, else auto-detect
    if compiler_config is not None:
        final_compiler_args = compiler_config.to_args()
    elif isinstance(kernel, compile.NKIPyKernel):
        final_compiler_args = compile.nkipy_compiler_args
    else:
        # assume is NKI
        final_compiler_args = compile.nki_compiler_args

    # Append legacy additional_compiler_args if provided
    if additional_compiler_args:
        final_compiler_args = final_compiler_args + " " + additional_compiler_args

    # always clean the build dir in baremetal mode
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    neff = compile.compile_to_neff(
        trace_kernel=kernel,
        output_dir=build_dir,
        neff_name=f"{name}.neff",
        save_artifacts=True,
        additional_compiler_args=final_compiler_args,
        target=target,
    )

    device_kernel = DeviceKernel.load_from_neff(neff, name)

    device_inputs = {}
    for intensor in ir.inputs:
        if "must_alias_input" in intensor.name:
            base_name = intensor.name.split(".must_alias_input")[0]
            np_tensor = boundargs.arguments[base_name]
        else:
            np_tensor = boundargs.arguments[intensor.name]
        device_inputs[intensor.name] = DeviceTensor.from_numpy(np_tensor)

    device_outputs = {}
    for outtensor in ir.outputs:
        np_output = np.zeros(outtensor.shape, dtype=outtensor.dtype)
        device_outputs[outtensor.name] = DeviceTensor.from_numpy(np_output)

    device_kernel(inputs=device_inputs, outputs=device_outputs, save_trace=save_trace)

    for outtensor in ir.outputs:
        result = device_outputs[outtensor.name].numpy()
        dst = boundargs.arguments[outtensor.name]
        np.copyto(dst=dst, src=result)

    # Return the output(s)
    if len(ir.outputs) == 1:
        return boundargs.arguments[ir.outputs[0].name]
    elif len(ir.outputs) > 1:
        return tuple(boundargs.arguments[out.name] for out in ir.outputs)
    return None
