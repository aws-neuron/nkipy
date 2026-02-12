# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spike Runtime for Neuron Kernels"""

import inspect
import logging
import os
from typing import Dict, Optional, Tuple

from nkipy.runtime.device_kernel import DeviceKernel
from nkipy.runtime.device_tensor import DeviceTensor

logger = logging.getLogger(__name__)


class CompiledKernel:
    def __init__(self, traced_kernel, compiled_artifact):
        self.traced_kernel = traced_kernel
        # check compiled artifact exists and get full path
        if not os.path.exists(compiled_artifact):
            raise FileNotFoundError(f"Compiled artifact not found: {compiled_artifact}")
        self.compiled_artifact = os.path.abspath(compiled_artifact)
        self.ir = traced_kernel._code


class BaremetalExecutor:
    def __init__(self, verbose: int = 0):
        """Initialize BaremetalExecutor.

        Args:
            verbose: Verbosity level (currently unused, spike singleton handles this)
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup handled by spike singleton and DeviceTensor
        pass

    def _prepare_io_tensors(
        self,
        compiled_kernel: CompiledKernel,
        device_kernel: DeviceKernel,
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, DeviceTensor], Dict[str, DeviceTensor]]:
        """Prepare input and output tensors for kernel execution.

        Args:
            compiled_kernel: CompiledKernel instance
            device_kernel: DeviceKernel instance
            *args, **kwargs: Arguments to pass to the kernel

        Returns:
            Tuple of (inputs_dict, outputs_dict)
        """
        # Bind arguments manually using inspect
        sig = inspect.signature(compiled_kernel.traced_kernel.func)
        boundargs = sig.bind(*args, **kwargs)
        boundargs.apply_defaults()

        # Prepare inputs using DeviceTensor
        inputs = {}
        for intensor in compiled_kernel.ir.inputs:
            real_name = (
                intensor.name.split(".must_alias_input")[0]
                if ".must_alias_input" in intensor.name
                else intensor.name
            )
            np_tensor = boundargs.arguments[real_name]
            inputs[intensor.name] = DeviceTensor.from_numpy(np_tensor)

        # Prepare outputs
        outputs = device_kernel.allocate_output_tensors()
        outputs_dict = {t.name: t for t in outputs}

        return inputs, outputs_dict

    def benchmark(
        self,
        compiled_kernel: CompiledKernel,
        *args,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 100,
        mode: str = "device",
        artifacts_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Benchmark kernel execution.

        Args:
            compiled_kernel: CompiledKernel instance
            args: Positional arguments to pass to the kernel
            kwargs: Keyword arguments to pass to the kernel
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            mode: Timing mode ('device' or 'host')
            artifacts_dir: Optional directory for artifacts

        Returns:
            Dict with benchmark statistics
        """
        # Load kernel using DeviceKernel
        device_kernel = DeviceKernel.load_from_neff(
            compiled_kernel.compiled_artifact,
            name=compiled_kernel.traced_kernel.__name__,
        )

        # Prepare inputs/outputs
        inputs, outputs = self._prepare_io_tensors(
            compiled_kernel, device_kernel, *args, **kwargs
        )

        # Use SpikeModel's benchmark method
        stats = device_kernel.benchmark(
            inputs=inputs,
            outputs=outputs,
            warmup_iter=warmup_iterations,
            benchmark_iter=benchmark_iterations,
            mode=mode,
        )
        return stats

    def run(
        self,
        compiled_kernel: CompiledKernel,
        *args,
        save_trace: bool = False,
        artifacts_dir: Optional[str] = None,
        **kwargs,
    ):
        """Execute kernel.

        Args:
            compiled_kernel: CompiledKernel instance
            args: Positional arguments to pass to the kernel
            kwargs: Keyword arguments to pass to the kernel
            save_trace: Whether to save execution trace
            artifacts_dir: Optional directory for artifacts

        Returns:
            Kernel execution result(s)
        """
        # Load kernel using DeviceKernel
        device_kernel = DeviceKernel.load_from_neff(
            compiled_kernel.compiled_artifact,
            name=compiled_kernel.traced_kernel.__name__,
        )

        # Prepare inputs/outputs
        inputs, outputs = self._prepare_io_tensors(
            compiled_kernel, device_kernel, *args, **kwargs
        )

        # Use SpikeModel's __call__ method
        if save_trace:
            base = os.path.abspath(artifacts_dir) if artifacts_dir else os.getcwd()
            ntff_name = os.path.join(base, "profile.ntff")
        else:
            ntff_name = None
        device_kernel(
            inputs=inputs, outputs=outputs, save_trace=save_trace, ntff_name=ntff_name
        )

        # Convert outputs back to numpy
        result = [outputs[t.name].numpy() for t in compiled_kernel.ir.outputs]
        return result[0] if len(result) == 1 else result
