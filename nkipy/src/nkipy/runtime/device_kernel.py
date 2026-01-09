# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit
import os
import shutil
import time
import types

from nkipy.core import compile
from nkipy.core.compile import CompilationTarget, _get_build_dir, compile_to_neff, trace
from nkipy.core.logger import get_logger
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import device_tensor
from nkipy.runtime.device_tensor import DeviceTensor
from spike import SpikeModel

if device_tensor._TORCH_ENABLED:
    import torch.distributed as dist

logger = get_logger()

# Device loaded kernels prevent loading multiple times
_LOADED_KERNELS = {}


# Note: kernel object from nanobind is RAII and Python GC managed, so no need to call
# `.unload_model()` explicitly. However, we do want to clear the dict to make sure
# it does before nanobind ref leak check.
def _cleanup_kernels():
    _LOADED_KERNELS.clear()


atexit.register(_cleanup_kernels)


class DeviceKernel(SpikeModel):
    """A wrapper class for executing compiled kernels."""

    def __init__(self, model_ref, name, neff_path):
        super().__init__(model_ref, name, neff_path)

    @classmethod
    def compile_and_load(
        cls,
        kernel,
        *args,
        name=None,
        additional_compiler_args=None,
        use_cached_if_exists=True,
        build_dir=None,
        target=CompilationTarget.DEFAULT,
        **kwargs,
    ):
        """Compile and load a kernel, returning a DeviceKernel instance.

        Args:
            kernel: The kernel function to compile
            name: Optional name for the kernel. If None, uses kernel.__name__
            additional_compiler_args: Optional additional compiler arguments to append
            use_cached_if_exists: If True, use cached neff if it exists.
            build_dir: Overriding the build directory for the kernel
            target: Compilation target for the kernel
            *args, **kwargs: Arguments for specialization (numpy array or DeviceTensor)

        Returns:
            DeviceKernel: A DeviceKernel instance with the compiled kernel
        """
        if name is None:
            # FIXME: this is likely to introduce unexpected conflict
            # need a more robust caching mechanism (hash etc)
            name = kernel.__name__

        if use_cached_if_exists and name in _LOADED_KERNELS:
            logger.info(f"Using loaded kernel: {name}")
            return _LOADED_KERNELS[name]

        # Convert DeviceTensors to numpy arrays for compilation
        numpy_args = []
        for arg in args:
            if isinstance(arg, DeviceTensor):
                numpy_args.append(arg.numpy())
            else:
                numpy_args.append(arg)

        numpy_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, DeviceTensor):
                numpy_kwargs[key] = value.numpy()
            else:
                numpy_kwargs[key] = value

        build_dir = build_dir or _get_build_dir()

        output_dir = f"{build_dir}/{name}"
        neff_path = f"{output_dir}/{name}.neff"

        if (
            device_tensor._TORCH_ENABLED
            and dist.is_initialized()
            and dist.get_rank() != 0
        ):
            logger.info(
                f"Rank {dist.get_rank()} is not the master rank, skipping compilation"
            )
        elif use_cached_if_exists and os.path.exists(neff_path):
            logger.info(f"Kernel found in '{neff_path}', using cached")
        else:
            # Clean output directory if it exists and we're recompiling
            if not use_cached_if_exists and os.path.exists(output_dir):
                logger.info(f"Cleaning output directory: {output_dir}")
                shutil.rmtree(output_dir)

            logger.info(f"Compiling kernel: {name}")
            time_start = time.time()
            if isinstance(kernel, types.FunctionType):
                # Treat untraced function as NKIPy
                traced_kernel = trace(kernel)
                compiler_args = compile.nkipy_compiler_args
            elif isinstance(kernel, NKIPyKernel):
                traced_kernel = kernel
                compiler_args = compile.nkipy_compiler_args
            else:
                logger.info("Continue as NKI kernel")
                traced_kernel = kernel
                compiler_args = compile.nki_compiler_args

            # Append user-provided additional compiler args if any
            if additional_compiler_args:
                compiler_args = compiler_args + " " + additional_compiler_args

            logger.debug(f"Compiler arguments: {compiler_args}")

            traced_kernel.specialize(*numpy_args, **numpy_kwargs)
            compile_to_neff(
                traced_kernel,
                output_dir=output_dir,
                neff_name=f"{name}.neff",
                additional_compiler_args=compiler_args,
                save_artifacts=True,
                target=target,
            )
            time_end = time.time()
            logger.info(f"Compile time: {time_end - time_start:.2f} seconds")

        if (
            device_tensor._TORCH_ENABLED
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            # make sure the lead is done with compilation
            dist.barrier()

            # Load with collective
            device_kernel = cls.load_from_neff(
                neff_path,
                name=name,
                cc_enabled=True,
                rank_id=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
        else:
            device_kernel = cls.load_from_neff(neff_path, name=name)

        if use_cached_if_exists:
            _LOADED_KERNELS[name] = device_kernel
        return device_kernel
