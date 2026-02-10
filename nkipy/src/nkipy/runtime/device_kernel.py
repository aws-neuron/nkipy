# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit
import hashlib
import os
import shutil
import time
import types

from nkipy.core import compile
from nkipy.core.backend.hlo import HLOModule
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


def _hlo_content_hash(hlo_module: HLOModule, compiler_args: str) -> str:
    """Compute a content hash from the HLO protobuf and compiler args.

    Hashing the HLO (instead of source code) ensures that different input
    shapes/dtypes produce different cache entries, even when the kernel
    source is identical.

    The HLO proto uses only ``repeated`` fields (no ``map`` fields), so
    ``SerializeToString()`` is deterministic for the same computation graph.
    """
    h = hashlib.sha256()
    h.update(hlo_module.to_proto().SerializeToString())
    h.update(compiler_args.encode("utf-8"))
    return h.hexdigest()[:12]


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
            name = kernel.__name__

        # Determine compiler args early so they can be included in the hash
        if isinstance(kernel, types.FunctionType):
            base_compiler_args = compile.nkipy_compiler_args
        elif isinstance(kernel, NKIPyKernel):
            base_compiler_args = compile.nkipy_compiler_args
        else:
            raise NotImplementedError(f"Unsupported kernel type: {type(kernel)}")
        if additional_compiler_args:
            full_compiler_args = base_compiler_args + " " + additional_compiler_args
        else:
            full_compiler_args = base_compiler_args

        # Convert DeviceTensors to numpy arrays for tracing/compilation
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

        # Trace and specialize BEFORE hashing so that the cache key
        # reflects the actual HLO (which depends on input shapes/dtypes),
        # not just the source code.
        if isinstance(kernel, types.FunctionType):
            traced_kernel = trace(kernel)
        elif isinstance(kernel, NKIPyKernel):
            traced_kernel = kernel
        else:
            logger.info("Continue as NKI kernel")
            traced_kernel = kernel

        traced_kernel.specialize(*numpy_args, **numpy_kwargs)

        # Compute content hash
        hlo_module = traced_kernel._code
        if isinstance(hlo_module, HLOModule):
            content_hash = _hlo_content_hash(hlo_module, full_compiler_args)
        else:
            raise NotImplementedError("Only HLOModule is supported for content hashing")
        cache_key = f"{name}_{content_hash}"

        if use_cached_if_exists and cache_key in _LOADED_KERNELS:
            logger.info(f"Using loaded kernel: {name} (hash={content_hash})")
            return _LOADED_KERNELS[cache_key]

        build_dir = build_dir or _get_build_dir()

        # Include hash in the directory path so different HLO → different NEFF
        output_dir = f"{build_dir}/{name}_{content_hash}"
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
            logger.info(
                f"Kernel found in '{neff_path}', using cached (hash={content_hash})"
            )
        else:
            # Clean output directory if it exists and we're recompiling
            if not use_cached_if_exists and os.path.exists(output_dir):
                logger.info(f"Cleaning output directory: {output_dir}")
                shutil.rmtree(output_dir)

            logger.info(f"Compiling kernel: {name} (hash={content_hash})")
            time_start = time.time()

            logger.debug(f"Compiler arguments: {full_compiler_args}")

            # traced_kernel is already specialized above; just compile.
            compile_to_neff(
                traced_kernel,
                output_dir=output_dir,
                neff_name=f"{name}.neff",
                additional_compiler_args=full_compiler_args,
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
            _LOADED_KERNELS[cache_key] = device_kernel
        return device_kernel
