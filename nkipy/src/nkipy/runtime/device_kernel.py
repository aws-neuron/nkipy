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

    # TODO: this SerializeToString can be slow for large HLO
    h.update(hlo_module.to_proto().SerializeToString())
    h.update(compiler_args.encode("utf-8"))
    return h.hexdigest()[:12]


def _is_distributed() -> bool:
    """Check if running in a multi-worker torch.distributed setting."""
    return (
        device_tensor._TORCH_ENABLED
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


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
        cc_enabled=None,
        rank_id=None,
        world_size=None,
        **kwargs,
    ):
        """Compile and load a kernel, returning a DeviceKernel instance.

        Collective-communication behaviour depends on how ``cc_enabled`` is
        supplied:

        * **None (default)** – auto-detected from ``torch.distributed``.
          When distributed, rank 0 traces/compiles and broadcasts the NEFF
          path to all workers (SPMD pattern).
        * **Explicitly set** – every rank traces and compiles independently
          (MPMD pattern).  This is required when each rank runs a different
          kernel or uses different input shapes.  Also useful for
          non-torch-distributed runtimes that manage their own ranks.

        Args:
            kernel: The kernel function to compile
            name: Optional name for the kernel. If None, uses kernel.__name__
            additional_compiler_args: Optional additional compiler arguments to append
            use_cached_if_exists: If True, use cached neff if it exists.
            build_dir: Overriding the build directory for the kernel
            target: Compilation target for the kernel
            cc_enabled: Enable collective communication for this kernel.
                Auto-detected from torch.distributed when None.
            rank_id: Worker rank for CC. Auto-detected when None.
            world_size: Total workers for CC. Auto-detected when None.
            *args, **kwargs: Arguments for specialization (numpy array or DeviceTensor)

        Returns:
            DeviceKernel: A DeviceKernel instance with the compiled kernel
        """
        if name is None:
            name = kernel.__name__

        distributed = _is_distributed()

        # When cc_enabled is explicitly set, every rank traces and compiles
        # independently (MPMD).  Namespace the build dir by rank to avoid
        # concurrent writes when different ranks produce the same content hash.
        # Only auto-detected distributed mode uses rank-0 compile + broadcast (SPMD).
        if cc_enabled is not None and rank_id is not None:
            mpmd_build_dir = os.path.join(build_dir or _get_build_dir(), f"rank_{rank_id}")
        else:
            mpmd_build_dir = build_dir

        if distributed and cc_enabled is None:
            if dist.get_rank() == 0:
                neff_path, cache_key = cls._trace_and_compile(
                    kernel,
                    name,
                    args,
                    kwargs,
                    additional_compiler_args=additional_compiler_args,
                    use_cached_if_exists=use_cached_if_exists,
                    build_dir=mpmd_build_dir,
                    target=target,
                )
                dist.broadcast_object_list([neff_path, cache_key], src=0)
            else:
                info = [None, None]
                dist.broadcast_object_list(info, src=0)
                neff_path, cache_key = info
        else:
            neff_path, cache_key = cls._trace_and_compile(
                kernel,
                name,
                args,
                kwargs,
                additional_compiler_args=additional_compiler_args,
                use_cached_if_exists=use_cached_if_exists,
                build_dir=mpmd_build_dir,
                target=target,
            )

        # Check in-memory cache (consistent across all ranks)
        if use_cached_if_exists and cache_key in _LOADED_KERNELS:
            logger.info(f"Using loaded kernel: {name} (cache_key={cache_key})")
            return _LOADED_KERNELS[cache_key]

        # Resolve CC parameters: explicit args take priority, then torch.distributed.
        if cc_enabled is None and distributed:
            cc_enabled = True
            # Barrier only needed when rank 0 compiled for all workers.
            dist.barrier()
        if rank_id is None and distributed:
            rank_id = dist.get_rank()
        if world_size is None and distributed:
            world_size = dist.get_world_size()

        # Load the compiled NEFF
        if cc_enabled:
            device_kernel = cls.load_from_neff(
                neff_path,
                name=name,
                cc_enabled=True,
                rank_id=rank_id,
                world_size=world_size,
            )
        else:
            device_kernel = cls.load_from_neff(neff_path, name=name)

        if use_cached_if_exists:
            _LOADED_KERNELS[cache_key] = device_kernel
        return device_kernel

    @classmethod
    def _trace_and_compile(
        cls,
        kernel,
        name,
        args,
        kwargs,
        *,
        additional_compiler_args,
        use_cached_if_exists,
        build_dir,
        target,
    ):
        """Trace, specialize, hash, and compile a kernel.

        Returns:
            tuple[str, str]: (neff_path, cache_key)
        """
        # Determine compiler args
        if not isinstance(kernel, (types.FunctionType, NKIPyKernel)):
            raise NotImplementedError(f"Unsupported kernel type: {type(kernel)}")

        compiler_args = compile.nkipy_compiler_args
        if additional_compiler_args:
            compiler_args = compiler_args + " " + additional_compiler_args

        # Convert DeviceTensors to numpy arrays for tracing
        numpy_args = [
            arg.numpy() if isinstance(arg, DeviceTensor) else arg for arg in args
        ]
        numpy_kwargs = {
            k: v.numpy() if isinstance(v, DeviceTensor) else v
            for k, v in kwargs.items()
        }

        # Trace and specialize
        if isinstance(kernel, types.FunctionType):
            traced_kernel = trace(kernel)
        else:
            traced_kernel = kernel

        traced_kernel.specialize(*numpy_args, **numpy_kwargs)

        # Compute content hash from HLO
        hlo_module = traced_kernel._code
        if not isinstance(hlo_module, HLOModule):
            raise NotImplementedError("Only HLOModule is supported for content hashing")

        content_hash = _hlo_content_hash(hlo_module, compiler_args)
        cache_key = f"{name}_{content_hash}"

        # Determine output paths
        build_dir = build_dir or _get_build_dir()
        output_dir = f"{build_dir}/{name}_{content_hash}"
        neff_path = f"{output_dir}/{name}.neff"

        # Compile if needed
        if use_cached_if_exists and os.path.exists(neff_path):
            logger.info(
                f"Kernel found in '{neff_path}', using cached (hash={content_hash})"
            )
        else:
            if not use_cached_if_exists and os.path.exists(output_dir):
                logger.info(f"Cleaning output directory: {output_dir}")
                shutil.rmtree(output_dir)

            logger.info(f"Compiling kernel: {name} (hash={content_hash})")
            logger.debug(f"Compiler arguments: {compiler_args}")

            time_start = time.time()
            compile_to_neff(
                traced_kernel,
                output_dir=output_dir,
                neff_name=f"{name}.neff",
                additional_compiler_args=compiler_args,
                save_artifacts=True,
                target=target,
            )
            logger.info(f"Compile time: {time.time() - time_start:.2f} seconds")

        return neff_path, cache_key
