# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-NKIPy project
"""Combined compilation utilities for NKIPy backend.

This module provides:
1. Backend compilation function (compile_fn) for torch.compile
2. local_compile decorator for conditional compilation with device transfer
3. NKIPY_BACKEND constant for use in torch.compile(backend=...)
"""

import builtins
import functools
from collections.abc import Sequence
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.fx as fx
from torch._decomp import core_aten_decompositions
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.utils import InputType
from vllm.logger import init_logger

import vllm_nkipy.envs as envs
from torch_to_nkipy.backend.nkipy_backend import nkipy_backend_fn

logger = init_logger(__name__)

F = TypeVar("F", bound=Callable)

# ============================================================================
# Module Configuration
# ============================================================================

# Whether to use NKIPy backend or CPU fallback for torch.compile
# Controlled via VLLM_NKIPY_COMPILE_USE_NKIPY env var (default: 1)
use_nkipy = envs.VLLM_NKIPY_COMPILE_USE_NKIPY == 1
logger.info(f"{use_nkipy=}")

# Module-level default for split_graph.
# This can be set by calling set_default_split_graph() before compilation.
# Used when torch.compile/local_compile decorators don't pass split_graph explicitly.
# Note: The backend now handles splitting automatically based on subgraph markers.
_default_split_graph: bool = True


# ============================================================================
# Configuration Functions
# ============================================================================


def set_default_split_graph(value: bool) -> None:
    """Set the default split_graph for all compile_fn calls.

    Call this before model loading to ensure all @torch.compile and
    @local_compile decorators use the correct backend.

    Note: The backend now handles splitting automatically based on subgraph
    markers in the graph. This flag is kept for configuration consistency.

    Args:
        value: Whether to enable graph splitting.
    """
    global _default_split_graph
    _default_split_graph = value
    logger.info(f"Set default split_graph={value}")


def get_default_split_graph() -> bool:
    """Get the current default split_graph value."""
    return _default_split_graph


def use_nkipy_backend():
    """Enable NKIPy backend (for testing/override)."""
    global use_nkipy
    use_nkipy = True


# ============================================================================
# Backend Compilation Functions
# ============================================================================


def test_on_host_compile_dcomposed(
    gm: fx.GraphModule, example_inputs: Sequence[InputType]
) -> Callable:
    """CPU fallback compiler for testing."""
    # gm.print_readable()
    return gm


def compile_fn(
    gm: fx.GraphModule,
    example_inputs: Sequence[InputType],
    options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
) -> Callable:
    """Main compile function for torch.compile backend.

    Args:
        gm: The FX GraphModule to compile.
        example_inputs: Example inputs for tracing.
        options: Optional compilation options passed to the backend.

    Returns:
        Compiled callable.

    Note:
        The backend now handles graph splitting automatically based on
        subgraph markers present in the graph. The split_graph configuration
        flag is kept for consistency but splitting is determined by markers.
    """
    # gm.print_readable()
    if use_nkipy:
        return nkipy_backend_fn(gm, example_inputs, options)
    else:
        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=test_on_host_compile_dcomposed,
            decompositions=core_aten_decompositions(),
            keep_inference_input_mutations=True,
        )


# ============================================================================
# Backend Constant
# ============================================================================

# Use this constant instead of `backend = compile_fn` pattern
# Example: torch.compile(backend=NKIPY_BACKEND, ...)
NKIPY_BACKEND = compile_fn


# ============================================================================
# Local Compile Decorator
# ============================================================================


def local_compile(
    *args, name=None, device=None, force=False, **kwargs
) -> Callable[[F], F]:
    """
    Conditionally apply torch.compile based on VLLM_NKIPY_ENABLE_COMPILE env var.

    When VLLM_NKIPY_ENABLE_COMPILE=1 (default), applies
    torch.compile with the given parameters.
    When VLLM_NKIPY_ENABLE_COMPILE=0, returns the function
    unchanged.

    Args:
        name: Optional name for this module. If provided, compilation is controlled by
              VLLM_NKIPY_COMPILE_MODULES env var (list of names to compile).
              If name is not in the list, the function is returned unchanged.
        device: Optional device to transfer tensors to before
                execution. If set, tensor inputs will be
                transferred to this device, the function
                executed, and outputs transferred back to
                the original device.
        force: If True, always compile regardless of environment settings.
        *args: Additional arguments to pass to torch.compile.
        **kwargs: Additional keyword arguments to pass to torch.compile.

    Returns:
        A decorator function
    """

    def decorator(func: F) -> F:
        # Check if torch.compile is enabled via env var (default to 1)
        enable_compile = envs.VLLM_NKIPY_ENABLE_COMPILE == 1 or force

        if not enable_compile:
            # Return the function unchanged
            return func

        # Check if selective compilation is enabled via VLLM_NKIPY_COMPILE_MODULES
        compile_modules = envs.VLLM_NKIPY_COMPILE_MODULES
        if compile_modules is not None and name is not None and not force:
            if name not in compile_modules:
                # This module is not in the compile list, return unchanged
                return func

        # Apply torch.compile with the specified parameters
        compiled_func = torch.compile(*args, **kwargs)(func)

        # Wrap with device transfer if needed
        if device is not None:
            return _wrap_with_device_transfer(compiled_func, device, name)

        return compiled_func

    return decorator


def _wrap_with_device_transfer(
    func: F, target_device: torch.device, func_name: Optional[str] = None
) -> F:
    """Wrap a function to handle device transfers for tensor inputs/outputs.

    Args:
        func: The function to wrap
        target_device: The device to transfer tensors to for execution
        func_name: Optional name for logging

    Returns:
        Wrapped function that handles device transfers
    """
    # Get output device from environment variable
    output_device_str = envs.VLLM_NKIPY_COMPILE_OUTPUT_DEVICE
    output_device = None
    if output_device_str is not None:
        output_device = torch.device(output_device_str)
        logger.info(
            "VLLM_NKIPY_COMPILE_OUTPUT_DEVICE set to: %s",
            output_device,
        )

    # Check if we should save inputs/outputs for this function
    save_io_modules = envs.VLLM_NKIPY_COMPILE_SAVE_IO
    should_save_io = (
        save_io_modules is not None
        and func_name is not None
        and func_name in save_io_modules
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Track original devices and whether transfer is needed
        original_devices = {}
        needs_transfer = False

        func_display_name = func_name or func.__name__

        # Helper to log tensor devices
        def log_tensor_devices(obj, prefix=""):
            devices = []
            if isinstance(obj, torch.Tensor):
                devices.append(
                    f"{prefix}: device={obj.device}, "
                    f"shape={obj.shape}, "
                    f"dtype={obj.dtype}"
                )
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    devices.extend(log_tensor_devices(item, f"{prefix}[{i}]"))
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    devices.extend(log_tensor_devices(v, f"{prefix}[{k}]"))
            return devices

        # Helper to transfer tensor to device
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, (list, tuple)):
                return type(obj)(to_device(item, device) for item in obj)
            elif isinstance(obj, dict):
                return {k: to_device(v, device) for k, v in obj.items()}
            return obj

        # Helper to save values (tensors and scalars) using debug_utils
        def save_values(obj, prefix=""):
            """Recursively save values to disk using debug_utils."""
            from vllm_nkipy.model_executor.models.debug_utils import log_value

            if isinstance(obj, torch.Tensor):
                # Convert to CPU before saving
                value_cpu = obj.detach().cpu()
                log_value(value_cpu, f"{func_display_name}_{prefix}")
            elif isinstance(obj, (int, float, bool, type(None))):
                # Save scalar values directly
                log_value(obj, f"{func_display_name}_{prefix}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    save_values(item, f"{prefix}_arg{i}" if prefix else f"arg{i}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    save_values(v, f"{prefix}_{k}" if prefix else str(k))
            else:
                # Try to save other types as well
                try:
                    log_value(obj, f"{func_display_name}_{prefix}")
                except Exception as e:
                    logger.warning(
                        "Could not save value of type %s: %s",
                        type(obj),
                        e,
                    )

        # Check original devices of tensor inputs
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                orig_device = arg.device
                original_devices[("arg", i)] = orig_device
                if orig_device != target_device:
                    needs_transfer = True

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                orig_device = value.device
                original_devices[("kwarg", key)] = orig_device
                if orig_device != target_device:
                    needs_transfer = True

        # If no transfer needed, just call the function
        if not needs_transfer:
            result = func(*args, **kwargs)
            # Still need to check if we should transfer output to a different device
            if output_device is not None:
                logger.debug(
                    "[%s] Output devices BEFORE "
                    "transfer to output device:",
                    func_display_name,
                )
                for device_info in log_tensor_devices(
                    result, "  result"
                ):
                    logger.debug(
                        "[%s]   %s",
                        func_display_name,
                        device_info,
                    )

                logger.debug(
                    "[%s] Transferring result to "
                    "output device: %s",
                    func_display_name,
                    output_device,
                )
                result = to_device(result, output_device)

                logger.debug(
                    "[%s] Output devices AFTER "
                    "transfer to %s:",
                    func_display_name,
                    output_device,
                )
                for device_info in log_tensor_devices(
                    result, "  result"
                ):
                    logger.debug(
                        "[%s]   %s",
                        func_display_name,
                        device_info,
                    )
            return result

        # Log the transfer with detailed device info
        logger.debug(
            "[%s] Transferring tensors to %s",
            func_display_name,
            target_device,
        )

        # Log input devices before transfer
        logger.debug(
            "[%s] Input devices BEFORE transfer:",
            func_display_name,
        )
        for i, arg in enumerate(args):
            for device_info in log_tensor_devices(
                arg, f"  arg[{i}]"
            ):
                logger.debug(
                    "[%s]   %s", func_display_name, device_info
                )
        for key, value in kwargs.items():
            for device_info in log_tensor_devices(
                value, f"  kwarg[{key}]"
            ):
                logger.debug(
                    "[%s]   %s", func_display_name, device_info
                )

        # Transfer inputs to target device
        args_on_device = to_device(args, target_device)
        kwargs_on_device = to_device(kwargs, target_device)

        # Log input devices after transfer
        logger.debug(
            "[%s] Input devices AFTER transfer to %s:",
            func_display_name,
            target_device,
        )
        for i, arg in enumerate(args_on_device):
            for device_info in log_tensor_devices(
                arg, f"  arg[{i}]"
            ):
                logger.debug(
                    "[%s]   %s", func_display_name, device_info
                )
        for key, value in kwargs_on_device.items():
            for device_info in log_tensor_devices(
                value, f"  kwarg[{key}]"
            ):
                logger.debug(
                    "[%s]   %s", func_display_name, device_info
                )

        # Save inputs if requested
        if should_save_io:
            logger.info(
                "[%s] Saving inputs to disk...",
                func_display_name,
            )
            save_values(args_on_device, "input_args")
            save_values(kwargs_on_device, "input_kwargs")

        # Execute function
        result = func(*args_on_device, **kwargs_on_device)

        # Save outputs if requested
        if should_save_io:
            logger.info(
                "[%s] Saving outputs to disk...",
                func_display_name,
            )
            save_values(result, "output")

        # Log output device before transfer
        logger.debug(
            "[%s] Output devices BEFORE transfer:",
            func_display_name,
        )
        for device_info in log_tensor_devices(
            result, "  result"
        ):
            logger.debug(
                "[%s]   %s", func_display_name, device_info
            )

        # Determine the output device
        # Use VLLM_NKIPY_COMPILE_OUTPUT_DEVICE if set, otherwise use original device
        if output_device is not None:
            final_device = output_device
            logger.debug(
                "[%s] Transferring result to "
                "output device: %s",
                func_display_name,
                final_device,
            )
        elif original_devices:
            final_device = next(iter(original_devices.values()))
            logger.debug(
                "[%s] Transferring result back to "
                "original device: %s",
                func_display_name,
                final_device,
            )
        else:
            # No transfer needed
            return result

        result = to_device(result, final_device)

        # Log output device after transfer
        logger.debug(
            "[%s] Output devices AFTER transfer to %s:",
            func_display_name,
            final_device,
        )
        for device_info in log_tensor_devices(
            result, "  result"
        ):
            logger.debug(
                "[%s]   %s", func_display_name, device_info
            )

        return result

    return wrapper
