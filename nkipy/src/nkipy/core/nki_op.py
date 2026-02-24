# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKI kernel integration for NKIPy - wraps NKI kernels as HLO custom-calls

This module provides two ways to use NKI kernels in NKIPy:

1. Direct @nki.jit support (lazy/dynamic):
   - Any kernel decorated with @nki.jit can be called directly during NKIPy tracing
   - Supports grid syntax: kernel[grid_x, grid_y](a, b)
   - Tracing happens at call time with actual operand shapes

2. wrap_nki_kernel for specialized ops (eager/static):
   - Pre-traces the kernel for specific operand shapes
   - Returns a NKICustomOp that only works with those shapes
   - Useful for explicit control over specialization

Supports two NKI frontends:
- Legacy frontend (neuronxcc.nki): Default, supports CPU execution
- Beta 2 frontend (nki): New frontend, hardware-only (no CPU execution support)
"""

import dataclasses
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from nkipy.core.backend import get_backend
from nkipy.core.backend.hlo import get_hlo_context
from nkipy.core.tensor import NKIPyTensorRef

# Conditional imports for both frontends
# Legacy frontend (neuronxcc.nki)
try:
    from neuronxcc.nki.compile import GenericKernel as LegacyGenericKernel
    from neuronxcc.nki.compiler.backends.neuron.FrameworkKernel import (
        UnifiedKernel as LegacyUnifiedKernel,
    )

    LEGACY_NKI_AVAILABLE = True
except ImportError:
    LegacyGenericKernel = None
    LegacyUnifiedKernel = None
    LEGACY_NKI_AVAILABLE = False

# Beta 2 frontend (nki)
try:
    from nki.compile import GenericKernel as Beta2GenericKernel
    from nki.compiler.backends.neuron.FrameworkKernel import (
        UnifiedKernel as Beta2UnifiedKernel,
    )

    BETA2_NKI_AVAILABLE = True
except ImportError:
    Beta2GenericKernel = None
    Beta2UnifiedKernel = None
    BETA2_NKI_AVAILABLE = False


def _get_platform_target_default() -> str:
    """Get the default platform target from the system."""
    try:
        from nkipy.core.compile import get_platform_target

        return get_platform_target().value
    except Exception:
        # Fallback to trn1 if detection fails
        return "trn1"


def _patch_nkipy_methods(kernel):
    """Patch NKIPy-specific methods onto a kernel instance.

    GenericKernel (from @nki.jit) doesn't implement the framework-specific methods.
    We patch them directly onto the instance to enable NKIPy tensor handling.
    """
    kernel.is_framework_tensor = lambda t: isinstance(t, (np.ndarray, NKIPyTensorRef))
    kernel.map_framework_tensor = lambda t: (t.shape, t.dtype)
    kernel.translate_to_neuron_dtype = lambda d: d
    kernel.opts = dataclasses.replace(kernel.opts, enable_const_rewrite=True)


# Create NKIOp classes for each frontend
if LEGACY_NKI_AVAILABLE:

    class LegacyNKIOp(LegacyUnifiedKernel):
        """NKIPy-specific wrapper for legacy NKI (beta1)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.opts = dataclasses.replace(self.opts, enable_const_rewrite=True)

        def translate_to_neuron_dtype(self, _dtype):
            return _dtype

        def is_framework_tensor(self, t):
            return isinstance(t, (np.ndarray, NKIPyTensorRef))

        def map_framework_tensor(self, t):
            return t.shape, t.dtype
else:
    LegacyNKIOp = None


if BETA2_NKI_AVAILABLE:

    class Beta2NKIOp(Beta2UnifiedKernel):
        """NKIPy-specific wrapper for Beta 2 NKI."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.opts = dataclasses.replace(self.opts, enable_const_rewrite=True)

        def translate_to_neuron_dtype(self, _dtype):
            return _dtype

        def is_framework_tensor(self, t):
            return isinstance(t, (np.ndarray, NKIPyTensorRef))

        def map_framework_tensor(self, t):
            return t.shape, t.dtype
else:
    Beta2NKIOp = None


# Alias for backward compatibility - defaults to legacy if available, otherwise beta2
NKIOp = LegacyNKIOp if LEGACY_NKI_AVAILABLE else Beta2NKIOp


def _build_hlo_custom_call(config, operands):
    """Build HLO custom-call operation from a TraceResult config."""
    if get_backend() != "hlo":
        raise NotImplementedError("Modes other than HLO are not implemented yet")

    ctx = get_hlo_context()

    # Build HLO operands: user inputs + constants
    hlo_operands = [
        op.backend_tensor for op in operands if isinstance(op, NKIPyTensorRef)
    ]

    for const in config.constant_values:
        const_tensor = ctx.build_op(
            "constant",
            operands=[],
            result_shape=const.shape,
            result_dtype=const.dtype,
            attributes={"value": const},
        )
        hlo_operands.append(const_tensor)

    output_shapes = [shape for dtype, shape in config.return_types]
    output_dtypes = [dtype for dtype, shape in config.return_types]

    custom_call_attrs = {
        "custom_call_target": "AwsNeuronCustomNativeKernel",
        "backend_config": config.dumped_config,
    }

    if config.has_collectives:
        custom_call_attrs["has_collectives"] = True

    if config.operand_output_aliases:
        custom_call_attrs["operand_output_aliases"] = config.operand_output_aliases

    # Single output vs tuple output
    if len(output_shapes) == 1 and not config.result_is_sequence:
        result_tensor = ctx.build_op(
            "custom-call",
            hlo_operands,
            output_shapes[0],
            output_dtypes[0],
            custom_call_attrs,
        )
        return NKIPyTensorRef(result_tensor)
    else:
        custom_call_attrs["is_tuple"] = True
        result_tensor = ctx.build_op(
            "custom-call", hlo_operands, output_shapes, output_dtypes, custom_call_attrs
        )

        results = []
        for i in range(len(output_shapes)):
            element_tensor = ctx.build_op(
                "get-tuple-element",
                [result_tensor],
                output_shapes[i],
                output_dtypes[i],
                {"tuple_index": i},
            )
            results.append(NKIPyTensorRef(element_tensor))
        return tuple(results)


def _generate_nki_custom_call(kernel, *args, **kwargs):
    """Generate HLO custom-call for an NKI kernel during NKIPy tracing."""
    _patch_nkipy_methods(kernel)

    # Convert NKIPyTensorRef to empty numpy arrays for NKI specialization
    # Especially important for NKI Beta2 frontend
    # which doesn't support NKIPyTensorRef during specialize
    numpy_args = []
    for arg in args:
        if isinstance(arg, NKIPyTensorRef):
            # Create empty numpy array with same shape/dtype
            numpy_args.append(np.empty(arg.shape, dtype=arg.dtype))
        else:
            # Maybe numpy or non-tensor args
            numpy_args.append(arg)

    numpy_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, NKIPyTensorRef):
            numpy_kwargs[key] = np.empty(value.shape, dtype=value.dtype)
        else:
            numpy_kwargs[key] = value

    with kernel.bind_arguments(*numpy_args, **numpy_kwargs) as boundargs:
        config = kernel.dump_config_with_boundargs(boundargs)

    operands = [arg for arg in args if isinstance(arg, (NKIPyTensorRef, np.ndarray))]
    if get_backend() == "cpu":
        raise NotImplementedError("CPU execution is not supported for NKI custom ops")
    return _build_hlo_custom_call(config, operands)


# Monkey-patch to intercept jit calls during NKIPy tracing
if LEGACY_NKI_AVAILABLE:
    _original_legacy_generic_kernel_call = LegacyGenericKernel.__call__

    def _patched_legacy_generic_kernel_call(self, *args, **kwargs):
        """Patched __call__ that intercepts calls during NKIPy tracing."""
        if get_backend() != "cpu":
            return _generate_nki_custom_call(self, *args, **kwargs)
        return _original_legacy_generic_kernel_call(self, *args, **kwargs)

    LegacyGenericKernel.__call__ = _patched_legacy_generic_kernel_call


if BETA2_NKI_AVAILABLE:
    _original_beta2_generic_kernel_call = Beta2GenericKernel.__call__

    def _patched_beta2_generic_kernel_call(self, *args, **kwargs):
        """Patched __call__ that intercepts calls during NKIPy tracing."""
        if get_backend() != "cpu":
            # Note: Create a disposable copy of a GenericKernel for NKI tracing.

            # This is requried only for Beta2. The frontend.Kernel (kernel.kernel)
            # accumulates state during specialize/trace and cannot be reused.
            fresh = self
            fresh.kernel = type(self.kernel)(self.func)

            return _generate_nki_custom_call(fresh, *args, **kwargs)
        return _original_beta2_generic_kernel_call(self, *args, **kwargs)

    Beta2GenericKernel.__call__ = _patched_beta2_generic_kernel_call


class NKICustomOp:
    """Backward-compatible NKI custom op class.

    This class provides the original API for wrapping NKI kernels.
    New code should use wrap_nki_kernel() or direct @nki.jit instead.
    """

    def __init__(
        self,
        kernel: Callable,
        operands: Iterable,
        grid: Optional[Tuple[int, ...]] = (),
        kernel_return: bool = True,
        compiler_args: str = "",
        is_nki_beta_2_version: bool = False,
        platform_target: Optional[str] = None,
    ):
        operands = list(operands)

        # Select the appropriate NKIOp class based on frontend
        if is_nki_beta_2_version:
            if not BETA2_NKI_AVAILABLE:
                raise ImportError(
                    "Beta 2 NKI frontend (nki) is not available. Please install nki."
                )
            NKIOpClass = Beta2NKIOp
        else:
            if not LEGACY_NKI_AVAILABLE:
                raise ImportError(
                    "Legacy NKI frontend (neuronxcc.nki) is not available."
                    " Please install neuronxcc."
                )
            NKIOpClass = LegacyNKIOp

        # Determine platform target
        if platform_target is None:
            platform_target = _get_platform_target_default()

        # Trace the kernel with the appropriate NKIOp class
        traced_kernel = NKIOpClass.trace(
            kernel,
            grid=grid,
            kernel_return=kernel_return,
            experimental_flags=compiler_args,
            enable_cache=False,
            platform_target=platform_target,
        )

        # Get TraceResult config
        self.config = traced_kernel.dump_config(*operands)

    def __call__(self, *operands):
        if get_backend() == "cpu":
            raise NotImplementedError(
                "CPU execution is not supported for NKI custom ops"
            )
        return _build_hlo_custom_call(self.config, operands)


def wrap_nki_kernel(
    kernel: Callable,
    operands: Iterable,
    grid: Optional[Tuple[int, ...]] = (),
    is_nki_beta_2_version: bool = False,
    platform_target: Optional[str] = None,
):
    """Wrap an NKI kernel for use in NKIPy's HLO tracing flow.

    Pre-traces the kernel for the given operand shapes and returns a NKICustomOp.

    Args:
        kernel: The NKI kernel function (or @nki.jit decorated kernel)
        operands: Example operands (numpy arrays) for tracing (shape and dtype)
        grid: SPMD grid configuration (ignored if kernel is already @nki.jit with grid)
        is_nki_beta_2_version: If True, use the new Beta 2 NKI frontend (nki package).
                               If False, use the legacy frontend (neuronxcc.nki).
                               Note: Beta 2 frontend does not support CPU execution.
        platform_target: Target platform (e.g., "trn1", "trn2"). If None, auto-detected.

    Returns:
        NKICustomOp that can be called during HLO tracing
    """
    return NKICustomOp(
        kernel,
        operands,
        grid,
        is_nki_beta_2_version=is_nki_beta_2_version,
        platform_target=platform_target,
    )
