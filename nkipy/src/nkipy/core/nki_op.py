# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKI kernel integration for NKIPy.

This module provides three ways to use NKI kernels in NKIPy:

1. Direct @nki.jit support (lazy/dynamic):
   - Any kernel decorated with @nki.jit can be called directly during NKIPy tracing
   - Supports grid syntax: kernel[grid_x, grid_y](a, b)
   - Tracing happens at call time with actual operand shapes

2. wrap_nki_kernel for specialized ops (eager/static):
   - Pre-traces the kernel for specific operand shapes
   - Returns a NKICustomOp that only works with those shapes
   - Useful for explicit control over specialization

3. nki_custom_op for cross-backend custom ops:
   - Accepts both @nki.jit (HLO backend) and kernel_builder (nkigen backend)
   - Dispatches to the correct implementation based on the active backend

Supports three NKI frontends:
- Legacy frontend (neuronxcc.nki): Default, supports CPU execution
- Beta 2 frontend (nki with GenericKernel): Hardware-only (no CPU execution support)
- Beta 3 frontend (nki with Kernel): Hardware-only, new compilation API
"""

import dataclasses
import inspect
from typing import Callable, Iterable, List, Optional, Tuple

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

# Beta 2 frontend (nki with GenericKernel)
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

# Beta 3 frontend (nki with Kernel + compile_kernel_to_nir)
try:
    from nki.framework.kernel import Kernel as Beta3Kernel
    from nki.framework.compiled import compile_kernel_to_nir as _beta3_compile_kernel_to_nir
    from nki.compiler.ncc_driver import CompileOptions as Beta3CompileOptions

    BETA3_NKI_AVAILABLE = True
except ImportError:
    Beta3Kernel = None
    _beta3_compile_kernel_to_nir = None
    Beta3CompileOptions = None
    BETA3_NKI_AVAILABLE = False


def _get_platform_target_default() -> str:
    """Get the default platform target from the system."""
    try:
        from nkipy.core.compile import get_platform_target

        return get_platform_target().value
    except Exception:
        # Fallback to trn1 if detection fails
        return "trn1"


# Beta 3 BIR artifacts must persist until neuronx-cc compiles the HLO module.
# Use a process-level temp directory that lives until the process exits.
_beta3_base_artifacts_dir = None
_beta3_artifacts_counter = 0


def _get_beta3_artifacts_dir() -> str:
    """Get a unique persistent directory for beta 3 BIR artifacts."""
    import os
    import tempfile

    global _beta3_base_artifacts_dir, _beta3_artifacts_counter
    if _beta3_base_artifacts_dir is None:
        _beta3_base_artifacts_dir = tempfile.mkdtemp(prefix="nkipy_beta3_bir_")
    _beta3_artifacts_counter += 1
    subdir = os.path.join(_beta3_base_artifacts_dir, str(_beta3_artifacts_counter))
    os.makedirs(subdir, exist_ok=True)
    return subdir


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


def _emit_hlo_custom_call(
    hlo_operands,
    tensor_operands,
    output_shapes,
    output_dtypes,
    backend_config,
    has_collectives,
    alias_map,
    is_tuple_return,
):
    """Emit the HLO custom-call op shared by all NKI frontends.

    Callers normalize their frontend-specific config into these arguments:

        hlo_operands:    backend tensors passed to the custom-call (inputs +
                         any frontend-managed constants).
        tensor_operands: the NKIPyTensorRef inputs, in operand order, used to
                         resolve output aliases back to their input tensor.
        output_shapes/output_dtypes: per-output result types.
        backend_config:  serialized kernel config blob.
        has_collectives: whether the kernel uses collectives.
        alias_map:       {input_operand_idx: output_idx} aliasing, indexed over
                         ``tensor_operands``.
        is_tuple_return: force a tuple result even for a single output.
    """
    ctx = get_hlo_context()

    custom_call_attrs = {
        "custom_call_target": "AwsNeuronCustomNativeKernel",
        "backend_config": backend_config,
    }

    if has_collectives:
        custom_call_attrs["has_collectives"] = True

    alias_map = alias_map or {}
    if alias_map:
        custom_call_attrs["operand_output_aliases"] = alias_map
    # Invert to output_idx -> input_operand_idx for result construction
    output_alias_map = {out_idx: in_idx for in_idx, out_idx in alias_map.items()}

    def _resolve(output_idx, result_tensor):
        """Wrap a result tensor, mutating the aliased input in place if any."""
        if output_idx in output_alias_map:
            original = tensor_operands[output_alias_map[output_idx]]
            original._is_mutated = True
            original.backend_tensor = result_tensor
            original._shape = result_tensor.shape
            original._dtype = result_tensor.dtype
            return original
        return NKIPyTensorRef(result_tensor)

    # Single output vs tuple output
    if len(output_shapes) == 1 and not is_tuple_return:
        result_tensor = ctx.build_op(
            "custom-call",
            hlo_operands,
            output_shapes[0],
            output_dtypes[0],
            custom_call_attrs,
        )
        return _resolve(0, result_tensor)

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
        results.append(_resolve(i, element_tensor))
    return tuple(results)


def _build_hlo_custom_call(config, operands):
    """Build HLO custom-call operation from a TraceResult config (beta 1/2)."""
    if get_backend() != "hlo":
        raise NotImplementedError("Modes other than HLO are not implemented yet")

    ctx = get_hlo_context()

    # Collect tensor operands (preserving order) for alias resolution
    tensor_operands = [op for op in operands if isinstance(op, NKIPyTensorRef)]

    # Build HLO operands: user inputs + constants
    hlo_operands = [op.backend_tensor for op in tensor_operands]

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

    return _emit_hlo_custom_call(
        hlo_operands=hlo_operands,
        tensor_operands=tensor_operands,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend_config=config.dumped_config,
        has_collectives=config.has_collectives,
        # NKI alias map: {input_operand_idx: output_idx}
        alias_map=config.operand_output_aliases,
        is_tuple_return=config.result_is_sequence,
    )


def _build_hlo_custom_call_beta3(framework_config, is_tuple_return, operands):
    """Build HLO custom-call operation from a beta 3 FrameworkConfig."""
    if get_backend() != "hlo":
        raise NotImplementedError("Modes other than HLO are not implemented yet")

    # Collect tensor operands (preserving order) for alias resolution
    tensor_operands = [op for op in operands if isinstance(op, NKIPyTensorRef)]

    # Build HLO operands (beta 3 handles constants internally)
    hlo_operands = [op.backend_tensor for op in tensor_operands]

    output_shapes = [tuple(spec.shape) for spec in framework_config.output_specs]
    output_dtypes = [np.dtype(spec.dtype) for spec in framework_config.output_specs]

    return _emit_hlo_custom_call(
        hlo_operands=hlo_operands,
        tensor_operands=tensor_operands,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend_config=framework_config.backend_config_b64,
        has_collectives=framework_config.has_collectives,
        # Beta 3 alias map: {input_idx: output_idx}
        alias_map=framework_config.operand_output_aliases,
        is_tuple_return=is_tuple_return,
    )


def _generate_nki_custom_call(kernel, *args, **kwargs):
    """Generate HLO custom-call for an NKI kernel during NKIPy tracing (beta 1/2)."""
    _patch_nkipy_methods(kernel)

    # Bind original args/kwargs to the kernel signature to get parameter-ordered
    # arguments. This is used both for NKI specialization and to collect tensor
    # operands in the correct order for the HLO custom call.
    func = getattr(kernel, "func", kernel)
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # Convert NKIPyTensorRef to empty numpy arrays for NKI specialization
    # Especially important for NKI Beta2 frontend
    # which doesn't support NKIPyTensorRef during specialize
    numpy_bound = {
        k: np.empty(v.shape, dtype=v.dtype) if isinstance(v, NKIPyTensorRef) else v
        for k, v in bound.arguments.items()
    }

    with kernel.bind_arguments(**numpy_bound) as boundargs:
        config = kernel.dump_config_with_boundargs(boundargs)

    # Collect tensor operands in parameter order (matching the traced config).
    operands = [
        v
        for v in bound.arguments.values()
        if isinstance(v, (NKIPyTensorRef, np.ndarray))
    ]
    if get_backend() == "cpu":
        raise NotImplementedError("CPU execution is not supported for NKI custom ops")
    return _build_hlo_custom_call(config, operands)


def _beta3_compile_and_get_config(kernel, numpy_inputs, platform_target=None, lnc=None):
    """Compile a beta 3 kernel and return (framework_config, is_tuple_return).

    The artifacts directory is managed by the caller via _get_beta3_artifacts_dir().
    """
    import os

    if platform_target is None:
        platform_target = _get_platform_target_default()
    if lnc is None:
        lnc = getattr(kernel, "lnc", 1) or 1

    artifacts_dir = _get_beta3_artifacts_dir()
    compile_opts = Beta3CompileOptions(
        target=platform_target,
        lnc=lnc,
        artifacts_dir=artifacts_dir,
        output_path=os.path.join(artifacts_dir, "kernel.neff"),
    )
    nir = _beta3_compile_kernel_to_nir(
        kernel, inputs=numpy_inputs, compile_opts=compile_opts, enable_cache=False
    )
    return nir.build_config(), nir.is_tuple_return


def _generate_nki_custom_call_beta3(kernel, *args, **kwargs):
    """Generate HLO custom-call for a beta 3 NKI kernel during NKIPy tracing."""
    func = getattr(kernel, "func", kernel)
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # Convert NKIPyTensorRef to empty numpy arrays for compilation
    numpy_inputs = {
        k: np.empty(v.shape, dtype=v.dtype) if isinstance(v, NKIPyTensorRef) else v
        for k, v in bound.arguments.items()
    }

    framework_config, is_tuple_return = _beta3_compile_and_get_config(kernel, numpy_inputs)

    # Collect tensor operands in parameter order
    operands = [
        v
        for v in bound.arguments.values()
        if isinstance(v, (NKIPyTensorRef, np.ndarray))
    ]
    if get_backend() == "cpu":
        raise NotImplementedError("CPU execution is not supported for NKI custom ops")
    return _build_hlo_custom_call_beta3(
        framework_config, is_tuple_return, operands
    )


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
            # No longer need disposable GenericKernel copy:
            # fixed in nki 2.28.0 release
            return _generate_nki_custom_call(self, *args, **kwargs)
        return _original_beta2_generic_kernel_call(self, *args, **kwargs)

    Beta2GenericKernel.__call__ = _patched_beta2_generic_kernel_call


if BETA3_NKI_AVAILABLE:
    _original_beta3_kernel_call = Beta3Kernel.__call__

    def _patched_beta3_kernel_call(self, *args, **kwargs):
        """Patched __call__ that intercepts calls during NKIPy tracing."""
        if get_backend() != "cpu":
            return _generate_nki_custom_call_beta3(self, *args, **kwargs)
        return _original_beta3_kernel_call(self, *args, **kwargs)

    Beta3Kernel.__call__ = _patched_beta3_kernel_call


class NKICustomOp:
    """HLO custom-call wrapper for a pre-traced NKI kernel.

    Pre-traces the kernel at construction time for specific operand shapes.
    Used by ``wrap_nki_kernel``.
    """

    def __init__(
        self,
        kernel: Callable,
        operands: Iterable,
        grid: Optional[Tuple[int, ...]] = (),
        kernel_return: bool = True,
        compiler_args: str = "",
        is_nki_beta_2_version: bool = False,
        is_nki_beta_3_version: bool = False,
        platform_target: Optional[str] = None,
    ):
        operands = list(operands)
        self._is_beta3 = is_nki_beta_3_version

        if platform_target is None:
            platform_target = _get_platform_target_default()

        if is_nki_beta_3_version:
            if not BETA3_NKI_AVAILABLE:
                raise ImportError(
                    "Beta 3 NKI frontend (nki.framework.kernel.Kernel) is not "
                    "available. Please install nki >= 0.4."
                )
            self._compile_beta3(kernel, operands, platform_target)
        elif is_nki_beta_2_version:
            if not BETA2_NKI_AVAILABLE:
                raise ImportError(
                    "Beta 2 NKI frontend (nki) is not available. Please install nki."
                )
            self._compile_beta2(
                kernel, operands, grid, kernel_return, compiler_args, platform_target
            )
        else:
            if not LEGACY_NKI_AVAILABLE:
                raise ImportError(
                    "Legacy NKI frontend (neuronxcc.nki) is not available."
                    " Please install neuronxcc."
                )
            self._compile_legacy(
                kernel, operands, grid, kernel_return, compiler_args, platform_target
            )

    def _compile_legacy(
        self, kernel, operands, grid, kernel_return, compiler_args, platform_target
    ):
        traced_kernel = LegacyNKIOp.trace(
            kernel,
            grid=grid,
            kernel_return=kernel_return,
            experimental_flags=compiler_args,
            enable_cache=False,
            platform_target=platform_target,
        )
        self.config = traced_kernel.dump_config(*operands)

    def _compile_beta2(
        self, kernel, operands, grid, kernel_return, compiler_args, platform_target
    ):
        traced_kernel = Beta2NKIOp.trace(
            kernel,
            grid=grid,
            kernel_return=kernel_return,
            experimental_flags=compiler_args,
            enable_cache=False,
            platform_target=platform_target,
        )
        self.config = traced_kernel.dump_config(*operands)

    def _compile_beta3(self, kernel, operands, platform_target):
        # Build inputs dict from operands matching kernel parameter names
        func = getattr(kernel, "func", kernel)
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        numpy_inputs = {}
        op_idx = 0
        for p in params:
            if op_idx < len(operands):
                numpy_inputs[p] = operands[op_idx]
                op_idx += 1
            else:
                break

        self._beta3_framework_config, self._beta3_is_tuple_return = (
            _beta3_compile_and_get_config(kernel, numpy_inputs, platform_target)
        )

    def __call__(self, *operands):
        if get_backend() == "cpu":
            raise NotImplementedError(
                "CPU execution is not supported for NKI custom ops"
            )
        if self._is_beta3:
            return _build_hlo_custom_call_beta3(
                self._beta3_framework_config, self._beta3_is_tuple_return, operands
            )
        return _build_hlo_custom_call(self.config, operands)


def wrap_nki_kernel(
    kernel: Callable,
    operands: Iterable,
    grid: Optional[Tuple[int, ...]] = (),
    is_nki_beta_2_version: bool = False,
    is_nki_beta_3_version: bool = False,
    platform_target: Optional[str] = None,
):
    """Wrap an NKI kernel for use in NKIPy's HLO tracing flow.

    Pre-traces the kernel for the given operand shapes and returns a NKICustomOp.

    Args:
        kernel: The NKI kernel function (or @nki.jit decorated kernel)
        operands: Example operands (numpy arrays) for tracing (shape and dtype)
        grid: SPMD grid configuration (ignored if kernel is already @nki.jit with grid)
        is_nki_beta_2_version: If True, use the Beta 2 NKI frontend (nki package
                               with GenericKernel). Note: does not support CPU execution.
        is_nki_beta_3_version: If True, use the Beta 3 NKI frontend (nki >= 0.4 with
                               compile_kernel_to_nir). Note: does not support CPU execution.
        platform_target: Target platform (e.g., "trn1", "trn2"). If None, auto-detected.

    Returns:
        NKICustomOp that can be called during HLO tracing
    """
    return NKICustomOp(
        kernel,
        operands,
        grid,
        is_nki_beta_2_version=is_nki_beta_2_version,
        is_nki_beta_3_version=is_nki_beta_3_version,
        platform_target=platform_target,
    )


# ---------------------------------------------------------------------------
# NkiGen custom op support
# ---------------------------------------------------------------------------


def _generate_nkigen_custom_call(kernel_builder, input_specs, output_specs, *args):
    """Compile a kernel_builder function and inline it during nkigen tracing."""
    from nkigen.builder import apply_custom_op

    return apply_custom_op(
        kernel_builder=kernel_builder,
        reference_fn=None,
        input_specs=input_specs,
        output_specs=output_specs,
        args=args,
    )


# ---------------------------------------------------------------------------
# Unified custom op interface
# ---------------------------------------------------------------------------


def nki_custom_op(
    *,
    nki_kernel: Optional[Callable] = None,
    kernel_builder: Optional[Callable] = None,
    input_specs: Optional[List[Tuple[Tuple[int, ...], str]]] = None,
    output_specs: Optional[List[Tuple[Tuple[int, ...], str]]] = None,
) -> "NKICustomOpHandle":
    """Create a cross-backend custom NKI op.

    Args:
        nki_kernel: ``@nki.jit`` decorated kernel for the HLO backend.
        kernel_builder: ``nki.compiler.kernel_builder`` function for the
            nkigen backend.  Requires ``input_specs`` and ``output_specs``.
        input_specs: List of ``((shape), dtype_str)`` for each input.
            Required when ``kernel_builder`` is provided.
        output_specs: List of ``((shape), dtype_str)`` for each output.
            Required when ``kernel_builder`` is provided.

    Returns:
        An ``NKICustomOpHandle`` callable that dispatches to the correct
        backend at call time.
    """
    if nki_kernel is None and kernel_builder is None:
        raise ValueError(
            "At least one of nki_kernel or kernel_builder must be provided."
        )
    if kernel_builder is not None:
        if input_specs is None or output_specs is None:
            raise ValueError(
                "input_specs and output_specs are required when kernel_builder "
                "is provided."
            )
    return NKICustomOpHandle(
        nki_kernel=nki_kernel,
        kernel_builder=kernel_builder,
        input_specs=input_specs,
        output_specs=output_specs,
    )


class NKICustomOpHandle:
    """Backend-aware callable wrapping a custom NKI op definition."""

    def __init__(
        self,
        *,
        nki_kernel: Optional[Callable],
        kernel_builder: Optional[Callable],
        input_specs: Optional[List[Tuple[Tuple[int, ...], str]]],
        output_specs: Optional[List[Tuple[Tuple[int, ...], str]]],
    ):
        self._nki_kernel = nki_kernel
        self._kernel_builder = kernel_builder
        self._input_specs = input_specs
        self._output_specs = output_specs

    def __call__(self, *args):
        backend = get_backend()

        if backend == "hlo":
            if self._nki_kernel is None:
                raise RuntimeError(
                    "nki_custom_op has no nki_kernel for the HLO backend. "
                    "Provide an @nki.jit decorated kernel via nki_kernel=."
                )
            if BETA3_NKI_AVAILABLE and isinstance(self._nki_kernel, Beta3Kernel):
                return _generate_nki_custom_call_beta3(self._nki_kernel, *args)
            return _generate_nki_custom_call(self._nki_kernel, *args)

        if backend == "nkigen":
            if self._kernel_builder is None:
                raise RuntimeError(
                    "nki_custom_op has no kernel_builder for the nkigen "
                    "backend. Provide a kernel_builder function via "
                    "kernel_builder=."
                )
            return _generate_nkigen_custom_call(
                self._kernel_builder, self._input_specs, self._output_specs,
                *args,
            )

        raise RuntimeError(
            f"nki_custom_op is not supported on backend '{backend}'. "
            f"Use the 'hlo' or 'nkigen' backend."
        )
