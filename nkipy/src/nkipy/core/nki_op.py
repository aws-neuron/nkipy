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

Beta 3 kernels compile through ``nki.framework.compiled.CompileKernel`` (see
``_NKIPyCompileKernel``), which is where the allocation/scheduling split between
NKI and the neuronx-cc backend is decided; ``BETA3_COMPILE_MODES`` documents the
available modes.
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

# Beta 3 frontend (nki with Kernel + CompileKernel)
try:
    from nki.framework.kernel import Kernel as Beta3Kernel
    from nki.framework.compiled import CompileKernel as Beta3CompileKernel

    BETA3_NKI_AVAILABLE = True
except ImportError:
    Beta3Kernel = None
    Beta3CompileKernel = None
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


# ---------------------------------------------------------------------------
# Beta 3 compilation modes
# ---------------------------------------------------------------------------

# How the NKI beta 3 MLIR pipeline and the neuronx-cc backend split
# responsibility for memory allocation and instruction scheduling:
#
#   "standard"          neuronx-cc owns allocation + scheduling. NKI's
#                       LinearScanAllocation and InstructionScheduling passes are
#                       off, so the BIR carries unallocated memory locations and
#                       neuronx-cc runs its coloring allocator and PSUM address
#                       rotation. This is the production path -- the same one
#                       nki.framework's own torch/JAX integrations use.
#
#   "integration-alloc" NKI owns allocation (LinearScanAllocation on), neuronx-cc
#                       still owns scheduling. The BIR is stamped
#                       ``sb_allocated``/``psum_allocated``, so neuronx-cc skips its
#                       allocator.
BETA3_COMPILE_MODES = ("standard", "integration-alloc")
BETA3_ALLOCATION_MODES = ("fast", "ring", "max-reuse")


def _beta3_supports_allocation_mode() -> bool:
    """Whether the installed nki exposes the ``nisa-allocation-mode`` option."""
    import glob
    import os

    try:
        from nki.compiler import _internal
    except ImportError:
        return False

    pattern = os.path.join(os.path.dirname(_internal.__file__), "_mlir_libs", "_nki*.so")
    for lib in glob.glob(pattern):
        try:
            with open(lib, "rb") as handle:
                if b"nisa-allocation-mode" in handle.read():
                    return True
        except OSError:
            continue
    return False


def _beta3_pipeline_options(compile_opts, compile_mode, allocation_mode):
    """Apply the beta 3 compile-mode pipeline options to ``compile_opts``."""
    if compile_mode not in BETA3_COMPILE_MODES:
        raise ValueError(
            f"nki_compile_mode={compile_mode!r} is invalid; expected one of: "
            f"{', '.join(BETA3_COMPILE_MODES)}"
        )
    if allocation_mode is not None and allocation_mode not in BETA3_ALLOCATION_MODES:
        raise ValueError(
            f"nisa_allocation_mode={allocation_mode!r} is invalid; expected one "
            f"of: {', '.join(BETA3_ALLOCATION_MODES)}"
        )
    if allocation_mode is not None and not _beta3_supports_allocation_mode():
        raise ValueError(
            "nisa_allocation_mode is not supported by the installed nki: it has "
            "no 'nisa-allocation-mode' pipeline option. Leave it as None to use "
            "NKI's built-in default."
        )

    if compile_mode == "standard":
        if allocation_mode is not None:
            raise ValueError(
                "nisa_allocation_mode is only meaningful with "
                'nki_compile_mode="integration-alloc"; under "standard" the '
                "neuronx-cc backend owns allocation."
            )
        # Turns off NKI's LinearScanAllocation + InstructionScheduling and sets
        # emit_reg_compute_as_affine_expr, matching the production integrations.
        return compile_opts.disable_backend_optimizations()

    options = [
        "enable-linear-scan-allocation=true",
        "enable-instruction-scheduling=false",
    ]
    if allocation_mode is not None:
        options.append(f"nisa-allocation-mode={allocation_mode}")
    return compile_opts.set_pipeline_options(*options)


if BETA3_NKI_AVAILABLE:

    @dataclasses.dataclass(frozen=True)
    class _NKIPyCompileKernel(Beta3CompileKernel):
        """``CompileKernel`` subclass carrying NKIPy's compilation policy.

        Using ``CompileKernel`` (rather than calling ``compile_kernel_to_nir``
        with a hand-built ``CompileOptions``) is what nki.framework expects of a
        framework integration: it is the layer that resolves the compile-mode
        pipeline options, so a bare ``CompileOptions`` silently inherits NKI's
        experimental defaults instead of the production ones.

        NKIPy deliberately keeps compile caching and artifact lifetime as the
        caller's concern, so NKI's caches are disabled here: ``__post_init__``
        skips creating the in-memory compile cache, and ``_make_trace_cache``
        suppresses the persistent trace cache on nki versions that have one.
        """

        nki_compile_mode: str = "standard"
        """Allocation/scheduling split; see ``BETA3_COMPILE_MODES``."""

        nisa_allocation_mode: Optional[str] = None
        """NKI address-assignment strategy; ``"integration-alloc"`` only."""

        def __post_init__(self):
            # Skip CompileKernel.__post_init__, which installs an in-memory
            # compile cache on the wrapped function. Cache management is left to
            # the caller, so go straight to Kernel.__post_init__.
            Beta3Kernel.__post_init__(self)

        def _make_trace_cache(self, inputs, compile_opts):
            """Disable NKI's cross-process trace cache (see class docstring).

            Only present on newer nki; harmless to define unconditionally.
            """
            return None

        def _compile_opts(self):
            # CompileKernel._compile_opts() applies
            # disable_backend_optimizations() whenever _enable_backend_opt is
            # False, which is right for "standard" but would contradict
            # "integration-alloc". Let it build the base options with that step
            # suppressed, then apply exactly one mode's options.
            base = dataclasses.replace(self, _enable_backend_opt=True)
            return _beta3_pipeline_options(
                Beta3CompileKernel._compile_opts(base),
                self.nki_compile_mode,
                self.nisa_allocation_mode,
            )


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


def _beta3_compile_and_get_config(
    kernel,
    numpy_inputs,
    platform_target=None,
    lnc=None,
    nki_compile_mode="standard",
    nisa_allocation_mode=None,
):
    """Compile a beta 3 kernel and return (framework_config, is_tuple_return).

    The artifacts directory is managed by the caller via _get_beta3_artifacts_dir().
    """
    if platform_target is None:
        platform_target = _get_platform_target_default()

    # Carry the user's @nki.jit settings (lnc, schedule, address_rotation, ...)
    # over to the CompileKernel rather than re-deriving them, so kernel options
    # keep working and new Kernel fields are picked up automatically.
    kernel_fields = {
        field.name: getattr(kernel, field.name)
        for field in dataclasses.fields(kernel)
        if field.name != "func"
    }
    if lnc is not None:
        kernel_fields["lnc"] = lnc
    kernel_fields.setdefault("lnc", 1)

    compile_kernel = _NKIPyCompileKernel(
        getattr(kernel, "func", kernel),
        **kernel_fields,
        target=platform_target,
        artifacts_dir=_get_beta3_artifacts_dir(),
        nki_compile_mode=nki_compile_mode,
        nisa_allocation_mode=nisa_allocation_mode,
    )

    # CompileKernel.compile() returns only (config, cache_hash); go through
    # _cached_compile_to_bir (as nki.framework's JAX integration does) to keep
    # the NirResult, which carries is_tuple_return. Inputs are already numpy
    # arrays, which NKI's frontends consume directly -- no per-framework tensor
    # conversion step is needed.
    nir = compile_kernel._cached_compile_to_bir(
        frontend=compile_kernel._frontend_cls(
            enable_backend_opt=compile_kernel._enable_backend_opt
        ),
        inputs=numpy_inputs,
        compile_opts=compile_kernel._compile_opts(),
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
        nki_compile_mode: str = "standard",
        nisa_allocation_mode: Optional[str] = None,
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
            self._compile_beta3(
                kernel,
                operands,
                platform_target,
                nki_compile_mode,
                nisa_allocation_mode,
            )
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

    def _compile_beta3(
        self,
        kernel,
        operands,
        platform_target,
        nki_compile_mode="standard",
        nisa_allocation_mode=None,
    ):
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
            _beta3_compile_and_get_config(
                kernel,
                numpy_inputs,
                platform_target,
                nki_compile_mode=nki_compile_mode,
                nisa_allocation_mode=nisa_allocation_mode,
            )
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
    nki_compile_mode: str = "standard",
    nisa_allocation_mode: Optional[str] = None,
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
                               CompileKernel). Note: does not support CPU execution.
        platform_target: Target platform (e.g., "trn1", "trn2"). If None, auto-detected.
        nki_compile_mode: Beta 3 only. ``"standard"`` (default) leaves memory
            allocation and instruction scheduling to the neuronx-cc backend.
            ``"integration-alloc"`` has NKI allocate instead; it is experimental.
        nisa_allocation_mode: Beta 3 ``"integration-alloc"`` only. NKI's
            address-assignment strategy: ``"fast"``, ``"ring"``, or
            ``"max-reuse"``. ``None`` (default) uses NKI's own default
            (``"fast"``). Raises ValueError if the installed nki has no
            ``nisa-allocation-mode`` pipeline option.

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
        nki_compile_mode=nki_compile_mode,
        nisa_allocation_mode=nisa_allocation_mode,
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
