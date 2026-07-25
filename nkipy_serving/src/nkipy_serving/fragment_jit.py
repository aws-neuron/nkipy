"""Fragment JIT for composable eager execution.

``@jit`` compiles and runs on NeuronCore by default.
Pass ``device=False`` for CPU-only numpy execution.

Example::

    @jit(name="embed", build_dir="build/fragments")
    def embed(input_ids, embeddings):
        return embeddings[input_ids]

    # swap to CPU for debug:
    attn_cpu = jit(attn_fn, device=False)
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ALIAS_SUFFIX = ".must_alias_input"
_OUTPUT_TENSORS_KWARG = "_nkipy_output_tensors"


# ---------------------------------------------------------------------------
# Specialization key helpers
# ---------------------------------------------------------------------------


def _arg_sig(arg) -> tuple:
    """Shape+dtype signature for specialization key."""
    if isinstance(arg, np.ndarray):
        return ("np", tuple(arg.shape), str(arg.dtype))
    if hasattr(arg, "shape") and hasattr(arg, "dtype"):
        return ("dev", tuple(arg.shape), str(arg.dtype))
    return ("static", type(arg).__name__, arg)


def _fn_fingerprint(func, version: Optional[str]) -> str:
    if version is not None:
        return f"v:{version}"
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        src = None
    if src is not None:
        return "src:" + hashlib.sha256(src.encode()).hexdigest()[:12]
    if hasattr(func, "__code__"):
        return "bc:" + hashlib.sha256(func.__code__.co_code).hexdigest()[:12]
    return "unknown"


# ---------------------------------------------------------------------------
# Fragment (internal — use device_jit / cpu_jit decorators)
# ---------------------------------------------------------------------------


class Fragment:
    """Composable kernel fragment. Use ``@device_jit`` or ``@cpu_jit``."""

    def __init__(
        self,
        func,
        *,
        device: bool = True,
        name: Optional[str] = None,
        build_dir: Optional[str] = None,
        additional_compiler_args: Optional[str] = None,
        version: Optional[str] = None,
        cc_enabled: Optional[bool] = None,
        rank_id: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self._func = func
        self._device = device
        self._name = name or func.__name__
        self._build_dir = build_dir
        self._additional_compiler_args = additional_compiler_args
        self._version = version
        self._cc_enabled = cc_enabled
        self._rank_id = rank_id
        self._world_size = world_size
        self._fingerprint = _fn_fingerprint(func, version)

        self._cache: dict[tuple, _CachedKernel] = {}
        self._hits = 0
        self._misses = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> bool:
        return self._device

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "entries": len(self._cache)}

    def clear_cache(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # -- CPU path ---------------------------------------------------------------

    def _call_cpu(self, args: tuple, kwargs: dict) -> Any:
        kwargs = dict(kwargs)
        kwargs.pop(_OUTPUT_TENSORS_KWARG, None)
        # Auto-download device tensors to numpy at the device→CPU boundary
        args = tuple(
            a.numpy() if hasattr(a, "tensor_ref") and hasattr(a, "numpy") else a
            for a in args
        )
        kwargs = {
            k: v.numpy() if hasattr(v, "tensor_ref") and hasattr(v, "numpy") else v
            for k, v in kwargs.items()
        }
        return self._func(*args, **kwargs)

    # -- Device path ------------------------------------------------------------

    def _spec_key(self, args: tuple, kwargs: dict) -> tuple:
        arg_sig = tuple(_arg_sig(a) for a in args)
        kwarg_sig = tuple((k, _arg_sig(v)) for k, v in sorted(kwargs.items()))
        return (
            self._fingerprint,
            self._version,
            arg_sig,
            kwarg_sig,
            self._additional_compiler_args or "",
            self._cc_enabled,
            self._rank_id,
            self._world_size,
            self._build_dir or "",
        )

    def _compile_lock_key(self, args: tuple, kwargs: dict) -> tuple:
        """Process-shared compile lock key, excluding rank-local cache roots."""

        arg_sig = tuple(_arg_sig(a) for a in args)
        kwarg_sig = tuple((k, _arg_sig(v)) for k, v in sorted(kwargs.items()))
        return (
            self._fingerprint,
            self._version,
            arg_sig,
            kwarg_sig,
            self._additional_compiler_args or "",
            self._cc_enabled,
        )

    def _get_or_compile(self, args: tuple, kwargs: dict) -> "_CachedKernel":
        key = self._spec_key(args, kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        logger.info("Fragment %s: compiling (miss #%d)", self._name, self._misses)
        compile_lock_key = self._compile_lock_key(args, kwargs)
        lock_name = (
            f"{self._name}_"
            f"{hashlib.sha1(repr(compile_lock_key).encode('utf-8')).hexdigest()[:12]}"
        )

        from nkipy.core.compile import CompilationTarget
        from nkipy.runtime.device_kernel import DeviceKernel
        from nkipy.runtime.device_tensor import DeviceTensor
        from spike import SpikeTensor

        def _norm(a):
            if isinstance(a, DeviceTensor):
                return a
            if isinstance(a, SpikeTensor):
                return DeviceTensor(
                    tensor_ref=a.tensor_ref,
                    shape=a.shape,
                    dtype=a.dtype,
                    name=a.name,
                )
            return a

        norm_args = tuple(_norm(a) for a in args)
        norm_kwargs = {k: _norm(v) for k, v in kwargs.items()}

        build_dir = None
        if self._build_dir:
            import os

            build_dir = os.path.join(self._build_dir, self._name)

        if self._cc_enabled:
            if self._rank_id is None or self._world_size is None:
                raise RuntimeError(
                    "Fragment collective kernels require rank_id/world_size"
                )
            from nkipy_serving.runtime.kernel_compile import (
                compile_neff_path_with_lock,
            )

            neff_path = compile_neff_path_with_lock(
                DeviceKernel,
                self._func,
                *norm_args,
                name=self._name,
                build_dir=build_dir,
                namespace="fragment_jit_collective",
                lock_name=lock_name,
                additional_compiler_args=self._additional_compiler_args,
                target=CompilationTarget.DEFAULT,
                **norm_kwargs,
            )
            from nkipy_serving.runtime.collective_load import (
                collective_load_barrier,
                rank_shared_build_dir,
            )

            collective_load_barrier(
                build_dir=rank_shared_build_dir(
                    self._build_dir,
                    namespace="fragment_collectives",
                ),
                name=self._name,
                rank_id=int(self._rank_id),
                world_size=int(self._world_size),
            )
            dk = DeviceKernel.load_from_neff(
                neff_path,
                name=self._name,
                cc_enabled=True,
                rank_id=int(self._rank_id),
                world_size=int(self._world_size),
            )
        else:
            from nkipy_serving.runtime.kernel_compile import (
                compile_and_load_with_lock,
            )

            dk = compile_and_load_with_lock(
                DeviceKernel,
                self._func,
                *norm_args,
                name=self._name,
                build_dir=build_dir,
                namespace="fragment_jit",
                lock_name=lock_name,
                additional_compiler_args=self._additional_compiler_args,
                target=CompilationTarget.DEFAULT,
                **norm_kwargs,
            )

        alias_output_names = set()
        for out_name in dk.output_tensors_info:
            input_alias = out_name + _ALIAS_SUFFIX
            if input_alias in dk.input_tensors_info:
                alias_output_names.add(out_name)

        ck = _CachedKernel(dk, alias_output_names)
        self._cache[key] = ck
        return ck

    def _call_device(self, args: tuple, kwargs: dict) -> Any:
        from nkipy.runtime.device_tensor import DeviceTensor
        from spike import SpikeTensor

        kwargs = dict(kwargs)
        requested_outputs = kwargs.pop(_OUTPUT_TENSORS_KWARG, None)
        if requested_outputs is None:
            provided_outputs: dict[str, Any] = {}
        elif isinstance(requested_outputs, dict):
            provided_outputs = dict(requested_outputs)
        else:
            raise TypeError(
                f"{_OUTPUT_TENSORS_KWARG} must be a dict of output name to tensor"
            )

        ck = self._get_or_compile(args, kwargs)
        dk = ck.device_kernel

        sig = inspect.signature(self._func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        inputs: dict[str, Any] = {}
        for param_name, value in bound.arguments.items():
            if isinstance(value, DeviceTensor):
                inputs[param_name] = value
            elif isinstance(value, SpikeTensor):
                inputs[param_name] = value
            elif isinstance(value, np.ndarray):
                inputs[param_name] = DeviceTensor.from_numpy(value, name=param_name)

        outputs = {}
        missing_outputs: list[str] = []
        for out_name in dk.output_tensors_info:
            if out_name in provided_outputs:
                continue
            if out_name in ck.alias_output_names:
                if out_name in inputs or out_name + _ALIAS_SUFFIX in inputs:
                    continue
            missing_outputs.append(out_name)
        auto_by_name = (
            {t.name: t for t in dk.allocate_output_tensors()} if missing_outputs else {}
        )

        for out_name in dk.output_tensors_info:
            if out_name in provided_outputs:
                outputs[out_name] = provided_outputs[out_name]
            elif out_name in ck.alias_output_names:
                if out_name in inputs:
                    outputs[out_name] = inputs[out_name]
                else:
                    alias_input = out_name + _ALIAS_SUFFIX
                    if alias_input in inputs:
                        outputs[out_name] = inputs[alias_input]
                    else:
                        outputs[out_name] = auto_by_name[out_name]
            else:
                outputs[out_name] = auto_by_name[out_name]

        dk(inputs=inputs, outputs=outputs)

        user_outputs = [
            v for k, v in sorted(outputs.items()) if k not in ck.alias_output_names
        ]
        if len(user_outputs) == 1:
            return user_outputs[0]
        return tuple(user_outputs)

    # -- Public interface -------------------------------------------------------

    def __call__(self, *args, **kwargs) -> Any:
        if self._device:
            return self._call_device(args, kwargs)
        return self._call_cpu(args, kwargs)

    def __repr__(self) -> str:
        mode = "device" if self._device else "cpu"
        stats = self.cache_stats
        return (
            f"Fragment({self._name!r}, {mode}, "
            f"entries={stats['entries']}, "
            f"hits={stats['hits']}, misses={stats['misses']})"
        )


class _CachedKernel:
    __slots__ = ("device_kernel", "alias_output_names")

    def __init__(self, device_kernel, alias_output_names: set[str]):
        self.device_kernel = device_kernel
        self.alias_output_names = alias_output_names


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def jit(
    func=None,
    *,
    device: bool = True,
    name: Optional[str] = None,
    build_dir: Optional[str] = None,
    additional_compiler_args: Optional[str] = None,
    version: Optional[str] = None,
    cc_enabled: Optional[bool] = None,
    rank_id: Optional[int] = None,
    world_size: Optional[int] = None,
):
    """Decorator: wrap a function as a composable Fragment.

    ``device=True`` (default): compile and run on NeuronCore.
    ``device=False``: numpy in → numpy out, no compilation.

    Usage::

        @jit
        def kernel(x, w): ...

        @jit(name="my_kernel", build_dir="build/fragments")
        def kernel(x, w): ...

        cpu_kernel = jit(kernel_fn, device=False)
    """

    def wrap(fn):
        return Fragment(
            fn,
            device=device,
            name=name or fn.__name__,
            build_dir=build_dir,
            additional_compiler_args=additional_compiler_args,
            version=version,
            cc_enabled=cc_enabled,
            rank_id=rank_id,
            world_size=world_size,
        )

    if func is not None:
        return wrap(func)
    return wrap
