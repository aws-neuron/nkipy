"""Shared DeviceKernel compile helpers.

These helpers are intentionally small: they serialize cold-cache compiles for
kernel entrypoints that are identical across local worker ranks.  The caller
still owns the kernel cache and runtime invocation.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nkipy_serving.runtime.collective_load import rank_shared_build_dir

logger = logging.getLogger(__name__)

_DEFAULT_BUILD_DIR = Path("/tmp/nkipy_serving_kernel_compile")
_GLOBAL_NEFF_CACHE_DIR_ENV = "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR"
_DEFAULT_GLOBAL_NEFF_CACHE_DIR = Path("/tmp/nkipy_serving_neff_catalog")
_RANK_DIR_RE = re.compile(r"rank_?\d+")
_SEALED_NEFF_NAMESPACES: dict[str, str] = {}


def shared_kernel_build_dir(
    build_dir: str | Path | None,
    *,
    namespace: str,
) -> str | None:
    """Map a per-rank build directory to a run-shared kernel cache directory."""

    return rank_shared_build_dir(build_dir, namespace=namespace)


def seal_kernel_compile_namespace(
    namespace: str,
    *,
    reason: str = "warmup complete",
) -> None:
    """Reject future cold compiles for a shared DeviceKernel namespace."""

    _SEALED_NEFF_NAMESPACES[str(namespace)] = str(reason)


def unseal_kernel_compile_namespace(namespace: str) -> None:
    """Allow cold compiles for a shared DeviceKernel namespace again."""

    _SEALED_NEFF_NAMESPACES.pop(str(namespace), None)


def is_kernel_compile_namespace_sealed(namespace: str) -> bool:
    return str(namespace) in _SEALED_NEFF_NAMESPACES


def _require_unsealed_kernel_compile_namespace(
    *,
    namespace: str,
    name: str,
    cache_key: str,
) -> None:
    reason = _SEALED_NEFF_NAMESPACES.get(str(namespace))
    if reason is None:
        return
    raise RuntimeError(
        "DeviceKernel late compile blocked after namespace seal: "
        f"namespace={namespace} name={name} cache_key={cache_key} reason={reason}"
    )


@contextmanager
def kernel_compile_lock(
    *,
    build_dir: str | Path | None,
    name: str,
) -> Any:
    """Serialize same-kernel cold-cache compiles across local processes."""

    root = Path(str(build_dir)) if build_dir is not None else _DEFAULT_BUILD_DIR
    lock_dir = root / ".kernel_compile_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe, digest = _safe_name_digest(str(name))
    with (lock_dir / f"{safe}_{digest}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def deterministic_nki_artifacts_dir(
    *,
    build_dir: str | Path | None,
    name: str,
) -> Any:
    """Provide a stable beta2-NKI artifact root while tracing a kernel."""

    if build_dir is None:
        yield
        return

    safe, digest = _safe_name_digest(str(name))
    root = Path(str(build_dir)) / ".nki_bir" / f"{safe}_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    key = "NKIPY_SERVING_NKI_BIR_ARTIFACTS_DIR"
    old = os.environ.get(key)
    os.environ[key] = str(root)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _safe_name_digest(name: str) -> tuple[str, str]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:180] or "kernel"
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:12]
    return safe, digest


def _is_stale_compile_artifact_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "Compilation artifacts already exist" in msg
        or "json.exception.parse_error" in msg
        or "Expecting value: line 1 column 1" in msg
        or "No such file or directory: 'mempressure.txt'" in msg
        or "PE0.json" in msg
    )


def _signature_value(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return (
            type(value).__name__,
            tuple(int(dim) for dim in tuple(shape or ())),
            str(dtype),
        )
    if isinstance(value, dict):
        return tuple(
            (str(key), _signature_value(val))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_signature_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def kernel_signature_cache_key(
    fn: Any,
    *,
    name: str,
    sample_args: tuple[Any, ...],
    kwargs: dict[str, Any],
    additional_compiler_args: str | None,
    target: Any,
) -> str:
    signature = (
        getattr(fn, "__module__", ""),
        getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
        str(name),
        tuple(_signature_value(arg) for arg in sample_args),
        _signature_value(kwargs),
        str(additional_compiler_args or ""),
        repr(target),
    )
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()


def canonical_neff_record_path(
    *,
    build_dir: str | Path | None,
    name: str,
    cache_key: str,
) -> Path:
    root = Path(str(build_dir)) if build_dir is not None else _DEFAULT_BUILD_DIR
    safe, name_digest = _safe_name_digest(str(name))
    key_digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()[:16]
    return root / ".canonical_neffs" / f"{safe}_{name_digest}_{key_digest}.path"


def _neff_record_path_value(record_path: Path) -> str | None:
    try:
        neff_path = record_path.read_text().strip()
    except FileNotFoundError:
        return None
    if neff_path and _has_rank_dir_component(neff_path):
        try:
            record_path.unlink()
        except OSError:
            pass
        return None
    if neff_path and Path(neff_path).exists():
        return neff_path
    return None


def _write_neff_record_path(record_path: Path, neff_path: str | Path) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = record_path.with_suffix(f"{record_path.suffix}.tmp.{os.getpid()}")
    tmp_path.write_text(f"{str(neff_path)}\n")
    tmp_path.replace(record_path)


def _has_rank_dir_component(path: str | Path) -> bool:
    return any(_RANK_DIR_RE.fullmatch(part) is not None for part in Path(path).parts)


def global_neff_record_path(
    *,
    namespace: str,
    name: str,
    cache_key: str,
) -> Path:
    root = Path(
        os.getenv(
            _GLOBAL_NEFF_CACHE_DIR_ENV,
            str(_DEFAULT_GLOBAL_NEFF_CACHE_DIR),
        )
    )
    safe, name_digest = _safe_name_digest(str(name))
    key_digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()[:16]
    return (
        root
        / re.sub(r"[^A-Za-z0-9_.-]+", "_", str(namespace))
        / f"{safe}_{name_digest}_{key_digest}.path"
    )


def read_global_neff_path(
    *,
    namespace: str,
    name: str,
    cache_key: str,
) -> str | None:
    return _neff_record_path_value(
        global_neff_record_path(
            namespace=namespace,
            name=name,
            cache_key=cache_key,
        )
    )


def write_global_neff_path(
    *,
    namespace: str,
    name: str,
    cache_key: str,
    neff_path: str | Path,
) -> None:
    _write_neff_record_path(
        global_neff_record_path(
            namespace=namespace,
            name=name,
            cache_key=cache_key,
        ),
        neff_path,
    )


def read_canonical_neff_path(
    *,
    build_dir: str | Path | None,
    name: str,
    cache_key: str,
) -> str | None:
    return _neff_record_path_value(
        canonical_neff_record_path(
            build_dir=build_dir,
            name=name,
            cache_key=cache_key,
        )
    )


def write_canonical_neff_path(
    *,
    build_dir: str | Path | None,
    name: str,
    cache_key: str,
    neff_path: str | Path,
) -> None:
    record_path = canonical_neff_record_path(
        build_dir=build_dir,
        name=name,
        cache_key=cache_key,
    )
    _write_neff_record_path(record_path, neff_path)


def read_cached_neff_path_with_source(
    *,
    build_dir: str | Path | None,
    namespace: str,
    name: str,
    cache_key: str,
) -> tuple[str | None, str]:
    neff_path = read_canonical_neff_path(
        build_dir=build_dir,
        name=name,
        cache_key=cache_key,
    )
    if neff_path is not None:
        write_global_neff_path(
            namespace=namespace,
            name=name,
            cache_key=cache_key,
            neff_path=neff_path,
        )
        return neff_path, "local"
    neff_path = read_global_neff_path(
        namespace=namespace,
        name=name,
        cache_key=cache_key,
    )
    if neff_path is not None:
        write_canonical_neff_path(
            build_dir=build_dir,
            name=name,
            cache_key=cache_key,
            neff_path=neff_path,
        )
        return neff_path, "global"
    return None, "miss"


def compile_and_load_with_lock(
    device_kernel_cls: Any,
    fn: Any,
    *sample_args: Any,
    name: str,
    build_dir: str | Path | None,
    namespace: str,
    lock_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Compile/load a DeviceKernel from a shared, file-locked build dir."""

    use_cached_if_exists = bool(kwargs.pop("use_cached_if_exists", True))
    cc_enabled = kwargs.pop("cc_enabled", None)
    rank_id = kwargs.pop("rank_id", None)
    world_size = kwargs.pop("world_size", None)
    is_spmd = kwargs.pop("is_spmd", None)
    collective_load_requested = any(
        value is not None for value in (cc_enabled, rank_id, world_size, is_spmd)
    )
    has_direct_neff_api = hasattr(device_kernel_cls, "_trace_and_compile") and hasattr(
        device_kernel_cls,
        "load_from_neff",
    )
    if not has_direct_neff_api:
        raise RuntimeError(
            "DeviceKernel must expose _trace_and_compile/load_from_neff "
            "so kernels are loaded from precompiled NEFF artifacts"
        )
    if not collective_load_requested:
        return compile_and_load_neff_with_lock(
            device_kernel_cls,
            fn,
            *sample_args,
            name=name,
            build_dir=build_dir,
            namespace=namespace,
            lock_name=lock_name,
            use_cached_if_exists=use_cached_if_exists,
            **kwargs,
        )

    if bool(cc_enabled) and (rank_id is None or world_size is None):
        raise ValueError("rank_id and world_size are required when cc_enabled=True")
    neff_path = compile_neff_path_with_lock(
        device_kernel_cls,
        fn,
        *sample_args,
        name=name,
        build_dir=build_dir,
        namespace=namespace,
        lock_name=lock_name,
        use_cached_if_exists=use_cached_if_exists,
        **kwargs,
    )
    load_kwargs: dict[str, Any] = {"name": name}
    if bool(cc_enabled):
        load_kwargs.update(
            cc_enabled=True,
            rank_id=int(rank_id),
            world_size=int(world_size),
        )
    return device_kernel_cls.load_from_neff(neff_path, **load_kwargs)


def compile_and_load_neff_with_lock(
    device_kernel_cls: Any,
    fn: Any,
    *sample_args: Any,
    name: str,
    build_dir: str | Path | None,
    namespace: str,
    lock_name: str | None = None,
    additional_compiler_args: str | None = None,
    use_cached_if_exists: bool = True,
    **kwargs: Any,
) -> Any:
    """Compile once in a shared dir, then load the resolved NEFF path.

    ``DeviceKernel.compile_and_load`` can still leave duplicate NEFF artifacts
    when many worker processes use a shared build directory. This helper keeps
    the content-hash trace under the file lock and loads the returned NEFF
    directly, so identical ranks reuse the same artifact path.
    """

    neff_path = compile_neff_path_with_lock(
        device_kernel_cls,
        fn,
        *sample_args,
        name=name,
        build_dir=build_dir,
        namespace=namespace,
        lock_name=lock_name,
        additional_compiler_args=additional_compiler_args,
        use_cached_if_exists=use_cached_if_exists,
        **kwargs,
    )
    return device_kernel_cls.load_from_neff(neff_path, name=name)


def compile_neff_path_with_lock(
    device_kernel_cls: Any,
    fn: Any,
    *sample_args: Any,
    name: str,
    build_dir: str | Path | None,
    namespace: str,
    lock_name: str | None = None,
    additional_compiler_args: str | None = None,
    use_cached_if_exists: bool = True,
    **kwargs: Any,
) -> str:
    """Compile a DeviceKernel to a NEFF path using local/global record caches."""

    shared_build_dir = shared_kernel_build_dir(build_dir, namespace=namespace)
    target = kwargs.pop("target", None)
    if target is None:
        from nkipy.core.compile import CompilationTarget

        target = CompilationTarget.DEFAULT
    cache_key = kernel_signature_cache_key(
        fn,
        name=name,
        sample_args=sample_args,
        kwargs=kwargs,
        additional_compiler_args=additional_compiler_args,
        target=target,
    )
    neff_path = None
    if use_cached_if_exists:
        neff_path, _ = read_cached_neff_path_with_source(
            build_dir=shared_build_dir,
            namespace=namespace,
            name=name,
            cache_key=cache_key,
        )
    if neff_path is None:
        with kernel_compile_lock(
            build_dir=shared_build_dir,
            name=lock_name or name,
        ):
            neff_path = (
                read_cached_neff_path_with_source(
                    build_dir=shared_build_dir,
                    namespace=namespace,
                    name=name,
                    cache_key=cache_key,
                )[0]
                if use_cached_if_exists
                else None
            )
            if neff_path is None:
                _require_unsealed_kernel_compile_namespace(
                    namespace=namespace,
                    name=name,
                    cache_key=cache_key,
                )
                try:
                    with deterministic_nki_artifacts_dir(
                        build_dir=shared_build_dir,
                        name=name,
                    ):
                        neff_path, _ = device_kernel_cls._trace_and_compile(
                            fn,
                            name,
                            sample_args,
                            kwargs,
                            additional_compiler_args=additional_compiler_args,
                            use_cached_if_exists=use_cached_if_exists,
                            build_dir=shared_build_dir,
                            target=target,
                        )
                except RuntimeError as exc:
                    if not (
                        use_cached_if_exists and _is_stale_compile_artifact_error(exc)
                    ):
                        raise
                    logger.warning(
                        "Kernel %s: retrying trace after stale artifact error",
                        name,
                    )
                    with deterministic_nki_artifacts_dir(
                        build_dir=shared_build_dir,
                        name=name,
                    ):
                        neff_path, _ = device_kernel_cls._trace_and_compile(
                            fn,
                            name,
                            sample_args,
                            kwargs,
                            additional_compiler_args=additional_compiler_args,
                            use_cached_if_exists=False,
                            build_dir=shared_build_dir,
                            target=target,
                        )
            if use_cached_if_exists:
                write_canonical_neff_path(
                    build_dir=shared_build_dir,
                    name=name,
                    cache_key=cache_key,
                    neff_path=str(neff_path),
                )
                write_global_neff_path(
                    namespace=namespace,
                    name=name,
                    cache_key=cache_key,
                    neff_path=str(neff_path),
                )
    return str(neff_path)
