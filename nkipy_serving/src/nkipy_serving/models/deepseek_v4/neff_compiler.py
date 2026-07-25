"""Compile/cache helpers for DSV4 product kernels."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkipy_serving.models._device_utils import _get_device_kernel_cls
from nkipy_serving.models.deepseek_v4.diagnostics import (
    rank_trace_allowed,
    stage_profile_enabled,
)
from nkipy_serving.profiling import ProfileWriter
from nkipy_serving.runtime.collective_load import collective_load_barrier
from nkipy_serving.runtime.kernel_compile import (
    kernel_compile_lock,
    kernel_signature_cache_key,
    read_canonical_neff_path,
    write_canonical_neff_path,
)

logger = logging.getLogger(__name__)
_PRODUCT_COMPILE_PROFILE_WRITERS: dict[str, ProfileWriter] = {}
_PRODUCT_RUNTIME_PROFILE_WRITERS: dict[str, ProfileWriter] = {}
_PRODUCT_KERNEL_OBJECT_NAMES: dict[int, str] = {}
_PRODUCT_LOADED_KERNEL_NAMES: dict[int, str] = {}
_PRODUCT_RUNTIME_LAYER_PROFILE_FIELDS: ContextVar[dict[str, Any] | None] = ContextVar(
    "dsv4_product_runtime_layer_profile_fields", default=None
)


@contextmanager
def product_runtime_layer_profile(fields: dict[str, Any] | None):
    token = _PRODUCT_RUNTIME_LAYER_PROFILE_FIELDS.set(dict(fields) if fields else None)
    try:
        yield
    finally:
        _PRODUCT_RUNTIME_LAYER_PROFILE_FIELDS.reset(token)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _product_rank_from_build_dir(build_dir: str | None) -> int | None:
    if build_dir is None:
        return None
    for part in Path(str(build_dir)).parts:
        match = re.fullmatch(r"rank_(\d+)", part)
        if match is not None:
            return int(match.group(1))
    return None


def _product_load_trace_path() -> Path | None:
    path = os.getenv("NKIPY_SERVING_DSV4_PRODUCT_LOAD_TRACE_PATH", "").strip()
    if path:
        return Path(path)
    if _env_flag("NKIPY_SERVING_DSV4_PRODUCT_LOAD_TRACE"):
        return Path("/tmp/dsv4_product_load_trace.jsonl")
    return None


def _product_load_trace_enabled() -> bool:
    return _product_load_trace_path() is not None


def _product_kernel_family(name: str) -> str:
    family = str(name)
    if family.startswith("dsv4_product_"):
        family = family[len("dsv4_product_") :]
    family = re.split(r"_t\d+(?:_|$)", family, maxsplit=1)[0]
    return family or str(name)


def _loaded_product_kernel_family_counts() -> list[dict[str, Any]]:
    counts = Counter(
        _product_kernel_family(name) for name in _PRODUCT_LOADED_KERNEL_NAMES.values()
    )
    return [
        {"family": family, "count": int(count)}
        for family, count in sorted(
            counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )
    ]


def _write_product_load_trace(
    *,
    event: str,
    name: str,
    build_dir: str | None,
    rank: int | None,
    neff_path: str | None,
    cc_enabled: bool,
    error: str = "",
) -> None:
    path = _product_load_trace_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "pid": os.getpid(),
            "event": str(event),
            "name": str(name),
            "family": _product_kernel_family(str(name)),
            "rank": int(rank) if rank is not None else None,
            "build_dir": str(build_dir) if build_dir is not None else None,
            "neff_path": str(neff_path) if neff_path is not None else None,
            "cc_enabled": bool(cc_enabled),
            "loaded_total": len(_PRODUCT_LOADED_KERNEL_NAMES),
            "loaded_families": _loaded_product_kernel_family_counts(),
            **({"error": str(error)} if error else {}),
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        logger.debug("failed to write DSV4 product load trace", exc_info=True)


def _record_product_kernel_loaded(
    *,
    key: int,
    name: str,
    build_dir: str | None,
    rank: int | None,
    neff_path: str | None,
    cc_enabled: bool,
    event: str = "load_done",
) -> None:
    if not _product_load_trace_enabled():
        return
    _PRODUCT_LOADED_KERNEL_NAMES[int(key)] = str(name)
    _write_product_load_trace(
        event=event,
        name=str(name),
        build_dir=build_dir,
        rank=rank,
        neff_path=neff_path,
        cc_enabled=cc_enabled,
    )


def _record_product_kernel_unloaded(
    *,
    key: int,
    name: str,
    build_dir: str | None,
    rank: int | None,
    neff_path: str | None,
    cc_enabled: bool,
) -> None:
    if not _product_load_trace_enabled():
        return
    _PRODUCT_LOADED_KERNEL_NAMES.pop(int(key), None)
    _write_product_load_trace(
        event="unload_done",
        name=str(name),
        build_dir=build_dir,
        rank=rank,
        neff_path=neff_path,
        cc_enabled=cc_enabled,
    )


def _product_compile_profile_writer(rank: int | None) -> ProfileWriter | None:
    if not stage_profile_enabled():
        return None
    if rank is not None and not rank_trace_allowed(int(rank)):
        return None
    label = str(int(rank)) if rank is not None else "unknown"
    writer = _PRODUCT_COMPILE_PROFILE_WRITERS.get(label)
    if writer is None:
        writer = ProfileWriter(f"dsv4_product_compile_rank_{label}", flush_every=1)
        _PRODUCT_COMPILE_PROFILE_WRITERS[label] = writer
    return writer


def _product_runtime_profile_writer(rank: int | None) -> ProfileWriter | None:
    if not stage_profile_enabled():
        return None
    if not _env_flag("NKIPY_SERVING_DSV4_KERNEL_RUNTIME_PROFILE"):
        return None
    if rank is not None and not rank_trace_allowed(int(rank)):
        return None
    label = str(int(rank)) if rank is not None else "unknown"
    writer = _PRODUCT_RUNTIME_PROFILE_WRITERS.get(label)
    if writer is None:
        writer = ProfileWriter(f"dsv4_product_runtime_rank_{label}", flush_every=1)
        _PRODUCT_RUNTIME_PROFILE_WRITERS[label] = writer
    return writer


def _product_kernel_name(kernel: Any) -> str:
    if isinstance(kernel, _ProductPrecompiledKernel):
        return str(kernel.name)
    name = getattr(kernel, "_dsv4_product_name", None)
    if name:
        return str(name)
    name = _PRODUCT_KERNEL_OBJECT_NAMES.get(id(kernel))
    if name:
        return str(name)
    name = getattr(kernel, "__name__", None)
    if name:
        return str(name)
    return type(kernel).__name__


def _tag_product_kernel_name(kernel: Any, name: str) -> Any:
    try:
        setattr(kernel, "_dsv4_product_name", str(name))
    except Exception:
        _PRODUCT_KERNEL_OBJECT_NAMES[id(kernel)] = str(name)
    else:
        _PRODUCT_KERNEL_OBJECT_NAMES[id(kernel)] = str(name)
    return kernel


def _write_product_runtime_profile(
    *,
    writer: ProfileWriter | None,
    kernel: Any,
    build_dir: str | None,
    rank: int | None,
    unload_after_call: bool,
    elapsed_s: float,
    load_s: float,
    call_s: float,
    unload_s: float,
    loaded_before: bool,
    status: str,
    error: str = "",
) -> None:
    if writer is None:
        return
    deferred_load = isinstance(kernel, _ProductPrecompiledKernel)
    layer_fields = _PRODUCT_RUNTIME_LAYER_PROFILE_FIELDS.get()
    writer.write(
        {
            "ts": time.time(),
            "stage": "run_product_kernel",
            "name": _product_kernel_name(kernel),
            "rank": int(rank) if rank is not None else None,
            "build_dir": str(build_dir) if build_dir is not None else None,
            "deferred_load": bool(deferred_load),
            "precompiled_neff": bool(deferred_load),
            "runtime_compile": False,
            "cc_enabled": bool(getattr(kernel, "cc_enabled", False)),
            "unload_after_call": bool(unload_after_call),
            "loaded_before": bool(loaded_before),
            "elapsed_s": round(float(elapsed_s), 6),
            "load_s": round(float(load_s), 6),
            "call_s": round(float(call_s), 6),
            "unload_s": round(float(unload_s), 6),
            "status": str(status),
            **(layer_fields or {}),
            **({"error": str(error)} if error else {}),
        }
    )


def _shape_summary(values: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            summary[name] = [int(dim) for dim in shape]
    return summary


def _write_product_runtime_event(
    *,
    writer: ProfileWriter | None,
    event: str,
    kernel: Any,
    build_dir: str | None,
    rank: int | None,
    unload_after_call: bool,
    loaded_before: bool,
    elapsed_s: float,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> None:
    if writer is None:
        return
    deferred_load = isinstance(kernel, _ProductPrecompiledKernel)
    layer_fields = _PRODUCT_RUNTIME_LAYER_PROFILE_FIELDS.get()
    writer.write(
        {
            "ts": time.time(),
            "stage": "run_product_kernel_phase",
            "event": str(event),
            "name": _product_kernel_name(kernel),
            "rank": int(rank) if rank is not None else None,
            "build_dir": str(build_dir) if build_dir is not None else None,
            "deferred_load": bool(deferred_load),
            "precompiled_neff": bool(deferred_load),
            "cc_enabled": bool(getattr(kernel, "cc_enabled", False)),
            "unload_after_call": bool(unload_after_call),
            "loaded_before": bool(loaded_before),
            "elapsed_s": round(float(elapsed_s), 6),
            **(layer_fields or {}),
            **({"input_shapes": _shape_summary(inputs)} if inputs else {}),
            **({"output_shapes": _shape_summary(outputs)} if outputs else {}),
        }
    )


def _write_product_compile_profile(
    *,
    writer: ProfileWriter | None,
    name: str,
    build_dir: str | None,
    shared_build_dir: str | None,
    rank: int | None,
    cc_enabled: bool,
    load: bool,
    status: str,
    cache_status: str,
    elapsed_s: float,
    lock_wait_s: float = 0.0,
    compile_s: float = 0.0,
    load_s: float = 0.0,
    barrier_s: float = 0.0,
    neff_path: str | None = None,
    canonical_neff_cache_key: str | None = None,
    error: str = "",
) -> None:
    if writer is None:
        return
    writer.write(
        {
            "ts": time.time(),
            "stage": "compile_product_kernel",
            "name": str(name),
            "rank": int(rank) if rank is not None else None,
            "build_dir": str(build_dir) if build_dir is not None else None,
            "shared_build_dir": (
                str(shared_build_dir) if shared_build_dir is not None else None
            ),
            "cc_enabled": bool(cc_enabled),
            "load_requested": bool(load),
            "deferred_load": not bool(load),
            "precompiled_neff": bool(neff_path),
            "cache_status": str(cache_status),
            "elapsed_s": round(float(elapsed_s), 6),
            "lock_wait_s": round(float(lock_wait_s), 6),
            "compile_s": round(float(compile_s), 6),
            "load_s": round(float(load_s), 6),
            "barrier_s": round(float(barrier_s), 6),
            "neff_path": str(neff_path) if neff_path else None,
            "canonical_neff_cache_key": (
                str(canonical_neff_cache_key)
                if canonical_neff_cache_key is not None
                else None
            ),
            "status": str(status),
            **({"error": str(error)} if error else {}),
        }
    )


def _product_shared_build_dir(build_dir: str | None) -> str | None:
    if build_dir is None:
        return None
    path = Path(str(build_dir))
    parts = path.parts
    for idx, part in enumerate(parts):
        if re.fullmatch(r"rank_\d+", part):
            return str(Path(*parts[:idx]) / "product")
    return str(path)


def _collective_load_barrier_metadata_for_groups(
    *,
    rank_id: int,
    world_size: int,
    replica_groups: tuple[tuple[int, ...], ...],
) -> tuple[int, int]:
    """Return subgroup-local rank/world for the file-backed load barrier.

    ``DeviceKernel.load_from_neff`` still receives the original global
    ``rank_id``/``world_size`` metadata. The Python-side file barrier should
    only wait on ranks that will actually load this grouped collective NEFF.
    """

    world_i = int(world_size)
    rank_i = int(rank_id)
    ranks = tuple(sorted({int(rank) for group in replica_groups for rank in group}))
    if not ranks:
        return rank_i, world_i
    if ranks == tuple(range(world_i)):
        return rank_i, world_i
    try:
        return ranks.index(rank_i), len(ranks)
    except ValueError as exc:
        raise RuntimeError(
            "collective load barrier rank is not part of replica group union: "
            f"rank_id={rank_i}, world_size={world_i}, groups={replica_groups}"
        ) from exc


@dataclass
class _ProductPrecompiledKernel:
    neff_path: str
    name: str
    cc_enabled: bool
    rank_id: int | None = None
    world_size: int | None = None
    load_barrier_name: str | None = None
    load_barrier_rank_id: int | None = None
    load_barrier_world_size: int | None = None
    loaded: Any | None = None

    def load(self, *, build_dir: str | None) -> Any:
        rank = (
            int(self.rank_id)
            if self.rank_id is not None
            else _product_rank_from_build_dir(build_dir)
        )
        if self.loaded is not None:
            if _env_flag("NKIPY_SERVING_DSV4_PRODUCT_LOAD_TRACE_REUSE"):
                _write_product_load_trace(
                    event="load_reuse",
                    name=self.name,
                    build_dir=build_dir,
                    rank=rank,
                    neff_path=self.neff_path,
                    cc_enabled=self.cc_enabled,
                )
            return self.loaded
        _write_product_load_trace(
            event="load_start",
            name=self.name,
            build_dir=build_dir,
            rank=rank,
            neff_path=self.neff_path,
            cc_enabled=self.cc_enabled,
        )
        try:
            if self.cc_enabled:
                if self.rank_id is None or self.world_size is None:
                    raise RuntimeError(
                        f"DSV4 product precompiled collective {self.name} is missing "
                        "rank/world metadata"
                    )
                if not self.load_barrier_name:
                    raise RuntimeError(
                        f"DSV4 product precompiled collective {self.name} is missing "
                        "load barrier metadata"
                    )
                barrier_rank_id = (
                    int(self.load_barrier_rank_id)
                    if self.load_barrier_rank_id is not None
                    else int(self.rank_id)
                )
                barrier_world_size = (
                    int(self.load_barrier_world_size)
                    if self.load_barrier_world_size is not None
                    else int(self.world_size)
                )
                collective_load_barrier(
                    build_dir=_product_shared_build_dir(build_dir),
                    name=str(self.load_barrier_name),
                    rank_id=barrier_rank_id,
                    world_size=barrier_world_size,
                )
                self.loaded = _get_device_kernel_cls().load_from_neff(
                    self.neff_path,
                    name=self.name,
                    cc_enabled=True,
                    rank_id=int(self.rank_id),
                    world_size=int(self.world_size),
                )
            else:
                self.loaded = _get_device_kernel_cls().load_from_neff(
                    self.neff_path,
                    name=self.name,
                )
        except Exception as exc:
            _write_product_load_trace(
                event="load_error",
                name=self.name,
                build_dir=build_dir,
                rank=rank,
                neff_path=self.neff_path,
                cc_enabled=self.cc_enabled,
                error=repr(exc),
            )
            raise
        _record_product_kernel_loaded(
            key=id(self),
            name=self.name,
            build_dir=build_dir,
            rank=rank,
            neff_path=self.neff_path,
            cc_enabled=self.cc_enabled,
        )
        return self.loaded

    def unload(self, *, build_dir: str | None = None) -> None:
        if self.loaded is None:
            return
        rank = (
            int(self.rank_id)
            if self.rank_id is not None
            else _product_rank_from_build_dir(build_dir)
        )
        _write_product_load_trace(
            event="unload_start",
            name=self.name,
            build_dir=build_dir,
            rank=rank,
            neff_path=self.neff_path,
            cc_enabled=self.cc_enabled,
        )
        try:
            model_ref = getattr(self.loaded, "model_ref", None)
            if model_ref is not None:
                from spike.spike_singleton import get_spike_singleton

                get_spike_singleton().unload_model(model_ref)
        finally:
            self.loaded = None
            _record_product_kernel_unloaded(
                key=id(self),
                name=self.name,
                build_dir=build_dir,
                rank=rank,
                neff_path=self.neff_path,
                cc_enabled=self.cc_enabled,
            )


def _run_product_kernel(
    kernel: Any,
    *,
    build_dir: str | None,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    unload_after_call: bool = True,
) -> None:
    rank = _product_rank_from_build_dir(build_dir)
    writer = _product_runtime_profile_writer(rank)
    if writer is not None:
        t0 = time.perf_counter()
        load_s = 0.0
        call_s = 0.0
        unload_s = 0.0
        loaded_before = bool(
            isinstance(kernel, _ProductPrecompiledKernel) and kernel.loaded is not None
        )
        status = "ok"
        error = ""
        try:
            _write_product_runtime_event(
                writer=writer,
                event="start",
                kernel=kernel,
                build_dir=build_dir,
                rank=rank,
                unload_after_call=unload_after_call,
                loaded_before=loaded_before,
                elapsed_s=time.perf_counter() - t0,
                inputs=inputs,
                outputs=outputs,
            )
            if isinstance(kernel, _ProductPrecompiledKernel):
                _write_product_runtime_event(
                    writer=writer,
                    event="load_start",
                    kernel=kernel,
                    build_dir=build_dir,
                    rank=rank,
                    unload_after_call=unload_after_call,
                    loaded_before=loaded_before,
                    elapsed_s=time.perf_counter() - t0,
                )
                t = time.perf_counter()
                loaded_kernel = kernel.load(build_dir=build_dir)
                load_s = time.perf_counter() - t
                _write_product_runtime_event(
                    writer=writer,
                    event="load_done",
                    kernel=kernel,
                    build_dir=build_dir,
                    rank=rank,
                    unload_after_call=unload_after_call,
                    loaded_before=loaded_before,
                    elapsed_s=time.perf_counter() - t0,
                )
                try:
                    _write_product_runtime_event(
                        writer=writer,
                        event="call_start",
                        kernel=kernel,
                        build_dir=build_dir,
                        rank=rank,
                        unload_after_call=unload_after_call,
                        loaded_before=loaded_before,
                        elapsed_s=time.perf_counter() - t0,
                        inputs=inputs,
                        outputs=outputs,
                    )
                    t = time.perf_counter()
                    loaded_kernel(inputs=inputs, outputs=outputs)
                    call_s = time.perf_counter() - t
                    _write_product_runtime_event(
                        writer=writer,
                        event="call_done",
                        kernel=kernel,
                        build_dir=build_dir,
                        rank=rank,
                        unload_after_call=unload_after_call,
                        loaded_before=loaded_before,
                        elapsed_s=time.perf_counter() - t0,
                    )
                finally:
                    if unload_after_call:
                        _write_product_runtime_event(
                            writer=writer,
                            event="unload_start",
                            kernel=kernel,
                            build_dir=build_dir,
                            rank=rank,
                            unload_after_call=unload_after_call,
                            loaded_before=loaded_before,
                            elapsed_s=time.perf_counter() - t0,
                        )
                        t = time.perf_counter()
                        kernel.unload(build_dir=build_dir)
                        unload_s = time.perf_counter() - t
                        _write_product_runtime_event(
                            writer=writer,
                            event="unload_done",
                            kernel=kernel,
                            build_dir=build_dir,
                            rank=rank,
                            unload_after_call=unload_after_call,
                            loaded_before=loaded_before,
                            elapsed_s=time.perf_counter() - t0,
                        )
            else:
                _write_product_runtime_event(
                    writer=writer,
                    event="call_start",
                    kernel=kernel,
                    build_dir=build_dir,
                    rank=rank,
                    unload_after_call=unload_after_call,
                    loaded_before=loaded_before,
                    elapsed_s=time.perf_counter() - t0,
                    inputs=inputs,
                    outputs=outputs,
                )
                t = time.perf_counter()
                kernel(inputs=inputs, outputs=outputs)
                call_s = time.perf_counter() - t
                _write_product_runtime_event(
                    writer=writer,
                    event="call_done",
                    kernel=kernel,
                    build_dir=build_dir,
                    rank=rank,
                    unload_after_call=unload_after_call,
                    loaded_before=loaded_before,
                    elapsed_s=time.perf_counter() - t0,
                )
        except Exception as exc:
            status = "error"
            error = repr(exc)
            raise
        finally:
            _write_product_runtime_profile(
                writer=writer,
                kernel=kernel,
                build_dir=build_dir,
                rank=rank,
                unload_after_call=unload_after_call,
                elapsed_s=time.perf_counter() - t0,
                load_s=load_s,
                call_s=call_s,
                unload_s=unload_s,
                loaded_before=loaded_before,
                status=status,
                error=error,
            )
        return
    if isinstance(kernel, _ProductPrecompiledKernel):
        loaded_kernel = kernel.load(build_dir=build_dir)
        try:
            loaded_kernel(inputs=inputs, outputs=outputs)
        finally:
            if unload_after_call:
                kernel.unload(build_dir=build_dir)
        return
    kernel(inputs=inputs, outputs=outputs)


def _resolve_product_kernel_for_load(
    kernel: Any,
    *,
    build_dir: str | None,
    load: bool,
) -> Any:
    if isinstance(kernel, _ProductPrecompiledKernel) and load:
        return kernel.load(build_dir=build_dir)
    return kernel


def _product_canonical_neff_record_path(
    *,
    build_dir: str | None,
    cache_key: str,
) -> Path:
    root = Path(build_dir or "/tmp/nkipy_serving_dsv4_product_compile")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_key))[:180] or "kernel"
    digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()[:12]
    return root / ".dsv4_product_canonical_neffs" / f"{safe}_{digest}.path"


def _read_neff_record_path(record_path: Path) -> str | None:
    try:
        neff_path = record_path.read_text().strip()
    except FileNotFoundError:
        return None
    if neff_path and Path(neff_path).exists():
        return neff_path
    return None


def _write_neff_record_path(record_path: Path, neff_path: str) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = record_path.with_suffix(f"{record_path.suffix}.tmp.{os.getpid()}")
    tmp_path.write_text(f"{str(neff_path)}\n")
    tmp_path.replace(record_path)


def _read_product_canonical_neff_path_with_source(
    *,
    build_dir: str | None,
    cache_key: str | None,
) -> tuple[str | None, str]:
    if not cache_key:
        return None, "canonical_miss"
    neff_path = _read_neff_record_path(
        _product_canonical_neff_record_path(
            build_dir=build_dir,
            cache_key=str(cache_key),
        )
    )
    if neff_path is not None:
        return neff_path, "canonical_hit"
    return None, "canonical_miss"


def _read_product_canonical_neff_path(
    *,
    build_dir: str | None,
    cache_key: str | None,
) -> str | None:
    neff_path, _ = _read_product_canonical_neff_path_with_source(
        build_dir=build_dir,
        cache_key=cache_key,
    )
    return neff_path


def _write_product_canonical_neff_path(
    *,
    build_dir: str | None,
    cache_key: str | None,
    neff_path: str,
) -> None:
    if not cache_key:
        return
    record_path = _product_canonical_neff_record_path(
        build_dir=build_dir,
        cache_key=str(cache_key),
    )
    _write_neff_record_path(record_path, str(neff_path))


def _read_product_signature_neff_path_with_source(
    *,
    build_dir: str | None,
    name: str,
    cache_key: str,
) -> tuple[str | None, str]:
    neff_path = read_canonical_neff_path(
        build_dir=build_dir,
        name=name,
        cache_key=cache_key,
    )
    if neff_path is not None:
        return neff_path, "canonical_hit"
    return None, "canonical_miss"


def _write_product_signature_neff_path(
    *,
    build_dir: str | None,
    name: str,
    cache_key: str,
    neff_path: str,
) -> None:
    write_canonical_neff_path(
        build_dir=build_dir,
        name=name,
        cache_key=cache_key,
        neff_path=str(neff_path),
    )


@dataclass(frozen=True)
class _ProductCompileCacheResult:
    neff_path: str
    cache_status: str
    lock_wait_s: float
    compile_s: float


def _default_compilation_target() -> Any:
    from nkipy.core.compile import CompilationTarget

    return CompilationTarget.DEFAULT


def _trace_product_neff_with_cache(
    *,
    device_kernel_cls: Any,
    fn: Any,
    sample_args: tuple[Any, ...],
    name: str,
    shared_build_dir: str | None,
    additional_compiler_args: str,
    compile_kwargs: dict[str, Any],
    target: Any,
    read_cached_neff: Any,
    write_cached_neff: Any,
) -> _ProductCompileCacheResult:
    lock_wait_s = 0.0
    compile_s = 0.0
    neff_path, cache_status = read_cached_neff()
    lock_t0 = time.perf_counter()
    if neff_path is None:
        with kernel_compile_lock(build_dir=shared_build_dir, name=name):
            lock_wait_s += time.perf_counter() - lock_t0
            neff_path, cache_status = read_cached_neff()
            if neff_path is None:
                compile_t0 = time.perf_counter()
                neff_path, _ = device_kernel_cls._trace_and_compile(
                    fn,
                    name,
                    sample_args,
                    compile_kwargs,
                    additional_compiler_args=additional_compiler_args,
                    use_cached_if_exists=True,
                    build_dir=shared_build_dir,
                    target=target,
                )
                compile_s += time.perf_counter() - compile_t0
                write_cached_neff(str(neff_path))
    return _ProductCompileCacheResult(
        neff_path=str(neff_path),
        cache_status=str(cache_status),
        lock_wait_s=lock_wait_s,
        compile_s=compile_s,
    )


def _trace_product_signature_neff_with_cache(
    *,
    device_kernel_cls: Any,
    fn: Any,
    sample_args: tuple[Any, ...],
    name: str,
    shared_build_dir: str | None,
    additional_compiler_args: str,
    kwargs: dict[str, Any],
) -> _ProductCompileCacheResult:
    """Trace a non-collective product kernel using the signature NEFF cache."""
    compile_kwargs = dict(kwargs)
    target = compile_kwargs.pop("target", None)
    if target is None:
        target = _default_compilation_target()
    cache_key = kernel_signature_cache_key(
        fn,
        name=name,
        sample_args=sample_args,
        kwargs=compile_kwargs,
        additional_compiler_args=additional_compiler_args,
        target=target,
    )
    return _trace_product_neff_with_cache(
        device_kernel_cls=device_kernel_cls,
        fn=fn,
        sample_args=sample_args,
        name=str(name),
        shared_build_dir=shared_build_dir,
        additional_compiler_args=additional_compiler_args,
        compile_kwargs=compile_kwargs,
        target=target,
        read_cached_neff=lambda: _read_product_signature_neff_path_with_source(
            build_dir=shared_build_dir,
            name=str(name),
            cache_key=cache_key,
        ),
        write_cached_neff=lambda neff_path: _write_product_signature_neff_path(
            build_dir=shared_build_dir,
            name=str(name),
            cache_key=cache_key,
            neff_path=str(neff_path),
        ),
    )


def _compile_product_kernel(
    fn: Any,
    *sample_args: Any,
    name: str,
    build_dir: str | None,
    additional_compiler_args: str,
    load: bool = True,
    **kwargs: Any,
) -> Any:
    shared_build_dir = _product_shared_build_dir(build_dir)
    load_barrier_name = kwargs.pop("load_barrier_name", None)
    load_barrier_rank_id = kwargs.pop("load_barrier_rank_id", None)
    load_barrier_world_size = kwargs.pop("load_barrier_world_size", None)
    canonical_neff_cache_key = kwargs.pop("canonical_neff_cache_key", None)
    cc_enabled = bool(kwargs.get("cc_enabled", False))
    profile_rank = (
        int(kwargs["rank_id"])
        if cc_enabled and kwargs.get("rank_id") is not None
        else _product_rank_from_build_dir(build_dir)
    )
    profile_writer = _product_compile_profile_writer(profile_rank)
    profile_t0 = time.perf_counter()
    profile_lock_wait_s = 0.0
    profile_compile_s = 0.0
    profile_load_s = 0.0
    profile_barrier_s = 0.0
    profile_cache_status = "not_started"
    profile_neff_path: str | None = None

    def _finish_profile(*, status: str = "ok", error: str = "") -> None:
        _write_product_compile_profile(
            writer=profile_writer,
            name=str(name),
            build_dir=build_dir,
            shared_build_dir=shared_build_dir,
            rank=profile_rank,
            cc_enabled=bool(cc_enabled),
            load=bool(load),
            status=status,
            cache_status=profile_cache_status,
            elapsed_s=time.perf_counter() - profile_t0,
            lock_wait_s=profile_lock_wait_s,
            compile_s=profile_compile_s,
            load_s=profile_load_s,
            barrier_s=profile_barrier_s,
            neff_path=profile_neff_path,
            canonical_neff_cache_key=canonical_neff_cache_key,
            error=error,
        )

    if cc_enabled:
        if not load_barrier_name:
            profile_cache_status = "invalid"
            _finish_profile(status="error", error="missing load_barrier_name")
            raise ValueError(
                "DSV4 product collective kernels require load_barrier_name"
            )
        # Collective model load can block until all ranks participate. Keep the
        # compile lock around NEFF generation only, then release it before load.
        device_kernel_cls = _get_device_kernel_cls()
        compile_kwargs = dict(kwargs)
        rank_id = compile_kwargs.pop("rank_id", None)
        world_size = compile_kwargs.pop("world_size", None)
        compile_kwargs.pop("cc_enabled", None)
        compile_kwargs.pop("is_spmd", None)
        target = compile_kwargs.pop("target", None)
        if rank_id is None or world_size is None:
            profile_cache_status = "invalid"
            _finish_profile(status="error", error="missing rank_id/world_size")
            raise ValueError(
                "DSV4 product collective kernels require rank_id/world_size"
            )
        if target is None:
            target = _default_compilation_target()
        barrier_rank_id = (
            int(load_barrier_rank_id)
            if load_barrier_rank_id is not None
            else int(rank_id)
        )
        barrier_world_size = (
            int(load_barrier_world_size)
            if load_barrier_world_size is not None
            else int(world_size)
        )
        try:
            cache_result = _trace_product_neff_with_cache(
                device_kernel_cls=device_kernel_cls,
                fn=fn,
                sample_args=sample_args,
                name=str(name),
                shared_build_dir=shared_build_dir,
                additional_compiler_args=additional_compiler_args,
                compile_kwargs=compile_kwargs,
                target=target,
                read_cached_neff=lambda: _read_product_canonical_neff_path_with_source(
                    build_dir=shared_build_dir,
                    cache_key=canonical_neff_cache_key,
                ),
                write_cached_neff=lambda neff_path: _write_product_canonical_neff_path(
                    build_dir=shared_build_dir,
                    cache_key=canonical_neff_cache_key,
                    neff_path=str(neff_path),
                ),
            )
            neff_path = cache_result.neff_path
            profile_cache_status = cache_result.cache_status
            profile_neff_path = cache_result.neff_path
            profile_lock_wait_s += cache_result.lock_wait_s
            profile_compile_s += cache_result.compile_s
            if not load:
                result = _ProductPrecompiledKernel(
                    neff_path=str(neff_path),
                    name=str(name),
                    cc_enabled=True,
                    rank_id=int(rank_id),
                    world_size=int(world_size),
                    load_barrier_name=str(load_barrier_name),
                    load_barrier_rank_id=barrier_rank_id,
                    load_barrier_world_size=barrier_world_size,
                )
                _finish_profile()
                return result
            if load_barrier_name:
                barrier_t0 = time.perf_counter()
                collective_load_barrier(
                    build_dir=shared_build_dir,
                    name=str(load_barrier_name),
                    rank_id=barrier_rank_id,
                    world_size=barrier_world_size,
                )
                profile_barrier_s += time.perf_counter() - barrier_t0
            load_t0 = time.perf_counter()
            _write_product_load_trace(
                event="direct_load_start",
                name=str(name),
                build_dir=build_dir,
                rank=profile_rank,
                neff_path=str(neff_path),
                cc_enabled=True,
            )
            try:
                result = device_kernel_cls.load_from_neff(
                    neff_path,
                    name=name,
                    cc_enabled=True,
                    rank_id=int(rank_id),
                    world_size=int(world_size),
                )
            except Exception as exc:
                _write_product_load_trace(
                    event="direct_load_error",
                    name=str(name),
                    build_dir=build_dir,
                    rank=profile_rank,
                    neff_path=str(neff_path),
                    cc_enabled=True,
                    error=repr(exc),
                )
                raise
            _tag_product_kernel_name(result, str(name))
            _record_product_kernel_loaded(
                key=id(result),
                name=str(name),
                build_dir=build_dir,
                rank=profile_rank,
                neff_path=str(neff_path),
                cc_enabled=True,
                event="direct_load_done",
            )
            profile_load_s += time.perf_counter() - load_t0
            _finish_profile()
            return result
        except Exception as exc:
            _finish_profile(status="error", error=repr(exc))
            raise
    if not load:
        device_kernel_cls = _get_device_kernel_cls()
        try:
            cache_result = _trace_product_signature_neff_with_cache(
                device_kernel_cls=device_kernel_cls,
                fn=fn,
                sample_args=sample_args,
                name=str(name),
                shared_build_dir=shared_build_dir,
                additional_compiler_args=additional_compiler_args,
                kwargs=kwargs,
            )
            neff_path = cache_result.neff_path
            profile_cache_status = cache_result.cache_status
            profile_neff_path = cache_result.neff_path
            profile_lock_wait_s += cache_result.lock_wait_s
            profile_compile_s += cache_result.compile_s
            result = _ProductPrecompiledKernel(
                neff_path=str(neff_path),
                name=str(name),
                cc_enabled=False,
            )
            _finish_profile()
            return result
        except Exception as exc:
            _finish_profile(status="error", error=repr(exc))
            raise
    try:
        device_kernel_cls = _get_device_kernel_cls()
        if not (
            hasattr(device_kernel_cls, "_trace_and_compile")
            and hasattr(device_kernel_cls, "load_from_neff")
        ):
            profile_cache_status = "invalid_device_kernel"
            raise RuntimeError(
                "DSV4 product kernels require DeviceKernel "
                "_trace_and_compile/load_from_neff so every kernel has a "
                "precompiled NEFF. compile_and_load fallback is disabled."
            )
        cache_result = _trace_product_signature_neff_with_cache(
            device_kernel_cls=device_kernel_cls,
            fn=fn,
            sample_args=sample_args,
            name=str(name),
            shared_build_dir=shared_build_dir,
            additional_compiler_args=additional_compiler_args,
            kwargs=kwargs,
        )
        neff_path = cache_result.neff_path
        profile_cache_status = cache_result.cache_status
        profile_neff_path = cache_result.neff_path
        profile_lock_wait_s += cache_result.lock_wait_s
        profile_compile_s += cache_result.compile_s
        load_t0 = time.perf_counter()
        _write_product_load_trace(
            event="direct_load_start",
            name=str(name),
            build_dir=build_dir,
            rank=profile_rank,
            neff_path=str(neff_path),
            cc_enabled=False,
        )
        try:
            result = device_kernel_cls.load_from_neff(neff_path, name=name)
        except Exception as exc:
            _write_product_load_trace(
                event="direct_load_error",
                name=str(name),
                build_dir=build_dir,
                rank=profile_rank,
                neff_path=str(neff_path),
                cc_enabled=False,
                error=repr(exc),
            )
            raise
        _tag_product_kernel_name(result, str(name))
        _record_product_kernel_loaded(
            key=id(result),
            name=str(name),
            build_dir=build_dir,
            rank=profile_rank,
            neff_path=str(neff_path),
            cc_enabled=False,
            event="direct_load_done",
        )
        profile_load_s += time.perf_counter() - load_t0
        _finish_profile()
        return result
    except Exception as exc:
        _finish_profile(status="error", error=repr(exc))
        raise
