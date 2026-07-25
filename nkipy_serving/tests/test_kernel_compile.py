from __future__ import annotations

import os

import numpy as np
import pytest

from nkipy_serving.models.deepseek_v4.neff_compiler import (
    _collective_load_barrier_metadata_for_groups,
)
from nkipy_serving.runtime import kernel_compile as kernel_compile_mod
from nkipy_serving.runtime.kernel_compile import compile_and_load_with_lock


def test_collective_load_barrier_metadata_uses_replica_group_union():
    groups = (tuple(range(16, 128)),)

    assert _collective_load_barrier_metadata_for_groups(
        rank_id=16,
        world_size=128,
        replica_groups=groups,
    ) == (0, 112)
    assert _collective_load_barrier_metadata_for_groups(
        rank_id=127,
        world_size=128,
        replica_groups=groups,
    ) == (111, 112)
    assert _collective_load_barrier_metadata_for_groups(
        rank_id=5,
        world_size=128,
        replica_groups=(tuple(range(16)),),
    ) == (5, 16)

    with pytest.raises(RuntimeError, match="not part of replica group union"):
        _collective_load_barrier_metadata_for_groups(
            rank_id=0,
            world_size=128,
            replica_groups=groups,
        )


def test_compile_and_load_with_lock_prefers_shared_neff_load(tmp_path):
    captured: dict[str, object] = {"traces": []}
    name = "kernel_test_direct_neff"
    sample = np.zeros((1,), dtype=np.float32)

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(fn, kernel_name, args, kwargs, **compile_kwargs):
            captured["traces"].append(
                {
                    "fn": fn,
                    "name": kernel_name,
                    "args": args,
                    "kwargs": kwargs,
                    "compile_kwargs": compile_kwargs,
                    "bir_dir": os.environ.get("NKIPY_SERVING_NKI_BIR_ARTIFACTS_DIR"),
                }
            )
            if compile_kwargs["use_cached_if_exists"]:
                raise RuntimeError(
                    "Compilation artifacts already exist in the output directory"
                )
            return str(tmp_path / "compiled.neff"), "cache"

        @staticmethod
        def load_from_neff(neff_path, **kwargs):
            captured["load"] = {"neff_path": neff_path, "kwargs": kwargs}
            return "loaded"

    prior_bir_dir = os.environ.get("NKIPY_SERVING_NKI_BIR_ARTIFACTS_DIR")
    out = compile_and_load_with_lock(
        _FakeDeviceKernel,
        lambda x: x,
        sample,
        name=name,
        build_dir=tmp_path / "rank_7" / "0123456789",
        namespace="shared_ns",
        static_arg=3,
    )

    assert out == "loaded"
    traces = captured["traces"]
    assert [trace["compile_kwargs"]["use_cached_if_exists"] for trace in traces] == [
        True,
        False,
    ]
    trace = traces[-1]
    assert trace["name"] == name
    assert trace["args"] == (sample,)
    assert trace["kwargs"] == {"static_arg": 3}
    assert trace["compile_kwargs"]["build_dir"] == str(
        tmp_path / "0123456789" / "shared_ns"
    )
    assert str(trace["bir_dir"]).startswith(
        str(tmp_path / "0123456789" / "shared_ns" / ".nki_bir")
    )
    assert captured["load"] == {
        "neff_path": str(tmp_path / "compiled.neff"),
        "kwargs": {"name": name},
    }
    assert os.environ.get("NKIPY_SERVING_NKI_BIR_ARTIFACTS_DIR") == prior_bir_dir


def test_compile_and_load_with_lock_loads_collectives_from_neff(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR",
        str(tmp_path / "global_neffs"),
    )
    captured: dict[str, object] = {}
    sample = np.zeros((1,), dtype=np.float32)

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(fn, kernel_name, args, kwargs, **compile_kwargs):
            captured["trace"] = {
                "fn": fn,
                "name": kernel_name,
                "args": args,
                "kwargs": kwargs,
                "compile_kwargs": compile_kwargs,
            }
            neff_path = tmp_path / "collective.neff"
            neff_path.write_text("fake-neff")
            return str(neff_path), "cache"

        @staticmethod
        def load_from_neff(neff_path, **kwargs):
            captured["load"] = {"neff_path": neff_path, "kwargs": kwargs}
            return "loaded"

        @staticmethod
        def compile_and_load(fn, *args, **kwargs):
            raise AssertionError("compile_and_load fallback must stay disabled")

    out = compile_and_load_with_lock(
        _FakeDeviceKernel,
        lambda x: x,
        sample,
        name="kernel_test_collective",
        build_dir=tmp_path / "rank_5" / "0123456789",
        namespace="collective_ns",
        cc_enabled=True,
        rank_id=5,
        world_size=8,
        is_spmd=True,
    )

    assert out == "loaded"
    trace = captured["trace"]
    assert trace["args"] == (sample,)
    assert trace["kwargs"] == {}
    assert trace["compile_kwargs"]["build_dir"] == str(
        tmp_path / "0123456789" / "collective_ns"
    )
    assert "cc_enabled" not in trace["compile_kwargs"]
    assert "rank_id" not in trace["compile_kwargs"]
    assert "world_size" not in trace["compile_kwargs"]
    assert "is_spmd" not in trace["compile_kwargs"]
    assert captured["load"] == {
        "neff_path": str(tmp_path / "collective.neff"),
        "kwargs": {
            "name": "kernel_test_collective",
            "cc_enabled": True,
            "rank_id": 5,
            "world_size": 8,
        },
    }


def test_compile_and_load_with_lock_requires_direct_neff_api_for_collectives(tmp_path):
    class _FakeDeviceKernel:
        @staticmethod
        def compile_and_load(*args, **kwargs):
            raise AssertionError("compile_and_load fallback must stay disabled")

    with pytest.raises(RuntimeError, match="precompiled NEFF"):
        compile_and_load_with_lock(
            _FakeDeviceKernel,
            lambda x: x,
            np.zeros((1,), dtype=np.float32),
            name="kernel_test_missing_neff_api",
            build_dir=tmp_path,
            namespace="collective_ns",
            cc_enabled=True,
            rank_id=0,
            world_size=1,
        )


def test_compile_and_load_with_lock_reuses_global_neff_without_lock(
    monkeypatch,
    tmp_path,
):
    traces: list[str] = []
    loads: list[dict[str, object]] = []
    lock_names: list[str] = []
    name = "kernel_test_global_neff"
    namespace = "attention_kernels"

    class _FakeLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(fn, kernel_name, args, kwargs, **compile_kwargs):
            del fn, args, kwargs, compile_kwargs
            traces.append(str(kernel_name))
            neff_path = tmp_path / f"compiled_{len(traces)}.neff"
            neff_path.write_text("fake-neff")
            return str(neff_path), "cache"

        @staticmethod
        def load_from_neff(neff_path, **kwargs):
            loads.append({"neff_path": str(neff_path), "kwargs": dict(kwargs)})
            return "loaded"

    monkeypatch.setenv(
        "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR",
        str(tmp_path / "global_neffs"),
    )
    monkeypatch.setattr(
        kernel_compile_mod,
        "kernel_compile_lock",
        lambda *, build_dir, name: lock_names.append(str(name)) or _FakeLock(),
    )

    sample = np.zeros((1,), dtype=np.float32)
    first = compile_and_load_with_lock(
        _FakeDeviceKernel,
        lambda x: x,
        sample,
        name=name,
        build_dir=tmp_path / "build_a" / "rank_0" / "0123456789",
        namespace=namespace,
        static_arg=3,
    )
    second = compile_and_load_with_lock(
        _FakeDeviceKernel,
        lambda x: x,
        np.ones((1,), dtype=np.float32),
        name=name,
        build_dir=tmp_path / "build_b" / "rank_1" / "0123456789",
        namespace=namespace,
        static_arg=3,
    )

    assert first == "loaded"
    assert second == "loaded"
    assert traces == [name]
    assert [load["neff_path"] for load in loads] == [loads[0]["neff_path"]] * 2
    assert lock_names == [name]


def test_sealed_kernel_compile_namespace_allows_hits_and_blocks_misses(
    monkeypatch,
    tmp_path,
):
    traces: list[str] = []
    loads: list[str] = []
    lock_names: list[str] = []
    namespace = "sealed_kernel_compile_ns"
    name = "kernel_test_sealed_global_hit"
    target = "fake-target"

    class _FakeLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(fn, kernel_name, args, kwargs, **compile_kwargs):
            del fn, args, kwargs, compile_kwargs
            traces.append(str(kernel_name))
            neff_path = tmp_path / f"compiled_{len(traces)}.neff"
            neff_path.write_text("fake-neff")
            return str(neff_path), "cache"

        @staticmethod
        def load_from_neff(neff_path, **kwargs):
            del kwargs
            loads.append(str(neff_path))
            return "loaded"

    def _identity(x):
        return x

    monkeypatch.setenv(
        "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR",
        str(tmp_path / "global_neffs"),
    )
    monkeypatch.setattr(
        kernel_compile_mod,
        "kernel_compile_lock",
        lambda *, build_dir, name: lock_names.append(str(name)) or _FakeLock(),
    )
    kernel_compile_mod.unseal_kernel_compile_namespace(namespace)

    sample = np.zeros((1,), dtype=np.float32)
    kwargs = {"static_arg": 7}
    cache_key = kernel_compile_mod.kernel_signature_cache_key(
        _identity,
        name=name,
        sample_args=(sample,),
        kwargs=kwargs,
        additional_compiler_args=None,
        target=target,
    )
    cached_neff = tmp_path / "cached.neff"
    cached_neff.write_text("fake-neff")
    kernel_compile_mod.write_global_neff_path(
        namespace=namespace,
        name=name,
        cache_key=cache_key,
        neff_path=cached_neff,
    )

    try:
        kernel_compile_mod.seal_kernel_compile_namespace(
            namespace,
            reason="test warmup complete",
        )
        assert kernel_compile_mod.is_kernel_compile_namespace_sealed(namespace)

        assert (
            kernel_compile_mod.compile_and_load_neff_with_lock(
                _FakeDeviceKernel,
                _identity,
                sample,
                name=name,
                build_dir=tmp_path / "build_a" / "rank_0" / "0123456789",
                namespace=namespace,
                target=target,
                **kwargs,
            )
            == "loaded"
        )

        with pytest.raises(
            RuntimeError,
            match="late compile blocked after namespace seal",
        ):
            kernel_compile_mod.compile_and_load_neff_with_lock(
                _FakeDeviceKernel,
                _identity,
                sample,
                name="kernel_test_sealed_miss",
                build_dir=tmp_path / "build_b" / "rank_1" / "0123456789",
                namespace=namespace,
                target=target,
                **kwargs,
            )
    finally:
        kernel_compile_mod.unseal_kernel_compile_namespace(namespace)

    assert traces == []
    assert loads == [str(cached_neff)]
    assert lock_names == ["kernel_test_sealed_miss"]


def test_compile_and_load_neff_local_hit_seeds_global_catalog(
    monkeypatch,
    tmp_path,
):
    loads: list[str] = []
    lock_names: list[str] = []
    name = "kernel_test_local_seed"
    namespace = "logits_processor"
    target = "fake-target"

    class _FakeLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _FakeDeviceKernel:
        @staticmethod
        def _trace_and_compile(*args, **kwargs):
            del args, kwargs
            raise AssertionError("local/global cache hit should avoid compile")

        @staticmethod
        def load_from_neff(neff_path, **kwargs):
            del kwargs
            loads.append(str(neff_path))
            return "loaded"

    def _identity(x):
        return x

    monkeypatch.setenv(
        "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR",
        str(tmp_path / "global_neffs"),
    )
    monkeypatch.setattr(
        kernel_compile_mod,
        "kernel_compile_lock",
        lambda *, build_dir, name: lock_names.append(str(name)) or _FakeLock(),
    )

    sample = np.zeros((1,), dtype=np.float32)
    kwargs = {"static_arg": 9}
    cache_key = kernel_compile_mod.kernel_signature_cache_key(
        _identity,
        name=name,
        sample_args=(sample,),
        kwargs=kwargs,
        additional_compiler_args=None,
        target=target,
    )
    local_neff = tmp_path / "local.neff"
    local_neff.write_text("fake-neff")
    shared_a = kernel_compile_mod.shared_kernel_build_dir(
        tmp_path / "build_a" / "rank_0" / "0123456789",
        namespace=namespace,
    )
    kernel_compile_mod.write_canonical_neff_path(
        build_dir=shared_a,
        name=name,
        cache_key=cache_key,
        neff_path=local_neff,
    )

    first = kernel_compile_mod.compile_and_load_neff_with_lock(
        _FakeDeviceKernel,
        _identity,
        sample,
        name=name,
        build_dir=tmp_path / "build_a" / "rank_0" / "0123456789",
        namespace=namespace,
        target=target,
        **kwargs,
    )
    second = kernel_compile_mod.compile_and_load_neff_with_lock(
        _FakeDeviceKernel,
        _identity,
        np.ones((1,), dtype=np.float32),
        name=name,
        build_dir=tmp_path / "build_b" / "rank_1" / "0123456789",
        namespace=namespace,
        target=target,
        **kwargs,
    )

    assert first == "loaded"
    assert second == "loaded"
    assert loads == [str(local_neff), str(local_neff)]
    assert lock_names == []


def test_neff_record_cache_rejects_rank_local_paths(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "NKIPY_SERVING_GLOBAL_NEFF_CACHE_DIR",
        str(tmp_path / "global_neffs"),
    )
    name = "kernel_test_rank_local_record"
    namespace = "logits_processor"
    cache_key = "rank-local-cache-key"
    bad_neff = tmp_path / "build_a" / "rank_7" / "0123456789" / "kernel" / "bad.neff"
    bad_neff.parent.mkdir(parents=True)
    bad_neff.write_text("fake-neff")

    shared_dir = kernel_compile_mod.shared_kernel_build_dir(
        tmp_path / "build_b" / "rank_1" / "0123456789",
        namespace=namespace,
    )
    kernel_compile_mod.write_global_neff_path(
        namespace=namespace,
        name=name,
        cache_key=cache_key,
        neff_path=bad_neff,
    )

    assert kernel_compile_mod.read_cached_neff_path_with_source(
        build_dir=shared_dir,
        namespace=namespace,
        name=name,
        cache_key=cache_key,
    ) == (None, "miss")
