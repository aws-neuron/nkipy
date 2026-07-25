from __future__ import annotations

import pytest

from nkipy_serving.runtime import collective_load


def test_collective_load_timeout_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S", raising=False)
    monkeypatch.delenv("NKIPY_SERVING_TP_WORKER_TIMEOUT_S", raising=False)

    assert collective_load._barrier_timeout_s(12.5) == 12.5

    monkeypatch.setenv("NKIPY_SERVING_TP_WORKER_TIMEOUT_S", "1234")
    assert collective_load._barrier_timeout_s(None) == 1234.0

    monkeypatch.setenv("NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S", "55")
    assert collective_load._barrier_timeout_s(None) == 55.0


def test_collective_load_run_id_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NKIPY_SERVING_COLLECTIVE_LOAD_RUN_ID", "run/abc")
    assert collective_load._barrier_run_id() == "run_abc"


def test_rank_shared_build_dir_strips_executor_rank_directory(tmp_path) -> None:
    namespace = "shared_neff"
    config_hash = "0123456789"
    for rank_dir in ("rank0", "rank_0"):
        build_dir = tmp_path / "build" / config_hash / rank_dir / "kernel"
        assert collective_load.rank_shared_build_dir(
            build_dir,
            namespace=namespace,
        ) == str(tmp_path / "build" / config_hash / namespace)


def test_rank_shared_build_dir_strips_worker_rank_and_preserves_hash(
    tmp_path,
) -> None:
    namespace = "shared_neff"
    config_hash = "0123456789"
    build_dir = tmp_path / "build" / "rank_7" / config_hash

    assert collective_load.rank_shared_build_dir(
        build_dir,
        namespace=namespace,
    ) == str(tmp_path / "build" / config_hash / namespace)


def test_rank_shared_build_dir_strips_nested_rank_scopes(tmp_path) -> None:
    namespace = "shared_neff"
    config_hash = "0123456789"
    build_dir = tmp_path / "build" / "rank_7" / config_hash / "rank7" / "kernel"

    assert collective_load.rank_shared_build_dir(
        build_dir,
        namespace=namespace,
    ) == str(tmp_path / "build" / config_hash / namespace)


def test_rank_shared_build_dir_adds_namespace_without_rank(tmp_path) -> None:
    namespace = "shared_neff"
    build_dir = tmp_path / "build" / "0123456789"

    assert collective_load.rank_shared_build_dir(
        build_dir,
        namespace=namespace,
    ) == str(build_dir / namespace)
