from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from nkipy_serving.models.deepseek_v4.device_weights import (
    _DSV4_PREPARED_WEIGHT_CACHE_VERSION,
    prepared_weight_rank_dir_for,
)
from nkipy_serving.models.deepseek_v4.rank_layout import local_expert_ids


def _load_validator():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "validate_dsv4_prepared_weights.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_dsv4_prepared_weights_under_test",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_VALIDATOR = _load_validator()
validate_main = _VALIDATOR.main
validate_prepared_weight_root = _VALIDATOR.validate_prepared_weight_root


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _write_rank_dir(
    root: Path,
    *,
    tp_degree: int = 2,
    ep_degree: int = 2,
    replica_degree: int = 2,
    lane: int = 0,
    tp_rank: int = 0,
    num_layers: int = 2,
    num_routed_experts: int = 8,
    metadata_overrides: dict[str, object] | None = None,
) -> Path:
    rank_dir = prepared_weight_rank_dir_for(
        root,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        replica_degree=replica_degree,
        lane=lane,
        tp_rank=tp_rank,
    )
    rank_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": _DSV4_PREPARED_WEIGHT_CACHE_VERSION,
        "source": "/checkpoint",
        "num_hidden_layers": num_layers,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "ep_degree": ep_degree,
        "replica_degree": replica_degree,
        "attention_lane": lane,
        "local_expert_ids": list(
            local_expert_ids(
                num_routed_experts,
                ep_degree,
                ep_rank=lane % ep_degree,
            )
        ),
        "bytes": 1,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (rank_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    _write_file(rank_dir / "dense.safetensors")
    for layer_id in range(num_layers):
        _write_file(rank_dir / f"layer_{layer_id:03d}.safetensors")
    return rank_dir


def _write_unique_replica_zero_root(root: Path) -> None:
    for lane in range(2):
        for tp_rank in range(2):
            _write_rank_dir(root, lane=lane, tp_rank=tp_rank)


def test_validate_dsv4_prepared_weights_accepts_replica_zero_fallback(tmp_path):
    root = tmp_path / "prepared"
    _write_unique_replica_zero_root(root)

    report = validate_prepared_weight_root(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
        num_hidden_layers=2,
        num_routed_experts=8,
        expected_count=4,
    )

    assert report.ok, [issue.format() for issue in report.issues]
    assert len(report.rank_dirs) == 4
    assert report.direct_runtime_ranks == 4
    assert report.fallback_runtime_ranks == 4
    assert report.runtime_ranks_covered == 8


def test_validate_dsv4_prepared_weights_rejects_missing_fallback_rank(tmp_path):
    root = tmp_path / "prepared"
    _write_unique_replica_zero_root(root)
    missing = prepared_weight_rank_dir_for(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
        lane=1,
        tp_rank=1,
    )
    for child in missing.iterdir():
        child.unlink()
    missing.rmdir()

    report = validate_prepared_weight_root(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
        num_hidden_layers=2,
        num_routed_experts=8,
    )

    messages = [issue.format() for issue in report.issues]
    assert not report.ok
    assert any("lane=1, tp_rank=1" in message for message in messages)
    assert any("lane=3, tp_rank=1" in message for message in messages)


def test_validate_dsv4_prepared_weights_rejects_metadata_topology_mismatch(tmp_path):
    root = tmp_path / "prepared"
    _write_unique_replica_zero_root(root)
    bad = prepared_weight_rank_dir_for(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
        lane=0,
        tp_rank=1,
    )
    meta_path = bad / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["tp_rank"] = 0
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_prepared_weight_root(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
    )

    assert not report.ok
    assert any(
        "directory tp rank mismatch" in issue.format() for issue in report.issues
    )


def test_validate_dsv4_prepared_weights_rejects_missing_layer_file(tmp_path):
    root = tmp_path / "prepared"
    _write_unique_replica_zero_root(root)
    rank_dir = prepared_weight_rank_dir_for(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
        lane=0,
        tp_rank=0,
    )
    (rank_dir / "layer_001.safetensors").unlink()

    report = validate_prepared_weight_root(
        root,
        tp_degree=2,
        ep_degree=2,
        replica_degree=2,
    )

    assert not report.ok
    assert any("layer_001.safetensors" in issue.format() for issue in report.issues)


def test_validate_dsv4_prepared_weights_cli_returns_error_for_bad_root(
    tmp_path, capsys
):
    missing = tmp_path / "missing"

    rc = validate_main(["--root", str(missing)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "prepared-weight root does not exist" in captured.out


def test_validate_dsv4_prepared_weights_cli_reports_invalid_metadata(tmp_path, capsys):
    root = tmp_path / "prepared"
    rank_dir = _write_rank_dir(root)
    meta_path = rank_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["num_hidden_layers"] = "invalid"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    rc = validate_main(["--root", str(root)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "num_hidden_layers" in captured.out


def test_validate_dsv4_prepared_weights_cli_accepts_valid_root(tmp_path, capsys):
    root = tmp_path / "prepared"
    _write_unique_replica_zero_root(root)

    rc = validate_main(
        [
            "--root",
            str(root),
            "--tp-degree",
            "2",
            "--ep-degree",
            "2",
            "--replica-degree",
            "2",
            "--num-hidden-layers",
            "2",
            "--num-routed-experts",
            "8",
            "--expected-count",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "[validate] OK" in captured.out
