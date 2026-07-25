"""Validate prepared DSV4 rank-weight cache roots.

Input: a prepared-weight root produced by ``scripts.prepare_dsv4_rank_weights``
or one rank directory under that root.

The runtime can load either a single rank directory or a shared prepared root:

    <root>/metadata.json
    <root>/tp8_ep8_rep1/lane00_tp00/metadata.json
    <root>/tp8_ep8_rep2/lane00_tp00/metadata.json

For replica_degree > 1, the loader accepts replica-zero prepared dirs for
runtime lanes in later replicas by falling back from ``lane`` to
``lane % ep_degree``. This validator checks that every runtime rank has either
a direct prepared dir or that replica-zero fallback dir.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nkipy_serving.models.deepseek_v4.device_weights import (
    _DSV4_PREPARED_WEIGHT_CACHE_VERSION,
)
from nkipy_serving.models.deepseek_v4.rank_layout import local_expert_ids

_LAYOUT_RE = re.compile(r"^tp(?P<tp>\d+)_ep(?P<ep>\d+)_rep(?P<rep>\d+)$")
_RANK_RE = re.compile(r"^lane(?P<lane>\d+)_tp(?P<tp_rank>\d+)$")
_REQUIRED_INT_FIELDS = (
    "version",
    "num_hidden_layers",
    "tp_degree",
    "tp_rank",
    "ep_degree",
    "replica_degree",
    "attention_lane",
)


@dataclass(frozen=True)
class ValidationIssue:
    message: str
    path: Path | None = None

    def format(self) -> str:
        if self.path is None:
            return self.message
        return f"{self.path}: {self.message}"


@dataclass(frozen=True)
class PreparedRankDir:
    path: Path
    metadata: dict[str, Any]
    metadata_issue: ValidationIssue | None = None
    layout_tp_degree: int | None = None
    layout_ep_degree: int | None = None
    layout_replica_degree: int | None = None
    dir_lane: int | None = None
    dir_tp_rank: int | None = None


@dataclass
class ValidationReport:
    root: Path
    rank_dirs: list[PreparedRankDir] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    direct_runtime_ranks: int = 0
    fallback_runtime_ranks: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def runtime_ranks_covered(self) -> int:
        return int(self.direct_runtime_ranks + self.fallback_runtime_ranks)


def _issue(message: str, path: Path | None = None) -> ValidationIssue:
    return ValidationIssue(message=message, path=path)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, ValidationIssue | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, _issue("metadata.json is missing", path.parent)
    except json.JSONDecodeError as exc:
        return None, _issue(f"metadata.json is invalid JSON: {exc}", path)
    if not isinstance(data, dict):
        return None, _issue("metadata.json must contain an object", path)
    return data, None


def _discover_rank_dirs(root: Path) -> list[Path]:
    if (root / "metadata.json").exists():
        return [root]
    return sorted({path.parent for path in root.rglob("metadata.json")})


def _parse_rank_dir(root: Path, rank_dir: Path) -> PreparedRankDir:
    meta, meta_issue = _read_json(rank_dir / "metadata.json")
    if meta_issue is not None:
        # The caller records the issue separately; keep a partial record so the
        # final summary can still mention this directory.
        meta = {}

    layout_match = _LAYOUT_RE.match(rank_dir.parent.name)
    rank_match = _RANK_RE.match(rank_dir.name)
    if rank_dir.resolve() == root.resolve():
        layout_match = None
        rank_match = None

    layout_tp = layout_ep = layout_rep = None
    if layout_match is not None:
        layout_tp = int(layout_match.group("tp"))
        layout_ep = int(layout_match.group("ep"))
        layout_rep = int(layout_match.group("rep"))

    dir_lane = dir_tp_rank = None
    if rank_match is not None:
        dir_lane = int(rank_match.group("lane"))
        dir_tp_rank = int(rank_match.group("tp_rank"))

    return PreparedRankDir(
        path=rank_dir,
        metadata=meta or {},
        metadata_issue=meta_issue,
        layout_tp_degree=layout_tp,
        layout_ep_degree=layout_ep,
        layout_replica_degree=layout_rep,
        dir_lane=dir_lane,
        dir_tp_rank=dir_tp_rank,
    )


def _int_field(
    rank: PreparedRankDir,
    key: str,
    issues: list[ValidationIssue],
) -> int | None:
    if key not in rank.metadata:
        issues.append(_issue(f"metadata missing {key!r}", rank.path))
        return None
    try:
        return int(rank.metadata[key])
    except (TypeError, ValueError):
        issues.append(
            _issue(
                f"metadata field {key!r} must be an integer, got {rank.metadata[key]!r}",
                rank.path,
            )
        )
        return None


def _infer_expected(
    ranks: list[PreparedRankDir],
    *,
    cli_value: int | None,
    metadata_key: str,
    layout_attr: str,
) -> int | None:
    if cli_value is not None:
        return int(cli_value)
    for rank in ranks:
        raw = rank.metadata.get(metadata_key)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
        layout_value = getattr(rank, layout_attr)
        if layout_value is not None:
            return int(layout_value)
    return None


def _positive_or_none(
    value: int | None,
    *,
    name: str,
    issues: list[ValidationIssue],
) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        issues.append(_issue(f"{name} must be positive, got {value}"))
        return None
    return value


def _require_file(path: Path, issues: list[ValidationIssue]) -> None:
    if not path.exists():
        issues.append(_issue("required file is missing", path))
        return
    if not path.is_file():
        issues.append(_issue("required path is not a file", path))
        return
    if path.stat().st_size <= 0:
        issues.append(_issue("required file is empty", path))


def _check_safetensors_header(path: Path, issues: list[ValidationIssue]) -> None:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="np") as handle:
            keys = list(handle.keys())
    except Exception as exc:  # pragma: no cover - exact safetensors errors vary.
        issues.append(_issue(f"safetensors header could not be opened: {exc}", path))
        return
    if not keys:
        issues.append(_issue("safetensors file contains no tensors", path))


def _validate_rank_files(
    rank: PreparedRankDir,
    cached_layers: int,
    issues: list[ValidationIssue],
    *,
    check_safetensors: bool,
) -> None:
    files = [rank.path / "dense.safetensors"]
    files.extend(
        rank.path / f"layer_{layer_id:03d}.safetensors"
        for layer_id in range(int(cached_layers))
    )
    for file_path in files:
        before = len(issues)
        _require_file(file_path, issues)
        if check_safetensors and len(issues) == before:
            _check_safetensors_header(file_path, issues)


def _validate_rank_metadata(
    rank: PreparedRankDir,
    issues: list[ValidationIssue],
    *,
    tp_degree: int | None,
    ep_degree: int | None,
    replica_degree: int | None,
    num_hidden_layers: int | None,
    num_routed_experts: int | None,
    check_safetensors: bool,
) -> tuple[int | None, tuple[int, ...] | None, int | None, int | None]:
    values = {key: _int_field(rank, key, issues) for key in _REQUIRED_INT_FIELDS}

    version = values["version"]
    if version is not None and int(version) != int(_DSV4_PREPARED_WEIGHT_CACHE_VERSION):
        issues.append(
            _issue(
                "metadata version mismatch: "
                f"cache={version}, expected={_DSV4_PREPARED_WEIGHT_CACHE_VERSION}",
                rank.path,
            )
        )

    expected_pairs = (
        ("tp_degree", tp_degree),
        ("ep_degree", ep_degree),
        ("replica_degree", replica_degree),
    )
    for key, expected in expected_pairs:
        cached = values[key]
        if cached is not None and int(cached) <= 0:
            issues.append(_issue(f"metadata {key} must be positive", rank.path))
        if expected is not None and cached is not None and int(cached) != int(expected):
            issues.append(
                _issue(
                    f"metadata {key} mismatch: cache={cached}, expected={expected}",
                    rank.path,
                )
            )

    if (
        rank.layout_tp_degree is not None
        and values["tp_degree"] is not None
        and values["tp_degree"] != rank.layout_tp_degree
    ):
        issues.append(
            _issue(
                "directory layout tp_degree mismatch: "
                f"dir={rank.layout_tp_degree}, metadata={values['tp_degree']}",
                rank.path,
            )
        )
    if (
        rank.layout_ep_degree is not None
        and values["ep_degree"] is not None
        and values["ep_degree"] != rank.layout_ep_degree
    ):
        issues.append(
            _issue(
                "directory layout ep_degree mismatch: "
                f"dir={rank.layout_ep_degree}, metadata={values['ep_degree']}",
                rank.path,
            )
        )
    if (
        rank.layout_replica_degree is not None
        and values["replica_degree"] is not None
        and values["replica_degree"] != rank.layout_replica_degree
    ):
        issues.append(
            _issue(
                "directory layout replica_degree mismatch: "
                f"dir={rank.layout_replica_degree}, metadata={values['replica_degree']}",
                rank.path,
            )
        )

    tp_rank = values["tp_rank"]
    lane = values["attention_lane"]
    if (
        rank.dir_tp_rank is not None
        and tp_rank is not None
        and tp_rank != rank.dir_tp_rank
    ):
        issues.append(
            _issue(
                f"directory tp rank mismatch: dir={rank.dir_tp_rank}, metadata={tp_rank}",
                rank.path,
            )
        )
    if rank.dir_lane is not None and lane is not None and lane != rank.dir_lane:
        issues.append(
            _issue(
                f"directory lane mismatch: dir={rank.dir_lane}, metadata={lane}",
                rank.path,
            )
        )
    if tp_degree is not None and tp_rank is not None and not (0 <= tp_rank < tp_degree):
        issues.append(
            _issue(f"tp_rank {tp_rank} is outside [0, {tp_degree})", rank.path)
        )
    if (
        ep_degree is not None
        and replica_degree is not None
        and lane is not None
        and not (0 <= lane < ep_degree * replica_degree)
    ):
        issues.append(
            _issue(
                f"attention_lane {lane} is outside [0, {ep_degree * replica_degree})",
                rank.path,
            )
        )

    cached_layers = values["num_hidden_layers"]
    if cached_layers is not None:
        if int(cached_layers) < 0:
            issues.append(_issue("num_hidden_layers must be non-negative", rank.path))
        if num_hidden_layers is not None and int(cached_layers) < int(
            num_hidden_layers
        ):
            issues.append(
                _issue(
                    "metadata num_hidden_layers is too small: "
                    f"cache={cached_layers}, expected_at_least={num_hidden_layers}",
                    rank.path,
                )
            )
        _validate_rank_files(
            rank,
            int(max(0, cached_layers)),
            issues,
            check_safetensors=check_safetensors,
        )

    raw_experts = rank.metadata.get("local_expert_ids")
    local_ids: tuple[int, ...] | None = None
    if not isinstance(raw_experts, list) or not raw_experts:
        issues.append(
            _issue("metadata local_expert_ids must be a non-empty list", rank.path)
        )
    else:
        try:
            local_ids = tuple(int(item) for item in raw_experts)
        except (TypeError, ValueError):
            issues.append(
                _issue("metadata local_expert_ids must contain integers", rank.path)
            )
    if (
        local_ids is not None
        and num_routed_experts is not None
        and ep_degree is not None
        and lane is not None
    ):
        expected = local_expert_ids(
            int(num_routed_experts),
            int(ep_degree),
            ep_rank=int(lane) % int(ep_degree),
        )
        if local_ids != expected:
            issues.append(
                _issue(
                    "metadata local_expert_ids mismatch: "
                    f"cache={local_ids}, expected={expected}",
                    rank.path,
                )
            )

    return cached_layers, local_ids, lane, tp_rank


def _validate_coverage(
    ranks: list[PreparedRankDir],
    issues: list[ValidationIssue],
    *,
    tp_degree: int | None,
    ep_degree: int | None,
    replica_degree: int | None,
    expected_count: int | None,
) -> tuple[int, int]:
    if expected_count is not None and len(ranks) != int(expected_count):
        issues.append(
            _issue(
                f"rank directory count mismatch: found={len(ranks)}, "
                f"expected={int(expected_count)}"
            )
        )

    is_single_rank_root = (
        len(ranks) == 1
        and ranks[0].path.joinpath("metadata.json").exists()
        and ranks[0].dir_lane is None
    )
    if not ranks or is_single_rank_root:
        return (1 if ranks else 0), 0
    if tp_degree is None or ep_degree is None or replica_degree is None:
        issues.append(_issue("cannot validate runtime coverage without tp/ep/replica"))
        return 0, 0

    by_lane_tp: dict[tuple[int, int], PreparedRankDir] = {}
    for rank in ranks:
        lane = rank.dir_lane
        tp_rank = rank.dir_tp_rank
        if lane is None or tp_rank is None:
            issues.append(
                _issue(
                    "rank dir must be named laneXX_tpYY under a tp*_ep*_rep* layout",
                    rank.path,
                )
            )
            continue
        key = (int(lane), int(tp_rank))
        if key in by_lane_tp:
            issues.append(_issue(f"duplicate rank dir for lane/tp {key}", rank.path))
        by_lane_tp[key] = rank

    direct = 0
    fallback = 0
    for lane in range(int(ep_degree) * int(replica_degree)):
        for tp_rank in range(int(tp_degree)):
            if (lane, tp_rank) in by_lane_tp:
                direct += 1
                continue
            fallback_key = (lane % int(ep_degree), tp_rank)
            if fallback_key in by_lane_tp:
                fallback += 1
                continue
            issues.append(
                _issue(
                    "missing prepared rank dir for runtime rank: "
                    f"lane={lane}, tp_rank={tp_rank}; "
                    f"fallback_lane={lane % int(ep_degree)}"
                )
            )
    return direct, fallback


def validate_prepared_weight_root(
    root: Path,
    *,
    tp_degree: int | None = None,
    ep_degree: int | None = None,
    replica_degree: int | None = None,
    num_hidden_layers: int | None = None,
    num_routed_experts: int | None = None,
    expected_count: int | None = None,
    check_safetensors: bool = False,
) -> ValidationReport:
    root = Path(root).expanduser().resolve()
    report = ValidationReport(root=root)
    if not root.exists():
        report.issues.append(_issue("prepared-weight root does not exist", root))
        return report
    if not root.is_dir():
        report.issues.append(_issue("prepared-weight root is not a directory", root))
        return report

    rank_paths = _discover_rank_dirs(root)
    if not rank_paths:
        report.issues.append(_issue("no prepared rank metadata found", root))
        return report

    ranks = [_parse_rank_dir(root, path) for path in rank_paths]
    report.rank_dirs = ranks
    tp_degree = _infer_expected(
        ranks,
        cli_value=tp_degree,
        metadata_key="tp_degree",
        layout_attr="layout_tp_degree",
    )
    ep_degree = _infer_expected(
        ranks,
        cli_value=ep_degree,
        metadata_key="ep_degree",
        layout_attr="layout_ep_degree",
    )
    replica_degree = _infer_expected(
        ranks,
        cli_value=replica_degree,
        metadata_key="replica_degree",
        layout_attr="layout_replica_degree",
    )
    tp_degree = _positive_or_none(
        tp_degree,
        name="tp_degree",
        issues=report.issues,
    )
    ep_degree = _positive_or_none(
        ep_degree,
        name="ep_degree",
        issues=report.issues,
    )
    replica_degree = _positive_or_none(
        replica_degree,
        name="replica_degree",
        issues=report.issues,
    )

    cached_layer_counts: set[int] = set()
    experts_by_lane_mod: dict[int, tuple[int, ...]] = {}
    for rank in ranks:
        if rank.metadata_issue is not None:
            report.issues.append(rank.metadata_issue)
            continue
        cached_layers, local_ids, lane, _tp_rank = _validate_rank_metadata(
            rank,
            report.issues,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            replica_degree=replica_degree,
            num_hidden_layers=num_hidden_layers,
            num_routed_experts=num_routed_experts,
            check_safetensors=check_safetensors,
        )
        if cached_layers is not None:
            cached_layer_counts.add(int(cached_layers))
        if local_ids is not None and lane is not None and ep_degree is not None:
            lane_mod = int(lane) % int(ep_degree)
            previous = experts_by_lane_mod.get(lane_mod)
            if previous is not None and previous != local_ids:
                report.issues.append(
                    _issue(
                        "local_expert_ids mismatch across TP ranks for "
                        f"lane_mod={lane_mod}: first={previous}, current={local_ids}",
                        rank.path,
                    )
                )
            experts_by_lane_mod[lane_mod] = local_ids

    if len(cached_layer_counts) > 1:
        report.issues.append(
            _issue(
                "rank metadata has inconsistent num_hidden_layers values: "
                f"{sorted(cached_layer_counts)}",
                root,
            )
        )

    direct, fallback = _validate_coverage(
        ranks,
        report.issues,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        replica_degree=replica_degree,
        expected_count=expected_count,
    )
    report.direct_runtime_ranks = direct
    report.fallback_runtime_ranks = fallback
    return report


def _print_report(report: ValidationReport) -> None:
    layer_counts: set[int] = set()
    for rank in report.rank_dirs:
        raw_layers = rank.metadata.get("num_hidden_layers")
        if raw_layers is None:
            continue
        try:
            layer_counts.add(int(raw_layers))
        except (TypeError, ValueError):
            pass
    layers = sorted(layer_counts)
    print(
        "[validate] "
        f"root={report.root} rank_dirs={len(report.rank_dirs)} "
        f"runtime_ranks_covered={report.runtime_ranks_covered} "
        f"direct_runtime_ranks={report.direct_runtime_ranks} "
        f"fallback_runtime_ranks={report.fallback_runtime_ranks} "
        f"layers={layers or 'unknown'}",
        flush=True,
    )
    if report.ok:
        print("[validate] OK", flush=True)
        return
    for issue in report.issues:
        print(f"[validate] ERROR {issue.format()}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate a prepared DeepSeek-V4 rank-weight cache root.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Prepared-weight root or one prepared rank directory.",
    )
    ap.add_argument("--tp-degree", type=int, default=None)
    ap.add_argument("--ep-degree", type=int, default=None)
    ap.add_argument("--replica-degree", type=int, default=None)
    ap.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Expected minimum layer count.",
    )
    ap.add_argument(
        "--num-routed-experts",
        type=int,
        default=None,
        help="Validate local_expert_ids against this global expert count.",
    )
    ap.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail if discovered rank-directory count differs.",
    )
    ap.add_argument(
        "--check-safetensors",
        action="store_true",
        help="Open safetensors headers to catch corrupt files.",
    )
    args = ap.parse_args(argv)

    report = validate_prepared_weight_root(
        args.root,
        tp_degree=args.tp_degree,
        ep_degree=args.ep_degree,
        replica_degree=args.replica_degree,
        num_hidden_layers=args.num_hidden_layers,
        num_routed_experts=args.num_routed_experts,
        expected_count=args.expected_count,
        check_safetensors=bool(args.check_safetensors),
    )
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
