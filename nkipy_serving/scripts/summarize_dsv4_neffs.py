"""Summarize DeepSeek-V4 NEFF artifacts in a build directory.

Raw ``find *.neff`` counts are misleading for DSV4 product runs because build
roots often contain stale runs, per-rank subdirectories, and multiple serving
shapes. This helper reports both raw counts and unique filename stems grouped
by top-level artifact family.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_TRAILING_HASH_RE = re.compile(r"_[0-9a-f]{8,40}$")
_BUCKET_MARKER_RE = re.compile(r"_(?:t|b|s)\d+(?:_|$)")


def _top_level(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else "."


def _normalized_stem(stem: str) -> str:
    """Collapse compile-variant hash suffixes without hiding shape changes."""
    return _TRAILING_HASH_RE.sub("_<hash>", stem)


def _family_from_stem(stem: str) -> str:
    """Return the operation family before token/request-bucket shape suffixes."""
    normalized = _normalized_stem(stem)
    match = _BUCKET_MARKER_RE.search(normalized)
    if match is None:
        return normalized
    return normalized[: match.start()]


def _matches_bucket(stem: str, buckets: tuple[int, ...]) -> bool:
    if not buckets:
        return True
    for bucket in buckets:
        token = int(bucket)
        patterns = (
            rf"(^|_)t{token}($|_)",
            rf"(^|_)b{token}($|_)",
            rf"(^|_)s{token}($|_)",
        )
        if any(re.search(pattern, stem) for pattern in patterns):
            return True
    return False


def summarize_build_dir(
    build_dir: str | Path,
    *,
    current_buckets: tuple[int, ...] = (),
    topn: int = 20,
) -> dict[str, Any]:
    root = Path(build_dir)
    files = sorted(root.rglob("*.neff")) if root.exists() else []
    top_counts = Counter(_top_level(root, path) for path in files)
    rank_counts = Counter(
        top for top in top_counts if re.fullmatch(r"rank_\d+", str(top))
    )
    top_level: dict[str, Any] = {}
    for top in sorted(top_counts):
        stems = Counter(path.stem for path in files if _top_level(root, path) == top)
        normalized_stems = Counter()
        for stem, count in stems.items():
            normalized_stems[_normalized_stem(stem)] += count
        current = Counter(
            {
                stem: count
                for stem, count in stems.items()
                if _matches_bucket(stem, current_buckets)
            }
        )
        current_normalized = Counter()
        for stem, count in current.items():
            current_normalized[_normalized_stem(stem)] += count
        families = Counter()
        for stem, count in stems.items():
            families[_family_from_stem(stem)] += count
        family_variants = Counter(_family_from_stem(stem) for stem in normalized_stems)
        current_families = Counter()
        for stem, count in current.items():
            current_families[_family_from_stem(stem)] += count
        current_family_variants = Counter(
            _family_from_stem(stem) for stem in current_normalized
        )
        top_level[top] = {
            "raw_neff_count": int(top_counts[top]),
            "unique_stem_count": len(stems),
            "normalized_unique_stem_count": len(normalized_stems),
            "family_count": len(families),
            "currentish_unique_stem_count": len(current),
            "currentish_normalized_unique_stem_count": len(current_normalized),
            "currentish_family_count": len(current_families),
            "top_stems": [
                {"stem": stem, "count": int(count)}
                for stem, count in stems.most_common(int(topn))
            ],
            "normalized_top_stems": [
                {"stem": stem, "count": int(count)}
                for stem, count in normalized_stems.most_common(int(topn))
            ],
            "currentish_top_stems": [
                {"stem": stem, "count": int(count)}
                for stem, count in current.most_common(int(topn))
            ],
            "currentish_normalized_top_stems": [
                {"stem": stem, "count": int(count)}
                for stem, count in current_normalized.most_common(int(topn))
            ],
            "top_families": [
                {"family": family, "count": int(count)}
                for family, count in families.most_common(int(topn))
            ],
            "top_family_variants": [
                {"family": family, "variants": int(count)}
                for family, count in family_variants.most_common(int(topn))
            ],
            "currentish_top_families": [
                {"family": family, "count": int(count)}
                for family, count in current_families.most_common(int(topn))
            ],
            "currentish_top_family_variants": [
                {"family": family, "variants": int(count)}
                for family, count in current_family_variants.most_common(int(topn))
            ],
        }
    unique_stems = Counter(path.stem for path in files)
    normalized_unique_stems = Counter()
    for stem, count in unique_stems.items():
        normalized_unique_stems[_normalized_stem(stem)] += count
    families = Counter()
    for stem, count in unique_stems.items():
        families[_family_from_stem(stem)] += count
    family_variants = Counter(
        _family_from_stem(stem) for stem in normalized_unique_stems
    )
    return {
        "build_dir": str(root),
        "exists": root.exists(),
        "raw_neff_count": len(files),
        "unique_stem_count": len(unique_stems),
        "normalized_unique_stem_count": len(normalized_unique_stems),
        "family_count": len(families),
        "rank_dir_count": len(rank_counts),
        "rank_raw_neff_count": int(
            sum(top_counts[top] for top in top_counts if top in rank_counts)
        ),
        "current_buckets": list(current_buckets),
        "top_level_counts": {str(k): int(v) for k, v in top_counts.items()},
        "top_level": top_level,
        "top_stems": [
            {"stem": stem, "count": int(count)}
            for stem, count in unique_stems.most_common(int(topn))
        ],
        "normalized_top_stems": [
            {"stem": stem, "count": int(count)}
            for stem, count in normalized_unique_stems.most_common(int(topn))
        ],
        "top_families": [
            {"family": family, "count": int(count)}
            for family, count in families.most_common(int(topn))
        ],
        "top_family_variants": [
            {"family": family, "variants": int(count)}
            for family, count in family_variants.most_common(int(topn))
        ],
    }


def format_text(summary: dict[str, Any]) -> str:
    lines = [
        f"build_dir: {summary['build_dir']}",
        f"exists: {summary['exists']}",
        f"raw_neff_count: {summary['raw_neff_count']:,}",
        f"unique_stem_count: {summary['unique_stem_count']:,}",
        f"normalized_unique_stem_count: {summary['normalized_unique_stem_count']:,}",
        f"family_count: {summary['family_count']:,}",
        f"rank_dir_count: {summary['rank_dir_count']:,}",
        f"rank_raw_neff_count: {summary['rank_raw_neff_count']:,}",
    ]
    buckets = summary.get("current_buckets") or []
    if buckets:
        lines.append(f"current_buckets: {buckets}")
    lines.append("top_level:")
    if int(summary.get("rank_dir_count", 0)) > 0:
        lines.append(
            "  "
            f"rank_*: dirs={summary['rank_dir_count']:,}, "
            f"raw={summary['rank_raw_neff_count']:,}"
        )
    for top, payload in sorted(summary.get("top_level", {}).items()):
        if re.fullmatch(r"rank_\d+", str(top)):
            continue
        lines.append(
            "  "
            f"{top}: raw={payload['raw_neff_count']:,}, "
            f"unique={payload['unique_stem_count']:,}, "
            f"normalized={payload['normalized_unique_stem_count']:,}, "
            f"families={payload['family_count']:,}, "
            f"currentish={payload['currentish_unique_stem_count']:,}, "
            "currentish_normalized="
            f"{payload['currentish_normalized_unique_stem_count']:,}, "
            f"currentish_families={payload['currentish_family_count']:,}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", help="NKI/Neuron build directory to inspect")
    parser.add_argument(
        "--current-bucket",
        dest="current_buckets",
        action="append",
        type=int,
        default=[],
        help="Token/request bucket to highlight; may be passed multiple times",
    )
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = summarize_build_dir(
        args.build_dir,
        current_buckets=tuple(int(v) for v in args.current_buckets),
        topn=int(args.topn),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
