from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _iter_records(paths: list[str]):
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for path in matches:
            with open(path) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"{path}:{line_no}: invalid JSON") from exc
                    record["_path"] = path
                    yield record


def _add(stats: dict[Any, list[float]], key: Any, record: dict[str, Any]) -> None:
    elapsed = float(record.get("elapsed_s", 0.0) or 0.0)
    compile_s = float(record.get("compile_s", 0.0) or 0.0)
    load_s = float(record.get("load_s", 0.0) or 0.0)
    barrier_s = float(record.get("barrier_s", 0.0) or 0.0)
    lock_s = float(record.get("lock_wait_s", 0.0) or 0.0)
    row = stats[key]
    row[0] += 1
    row[1] += elapsed
    row[2] = max(row[2], elapsed)
    row[3] += compile_s
    row[4] += load_s
    row[5] += barrier_s
    row[6] += lock_s


def _print_table(title: str, stats: dict[Any, list[float]], *, top: int) -> None:
    print(f"\n{title}")
    print("count  elapsed_s  max_s  compile_s  load_s  barrier_s  lock_wait_s  key")
    for key, row in sorted(stats.items(), key=lambda item: item[1][1], reverse=True)[
        :top
    ]:
        count, elapsed, max_s, compile_s, load_s, barrier_s, lock_s = row
        print(
            f"{int(count):5d}  {elapsed:9.3f}  {max_s:5.3f}  "
            f"{compile_s:9.3f}  {load_s:6.3f}  {barrier_s:9.3f}  "
            f"{lock_s:11.3f}  {key}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize DSV4 product compile/load JSONL profiles."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["/tmp/nkipy_serving_profile/dsv4_product_compile_rank_*.jsonl"],
        help="JSONL file paths or globs.",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    by_name: dict[Any, list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    by_cache: dict[Any, list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    by_rank: dict[Any, list[float]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    total = 0

    for record in _iter_records(args.paths):
        total += 1
        _add(by_name, record.get("name", "<unknown>"), record)
        _add(
            by_cache,
            (
                record.get("cache_status"),
                "cc" if record.get("cc_enabled") else "noncc",
                "load" if record.get("load_requested") else "defer_load",
                record.get("status"),
            ),
            record,
        )
        _add(by_rank, record.get("rank"), record)

    print(f"records={total}")
    _print_table("By cache status", by_cache, top=args.top)
    _print_table("By rank", by_rank, top=args.top)
    _print_table("By kernel name", by_name, top=args.top)


if __name__ == "__main__":
    main()
