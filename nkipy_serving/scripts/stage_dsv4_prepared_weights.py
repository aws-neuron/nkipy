"""Stage prepared DSV4 rank weights from shared storage to local disk.

Input: a prepared-weight root produced by ``scripts.prepare_dsv4_rank_weights``.

Output: a local prepared-weight cache root that can be assigned to
``dsv4_prepared_weight_local_dir`` or
``NKIPY_SERVING_DSV4_PREPARED_WEIGHT_LOCAL_DIR``.

Runtime can stage a rank lazily when both runtime config fields are set:

    dsv4_prepared_weight_dir=/path/to/prepared-root
    dsv4_prepared_weight_local_dir=/tmp/dsv4_prepared

For serving, pre-stage first to avoid many workers copying from shared storage during
executor startup:

    uv run python -m scripts.stage_dsv4_prepared_weights \
        --src-root /path/to/DeepSeek-V4-Flash-neuron-fp8-noscale-prepared-tp8-ep8-r1 \
        --local-root /tmp/dsv4_prepared_dsv4_fp8_tp8_ep8_r1 \
        --jobs 8 --expected-count 64
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from nkipy_serving.models.deepseek_v4.device_weights import (
    prepared_weight_local_rank_dir,
    stage_prepared_weight_rank_dir_local,
)


@dataclass(frozen=True)
class _StageResult:
    src: Path
    dst: Path
    status: str
    bytes: int
    elapsed_s: float


def _rank_dirs(root: Path) -> list[Path]:
    if (root / "metadata.json").exists():
        return [root]
    return sorted({path.parent for path in root.rglob("metadata.json")})


def _rank_dir_bytes(path: Path) -> int:
    total = 0
    for item in path.iterdir():
        if item.is_file():
            total += int(item.stat().st_size)
    return total


def _metadata_matches(src: Path, dst: Path) -> bool:
    src_meta = src / "metadata.json"
    dst_meta = dst / "metadata.json"
    return (
        src_meta.exists()
        and dst_meta.exists()
        and src_meta.read_bytes() == dst_meta.read_bytes()
    )


def _metadata_summary(path: Path) -> str:
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return "metadata=missing"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "metadata=invalid"
    layer_count = data.get("num_hidden_layers", "?")
    byte_count = data.get("bytes")
    if byte_count is None:
        return f"layers={layer_count}"
    return f"layers={layer_count} meta_bytes={int(byte_count) / 1e9:.3f}GB"


def _stage_one(
    src_root: Path,
    local_root: Path,
    rank_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> _StageResult:
    dst = prepared_weight_local_rank_dir(src_root, rank_dir, local_root)
    bytes_ = _rank_dir_bytes(rank_dir)
    if dry_run:
        if _metadata_matches(rank_dir, dst):
            status = "exists"
        elif dst.exists():
            status = "stale"
        else:
            status = "missing"
        return _StageResult(rank_dir, dst, status, bytes_, 0.0)

    if overwrite and dst.exists():
        shutil.rmtree(dst)
    if _metadata_matches(rank_dir, dst):
        return _StageResult(rank_dir, dst, "skip", bytes_, 0.0)

    t0 = time.perf_counter()
    staged = stage_prepared_weight_rank_dir_local(
        src_root,
        rank_dir,
        local_root,
    )
    elapsed = time.perf_counter() - t0
    return _StageResult(rank_dir, staged, "copied", bytes_, elapsed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-stage prepared DSV4 rank weights onto local disk.",
    )
    ap.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Prepared-weight source root on shared storage.",
    )
    ap.add_argument(
        "--local-root",
        type=Path,
        required=True,
        help="Local prepared-weight cache root.",
    )
    ap.add_argument(
        "--jobs", type=int, default=8, help="Parallel rank-directory copies."
    )
    ap.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail if discovered rank count differs.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report rank dirs and local cache status.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Force recopy even when metadata already matches.",
    )
    args = ap.parse_args(argv)

    src_root = args.src_root.resolve()
    local_root = args.local_root.resolve()
    rank_dirs = _rank_dirs(src_root)
    if args.expected_count is not None and len(rank_dirs) != int(args.expected_count):
        raise RuntimeError(
            f"Expected {int(args.expected_count)} prepared rank dirs under {src_root}, "
            f"found {len(rank_dirs)}"
        )
    total_bytes = sum(_rank_dir_bytes(path) for path in rank_dirs)
    print(
        f"[stage] src_root={src_root} local_root={local_root} "
        f"ranks={len(rank_dirs)} bytes={total_bytes / 1e9:.3f}GB "
        f"jobs={max(1, int(args.jobs))} dry_run={bool(args.dry_run)}",
        flush=True,
    )
    if rank_dirs:
        print(
            f"[stage] first_rank={rank_dirs[0]} {_metadata_summary(rank_dirs[0])}",
            flush=True,
        )

    t0 = time.perf_counter()
    copied = 0
    skipped = 0
    stale_or_missing = 0
    done = 0
    jobs = max(1, int(args.jobs))

    def _handle(result: _StageResult) -> None:
        nonlocal copied, skipped, stale_or_missing, done
        done += 1
        if result.status == "copied":
            copied += 1
        elif result.status == "skip" or result.status == "exists":
            skipped += 1
        else:
            stale_or_missing += 1
        if (
            done == 1
            or done == len(rank_dirs)
            or result.status == "copied"
            or done % max(1, min(16, len(rank_dirs))) == 0
        ):
            print(
                f"[stage-progress] done={done}/{len(rank_dirs)} "
                f"status={result.status} "
                f"bytes={result.bytes / 1e9:.3f}GB "
                f"elapsed={time.perf_counter() - t0:.1f}s "
                f"rank={result.src.name}",
                flush=True,
            )

    if jobs == 1 or len(rank_dirs) <= 1:
        for rank_dir in rank_dirs:
            _handle(
                _stage_one(
                    src_root,
                    local_root,
                    rank_dir,
                    dry_run=bool(args.dry_run),
                    overwrite=bool(args.overwrite),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(
                    _stage_one,
                    src_root,
                    local_root,
                    rank_dir,
                    dry_run=bool(args.dry_run),
                    overwrite=bool(args.overwrite),
                )
                for rank_dir in rank_dirs
            ]
            for fut in as_completed(futures):
                _handle(fut.result())

    print(
        f"[stage] done ranks={len(rank_dirs)} copied={copied} "
        f"skipped={skipped} stale_or_missing={stale_or_missing} "
        f"elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
