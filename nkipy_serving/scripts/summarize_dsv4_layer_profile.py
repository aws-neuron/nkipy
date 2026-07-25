"""Summarize DSV4 product runtime profiles by layer graph key."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_records(paths: list[str]):
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for path in matches:
            input_path = Path(path)
            if input_path.is_dir():
                expanded_paths = sorted(input_path.rglob("*.jsonl"))
            else:
                expanded_paths = [input_path]
            for expanded_path in expanded_paths:
                with open(expanded_path) as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"{expanded_path}:{line_no}: invalid JSON"
                            ) from exc
                        record["_path"] = str(expanded_path)
                        yield record


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_worker_step(record: dict[str, Any]) -> bool:
    path = Path(str(record.get("_path", ""))).name
    return (
        path.startswith("worker_") and path.endswith("_steps.jsonl") and "ts" in record
    )


def _derive_serving_window(records: list[dict[str, Any]]) -> dict[str, Any]:
    worker_rows = 0
    start_ts: float | None = None
    end_ts: float | None = None
    for record in records:
        if not _is_worker_step(record):
            continue
        ts = _as_float(record.get("ts"))
        if ts is None:
            continue
        duration = _as_float(record.get("t_total"))
        if duration is None:
            duration = _as_float(record.get("t_model_forward"))
        if duration is None:
            duration = 0.0
        row_start = ts - max(float(duration), 0.0)
        worker_rows += 1
        start_ts = row_start if start_ts is None else min(start_ts, row_start)
        end_ts = ts if end_ts is None else max(end_ts, ts)
    return {
        "worker_step_rows": int(worker_rows),
        "worker_window_start_ts": start_ts,
        "worker_window_end_ts": end_ts,
    }


def _overlaps_time_window(
    record: dict[str, Any],
    *,
    min_ts: float | None,
    max_ts: float | None,
) -> bool:
    if min_ts is None and max_ts is None:
        return True
    end_ts = _as_float(record.get("ts"))
    if end_ts is None:
        return False
    duration = _as_float(record.get("elapsed_s"))
    if duration is None:
        duration = _as_float(record.get("call_s"))
    start_ts = end_ts - max(float(duration or 0.0), 0.0)
    if min_ts is not None and end_ts < min_ts:
        return False
    if max_ts is not None and start_ts > max_ts:
        return False
    return True


def _new_row() -> dict[str, Any]:
    return {
        "calls": 0,
        "elapsed_s": 0.0,
        "call_s": 0.0,
        "load_s": 0.0,
        "unload_s": 0.0,
        "call_samples_s": [],
        "names": Counter(),
        "layers": Counter(),
    }


def _new_forward_row() -> dict[str, Any]:
    return {
        "calls": 0,
        "elapsed_s": 0.0,
        "elapsed_samples_s": [],
        "stages": Counter(),
        "stage_elapsed_s": defaultdict(float),
        "layers": Counter(),
    }


def _add(row: dict[str, Any], record: dict[str, Any]) -> None:
    row["calls"] += 1
    row["elapsed_s"] += float(record.get("elapsed_s", 0.0) or 0.0)
    row["call_s"] += float(record.get("call_s", 0.0) or 0.0)
    row["load_s"] += float(record.get("load_s", 0.0) or 0.0)
    row["unload_s"] += float(record.get("unload_s", 0.0) or 0.0)
    row["call_samples_s"].append(float(record.get("call_s", 0.0) or 0.0))
    row["names"][str(record.get("name", "<unknown>"))] += 1
    row["layers"][str(record.get("layer_graph_key", "<none>"))] += 1


def _add_forward(row: dict[str, Any], record: dict[str, Any]) -> None:
    elapsed = float(record.get("elapsed_s", 0.0) or 0.0)
    stage = str(record.get("stage", "<unknown>"))
    row["calls"] += 1
    row["elapsed_s"] += elapsed
    row["elapsed_samples_s"].append(elapsed)
    row["stages"][stage] += 1
    row["stage_elapsed_s"][stage] += elapsed
    row["layers"][str(record.get("layer_graph_key", "<none>"))] += 1


def _counter_items(counter: Counter[str], top: int) -> list[dict[str, Any]]:
    return [
        {"key": str(key), "count": int(count)}
        for key, count in counter.most_common(int(top))
    ]


def _elapsed_counter_items(
    counter: dict[str, float],
    *,
    counts: Counter[str],
    top: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            "key": str(key),
            "count": int(counts.get(key, 0)),
            "elapsed_s": round(float(elapsed), 6),
        }
        for key, elapsed in counter.items()
    ]
    return sorted(
        rows,
        key=lambda row: float(row["elapsed_s"]),
        reverse=True,
    )[: int(top)]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(float(q) * len(ordered)) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def _finalize_row(row: dict[str, Any], *, top: int) -> dict[str, Any]:
    samples = [float(value) for value in row["call_samples_s"]]
    steady_samples = samples[1:]
    calls = int(row["calls"])
    call_total = float(row["call_s"])
    steady_total = float(sum(steady_samples))
    return {
        "calls": calls,
        "unique_kernel_names": len(row["names"]),
        "layer_keys": len(row["layers"]),
        "elapsed_s": round(float(row["elapsed_s"]), 6),
        "call_s": round(call_total, 6),
        "call_avg_s": round(call_total / calls, 6) if calls else 0.0,
        "call_min_s": round(min(samples), 6) if samples else 0.0,
        "call_p50_s": round(_quantile(samples, 0.50), 6),
        "call_p95_s": round(_quantile(samples, 0.95), 6),
        "call_max_s": round(max(samples), 6) if samples else 0.0,
        "call_first_s": round(samples[0], 6) if samples else 0.0,
        "call_steady_count": len(steady_samples),
        "call_steady_s": round(steady_total, 6),
        "call_steady_avg_s": (
            round(steady_total / len(steady_samples), 6) if steady_samples else 0.0
        ),
        "load_s": round(float(row["load_s"]), 6),
        "unload_s": round(float(row["unload_s"]), 6),
        "top_kernel_names": _counter_items(row["names"], top),
        "top_layer_keys": _counter_items(row["layers"], top),
    }


def _finalize_forward_row(row: dict[str, Any], *, top: int) -> dict[str, Any]:
    samples = [float(value) for value in row["elapsed_samples_s"]]
    steady_samples = samples[1:]
    calls = int(row["calls"])
    elapsed_total = float(row["elapsed_s"])
    steady_total = float(sum(steady_samples))
    return {
        "calls": calls,
        "unique_stages": len(row["stages"]),
        "layer_keys": len(row["layers"]),
        "elapsed_s": round(elapsed_total, 6),
        "elapsed_avg_s": round(elapsed_total / calls, 6) if calls else 0.0,
        "elapsed_min_s": round(min(samples), 6) if samples else 0.0,
        "elapsed_p50_s": round(_quantile(samples, 0.50), 6),
        "elapsed_p95_s": round(_quantile(samples, 0.95), 6),
        "elapsed_max_s": round(max(samples), 6) if samples else 0.0,
        "elapsed_first_s": round(samples[0], 6) if samples else 0.0,
        "elapsed_steady_count": len(steady_samples),
        "elapsed_steady_s": round(steady_total, 6),
        "elapsed_steady_avg_s": (
            round(steady_total / len(steady_samples), 6) if steady_samples else 0.0
        ),
        "top_stages": _counter_items(row["stages"], top),
        "top_stage_times": _elapsed_counter_items(
            row["stage_elapsed_s"],
            counts=row["stages"],
            top=top,
        ),
        "top_layer_keys": _counter_items(row["layers"], top),
    }


_SORT_KEYS = {
    "call_s",
    "call_avg_s",
    "call_first_s",
    "call_steady_s",
    "call_steady_avg_s",
    "call_p50_s",
    "call_p95_s",
    "call_max_s",
    "load_s",
}

_FORWARD_SORT_KEYS = {
    "elapsed_s",
    "elapsed_avg_s",
    "elapsed_first_s",
    "elapsed_steady_s",
    "elapsed_steady_avg_s",
    "elapsed_p50_s",
    "elapsed_p95_s",
    "elapsed_max_s",
}


def summarize_layer_profile(
    paths: list[str],
    *,
    top: int = 20,
    sort_by: str = "call_s",
    min_ts: float | None = None,
    max_ts: float | None = None,
    serving_window: bool = False,
) -> dict[str, Any]:
    sort_by = str(sort_by)
    if sort_by not in _SORT_KEYS and sort_by not in _FORWARD_SORT_KEYS:
        raise ValueError(
            "unsupported sort key "
            f"{sort_by!r}; expected one of {sorted(_SORT_KEYS | _FORWARD_SORT_KEYS)}"
        )
    records = list(_iter_records(paths))
    serving_window_info = _derive_serving_window(records)
    if serving_window:
        if serving_window_info["worker_window_start_ts"] is None:
            raise ValueError(
                "--serving-window requires worker_*_steps.jsonl rows in the input"
            )
        worker_start = float(serving_window_info["worker_window_start_ts"])
        worker_end = float(serving_window_info["worker_window_end_ts"])
        min_ts = worker_start if min_ts is None else max(float(min_ts), worker_start)
        max_ts = worker_end if max_ts is None else min(float(max_ts), worker_end)
    if min_ts is not None and max_ts is not None and float(min_ts) > float(max_ts):
        raise ValueError(f"invalid timestamp window: min_ts={min_ts} > max_ts={max_ts}")

    by_layer: dict[str, dict[str, Any]] = defaultdict(_new_row)
    by_variant: dict[str, dict[str, Any]] = defaultdict(_new_row)
    by_name: dict[str, dict[str, Any]] = defaultdict(_new_row)
    by_forward_layer: dict[str, dict[str, Any]] = defaultdict(_new_forward_row)
    by_forward_variant: dict[str, dict[str, Any]] = defaultdict(_new_forward_row)
    by_forward_stage: dict[str, dict[str, Any]] = defaultdict(_new_forward_row)
    runtime_rows = 0
    forward_rows = 0
    no_layer_rows = 0
    timestamp_filtered_rows = 0

    for record in records:
        layer_key = str(record.get("layer_graph_key") or "<none>")
        variant_key = str(record.get("layer_variant_key") or "<none>")
        stage = str(record.get("stage") or "<unknown>")
        if record.get("stage") == "run_product_kernel":
            if not _overlaps_time_window(record, min_ts=min_ts, max_ts=max_ts):
                timestamp_filtered_rows += 1
                continue
            runtime_rows += 1
            kernel_name = str(record.get("name") or "<unknown>")
            if layer_key == "<none>":
                no_layer_rows += 1
            _add(by_layer[layer_key], record)
            _add(by_variant[variant_key], record)
            _add(by_name[kernel_name], record)
            continue
        if layer_key == "<none>":
            continue
        if not _overlaps_time_window(record, min_ts=min_ts, max_ts=max_ts):
            timestamp_filtered_rows += 1
            continue
        forward_rows += 1
        _add_forward(by_forward_layer[layer_key], record)
        _add_forward(by_forward_variant[variant_key], record)
        _add_forward(by_forward_stage[stage], record)

    def _top_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        runtime_sort_by = sort_by if sort_by in _SORT_KEYS else "call_s"
        finalized = [
            {"key": key, **_finalize_row(row, top=top)} for key, row in rows.items()
        ]
        return sorted(
            finalized,
            key=lambda row: float(row.get(runtime_sort_by, 0.0) or 0.0),
            reverse=True,
        )[: int(top)]

    def _top_forward_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        forward_sort_by = sort_by if sort_by in _FORWARD_SORT_KEYS else "elapsed_s"
        finalized = [
            {"key": key, **_finalize_forward_row(row, top=top)}
            for key, row in rows.items()
        ]
        return sorted(
            finalized,
            key=lambda row: float(row.get(forward_sort_by, 0.0) or 0.0),
            reverse=True,
        )[: int(top)]

    return {
        "runtime_rows": int(runtime_rows),
        "forward_rows": int(forward_rows),
        "sort_by": sort_by,
        "time_filter": "serving_window"
        if serving_window
        else ("manual" if min_ts is not None or max_ts is not None else "none"),
        "min_ts": round(float(min_ts), 6) if min_ts is not None else None,
        "max_ts": round(float(max_ts), 6) if max_ts is not None else None,
        "timestamp_filtered_rows": int(timestamp_filtered_rows),
        **serving_window_info,
        "layer_keys": len(by_layer),
        "variant_keys": len(by_variant),
        "unique_kernel_names": len(by_name),
        "rows_without_layer_key": int(no_layer_rows),
        "top_variants": _top_rows(by_variant),
        "top_layers": _top_rows(by_layer),
        "top_kernel_names": _top_rows(by_name),
        "forward_layer_keys": len(by_forward_layer),
        "forward_variant_keys": len(by_forward_variant),
        "forward_stage_keys": len(by_forward_stage),
        "top_forward_variants": _top_forward_rows(by_forward_variant),
        "top_forward_layers": _top_forward_rows(by_forward_layer),
        "top_forward_stages": _top_forward_rows(by_forward_stage),
    }


def _print_text(summary: dict[str, Any]) -> None:
    print(
        "runtime_rows={runtime_rows} layer_keys={layer_keys} "
        "variant_keys={variant_keys} unique_kernel_names={unique_kernel_names} "
        "rows_without_layer_key={rows_without_layer_key}".format(**summary)
    )
    if summary.get("time_filter") != "none":
        print(
            "time_filter={time_filter} min_ts={min_ts} max_ts={max_ts} "
            "timestamp_filtered_rows={timestamp_filtered_rows} "
            "worker_step_rows={worker_step_rows}".format(**summary)
        )
    for title, key in (
        ("Top variants", "top_variants"),
        ("Top layers", "top_layers"),
        ("Top kernel names", "top_kernel_names"),
    ):
        print(f"\n{title}")
        print(
            "calls  unique  layers  total_s   avg_s  first_s  steady_avg_s"
            "    p50_s    p95_s    max_s  load_s  key"
        )
        for row in summary[key]:
            print(
                f"{row['calls']:5d}  {row['unique_kernel_names']:6d}  "
                f"{row['layer_keys']:6d}  {row['call_s']:7.3f}  "
                f"{row['call_avg_s']:6.3f}  {row['call_first_s']:7.3f}  "
                f"{row['call_steady_avg_s']:12.3f}  "
                f"{row['call_p50_s']:7.3f}  {row['call_p95_s']:7.3f}  "
                f"{row['call_max_s']:7.3f}  "
                f"{row['load_s']:6.3f}  {row['key']}"
            )
    if not summary.get("forward_rows"):
        return
    print("\nTop forward stages")
    print(
        "calls  stages  layers  total_s   avg_s  first_s  steady_avg_s"
        "    p50_s    p95_s    max_s  key"
    )
    for row in summary["top_forward_stages"]:
        print(
            f"{row['calls']:5d}  {row['unique_stages']:6d}  "
            f"{row['layer_keys']:6d}  {row['elapsed_s']:7.3f}  "
            f"{row['elapsed_avg_s']:6.3f}  {row['elapsed_first_s']:7.3f}  "
            f"{row['elapsed_steady_avg_s']:12.3f}  "
            f"{row['elapsed_p50_s']:7.3f}  {row['elapsed_p95_s']:7.3f}  "
            f"{row['elapsed_max_s']:7.3f}  {row['key']}"
        )
    print("\nForward-stage fragmentation")
    print(
        "calls  stages  layers  total_s   avg_s  first_s  steady_avg_s"
        "    p50_s    p95_s    max_s  key"
    )
    for row in summary["top_forward_layers"]:
        print(
            f"{row['calls']:5d}  {row['unique_stages']:6d}  "
            f"{row['layer_keys']:6d}  {row['elapsed_s']:7.3f}  "
            f"{row['elapsed_avg_s']:6.3f}  {row['elapsed_first_s']:7.3f}  "
            f"{row['elapsed_steady_avg_s']:12.3f}  "
            f"{row['elapsed_p50_s']:7.3f}  {row['elapsed_p95_s']:7.3f}  "
            f"{row['elapsed_max_s']:7.3f}  {row['key']}"
        )
        top_stages = ", ".join(
            f"{item['key']}:{item['count']}" for item in row["top_stages"][:5]
        )
        print(f"       top_stages: {top_stages}")
        top_stage_times = ", ".join(
            f"{item['key']}:{item['elapsed_s']:.3f}s"
            for item in row["top_stage_times"][:5]
        )
        print(f"       top_stage_times: {top_stage_times}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize DSV4 product runtime profiles by layer graph key."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["/tmp/nkipy_serving_profile/dsv4_product_runtime_rank_*.jsonl"],
        help="Runtime JSONL file paths or globs.",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--sort-by",
        choices=sorted(_SORT_KEYS | _FORWARD_SORT_KEYS),
        default="call_s",
        help="Metric used to rank rows.",
    )
    parser.add_argument(
        "--serving-window",
        action="store_true",
        help=(
            "Restrict product rows to the request window derived from "
            "worker_*_steps.jsonl records in the input."
        ),
    )
    parser.add_argument(
        "--min-ts",
        type=float,
        default=None,
        help="Only include product rows whose execution overlaps this timestamp or later.",
    )
    parser.add_argument(
        "--max-ts",
        type=float,
        default=None,
        help="Only include product rows whose execution overlaps this timestamp or earlier.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    summary = summarize_layer_profile(
        args.paths,
        top=int(args.top),
        sort_by=str(args.sort_by),
        min_ts=args.min_ts,
        max_ts=args.max_ts,
        serving_window=bool(args.serving_window),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_text(summary)


if __name__ == "__main__":
    main()
