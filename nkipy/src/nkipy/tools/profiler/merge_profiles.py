"""Merge scalene CPU profiles with kernel device profiles.

Usage::

    python -m nkipy.tools.profiler scalene.json kernel_profile.json merged.json
    python -m nkipy.tools.profiler --kernel-only kernel_profile.json output.json
    python -m nkipy.tools.profiler --kernel-only kp_rank0.json kp_rank1.json output.json
"""

import copy
import json
import linecache
import os
import sys
from collections import defaultdict
from pathlib import Path

from .kernel_profiler import KernelProfileResult
from .trace_timeline import _parse_trace_timeline

# Default values for scalene LineData fields the GUI accesses unconditionally.
_LINE_DEFAULTS = {
    "line": "",
    "n_cpu_percent_python": 0.0,
    "n_cpu_percent_c": 0.0,
    "n_sys_percent": 0.0,
    "n_core_utilization": 0.0,
    "n_peak_mb": 0.0,
    "n_avg_mb": 0.0,
    "n_python_fraction": 0.0,
    "n_copy_mb_s": 0.0,
    "n_copy_mb": 0.0,
    "n_gpu_percent": 0.0,
    "n_gpu_peak_memory_mb": 0.0,
    "n_gpu_avg_memory_mb": 0.0,
    "n_usage_fraction": 0.0,
    "n_malloc_mb": 0.0,
    "n_mallocs": 0,
    "n_growth_mb": 0.0,
    "memory_samples": [],
    "start_region_line": 0,
    "end_region_line": 0,
    "start_function_line": 0,
    "end_function_line": 0,
    "start_outermost_loop": 0,
    "end_outermost_loop": 0,
    "cpu_samples_list": [],
    "async_task_names": [],
    "is_coroutine": False,
    "n_async_await_percent": 0.0,
    "n_async_concurrency_mean": 0.0,
    "n_async_concurrency_peak": 0.0,
}

_FILE_DEFAULTS = {
    "functions": [],
    "imports": [],
    "percent_cpu_time": 0.0,
}


def _load_scalene_json(scalene_path: str | Path) -> dict:
    """Load scalene JSON, handling both raw JSON and HTML-embedded JSON."""
    content = Path(scalene_path).read_text().strip()
    if "<!DOCTYPE html>" in content:
        marker = "const profile = "
        json_start = content.find(marker) + len(marker)
        if json_start > len(marker) - 1:
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(content[json_start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = json_start + i + 1
                        break
            content = content[json_start:json_end]
    return json.loads(content)


def _normalize_path(p: str) -> str:
    """Normalize a file path for comparison."""
    try:
        return os.path.realpath(p)
    except (OSError, ValueError):
        return p


def _match_filename(kernel_filename: str, scalene_filenames: list[str]) -> str | None:
    """Match a kernel filename to one of the scalene profile filenames.

    Tries realpath match first, then falls back to basename match.
    """
    norm_kernel = _normalize_path(kernel_filename)
    for sf in scalene_filenames:
        if _normalize_path(sf) == norm_kernel:
            return sf

    # Fallback: basename match
    kernel_base = os.path.basename(kernel_filename)
    for sf in scalene_filenames:
        if os.path.basename(sf) == kernel_base:
            return sf

    return None


def _calculate_cpu_sample_overlap(
    cpu_samples_list: list[float],
    nc_intervals_sec: list[tuple[float, float]],
    start_time_absolute: float,
    start_time_perf: float,
) -> float:
    """Calculate what percent of CPU samples overlap with NC execution intervals.

    Args:
        cpu_samples_list: CPU sample timestamps (perf_counter values).
        nc_intervals_sec: (start, stop) in absolute seconds for nc_exec_running.
        start_time_absolute: Scalene's absolute start time (Unix timestamp seconds).
        start_time_perf: Scalene's perf_counter start time (seconds).

    Returns:
        Overlap percentage (0-100).
    """
    if not cpu_samples_list or not nc_intervals_sec:
        return 0.0

    overlap_count = 0
    for sample_perf in cpu_samples_list:
        sample_abs = start_time_absolute + (sample_perf - start_time_perf)
        for start, end in nc_intervals_sec:
            if start <= sample_abs <= end:
                overlap_count += 1
                break

    return (overlap_count / len(cpu_samples_list) * 100) if cpu_samples_list else 0.0


def _print_utilization_summary(
    profile: KernelProfileResult,
    timeline,
    total_nc_time: float,
    total_nrt_time: float,
    line_nc_durations: dict,
    line_nrt_durations: dict,
    line_counts: dict,
    line_kernel_names: dict,
) -> None:
    """Print utilization summary to stdout."""
    wall_ms = (profile.wall_stop_ns - profile.wall_start_ns) / 1_000_000.0
    idle_ms = wall_ms - total_nc_time if wall_ms > total_nc_time else 0.0
    util_pct = (total_nc_time / wall_ms * 100) if wall_ms > 0 else 0.0

    print("\n=== Device Profiling Summary ===")
    print(f"Wall time:        {wall_ms:.1f} ms")
    print(f"NRT execution:    {total_nrt_time:.1f} ms (host-side)")
    print(f"NC execution:     {total_nc_time:.1f} ms ({util_pct:.1f}% of wall)")
    if total_nrt_time > 0:
        print(
            f"NC/NRT ratio:     {total_nc_time / total_nrt_time:.2f} "
            f"({'async' if total_nc_time > total_nrt_time else 'sync'})"
        )
    print(f"Device idle:      {idle_ms:.1f} ms ({100 - util_pct:.1f}%)")

    if timeline.gap_breakdown_ms and idle_ms > 0:
        sorted_gaps = sorted(
            timeline.gap_breakdown_ms.items(), key=lambda x: x[1], reverse=True
        )
        category_labels = {
            "d2h_transfer": "D2H transfers",
            "h2d_transfer": "H2D transfers",
            "memory_mgmt": "Memory mgmt",
            "exec_overhead": "Exec overhead",
            "sync_wait": "Sync waits",
        }
        for category, ms in sorted_gaps:
            label = category_labels.get(category, category)
            pct_of_idle = (ms / idle_ms * 100) if idle_ms > 0 else 0.0
            print(f"  {label + ':':18s} {ms:6.1f} ms ({pct_of_idle:4.1f}% of idle)")

    # Per-kernel-name aggregation
    kernel_agg: dict[str, dict] = {}
    for kc in profile.kernel_calls:
        key = (kc.filename, kc.lineno)
        nc_dur = line_nc_durations.get(key, 0.0) / line_counts.get(key, 1)
        nrt_dur = line_nrt_durations.get(key, 0.0) / line_counts.get(key, 1)
        name = kc.kernel_name
        if name not in kernel_agg:
            kernel_agg[name] = {"nc_time_ms": 0.0, "nrt_time_ms": 0.0, "count": 0}
        kernel_agg[name]["nc_time_ms"] += nc_dur
        kernel_agg[name]["nrt_time_ms"] += nrt_dur
        kernel_agg[name]["count"] += 1

    if kernel_agg:
        print("\nTop kernels:")
        sorted_kernels = sorted(
            kernel_agg.items(), key=lambda x: x[1]["nc_time_ms"], reverse=True
        )
        for name, agg in sorted_kernels:
            nc_pct = (
                (agg["nc_time_ms"] / total_nc_time * 100) if total_nc_time > 0 else 0.0
            )
            nc_avg = agg["nc_time_ms"] / agg["count"] if agg["count"] > 0 else 0.0
            nrt_avg = agg["nrt_time_ms"] / agg["count"] if agg["count"] > 0 else 0.0
            print(
                f"  {name:30s} {agg['nc_time_ms']:8.1f} ms ({nc_pct:5.1f}%)  "
                f"{agg['count']:5d} calls  nc={nc_avg:.3f} nrt={nrt_avg:.3f} ms/call"
            )


def merge_scalene_and_kernel_profiles(
    scalene_json_path: str | Path,
    kernel_profile_path: str | Path,
    output_path: str | Path,
) -> None:
    """Merge a scalene CPU profile with a kernel device profile.

    The merged output adds per-line device timing fields (``nrt_time_ms``,
    ``nc_time_ms``, etc.) to the scalene JSON so the scalene GUI can render
    NeuronCore time bars alongside CPU time.

    Args:
        scalene_json_path: Path to scalene output JSON (or HTML).
        kernel_profile_path: Path to KernelProfileResult JSON.
        output_path: Where to write the merged JSON.
    """
    scalene_data = _load_scalene_json(scalene_json_path)
    profile = KernelProfileResult.load(kernel_profile_path)

    # Parse full timeline for utilization analysis
    timeline = _parse_trace_timeline(profile.events_json)
    nc_durations = timeline.nc_durations_ms
    nrt_durations = timeline.nrt_durations_ms

    # Correlate by call index: kernel_calls[i] -> nc/nrt durations[i]
    # Aggregate per (filename, lineno)
    line_nc_durations: dict[tuple[str, int], float] = defaultdict(float)
    line_nrt_durations: dict[tuple[str, int], float] = defaultdict(float)
    line_counts: dict[tuple[str, int], int] = defaultdict(int)
    line_kernel_names: dict[tuple[str, int], str] = {}

    for i, kc in enumerate(profile.kernel_calls):
        nc_dur = nc_durations[i] if i < len(nc_durations) else 0.0
        nrt_dur = nrt_durations[i] if i < len(nrt_durations) else 0.0
        key = (kc.filename, kc.lineno)
        line_nc_durations[key] += nc_dur
        line_nrt_durations[key] += nrt_dur
        line_counts[key] += 1
        if key not in line_kernel_names:
            line_kernel_names[key] = kc.kernel_name

    total_nc_time = sum(nc_durations) if nc_durations else 0.0
    total_nrt_time = sum(nrt_durations) if nrt_durations else 0.0

    # CPU sample overlap: convert NC host intervals to seconds
    nc_intervals_sec: list[tuple[float, float]] = []
    for start_ns, stop_ns in timeline.nc_host_intervals_ns:
        nc_intervals_sec.append((start_ns / 1e9, stop_ns / 1e9))

    start_time_absolute = scalene_data.get("start_time_absolute", 0.0)
    start_time_perf = scalene_data.get("start_time_perf", 0.0)

    # Inject into scalene JSON, creating missing file/line entries as needed.
    if "files" not in scalene_data:
        scalene_data["files"] = {}

    scalene_filenames = list(scalene_data["files"].keys())

    # Process CPU sample overlap for existing lines
    if nc_intervals_sec and start_time_absolute and start_time_perf:
        for filename, file_data in scalene_data["files"].items():
            for line_data in file_data.get("lines", []):
                cpu_samples = line_data.get("cpu_samples_list", [])
                if cpu_samples:
                    overlap_pct = _calculate_cpu_sample_overlap(
                        cpu_samples,
                        nc_intervals_sec,
                        start_time_absolute,
                        start_time_perf,
                    )
                    line_data["cpu_samples_nc_overlap_percent"] = overlap_pct

    for kernel_filename, lineno in line_nc_durations:
        matched_file = _match_filename(kernel_filename, scalene_filenames)

        if matched_file is None:
            matched_file = kernel_filename
            scalene_data["files"][matched_file] = {
                "lines": [],
                **copy.deepcopy(_FILE_DEFAULTS),
            }
            scalene_filenames.append(matched_file)

        file_data = scalene_data["files"][matched_file]
        if "lines" not in file_data:
            file_data["lines"] = []
        # Ensure file-level defaults exist
        for k, v in _FILE_DEFAULTS.items():
            if k not in file_data:
                file_data[k] = copy.deepcopy(v)

        # Find existing line entry or create one with GUI-required defaults
        line_data = None
        for ld in file_data["lines"]:
            if ld.get("lineno") == lineno:
                line_data = ld
                break
        if line_data is None:
            source_line = linecache.getline(kernel_filename, lineno).rstrip()
            line_data = {
                "lineno": lineno,
                **copy.deepcopy(_LINE_DEFAULTS),
                "line": source_line,
            }
            file_data["lines"].append(line_data)

        key = (kernel_filename, lineno)
        nc_ms = line_nc_durations[key]
        nrt_ms = line_nrt_durations.get(key, 0.0)
        nc_percent = (nc_ms / total_nc_time * 100) if total_nc_time > 0 else 0.0
        nrt_percent = (nrt_ms / total_nrt_time * 100) if total_nrt_time > 0 else 0.0

        line_data["nc_time_ms"] = nc_ms
        line_data["nc_percent"] = nc_percent
        line_data["nc_execute_count"] = line_counts[key]
        line_data["nrt_time_ms"] = nrt_ms
        line_data["nrt_percent"] = nrt_percent
        if nrt_ms > 0:
            line_data["nc_nrt_ratio"] = nc_ms / nrt_ms

    # Per-kernel-name aggregation -> function entries in scalene JSON
    kernel_agg: dict[str, dict] = defaultdict(
        lambda: {"nc_time_ms": 0.0, "nrt_time_ms": 0.0, "count": 0, "filename": ""}
    )
    for i, kc in enumerate(profile.kernel_calls):
        nc_dur = nc_durations[i] if i < len(nc_durations) else 0.0
        nrt_dur = nrt_durations[i] if i < len(nrt_durations) else 0.0
        agg = kernel_agg[kc.kernel_name]
        agg["nc_time_ms"] += nc_dur
        agg["nrt_time_ms"] += nrt_dur
        agg["count"] += 1
        if not agg["filename"]:
            agg["filename"] = kc.filename

    for kernel_name, agg in kernel_agg.items():
        filename = agg["filename"]
        matched_file = _match_filename(filename, scalene_filenames)
        if matched_file is None:
            matched_file = filename
            if matched_file not in scalene_data["files"]:
                scalene_data["files"][matched_file] = {
                    "lines": [],
                    **copy.deepcopy(_FILE_DEFAULTS),
                }
                scalene_filenames.append(matched_file)

        file_data = scalene_data["files"][matched_file]
        if "functions" not in file_data:
            file_data["functions"] = []

        nc_pct = (agg["nc_time_ms"] / total_nc_time * 100) if total_nc_time > 0 else 0.0
        nrt_pct = (
            (agg["nrt_time_ms"] / total_nrt_time * 100) if total_nrt_time > 0 else 0.0
        )
        func_entry = {
            **copy.deepcopy(_LINE_DEFAULTS),
            "line": kernel_name,
            "nc_time_ms": agg["nc_time_ms"],
            "nc_percent": nc_pct,
            "nc_execute_count": agg["count"],
            "nrt_time_ms": agg["nrt_time_ms"],
            "nrt_percent": nrt_pct,
        }
        if agg["nrt_time_ms"] > 0:
            func_entry["nc_nrt_ratio"] = agg["nc_time_ms"] / agg["nrt_time_ms"]
        file_data["functions"].append(func_entry)

    # Top-level metadata
    scalene_data["neuron_total_nc_time_ms"] = total_nc_time
    scalene_data["neuron_total_time_ms"] = total_nrt_time
    scalene_data["neuron_nc_event_count"] = len(nc_durations)
    scalene_data["neuron_event_count"] = len(nrt_durations)

    Path(output_path).write_text(json.dumps(scalene_data, indent=2))
    print(f"Merged profile written to {output_path}")
    print(
        f"Total NRT time: {total_nrt_time:.2f}ms, NC time: {total_nc_time:.2f}ms "
        f"({len(nc_durations)} events)"
    )

    # Print utilization summary
    _print_utilization_summary(
        profile,
        timeline,
        total_nc_time,
        total_nrt_time,
        line_nc_durations,
        line_nrt_durations,
        line_counts,
        line_kernel_names,
    )


def merge_kernel_only(
    kernel_profile_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """Merge one or more kernel profiles without a scalene CPU profile.

    Creates a scalene-compatible JSON with device timing only.

    Args:
        kernel_profile_paths: Paths to KernelProfileResult JSON files.
        output_path: Where to write the merged JSON.
    """
    scalene_data: dict = {"files": {}}
    all_summaries: list[dict] = []

    for rank_idx, kp_path in enumerate(kernel_profile_paths):
        profile = KernelProfileResult.load(kp_path)
        timeline = _parse_trace_timeline(profile.events_json)
        nc_durations = timeline.nc_durations_ms
        nrt_durations = timeline.nrt_durations_ms

        total_nc_time = sum(nc_durations) if nc_durations else 0.0
        total_nrt_time = sum(nrt_durations) if nrt_durations else 0.0
        wall_ms = (profile.wall_stop_ns - profile.wall_start_ns) / 1_000_000.0
        util_pct = (total_nc_time / wall_ms * 100) if wall_ms > 0 else 0.0

        all_summaries.append(
            {
                "rank": rank_idx,
                "total_nc_ms": total_nc_time,
                "total_nrt_ms": total_nrt_time,
                "wall_ms": wall_ms,
                "util_pct": util_pct,
                "n_events": len(nc_durations),
                "timeline": timeline,
                "profile": profile,
            }
        )

        # Aggregate per (filename, lineno)
        line_nc_durations: dict[tuple[str, int], float] = defaultdict(float)
        line_nrt_durations: dict[tuple[str, int], float] = defaultdict(float)
        line_counts: dict[tuple[str, int], int] = defaultdict(int)
        line_kernel_names: dict[tuple[str, int], str] = {}

        for i, kc in enumerate(profile.kernel_calls):
            nc_dur = nc_durations[i] if i < len(nc_durations) else 0.0
            nrt_dur = nrt_durations[i] if i < len(nrt_durations) else 0.0
            key = (kc.filename, kc.lineno)
            line_nc_durations[key] += nc_dur
            line_nrt_durations[key] += nrt_dur
            line_counts[key] += 1
            if key not in line_kernel_names:
                line_kernel_names[key] = kc.kernel_name

        scalene_filenames = list(scalene_data["files"].keys())

        for kernel_filename, lineno in line_nc_durations:
            matched_file = _match_filename(kernel_filename, scalene_filenames)

            if matched_file is None:
                matched_file = kernel_filename
                if matched_file not in scalene_data["files"]:
                    scalene_data["files"][matched_file] = {
                        "lines": [],
                        **copy.deepcopy(_FILE_DEFAULTS),
                    }
                    scalene_filenames.append(matched_file)

            file_data = scalene_data["files"][matched_file]
            if "lines" not in file_data:
                file_data["lines"] = []
            for k, v in _FILE_DEFAULTS.items():
                if k not in file_data:
                    file_data[k] = copy.deepcopy(v)

            line_data = None
            for ld in file_data["lines"]:
                if ld.get("lineno") == lineno:
                    line_data = ld
                    break
            if line_data is None:
                source_line = linecache.getline(kernel_filename, lineno).rstrip()
                line_data = {
                    "lineno": lineno,
                    **copy.deepcopy(_LINE_DEFAULTS),
                    "line": source_line,
                }
                file_data["lines"].append(line_data)

            key = (kernel_filename, lineno)
            nc_ms = line_nc_durations[key]
            nrt_ms = line_nrt_durations.get(key, 0.0)
            nc_percent = (nc_ms / total_nc_time * 100) if total_nc_time > 0 else 0.0
            nrt_percent = (nrt_ms / total_nrt_time * 100) if total_nrt_time > 0 else 0.0

            # Accumulate across ranks
            line_data["nc_time_ms"] = line_data.get("nc_time_ms", 0.0) + nc_ms
            line_data["nc_percent"] = nc_percent
            line_data["nc_execute_count"] = (
                line_data.get("nc_execute_count", 0) + line_counts[key]
            )
            line_data["nrt_time_ms"] = line_data.get("nrt_time_ms", 0.0) + nrt_ms
            line_data["nrt_percent"] = nrt_percent
            if nrt_ms > 0:
                line_data["nc_nrt_ratio"] = nc_ms / nrt_ms

    # Global metadata
    total_nc_all = sum(s["total_nc_ms"] for s in all_summaries)
    total_nrt_all = sum(s["total_nrt_ms"] for s in all_summaries)
    total_events_all = sum(s["n_events"] for s in all_summaries)
    max_wall_ms = max(s["wall_ms"] for s in all_summaries) if all_summaries else 0.0
    scalene_data["neuron_total_nc_time_ms"] = total_nc_all
    scalene_data["neuron_total_time_ms"] = total_nrt_all
    scalene_data["neuron_nc_event_count"] = total_events_all

    # Scalene GUI required top-level fields
    scalene_data.setdefault("elapsed_time_sec", max_wall_ms / 1000.0)
    scalene_data.setdefault("gpu", False)
    scalene_data.setdefault("gpu_device", "")
    scalene_data.setdefault("memory", False)
    scalene_data.setdefault("max_footprint_mb", 0)
    scalene_data.setdefault("samples", [])
    scalene_data.setdefault("growth_rate", 0.0)

    Path(output_path).write_text(json.dumps(scalene_data, indent=2))

    n_ranks = len(all_summaries)
    if n_ranks > 1:
        print(f"\n=== Device Profiling Summary ({n_ranks} ranks) ===")
        for s in all_summaries:
            print(
                f"Rank {s['rank']}: {s['total_nc_ms']:.1f}ms NC "
                f"({s['util_pct']:.1f}% util)"
            )
    else:
        s = all_summaries[0]
        _print_utilization_summary(
            s["profile"],
            s["timeline"],
            s["total_nc_ms"],
            s["total_nrt_ms"],
            line_nc_durations,
            line_nrt_durations,
            line_counts,
            line_kernel_names,
        )

    print(f"\nMerged profile written to {output_path}")
