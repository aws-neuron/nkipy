"""Trace timeline parsing for NRT system trace events."""

import json
from dataclasses import dataclass, field


@dataclass
class TraceTimeline:
    """Parsed timeline from a device trace, including gap analysis.

    Attributes:
        nc_durations_ms: Device-clock durations for each nc_exec_running event.
        nc_host_intervals_ns: Host-clock (start_ns, stop_ns) for nc_exec_running.
        nrt_durations_ms: Host-clock durations for each nrt_execute event.
        nrt_host_intervals_ns: Host-clock (start_ns, stop_ns) for nrt_execute.
        wall_span_ns: (first_event_ts, last_event_ts) across all events.
        gap_breakdown_ms: Category -> total ms during NC-idle gaps.
    """

    nc_durations_ms: list[float] = field(default_factory=list)
    nc_host_intervals_ns: list[tuple[int, int]] = field(default_factory=list)
    nrt_durations_ms: list[float] = field(default_factory=list)
    nrt_host_intervals_ns: list[tuple[int, int]] = field(default_factory=list)
    wall_span_ns: tuple[int, int] = (0, 0)
    gap_breakdown_ms: dict[str, float] = field(default_factory=dict)


# Mapping from event_type to human-readable gap category
_GAP_CATEGORIES: dict[str, str] = {
    # D2H transfers
    "nrt_tensor_read": "d2h_transfer",
    "dmem_buf_copyout": "d2h_transfer",
    # H2D transfers
    "nrt_tensor_write": "h2d_transfer",
    "dmem_buf_copyin": "h2d_transfer",
    # Memory management
    "nrt_tensor_allocate": "memory_mgmt",
    "nrt_tensor_free": "memory_mgmt",
    "nrt_dma_mem_alloc": "memory_mgmt",
    "nrt_dma_mem_dealloc": "memory_mgmt",
    # Exec overhead
    "nrt_execute": "exec_overhead",
    "nrt_model_submit": "exec_overhead",
    "kbl_exec_pre": "exec_overhead",
    "kbl_exec_post": "exec_overhead",
    # Sync waits
    "nrt_async_sema_wait": "sync_wait",
    "tensor_block_while_exec": "sync_wait",
}


def _parse_trace_timeline(events_json: str) -> TraceTimeline:
    """Parse a full trace timeline with gap analysis from NRT sys trace JSON.

    Parses nc_exec_running events for device durations and host intervals,
    then analyzes gaps between consecutive NC executions to determine what
    the device was doing while idle (transfers, memory ops, sync waits, etc.).

    Returns:
        TraceTimeline with durations, host intervals, wall span, and gap breakdown.
    """
    if not events_json:
        return TraceTimeline()

    root = json.loads(events_json)
    events = root.get("events", [])

    if not events:
        return TraceTimeline()

    # Collect start/stop pairs for all event types using tracking_id + timestamp_ns
    # Key: (event_type, tracking_id) -> start timestamp_ns (host clock)
    starts: dict[tuple[str, int], int] = {}
    # Intervals per event type: event_type -> list of (start_ns, stop_ns)
    type_intervals: dict[str, list[tuple[int, int]]] = {}

    # NC-specific: device clock durations
    nc_device_starts: dict[int, int] = {}  # tracking_id -> nc_timestamp_ns
    nc_durations_ms: list[float] = []
    nc_host_intervals_ns: list[tuple[int, int]] = []

    # NRT-specific: host clock durations for nrt_execute
    nrt_durations_ms: list[float] = []
    nrt_host_intervals_ns: list[tuple[int, int]] = []

    # Track wall span
    min_ts = float("inf")
    max_ts = float("-inf")

    for event in events:
        event_type = event.get("event_type")
        phase = event.get("phase")
        tracking_id = event.get("tracking_id")
        ts_ns = event.get("timestamp_ns")
        data = event.get("data", {})

        if tracking_id is None:
            continue

        # Track wall span across all events with host timestamps
        if ts_ns is not None:
            if ts_ns < min_ts:
                min_ts = ts_ns
            if ts_ns > max_ts:
                max_ts = ts_ns

        # NC exec running: collect both device-clock durations and host intervals
        if event_type == "nc_exec_running":
            nc_ts = data.get("nc_timestamp_ns")
            if phase == "start":
                if nc_ts is not None:
                    nc_device_starts[tracking_id] = nc_ts
                if ts_ns is not None:
                    starts[("nc_exec_running", tracking_id)] = ts_ns
            elif phase == "stop":
                # Device clock duration
                if nc_ts is not None:
                    start_nc = nc_device_starts.pop(tracking_id, None)
                    if start_nc is not None:
                        nc_durations_ms.append((nc_ts - start_nc) / 1_000_000.0)
                # Host clock interval
                if ts_ns is not None:
                    start_host = starts.pop(("nc_exec_running", tracking_id), None)
                    if start_host is not None:
                        nc_host_intervals_ns.append((start_host, ts_ns))
            continue

        # nrt_execute: collect per-call host-clock durations AND feed into
        # gap analysis (exec_overhead category). Does not have device clock.
        if event_type == "nrt_execute":
            if phase == "start" and ts_ns is not None:
                starts[("nrt_execute", tracking_id)] = ts_ns
            elif phase == "stop" and ts_ns is not None:
                start_ts = starts.pop(("nrt_execute", tracking_id), None)
                if start_ts is not None:
                    nrt_durations_ms.append((ts_ns - start_ts) / 1_000_000.0)
                    nrt_host_intervals_ns.append((start_ts, ts_ns))
                    # Also record for gap analysis
                    type_intervals.setdefault("nrt_execute", []).append(
                        (start_ts, ts_ns)
                    )
            continue

        # All other categorized event types: collect host-clock intervals
        if event_type in _GAP_CATEGORIES:
            if phase == "start" and ts_ns is not None:
                starts[(event_type, tracking_id)] = ts_ns
            elif phase == "stop" and ts_ns is not None:
                start_ts = starts.pop((event_type, tracking_id), None)
                if start_ts is not None:
                    type_intervals.setdefault(event_type, []).append((start_ts, ts_ns))

    # Compute wall span
    if min_ts == float("inf"):
        wall_span_ns = (0, 0)
    else:
        wall_span_ns = (int(min_ts), int(max_ts))

    # Sort NC host intervals by start time
    nc_host_intervals_ns.sort(key=lambda x: x[0])

    # Compute gaps between consecutive NC executions
    gap_breakdown_ms: dict[str, float] = {}
    if len(nc_host_intervals_ns) >= 2:
        gaps: list[tuple[int, int]] = []
        for i in range(len(nc_host_intervals_ns) - 1):
            gap_start = nc_host_intervals_ns[i][1]
            gap_end = nc_host_intervals_ns[i + 1][0]
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end))

        # For each gap, check which other event intervals overlap and attribute time
        for gap_start, gap_end in gaps:
            for event_type, intervals in type_intervals.items():
                category = _GAP_CATEGORIES[event_type]
                for iv_start, iv_end in intervals:
                    # Compute overlap with this gap
                    overlap_start = max(gap_start, iv_start)
                    overlap_end = min(gap_end, iv_end)
                    if overlap_end > overlap_start:
                        overlap_ms = (overlap_end - overlap_start) / 1_000_000.0
                        gap_breakdown_ms[category] = (
                            gap_breakdown_ms.get(category, 0.0) + overlap_ms
                        )

    # Sort NRT host intervals by start time
    nrt_host_intervals_ns.sort(key=lambda x: x[0])

    return TraceTimeline(
        nc_durations_ms=nc_durations_ms,
        nc_host_intervals_ns=nc_host_intervals_ns,
        nrt_durations_ms=nrt_durations_ms,
        nrt_host_intervals_ns=nrt_host_intervals_ns,
        wall_span_ns=wall_span_ns,
        gap_breakdown_ms=gap_breakdown_ms,
    )
