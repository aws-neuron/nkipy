"""Tests for KernelProfiler and merge_profiles."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nkipy.tools.profiler.kernel_profiler import (
    KernelExecution,
    KernelProfiler,
    KernelProfileResult,
    _find_user_frame,
    _get_current_rank,
)
from nkipy.tools.profiler.merge_profiles import (
    _calculate_cpu_sample_overlap,
    merge_kernel_only,
    merge_scalene_and_kernel_profiles,
)
from nkipy.tools.profiler.trace_timeline import TraceTimeline, _parse_trace_timeline


# --- Helpers for building synthetic trace events ---


def _make_nc_exec_events(
    durations_ms: list[float],
    base_host_ns: int = 1_000_000_000,
    base_device_ns: int = 1_000_000_000,
    gap_ns: int = 1_000_000,
) -> list[dict]:
    """Build synthetic nc_exec_running start/stop event pairs."""
    events = []
    host_ts = base_host_ns
    device_ts = base_device_ns
    for tid, dur_ms in enumerate(durations_ms):
        dur_ns = int(dur_ms * 1_000_000)
        events.append(
            {
                "event_type": "nc_exec_running",
                "phase": "start",
                "tracking_id": tid,
                "timestamp_ns": host_ts,
                "data": {"nc_timestamp_ns": device_ts},
            }
        )
        events.append(
            {
                "event_type": "nc_exec_running",
                "phase": "stop",
                "tracking_id": tid,
                "timestamp_ns": host_ts + dur_ns,
                "data": {"nc_timestamp_ns": device_ts + dur_ns},
            }
        )
        host_ts += dur_ns + gap_ns
        device_ts += dur_ns + gap_ns
    return events


def _make_nrt_execute_events(
    durations_ms: list[float],
    base_host_ns: int = 1_000_000_000,
    gap_ns: int = 1_000_000,
    base_tracking_id: int = 5000,
) -> list[dict]:
    """Build synthetic nrt_execute start/stop event pairs (host clock only)."""
    events = []
    host_ts = base_host_ns
    for i, dur_ms in enumerate(durations_ms):
        tid = base_tracking_id + i
        dur_ns = int(dur_ms * 1_000_000)
        events.append(
            {
                "event_type": "nrt_execute",
                "phase": "start",
                "tracking_id": tid,
                "timestamp_ns": host_ts,
                "data": {},
            }
        )
        events.append(
            {
                "event_type": "nrt_execute",
                "phase": "stop",
                "tracking_id": tid,
                "timestamp_ns": host_ts + dur_ns,
                "data": {},
            }
        )
        host_ts += dur_ns + gap_ns
    return events


def _make_categorized_events(
    event_type: str,
    intervals_ns: list[tuple[int, int]],
    base_tracking_id: int = 1000,
) -> list[dict]:
    """Build synthetic start/stop pairs for a categorized event type."""
    events = []
    for i, (start_ns, stop_ns) in enumerate(intervals_ns):
        tid = base_tracking_id + i
        events.append(
            {
                "event_type": event_type,
                "phase": "start",
                "tracking_id": tid,
                "timestamp_ns": start_ns,
                "data": {},
            }
        )
        events.append(
            {
                "event_type": event_type,
                "phase": "stop",
                "tracking_id": tid,
                "timestamp_ns": stop_ns,
                "data": {},
            }
        )
    return events


def _events_json(events: list[dict]) -> str:
    return json.dumps({"events": events})


# --- TraceTimeline parsing ---


class TestParseTraceTimeline:
    def test_empty_input(self):
        tl = _parse_trace_timeline("")
        assert tl.nc_durations_ms == []
        assert tl.nc_host_intervals_ns == []
        assert tl.nrt_durations_ms == []
        assert tl.nrt_host_intervals_ns == []
        assert tl.wall_span_ns == (0, 0)
        assert tl.gap_breakdown_ms == {}

    def test_empty_events(self):
        tl = _parse_trace_timeline('{"events": []}')
        assert tl.nc_durations_ms == []
        assert tl.nrt_durations_ms == []

    def test_nc_durations_and_host_intervals(self):
        events = _make_nc_exec_events([5.0, 3.0])
        tl = _parse_trace_timeline(_events_json(events))

        assert len(tl.nc_durations_ms) == 2
        assert tl.nc_durations_ms[0] == pytest.approx(5.0)
        assert tl.nc_durations_ms[1] == pytest.approx(3.0)

        assert len(tl.nc_host_intervals_ns) == 2
        # Intervals should be sorted by start
        assert tl.nc_host_intervals_ns[0][0] < tl.nc_host_intervals_ns[1][0]

    def test_nrt_execute_parsed_separately(self):
        """nrt_execute events produce separate durations from nc_exec_running."""
        nc_events = _make_nc_exec_events([5.0, 3.0])
        # nrt_execute is shorter (host-side async submit)
        nrt_events = _make_nrt_execute_events([0.1, 0.08])
        tl = _parse_trace_timeline(_events_json(nc_events + nrt_events))

        assert len(tl.nc_durations_ms) == 2
        assert len(tl.nrt_durations_ms) == 2
        assert tl.nrt_durations_ms[0] == pytest.approx(0.1)
        assert tl.nrt_durations_ms[1] == pytest.approx(0.08)
        assert len(tl.nrt_host_intervals_ns) == 2

    def test_nrt_execute_in_gap_analysis(self):
        """nrt_execute events also feed into gap breakdown as exec_overhead."""
        # NC: 0-5ms, gap 5-10ms, NC: 10-15ms
        nc_events = _make_nc_exec_events(
            [5.0, 5.0],
            base_host_ns=0,
            gap_ns=5_000_000,
        )
        # nrt_execute during the gap: 6-7ms
        nrt_events = _make_nrt_execute_events(
            [1.0],
            base_host_ns=6_000_000,
            base_tracking_id=9000,
        )
        tl = _parse_trace_timeline(_events_json(nc_events + nrt_events))
        assert "exec_overhead" in tl.gap_breakdown_ms
        assert tl.gap_breakdown_ms["exec_overhead"] == pytest.approx(1.0)

    def test_wall_span(self):
        events = _make_nc_exec_events([10.0], base_host_ns=500)
        tl = _parse_trace_timeline(_events_json(events))

        assert tl.wall_span_ns[0] == 500
        assert tl.wall_span_ns[1] == 500 + 10_000_000  # 10ms in ns

    def test_gap_breakdown_with_transfer(self):
        """Transfer event during NC gap gets attributed correctly."""
        # NC exec: 0-5ms, then 10-15ms (gap from 5ms to 10ms)
        nc_events = _make_nc_exec_events(
            [5.0, 5.0],
            base_host_ns=0,
            gap_ns=5_000_000,  # 5ms gap
        )

        # D2H transfer during the gap: 5.5ms to 8ms (host clock)
        d2h_events = _make_categorized_events(
            "nrt_tensor_read",
            [(5_500_000, 8_000_000)],  # 2.5ms within the 5ms gap
        )

        tl = _parse_trace_timeline(_events_json(nc_events + d2h_events))

        assert "d2h_transfer" in tl.gap_breakdown_ms
        assert tl.gap_breakdown_ms["d2h_transfer"] == pytest.approx(2.5)

    def test_gap_breakdown_multiple_categories(self):
        """Multiple event types in a single gap."""
        # NC: 0-2ms, gap 2-7ms, NC: 7-9ms
        nc_events = _make_nc_exec_events(
            [2.0, 2.0],
            base_host_ns=0,
            gap_ns=5_000_000,  # 5ms gap
        )

        # H2D during gap: 2.5-4ms (1.5ms)
        h2d = _make_categorized_events(
            "nrt_tensor_write",
            [(2_500_000, 4_000_000)],
            base_tracking_id=100,
        )
        # Sync wait during gap: 4-6ms (2ms)
        sync = _make_categorized_events(
            "nrt_async_sema_wait",
            [(4_000_000, 6_000_000)],
            base_tracking_id=200,
        )

        tl = _parse_trace_timeline(_events_json(nc_events + h2d + sync))

        assert tl.gap_breakdown_ms.get("h2d_transfer", 0.0) == pytest.approx(1.5)
        assert tl.gap_breakdown_ms.get("sync_wait", 0.0) == pytest.approx(2.0)

    def test_event_outside_gap_not_attributed(self):
        """Events that don't overlap NC gaps are not counted."""
        # NC: 0-10ms (single long exec, no gap)
        nc_events = _make_nc_exec_events([10.0], base_host_ns=0)

        # D2H happening during NC exec (not a gap)
        d2h = _make_categorized_events(
            "nrt_tensor_read",
            [(2_000_000, 5_000_000)],
        )

        tl = _parse_trace_timeline(_events_json(nc_events + d2h))

        # No gaps exist (single NC execution), so no breakdown
        assert tl.gap_breakdown_ms == {}

    def test_partial_overlap_with_gap(self):
        """Event that partially overlaps a gap only counts the overlapping portion."""
        # NC: 0-5ms, gap 5-10ms, NC: 10-15ms
        nc_events = _make_nc_exec_events(
            [5.0, 5.0],
            base_host_ns=0,
            gap_ns=5_000_000,
        )

        # Memory event spans 3-7ms (overlaps gap from 5-7ms = 2ms)
        mem = _make_categorized_events(
            "nrt_tensor_allocate",
            [(3_000_000, 7_000_000)],
        )

        tl = _parse_trace_timeline(_events_json(nc_events + mem))
        assert tl.gap_breakdown_ms.get("memory_mgmt", 0.0) == pytest.approx(2.0)


# --- KernelProfileResult save/load roundtrip ---


class TestKernelProfileResult:
    def test_save_load_roundtrip(self, tmp_path):
        original = KernelProfileResult(
            kernel_calls=[
                KernelExecution(
                    filename="/home/user/model.py",
                    lineno=42,
                    kernel_name="attention",
                    call_index=0,
                ),
                KernelExecution(
                    filename="/home/user/model.py",
                    lineno=55,
                    kernel_name="ffn",
                    call_index=1,
                ),
            ],
            events_json='{"events": []}',
            wall_start_ns=1000000000,
            wall_stop_ns=2000000000,
        )

        path = tmp_path / "profile.json"
        original.save(path)

        loaded = KernelProfileResult.load(path)
        assert len(loaded.kernel_calls) == 2
        assert loaded.kernel_calls[0].filename == "/home/user/model.py"
        assert loaded.kernel_calls[0].lineno == 42
        assert loaded.kernel_calls[0].kernel_name == "attention"
        assert loaded.kernel_calls[0].call_index == 0
        assert loaded.kernel_calls[1].lineno == 55
        assert loaded.kernel_calls[1].kernel_name == "ffn"
        assert loaded.events_json == '{"events": []}'
        assert loaded.wall_start_ns == 1000000000
        assert loaded.wall_stop_ns == 2000000000

    def test_save_load_empty(self, tmp_path):
        original = KernelProfileResult()
        path = tmp_path / "empty.json"
        original.save(path)

        loaded = KernelProfileResult.load(path)
        assert loaded.kernel_calls == []
        assert loaded.events_json == ""
        assert loaded.wall_start_ns == 0
        assert loaded.wall_stop_ns == 0

    def test_load_legacy_format(self, tmp_path):
        """Loading a file without wall_start_ns/wall_stop_ns defaults to 0."""
        data = {
            "kernel_calls": [],
            "events_json": "",
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(data))

        loaded = KernelProfileResult.load(path)
        assert loaded.wall_start_ns == 0
        assert loaded.wall_stop_ns == 0


# --- KernelProfiler monkey-patching ---


class TestKernelProfiler:
    def test_monkey_patch_records_calls(self, tmp_path):
        """Verify monkey-patching records kernel name and restores on exit."""
        from spike.spike_model import SpikeModel

        original_call = SpikeModel.__call__

        output_path = tmp_path / "kp.json"

        # Mock SystemTraceSession to avoid needing hardware
        mock_trace = MagicMock()
        mock_trace.__enter__ = MagicMock(return_value=mock_trace)
        mock_trace.__exit__ = MagicMock(return_value=False)
        mock_trace.fetch_events_json = MagicMock(return_value='{"events": []}')

        # Also mock the original __call__ so we don't need a real model
        with (
            patch(
                "nkipy.tools.profiler.kernel_profiler.SystemTraceSession",
                return_value=mock_trace,
            ),
            patch.object(SpikeModel, "__call__", return_value=None) as mock_call,
        ):
            profiler = KernelProfiler(core_id=0, scalene=False, output_path=output_path)
            with profiler:
                # SpikeModel.__call__ should be patched (not the mock we set)
                assert SpikeModel.__call__ is not mock_call

                # Simulate kernel calls via the monkey-patched __call__
                mock_model = MagicMock()
                mock_model.name = "test_kernel"
                SpikeModel.__call__(mock_model, inputs={}, outputs={})

            # After exit, __call__ should be restored to the mock
            assert SpikeModel.__call__ is mock_call

        # Check recorded calls
        result = profiler.result
        assert len(result.kernel_calls) == 1
        assert result.kernel_calls[0].kernel_name == "test_kernel"
        assert result.kernel_calls[0].call_index == 0
        # Source location should point to this test file
        assert "test_kernel_profiler.py" in result.kernel_calls[0].filename

    def test_output_saved_to_disk(self, tmp_path):
        """Verify output JSON file is written on exit."""
        output_path = tmp_path / "kp.json"

        mock_trace = MagicMock()
        mock_trace.__enter__ = MagicMock(return_value=mock_trace)
        mock_trace.__exit__ = MagicMock(return_value=False)
        mock_trace.fetch_events_json = MagicMock(return_value='{"events": []}')

        with patch(
            "nkipy.tools.profiler.kernel_profiler.SystemTraceSession",
            return_value=mock_trace,
        ):
            with KernelProfiler(core_id=0, scalene=False, output_path=output_path):
                pass

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "kernel_calls" in data
        assert "events_json" in data
        assert "wall_start_ns" in data
        assert "wall_stop_ns" in data
        assert data["wall_start_ns"] > 0
        assert data["wall_stop_ns"] >= data["wall_start_ns"]

    def test_scalene_import_failure_is_silent(self, tmp_path):
        """When scalene is not installed, profiler should still work."""
        output_path = tmp_path / "kp.json"

        mock_trace = MagicMock()
        mock_trace.__enter__ = MagicMock(return_value=mock_trace)
        mock_trace.__exit__ = MagicMock(return_value=False)
        mock_trace.fetch_events_json = MagicMock(return_value='{"events": []}')

        with patch(
            "nkipy.tools.profiler.kernel_profiler.SystemTraceSession",
            return_value=mock_trace,
        ):
            # scalene=True but import will fail in test environment
            with KernelProfiler(core_id=0, scalene=True, output_path=output_path):
                pass

        assert output_path.exists()


# --- Rank filtering ---


class TestRankFiltering:
    def test_get_current_rank_no_dist(self):
        """Without torch.distributed, rank is 0."""
        assert _get_current_rank() == 0

    def test_non_target_rank_is_noop(self, tmp_path):
        """Profiler is a no-op on non-target ranks."""
        output_path = tmp_path / "kp.json"

        with patch(
            "nkipy.tools.profiler.kernel_profiler._get_current_rank", return_value=1
        ):
            profiler = KernelProfiler(
                core_id=0,
                scalene=False,
                output_path=output_path,
                target_ranks=[0],
            )
            with profiler:
                pass

        # Non-target rank should not write output
        assert not output_path.exists()
        assert profiler.result is None

    def test_target_rank_profiles_normally(self, tmp_path):
        """Profiler works normally on a target rank."""
        output_path = tmp_path / "kp.json"

        mock_trace = MagicMock()
        mock_trace.__enter__ = MagicMock(return_value=mock_trace)
        mock_trace.__exit__ = MagicMock(return_value=False)
        mock_trace.fetch_events_json = MagicMock(return_value='{"events": []}')

        with (
            patch(
                "nkipy.tools.profiler.kernel_profiler._get_current_rank", return_value=0
            ),
            patch(
                "nkipy.tools.profiler.kernel_profiler.SystemTraceSession",
                return_value=mock_trace,
            ),
        ):
            profiler = KernelProfiler(
                core_id=0,
                scalene=False,
                output_path=output_path,
                target_ranks=[0],
            )
            with profiler:
                pass

        assert output_path.exists()
        assert profiler.result is not None

    def test_multi_rank_output_path(self, tmp_path):
        """With multiple target ranks, output includes rank suffix."""
        output_path = tmp_path / "kernel_profile.json"

        mock_trace = MagicMock()
        mock_trace.__enter__ = MagicMock(return_value=mock_trace)
        mock_trace.__exit__ = MagicMock(return_value=False)
        mock_trace.fetch_events_json = MagicMock(return_value='{"events": []}')

        with (
            patch(
                "nkipy.tools.profiler.kernel_profiler._get_current_rank", return_value=2
            ),
            patch(
                "nkipy.tools.profiler.kernel_profiler.SystemTraceSession",
                return_value=mock_trace,
            ),
        ):
            with KernelProfiler(
                core_id=2,
                scalene=False,
                output_path=output_path,
                target_ranks=[0, 1, 2, 3],
            ):
                pass

        expected = tmp_path / "kernel_profile_rank2.json"
        assert expected.exists()

    def test_no_target_ranks_profiles_all(self, tmp_path):
        """target_ranks=None means profile on any rank (default behavior)."""
        output_path = tmp_path / "kp.json"

        mock_trace = MagicMock()
        mock_trace.__enter__ = MagicMock(return_value=mock_trace)
        mock_trace.__exit__ = MagicMock(return_value=False)
        mock_trace.fetch_events_json = MagicMock(return_value='{"events": []}')

        with (
            patch(
                "nkipy.tools.profiler.kernel_profiler._get_current_rank", return_value=5
            ),
            patch(
                "nkipy.tools.profiler.kernel_profiler.SystemTraceSession",
                return_value=mock_trace,
            ),
        ):
            profiler = KernelProfiler(
                core_id=5,
                scalene=False,
                output_path=output_path,
                target_ranks=None,
            )
            with profiler:
                pass

        assert output_path.exists()


# --- CPU sample overlap ---


class TestCPUSampleOverlap:
    def test_full_overlap(self):
        """All samples fall within NC intervals."""
        samples = [10.0, 11.0, 12.0]
        # NC interval covers 9-13 seconds absolute
        nc_intervals = [(9.0, 13.0)]
        pct = _calculate_cpu_sample_overlap(
            samples,
            nc_intervals,
            start_time_absolute=0.0,
            start_time_perf=0.0,
        )
        assert pct == pytest.approx(100.0)

    def test_no_overlap(self):
        """No samples overlap NC intervals."""
        samples = [1.0, 2.0, 3.0]
        nc_intervals = [(10.0, 20.0)]
        pct = _calculate_cpu_sample_overlap(
            samples,
            nc_intervals,
            start_time_absolute=0.0,
            start_time_perf=0.0,
        )
        assert pct == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Some samples overlap, some don't."""
        samples = [1.0, 5.0, 15.0, 25.0]
        nc_intervals = [(4.0, 6.0), (14.0, 16.0)]
        pct = _calculate_cpu_sample_overlap(
            samples,
            nc_intervals,
            start_time_absolute=0.0,
            start_time_perf=0.0,
        )
        assert pct == pytest.approx(50.0)

    def test_perf_to_absolute_conversion(self):
        """Verify perf_counter -> absolute time conversion."""
        # perf starts at 100, absolute starts at 1000
        samples = [105.0, 110.0]  # absolute: 1005, 1010
        nc_intervals = [(1004.0, 1006.0)]  # covers sample at 1005
        pct = _calculate_cpu_sample_overlap(
            samples,
            nc_intervals,
            start_time_absolute=1000.0,
            start_time_perf=100.0,
        )
        assert pct == pytest.approx(50.0)

    def test_empty_samples(self):
        pct = _calculate_cpu_sample_overlap([], [(1.0, 2.0)], 0.0, 0.0)
        assert pct == pytest.approx(0.0)

    def test_empty_intervals(self):
        pct = _calculate_cpu_sample_overlap([1.0, 2.0], [], 0.0, 0.0)
        assert pct == pytest.approx(0.0)


# --- merge_profiles ---


class TestMergeProfiles:
    def _make_scalene_json(
        self, path: Path, lines: list[dict], filename: str = "model.py"
    ):
        """Create a minimal scalene JSON file."""
        data = {
            "files": {
                filename: {
                    "lines": lines,
                }
            }
        }
        path.write_text(json.dumps(data))

    def _make_kernel_profile(
        self,
        path: Path,
        calls: list[tuple[str, int, str]],
        durations_ms: list[float],
        nrt_durations_ms: list[float] | None = None,
        wall_start_ns: int = 0,
        wall_stop_ns: int = 0,
    ):
        """Create a kernel profile with matching trace events.

        Args:
            nrt_durations_ms: Host-side nrt_execute durations. If None,
                defaults to 20% of each nc duration (simulating async exec).
        """
        kernel_calls = [
            KernelExecution(
                filename=filename, lineno=lineno, kernel_name=name, call_index=i
            )
            for i, (filename, lineno, name) in enumerate(calls)
        ]

        if nrt_durations_ms is None:
            nrt_durations_ms = [d * 0.2 for d in durations_ms]

        # Build synthetic nc_exec_running + nrt_execute events
        nc_events = _make_nc_exec_events(durations_ms)
        nrt_events = _make_nrt_execute_events(nrt_durations_ms)
        events_json = json.dumps({"events": nc_events + nrt_events})

        result = KernelProfileResult(
            kernel_calls=kernel_calls,
            events_json=events_json,
            wall_start_ns=wall_start_ns,
            wall_stop_ns=wall_stop_ns,
        )
        result.save(path)

    def test_basic_merge(self, tmp_path):
        """Verify merge injects nc_time_ms and nrt_time_ms into scalene JSON."""
        scalene_path = tmp_path / "scalene.json"
        kernel_path = tmp_path / "kernel.json"
        output_path = tmp_path / "merged.json"

        # Scalene has two lines of interest
        self._make_scalene_json(
            scalene_path,
            [
                {"lineno": 10, "n_cpu_percent_python": 50.0},
                {"lineno": 20, "n_cpu_percent_python": 30.0},
                {"lineno": 30, "n_cpu_percent_python": 20.0},
            ],
        )

        # Two kernel calls at lines 10 and 20
        # nc=5.0,3.0  nrt=1.0,0.6 (default 20%)
        self._make_kernel_profile(
            kernel_path,
            calls=[
                ("model.py", 10, "attention"),
                ("model.py", 20, "ffn"),
            ],
            durations_ms=[5.0, 3.0],
        )

        merge_scalene_and_kernel_profiles(scalene_path, kernel_path, output_path)

        merged = json.loads(output_path.read_text())

        # Check line 10: nc and nrt are separate
        line10 = merged["files"]["model.py"]["lines"][0]
        assert line10["nc_time_ms"] == pytest.approx(5.0)
        assert line10["nrt_time_ms"] == pytest.approx(1.0)
        assert line10["nc_execute_count"] == 1
        assert line10["nc_nrt_ratio"] == pytest.approx(5.0)

        # Check line 20
        line20 = merged["files"]["model.py"]["lines"][1]
        assert line20["nc_time_ms"] == pytest.approx(3.0)
        assert line20["nrt_time_ms"] == pytest.approx(0.6)

        # Check line 30 has no nc_time_ms
        line30 = merged["files"]["model.py"]["lines"][2]
        assert "nc_time_ms" not in line30

        # Check metadata
        assert merged["neuron_total_nc_time_ms"] == pytest.approx(8.0)
        assert merged["neuron_total_time_ms"] == pytest.approx(1.6)
        assert merged["neuron_nc_event_count"] == 2
        assert merged["neuron_event_count"] == 2

    def test_aggregation_same_line(self, tmp_path):
        """Multiple kernel calls on the same line aggregate durations."""
        scalene_path = tmp_path / "scalene.json"
        kernel_path = tmp_path / "kernel.json"
        output_path = tmp_path / "merged.json"

        self._make_scalene_json(
            scalene_path,
            [{"lineno": 10, "n_cpu_percent_python": 100.0}],
        )

        # Three kernel calls all on line 10
        self._make_kernel_profile(
            kernel_path,
            calls=[
                ("model.py", 10, "layer_kernel"),
                ("model.py", 10, "layer_kernel"),
                ("model.py", 10, "layer_kernel"),
            ],
            durations_ms=[2.0, 3.0, 1.5],
        )

        merge_scalene_and_kernel_profiles(scalene_path, kernel_path, output_path)

        merged = json.loads(output_path.read_text())
        line10 = merged["files"]["model.py"]["lines"][0]
        assert line10["nc_time_ms"] == pytest.approx(6.5)
        assert line10["nc_execute_count"] == 3
        assert line10["nc_percent"] == pytest.approx(100.0)

    def test_html_embedded_scalene(self, tmp_path):
        """Merge handles scalene HTML output with embedded JSON."""
        scalene_path = tmp_path / "scalene.html"
        kernel_path = tmp_path / "kernel.json"
        output_path = tmp_path / "merged.json"

        scalene_json = json.dumps(
            {
                "files": {
                    "model.py": {
                        "lines": [{"lineno": 5, "n_cpu_percent_python": 100.0}]
                    }
                }
            }
        )
        html = f"<!DOCTYPE html><html>const profile = {scalene_json};</html>"
        scalene_path.write_text(html)

        self._make_kernel_profile(
            kernel_path,
            calls=[("model.py", 5, "kernel")],
            durations_ms=[10.0],
        )

        merge_scalene_and_kernel_profiles(scalene_path, kernel_path, output_path)

        merged = json.loads(output_path.read_text())
        assert merged["files"]["model.py"]["lines"][0]["nc_time_ms"] == 10.0

    def test_path_matching_by_basename(self, tmp_path):
        """Kernel paths with different prefixes match by basename."""
        scalene_path = tmp_path / "scalene.json"
        kernel_path = tmp_path / "kernel.json"
        output_path = tmp_path / "merged.json"

        # Scalene uses a different absolute path
        self._make_scalene_json(
            scalene_path,
            [{"lineno": 10, "n_cpu_percent_python": 100.0}],
            filename="/opt/project/model.py",
        )

        # Kernel profile uses yet another path
        self._make_kernel_profile(
            kernel_path,
            calls=[("/home/user/project/model.py", 10, "kernel")],
            durations_ms=[7.0],
        )

        merge_scalene_and_kernel_profiles(scalene_path, kernel_path, output_path)

        merged = json.loads(output_path.read_text())
        line10 = merged["files"]["/opt/project/model.py"]["lines"][0]
        assert line10["nc_time_ms"] == 7.0

    def test_per_kernel_function_entries(self, tmp_path):
        """Per-kernel-name aggregation creates function entries with nrt/nc."""
        scalene_path = tmp_path / "scalene.json"
        kernel_path = tmp_path / "kernel.json"
        output_path = tmp_path / "merged.json"

        self._make_scalene_json(
            scalene_path,
            [{"lineno": 10}, {"lineno": 20}],
        )

        self._make_kernel_profile(
            kernel_path,
            calls=[
                ("model.py", 10, "attention"),
                ("model.py", 10, "attention"),
                ("model.py", 20, "ffn"),
            ],
            durations_ms=[3.0, 4.0, 5.0],
        )

        merge_scalene_and_kernel_profiles(scalene_path, kernel_path, output_path)

        merged = json.loads(output_path.read_text())
        functions = merged["files"]["model.py"]["functions"]
        assert len(functions) == 2

        func_by_name = {f["line"]: f for f in functions}
        assert "attention" in func_by_name
        assert "ffn" in func_by_name

        attn = func_by_name["attention"]
        assert attn["nc_time_ms"] == pytest.approx(7.0)
        assert attn["nc_execute_count"] == 2
        # nrt is 20% of nc by default
        assert attn["nrt_time_ms"] == pytest.approx(1.4)
        assert "nc_nrt_ratio" in attn

        ffn = func_by_name["ffn"]
        assert ffn["nc_time_ms"] == pytest.approx(5.0)
        assert ffn["nc_execute_count"] == 1

    def test_cpu_overlap_injected(self, tmp_path):
        """CPU sample overlap is computed and injected into line data."""
        scalene_path = tmp_path / "scalene.json"
        kernel_path = tmp_path / "kernel.json"
        output_path = tmp_path / "merged.json"

        # Scalene with start times and cpu_samples_list
        data = {
            "start_time_absolute": 1000.0,
            "start_time_perf": 100.0,
            "files": {
                "model.py": {
                    "lines": [
                        {
                            "lineno": 10,
                            "cpu_samples_list": [105.0, 106.0, 107.0, 108.0],
                        },
                    ],
                }
            },
        }
        scalene_path.write_text(json.dumps(data))

        # NC host intervals will be at ~1s = 1_000_000_000ns
        # cpu_samples at absolute 1005, 1006, 1007, 1008 -- these are in seconds
        # NC intervals at 1s in ns -- that's 1.0s in seconds
        # So samples at 1005s won't overlap NC intervals at 1.0s
        # Let's adjust: make NC intervals match sample times
        events = []
        # NC interval from 1004.5s to 1006.5s in ns
        events.append(
            {
                "event_type": "nc_exec_running",
                "phase": "start",
                "tracking_id": 0,
                "timestamp_ns": 1_004_500_000_000,  # 1004.5s
                "data": {"nc_timestamp_ns": 1000},
            }
        )
        events.append(
            {
                "event_type": "nc_exec_running",
                "phase": "stop",
                "tracking_id": 0,
                "timestamp_ns": 1_006_500_000_000,  # 1006.5s
                "data": {"nc_timestamp_ns": 3_000_000},  # 2ms device time
            }
        )

        result = KernelProfileResult(
            kernel_calls=[
                KernelExecution("model.py", 10, "kernel", 0),
            ],
            events_json=json.dumps({"events": events}),
            wall_start_ns=1_004_000_000_000,
            wall_stop_ns=1_007_000_000_000,
        )
        result.save(kernel_path)

        merge_scalene_and_kernel_profiles(scalene_path, kernel_path, output_path)

        merged = json.loads(output_path.read_text())
        line10 = merged["files"]["model.py"]["lines"][0]
        # Samples at abs 1005, 1006 overlap NC [1004.5, 1006.5]
        # Samples at abs 1007, 1008 don't
        assert line10["cpu_samples_nc_overlap_percent"] == pytest.approx(50.0)


# --- merge_kernel_only ---


class TestMergeKernelOnly:
    def _make_kernel_profile(
        self,
        path: Path,
        calls: list[tuple[str, int, str]],
        durations_ms: list[float],
        nrt_durations_ms: list[float] | None = None,
        wall_start_ns: int = 0,
        wall_stop_ns: int = 100_000_000,
    ):
        kernel_calls = [
            KernelExecution(
                filename=filename, lineno=lineno, kernel_name=name, call_index=i
            )
            for i, (filename, lineno, name) in enumerate(calls)
        ]
        if nrt_durations_ms is None:
            nrt_durations_ms = [d * 0.2 for d in durations_ms]
        nc_events = _make_nc_exec_events(durations_ms)
        nrt_events = _make_nrt_execute_events(nrt_durations_ms)
        result = KernelProfileResult(
            kernel_calls=kernel_calls,
            events_json=json.dumps({"events": nc_events + nrt_events}),
            wall_start_ns=wall_start_ns,
            wall_stop_ns=wall_stop_ns,
        )
        result.save(path)

    def test_single_rank_kernel_only(self, tmp_path):
        """kernel-only merge with a single profile."""
        kp = tmp_path / "kernel.json"
        out = tmp_path / "merged.json"

        self._make_kernel_profile(
            kp,
            calls=[("model.py", 10, "attn"), ("model.py", 20, "ffn")],
            durations_ms=[5.0, 3.0],
        )

        merge_kernel_only([kp], out)

        merged = json.loads(out.read_text())
        assert merged["neuron_total_nc_time_ms"] == pytest.approx(8.0)
        assert merged["neuron_total_time_ms"] == pytest.approx(1.6)
        assert merged["neuron_nc_event_count"] == 2
        assert "model.py" in merged["files"]

        # nrt and nc should be separate
        line10 = merged["files"]["model.py"]["lines"][0]
        assert line10["nc_time_ms"] == pytest.approx(5.0)
        assert line10["nrt_time_ms"] == pytest.approx(1.0)

    def test_multi_rank_kernel_only(self, tmp_path):
        """kernel-only merge with multiple rank profiles."""
        kp0 = tmp_path / "kernel_rank0.json"
        kp1 = tmp_path / "kernel_rank1.json"
        out = tmp_path / "merged.json"

        self._make_kernel_profile(
            kp0,
            calls=[("model.py", 10, "attn")],
            durations_ms=[5.0],
        )
        self._make_kernel_profile(
            kp1,
            calls=[("model.py", 10, "attn")],
            durations_ms=[6.0],
        )

        merge_kernel_only([kp0, kp1], out)

        merged = json.loads(out.read_text())
        # Both ranks contribute
        assert merged["neuron_total_nc_time_ms"] == pytest.approx(11.0)
        assert merged["neuron_nc_event_count"] == 2

        # Line data accumulates across ranks
        line10 = merged["files"]["model.py"]["lines"][0]
        assert line10["nc_time_ms"] == pytest.approx(11.0)
        assert line10["nc_execute_count"] == 2
        # nrt also accumulates
        assert line10["nrt_time_ms"] == pytest.approx(2.2)  # 1.0 + 1.2
