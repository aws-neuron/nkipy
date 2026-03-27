"""Kernel profiler for correlating device traces with Python source lines."""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spike import SpikeModel
from spike._spike import SystemTraceSession

from .trace_timeline import _parse_trace_timeline

logger = logging.getLogger(__name__)


# Paths within the framework that should be skipped when walking frames
_SKIP_PATH_FRAGMENTS = (
    os.sep + os.path.join("spike", "src", "spike") + os.sep,
    os.sep + os.path.join("nkipy", "src", "nkipy", "runtime") + os.sep,
    os.sep + os.path.join("nkipy", "src", "nkipy", "core") + os.sep,
    os.sep + os.path.join("nkipy", "src", "nkipy", "tools", "profiler") + os.sep,
)


def _find_user_frame():
    """Walk the call stack to find the first user-code frame.

    Skips frames inside spike/ and nkipy/ internals so that source lines
    point to the actual model code (e.g. model.py) rather than framework
    wrappers. The merge step handles creating entries for files that scalene
    didn't profile.

    Uses sys._getframe() for speed (avoids reading source lines unlike
    inspect.stack()).
    """
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not any(frag in filename for frag in _SKIP_PATH_FRAGMENTS):
            return filename, frame.f_lineno
        frame = frame.f_back
    return "<unknown>", 0


@dataclass
class KernelExecution:
    """A single kernel execution with its source location."""

    filename: str
    lineno: int
    kernel_name: str
    call_index: int


@dataclass
class KernelProfileResult:
    """Result of a kernel profiling session."""

    kernel_calls: list[KernelExecution] = field(default_factory=list)
    events_json: str = ""
    wall_start_ns: int = 0
    wall_stop_ns: int = 0

    def save(self, path: str | Path) -> None:
        """Save the profile result to a JSON file."""
        data = {
            "kernel_calls": [
                {
                    "filename": kc.filename,
                    "lineno": kc.lineno,
                    "kernel_name": kc.kernel_name,
                    "call_index": kc.call_index,
                }
                for kc in self.kernel_calls
            ],
            "events_json": self.events_json,
            "wall_start_ns": self.wall_start_ns,
            "wall_stop_ns": self.wall_stop_ns,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KernelProfileResult":
        """Load a profile result from a JSON file."""
        data = json.loads(Path(path).read_text())
        kernel_calls = [KernelExecution(**kc) for kc in data["kernel_calls"]]
        return cls(
            kernel_calls=kernel_calls,
            events_json=data["events_json"],
            wall_start_ns=data.get("wall_start_ns", 0),
            wall_stop_ns=data.get("wall_stop_ns", 0),
        )


def _get_current_rank() -> int:
    """Get the current distributed rank, or 0 if not in a distributed context."""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass
    return 0


class KernelProfiler:
    """Context manager that profiles kernel executions with device traces.

    Simultaneously:
    - Starts a SystemTraceSession to capture device timing
    - Monkey-patches SpikeModel.__call__ to record source locations
    - Optionally starts/stops scalene CPU profiling

    Usage::

        with KernelProfiler(core_id=0, output_path="kernel_profile.json"):
            model.forward(input_ids, attention_mask)

    For distributed runs, use ``target_ranks`` to control which ranks profile::

        with KernelProfiler(core_id=0, target_ranks=[0]):
            model.forward(input_ids, attention_mask)

    When run under ``scalene --off``, the profiler controls exactly which
    region gets CPU-profiled, keeping compilation/warmup out of the profile.
    """

    def __init__(
        self,
        core_id: Optional[int] = None,
        scalene: bool = True,
        output_path: str | Path = "kernel_profile.json",
        target_ranks: Optional[list[int]] = None,
    ):
        self._core_id = core_id
        self._scalene_enabled = scalene
        self._output_path = Path(output_path)
        self._target_ranks = target_ranks
        self._result: Optional[KernelProfileResult] = None
        self._trace: Optional[SystemTraceSession] = None
        self._original_call: Optional[object] = None
        self._kernel_calls: list[KernelExecution] = []
        self._call_index = 0
        self._scalene_profiler = None
        self._wall_start_ns = 0
        self._wall_stop_ns = 0
        self._active = True  # Whether this rank is actively profiling

    def __enter__(self):
        # Check if current rank should be profiled
        if self._target_ranks is not None:
            rank = _get_current_rank()
            if rank not in self._target_ranks:
                self._active = False
                return self

        self._kernel_calls = []
        self._call_index = 0

        # Monkey-patch SpikeModel.__call__
        self._original_call = SpikeModel.__call__
        profiler_self = self

        original_call = self._original_call

        def patched_call(model_self, *args, **kwargs):
            filename, lineno = _find_user_frame()
            profiler_self._kernel_calls.append(
                KernelExecution(
                    filename=filename,
                    lineno=lineno,
                    kernel_name=model_self.name,
                    call_index=profiler_self._call_index,
                )
            )
            profiler_self._call_index += 1
            return original_call(model_self, *args, **kwargs)

        SpikeModel.__call__ = patched_call

        # Start device trace
        self._trace = SystemTraceSession(self._core_id)
        self._trace.__enter__()

        # Start scalene profiling if enabled
        if self._scalene_enabled:
            try:
                from scalene import scalene_profiler

                self._scalene_profiler = scalene_profiler
                scalene_profiler.start()
            except (ImportError, SystemExit, Exception) as e:
                logger.debug(f"Scalene not available, skipping: {e}")
                self._scalene_profiler = None

        # Record wall clock start
        self._wall_start_ns = time.time_ns()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._active:
            return False

        # Record wall clock stop
        self._wall_stop_ns = time.time_ns()

        # Stop scalene profiling first (before any cleanup overhead)
        if self._scalene_profiler is not None:
            try:
                self._scalene_profiler.stop()
            except Exception as e:
                logger.debug(f"Scalene stop failed: {e}")

        # Restore original __call__
        if self._original_call is not None:
            SpikeModel.__call__ = self._original_call
            self._original_call = None

        # Fetch trace events and stop session
        events_json = ""
        if self._trace is not None:
            events_json = self._trace.fetch_events_json()
            self._trace.__exit__(exc_type, exc_val, exc_tb)
            self._trace = None

        # Determine output path (include rank suffix for multi-rank)
        output_path = self._output_path
        if self._target_ranks is not None and len(self._target_ranks) > 1:
            rank = _get_current_rank()
            stem = output_path.stem
            output_path = output_path.with_name(
                f"{stem}_rank{rank}{output_path.suffix}"
            )

        # Build result
        self._result = KernelProfileResult(
            kernel_calls=list(self._kernel_calls),
            events_json=events_json,
            wall_start_ns=self._wall_start_ns,
            wall_stop_ns=self._wall_stop_ns,
        )

        # Save to disk
        self._result.save(output_path)

        # Print one-line summary
        timeline = _parse_trace_timeline(events_json)
        total_nc_ms = sum(timeline.nc_durations_ms)
        wall_ms = (self._wall_stop_ns - self._wall_start_ns) / 1_000_000.0
        util_pct = (total_nc_ms / wall_ms * 100) if wall_ms > 0 else 0.0
        n_calls = len(self._kernel_calls)
        print(
            f"Kernel profile: {n_calls} calls, {total_nc_ms:.1f}ms NC, "
            f"{util_pct:.1f}% utilization -> {output_path}"
        )

        return False

    @property
    def result(self) -> Optional[KernelProfileResult]:
        """Access the profile result after the context manager exits."""
        return self._result
