"""Lightweight serving profiler gated by NKIPY_SERVING_PROFILE=1.

When enabled, writes per-step JSONL trace files to NKIPY_SERVING_PROFILE_DIR
(default: /tmp/nkipy_serving_profile/).

Usage in scheduler / worker:
    from nkipy_serving.profiling import PROFILING_ENABLED, ProfileWriter, StepTimer

    if PROFILING_ENABLED:
        writer = ProfileWriter("scheduler_steps")
        ...
        timer = StepTimer()
        timer.mark("batch_build")
        ...
        timer.mark("dispatch")
        writer.write({**timer.elapsed(), "step": n})
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_env = os.getenv("NKIPY_SERVING_PROFILE", "").strip().lower()
PROFILING_ENABLED: bool = _env in {"1", "true", "yes", "on"}

PROFILE_DIR: Path = Path(
    os.getenv("NKIPY_SERVING_PROFILE_DIR", "/tmp/nkipy_serving_profile")
)


class ProfileWriter:
    """Buffered JSONL writer. One instance per trace file."""

    def __init__(self, name: str, flush_every: int = 50):
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._path = PROFILE_DIR / f"{name}.jsonl"
        self._fh = open(self._path, "a")
        self._flush_every = flush_every
        self._count = 0

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


class StepTimer:
    """Accumulates named timing marks within a single step.

    Usage:
        t = StepTimer()
        # ... do work A ...
        t.mark("phase_a")
        # ... do work B ...
        t.mark("phase_b")
        durations = t.elapsed()
        # {"t_phase_a": 0.00023, "t_phase_b": 0.00105, "t_total": 0.00128}
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._marks: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self._marks.append((name, now - self._last))
        self._last = now

    def elapsed(self) -> dict[str, float]:
        total = time.perf_counter() - self._t0
        out = {f"t_{name}": round(dur, 6) for name, dur in self._marks}
        # Use raw sum (not rounded) so t_total >= sum of individual marks
        # even when float rounding of the sum would shave off a sub-us amount.
        out["t_total"] = max(round(total, 6), sum(out.values()))
        return out


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def startup_profiling_enabled() -> bool:
    """Return whether startup phase JSONL should be emitted.

    DSV4 startup work currently uses ``NKIPY_SERVING_DSV4_STAGE_PROFILE`` for
    compile/forward JSONL. Treat it as enabling startup JSONL too, while also
    allowing a model-agnostic explicit flag.
    """
    return _env_flag("NKIPY_SERVING_STARTUP_PROFILE") or _env_flag(
        "NKIPY_SERVING_DSV4_STAGE_PROFILE"
    )


_STARTUP_PROFILE_WRITERS: dict[str, ProfileWriter] = {}


def _startup_profile_writer(component: str, rank: int | None) -> ProfileWriter | None:
    if not startup_profiling_enabled():
        return None
    label = "unknown" if rank is None else str(int(rank))
    key = f"{component}:{label}"
    writer = _STARTUP_PROFILE_WRITERS.get(key)
    if writer is None:
        writer = ProfileWriter(f"{component}_rank_{label}", flush_every=1)
        _STARTUP_PROFILE_WRITERS[key] = writer
    return writer


class StartupProfiler:
    """Per-process startup phase profiler.

    Records compact JSONL phase rows. It intentionally does not apply rank
    filters: startup diagnosis needs all-rank timing, even when log printing is
    filtered down to one rank.
    """

    def __init__(
        self,
        component: str,
        *,
        rank: int | None = None,
        **base_fields: Any,
    ) -> None:
        self.component = str(component)
        self.rank = int(rank) if rank is not None else _rank_from_env()
        self._base_fields = dict(base_fields)
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._writer = _startup_profile_writer(self.component, self.rank)

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def record(
        self,
        stage: str,
        *,
        event: str = "phase",
        elapsed_s: float | None = None,
        total_elapsed_s: float | None = None,
        **fields: Any,
    ) -> None:
        if self._writer is None:
            return
        now_wall = time.time()
        now = time.perf_counter()
        if elapsed_s is None:
            elapsed_s = now - self._last
        if total_elapsed_s is None:
            total_elapsed_s = now - self._t0
        self._last = now
        self._writer.write(
            {
                "ts": now_wall,
                "component": self.component,
                "stage": str(stage),
                "event": str(event),
                "rank": self.rank,
                "pid": os.getpid(),
                "elapsed_s": round(float(elapsed_s), 6),
                "total_elapsed_s": round(float(total_elapsed_s), 6),
                **self._base_fields,
                **fields,
            }
        )

    @contextmanager
    def phase(self, stage: str, **fields: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                stage,
                event="phase",
                elapsed_s=time.perf_counter() - start,
                **fields,
            )


def _rank_from_env() -> int | None:
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None
