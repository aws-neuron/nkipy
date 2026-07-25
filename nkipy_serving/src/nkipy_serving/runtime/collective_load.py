"""File-backed synchronization helpers for deferred CC kernel loads."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

_BARRIER_COUNTS: dict[str, int] = {}
_DEFAULT_BARRIER_TIMEOUT_S = 900.0
_RANK_DIR_RE = re.compile(r"rank_?\d+")
_CONFIG_HASH_RE = re.compile(r"[0-9a-f]{10}")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:180] or "barrier"


def _is_rank_dir(part: str) -> bool:
    return _RANK_DIR_RE.fullmatch(part) is not None


def _is_config_hash(part: str) -> bool:
    return _CONFIG_HASH_RE.fullmatch(part) is not None


def _barrier_run_id() -> str:
    raw = os.getenv("NKIPY_SERVING_COLLECTIVE_LOAD_RUN_ID")
    if raw:
        return _safe_name(raw)
    return f"pg{os.getpgrp()}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _barrier_timeout_s(timeout_s: float | None) -> float:
    if timeout_s is not None:
        return float(timeout_s)
    for env_name in (
        "NKIPY_SERVING_COLLECTIVE_LOAD_TIMEOUT_S",
        "NKIPY_SERVING_TP_WORKER_TIMEOUT_S",
    ):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return _DEFAULT_BARRIER_TIMEOUT_S


def rank_shared_build_dir(
    build_dir: str | os.PathLike[str] | None,
    *,
    namespace: str,
) -> str | None:
    """Map rank-local compile dirs to a config-scoped shared namespace.

    Runtime worker roots can be sharded as ``.../rank_N/<config_hash>`` while
    executors also append ``rankN`` below the config hash.  Shared kernel caches
    should remove those rank-only components without losing the config hash.
    """
    if build_dir is None:
        return None
    path = Path(str(build_dir))
    parts = path.parts
    rank_indices = [idx for idx, part in enumerate(parts) if _is_rank_dir(part)]
    if not rank_indices:
        return str(path / namespace)

    if len(rank_indices) >= 2:
        first = rank_indices[0]
        last = rank_indices[-1]
        base_parts = (*parts[:first], *parts[first + 1 : last])
    else:
        idx = rank_indices[0]
        prev_is_hash = idx > 0 and _is_config_hash(parts[idx - 1])
        next_is_hash = idx + 1 < len(parts) and _is_config_hash(parts[idx + 1])
        if next_is_hash and not prev_is_hash:
            base_parts = (*parts[:idx], *parts[idx + 1 :])
        else:
            base_parts = parts[:idx]

    return str(Path(*base_parts) / namespace)


def collective_load_barrier(
    *,
    build_dir: str | None,
    name: str,
    rank_id: int,
    world_size: int,
    timeout_s: float | None = None,
) -> None:
    """Wait until all local ranks reach a collective kernel load boundary."""
    world = int(world_size)
    if world <= 1:
        return

    rank = int(rank_id)
    if rank < 0 or rank >= world:
        raise RuntimeError(f"invalid collective barrier rank {rank} for world {world}")

    key = f"{name}:{world}"
    generation = _BARRIER_COUNTS.get(key, 0)
    _BARRIER_COUNTS[key] = generation + 1

    if os.getenv("NKIPY_SERVING_DSV4_BARRIER_TRACE"):
        try:
            with open("/tmp/_dsv4_barrier_trace.log", "a") as _bt:
                _bt.write(f"rank={rank} gen={generation} name={name}\n")
        except Exception:
            pass

    root = Path(build_dir or "/tmp/nkipy_serving_collective_load")
    barrier_dir = (
        root
        / ".collective_load_barriers"
        / _barrier_run_id()
        / f"{_safe_name(name)}_w{world}_g{generation}"
    )
    barrier_dir.mkdir(parents=True, exist_ok=True)
    marker = barrier_dir / f"rank_{rank}"
    marker.write_text(f"{os.getpid()}\n{time.time()}\n", encoding="utf-8")

    timeout = _barrier_timeout_s(timeout_s)
    deadline = time.monotonic() + timeout
    while True:
        live = 0
        for path in barrier_dir.glob("rank_*"):
            try:
                pid_s = path.read_text(encoding="utf-8").splitlines()[0]
                pid = int(pid_s)
            except (OSError, ValueError, IndexError):
                continue
            if _pid_alive(pid):
                live += 1
            else:
                try:
                    path.unlink()
                except OSError:
                    continue
        if live >= world:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out waiting for collective load barrier "
                f"{name!r} after {timeout:.1f}s: "
                f"{live}/{world} live ranks reached"
            )
        time.sleep(0.25)
