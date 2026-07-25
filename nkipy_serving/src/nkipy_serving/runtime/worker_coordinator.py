"""Multi-process worker coordinator for serving.

For total_workers > 1, the coordinator spawns all workers (0..N-1) as
processes and broadcasts forward_step commands so all ranks execute device
kernels (with all-reduce) in lockstep.  Workers only run forward passes —
scheduling, KV management, and sampling happen in the scheduler process.

Each worker has a global rank. With EP enabled:
  - tp_rank = global_rank % tp_degree
  - ep_rank = global_rank // tp_degree
Only workers with ep_rank == 0 return sampled outputs to the coordinator.

ForwardBatch data is passed via shared memory to avoid per-step
serialization overhead.  Dispatch and collect use SHM-based spin signaling
for low-latency wakeup.

Both nkipy and numpy backends return a compact dict (next_token_ids +
optional logprobs) that is written to SHM output slots.  A single
mp.Queue (result_queue) is used only for startup lifecycle signaling
(worker_ready, worker_crash) — never on the per-step forward path.
Shutdown uses SHM command broadcast; no per-worker queues exist.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import struct
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, fields, replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.config import RuntimeConfig, validate_runtime_config
from nkipy_serving.profiling import (
    PROFILING_ENABLED,
    ProfileWriter,
    StartupProfiler,
    StepTimer,
)
from nkipy_serving.runtime.diagnostics import env_flag, env_rank_filter_allows
from nkipy_serving.sampling.constants import LOGPROBS_K_MAX

_DEFAULT_WORKER_TIMEOUT_S = 1800
_DEFAULT_SPIN_BUSY_LOOP_S = 0.05
_DEFAULT_SPIN_IDLE_SLEEP_S = 0.0005
_LOGGER = logging.getLogger(__name__)


def _worker_startup_trace_rank_allowed(rank: int) -> bool:
    return env_rank_filter_allows(
        rank,
        "NKIPY_SERVING_DSV4_RANK_TRACE_FILTER",
        "NKIPY_SERVING_DSV4_WARMUP_TRACE_RANKS",
    )


def _worker_startup_trace(message: str, *, rank: int | None = None) -> None:
    if not env_flag("NKIPY_SERVING_DSV4_WARMUP_TRACE"):
        return
    if rank is not None and not _worker_startup_trace_rank_allowed(int(rank)):
        return
    _LOGGER.info("DSV4 worker startup %s", message)


def _round_startup_s(value: float) -> float:
    return round(float(value), 6)


def _startup_field_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_startup_field_value(item) for item in value]
    if isinstance(value, list):
        return [_startup_field_value(item) for item in value]
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_worker_startup(
    per_rank: dict[int, dict[str, Any]],
    *,
    total_workers: int,
    slowest_limit: int = 8,
) -> dict[str, Any]:
    """Build a compact startup summary suitable for server-info payloads."""
    total_values: list[float] = []
    stage_max: dict[str, dict[str, Any]] = {}
    ranked: list[tuple[float, int, dict[str, Any]]] = []

    for rank, rank_summary in sorted(per_rank.items()):
        rank_id = int(rank)
        total = _float_or_none(rank_summary.get("total_elapsed_s"))
        if total is None:
            continue
        total_values.append(total)
        ranked.append((total, rank_id, rank_summary))
        for stage in rank_summary.get("stages", []):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage", ""))
            elapsed = _float_or_none(stage.get("elapsed_s"))
            if not stage_name or elapsed is None:
                continue
            previous = stage_max.get(stage_name)
            if previous is not None and elapsed <= float(previous["elapsed_s"]):
                continue
            stage_max[stage_name] = {
                "rank": rank_id,
                "elapsed_s": _round_startup_s(elapsed),
                "total_elapsed_s": _round_startup_s(
                    _float_or_none(stage.get("total_elapsed_s")) or 0.0
                ),
            }

    ranked.sort(key=lambda item: (-item[0], item[1]))
    slowest_ranks: list[dict[str, Any]] = []
    for total, rank_id, rank_summary in ranked[: max(0, int(slowest_limit))]:
        slowest_ranks.append(
            {
                "rank": rank_id,
                "tp_rank": rank_summary.get("tp_rank"),
                "ep_rank": rank_summary.get("ep_rank"),
                "visible_core": rank_summary.get("visible_core"),
                "total_elapsed_s": _round_startup_s(total),
                "stages": list(rank_summary.get("stages", [])),
            }
        )

    return {
        "total_workers": int(total_workers),
        "ready_workers": len(per_rank),
        "max_total_elapsed_s": (
            _round_startup_s(max(total_values)) if total_values else 0.0
        ),
        "mean_total_elapsed_s": (
            _round_startup_s(sum(total_values) / len(total_values))
            if total_values
            else 0.0
        ),
        "slowest_ranks": slowest_ranks,
        "stage_max_elapsed_s": stage_max,
    }


def _sched_yield() -> None:
    """Yield CPU time without introducing fixed sleep latency."""
    try:
        os.sched_yield()
    except AttributeError:
        time.sleep(0)


class _SpinTimer:
    """Yield-based polling helper for short waits."""

    def record_activity(self) -> None:
        return None

    def spin(self) -> None:
        _sched_yield()


class _SpinSleepTimer(_SpinTimer):
    """Yield for hot loops; sleep briefly after sustained inactivity."""

    def __init__(self, busy_loop_s: float, idle_sleep_s: float):
        self._busy_loop_s = busy_loop_s
        self._idle_sleep_s = idle_sleep_s
        self._last_activity = time.monotonic()

    def record_activity(self) -> None:
        self._last_activity = time.monotonic()

    def spin(self) -> None:
        if time.monotonic() - self._last_activity >= self._busy_loop_s:
            time.sleep(self._idle_sleep_s)
        else:
            _sched_yield()


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float value for {name}: {raw!r}") from exc


def _make_spin_timer() -> _SpinTimer:
    enabled = os.getenv("NKIPY_SERVING_SPIN_SLEEP_WHEN_IDLE", "0").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return _SpinTimer()
    return _SpinSleepTimer(
        busy_loop_s=_parse_float_env(
            "NKIPY_SERVING_SPIN_BUSY_LOOP_S",
            _DEFAULT_SPIN_BUSY_LOOP_S,
        ),
        idle_sleep_s=_parse_float_env(
            "NKIPY_SERVING_SPIN_IDLE_SLEEP_S",
            _DEFAULT_SPIN_IDLE_SLEEP_S,
        ),
    )


def _visible_core_for_rank(runtime_config: RuntimeConfig, rank: int) -> int:
    """Map a global worker rank onto a physical/logical Neuron core index.

    This is a placement-only setting. Logical TP/EP ranks, communicator ranks,
    and compiled kernel variants stay keyed by the original global rank.
    """
    return int(runtime_config.device_offset) + int(rank)


def _rank_scoped_runtime_config(
    runtime_config: RuntimeConfig,
    *,
    rank: int,
) -> RuntimeConfig:
    """Give spawned workers isolated NKI build workspaces.

    NKI/neuronx-cc compilation writes temporary artifacts under
    ``config_build_dir()``. When TP workers compile the same kernel lazily in
    parallel, sharing that directory can corrupt compiler sidecar files. Keep
    the logical config hash stable, but shard the writable root by rank.
    """
    if int(runtime_config.total_workers) <= 1:
        return runtime_config
    build_root = Path(runtime_config.nkipy_build_dir) / f"rank_{int(rank)}"
    return replace(runtime_config, nkipy_build_dir=str(build_root))


def _make_collective_load_run_id() -> str:
    """Return one run id to share across spawned worker processes."""
    configured = os.getenv("NKIPY_SERVING_COLLECTIVE_LOAD_RUN_ID")
    if configured:
        return configured
    return f"wc{os.getpid()}_{uuid.uuid4().hex[:12]}"


def _log_worker_command_failure(
    rank: int, cmd: str, exc: BaseException, *, tb: bool = False
) -> None:
    msg = f"Worker command failed: rank={rank} cmd={cmd} error={exc!r}"
    if tb:
        _LOGGER.error(msg, exc_info=True)
    else:
        _LOGGER.warning(msg)


def _close_shared_memory(shm: SharedMemory) -> None:
    try:
        shm.close()
    except (BufferError, OSError):
        return


def _unlink_shared_memory(shm: SharedMemory) -> None:
    try:
        shm.unlink()
    except OSError:
        return


# ---------------------------------------------------------------------------
# Shared memory buffers for ForwardBatch fields
# ---------------------------------------------------------------------------


@dataclass
class _SharedBatchBuffers:
    """Holds SharedMemory segments for ForwardBatch numpy arrays."""

    input_ids: SharedMemory
    positions: SharedMemory
    slot_mapping: SharedMemory
    seq_lens: SharedMemory
    block_tables: SharedMemory
    query_start_loc: SharedMemory
    sample_mask: SharedMemory
    temperatures: SharedMemory
    top_ks: SharedMemory
    top_ps: SharedMemory
    min_ps: SharedMemory
    uniform_u: SharedMemory
    state_owner_ids: SharedMemory
    dp_attention_lane_token_counts: SharedMemory
    dp_attention_lane_batch_sizes: SharedMemory
    dp_attention_lane_token_offsets: SharedMemory
    dp_attention_lane_batch_offsets: SharedMemory

    def names(self) -> dict[str, str]:
        """Return mapping of field name → shared memory name for workers."""
        return {f.name: getattr(self, f.name).name for f in fields(self)}

    def close_and_unlink(self) -> None:
        for f in fields(self):
            shm = getattr(self, f.name, None)
            if shm is not None:
                _close_shared_memory(shm)
                _unlink_shared_memory(shm)


def _compute_shm_sizes(runtime_config: RuntimeConfig) -> dict[str, int]:
    """Compute max shared memory buffer sizes from runtime config."""
    max_tokens = (
        max(runtime_config.token_buckets) if runtime_config.token_buckets else 4096
    )
    max_bs = (
        max(runtime_config.request_buckets) if runtime_config.request_buckets else 32
    )
    max_lanes = max(1, int(runtime_config.attention_dp_degree))
    block_size = runtime_config.kv_cache_block_size
    # Conservative upper bound for max blocks per request.
    max_seq_len = runtime_config.max_context_len
    max_blocks_per_req = (max_seq_len + block_size - 1) // block_size
    return {
        "input_ids": max_tokens * 4,  # int32
        "positions": max_tokens * 4,  # int32
        "slot_mapping": max_tokens * 8,  # int64
        "seq_lens": max_bs * 8,  # int64
        "block_tables": max_bs * max_blocks_per_req * 8,  # int64
        "query_start_loc": (max_bs + 1) * 8,  # int64
        "sample_mask": max_bs,  # bool
        "temperatures": max_bs * 4,  # float32
        "top_ks": max_bs * 4,  # int32
        "top_ps": max_bs * 4,  # float32
        "min_ps": max_bs * 4,  # float32
        "uniform_u": max_bs * 4,  # float32
        "state_owner_ids": max_bs * 4,  # int32
        "dp_attention_lane_token_counts": max_lanes * 4,  # int32
        "dp_attention_lane_batch_sizes": max_lanes * 4,  # int32
        "dp_attention_lane_token_offsets": (max_lanes + 1) * 4,  # int32
        "dp_attention_lane_batch_offsets": (max_lanes + 1) * 4,  # int32
    }


def _allocate_shared_buffers(runtime_config: RuntimeConfig) -> _SharedBatchBuffers:
    """Allocate shared memory segments for ForwardBatch fields."""
    sizes = _compute_shm_sizes(runtime_config)
    return _SharedBatchBuffers(
        input_ids=SharedMemory(create=True, size=max(sizes["input_ids"], 1)),
        positions=SharedMemory(create=True, size=max(sizes["positions"], 1)),
        slot_mapping=SharedMemory(create=True, size=max(sizes["slot_mapping"], 1)),
        seq_lens=SharedMemory(create=True, size=max(sizes["seq_lens"], 1)),
        block_tables=SharedMemory(create=True, size=max(sizes["block_tables"], 1)),
        query_start_loc=SharedMemory(
            create=True, size=max(sizes["query_start_loc"], 1)
        ),
        sample_mask=SharedMemory(create=True, size=max(sizes["sample_mask"], 1)),
        temperatures=SharedMemory(create=True, size=max(sizes["temperatures"], 1)),
        top_ks=SharedMemory(create=True, size=max(sizes["top_ks"], 1)),
        top_ps=SharedMemory(create=True, size=max(sizes["top_ps"], 1)),
        min_ps=SharedMemory(create=True, size=max(sizes["min_ps"], 1)),
        uniform_u=SharedMemory(create=True, size=max(sizes["uniform_u"], 1)),
        state_owner_ids=SharedMemory(
            create=True, size=max(sizes["state_owner_ids"], 1)
        ),
        dp_attention_lane_token_counts=SharedMemory(
            create=True, size=max(sizes["dp_attention_lane_token_counts"], 1)
        ),
        dp_attention_lane_batch_sizes=SharedMemory(
            create=True, size=max(sizes["dp_attention_lane_batch_sizes"], 1)
        ),
        dp_attention_lane_token_offsets=SharedMemory(
            create=True, size=max(sizes["dp_attention_lane_token_offsets"], 1)
        ),
        dp_attention_lane_batch_offsets=SharedMemory(
            create=True, size=max(sizes["dp_attention_lane_batch_offsets"], 1)
        ),
    )


def _validate_forward_batch_layout(batch: ForwardBatch) -> None:
    expected_shapes = {
        "input_ids": (int(batch.token_bucket),),
        "positions": (int(batch.token_bucket),),
        "slot_mapping": (int(batch.token_bucket),),
        "seq_lens": (int(batch.batch_size),),
        "block_tables": (int(batch.batch_size), int(batch.block_tables.shape[1])),
        "query_start_loc": (int(batch.batch_size) + 1,),
        "sample_mask": (int(batch.batch_size),),
        "temperatures": (int(batch.batch_size),),
        "top_ks": (int(batch.batch_size),),
        "top_ps": (int(batch.batch_size),),
        "min_ps": (int(batch.batch_size),),
        "uniform_u": (int(batch.batch_size),),
        "state_owner_ids": (int(batch.batch_size),),
        "dp_attention_lane_token_counts": (int(batch.dp_attention_num_lanes),),
        "dp_attention_lane_batch_sizes": (int(batch.dp_attention_num_lanes),),
        "dp_attention_lane_token_offsets": (int(batch.dp_attention_num_lanes) + 1,),
        "dp_attention_lane_batch_offsets": (int(batch.dp_attention_num_lanes) + 1,),
    }
    for name, expected_shape in expected_shapes.items():
        arr = getattr(batch, name)
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"Unexpected shape for {name}: got {arr.shape}, expected {expected_shape}"
            )


def _write_batch_to_shm(
    batch: ForwardBatch,
    shm_bufs: _SharedBatchBuffers,
) -> _ForwardBatchMetadata:
    """Write ForwardBatch arrays into shared memory and return compact metadata."""
    _validate_forward_batch_layout(batch)
    _FIELDS = [
        ("input_ids", batch.input_ids),
        ("positions", batch.positions),
        ("slot_mapping", batch.slot_mapping),
        ("seq_lens", batch.seq_lens),
        ("block_tables", batch.block_tables),
        ("query_start_loc", batch.query_start_loc),
        ("sample_mask", batch.sample_mask),
        ("temperatures", batch.temperatures),
        ("top_ks", batch.top_ks),
        ("top_ps", batch.top_ps),
        ("min_ps", batch.min_ps),
        ("uniform_u", batch.uniform_u),
        ("state_owner_ids", batch.state_owner_ids),
        ("dp_attention_lane_token_counts", batch.dp_attention_lane_token_counts),
        ("dp_attention_lane_batch_sizes", batch.dp_attention_lane_batch_sizes),
        ("dp_attention_lane_token_offsets", batch.dp_attention_lane_token_offsets),
        ("dp_attention_lane_batch_offsets", batch.dp_attention_lane_batch_offsets),
    ]
    for name, arr in _FIELDS:
        expected_dtype = _SHM_BATCH_DTYPES[name]
        if np.dtype(arr.dtype) != expected_dtype:
            raise TypeError(
                f"Unexpected dtype for {name}: got {arr.dtype}, expected {expected_dtype}"
            )
        shm = getattr(shm_bufs, name)
        nbytes = arr.nbytes
        dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
        np.copyto(dst, arr)
    return _ForwardBatchMetadata(
        forward_mode=1 if batch.forward_mode == ForwardMode.DECODE else 0,
        batch_size=int(batch.batch_size),
        requested_topk=int(batch.requested_topk),
        token_bucket=int(batch.token_bucket),
        real_total_tokens=int(batch.real_total_tokens),
        block_table_width=int(batch.block_tables.shape[1]),
        use_full_sampler=1 if batch.use_full_sampler else 0,
        needs_logprobs=1 if batch.needs_logprobs else 0,
        logprobs_k=int(batch.logprobs_k),
        attention_lane=int(batch.attention_lane),
        dp_attention_superstep=1 if batch.dp_attention_superstep else 0,
        dp_attention_num_lanes=int(batch.dp_attention_num_lanes),
    )


def _read_batch_from_shm(
    metadata: _ForwardBatchMetadata | dict[str, Any],
    shm_bufs: dict[str, SharedMemory],
) -> ForwardBatch:
    """Reconstruct ForwardBatch from shared memory using metadata."""
    if isinstance(metadata, _ForwardBatchMetadata):
        batch_size = int(metadata.batch_size)
        requested_topk = int(metadata.requested_topk)
        token_bucket = int(metadata.token_bucket)
        real_total_tokens = int(metadata.real_total_tokens)
        block_table_width = int(metadata.block_table_width)
        use_full_sampler = bool(metadata.use_full_sampler)
        needs_logprobs = bool(metadata.needs_logprobs)
        logprobs_k = int(metadata.logprobs_k)
        attention_lane = int(metadata.attention_lane)
        dp_attention_superstep = bool(metadata.dp_attention_superstep)
        dp_attention_num_lanes = int(metadata.dp_attention_num_lanes)
        mode = ForwardMode.DECODE if metadata.forward_mode == 1 else ForwardMode.EXTEND
    else:
        mode_str = metadata["forward_mode"]
        batch_size = metadata["batch_size"]
        requested_topk = metadata.get("requested_topk", 1)
        token_bucket = metadata.get("token_bucket", 0)
        real_total_tokens = metadata.get("real_total_tokens", 0)
        block_table_width = int(metadata["block_tables_shape"][1])
        use_full_sampler = bool(metadata.get("use_full_sampler", False))
        needs_logprobs = bool(metadata.get("needs_logprobs", False))
        logprobs_k = int(metadata.get("logprobs_k", 0))
        attention_lane = int(metadata.get("attention_lane", -1))
        dp_attention_superstep = bool(metadata.get("dp_attention_superstep", False))
        dp_attention_num_lanes = int(metadata.get("dp_attention_num_lanes", 1))
        mode = ForwardMode.EXTEND if mode_str == "extend" else ForwardMode.DECODE

    def _arr(name: str) -> np.ndarray:
        if isinstance(metadata, _ForwardBatchMetadata):
            dtype = _SHM_BATCH_DTYPES[name]
            if name in {"input_ids", "positions", "slot_mapping"}:
                shape = (token_bucket,)
            elif name in {
                "seq_lens",
                "sample_mask",
                "temperatures",
                "top_ks",
                "top_ps",
                "min_ps",
                "uniform_u",
                "state_owner_ids",
            }:
                shape = (batch_size,)
            elif name == "block_tables":
                shape = (batch_size, block_table_width)
            elif name == "query_start_loc":
                shape = (batch_size + 1,)
            elif name in {
                "dp_attention_lane_token_counts",
                "dp_attention_lane_batch_sizes",
            }:
                shape = (dp_attention_num_lanes,)
            elif name in {
                "dp_attention_lane_token_offsets",
                "dp_attention_lane_batch_offsets",
            }:
                shape = (dp_attention_num_lanes + 1,)
            else:
                raise KeyError(name)
        else:
            if f"{name}_shape" not in metadata:
                if name == "state_owner_ids":
                    return np.arange(batch_size, dtype=np.int32)
                if name == "temperatures":
                    return np.ones((batch_size,), dtype=np.float32)
                if name == "top_ks":
                    return np.ones((batch_size,), dtype=np.int32)
                if name == "top_ps":
                    return np.ones((batch_size,), dtype=np.float32)
                if name == "min_ps":
                    return np.zeros((batch_size,), dtype=np.float32)
                if name == "uniform_u":
                    return np.zeros((batch_size,), dtype=np.float32)
                if name in {
                    "dp_attention_lane_token_counts",
                    "dp_attention_lane_batch_sizes",
                }:
                    return np.zeros((dp_attention_num_lanes,), dtype=np.int32)
                if name in {
                    "dp_attention_lane_token_offsets",
                    "dp_attention_lane_batch_offsets",
                }:
                    return np.zeros((dp_attention_num_lanes + 1,), dtype=np.int32)
                raise KeyError(name)
            shape = tuple(metadata[f"{name}_shape"])
            dtype = np.dtype(metadata[f"{name}_dtype"])
        if name not in shm_bufs:
            if name == "temperatures":
                return np.ones(shape, dtype=np.float32)
            if name == "top_ks":
                return np.ones(shape, dtype=np.int32)
            if name == "top_ps":
                return np.ones(shape, dtype=np.float32)
            if name == "min_ps":
                return np.zeros(shape, dtype=np.float32)
            if name == "uniform_u":
                return np.zeros(shape, dtype=np.float32)
            if name == "state_owner_ids":
                return np.arange(shape[0], dtype=np.int32)
            if name.startswith("dp_attention_lane_"):
                return np.zeros(shape, dtype=np.int32)
            raise KeyError(name)
        shm = shm_bufs[name]
        nbytes = int(np.prod(shape)) * dtype.itemsize
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf[:nbytes]).copy()

    return ForwardBatch(
        forward_mode=mode,
        batch_size=batch_size,
        input_ids=_arr("input_ids"),
        positions=_arr("positions"),
        seq_lens=_arr("seq_lens"),
        slot_mapping=_arr("slot_mapping"),
        block_tables=_arr("block_tables"),
        query_start_loc=_arr("query_start_loc"),
        sample_mask=_arr("sample_mask"),
        requested_topk=int(requested_topk),
        token_bucket=token_bucket,
        real_total_tokens=real_total_tokens,
        use_full_sampler=use_full_sampler,
        needs_logprobs=needs_logprobs,
        logprobs_k=logprobs_k,
        temperatures=_arr("temperatures"),
        top_ks=_arr("top_ks"),
        top_ps=_arr("top_ps"),
        min_ps=_arr("min_ps"),
        uniform_u=_arr("uniform_u"),
        state_owner_ids=_arr("state_owner_ids"),
        attention_lane=attention_lane,
        dp_attention_superstep=dp_attention_superstep,
        dp_attention_num_lanes=dp_attention_num_lanes,
        dp_attention_lane_token_counts=_arr("dp_attention_lane_token_counts"),
        dp_attention_lane_batch_sizes=_arr("dp_attention_lane_batch_sizes"),
        dp_attention_lane_token_offsets=_arr("dp_attention_lane_token_offsets"),
        dp_attention_lane_batch_offsets=_arr("dp_attention_lane_batch_offsets"),
    )


# ---------------------------------------------------------------------------
# SHM step protocol: command block + per-worker status + output slots
# ---------------------------------------------------------------------------
#
# Memory layout (all in a single SharedMemory segment "_step_ctrl"):
#
# Offset 0:  command block (written by coordinator, read by all workers)
#   [0..7]   generation (uint64) — monotonically increasing step counter
#   [8..11]  cmd (uint32) — 0=nop, 1=forward_step, 2=shutdown,
#             3=reload_weights, 4=flush_cache, 5=clear_request_state,
#             6=checkpoint_request_state, 7=restore_request_state
#   [12..15] metadata_len (uint32)
#   [16..4095] metadata payload (msgpack; compact fixed-schema scalars only)
#
# Offset 4096: per-worker status slots (one per worker)
#   Per slot (256 bytes, cache-line aligned):
#     [0..7]   generation (uint64) — echoes command generation when done
#     [8..11]  status (uint32) — 0=idle, 1=ok, 2=error
#     [12..15] error_len (uint32) — length of error string
#     [16..255] error_msg (240 bytes, utf-8)
#
# Offset 4096 + worker_slots: output slots (one per output rank)
#   Per output slot (fixed size, see _OUTPUT_SLOT_SIZE):
#     [0..7]   generation (uint64)
#     [8..11]  output_type (uint32) — 0=none, 1=top1, 2=topk, 3=sampled_ids
#     [12..15] bs (uint32) — batch size
#     [16..19] vocab_offset (int32)
#     [20..23] candidate_k (uint32)
#     [24..31] reserved
#     [32..]   packed payload
#                top1_values[bs] f32 + top1_indices[bs] i32

_CMD_BLOCK_SIZE = 4096
_CMD_METADATA_OFFSET = 16
_WORKER_ERROR_BYTES = 240
_WORKER_SLOT_SIZE = 256
_OUTPUT_SLOT_HEADER_SIZE = 32

_CMD_NOP = 0
_CMD_FORWARD_STEP = 1
_CMD_SHUTDOWN = 2
_CMD_RELOAD_WEIGHTS = 3
_CMD_FLUSH_CACHE = 4
_CMD_CLEAR_REQUEST_STATE = 5
_CMD_CHECKPOINT_REQUEST_STATE = 6
_CMD_RESTORE_REQUEST_STATE = 7

_STATUS_IDLE = 0
_STATUS_OK = 1
_STATUS_ERROR = 2

_OUTPUT_TYPE_NONE = 0
_OUTPUT_TYPE_TOP1 = 1
_OUTPUT_TYPE_TOPK = 2
_OUTPUT_TYPE_IDS = 3
_OUTPUT_TYPE_IDS_WITH_LOGPROBS = 4


def _align_to(value: int, alignment: int = 64) -> int:
    return ((value + alignment - 1) // alignment) * alignment


_SHM_BATCH_DTYPES: dict[str, np.dtype[Any]] = {
    "input_ids": np.dtype(np.int32),
    "positions": np.dtype(np.int32),
    "slot_mapping": np.dtype(np.int64),
    "seq_lens": np.dtype(np.int64),
    "block_tables": np.dtype(np.int64),
    "query_start_loc": np.dtype(np.int64),
    "sample_mask": np.dtype(np.bool_),
    "temperatures": np.dtype(np.float32),
    "top_ks": np.dtype(np.int32),
    "top_ps": np.dtype(np.float32),
    "min_ps": np.dtype(np.float32),
    "uniform_u": np.dtype(np.float32),
    "state_owner_ids": np.dtype(np.int32),
    "dp_attention_lane_token_counts": np.dtype(np.int32),
    "dp_attention_lane_batch_sizes": np.dtype(np.int32),
    "dp_attention_lane_token_offsets": np.dtype(np.int32),
    "dp_attention_lane_batch_offsets": np.dtype(np.int32),
}


class _ForwardBatchMetadata(msgspec.Struct, frozen=True):
    forward_mode: int
    batch_size: int
    requested_topk: int
    token_bucket: int
    real_total_tokens: int
    block_table_width: int
    use_full_sampler: int
    needs_logprobs: int = 0
    logprobs_k: int = 0
    attention_lane: int = -1
    dp_attention_superstep: int = 0
    dp_attention_num_lanes: int = 1


_METADATA_ENCODER = msgspec.msgpack.Encoder()
_METADATA_DECODER = msgspec.msgpack.Decoder(type=_ForwardBatchMetadata)


class _ReloadWeightsMetadata(msgspec.Struct, frozen=True):
    model_path: str


_RELOAD_METADATA_DECODER = msgspec.msgpack.Decoder(type=_ReloadWeightsMetadata)


class _ClearRequestStateMetadata(msgspec.Struct, frozen=True):
    owner_ids: list[int]


_CLEAR_REQUEST_STATE_METADATA_DECODER = msgspec.msgpack.Decoder(
    type=_ClearRequestStateMetadata,
)


class _CheckpointRequestStateMetadata(msgspec.Struct, frozen=True):
    checkpoint_id: str
    owner_id: int
    seq_len: int
    num_tokens: int


_CHECKPOINT_REQUEST_STATE_METADATA_DECODER = msgspec.msgpack.Decoder(
    type=_CheckpointRequestStateMetadata,
)


class _RestoreRequestStateMetadata(msgspec.Struct, frozen=True):
    checkpoint_id: str


_RESTORE_REQUEST_STATE_METADATA_DECODER = msgspec.msgpack.Decoder(
    type=_RestoreRequestStateMetadata,
)


def _compute_output_slot_size(
    max_batch_size: int,
    max_candidate_width: int,
) -> int:
    # Greedy top-k payload: bs * candidate_width * 4 * 2 (values + indices).
    greedy_bytes = max_batch_size * max_candidate_width * 4 * 2
    # Logprobs payload: token_ids + chosen + topk_vals + topk_ids.
    logprobs_bytes = max_batch_size * (4 + 4 + 4 * LOGPROBS_K_MAX + 4 * LOGPROBS_K_MAX)
    payload_bytes = max(greedy_bytes, logprobs_bytes)
    return _align_to(_OUTPUT_SLOT_HEADER_SIZE + payload_bytes)


def _compute_ctrl_shm_layout(runtime_config: RuntimeConfig) -> tuple[int, int]:
    output_slot_size = _compute_output_slot_size(
        runtime_config.max_requests,
        max(1, int(runtime_config.dense_local_topk)),
    )
    # One output slot per (attention lane, TP column). Single-lane configs use
    # tp_degree slots; DSV4 uses attention_dp_degree * tp_degree slots so each
    # lane can publish sampled tokens.
    output_slot_count = runtime_config.tp_degree * runtime_config.attention_dp_degree
    total_size = (
        _CMD_BLOCK_SIZE
        + _WORKER_SLOT_SIZE * runtime_config.total_workers
        + output_slot_size * output_slot_count
    )
    return total_size, output_slot_size


def _cmd_block_write(
    buf: memoryview,
    generation: int,
    cmd: int,
    metadata_payload: bytes,
) -> None:
    """Write command block. Coordinator only."""
    # Write metadata first (before generation), so workers see consistent data.
    meta_len = len(metadata_payload)
    if meta_len > _CMD_BLOCK_SIZE - _CMD_METADATA_OFFSET:
        raise ValueError(f"Metadata payload too large: {meta_len} bytes")
    buf[_CMD_METADATA_OFFSET : _CMD_METADATA_OFFSET + meta_len] = metadata_payload
    struct.pack_into("I", buf, 8, cmd)
    struct.pack_into("I", buf, 12, meta_len)
    # Write generation LAST (release fence) — workers poll on this.
    struct.pack_into("Q", buf, 0, generation)


def _cmd_block_read_generation(buf: memoryview) -> int:
    return struct.unpack_from("Q", buf, 0)[0]


def _cmd_block_read(buf: memoryview) -> tuple[int, int, bytes]:
    """Read command block. Returns (generation, cmd, metadata_payload)."""
    generation = struct.unpack_from("Q", buf, 0)[0]
    cmd = struct.unpack_from("I", buf, 8)[0]
    meta_len = struct.unpack_from("I", buf, 12)[0]
    if meta_len > _CMD_BLOCK_SIZE - _CMD_METADATA_OFFSET:
        raise RuntimeError(f"Corrupt command metadata length: {meta_len}")
    metadata_payload = bytes(
        buf[_CMD_METADATA_OFFSET : _CMD_METADATA_OFFSET + meta_len]
    )
    return generation, cmd, metadata_payload


def _encode_forward_batch_metadata(metadata: _ForwardBatchMetadata) -> bytes:
    return _METADATA_ENCODER.encode(metadata)


def _encode_reload_weights_metadata(model_path: str) -> bytes:
    return _METADATA_ENCODER.encode(_ReloadWeightsMetadata(model_path=str(model_path)))


def _encode_clear_request_state_metadata(owner_ids: list[int]) -> bytes:
    clean = sorted({int(v) for v in owner_ids if int(v) >= 0})
    return _METADATA_ENCODER.encode(_ClearRequestStateMetadata(owner_ids=clean))


def _encode_checkpoint_request_state_metadata(
    *,
    checkpoint_id: str,
    owner_id: int,
    seq_len: int,
    num_tokens: int,
) -> bytes:
    return _METADATA_ENCODER.encode(
        _CheckpointRequestStateMetadata(
            checkpoint_id=str(checkpoint_id),
            owner_id=int(owner_id),
            seq_len=int(seq_len),
            num_tokens=int(num_tokens),
        )
    )


def _encode_restore_request_state_metadata(checkpoint_id: str) -> bytes:
    return _METADATA_ENCODER.encode(
        _RestoreRequestStateMetadata(checkpoint_id=str(checkpoint_id))
    )


def _decode_forward_batch_metadata(payload: bytes) -> _ForwardBatchMetadata:
    try:
        return _METADATA_DECODER.decode(payload)
    except msgspec.DecodeError as exc:
        raise RuntimeError(f"Corrupt command metadata payload: {exc}") from exc


def _worker_slot_offset(rank: int) -> int:
    return _CMD_BLOCK_SIZE + _WORKER_SLOT_SIZE * rank


def _worker_slot_write_done(
    buf: memoryview, rank: int, generation: int, status: int, error: str = ""
) -> None:
    """Worker writes its completion status."""
    off = _worker_slot_offset(rank)
    # Write status and error first.
    struct.pack_into("I", buf, off + 8, status)
    err_bytes = error.encode("utf-8")[:_WORKER_ERROR_BYTES]
    struct.pack_into("I", buf, off + 12, len(err_bytes))
    buf[off + 16 : off + 16 + len(err_bytes)] = err_bytes
    # Write generation LAST (release).
    struct.pack_into("Q", buf, off, generation)


def _worker_slot_read(buf: memoryview, rank: int) -> tuple[int, int, str]:
    """Coordinator reads worker slot. Returns (generation, status, error)."""
    off = _worker_slot_offset(rank)
    generation = struct.unpack_from("Q", buf, off)[0]
    status = struct.unpack_from("I", buf, off + 8)[0]
    err_len = struct.unpack_from("I", buf, off + 12)[0]
    error = bytes(buf[off + 16 : off + 16 + min(err_len, _WORKER_ERROR_BYTES)]).decode(
        "utf-8",
        errors="replace",
    )
    return generation, status, error


def _output_slot_offset(
    total_workers: int, output_idx: int, output_slot_size: int
) -> int:
    return (
        _CMD_BLOCK_SIZE
        + _WORKER_SLOT_SIZE * total_workers
        + output_slot_size * output_idx
    )


def _output_slot_write(
    buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
    output_type: int,
    arrays: list[np.ndarray],
    bs: int,
    vocab_offset: int = 0,
    candidate_k: int = 1,
) -> None:
    """Generic output slot writer. *arrays* are packed sequentially as payload."""
    payload_size = sum(a.nbytes for a in arrays)
    required = _OUTPUT_SLOT_HEADER_SIZE + payload_size
    if required > output_slot_size:
        raise ValueError(
            f"Output slot too small: required={required}, slot_size={output_slot_size}"
        )
    off = _output_slot_offset(total_workers, output_idx, output_slot_size)
    data_off = off + _OUTPUT_SLOT_HEADER_SIZE
    for arr in arrays:
        arr_bytes = arr.tobytes()
        buf[data_off : data_off + len(arr_bytes)] = arr_bytes
        data_off += len(arr_bytes)
    struct.pack_into("I", buf, off + 12, bs)
    struct.pack_into("i", buf, off + 16, vocab_offset)
    struct.pack_into("I", buf, off + 20, candidate_k)
    buf[off + 24 : off + 32] = b"\x00" * 8
    struct.pack_into("I", buf, off + 8, output_type)
    struct.pack_into("Q", buf, off, generation)


def _output_slot_write_topk(
    buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
    topk_values: np.ndarray,
    topk_indices: np.ndarray,
    vocab_offset: int,
) -> None:
    vals = np.asarray(topk_values, dtype=np.float32)
    idx = np.asarray(topk_indices, dtype=np.int32)
    bs, candidate_k = vals.shape
    _output_slot_write(
        buf,
        total_workers,
        output_idx,
        output_slot_size,
        generation,
        _OUTPUT_TYPE_TOPK,
        [vals, idx],
        bs,
        vocab_offset,
        candidate_k,
    )


def _output_slot_write_top1(
    buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
    top1_values: np.ndarray,
    top1_indices: np.ndarray,
    vocab_offset: int,
) -> None:
    vals = np.asarray(top1_values, dtype=np.float32).reshape((-1,))
    idx = np.asarray(top1_indices, dtype=np.int32).reshape((-1,))
    _output_slot_write(
        buf,
        total_workers,
        output_idx,
        output_slot_size,
        generation,
        _OUTPUT_TYPE_TOP1,
        [vals, idx],
        int(vals.shape[0]),
        vocab_offset,
        1,
    )


def _output_slot_write_ids(
    buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
    next_token_ids: np.ndarray,
) -> None:
    ids = np.asarray(next_token_ids, dtype=np.int32).reshape((-1,))
    _output_slot_write(
        buf,
        total_workers,
        output_idx,
        output_slot_size,
        generation,
        _OUTPUT_TYPE_IDS,
        [ids],
        int(ids.shape[0]),
    )


def _output_slot_write_ids_with_logprobs(
    buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
    next_token_ids: np.ndarray,
    chosen_logprobs: np.ndarray,
    topk_logprob_vals: np.ndarray,
    topk_logprob_ids: np.ndarray,
) -> None:
    ids = np.asarray(next_token_ids, dtype=np.int32).reshape((-1,))
    chosen = np.asarray(chosen_logprobs, dtype=np.float32).reshape((-1,))
    topk_v = np.asarray(topk_logprob_vals, dtype=np.float32)
    topk_i = np.asarray(topk_logprob_ids, dtype=np.int32)
    logprobs_k = int(topk_v.shape[1]) if topk_v.ndim == 2 else 0
    _output_slot_write(
        buf,
        total_workers,
        output_idx,
        output_slot_size,
        generation,
        _OUTPUT_TYPE_IDS_WITH_LOGPROBS,
        [ids, chosen, topk_v, topk_i],
        int(ids.shape[0]),
        candidate_k=logprobs_k,
    )


def _output_slot_read(
    buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
) -> dict[str, Any]:
    """Coordinator reads a generation-matched output slot."""
    off = _output_slot_offset(total_workers, output_idx, output_slot_size)
    slot_generation = struct.unpack_from("Q", buf, off)[0]
    if slot_generation != generation:
        return {}
    out_type = struct.unpack_from("I", buf, off + 8)[0]
    if out_type not in {
        _OUTPUT_TYPE_TOP1,
        _OUTPUT_TYPE_TOPK,
        _OUTPUT_TYPE_IDS,
        _OUTPUT_TYPE_IDS_WITH_LOGPROBS,
    }:
        return {}
    bs = struct.unpack_from("I", buf, off + 12)[0]
    candidate_k = struct.unpack_from("I", buf, off + 20)[0]
    if candidate_k <= 0:
        raise RuntimeError(
            f"Corrupt output slot for generation {generation}: invalid candidate_k={candidate_k}"
        )
    if out_type == _OUTPUT_TYPE_IDS:
        required = _OUTPUT_SLOT_HEADER_SIZE + bs * 4
    elif out_type == _OUTPUT_TYPE_IDS_WITH_LOGPROBS:
        # token_ids[bs] + chosen[bs] + topk_vals[bs,k] + topk_ids[bs,k]
        required = (
            _OUTPUT_SLOT_HEADER_SIZE
            + bs * 4
            + bs * 4
            + bs * candidate_k * 4
            + bs * candidate_k * 4
        )
    else:
        required = _OUTPUT_SLOT_HEADER_SIZE + bs * candidate_k * 8
    if required > output_slot_size:
        raise RuntimeError(
            f"Corrupt output slot for generation {generation}: "
            f"required={required}, slot_size={output_slot_size}"
        )
    data_off = off + _OUTPUT_SLOT_HEADER_SIZE
    if out_type == _OUTPUT_TYPE_IDS:
        token_ids = np.frombuffer(
            bytes(buf[data_off : data_off + bs * 4]),
            dtype=np.int32,
        ).copy()
        return {"next_token_ids": token_ids}
    if out_type == _OUTPUT_TYPE_IDS_WITH_LOGPROBS:
        token_ids = np.frombuffer(
            bytes(buf[data_off : data_off + bs * 4]),
            dtype=np.int32,
        ).copy()
        data_off += bs * 4
        chosen = np.frombuffer(
            bytes(buf[data_off : data_off + bs * 4]),
            dtype=np.float32,
        ).copy()
        data_off += bs * 4
        topk_vals = (
            np.frombuffer(
                bytes(buf[data_off : data_off + bs * candidate_k * 4]),
                dtype=np.float32,
            )
            .copy()
            .reshape((bs, candidate_k))
        )
        data_off += bs * candidate_k * 4
        topk_ids = (
            np.frombuffer(
                bytes(buf[data_off : data_off + bs * candidate_k * 4]),
                dtype=np.int32,
            )
            .copy()
            .reshape((bs, candidate_k))
        )
        return {
            "next_token_ids": token_ids,
            "chosen_logprobs": chosen,
            "topk_logprob_vals": topk_vals,
            "topk_logprob_ids": topk_ids,
        }
    vals = (
        np.frombuffer(
            bytes(buf[data_off : data_off + bs * candidate_k * 4]),
            dtype=np.float32,
        )
        .copy()
        .reshape((bs, candidate_k))
    )
    data_off += bs * candidate_k * 4
    idx = (
        np.frombuffer(
            bytes(buf[data_off : data_off + bs * candidate_k * 4]),
            dtype=np.int32,
        )
        .copy()
        .reshape((bs, candidate_k))
    )
    vocab_off = struct.unpack_from("i", buf, off + 16)[0]
    if out_type == _OUTPUT_TYPE_TOP1:
        return {
            "top1_values": vals.reshape((bs,)),
            "top1_indices": idx.reshape((bs,)),
            "vocab_offset": np.asarray([vocab_off], dtype=np.int32),
        }
    return {
        "topk_values": vals,
        "topk_indices": idx,
        "vocab_offset": np.asarray([vocab_off], dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker_main(
    rank: int,
    runtime_config_dict: dict[str, Any],
    result_queue: mp.Queue,
    root_comm_id: str,
    shared_buffer_names: dict[str, str] | None = None,
    ctrl_shm_name: str | None = None,
    output_slot_size: int = 0,
    collective_load_run_id: str | None = None,
) -> None:
    """Worker process for global ranks 0..total_workers-1.

    Each worker derives its tp_rank and ep_rank from the global rank:
      tp_rank = rank % tp_degree
      ep_rank = rank // tp_degree

    Builds a ModelRunner with rank-local sharded weights and executes
    forward_step commands received from the coordinator.  NRT collectives
    synchronize automatically across ranks.

    Only workers with ep_rank == 0 return sampled outputs to the coordinator;
    other EP ranks just signal completion.
    """
    if ctrl_shm_name is None:
        raise RuntimeError("SHM control required")

    shm_bufs: dict[str, SharedMemory] | None = None
    ctrl_shm: SharedMemory | None = None
    try:
        startup_t0 = time.monotonic()
        startup_last_t = startup_t0
        startup_stages: list[dict[str, Any]] = []
        startup_profiler = StartupProfiler("worker_startup", rank=rank)

        def trace_stage(stage: str, **fields: Any) -> None:
            nonlocal startup_last_t
            now = time.monotonic()
            elapsed_s = now - startup_last_t
            total_elapsed_s = now - startup_t0
            startup_last_t = now
            safe_fields = {
                str(key): _startup_field_value(value) for key, value in fields.items()
            }
            startup_stages.append(
                {
                    "stage": str(stage),
                    "elapsed_s": _round_startup_s(elapsed_s),
                    "total_elapsed_s": _round_startup_s(total_elapsed_s),
                    **safe_fields,
                }
            )
            startup_profiler.record(
                stage,
                elapsed_s=elapsed_s,
                total_elapsed_s=total_elapsed_s,
                **safe_fields,
            )
            field_text = " ".join(
                f"{key}={value}" for key, value in safe_fields.items()
            )
            _worker_startup_trace(
                f"rank={rank} {stage}"
                + (f" {field_text}" if field_text else "")
                + f" elapsed_s={total_elapsed_s:.1f}",
                rank=rank,
            )

        trace_stage("process start")
        if collective_load_run_id:
            os.environ["NKIPY_SERVING_COLLECTIVE_LOAD_RUN_ID"] = str(
                collective_load_run_id
            )
        runtime_config = RuntimeConfig(**runtime_config_dict)
        validate_runtime_config(runtime_config)
        runtime_config = _rank_scoped_runtime_config(
            runtime_config,
            rank=rank,
        )
        trace_stage("runtime config ready")
        tp_degree = runtime_config.tp_degree
        total_workers = runtime_config.total_workers

        tp_rank = rank % tp_degree
        ep_rank = rank // tp_degree
        # Per-lane output. Workers whose row is within the attention-DP lane
        # range own an output slot at (row * tp_degree + tp_rank).
        # Single-lane configs publish only row 0.
        attention_dp_degree = int(runtime_config.attention_dp_degree)
        row = ep_rank
        is_output_rank = row < attention_dp_degree
        output_idx = row * tp_degree + tp_rank if is_output_rank else -1

        visible_core = _visible_core_for_rank(runtime_config, rank)
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(visible_core)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("LOG_NKI_KERNEL_CALL", "0")

        # CC-related env vars only needed for multi-rank collective ops.
        if total_workers > 1:
            if ":" not in root_comm_id:
                raise RuntimeError(
                    f"NEURON_RT_ROOT_COMM_ID must be host:port, got {root_comm_id}"
                )
            master_addr, master_port = root_comm_id.split(":", 1)
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(total_workers)
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["NEURON_RT_ROOT_COMM_ID"] = root_comm_id
        trace_stage("env ready")

        # Open shared memory segments if provided.
        if shared_buffer_names is not None:
            shm_bufs = {
                name: SharedMemory(name=shm_name, create=False)
                for name, shm_name in shared_buffer_names.items()
            }
        trace_stage("data shm ready")

        # Open step-control SHM.
        ctrl_shm = SharedMemory(name=ctrl_shm_name, create=False)
        ctrl_buf = ctrl_shm.buf
        trace_stage("control shm ready")

        # Build worker's ModelRunner with rank-sharded weights + own KV pool.
        from nkipy_serving.mem_cache.memory_pool import MHATokenToKVPool
        from nkipy_serving.model_executor.model_runner import ModelRunner
        from nkipy_serving.models.registry import resolve_model_spec

        trace_stage("imports ready")

        spec = resolve_model_spec(runtime_config.model_id)
        model_config = spec.build_config(
            runtime_config, tp_rank=tp_rank, ep_rank=ep_rank
        )
        num_kv_heads, head_dim, num_layers, kv_dtype = spec.build_kv_metadata(
            model_config
        )
        trace_stage(
            "model metadata ready",
            layers=int(num_layers),
            kv_heads=int(num_kv_heads),
        )

        kv_pool = MHATokenToKVPool(
            size=runtime_config.kv_pool_size,
            page_size=runtime_config.kv_cache_block_size,
            dtype=kv_dtype,
            head_num=num_kv_heads,
            head_dim=head_dim,
            layer_num=num_layers,
        )
        trace_stage("kv pool ready")

        model_runner = ModelRunner(
            runtime_config=runtime_config,
            kv_pool=kv_pool,
            tp_rank=tp_rank,
            ep_rank=ep_rank,
        )
        trace_stage("model runner ready")

        # Warmup: compile and first-touch representative bucketed kernel paths
        # before signaling ready.
        from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings

        paddings = build_precompile_paddings(runtime_config)
        trace_stage(
            "warmup start",
            token_buckets=tuple(getattr(paddings, "token_paddings", ())),
            bs_buckets=tuple(getattr(paddings, "bs_paddings", ())),
        )
        model_runner.warmup(paddings)
        trace_stage("warmup done")

        # Per-rank lane metadata: only populated by model executors that
        # implement `.lane_metadata` (currently DeepSeek-V4). For other models
        # this stays None and the coordinator records an empty dict.
        lane_md: dict[str, Any] | None = None
        executor = getattr(model_runner, "_executor", None)
        if executor is not None and hasattr(executor, "lane_metadata"):
            try:
                lane_md = executor.lane_metadata
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to read worker lane metadata: rank=%s error=%r",
                    rank,
                    exc,
                )
                lane_md = None
        startup_summary = {
            "rank": int(rank),
            "tp_rank": int(tp_rank),
            "ep_rank": int(ep_rank),
            "visible_core": int(visible_core),
            "total_elapsed_s": _round_startup_s(time.monotonic() - startup_t0),
            "stages": list(startup_stages),
        }
        result_queue.put(
            {
                "cmd": "worker_ready",
                "rank": rank,
                "ok": True,
                "lane_metadata": lane_md,
                "startup_summary": startup_summary,
            }
        )
        trace_stage("worker ready sent")

        # Profiling (gated by NKIPY_SERVING_PROFILE=1).
        worker_profile_writer: ProfileWriter | None = None
        if PROFILING_ENABLED:
            worker_profile_writer = ProfileWriter(f"worker_{rank}_steps")

        _worker_loop_shm_spin(
            rank=rank,
            tp_rank=tp_rank,
            ep_rank=ep_rank,
            row=row,
            attention_dp_degree=attention_dp_degree,
            is_output_rank=is_output_rank,
            output_idx=output_idx,
            total_workers=total_workers,
            output_slot_size=output_slot_size,
            model_runner=model_runner,
            shm_bufs=shm_bufs,
            ctrl_buf=ctrl_buf,
            result_queue=result_queue,
            worker_profile_writer=worker_profile_writer,
        )

    except KeyboardInterrupt:
        return
    except Exception as exc:
        tb = traceback.format_exc()
        _log_worker_command_failure(rank, "worker_init", exc, tb=True)
        result_queue.put(
            {
                "cmd": "worker_crash",
                "rank": rank,
                "request_id": None,
                "ok": False,
                "error": repr(exc),
                "traceback": tb,
            }
        )
    finally:
        if shm_bufs is not None:
            for shm in shm_bufs.values():
                _close_shared_memory(shm)
        if ctrl_shm is not None:
            _close_shared_memory(ctrl_shm)


def _write_forward_output_to_shm(
    forward_out: dict[str, Any],
    ctrl_buf: memoryview,
    total_workers: int,
    output_idx: int,
    output_slot_size: int,
    generation: int,
) -> None:
    """Dispatch forward output to the appropriate SHM slot writer.

    Logprobs (chosen_logprobs, topk_logprob_*) are written alongside
    sampled token IDs using the ids-with-logprobs slot format.  The greedy
    top1/topk paths keep their rank-local format for TP merge by the
    coordinator's ``_combine_rank_outputs``.
    """
    if "chosen_logprobs" in forward_out and "next_token_ids" in forward_out:
        _output_slot_write_ids_with_logprobs(
            ctrl_buf,
            total_workers,
            output_idx,
            output_slot_size,
            generation,
            np.asarray(forward_out["next_token_ids"], dtype=np.int32),
            np.asarray(forward_out["chosen_logprobs"], dtype=np.float32),
            np.asarray(forward_out["topk_logprob_vals"], dtype=np.float32),
            np.asarray(forward_out["topk_logprob_ids"], dtype=np.int32),
        )
    elif "next_token_ids" in forward_out:
        _output_slot_write_ids(
            ctrl_buf,
            total_workers,
            output_idx,
            output_slot_size,
            generation,
            np.asarray(forward_out["next_token_ids"], dtype=np.int32),
        )
    elif "topk_values" in forward_out:
        _output_slot_write_topk(
            ctrl_buf,
            total_workers,
            output_idx,
            output_slot_size,
            generation,
            np.asarray(forward_out["topk_values"], dtype=np.float32),
            np.asarray(forward_out["topk_indices"], dtype=np.int32),
            int(np.asarray(forward_out["vocab_offset"]).flat[0]),
        )
    else:
        _output_slot_write_top1(
            ctrl_buf,
            total_workers,
            output_idx,
            output_slot_size,
            generation,
            np.asarray(forward_out["top1_values"], dtype=np.float32),
            np.asarray(forward_out["top1_indices"], dtype=np.int32),
            int(np.asarray(forward_out["vocab_offset"]).flat[0]),
        )


def _slice_forward_output_for_dp_attention_lane(
    forward_out: dict[str, Any],
    batch: ForwardBatch,
    lane: int,
) -> dict[str, Any]:
    """Slice full-superstep sampled outputs down to one attention lane."""
    lane_i = int(lane)
    if lane_i < 0 or lane_i >= int(batch.dp_attention_num_lanes):
        raise RuntimeError(
            f"DP-attention lane={lane_i} out of range for "
            f"num_lanes={batch.dp_attention_num_lanes}"
        )
    offsets = np.asarray(batch.dp_attention_lane_batch_offsets, dtype=np.int32)
    start = int(offsets[lane_i])
    end = int(offsets[lane_i + 1])
    if end < start:
        raise RuntimeError(
            f"DP-attention lane offsets must be monotonic, got {offsets.tolist()}"
        )
    full_bs = int(batch.batch_size)
    slice_keys = {
        "next_token_ids",
        "chosen_logprobs",
        "topk_logprob_vals",
        "topk_logprob_ids",
        "top1_values",
        "top1_indices",
        "topk_values",
        "topk_indices",
    }
    out: dict[str, Any] = {}
    for key, value in forward_out.items():
        if key in slice_keys:
            arr = np.asarray(value)
            if arr.shape[:1] != (full_bs,):
                raise RuntimeError(
                    "DP-attention output slice expects first dimension to be "
                    f"batch_size={full_bs}; key={key}, shape={arr.shape}"
                )
            out[key] = arr[start:end]
            continue
        out[key] = value
    return out


def _worker_loop_shm_spin(
    *,
    rank: int,
    tp_rank: int,
    ep_rank: int,
    row: int,
    attention_dp_degree: int,
    is_output_rank: bool,
    output_idx: int,
    total_workers: int,
    output_slot_size: int,
    model_runner: object,
    shm_bufs: dict[str, SharedMemory] | None,
    ctrl_buf: memoryview,
    result_queue: mp.Queue,
    worker_profile_writer: ProfileWriter | None,
) -> None:
    """Main worker loop using SHM generation polling (fast path)."""
    last_gen: int = 0
    worker_step_count = 0
    poll_timer = _make_spin_timer()

    while True:
        gen = _cmd_block_read_generation(ctrl_buf)
        if gen <= last_gen:
            poll_timer.spin()
            continue

        poll_timer.record_activity()
        gen, cmd, meta_payload = _cmd_block_read(ctrl_buf)
        if gen <= last_gen:
            # The coordinator writes metadata first and generation last. If we
            # race with a newer publish here, it is safe to skip the older
            # generation because the coordinator never has multiple in-flight
            # steps: dispatch is always followed by collect before the next
            # publish.
            continue
        last_gen = gen

        if cmd == _CMD_SHUTDOWN:
            _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)
            if worker_profile_writer is not None:
                worker_profile_writer.close()
            result_queue.put({"cmd": "shutdown_ack", "rank": rank})
            return

        if cmd == _CMD_RELOAD_WEIGHTS:
            try:
                metadata = _RELOAD_METADATA_DECODER.decode(meta_payload)
                model_runner.reload_weights_from_disk(metadata.model_path)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)
            except Exception as exc:
                _log_worker_command_failure(rank, "reload_weights", exc)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_ERROR, repr(exc))
            continue

        if cmd == _CMD_FLUSH_CACHE:
            try:
                model_runner.flush_cache()
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)
            except Exception as exc:
                _log_worker_command_failure(rank, "flush_cache", exc)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_ERROR, repr(exc))
            continue

        if cmd == _CMD_CLEAR_REQUEST_STATE:
            try:
                metadata = _CLEAR_REQUEST_STATE_METADATA_DECODER.decode(meta_payload)
                model_runner.clear_request_state(metadata.owner_ids)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)
            except Exception as exc:
                _log_worker_command_failure(rank, "clear_request_state", exc)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_ERROR, repr(exc))
            continue

        if cmd == _CMD_CHECKPOINT_REQUEST_STATE:
            try:
                metadata = _CHECKPOINT_REQUEST_STATE_METADATA_DECODER.decode(
                    meta_payload
                )
                model_runner.checkpoint_request_state(
                    checkpoint_id=metadata.checkpoint_id,
                    owner_id=metadata.owner_id,
                    seq_len=metadata.seq_len,
                    num_tokens=metadata.num_tokens,
                )
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)
            except Exception as exc:
                _log_worker_command_failure(rank, "checkpoint_request_state", exc)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_ERROR, repr(exc))
            continue

        if cmd == _CMD_RESTORE_REQUEST_STATE:
            try:
                metadata = _RESTORE_REQUEST_STATE_METADATA_DECODER.decode(meta_payload)
                model_runner.restore_request_state(metadata.checkpoint_id)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)
            except Exception as exc:
                _log_worker_command_failure(rank, "restore_request_state", exc)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_ERROR, repr(exc))
            continue

        if cmd == _CMD_FORWARD_STEP:
            try:
                timer = StepTimer() if worker_profile_writer is not None else None

                metadata = _decode_forward_batch_metadata(meta_payload)
                if shm_bufs is not None:
                    batch = _read_batch_from_shm(metadata, shm_bufs)
                else:
                    raise RuntimeError("SHM spin mode requires shared memory buffers")

                if timer is not None:
                    timer.mark("shm_read")

                owner_lane = int(batch.attention_lane)
                if attention_dp_degree <= 1:
                    should_run_forward = True
                elif batch.dp_attention_superstep:
                    # Conservative execution: every row participates
                    # in the same replica-local model forward so MoE/FFN
                    # collectives stay synchronized. Each row publishes only
                    # its own lane slice below. A later optimization can make
                    # attention lane-local and gather before MoE.
                    should_run_forward = True
                else:
                    if owner_lane < 0 or owner_lane >= attention_dp_degree:
                        raise RuntimeError(
                            f"attention_lane={owner_lane} out of range for "
                            f"attention_dp_degree={attention_dp_degree}"
                        )
                    # Multi-lane V4 batches are consumed by exactly one TP row.
                    should_run_forward = row == owner_lane
                forward_out = (
                    model_runner.forward(batch) if should_run_forward else None
                )

                if timer is not None:
                    timer.mark("model_forward")

                # Write output to SHM slot for the active lane only. Non-owner
                # lanes still acknowledge the command so lockstep stays intact.
                if (
                    is_output_rank
                    and should_run_forward
                    and (
                        not batch.dp_attention_superstep
                        or int(batch.dp_attention_lane_batch_sizes[row]) > 0
                    )
                ):
                    out_to_write = forward_out
                    if batch.dp_attention_superstep:
                        out_to_write = _slice_forward_output_for_dp_attention_lane(
                            forward_out,
                            batch,
                            row,
                        )
                    _write_forward_output_to_shm(
                        out_to_write,
                        ctrl_buf,
                        total_workers,
                        output_idx,
                        output_slot_size,
                        gen,
                    )

                # Signal completion.
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_OK)

                if timer is not None and worker_profile_writer is not None:
                    timer.mark("result_put")
                    worker_step_count += 1
                    worker_profile_writer.write(
                        {
                            "step": worker_step_count,
                            "rank": rank,
                            "ts": time.time(),
                            "forward_mode": batch.forward_mode.value,
                            "batch_size": batch.batch_size,
                            "token_bucket": batch.token_bucket,
                            "real_tokens": batch.real_total_tokens,
                            **timer.elapsed(),
                        }
                    )

            except Exception as exc:
                _log_worker_command_failure(rank, "forward_step", exc, tb=True)
                _worker_slot_write_done(ctrl_buf, rank, gen, _STATUS_ERROR, repr(exc))
            continue

        _worker_slot_write_done(
            ctrl_buf,
            rank,
            gen,
            _STATUS_ERROR,
            f"Unknown worker cmd: {cmd}",
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class WorkerCoordinator:
    """Coordinates TP+EP workers for lockstep forward execution.

    Spawns total_workers = tp_degree * ep_degree worker processes.
    For ep_degree=1 (no EP), this is equivalent to TP-only coordination.

    Uses shared memory for ForwardBatch data and SHM spin signaling to
    avoid mp.Queue overhead on dispatch and collect paths.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        total_workers = runtime_config.total_workers

        # For multi-worker, need NEURON_RT_ROOT_COMM_ID for CC bootstrap.
        root_comm_id = ""
        if total_workers > 1:
            root_comm_id = os.getenv("NEURON_RT_ROOT_COMM_ID", "localhost:62182")
            if ":" not in root_comm_id:
                raise RuntimeError(
                    f"NEURON_RT_ROOT_COMM_ID must be host:port, got {root_comm_id}"
                )

        self.runtime_config = runtime_config
        self.root_comm_id = root_comm_id
        self._collective_load_run_id = _make_collective_load_run_id()
        self.timeout_s = int(
            os.getenv(
                "NKIPY_SERVING_TP_WORKER_TIMEOUT_S",
                str(_DEFAULT_WORKER_TIMEOUT_S),
            )
        )

        if runtime_config.dsv4_prepared_weight_prestage:
            from nkipy_serving.models.deepseek_v4.prepared_weights import (
                prestage_prepared_weights,
            )

            prestage_prepared_weights(runtime_config)

        self._shm_bufs = _allocate_shared_buffers(runtime_config)
        self._ctx = mp.get_context("spawn")
        self._result_queue: mp.Queue = self._ctx.Queue()
        self._processes: dict[int, mp.Process] = {}
        self._last_forward_output: dict[str, Any] | None = None

        self._prof_shm_write_dur: float = 0.0
        self._prof_broadcast_dur: float = 0.0
        self._prof_collect_dur: float = 0.0
        self._prof_first_result_dur: float = 0.0
        self._prof_combine_dur: float = 0.0

        # Per-rank lane metadata reported at `worker_ready` time.
        # Populated only for models whose executor exposes `.lane_metadata`
        # (currently DeepSeek-V4). Accessed through `lane_metadata()`.
        self._lane_metadata: dict[int, dict[str, Any]] = {}
        self._worker_startup_summaries: dict[int, dict[str, Any]] = {}

        tp_degree = runtime_config.tp_degree
        attention_dp_degree = int(runtime_config.attention_dp_degree)
        # One output slot per (attention lane, TP column). Single-lane configs
        # reduce to `range(tp_degree)`.
        self._output_ranks = set(range(tp_degree * attention_dp_degree))
        self._total_workers = total_workers
        self._tp_degree = tp_degree
        self._attention_dp_degree = attention_dp_degree
        self._active_forward_output_ranks = set(range(tp_degree))
        self._forward_output_ranks_by_request: dict[str, set[int]] = {}
        self._active_forward_dp_lane_batch_sizes: np.ndarray | None = None
        self._forward_dp_lane_batch_sizes_by_request: dict[str, np.ndarray] = {}

        ctrl_size, self._ctrl_output_slot_size = _compute_ctrl_shm_layout(
            runtime_config
        )
        self._ctrl_shm = SharedMemory(create=True, size=ctrl_size)
        self._ctrl_shm.buf[:ctrl_size] = b"\x00" * ctrl_size
        self._generation = 0

        shared_buffer_names = self._shm_bufs.names()
        base_cfg = asdict(runtime_config)
        for global_rank in range(total_workers):
            rank_cfg = dict(base_cfg)
            proc = self._ctx.Process(
                target=_worker_main,
                args=(
                    global_rank,
                    rank_cfg,
                    self._result_queue,
                    self.root_comm_id,
                    shared_buffer_names,
                    self._ctrl_shm.name,
                    self._ctrl_output_slot_size,
                    self._collective_load_run_id,
                ),
            )
            proc.start()
            self._processes[global_rank] = proc

        try:
            self._wait_workers_ready()
        except BaseException:
            self.shutdown()
            raise

    def _wait_workers_ready(self) -> None:
        expected = set(self._processes.keys())
        ready: set[int] = set()
        deadline = time.monotonic() + self.timeout_s
        last_trace = 0.0

        while ready != expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Timed out waiting for workers. ready={sorted(ready)}, "
                    f"expected={sorted(expected)}"
                )
            try:
                result = self._result_queue.get(timeout=min(5.0, max(0.1, remaining)))
            except queue.Empty:
                now = time.monotonic()
                if now - last_trace >= 30.0:
                    missing = sorted(expected - ready)
                    _worker_startup_trace(
                        "waiting for workers "
                        f"ready={len(ready)}/{len(expected)} "
                        f"missing_head={missing[:16]} "
                        f"remaining_s={max(0.0, remaining):.1f}"
                    )
                    last_trace = now
                continue

            if result.get("cmd") == "worker_crash":
                trace = str(result.get("traceback") or "").strip()
                raise RuntimeError(
                    f"Worker crashed during init: rank={result.get('rank')}, "
                    f"error={result.get('error')}" + (f"\n{trace}" if trace else "")
                )
            if result.get("cmd") == "worker_ready":
                rank = int(result["rank"])
                ready.add(rank)
                _worker_startup_trace(
                    f"worker ready rank={rank} ready={len(ready)}/{len(expected)}"
                )
                md = result.get("lane_metadata")
                if isinstance(md, dict):
                    self._lane_metadata[rank] = md
                startup_summary = result.get("startup_summary")
                if isinstance(startup_summary, dict):
                    self._worker_startup_summaries[rank] = startup_summary

    def _assert_alive(self) -> None:
        dead = [rank for rank, proc in self._processes.items() if not proc.is_alive()]
        if dead:
            raise RuntimeError(f"Worker processes exited unexpectedly: ranks={dead}")

    def _drain_async_result_queue(self) -> None:
        """Drain the result queue, raising on worker crashes."""
        while True:
            try:
                message = self._result_queue.get_nowait()
            except queue.Empty:
                return
            if message.get("cmd") == "worker_crash":
                trace = str(message.get("traceback") or "").strip()
                raise RuntimeError(
                    f"Worker crashed: rank={message.get('rank')}, "
                    f"error={message.get('error')}" + (f"\n{trace}" if trace else "")
                )

    def _dispatch_worker_command_shm(
        self, cmd: int, metadata_payload: bytes = b""
    ) -> int:
        self._assert_alive()
        self._generation += 1
        _cmd_block_write(self._ctrl_shm.buf, self._generation, cmd, metadata_payload)
        return self._generation

    def _collect_worker_command_shm(self, generation: int, op_name: str) -> None:
        total = self._total_workers
        expected = set(range(total))
        completed: set[int] = set()
        deadline = time.monotonic() + self.timeout_s
        poll_timer = _make_spin_timer()
        next_liveness_check = time.monotonic()

        while completed != expected:
            made_progress = False
            self._drain_async_result_queue()

            for rank in range(total):
                if rank in completed:
                    continue
                worker_gen, status, error = _worker_slot_read(self._ctrl_shm.buf, rank)
                if worker_gen < generation:
                    continue

                completed.add(rank)
                poll_timer.record_activity()
                made_progress = True

                if status == _STATUS_ERROR:
                    raise RuntimeError(
                        f"Worker {op_name} failed: rank={rank}, error={error}"
                    )
                if status != _STATUS_OK:
                    raise RuntimeError(
                        f"Worker {op_name} failed: rank={rank}, unexpected status={status}"
                    )

            if completed == expected:
                break

            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for worker {op_name}. "
                    f"received={sorted(completed)}, expected={sorted(expected)}"
                )
            if now >= next_liveness_check:
                self._assert_alive()
                next_liveness_check = now + 0.1
            if not made_progress:
                poll_timer.spin()

        self._drain_async_result_queue()

    def _output_ranks_for_attention_lane(self, attention_lane: int) -> set[int]:
        """Return the SHM output slots expected for one scheduler lane."""
        if self._attention_dp_degree <= 1:
            return set(range(self._tp_degree))

        lane = int(attention_lane)
        if lane < 0 or lane >= self._attention_dp_degree:
            raise ValueError(
                "attention_lane must be in [0, attention_dp_degree) when "
                f"attention_dp_degree > 1; got lane={lane}, "
                f"attention_dp_degree={self._attention_dp_degree}"
            )
        start = lane * self._tp_degree
        return set(range(start, start + self._tp_degree))

    def _output_ranks_for_dp_attention_superstep(
        self,
        lane_batch_sizes: np.ndarray,
    ) -> set[int]:
        """Return output slots expected for all active lanes in a superstep."""
        sizes = np.asarray(lane_batch_sizes, dtype=np.int32).reshape(-1)
        if sizes.shape != (self._attention_dp_degree,):
            raise ValueError(
                "lane_batch_sizes must be [attention_dp_degree], got "
                f"{sizes.shape} for attention_dp_degree={self._attention_dp_degree}"
            )
        ranks: set[int] = set()
        for lane, batch_size in enumerate(sizes):
            if int(batch_size) <= 0:
                continue
            ranks.update(range(lane * self._tp_degree, (lane + 1) * self._tp_degree))
        return ranks

    def _expected_output_ranks_for_request(self, request_id: str | None) -> set[int]:
        by_request = getattr(self, "_forward_output_ranks_by_request", {})
        if request_id is not None and request_id in by_request:
            return set(by_request.pop(request_id))
        return set(getattr(self, "_active_forward_output_ranks", self._output_ranks))

    def dispatch_forward_step(self, forward_batch: ForwardBatch) -> str:
        """Broadcast a forward_step command to all workers via SHM."""
        self._assert_alive()
        if self._attention_dp_degree > 1:
            if not bool(forward_batch.dp_attention_superstep):
                raise RuntimeError(
                    "attention_dp_degree > 1 requires a DP-attention "
                    "superstep ForwardBatch. Refusing to run the old "
                    "single-lane owner-row path."
                )
        request_id = uuid.uuid4().hex
        self._last_forward_output = None
        dp_lane_batch_sizes = None
        if self._attention_dp_degree > 1 and bool(forward_batch.dp_attention_superstep):
            dp_lane_batch_sizes = (
                np.asarray(
                    forward_batch.dp_attention_lane_batch_sizes,
                    dtype=np.int32,
                )
                .reshape(-1)
                .copy()
            )
            expected_output_ranks = self._output_ranks_for_dp_attention_superstep(
                dp_lane_batch_sizes
            )
        else:
            expected_output_ranks = self._output_ranks_for_attention_lane(
                forward_batch.attention_lane
            )
        self._active_forward_output_ranks = set(expected_output_ranks)
        self._forward_output_ranks_by_request[request_id] = set(expected_output_ranks)
        self._active_forward_dp_lane_batch_sizes = dp_lane_batch_sizes
        if dp_lane_batch_sizes is not None:
            self._forward_dp_lane_batch_sizes_by_request[request_id] = (
                dp_lane_batch_sizes
            )

        if PROFILING_ENABLED:
            _shm_t0 = time.perf_counter()

        metadata = _write_batch_to_shm(forward_batch, self._shm_bufs)

        if PROFILING_ENABLED:
            self._prof_shm_write_dur = time.perf_counter() - _shm_t0

        # Write command block to SHM.
        self._generation += 1
        meta_payload = _encode_forward_batch_metadata(metadata)

        if PROFILING_ENABLED:
            _bcast_t0 = time.perf_counter()

        _cmd_block_write(
            self._ctrl_shm.buf,
            self._generation,
            _CMD_FORWARD_STEP,
            meta_payload,
        )

        if PROFILING_ENABLED:
            self._prof_broadcast_dur = time.perf_counter() - _bcast_t0

        return request_id

    def _dispatch_and_collect(
        self,
        shm_cmd: int,
        shm_payload: bytes = b"",
        *,
        op_name: str,
    ) -> None:
        """Dispatch a command to all workers and wait for completion."""
        gen = self._dispatch_worker_command_shm(shm_cmd, shm_payload)
        self._collect_worker_command_shm(gen, op_name)

    def reload_weights(self, model_path: str) -> None:
        """Rewrite model weights in-place on all workers from a local/HF snapshot."""
        self._dispatch_and_collect(
            _CMD_RELOAD_WEIGHTS,
            _encode_reload_weights_metadata(model_path),
            op_name="reload_weights",
        )

    def flush_cache(self) -> None:
        """Clear worker-side KV cache state on all workers."""
        self._dispatch_and_collect(_CMD_FLUSH_CACHE, op_name="flush_cache")

    def clear_request_state(self, owner_ids: list[int]) -> None:
        """Clear request-owned model state rows on all workers."""
        clean = sorted({int(v) for v in owner_ids if int(v) >= 0})
        if not clean:
            return
        self._dispatch_and_collect(
            _CMD_CLEAR_REQUEST_STATE,
            _encode_clear_request_state_metadata(clean),
            op_name="clear_request_state",
        )

    def checkpoint_request_state(
        self,
        *,
        owner_id: int,
        seq_len: int,
        num_tokens: int,
        checkpoint_id: str | None = None,
    ) -> str:
        """Checkpoint request-owned model state rows on all workers."""
        if int(owner_id) < 0:
            raise ValueError("owner_id must be non-negative")
        if int(seq_len) < 0:
            raise ValueError("seq_len must be non-negative")
        if int(num_tokens) < 0:
            raise ValueError("num_tokens must be non-negative")
        clean_id = str(checkpoint_id or uuid.uuid4().hex)
        if not clean_id:
            raise ValueError("checkpoint_id must be non-empty")
        self._dispatch_and_collect(
            _CMD_CHECKPOINT_REQUEST_STATE,
            _encode_checkpoint_request_state_metadata(
                checkpoint_id=clean_id,
                owner_id=int(owner_id),
                seq_len=int(seq_len),
                num_tokens=int(num_tokens),
            ),
            op_name="checkpoint_request_state",
        )
        return clean_id

    def restore_request_state(self, checkpoint_id: str) -> None:
        """Restore request-owned model state rows on all workers."""
        clean_id = str(checkpoint_id)
        if not clean_id:
            raise ValueError("checkpoint_id must be non-empty")
        self._dispatch_and_collect(
            _CMD_RESTORE_REQUEST_STATE,
            _encode_restore_request_state_metadata(clean_id),
            op_name="restore_request_state",
        )

    def collect_forward_step(self, request_id: str) -> None:
        """Wait for all workers to complete the forward step."""
        if PROFILING_ENABLED:
            _collect_t0 = time.perf_counter()
            _first_result_dur = None
        else:
            _collect_t0 = 0.0
            _first_result_dur = None

        gen = self._generation
        total = self._total_workers
        expected = set(range(total))
        completed: set[int] = set()
        deadline = time.monotonic() + self.timeout_s
        poll_timer = _make_spin_timer()
        next_liveness_check = time.monotonic()
        while completed != expected:
            made_progress = False
            self._drain_async_result_queue()

            for rank in range(total):
                if rank in completed:
                    continue
                worker_gen, status, error = _worker_slot_read(self._ctrl_shm.buf, rank)
                if worker_gen < gen:
                    continue

                if PROFILING_ENABLED and _first_result_dur is None:
                    _first_result_dur = time.perf_counter() - _collect_t0

                completed.add(rank)
                poll_timer.record_activity()
                made_progress = True

                if status == _STATUS_ERROR:
                    raise RuntimeError(
                        f"Worker forward failed: rank={rank}, error={error}"
                    )
                if status != _STATUS_OK:
                    raise RuntimeError(
                        f"Worker forward failed: rank={rank}, unexpected status={status}"
                    )

            if completed == expected:
                break

            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for forward step generation {gen}. "
                    f"received={sorted(completed)}, expected={sorted(expected)}"
                )
            if now >= next_liveness_check:
                self._assert_alive()
                next_liveness_check = now + 0.1
            if not made_progress:
                poll_timer.spin()

        self._drain_async_result_queue()

        expected_output_ranks = self._expected_output_ranks_for_request(request_id)
        rank_outputs: dict[int, dict[str, Any]] = {}
        for rank in sorted(expected_output_ranks):
            out = _output_slot_read(
                self._ctrl_shm.buf,
                total,
                rank,
                self._ctrl_output_slot_size,
                gen,
            )
            if out:
                rank_outputs[rank] = out

        if not rank_outputs:
            raise RuntimeError(
                f"No SHM outputs published for generation {gen}. "
                "Distributed nkipy workers must return sampled outputs."
            )

        missing_ranks = sorted(expected_output_ranks.difference(rank_outputs.keys()))
        if missing_ranks:
            raise RuntimeError(
                f"Missing SHM outputs for generation {gen}: ranks={missing_ranks}"
            )

        if PROFILING_ENABLED:
            self._prof_collect_dur = time.perf_counter() - _collect_t0
            self._prof_first_result_dur = _first_result_dur or 0.0

        # Aggregate outputs.
        if PROFILING_ENABLED:
            _combine_t0 = time.perf_counter()
        dp_lane_batch_sizes = getattr(
            self,
            "_forward_dp_lane_batch_sizes_by_request",
            {},
        ).pop(request_id, None)
        self._last_forward_output = self._combine_forward_rank_outputs(
            rank_outputs,
            dp_lane_batch_sizes=dp_lane_batch_sizes,
        )
        if PROFILING_ENABLED:
            self._prof_combine_dur = time.perf_counter() - _combine_t0

    # -- Non-blocking poll/collect for overlap scheduling ------------------

    def poll_forward_step(self) -> bool:
        """Non-blocking check: have all workers completed the current step?

        Returns True when every worker has written a status for the current
        generation.  Does NOT read output slots — call
        ``collect_forward_step_result`` after this returns True.
        """
        gen = self._generation
        total = self._total_workers
        self._drain_async_result_queue()
        for rank in range(total):
            worker_gen, status, error = _worker_slot_read(self._ctrl_shm.buf, rank)
            if worker_gen < gen:
                return False
            if status == _STATUS_ERROR:
                raise RuntimeError(f"Worker forward failed: rank={rank}, error={error}")
            if status != _STATUS_OK:
                raise RuntimeError(
                    f"Worker forward failed: rank={rank}, unexpected status={status}"
                )
        return True

    def collect_forward_step_result(self) -> None:
        """Read worker outputs after ``poll_forward_step`` returned True.

        Populates ``last_forward_output`` and profiling counters, just like
        the tail of ``collect_forward_step``.
        """
        gen = self._generation
        total = self._total_workers

        expected_output_ranks = set(
            getattr(self, "_active_forward_output_ranks", self._output_ranks)
        )
        getattr(self, "_forward_output_ranks_by_request", {}).clear()
        rank_outputs: dict[int, dict[str, Any]] = {}
        for rank in sorted(expected_output_ranks):
            out = _output_slot_read(
                self._ctrl_shm.buf,
                total,
                rank,
                self._ctrl_output_slot_size,
                gen,
            )
            if out:
                rank_outputs[rank] = out

        if not rank_outputs:
            raise RuntimeError(
                f"No SHM outputs published for generation {gen}. "
                "Distributed nkipy workers must return sampled outputs."
            )

        missing_ranks = sorted(expected_output_ranks.difference(rank_outputs.keys()))
        if missing_ranks:
            raise RuntimeError(
                f"Missing SHM outputs for generation {gen}: ranks={missing_ranks}"
            )

        dp_lane_batch_sizes = getattr(
            self,
            "_active_forward_dp_lane_batch_sizes",
            None,
        )
        self._active_forward_dp_lane_batch_sizes = None
        self._last_forward_output = self._combine_forward_rank_outputs(
            rank_outputs,
            dp_lane_batch_sizes=dp_lane_batch_sizes,
        )

    def _combine_forward_rank_outputs(
        self,
        rank_outputs: dict[int, dict[str, Any]],
        *,
        dp_lane_batch_sizes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if dp_lane_batch_sizes is None:
            return self._combine_rank_outputs(self._guard_single_lane(rank_outputs))
        return self._combine_dp_attention_superstep_outputs(
            rank_outputs,
            lane_batch_sizes=dp_lane_batch_sizes,
        )

    def _combine_dp_attention_superstep_outputs(
        self,
        rank_outputs: dict[int, dict[str, Any]],
        *,
        lane_batch_sizes: np.ndarray,
    ) -> dict[str, Any]:
        """Merge TP outputs per lane, then concatenate lanes in superstep order."""
        sizes = np.asarray(lane_batch_sizes, dtype=np.int32).reshape(-1)
        if sizes.shape != (self._attention_dp_degree,):
            raise ValueError(
                "lane_batch_sizes must be [attention_dp_degree], got "
                f"{sizes.shape} for attention_dp_degree={self._attention_dp_degree}"
            )
        lane_outputs: list[dict[str, Any]] = []
        for lane, batch_size in enumerate(sizes):
            bs = int(batch_size)
            if bs <= 0:
                continue
            expected = set(range(lane * self._tp_degree, (lane + 1) * self._tp_degree))
            missing = sorted(expected.difference(rank_outputs.keys()))
            if missing:
                raise RuntimeError(
                    "Missing DP-attention superstep SHM outputs for "
                    f"lane={lane}: ranks={missing}"
                )
            lane_out = self._combine_rank_outputs(
                {rank: rank_outputs[rank] for rank in sorted(expected)}
            )
            ids = np.asarray(lane_out["next_token_ids"], dtype=np.int32).reshape(-1)
            if ids.shape != (bs,):
                raise RuntimeError(
                    "DP-attention lane output batch size mismatch: "
                    f"lane={lane}, expected={bs}, got={ids.shape}"
                )
            lane_outputs.append(lane_out)
        if not lane_outputs:
            return {"next_token_ids": np.zeros((0,), dtype=np.int32)}

        result: dict[str, Any] = {
            "next_token_ids": np.concatenate(
                [
                    np.asarray(out["next_token_ids"], dtype=np.int32).reshape(-1)
                    for out in lane_outputs
                ]
            )
        }
        optional_keys = (
            ("chosen_logprobs", np.float32),
            ("topk_logprob_vals", np.float32),
            ("topk_logprob_ids", np.int32),
        )
        for key, dtype in optional_keys:
            if all(key in out for out in lane_outputs):
                result[key] = np.concatenate(
                    [np.asarray(out[key], dtype=dtype) for out in lane_outputs],
                    axis=0,
                )
        return result

    def _guard_single_lane(
        self,
        rank_outputs: dict[int, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """Require one TP row per combine call when attention-DP is active."""
        if self._attention_dp_degree <= 1 or not rank_outputs:
            return rank_outputs
        rows = {idx // self._tp_degree for idx in rank_outputs}
        if len(rows) > 1:
            raise RuntimeError(
                "Multi-lane output collection: slots from multiple rows "
                f"({sorted(rows)}) arrived in the same combine call. "
                "The scheduler must collect one attention lane at a time."
            )
        return rank_outputs

    @staticmethod
    def _combine_rank_outputs(
        rank_outputs: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """Combine per-rank forward outputs into a scheduler-friendly result.

        Both nkipy and numpy backends produce compact sampled-output dicts.
        Supported per-rank formats:
          - next_token_ids: [bs] int32 (already sampled; identical across ranks)
          - top1_values/top1_indices + vocab_offset (TP merge by best value)
          - topk_values/topk_indices + vocab_offset (TP merge by best value)
        Optional logprobs (chosen_logprobs, topk_logprob_vals/ids) are
        passed through from rank 0.

        The function assumes all `rank_outputs` come from one TP group.
        For V4 multi-lane, the caller groups by lane before invoking (the
        guard is in ``collect_forward_step`` / ``collect_forward_step_result``).
        """
        ranks = sorted(rank_outputs.keys())
        if not ranks:
            return {"next_token_ids": np.zeros((0,), dtype=np.int32)}

        if all(
            isinstance(rank_outputs[r], dict) and "next_token_ids" in rank_outputs[r]
            for r in ranks
        ):
            base = np.asarray(
                rank_outputs[ranks[0]]["next_token_ids"], dtype=np.int32
            ).reshape((-1,))
            for rank in ranks[1:]:
                cur = np.asarray(
                    rank_outputs[rank]["next_token_ids"], dtype=np.int32
                ).reshape((-1,))
                if cur.shape != base.shape or not np.array_equal(cur, base):
                    raise RuntimeError(
                        "Mismatched sampled token ids across output ranks: "
                        f"rank0_shape={base.shape} rank{rank}_shape={cur.shape}"
                    )
            result: dict[str, Any] = {
                "next_token_ids": base.astype(np.int32, copy=False)
            }
            # Pass through logprobs from rank 0 (identical across ranks).
            rank0_out = rank_outputs[ranks[0]]
            if "chosen_logprobs" in rank0_out:
                result["chosen_logprobs"] = np.asarray(
                    rank0_out["chosen_logprobs"], dtype=np.float32
                )
                result["topk_logprob_vals"] = np.asarray(
                    rank0_out["topk_logprob_vals"], dtype=np.float32
                )
                result["topk_logprob_ids"] = np.asarray(
                    rank0_out["topk_logprob_ids"], dtype=np.int32
                )
            return result

        top1_vals_list: list[np.ndarray] = []
        top1_idx_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []
        idx_list: list[np.ndarray] = []
        offsets: list[int] = []
        all_top1 = True

        for r in ranks:
            out = rank_outputs[r]
            if not isinstance(out, dict) or "vocab_offset" not in out:
                raise RuntimeError(
                    f"Unsupported worker output for rank={r}: "
                    f"keys={list(out) if isinstance(out, dict) else type(out)}"
                )
            if "topk_values" in out and "topk_indices" in out:
                all_top1 = False
                vals = np.asarray(out["topk_values"], dtype=np.float32)
                idx = np.asarray(out["topk_indices"], dtype=np.int32)
            elif "top1_values" in out and "top1_indices" in out:
                top1_vals = np.asarray(out["top1_values"], dtype=np.float32).reshape(
                    (-1,)
                )
                top1_idx = np.asarray(out["top1_indices"], dtype=np.int32).reshape(
                    (-1,)
                )
                top1_vals_list.append(top1_vals)
                top1_idx_list.append(top1_idx)
                vals = top1_vals.reshape((-1, 1))
                idx = top1_idx.reshape((-1, 1))
            else:
                raise RuntimeError(
                    f"Unsupported worker output for rank={r}: keys={list(out)}"
                )
            off_arr = np.asarray(out["vocab_offset"], dtype=np.int32).reshape((-1,))
            if off_arr.size != 1:
                raise RuntimeError(
                    f"Expected vocab_offset to be a scalar array, got shape={off_arr.shape}"
                )
            offsets.append(int(off_arr[0]))
            vals_list.append(vals)
            idx_list.append(idx)

        if all_top1:
            vals_stack = np.stack(top1_vals_list, axis=0)  # [tp, bs]
            idx_stack = np.stack(top1_idx_list, axis=0)  # [tp, bs]
            if vals_stack.ndim != 2 or idx_stack.ndim != 2:
                raise RuntimeError(
                    "Expected stacked top-1 arrays to be rank-2, "
                    f"got {vals_stack.shape=} {idx_stack.shape=}"
                )
            bs = int(vals_stack.shape[1])
            if bs == 0:
                return {"next_token_ids": np.zeros((0,), dtype=np.int32)}
            offsets_arr = np.asarray(offsets, dtype=np.int32)[:, None]
            global_ids = idx_stack + offsets_arr
            best_rank = np.argmax(vals_stack, axis=0).astype(np.int32)
            rows = np.arange(bs, dtype=np.int32)
            result = {"next_token_ids": global_ids[best_rank, rows].astype(np.int32)}
            # Pass through logprobs from rank 0 if present.
            rank0_out = rank_outputs[ranks[0]]
            if "chosen_logprobs" in rank0_out:
                result["chosen_logprobs"] = np.asarray(
                    rank0_out["chosen_logprobs"], dtype=np.float32
                )
            if "topk_logprob_vals" in rank0_out:
                result["topk_logprob_vals"] = np.asarray(
                    rank0_out["topk_logprob_vals"], dtype=np.float32
                )
                result["topk_logprob_ids"] = np.asarray(
                    rank0_out["topk_logprob_ids"], dtype=np.int32
                )
            return result

        candidate_widths = {int(vals.shape[1]) for vals in vals_list}
        if len(candidate_widths) != 1:
            raise RuntimeError(
                f"Mismatched per-rank candidate widths: {sorted(candidate_widths)}"
            )
        vals_stack = np.stack(vals_list, axis=0)  # [tp, bs, k]
        idx_stack = np.stack(idx_list, axis=0)  # [tp, bs, k]
        offsets_arr = np.asarray(offsets, dtype=np.int32)  # [tp]

        if vals_stack.ndim != 3 or idx_stack.ndim != 3:
            raise RuntimeError(
                "Expected stacked candidate arrays to be rank-3, "
                f"got {vals_stack.shape=} {idx_stack.shape=}"
            )
        bs = int(vals_stack.shape[1])
        if bs == 0:
            return {"next_token_ids": np.zeros((0,), dtype=np.int32)}

        global_ids = idx_stack + offsets_arr[:, None, None]
        vals_flat = np.transpose(vals_stack, (1, 0, 2)).reshape((bs, -1))
        ids_flat = np.transpose(global_ids, (1, 0, 2)).reshape((bs, -1))
        best_pos = np.argmax(vals_flat, axis=1).astype(np.int32)
        rows = np.arange(bs, dtype=np.int32)
        next_ids = ids_flat[rows, best_pos].astype(np.int32)
        return {"next_token_ids": next_ids}

    @property
    def last_forward_output(self) -> dict[str, Any] | None:
        """Output from the last forward step (sampled token ids + optional logprobs)."""
        return self._last_forward_output

    @property
    def last_ipc_profile(self) -> dict[str, float] | None:
        """IPC timing breakdown from the last forward step (profiling only)."""
        if not PROFILING_ENABLED:
            return None
        return {
            "t_shm_write": round(self._prof_shm_write_dur, 6),
            "t_broadcast": round(self._prof_broadcast_dur, 6),
            "t_collect_total": round(self._prof_collect_dur, 6),
            "t_first_result": round(self._prof_first_result_dur, 6),
            "t_combine_outputs": round(self._prof_combine_dur, 6),
        }

    def lane_metadata(self) -> dict[int, dict[str, Any]]:
        """Return the per-rank lane/group metadata reported at worker_ready.

        Only populated for models whose executor exposes `.lane_metadata`
        (DeepSeek-V4 family). Empty dict for others. Shape of each value:
        `{rank, row, col, replica, attn_lane, tp_group, moe_ep_group, ...}`.
        """
        return dict(self._lane_metadata)

    def startup_summary(self) -> dict[str, Any]:
        """Return aggregate worker startup timings reported with worker_ready."""
        return _summarize_worker_startup(
            dict(self._worker_startup_summaries),
            total_workers=self._total_workers,
        )

    def shutdown(self) -> None:
        self._generation += 1
        _cmd_block_write(self._ctrl_shm.buf, self._generation, _CMD_SHUTDOWN, b"{}")

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            alive = [proc for proc in self._processes.values() if proc.is_alive()]
            if not alive:
                break
            time.sleep(0.1)

        for proc in self._processes.values():
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

        self._processes.clear()

        # Clean up shared memory.
        self._shm_bufs.close_and_unlink()
        _close_shared_memory(self._ctrl_shm)
        _unlink_shared_memory(self._ctrl_shm)
