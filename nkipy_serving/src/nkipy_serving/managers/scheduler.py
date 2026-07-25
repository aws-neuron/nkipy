"""Scheduler subprocess for runtime model execution.

Architecture: ScheduleBatch encapsulates batch building and result routing.
The main loop (_run_single_batching_step) creates ScheduleBatch objects via
_get_next_batches(), calls build_forward_batch() + forward + process_results().
Mixed chunk combines extend and decode into a single EXTEND forward pass.
"""

from __future__ import annotations

import copy
import json
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.config import (
    RuntimeConfig,
    configure_runtime_environment,
    validate_runtime_config,
)
from nkipy_serving.managers.io_struct import GenerateReqInput
from nkipy_serving.mem_cache import BasePrefixCache, create_prefix_cache
from nkipy_serving.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)
from nkipy_serving.mem_cache.common import PrefixCacheReq
from nkipy_serving.mem_cache.memory_pool import ReqToTokenPool, SchedulerKVPoolStub
from nkipy_serving.profiling import PROFILING_ENABLED, ProfileWriter, StepTimer
from nkipy_serving.runtime.precompile_paddings import (
    PrecompilePaddings,
    build_precompile_paddings,
)
from nkipy_serving.runtime.shape_guard import select_bucket
from nkipy_serving.runtime.worker_coordinator import WorkerCoordinator
from nkipy_serving.sampling.constants import LOGPROBS_K_MAX
from nkipy_serving.sampling.params import SamplingParams
from nkipy_serving.sampling.random_state import assign_seed, draw_uniform


@dataclass(frozen=True)
class _Dsv4KVPressureModel:
    """Scheduler-side estimate of request-local DSV4 mutable-state bytes."""

    compress_ratios: tuple[int, ...]
    sliding_window: int
    head_dim: int
    index_head_dim: int
    max_context_len: int
    cache_bytes_per_elem: int = 2
    state_bytes_per_elem: int = 4

    @property
    def routing_policy(self) -> str:
        return "least_dsv4_kv_byte_pressure_round_robin_tie_break"

    @property
    def pressure_unit(self) -> str:
        return "estimated_dsv4_kv_state_bytes"

    def estimate_bytes(self, token_slots: int) -> int:
        seq_len = max(1, min(int(token_slots), int(self.max_context_len)))
        total = 0
        for ratio in self.compress_ratios:
            total += self._swa_bytes(seq_len)
            if ratio > 0:
                total += self._compressor_bytes(seq_len, int(ratio), self.head_dim)
                if int(ratio) == 4:
                    total += self._compressor_bytes(
                        seq_len,
                        int(ratio),
                        self.index_head_dim,
                    )
        return int(total)

    def estimate_allocated_bytes(
        self,
        *,
        max_requests: int,
        num_slots_per_layer: int,
        max_seq_len: int | None = None,
    ) -> int:
        """Estimate the DSV4 mutable state allocation per worker."""
        allocation_seq_len = (
            int(self.max_context_len) if max_seq_len is None else int(max_seq_len)
        )
        total = (
            len(self.compress_ratios)
            * int(num_slots_per_layer)
            * int(self.head_dim)
            * int(self.cache_bytes_per_elem)
        )
        for ratio in self.compress_ratios:
            if ratio <= 0:
                continue
            total += self._allocated_compressor_bytes(
                max_requests=max_requests,
                ratio=int(ratio),
                head_dim=int(self.head_dim),
                max_seq_len=allocation_seq_len,
            )
            if int(ratio) == 4:
                total += self._allocated_compressor_bytes(
                    max_requests=max_requests,
                    ratio=int(ratio),
                    head_dim=int(self.index_head_dim),
                    max_seq_len=allocation_seq_len,
                )
        return int(total)

    def summary(
        self,
        *,
        max_requests: int,
        num_slots_per_layer: int,
        max_seq_len: int | None = None,
    ) -> dict[str, Any]:
        allocation_seq_len = (
            int(self.max_context_len) if max_seq_len is None else int(max_seq_len)
        )
        kind_counts = {
            "full": sum(1 for ratio in self.compress_ratios if int(ratio) == 0),
            "c4a": sum(1 for ratio in self.compress_ratios if int(ratio) == 4),
            "c128a": sum(1 for ratio in self.compress_ratios if int(ratio) == 128),
        }
        out = {
            "num_layers": len(self.compress_ratios),
            "layer_kind_counts": kind_counts,
            "sliding_window": int(self.sliding_window),
            "head_dim": int(self.head_dim),
            "index_head_dim": int(self.index_head_dim),
            "max_context_len": int(self.max_context_len),
            "num_slots_per_layer": int(num_slots_per_layer),
            "max_requests": int(max_requests),
            "estimated_bytes_per_full_context_request": self.estimate_bytes(
                int(self.max_context_len)
            ),
            "estimated_static_state_bytes_per_worker": self.estimate_allocated_bytes(
                max_requests=max_requests,
                num_slots_per_layer=num_slots_per_layer,
                max_seq_len=allocation_seq_len,
            ),
            "state_size": int(allocation_seq_len),
        }
        return out

    def _swa_bytes(self, seq_len: int) -> int:
        rows = min(int(seq_len), int(self.sliding_window))
        return rows * int(self.head_dim) * int(self.cache_bytes_per_elem)

    def _compressor_bytes(self, seq_len: int, ratio: int, head_dim: int) -> int:
        overlap_factor = 2 if ratio == 4 else 1
        ring_rows = min(int(seq_len), overlap_factor * int(ratio))
        state_width = overlap_factor * int(head_dim)
        packed_width = 2 * state_width
        ring_bytes = ring_rows * packed_width * int(self.state_bytes_per_elem)
        compressed_rows = min(
            (int(seq_len) + int(ratio) - 1) // int(ratio),
            max(1, int(self.max_context_len) // int(ratio)),
        )
        cache_bytes = compressed_rows * int(head_dim) * int(self.cache_bytes_per_elem)
        return ring_bytes + cache_bytes

    def _allocated_compressor_bytes(
        self,
        *,
        max_requests: int,
        ratio: int,
        head_dim: int,
        max_seq_len: int,
    ) -> int:
        overlap_factor = 2 if ratio == 4 else 1
        ring_size = overlap_factor * int(ratio)
        state_width = overlap_factor * int(head_dim)
        packed_width = 2 * state_width
        state_rows = int(max_requests) * ring_size
        state_bytes = state_rows * packed_width * int(self.state_bytes_per_elem)
        cache_rows = int(max_requests) * max(1, int(max_seq_len) // int(ratio))
        cache_bytes = cache_rows * int(head_dim) * int(self.cache_bytes_per_elem)
        return state_bytes + cache_bytes


def _read_local_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_dsv4_hf_config_for_scheduler(
    runtime_config: RuntimeConfig,
) -> dict[str, Any] | None:
    """Best-effort local DSV4 config read for scheduler pressure estimates."""

    candidates = [
        str(getattr(runtime_config, "model_id", "") or ""),
        str(getattr(runtime_config, "hf_model_id", "") or ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        local_dir = Path(candidate)
        if local_dir.is_dir():
            data = _read_local_json(local_dir / "config.json")
            if data is not None:
                return data

    try:
        from nkipy_serving.models.reload_utils import resolve_model_snapshot_path
    except ImportError:
        return None

    for candidate in candidates:
        if not candidate:
            continue
        try:
            snapshot = resolve_model_snapshot_path(
                candidate,
                revision=getattr(runtime_config, "hf_revision", None),
                local_files_only=bool(
                    getattr(runtime_config, "hf_local_files_only", True)
                ),
            )
        except (OSError, RuntimeError, ValueError):
            continue
        data = _read_local_json(Path(snapshot) / "config.json")
        if data is not None:
            return data
    return None


def _build_dsv4_kv_pressure_model(
    runtime_config: RuntimeConfig,
) -> _Dsv4KVPressureModel | None:
    if not _runtime_uses_dsv4_request_state(runtime_config):
        return None
    cfg = _load_dsv4_hf_config_for_scheduler(runtime_config)
    if not cfg or cfg.get("model_type") != "deepseek_v4":
        return None
    raw_ratios = cfg.get("compress_ratios")
    if not isinstance(raw_ratios, list):
        return None
    try:
        num_layers = int(cfg["num_hidden_layers"])
        if getattr(runtime_config, "hf_num_hidden_layers", None) is not None:
            num_layers = min(
                num_layers,
                int(getattr(runtime_config, "hf_num_hidden_layers")),
            )
        ratios = tuple(int(v) for v in raw_ratios[:num_layers])
        if len(ratios) != num_layers or any(r not in (0, 4, 128) for r in ratios):
            return None
        return _Dsv4KVPressureModel(
            compress_ratios=ratios,
            sliding_window=int(cfg["sliding_window"]),
            head_dim=int(cfg["head_dim"]),
            index_head_dim=int(cfg.get("index_head_dim", cfg["head_dim"])),
            max_context_len=int(getattr(runtime_config, "max_context_len")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _runtime_uses_dsv4_request_state(runtime_config: RuntimeConfig) -> bool:
    model_id = str(getattr(runtime_config, "model_id", ""))
    hf_model = str(getattr(runtime_config, "hf_model_id", "") or "")
    return (
        model_id.startswith("deepseek-ai/DeepSeek-V4")
        or model_id == "deepseek-v4"
        or hf_model.startswith("deepseek-ai/DeepSeek-V4")
        or str(getattr(runtime_config, "attention_backend", ""))
        == "Dsv4SparseAttention"
    )


# ---------------------------------------------------------------------------
# Scheduler-side tokenizer service (replaces TokenizerManager proxy override)
# ---------------------------------------------------------------------------


class SchedulerTokenizerService:
    """Lightweight tokenizer service for the scheduler subprocess.

    Provides encode/decode without the full TokenizerManager proxy machinery.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        from nkipy_serving.tokenization.hf_tokenizer import HfTokenizer

        self.tokenizer = HfTokenizer(
            model_id=runtime_config.tokenizer_model_id,
            revision=runtime_config.tokenizer_revision,
            local_files_only=runtime_config.tokenizer_local_files_only,
        )
        self.served_model_name = runtime_config.model_id

    def encode_prompt(self, prompt: str) -> np.ndarray:
        return self.tokenizer.encode(prompt)

    def decode_ids(self, token_ids: np.ndarray) -> str:
        return self.tokenizer.decode(token_ids)

    def decode_one_token(self, token_id: int) -> str:
        return self.tokenizer.decode(np.asarray([int(token_id)], dtype=np.int32))


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------


@dataclass
class _RequestState:
    request_id: str
    req: GenerateReqInput
    prompt_ids: np.ndarray
    stream: bool
    submitted_at: float = field(default_factory=time.time)
    first_scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    generated_ids: list[int] = field(default_factory=list)
    extend_done: bool = False
    extend_offset: int = 0  # prompt tokens processed so far (for chunked prefill)
    prefix_hit_length: int = 0  # tokens reused from prefix cache
    prefix_cache_node: object | None = None  # cache node for lock management
    # KV pool tracking.
    req_pool_idx: int = -1
    out_cache_loc: np.ndarray | None = None  # allocated slot indices
    seq_len: int = 0  # current sequence length (prompt + generated so far)
    # Stop sequence tracking.
    stop_strs: list[str] = field(default_factory=list)
    stop_str_max_len: int = 0  # max TOKEN count of any stop string (for tail window)
    stop_str_max_char_len: int = 0  # max CHARACTER length of any stop string
    stop_token_ids: set[int] = field(default_factory=set)
    no_stop_trim: bool = False
    finish_reason: str = "length"  # "length" | "stop"
    # Logprob tracking.
    return_logprob: bool = False
    logprobs_k: int = 0  # 0 = disabled, N = return top-N
    top_logprobs_num: int = 0
    logprob_start_len: int = -1
    token_logprobs: list[tuple[float, int, str | None]] = field(default_factory=list)
    top_logprobs: list[list[tuple[float, int, str | None]] | None] = field(
        default_factory=list
    )
    input_token_logprobs: list[tuple[float, int, str | None]] = field(
        default_factory=list
    )
    input_top_logprobs: list[list[tuple[float, int, str | None]] | None] = field(
        default_factory=list
    )
    # Incremental detokenization for stop-string checks.
    decode_offset: int = 0
    decoded_text: str = ""
    sampling_params: SamplingParams | None = None
    sampling_seed: int = 0  # concrete seed for stateless RNG (0 = greedy)
    # DP-attention lane ownership. -1 means "not assigned yet" or single-lane.
    # Set once at admission and immutable for the lifetime of the request.
    attention_lane: int = -1

    @property
    def last_token_id(self) -> int:
        """Last token: most recent generated, or final prompt token."""
        return (
            self.generated_ids[-1] if self.generated_ids else int(self.prompt_ids[-1])
        )


@dataclass
class _RequestStateCheckpoint:
    """Scheduler-side request state paired with worker DSV4 state snapshots."""

    request_id: str
    owner_id: int
    seq_len: int
    extend_done: bool
    extend_offset: int
    generated_ids: list[int]
    decode_offset: int
    decoded_text: str
    finish_reason: str
    first_token_ts: float
    token_logprobs: list[Any]
    top_logprobs: list[Any]
    input_token_logprobs: list[Any]
    input_top_logprobs: list[Any]
    pending_token_outputs_len: int
    total_generated_tokens: int
    decode_tokens_since_last: int
    prefill_tokens_since_last: int


@dataclass
class _SchedulerMetrics:
    total_submitted: int = 0
    total_completed: int = 0
    total_aborted: int = 0
    total_errors: int = 0
    total_generated_tokens: int = 0
    total_stream_events: int = 0
    total_timed_out: int = 0
    scheduler_steps: int = 0
    max_extend_batch_size: int = 0
    max_decode_batch_size: int = 0
    # Throughput tracking.
    last_decode_throughput: float = 0.0  # tokens/sec
    last_prefill_throughput: float = 0.0  # tokens/sec
    # Latency tracking.
    total_time_to_first_token_s: float = 0.0
    total_requests_with_ttft: int = 0
    # Cache tracking.
    total_prefix_cache_hit_tokens: int = 0
    total_prefix_cache_total_tokens: int = 0
    # DP-attention lane counters. Empty on single-lane configs.
    lane_submitted: list[int] = field(default_factory=list)
    lane_completed: list[int] = field(default_factory=list)
    lane_aborted: list[int] = field(default_factory=list)
    lane_errors: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: convert flat slot indices to block table
# ---------------------------------------------------------------------------


def _slots_to_block_table(
    slots: np.ndarray,
    block_size: int,
    max_blocks: int,
) -> np.ndarray:
    """Convert flat cache slot indices to a block ID table.

    Each slot index maps to block_id = slot // block_size.
    Returns a 1-D array of unique block IDs in order, padded to max_blocks.
    """
    block_ids = slots // block_size
    # Unique blocks in order of first appearance (vectorized).
    _, first_idx = np.unique(block_ids, return_index=True)
    unique_ordered = block_ids[np.sort(first_idx)]
    result = np.zeros(max_blocks, dtype=np.int64)
    n = min(len(unique_ordered), max_blocks)
    result[:n] = unique_ordered[:n]
    return result


# ---------------------------------------------------------------------------
# Schedule batch: encapsulates batch building + result routing
# ---------------------------------------------------------------------------


class ScheduleBatch:
    """Lightweight batch object that encapsulates state grouping, ForwardBatch
    construction, and result routing.

    Created fresh each step by _SchedulerCore._get_next_batches().  Does NOT
    own _RequestState lifecycle (allocation / deallocation stays in _SchedulerCore).
    """

    def __init__(
        self,
        extend_states: list[_RequestState],
        decode_states: list[_RequestState],
        *,
        block_size: int,
        chunked_prefill_size: int,
        paddings: PrecompilePaddings,
        requested_topk: int = 1,
        mixed: bool = False,
        attention_lane: int = -1,
    ):
        self.extend_states = extend_states
        self.decode_states = decode_states
        self._block_size = block_size
        self._chunked_prefill_size = chunked_prefill_size
        self._paddings = paddings
        self._requested_topk = int(requested_topk)
        self._mixed = mixed
        # All requests in one ScheduleBatch must share a lane.
        # -1 means single-lane.
        self.attention_lane = int(attention_lane)
        self._forward_batch: ForwardBatch | None = None

    @property
    def is_empty(self) -> bool:
        return not self.extend_states and not self.decode_states

    def _require_forward_batch(self) -> ForwardBatch:
        fb = self._forward_batch
        if fb is None:
            raise RuntimeError(
                "ScheduleBatch.process_results called before build_forward_batch"
            )
        return fb

    @staticmethod
    def _require_cache_slots(state: _RequestState) -> np.ndarray:
        slots = state.out_cache_loc
        if slots is None:
            raise RuntimeError(
                f"request {state.request_id!r} has no KV cache slots allocated"
            )
        return slots

    # -- Build --

    def build_forward_batch(self) -> ForwardBatch:
        """Build a single ForwardBatch from the states in this batch."""
        now = time.time()
        for state in self._all_states():
            if state.first_scheduled_ts <= 0.0:
                state.first_scheduled_ts = now
        if self._mixed and (self.extend_states or self.decode_states):
            fb = self._build_mixed_batch()
        elif self.extend_states:
            fb = self._build_extend_batch()
        else:
            fb = self._build_decode_batch()
        self._forward_batch = fb
        return fb

    # -- Process results --

    def process_results(
        self, forward_output: dict[str, Any], scheduler: _SchedulerCore
    ) -> None:
        """Route per-request forward outputs to the correct processing logic."""
        fb = self._require_forward_batch()
        qsl = fb.query_start_loc
        ids = np.asarray(forward_output["next_token_ids"], dtype=np.int32)

        if self._mixed and self.extend_states and self.decode_states:
            n_ext = len(self.extend_states)
            scheduler._process_extend_output(
                self.extend_states, qsl[: n_ext + 1], ids[:n_ext]
            )
            scheduler._process_decode_output(self.decode_states, ids[n_ext:])
        elif self.extend_states:
            scheduler._process_extend_output(self.extend_states, qsl, ids)
        else:
            scheduler._process_decode_output(self.decode_states, ids)

        # Process device-computed logprobs if present in forward output.
        if "chosen_logprobs" in forward_output:
            all_states = (
                self.extend_states + self.decode_states
                if self._mixed
                else (self.extend_states or self.decode_states)
            )
            scheduler._process_logprobs_output(all_states, forward_output)

    # -- Private build helpers --

    def _all_states(self) -> list[_RequestState]:
        """Combined state list in the order used for batch building."""
        if self._mixed:
            return self.extend_states + self.decode_states
        return self.extend_states or self.decode_states

    def _state_is_greedy(self, state: _RequestState) -> bool:
        params = state.sampling_params
        return params is None or int(params.top_k) == 1

    def _use_full_sampler(self, states: list[_RequestState]) -> bool:
        return any(not self._state_is_greedy(state) for state in states)

    def _build_sampling_vectors(
        self,
        states: list[_RequestState],
        sample_mask: np.ndarray,
        *,
        use_full_sampler: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = len(states)
        temperatures = np.ones((batch_size,), dtype=np.float32)
        top_ks = np.ones((batch_size,), dtype=np.int32)
        top_ps = np.ones((batch_size,), dtype=np.float32)
        min_ps = np.zeros((batch_size,), dtype=np.float32)
        uniform_u = np.zeros((batch_size,), dtype=np.float32)

        for i, state in enumerate(states):
            params = state.sampling_params
            if params is None:
                continue
            temperatures[i] = np.float32(params.temperature)
            top_ks[i] = np.int32(params.top_k)
            top_ps[i] = np.float32(params.top_p)
            min_ps[i] = np.float32(params.min_p)
            if use_full_sampler and bool(sample_mask[i]):
                uniform_u[i] = draw_uniform(
                    state.sampling_seed,
                    len(state.generated_ids),
                )
        return temperatures, top_ks, top_ps, min_ps, uniform_u

    def _build_extend_batch(self) -> ForwardBatch:
        return self._build_batch_impl(self.extend_states, ForwardMode.EXTEND)

    def _build_decode_batch(self) -> ForwardBatch:
        return self._build_decode_batch_fast(self.decode_states)

    def _build_decode_batch_fast(self, states: list[_RequestState]) -> ForwardBatch:
        """Optimized batch builder for pure DECODE (all 1-token contributions)."""
        batch_size = len(states)

        input_ids_raw = np.empty(batch_size, dtype=np.int32)
        positions_raw = np.empty(batch_size, dtype=np.int32)
        slot_mappings_raw = np.empty(batch_size, dtype=np.int64)
        seq_lens = np.empty(batch_size, dtype=np.int64)
        sample_mask = np.ones(batch_size, dtype=np.bool_)
        state_owner_ids = np.empty(batch_size, dtype=np.int32)

        max_blocks = 0
        for state in states:
            n_blocks = (state.seq_len + self._block_size - 1) // self._block_size
            if n_blocks > max_blocks:
                max_blocks = n_blocks

        block_tables = np.zeros((batch_size, max(max_blocks, 1)), dtype=np.int64)

        for i, state in enumerate(states):
            input_ids_raw[i] = state.last_token_id
            positions_raw[i] = state.seq_len - 1
            slot_idx = state.seq_len - 1
            slots = self._require_cache_slots(state)
            slot_mappings_raw[i] = int(slots[slot_idx])
            seq_lens[i] = state.seq_len
            state_owner_ids[i] = int(state.req_pool_idx)
            block_tables[i] = _slots_to_block_table(
                slots[: state.seq_len],
                self._block_size,
                max(max_blocks, 1),
            )

        query_start_loc = np.arange(batch_size + 1, dtype=np.int64)

        token_bucket = select_bucket(batch_size, self._paddings.bs_paddings, "request")

        input_ids = np.zeros(token_bucket, dtype=np.int32)
        input_ids[:batch_size] = input_ids_raw
        positions = np.zeros(token_bucket, dtype=np.int32)
        positions[:batch_size] = positions_raw
        slot_mapping = np.zeros(token_bucket, dtype=np.int64)
        slot_mapping[:batch_size] = slot_mappings_raw
        needs_logprobs = any(s.logprobs_k > 0 for s in states)
        logprobs_k = max((s.logprobs_k for s in states), default=0)
        use_full_sampler = self._use_full_sampler(states)
        # Force device sampler when logprobs are requested (needs all-gather).
        if needs_logprobs and not use_full_sampler:
            use_full_sampler = True
        temperatures, top_ks, top_ps, min_ps, uniform_u = self._build_sampling_vectors(
            states,
            sample_mask,
            use_full_sampler=use_full_sampler,
        )

        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            sample_mask=sample_mask,
            requested_topk=self._requested_topk,
            token_bucket=token_bucket,
            real_total_tokens=batch_size,
            use_full_sampler=use_full_sampler,
            needs_logprobs=needs_logprobs,
            logprobs_k=logprobs_k,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            uniform_u=uniform_u,
            state_owner_ids=state_owner_ids,
            attention_lane=self.attention_lane,
        )

    def _build_mixed_batch(self) -> ForwardBatch:
        """Build a single EXTEND-mode batch with extend states first, then
        decode states each contributing 1 token (decode-as-extend)."""
        combined = self.extend_states + self.decode_states
        return self._build_batch_impl(combined, ForwardMode.EXTEND)

    def _build_batch_impl(
        self, states: list[_RequestState], mode: ForwardMode
    ) -> ForwardBatch:
        """Core batch builder.  Moved from _SchedulerCore._build_forward_batch.

        For EXTEND mode: if a state has extend_done=True (decode-as-extend in
        mixed batch), it contributes 1 token at position seq_len-1.
        """
        batch_size = len(states)
        all_input_ids: list[np.ndarray] = []
        all_positions: list[np.ndarray] = []
        all_slot_mappings: list[np.ndarray] = []
        seq_lens_list: list[int] = []
        sample_mask_list: list[bool] = []
        state_owner_list: list[int] = []
        query_offsets: list[int] = [0]

        def _effective_seq_len_for_batch(state: _RequestState) -> int:
            if mode != ForwardMode.EXTEND or state.extend_done:
                return int(state.seq_len)
            full_prompt_len = int(state.prompt_ids.size)
            if full_prompt_len == 0:
                return 1
            offset = int(state.extend_offset)
            remaining = full_prompt_len - offset
            chunk_size = self._chunked_prefill_size
            if chunk_size > 0 and remaining > chunk_size:
                chunk_len = chunk_size
            else:
                chunk_len = remaining
            return max(offset + chunk_len, 1)

        max_blocks = 0
        effective_seq_lens = []
        for state in states:
            effective_seq_len = _effective_seq_len_for_batch(state)
            effective_seq_lens.append(effective_seq_len)
            n_blocks = (effective_seq_len + self._block_size - 1) // self._block_size
            if n_blocks > max_blocks:
                max_blocks = n_blocks

        block_tables = np.zeros((batch_size, max(max_blocks, 1)), dtype=np.int64)

        for i, state in enumerate(states):
            seq_len = state.seq_len
            effective_seq_len = effective_seq_lens[i]
            slots = self._require_cache_slots(state)

            if mode == ForwardMode.EXTEND:
                if state.extend_done:
                    # Decode-as-extend: 1 token (mixed batch).
                    all_input_ids.append(
                        np.asarray([state.last_token_id], dtype=np.int32)
                    )
                    all_positions.append(np.asarray([seq_len - 1], dtype=np.int32))
                    slot_idx = seq_len - 1
                    all_slot_mappings.append(
                        np.asarray([int(slots[slot_idx])], dtype=np.int64)
                    )
                    seq_lens_list.append(seq_len)
                    query_offsets.append(query_offsets[-1] + 1)
                    sample_mask_list.append(True)
                    state_owner_list.append(int(state.req_pool_idx))
                else:
                    # Prompt tokens, possibly chunked.
                    full_prompt_len = int(state.prompt_ids.size)
                    if full_prompt_len == 0:
                        full_prompt_len = 1
                        state.prompt_ids = np.asarray([0], dtype=np.int32)
                    offset = state.extend_offset
                    remaining = full_prompt_len - offset
                    chunk_size = self._chunked_prefill_size
                    if chunk_size > 0 and remaining > chunk_size:
                        chunk_len = chunk_size
                    else:
                        chunk_len = remaining
                    prompt_ids = state.prompt_ids[offset : offset + chunk_len]
                    all_input_ids.append(prompt_ids.astype(np.int32))
                    all_positions.append(
                        np.arange(offset, offset + chunk_len, dtype=np.int32)
                    )
                    all_slot_mappings.append(
                        slots[offset : offset + chunk_len].astype(np.int64)
                    )
                    seq_lens_list.append(offset + chunk_len)
                    query_offsets.append(query_offsets[-1] + chunk_len)
                    sample_mask_list.append(offset + chunk_len >= full_prompt_len)
                    state_owner_list.append(int(state.req_pool_idx))
            else:
                # Single decode token.
                all_input_ids.append(np.asarray([state.last_token_id], dtype=np.int32))
                all_positions.append(np.asarray([seq_len - 1], dtype=np.int32))
                slot_idx = seq_len - 1
                all_slot_mappings.append(
                    np.asarray([int(slots[slot_idx])], dtype=np.int64)
                )
                seq_lens_list.append(seq_len)
                query_offsets.append(query_offsets[-1] + 1)
                sample_mask_list.append(True)
                state_owner_list.append(int(state.req_pool_idx))

            # Build block table for this request.
            block_tables[i] = _slots_to_block_table(
                slots[:effective_seq_len],
                self._block_size,
                max(max_blocks, 1),
            )

        raw_input_ids = np.concatenate(all_input_ids)
        raw_positions = np.concatenate(all_positions)
        raw_slot_mapping = np.concatenate(all_slot_mappings)
        total_tokens = raw_input_ids.shape[0]

        # Use token buckets for the compute surface; batch padding only affects
        # request metadata and sampling rows.
        token_bucket = select_bucket(
            total_tokens, self._paddings.token_paddings, "token"
        )

        # Pad to token_bucket.
        input_ids = np.zeros(token_bucket, dtype=np.int32)
        input_ids[:total_tokens] = raw_input_ids
        positions = np.zeros(token_bucket, dtype=np.int32)
        positions[:total_tokens] = raw_positions
        slot_mapping = np.zeros(token_bucket, dtype=np.int64)
        slot_mapping[:total_tokens] = raw_slot_mapping
        sample_mask = np.asarray(sample_mask_list, dtype=np.bool_)
        needs_logprobs = any(s.logprobs_k > 0 for s in states)
        logprobs_k = max((s.logprobs_k for s in states), default=0)
        use_full_sampler = self._use_full_sampler(states)
        if needs_logprobs and not use_full_sampler:
            use_full_sampler = True
        temperatures, top_ks, top_ps, min_ps, uniform_u = self._build_sampling_vectors(
            states,
            sample_mask,
            use_full_sampler=use_full_sampler,
        )

        return ForwardBatch(
            forward_mode=mode,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            seq_lens=np.asarray(seq_lens_list, dtype=np.int64),
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            query_start_loc=np.asarray(query_offsets, dtype=np.int64),
            sample_mask=sample_mask,
            requested_topk=self._requested_topk,
            token_bucket=token_bucket,
            real_total_tokens=total_tokens,
            use_full_sampler=use_full_sampler,
            needs_logprobs=needs_logprobs,
            logprobs_k=logprobs_k,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            uniform_u=uniform_u,
            state_owner_ids=np.asarray(state_owner_list, dtype=np.int32),
            attention_lane=self.attention_lane,
        )


class DpAttentionSuperstepBatch:
    """Replica-local DP-attention superstep built from lane-local batches.

    Attention remains lane-local, but MoE/FFN must see one synchronized token
    layout across all lanes in the replica. This wrapper preserves each
    lane-local `ScheduleBatch` for result routing while exposing one combined
    `ForwardBatch` with lane offset metadata for the model path.
    """

    def __init__(
        self,
        lane_batches: list[ScheduleBatch],
        *,
        num_lanes: int,
        paddings: PrecompilePaddings,
    ):
        if not lane_batches:
            raise ValueError("lane_batches must not be empty")
        lanes = [int(batch.attention_lane) for batch in lane_batches]
        if any(lane < 0 or lane >= int(num_lanes) for lane in lanes):
            raise ValueError(
                f"lane_batches must be in [0, {int(num_lanes)}), got {lanes}"
            )
        if len(set(lanes)) != len(lanes):
            raise ValueError(f"lane_batches must have unique lanes, got {lanes}")
        self._lane_batches = sorted(lane_batches, key=lambda b: b.attention_lane)
        self._num_lanes = int(num_lanes)
        self._paddings = paddings
        self.extend_states = [
            state for batch in self._lane_batches for state in batch.extend_states
        ]
        self.decode_states = [
            state for batch in self._lane_batches for state in batch.decode_states
        ]
        self.attention_lane = -1
        self._forward_batch: ForwardBatch | None = None

    @property
    def is_empty(self) -> bool:
        return not self.extend_states and not self.decode_states

    @property
    def lane_batches(self) -> tuple[ScheduleBatch, ...]:
        return tuple(self._lane_batches)

    def build_forward_batch(self) -> ForwardBatch:
        lane_fbs = [
            (batch.attention_lane, batch.build_forward_batch())
            for batch in self._lane_batches
        ]
        modes = {fb.forward_mode for _, fb in lane_fbs}
        if len(modes) != 1:
            raise ValueError(f"DP-attention superstep got mixed modes: {modes}")
        mode = lane_fbs[0][1].forward_mode

        token_counts = np.zeros((self._num_lanes,), dtype=np.int32)
        batch_sizes = np.zeros((self._num_lanes,), dtype=np.int32)
        lane_by_id = {lane: fb for lane, fb in lane_fbs}

        input_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        slot_parts: list[np.ndarray] = []
        seq_lens_parts: list[np.ndarray] = []
        sample_mask_parts: list[np.ndarray] = []
        state_owner_parts: list[np.ndarray] = []
        temp_parts: list[np.ndarray] = []
        top_k_parts: list[np.ndarray] = []
        top_p_parts: list[np.ndarray] = []
        min_p_parts: list[np.ndarray] = []
        uniform_parts: list[np.ndarray] = []
        query_offsets: list[int] = [0]
        max_block_width = max(int(fb.block_tables.shape[1]) for _, fb in lane_fbs)
        block_tables_parts: list[np.ndarray] = []

        for lane in range(self._num_lanes):
            fb = lane_by_id.get(lane)
            if fb is None:
                continue
            real_tokens = int(fb.real_total_tokens)
            bs = int(fb.batch_size)
            token_counts[lane] = real_tokens
            batch_sizes[lane] = bs
            input_parts.append(fb.input_ids[:real_tokens])
            position_parts.append(fb.positions[:real_tokens])
            slot_parts.append(fb.slot_mapping[:real_tokens])
            seq_lens_parts.append(fb.seq_lens)
            sample_mask_parts.append(fb.sample_mask)
            state_owner_parts.append(fb.state_owner_ids)
            temp_parts.append(fb.temperatures)
            top_k_parts.append(fb.top_ks)
            top_p_parts.append(fb.top_ps)
            min_p_parts.append(fb.min_ps)
            uniform_parts.append(fb.uniform_u)

            q_lens = np.diff(fb.query_start_loc.astype(np.int64, copy=False))
            for q_len in q_lens:
                query_offsets.append(query_offsets[-1] + int(q_len))

            block_tables = np.zeros((bs, max_block_width), dtype=np.int64)
            width = int(fb.block_tables.shape[1])
            block_tables[:, :width] = fb.block_tables
            block_tables_parts.append(block_tables)

        total_tokens = int(token_counts.sum())
        batch_size = int(batch_sizes.sum())
        if mode == ForwardMode.DECODE:
            token_bucket = select_bucket(
                max(total_tokens, 1), self._paddings.bs_paddings, "request"
            )
        else:
            token_bucket = select_bucket(
                max(total_tokens, 1), self._paddings.token_paddings, "token"
            )

        input_ids = np.zeros((token_bucket,), dtype=np.int32)
        positions = np.zeros((token_bucket,), dtype=np.int32)
        slot_mapping = np.zeros((token_bucket,), dtype=np.int64)
        if total_tokens > 0:
            input_ids[:total_tokens] = np.concatenate(input_parts).astype(
                np.int32, copy=False
            )
            positions[:total_tokens] = np.concatenate(position_parts).astype(
                np.int32, copy=False
            )
            slot_mapping[:total_tokens] = np.concatenate(slot_parts).astype(
                np.int64, copy=False
            )

        lane_token_offsets = np.zeros((self._num_lanes + 1,), dtype=np.int32)
        lane_batch_offsets = np.zeros((self._num_lanes + 1,), dtype=np.int32)
        lane_token_offsets[1:] = np.cumsum(token_counts, dtype=np.int32)
        lane_batch_offsets[1:] = np.cumsum(batch_sizes, dtype=np.int32)

        fb = ForwardBatch(
            forward_mode=mode,
            batch_size=batch_size,
            input_ids=input_ids,
            positions=positions,
            seq_lens=np.concatenate(seq_lens_parts).astype(np.int64, copy=False),
            slot_mapping=slot_mapping,
            block_tables=np.concatenate(block_tables_parts, axis=0).astype(
                np.int64, copy=False
            ),
            query_start_loc=np.asarray(query_offsets, dtype=np.int64),
            sample_mask=np.concatenate(sample_mask_parts).astype(np.bool_, copy=False),
            requested_topk=max(int(fb.requested_topk) for _, fb in lane_fbs),
            token_bucket=token_bucket,
            real_total_tokens=total_tokens,
            use_full_sampler=any(bool(fb.use_full_sampler) for _, fb in lane_fbs),
            needs_logprobs=any(bool(fb.needs_logprobs) for _, fb in lane_fbs),
            logprobs_k=max(int(fb.logprobs_k) for _, fb in lane_fbs),
            temperatures=np.concatenate(temp_parts).astype(np.float32, copy=False),
            top_ks=np.concatenate(top_k_parts).astype(np.int32, copy=False),
            top_ps=np.concatenate(top_p_parts).astype(np.float32, copy=False),
            min_ps=np.concatenate(min_p_parts).astype(np.float32, copy=False),
            uniform_u=np.concatenate(uniform_parts).astype(np.float32, copy=False),
            state_owner_ids=np.concatenate(state_owner_parts).astype(
                np.int32, copy=False
            ),
            attention_lane=-1,
            dp_attention_superstep=True,
            dp_attention_num_lanes=self._num_lanes,
            dp_attention_lane_token_counts=token_counts,
            dp_attention_lane_batch_sizes=batch_sizes,
            dp_attention_lane_token_offsets=lane_token_offsets,
            dp_attention_lane_batch_offsets=lane_batch_offsets,
        )
        self._forward_batch = fb
        return fb

    def process_results(
        self,
        forward_output: dict[str, Any],
        scheduler: _SchedulerCore,
    ) -> None:
        fb = self._forward_batch
        if fb is None:
            raise RuntimeError(
                "DpAttentionSuperstepBatch.process_results called before "
                "build_forward_batch"
            )
        batch_offsets = fb.dp_attention_lane_batch_offsets
        for batch in self._lane_batches:
            lane = int(batch.attention_lane)
            start = int(batch_offsets[lane])
            end = int(batch_offsets[lane + 1])
            lane_out: dict[str, Any] = {
                "next_token_ids": np.asarray(
                    forward_output["next_token_ids"], dtype=np.int32
                )[start:end],
            }
            if "chosen_logprobs" in forward_output:
                lane_out["chosen_logprobs"] = np.asarray(
                    forward_output["chosen_logprobs"], dtype=np.float32
                )[start:end]
                lane_out["topk_logprob_vals"] = np.asarray(
                    forward_output["topk_logprob_vals"], dtype=np.float32
                )[start:end]
                lane_out["topk_logprob_ids"] = np.asarray(
                    forward_output["topk_logprob_ids"], dtype=np.int32
                )[start:end]
            batch.process_results(lane_out, scheduler)


# ---------------------------------------------------------------------------
# Scheduler core
# ---------------------------------------------------------------------------


class _SchedulerCore:
    def __init__(
        self,
        manager: SchedulerTokenizerService,
        runtime_config: RuntimeConfig,
        response_queue,
        kv_pool,
        req_to_token_pool: ReqToTokenPool,
        token_allocator: BaseTokenToKVPoolAllocator,
        worker_coordinator: WorkerCoordinator,
        prefix_cache: BasePrefixCache | None = None,
        paddings: PrecompilePaddings | None = None,
        eos_token_ids: set[int] | None = None,
    ):
        self.manager = manager
        self.runtime_config = runtime_config
        self.response_queue = response_queue
        self.kv_pool = kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.token_allocator = token_allocator
        self.worker_coordinator = worker_coordinator
        self.prefix_cache = prefix_cache
        self._eos_token_ids: set[int] = (
            eos_token_ids if eos_token_ids is not None else set()
        )

        self.waiting_queue: list[_RequestState] = []
        self.running_batch: list[_RequestState] = []
        self.requests_by_id: dict[str, _RequestState] = {}
        self._request_state_checkpoints: dict[str, _RequestStateCheckpoint] = {}

        self.paused: bool = False
        self.shutdown_requested: bool = False
        self.metrics = _SchedulerMetrics()

        self._block_size = kv_pool.block_size
        self._max_batch_size = runtime_config.max_requests
        self._chunked_prefill_size = (
            runtime_config.chunked_prefill_size
        )  # -1 = disabled
        self._is_mixed_chunk = (
            self._chunked_prefill_size > 0 and runtime_config.enable_mixed_chunk
        )
        self._request_timeout_s = runtime_config.request_timeout_s
        self._paddings = (
            paddings
            if paddings is not None
            else build_precompile_paddings(runtime_config)
        )
        self._waiting_queue_dirty = False  # Track if waiting queue needs re-sort
        self._clear_request_state_on_free = _runtime_uses_dsv4_request_state(
            runtime_config,
        )

        # Throughput measurement state.
        self._last_decode_tic: float = time.time()
        self._decode_tokens_since_last: int = 0
        self._last_prefill_tic: float = time.time()
        self._prefill_tokens_since_last: int = 0
        self._metrics_update_interval: int = 100

        # Pending token outputs batched per step for the detokenizer.
        self._pending_token_outputs: list[dict[str, Any]] = []

        # Overlap scheduling state.
        self._overlap_enabled = bool(runtime_config.overlap_schedule)

        # DP-attention lane state. When attention_dp_degree == 1,
        # _assign_next_lane() returns -1.
        self._attention_dp_degree: int = int(runtime_config.attention_dp_degree)
        self._next_lane_cursor: int = 0
        self._lane_pressure_model = _build_dsv4_kv_pressure_model(runtime_config)
        self._init_lane_metrics()

        # Profiling (gated by NKIPY_SERVING_PROFILE=1).
        self._profile_writer: ProfileWriter | None = None
        self._ipc_profile_writer: ProfileWriter | None = None
        if PROFILING_ENABLED:
            self._profile_writer = ProfileWriter("scheduler_steps")
            self._ipc_profile_writer = ProfileWriter("ipc_breakdown")

    def _prefix_cache_available(self) -> bool:
        """Whether generic KV prefix reuse is safe for this runtime.

        DeepSeek-V4 owns additional per-request device state outside the
        generic KV pool (SWA/compressor/indexer rings). Reusing only generic
        KV slots from the prefix cache would leave that state inconsistent, so
        reuse/donation stays disabled until an explicit snapshot/restore path
        exists.
        """
        return self.prefix_cache is not None and not self._clear_request_state_on_free

    def _init_lane_metrics(self) -> None:
        if self._attention_dp_degree <= 1:
            return
        zeros = [0 for _ in range(self._attention_dp_degree)]
        self.metrics.lane_submitted = list(zeros)
        self.metrics.lane_completed = list(zeros)
        self.metrics.lane_aborted = list(zeros)
        self.metrics.lane_errors = list(zeros)

    def _record_lane_metric(self, field_name: str, lane: int) -> None:
        if self._attention_dp_degree <= 1:
            return
        if lane < 0 or lane >= self._attention_dp_degree:
            raise RuntimeError(
                f"attention lane {lane} out of range for "
                f"attention_dp_degree={self._attention_dp_degree}"
            )
        counters = getattr(self.metrics, field_name)
        if len(counters) != self._attention_dp_degree:
            setattr(
                self.metrics, field_name, [0 for _ in range(self._attention_dp_degree)]
            )
            counters = getattr(self.metrics, field_name)
        counters[lane] += 1

    def _lane_count_vector(self, states: list[_RequestState]) -> list[int]:
        counts = [0 for _ in range(max(0, self._attention_dp_degree))]
        if self._attention_dp_degree <= 1:
            return []
        for state in states:
            lane = int(state.attention_lane)
            if 0 <= lane < self._attention_dp_degree:
                counts[lane] += 1
        return counts

    def _request_token_pressure(self, state: _RequestState) -> int:
        """Rounded generic token-slot pressure for one request."""
        if state.out_cache_loc is not None:
            return int(state.out_cache_loc.size)
        prompt_len = max(1, int(state.prompt_ids.size))
        try:
            max_new = int(getattr(state.req, "max_new_tokens", 0))
        except (TypeError, ValueError):
            max_new = 0
        total = min(
            prompt_len + max(0, max_new), self.req_to_token_pool.max_context_len
        )
        return ((total + self._block_size - 1) // self._block_size) * self._block_size

    def _max_request_token_capacity(self) -> int:
        """Maximum token slots one request can ever allocate.

        ``max_context_len`` can exceed the physical KV pool for long-context
        configs that rely on chunk/bucket coverage. Admission must cap by the
        usable allocator capacity too; otherwise a request with an oversized
        generation budget waits forever for slots that can never exist.
        """
        context_limit = max(0, int(self.req_to_token_pool.max_context_len))
        allocator_size = int(getattr(self.token_allocator, "size", context_limit) or 0)
        page_size = max(
            1, int(getattr(self.token_allocator, "page_size", self._block_size) or 1)
        )
        num_pages = getattr(self.token_allocator, "num_pages", None)
        if num_pages is not None:
            # Paged allocators reserve page 0 so token index 0 remains a sink.
            allocator_limit = max(0, (int(num_pages) - 1) * page_size)
        else:
            allocator_limit = max(0, allocator_size)
        limit = min(context_limit, allocator_limit)
        block = max(1, int(self._block_size))
        return (int(limit) // block) * block

    def _cap_max_new_tokens_to_capacity(
        self,
        req: GenerateReqInput,
        *,
        prompt_len: int,
    ) -> None:
        total_limit = int(self._max_request_token_capacity())
        prompt_tokens = max(1, int(prompt_len))
        if total_limit <= prompt_tokens:
            req.max_new_tokens = 0
            return
        max_new_cap = total_limit - prompt_tokens
        if int(req.max_new_tokens) > max_new_cap:
            req.max_new_tokens = int(max_new_cap)

    def _request_lane_pressure(self, state: _RequestState) -> int:
        token_slots = self._request_token_pressure(state)
        model = getattr(self, "_lane_pressure_model", None)
        if model is None:
            return token_slots
        return model.estimate_bytes(token_slots)

    def _lane_pressure_vector(self, states: list[_RequestState]) -> list[int]:
        pressure = [0 for _ in range(max(0, self._attention_dp_degree))]
        if self._attention_dp_degree <= 1:
            return []
        for state in states:
            lane = int(state.attention_lane)
            if 0 <= lane < self._attention_dp_degree:
                pressure[lane] += self._request_lane_pressure(state)
        return pressure

    def _lane_inflight_pressure(self) -> list[int]:
        if self._attention_dp_degree <= 1:
            return []
        return self._lane_pressure_vector(list(self.requests_by_id.values()))

    def _lane_pressure_model_payload(self) -> dict[str, Any] | None:
        model = getattr(self, "_lane_pressure_model", None)
        if model is None:
            return None
        owner_swa_slots = int(self._max_batch_size) * int(model.sliding_window)
        kv_slots = int(getattr(self.kv_pool, "size", 0))
        num_slots_per_layer = max(
            kv_slots + 1,
            int(self._block_size) + 1,
            owner_swa_slots + 1,
        )
        state_size = int(getattr(self.runtime_config, "dsv4_state_size", 0))
        if state_size <= 0:
            raise RuntimeError(
                "dsv4_state_size must be > 0 when using the DSV4 pressure model"
            )
        payload = model.summary(
            max_requests=int(self._max_batch_size),
            num_slots_per_layer=num_slots_per_layer,
            max_seq_len=state_size,
        )
        return payload

    def _lane_metrics_payload(self) -> dict[str, Any]:
        if self._attention_dp_degree <= 1:
            return {}
        inflight_states = list(self.requests_by_id.values())
        model = getattr(self, "_lane_pressure_model", None)
        pressure_unit = (
            model.pressure_unit if model is not None else "rounded_kv_token_slots"
        )
        routing_policy = (
            model.routing_policy
            if model is not None
            else "least_token_pressure_round_robin_tie_break"
        )
        waiting_pressure = self._lane_pressure_vector(self.waiting_queue)
        running_pressure = self._lane_pressure_vector(self.running_batch)
        inflight_pressure = self._lane_pressure_vector(inflight_states)
        return {
            "attention_dp_degree": int(self._attention_dp_degree),
            "routing_policy": routing_policy,
            "pressure_unit": pressure_unit,
            "pressure_model": self._lane_pressure_model_payload(),
            "next_lane_cursor": int(self._next_lane_cursor % self._attention_dp_degree),
            "submitted": list(self.metrics.lane_submitted),
            "completed": list(self.metrics.lane_completed),
            "aborted": list(self.metrics.lane_aborted),
            "errors": list(self.metrics.lane_errors),
            "waiting": self._lane_count_vector(self.waiting_queue),
            "running": self._lane_count_vector(self.running_batch),
            "inflight": self._lane_count_vector(inflight_states),
            "waiting_pressure": waiting_pressure,
            "running_pressure": running_pressure,
            "inflight_pressure": inflight_pressure,
        }

    # -- Lane assignment --

    def _assign_next_lane(self) -> int:
        """Least-pressure lane assignment with round-robin tie-break.

        Returns -1 when `attention_dp_degree == 1`. DSV4 local configs use
        estimated mutable KV/state bytes; generic configs fall back to rounded
        token slots.
        """
        if self._attention_dp_degree <= 1:
            return -1
        pressure = self._lane_inflight_pressure()
        if not pressure:
            lane = self._next_lane_cursor % self._attention_dp_degree
            self._next_lane_cursor = (lane + 1) % self._attention_dp_degree
            return lane
        min_pressure = min(pressure)
        start = self._next_lane_cursor % self._attention_dp_degree
        lane = next(
            candidate
            for offset in range(self._attention_dp_degree)
            for candidate in ((start + offset) % self._attention_dp_degree,)
            if pressure[candidate] == min_pressure
        )
        self._next_lane_cursor = (lane + 1) % self._attention_dp_degree
        return lane

    # -- Response helpers --

    def _flush_token_outputs(self) -> None:
        """Send pending token outputs to the detokenizer as a batch."""
        if not self._pending_token_outputs:
            return
        self._send_response(
            {
                "type": "batch_tokens",
                "outputs": self._pending_token_outputs,
            }
        )
        self._pending_token_outputs = []

    def _send_response(self, msg: dict[str, Any]) -> None:
        """Send a response message via the response channel (ZMQ or queue)."""
        if hasattr(self.response_queue, "send_pyobj"):
            self.response_queue.send_pyobj(msg)
        else:
            self.response_queue.put(msg)

    def _control_response(
        self,
        control_id: str | None,
        ok: bool,
        **payload: Any,
    ) -> None:
        if control_id is None:
            return
        out = {"control_id": str(control_id), "ok": bool(ok)}
        out.update(payload)
        self._send_response(out)

    def _finalize_request_success(self, state: _RequestState) -> None:
        # Trim stop token from generated_ids when finish_reason is "stop".
        if (
            state.finish_reason == "stop"
            and not state.no_stop_trim
            and state.generated_ids
            and state.stop_token_ids
            and state.generated_ids[-1] in state.stop_token_ids
        ):
            state.generated_ids.pop()

        finish_reason = "score" if state.req.score else state.finish_reason

        # Send finish message to detokenizer for text decode + response formatting.
        finish_msg: dict[str, Any] = {
            "type": "finish",
            "request_id": state.request_id,
            "generated_ids": list(state.generated_ids),
            "prompt_ids": state.prompt_ids.tolist(),
            "finish_reason": finish_reason,
            "stop_strs": list(state.stop_strs),
            "no_stop_trim": bool(state.no_stop_trim),
            "first_scheduled_ts": float(state.first_scheduled_ts),
            "first_token_ts": float(state.first_token_ts),
            "cached_tokens": int(state.prefix_hit_length),
            "metadata": asdict(state.req),
        }
        if state.return_logprob:
            finish_msg["logprob_data"] = {
                "return_text_in_logprobs": bool(state.req.return_text_in_logprobs),
                "top_logprobs_num": int(state.top_logprobs_num),
                "logprob_start_len": int(state.logprob_start_len),
                "token_logprobs": state.token_logprobs,
                "top_logprobs": state.top_logprobs,
                "input_token_logprobs": state.input_token_logprobs
                if state.logprob_start_len >= 0
                else None,
                "input_top_logprobs": state.input_top_logprobs
                if state.logprob_start_len >= 0 and state.top_logprobs_num > 0
                else None,
            }
        self._send_response(finish_msg)
        self.metrics.total_completed += 1
        self._record_lane_metric("lane_completed", int(state.attention_lane))
        self.requests_by_id.pop(state.request_id, None)

    def _finalize_request_error(
        self, state: _RequestState, error: str, aborted: bool
    ) -> None:
        self._send_response(
            {
                "type": "final",
                "request_id": state.request_id,
                "ok": False,
                "error": str(error),
                "aborted": aborted,
                "finish_reason": "abort" if aborted else "error",
            }
        )
        if aborted:
            self.metrics.total_aborted += 1
            self._record_lane_metric("lane_aborted", int(state.attention_lane))
        else:
            self.metrics.total_errors += 1
            self._record_lane_metric("lane_errors", int(state.attention_lane))
        self.requests_by_id.pop(state.request_id, None)

    def _remaining_tokens(self, state: _RequestState) -> int:
        return max(0, int(state.req.max_new_tokens) - len(state.generated_ids))

    def _check_stop_strings(self, state: _RequestState) -> bool:
        if not state.stop_strs or not state.decoded_text:
            return False
        # Search the tail of the accumulated decoded text (no tokenizer call needed).
        # Buffer beyond max stop-string length accounts for multi-character tokens
        # that may have been appended in a single commit.
        tail_window = state.stop_str_max_char_len + 16
        tail_text = state.decoded_text[-tail_window:]
        for stop_str in state.stop_strs:
            if stop_str in tail_text:
                state.finish_reason = "stop"
                return True
        return False

    def _state_is_finished(self, state: _RequestState) -> bool:
        if not state.extend_done:
            return False
        if self._remaining_tokens(state) <= 0:
            state.finish_reason = "length"
            return True
        # Check stop_token_ids (O(1) set lookup) before stop strings.
        if (
            state.stop_token_ids
            and state.generated_ids
            and state.generated_ids[-1] in state.stop_token_ids
        ):
            state.finish_reason = "stop"
            return True
        if self._check_stop_strings(state):
            return True
        return False

    def _emit_token(self, state: _RequestState, token_id: int) -> None:
        state.generated_ids.append(int(token_id))
        self.metrics.total_generated_tokens += 1

        # TTFT tracking: first generated token.
        if len(state.generated_ids) == 1:
            now = time.time()
            state.first_token_ts = now
            ttft = now - state.submitted_at
            self.metrics.total_time_to_first_token_s += ttft
            self.metrics.total_requests_with_ttft += 1

        # Decode for stop-string check only (full decode in detokenizer).
        if state.stop_strs:
            suffix_ids = np.asarray(
                state.generated_ids[state.decode_offset :], dtype=np.int32
            )
            new_text = self.manager.decode_ids(suffix_ids)
            if new_text and not new_text.endswith("\ufffd"):
                state.decoded_text += new_text
                state.decode_offset = len(state.generated_ids)

        # Send token event to detokenizer (fire-and-forget).
        self._pending_token_outputs.append(
            {
                "request_id": state.request_id,
                "token_id": int(token_id),
                "stream": bool(state.stream),
            }
        )
        if state.stream:
            self.metrics.total_stream_events += 1

    def _logprob_token_text(
        self,
        state: _RequestState,
        token_id: int,
    ) -> str | None:
        """Always None — the detokenizer decodes logprob token text."""
        return None

    # -- Message handling --

    def _validate_request_input(self, prompt_ids: np.ndarray) -> str | None:
        """Validate input length against context and bucket limits.

        Returns an error message if validation fails, None if OK.
        """
        prompt_len = int(prompt_ids.size)
        max_context = self.req_to_token_pool.max_context_len

        if prompt_len >= max_context:
            return (
                f"Input length ({prompt_len} tokens) exceeds or equals "
                f"the maximum context length ({max_context} tokens)."
            )
        max_request_tokens = self._max_request_token_capacity()
        if max_request_tokens > 0 and prompt_len > max_request_tokens:
            return (
                f"Input length ({prompt_len} tokens) exceeds usable KV capacity "
                f"({max_request_tokens} tokens)."
            )

        if self._chunked_prefill_size <= 0:
            max_token_bucket = max(self._paddings.token_paddings)
            if prompt_len > max_token_bucket:
                return (
                    f"Input length ({prompt_len} tokens) exceeds "
                    f"the maximum token bucket ({max_token_bucket}) "
                    f"and chunked prefill is disabled."
                )

        return None

    def _handle_generate_cmd(self, msg: dict[str, Any]) -> None:
        request_id = str(msg.get("request_id", ""))
        if not request_id:
            return
        if request_id in self.requests_by_id:
            self._send_response(
                {
                    "type": "final",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"Duplicate request_id: {request_id}",
                }
            )
            return

        try:
            req_payload = msg.get("req", {})
            req = GenerateReqInput(**req_payload)
            if req.max_new_tokens < 0:
                raise RuntimeError(
                    f"max_new_tokens must be >= 0, got {req.max_new_tokens}"
                )
            if int(req.logprob_start_len) < -1:
                raise RuntimeError(
                    f"logprob_start_len must be >= -1, got {req.logprob_start_len}"
                )
            prompt_text = req.prompt if req.prompt is not None else req.text
            if req.input_ids is not None:
                # input_ids takes priority (the tokenizer manager pre-tokenizes
                # text prompts and sends both input_ids and prompt/text).
                prompt_ids = np.asarray(req.input_ids, dtype=np.int32)
            else:
                if prompt_text is None:
                    raise RuntimeError("Either prompt/text or input_ids is required")
                prompt_ids = self.manager.encode_prompt(prompt_text)
            if prompt_ids.size <= 0:
                if req.input_ids is not None:
                    raise RuntimeError("input_ids must encode to at least one token")
                raise RuntimeError("prompt must encode to at least one token")

            input_error = self._validate_request_input(prompt_ids)
            if input_error is not None:
                raise RuntimeError(input_error)
            self._cap_max_new_tokens_to_capacity(
                req,
                prompt_len=int(prompt_ids.size),
            )

            if req.score:
                raise RuntimeError(
                    "Score mode is not supported. Use logprobs on /generate instead."
                )
            if req.logprob_start_len >= 0:
                raise RuntimeError(
                    "Prompt logprobs (logprob_start_len >= 0) are not yet supported. "
                    "Output token logprobs are available via the logprobs parameter."
                )
            state = _RequestState(
                request_id=request_id,
                req=req,
                prompt_ids=prompt_ids,
                stream=bool(req.stream),
                attention_lane=self._assign_next_lane(),
            )
            sampling_params = SamplingParams(
                max_new_tokens=req.max_new_tokens,
                stop=req.stop,
                stop_token_ids=req.stop_token_ids,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                repetition_penalty=req.repetition_penalty,
                ignore_eos=req.ignore_eos,
                no_stop_trim=req.no_stop_trim,
                sampling_seed=req.seed,
            )
            tokenizer_like = getattr(self.manager, "tokenizer", self.manager)
            vocab_size = getattr(tokenizer_like, "vocab_size", None)
            if vocab_size is None:
                vocab_size = getattr(self.manager, "vocab_size", None)
            if vocab_size is not None:
                sampling_params.verify(int(vocab_size))
            state.sampling_params = sampling_params
            state.sampling_seed = assign_seed(sampling_params.sampling_seed)

            # Parse stop sequences.
            stop_raw = req.stop
            if stop_raw is not None:
                strs = [stop_raw] if isinstance(stop_raw, str) else list(stop_raw)
                state.stop_strs = [s for s in strs if s]
                if state.stop_strs:
                    state.stop_str_max_len = max(
                        len(self.manager.encode_prompt(s)) for s in state.stop_strs
                    )
                    state.stop_str_max_char_len = max(len(s) for s in state.stop_strs)

            # Parse stop_token_ids.
            raw_stop_ids = req.stop_token_ids
            if raw_stop_ids:
                state.stop_token_ids = set(int(t) for t in raw_stop_ids)

            # Merge EOS token IDs unless the request opted out.
            if not req.ignore_eos and self._eos_token_ids:
                state.stop_token_ids |= self._eos_token_ids

            # Parse no_stop_trim.
            state.no_stop_trim = bool(req.no_stop_trim)

            # Logprob semantics:
            # - return_logprob=True, top_logprobs_num=0 => chosen token only
            # - top_logprobs_num=N => chosen token + top-N
            state.return_logprob = bool(req.return_logprob)
            if int(req.top_logprobs_num) > 0:
                state.return_logprob = True
            if state.top_logprobs_num <= 0:
                state.top_logprobs_num = min(
                    max(0, int(req.top_logprobs_num)), LOGPROBS_K_MAX
                )
            state.logprob_start_len = int(req.logprob_start_len)
            if state.return_logprob:
                state.logprobs_k = max(1, state.top_logprobs_num)

            needs_prefill_scoring = bool(req.score or state.logprob_start_len >= 0)

            # Pre-compute prefix match at admission time (avoids re-matching every step).
            if self._prefix_cache_available() and not needs_prefill_scoring:
                match = self.prefix_cache.match_prefix(state.prompt_ids.tolist())
                state.prefix_hit_length = match.host_hit_length

            self.waiting_queue.append(state)
            self._waiting_queue_dirty = True
            self.requests_by_id[request_id] = state
            self.metrics.total_submitted += 1
            self._record_lane_metric("lane_submitted", int(state.attention_lane))

            if req.max_new_tokens == 0 and not needs_prefill_scoring:
                self.waiting_queue.pop()
                self._finalize_request_success(state)
        except Exception as exc:
            self._send_response(
                {
                    "type": "final",
                    "request_id": request_id,
                    "ok": False,
                    "error": repr(exc),
                }
            )

    def _filter_queue(
        self,
        queue: list[_RequestState],
        predicate,
        error: str,
        *,
        free_resources: bool = False,
    ) -> tuple[list[_RequestState], int]:
        """Remove states matching *predicate*, finalize them as errors. Returns (kept, removed_count)."""
        kept: list[_RequestState] = []
        removed = 0
        for state in queue:
            if predicate(state):
                if free_resources:
                    self._free_request_resources(state)
                self._finalize_request_error(state, error=error, aborted=True)
                removed += 1
            else:
                kept.append(state)
        return kept, removed

    def _abort_requests(self, request_id: str | None) -> int:
        pred = (
            (lambda s: True)
            if request_id is None
            else (lambda s: s.request_id == request_id)
        )
        msg = "Aborted by scheduler control command."
        self.waiting_queue, c1 = self._filter_queue(self.waiting_queue, pred, msg)
        self.running_batch, c2 = self._filter_queue(
            self.running_batch, pred, msg, free_resources=True
        )
        return c1 + c2

    def _ensure_no_active_requests(self) -> None:
        if self.waiting_queue or self.running_batch or self.requests_by_id:
            raise RuntimeError(
                "Cannot mutate runtime state while requests are active. "
                "Use abort_all_requests=true to force abort before reload/flush."
            )

    def _clear_runtime_caches(self) -> None:
        self.req_to_token_pool.clear()
        self.token_allocator.clear()
        self.kv_pool.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.reset()
        self.waiting_queue = []
        self.running_batch = []
        self.requests_by_id.clear()
        self._request_state_checkpoints.clear()
        self._waiting_queue_dirty = False

    def _flush_cache(self, *, abort_all_requests: bool) -> int:
        aborted_count = self._abort_requests(None) if abort_all_requests else 0
        if not abort_all_requests:
            self._ensure_no_active_requests()
        try:
            self.worker_coordinator.flush_cache()
        except Exception:
            self.paused = True
            raise
        self._clear_runtime_caches()
        return aborted_count

    def _reload_weights_from_disk(
        self,
        *,
        model_path: str,
        abort_all_requests: bool,
    ) -> int:
        aborted_count = self._abort_requests(None) if abort_all_requests else 0
        if not abort_all_requests:
            self._ensure_no_active_requests()
        try:
            self.worker_coordinator.reload_weights(model_path)
            self.worker_coordinator.flush_cache()
        except Exception:
            self.paused = True
            raise
        self._clear_runtime_caches()
        return aborted_count

    def _checkpoint_request_state(
        self,
        *,
        request_id: str,
        checkpoint_id: str | None,
        num_tokens: int,
    ) -> dict[str, Any]:
        """Checkpoint scheduler + DSV4 worker state for speculative rollback.

        The device snapshot is bounded to future positions
        ``[seq_len, seq_len + num_tokens)``. The scheduler snapshot is restored
        only before speculative tokens become user-visible; MTP should call this
        path around its internal draft/accept loop, not after normal decode
        tokens have been flushed to the detokenizer.
        """
        rid = str(request_id)
        if not rid:
            raise RuntimeError("request_id is required for checkpoint_request_state")
        ntokens = int(num_tokens)
        if ntokens < 0:
            raise RuntimeError("num_tokens must be non-negative")
        state = self.requests_by_id.get(rid)
        if state is None:
            raise RuntimeError(f"unknown active request_id: {rid}")
        if int(state.req_pool_idx) < 0:
            raise RuntimeError(
                f"request {rid} has no request-state owner yet; "
                "checkpoint after scheduler admission"
            )
        clean_id = str(checkpoint_id or f"{rid}:{time.time_ns()}")
        if not clean_id:
            raise RuntimeError("checkpoint_id must be non-empty")
        if clean_id in self._request_state_checkpoints:
            raise RuntimeError(f"duplicate request-state checkpoint_id: {clean_id}")

        checkpoint_fn = getattr(
            self.worker_coordinator, "checkpoint_request_state", None
        )
        if not callable(checkpoint_fn):
            raise RuntimeError(
                "worker coordinator does not support request-state checkpoint"
            )
        returned_id = checkpoint_fn(
            checkpoint_id=clean_id,
            owner_id=int(state.req_pool_idx),
            seq_len=int(state.seq_len),
            num_tokens=ntokens,
        )
        if returned_id is not None:
            clean_id = str(returned_id)

        self._request_state_checkpoints[clean_id] = _RequestStateCheckpoint(
            request_id=rid,
            owner_id=int(state.req_pool_idx),
            seq_len=int(state.seq_len),
            extend_done=bool(state.extend_done),
            extend_offset=int(state.extend_offset),
            generated_ids=list(state.generated_ids),
            decode_offset=int(state.decode_offset),
            decoded_text=str(state.decoded_text),
            finish_reason=str(state.finish_reason),
            first_token_ts=float(state.first_token_ts),
            token_logprobs=copy.deepcopy(state.token_logprobs),
            top_logprobs=copy.deepcopy(state.top_logprobs),
            input_token_logprobs=copy.deepcopy(state.input_token_logprobs),
            input_top_logprobs=copy.deepcopy(state.input_top_logprobs),
            pending_token_outputs_len=len(self._pending_token_outputs),
            total_generated_tokens=int(self.metrics.total_generated_tokens),
            decode_tokens_since_last=int(self._decode_tokens_since_last),
            prefill_tokens_since_last=int(self._prefill_tokens_since_last),
        )
        return {
            "checkpoint_id": clean_id,
            "request_id": rid,
            "owner_id": int(state.req_pool_idx),
            "seq_len": int(state.seq_len),
            "num_tokens": ntokens,
        }

    def _restore_request_state(self, checkpoint_id: str) -> dict[str, Any]:
        """Restore a scheduler/worker request-state checkpoint."""
        clean_id = str(checkpoint_id)
        if not clean_id:
            raise RuntimeError("checkpoint_id is required for restore_request_state")
        checkpoint = self._request_state_checkpoints.get(clean_id)
        if checkpoint is None:
            raise RuntimeError(f"unknown request-state checkpoint_id: {clean_id}")
        state = self.requests_by_id.get(checkpoint.request_id)
        if state is None:
            raise RuntimeError(
                f"checkpoint request is no longer active: {checkpoint.request_id}"
            )
        if int(state.req_pool_idx) != int(checkpoint.owner_id):
            raise RuntimeError(
                "request-state owner changed before restore: "
                f"{state.req_pool_idx} != {checkpoint.owner_id}"
            )

        restore_fn = getattr(self.worker_coordinator, "restore_request_state", None)
        if not callable(restore_fn):
            raise RuntimeError(
                "worker coordinator does not support request-state restore"
            )
        restore_fn(clean_id)

        state.seq_len = int(checkpoint.seq_len)
        state.extend_done = bool(checkpoint.extend_done)
        state.extend_offset = int(checkpoint.extend_offset)
        state.generated_ids = list(checkpoint.generated_ids)
        state.decode_offset = int(checkpoint.decode_offset)
        state.decoded_text = str(checkpoint.decoded_text)
        state.finish_reason = str(checkpoint.finish_reason)
        state.first_token_ts = float(checkpoint.first_token_ts)
        state.token_logprobs = copy.deepcopy(checkpoint.token_logprobs)
        state.top_logprobs = copy.deepcopy(checkpoint.top_logprobs)
        state.input_token_logprobs = copy.deepcopy(checkpoint.input_token_logprobs)
        state.input_top_logprobs = copy.deepcopy(checkpoint.input_top_logprobs)
        self._pending_token_outputs = self._pending_token_outputs[
            : checkpoint.pending_token_outputs_len
        ]
        self.metrics.total_generated_tokens = int(checkpoint.total_generated_tokens)
        self._decode_tokens_since_last = int(checkpoint.decode_tokens_since_last)
        self._prefill_tokens_since_last = int(checkpoint.prefill_tokens_since_last)
        del self._request_state_checkpoints[clean_id]
        return {
            "checkpoint_id": clean_id,
            "request_id": checkpoint.request_id,
            "owner_id": int(checkpoint.owner_id),
            "seq_len": int(checkpoint.seq_len),
        }

    def _handle_control_cmd(self, msg: dict[str, Any]) -> None:
        cmd = str(msg.get("cmd", ""))
        control_id = msg.get("control_id")
        try:
            if cmd == "pause":
                self.paused = True
                self._control_response(control_id, True)
                return

            if cmd == "resume":
                self.paused = False
                self._control_response(control_id, True)
                return

            if cmd == "abort":
                request_id = msg.get("request_id")
                rid = (
                    str(request_id)
                    if request_id is not None and str(request_id)
                    else None
                )
                aborted_count = self._abort_requests(rid)
                self._control_response(control_id, True, aborted_count=aborted_count)
                return

            if cmd == "flush_cache":
                aborted_count = self._flush_cache(
                    abort_all_requests=bool(msg.get("abort_all_requests", True))
                )
                self._control_response(control_id, True, aborted_count=aborted_count)
                return

            if cmd == "reload_weights_from_disk":
                model_path = str(msg.get("model_path", "")).strip()
                if not model_path:
                    raise RuntimeError(
                        "model_path is required for reload_weights_from_disk"
                    )
                aborted_count = self._reload_weights_from_disk(
                    model_path=model_path,
                    abort_all_requests=bool(msg.get("abort_all_requests", True)),
                )
                self._control_response(control_id, True, aborted_count=aborted_count)
                return

            if cmd == "get_lane_metadata":
                lane_md = self.worker_coordinator.lane_metadata()
                lane_routes = []
                if self._attention_dp_degree > 1:
                    from nkipy_serving.models.deepseek_v4.rank_layout import (
                        build_attention_dp_lane_routes,
                    )

                    lane_routes = [
                        route.to_dict()
                        for route in build_attention_dp_lane_routes(
                            self.runtime_config.tp_degree,
                            self.runtime_config.ep_degree,
                            self.runtime_config.replica_degree,
                        )
                    ]
                    if len(lane_routes) != self._attention_dp_degree:
                        raise RuntimeError(
                            "attention-DP lane route count mismatch: "
                            f"routes={len(lane_routes)}, "
                            f"attention_dp_degree={self._attention_dp_degree}"
                        )
                self._control_response(
                    control_id,
                    True,
                    lane_metadata=lane_md,
                    lane_routes=lane_routes,
                    attention_dp_degree=self._attention_dp_degree,
                    tp_degree=self.runtime_config.tp_degree,
                    ep_degree=self.runtime_config.ep_degree,
                    replica_degree=self.runtime_config.replica_degree,
                    total_workers=self.runtime_config.total_workers,
                )
                return

            if cmd == "checkpoint_request_state":
                result = self._checkpoint_request_state(
                    request_id=str(msg.get("request_id", "")),
                    checkpoint_id=msg.get("checkpoint_id"),
                    num_tokens=int(msg.get("num_tokens", 0)),
                )
                self._control_response(control_id, True, **result)
                return

            if cmd == "restore_request_state":
                result = self._restore_request_state(
                    checkpoint_id=str(msg.get("checkpoint_id", "")),
                )
                self._control_response(control_id, True, **result)
                return

            if cmd == "get_metrics":
                metrics = {
                    "paused": self.paused,
                    "waiting_queue_size": len(self.waiting_queue),
                    "running_batch_size": len(self.running_batch),
                    "inflight_requests": len(self.requests_by_id),
                    "total_submitted": self.metrics.total_submitted,
                    "total_completed": self.metrics.total_completed,
                    "total_aborted": self.metrics.total_aborted,
                    "total_errors": self.metrics.total_errors,
                    "total_generated_tokens": self.metrics.total_generated_tokens,
                    "total_stream_events": self.metrics.total_stream_events,
                    "total_timed_out": self.metrics.total_timed_out,
                    "scheduler_steps": self.metrics.scheduler_steps,
                    "max_extend_batch_size": self.metrics.max_extend_batch_size,
                    "max_decode_batch_size": self.metrics.max_decode_batch_size,
                    "kv_pool_available": self.token_allocator.available_size(),
                    # Runtime metrics.
                    "last_decode_throughput": self.metrics.last_decode_throughput,
                    "last_prefill_throughput": self.metrics.last_prefill_throughput,
                    "total_time_to_first_token_s": self.metrics.total_time_to_first_token_s,
                    "total_requests_with_ttft": self.metrics.total_requests_with_ttft,
                    "total_prefix_cache_hit_tokens": self.metrics.total_prefix_cache_hit_tokens,
                    "total_prefix_cache_total_tokens": self.metrics.total_prefix_cache_total_tokens,
                    "attention_dp_lane_metrics": self._lane_metrics_payload(),
                    "worker_startup": self.worker_coordinator.startup_summary(),
                }
                self._control_response(control_id, True, metrics=metrics)
                return

            self._control_response(
                control_id,
                False,
                error=f"Unknown scheduler control cmd: {cmd}",
            )
        except Exception as exc:
            self._control_response(control_id, False, error=repr(exc))

    def handle_message(self, msg: dict[str, Any]) -> None:
        cmd = str(msg.get("cmd", ""))
        if cmd == "shutdown":
            self.shutdown_requested = True
            self._send_response({"cmd": "shutdown_ack", "ok": True})
            return

        if cmd == "generate":
            self._handle_generate_cmd(msg)
            return

        if cmd in {
            "abort",
            "pause",
            "resume",
            "get_metrics",
            "get_lane_metadata",
            "flush_cache",
            "reload_weights_from_disk",
            "checkpoint_request_state",
            "restore_request_state",
        }:
            self._handle_control_cmd(msg)
            return

        request_id = msg.get("request_id")
        if request_id is not None:
            self._send_response(
                {
                    "type": "final",
                    "request_id": str(request_id),
                    "ok": False,
                    "error": f"Unknown scheduler cmd: {cmd}",
                }
            )
            return
        self._control_response(
            msg.get("control_id"),
            False,
            error=f"Unknown scheduler cmd: {cmd}",
        )

    # -- KV pool resource management --

    def _allocate_request_resources(self, state: _RequestState) -> bool:
        """Allocate req pool slot + KV cache slots for the prompt.

        If prefix cache is enabled and hits, reuse cached KV slot indices
        for the prefix and only allocate new slots for the remainder.
        Sets ``extend_offset`` so EXTEND skips cached prefix tokens.
        """
        prompt_len = int(state.prompt_ids.size)
        if prompt_len <= 0:
            prompt_len = 1  # at least one token

        # Need slots for prompt + max_new_tokens.
        self._cap_max_new_tokens_to_capacity(state.req, prompt_len=prompt_len)
        total_needed = prompt_len + int(state.req.max_new_tokens)
        if total_needed > self.req_to_token_pool.max_context_len:
            total_needed = self.req_to_token_pool.max_context_len
            state.req.max_new_tokens = max(0, total_needed - prompt_len)
        # Paged attention assumes token position i maps to offset (i % block_size)
        # within each block. Ensure allocations start on a block boundary by
        # reserving whole blocks per request.
        total_needed = (
            (total_needed + self._block_size - 1) // self._block_size
        ) * self._block_size

        # Check prefix cache for reusable KV slots.
        hit_length = 0
        cached_indices: np.ndarray | None = None
        cache_node = None
        needs_prefill_scoring = bool(state.req.score or state.logprob_start_len >= 0)
        if self._prefix_cache_available() and not needs_prefill_scoring:
            match = self.prefix_cache.match_prefix(state.prompt_ids.tolist())
            if match.host_hit_length > 0:
                # The attention backends assume token position i maps to offset
                # (i % block_size) within a paged KV cache. To preserve this
                # mapping, only reuse prefix cache hits in full block-size
                # chunks so that the first non-cached token starts on a block
                # boundary.
                hit_length = (
                    int(match.host_hit_length) // self._block_size
                ) * self._block_size
                if hit_length > 0:
                    cached_indices = match.device_indices
                    cache_node = match.last_host_node
                    # Lock the cache node so it isn't evicted while in use.
                    self.prefix_cache.inc_lock_ref(cache_node)

        # Allocate req pool slot.
        req_slots = self.req_to_token_pool.alloc(1)
        if req_slots is None:
            if cache_node is not None:
                self.prefix_cache.dec_lock_ref(cache_node)
            return False
        state.req_pool_idx = req_slots[0]

        # Allocate new KV slots for the non-cached portion.  Evict
        # unlocked prefix-cache entries first if the allocator is short.
        new_slots_needed = total_needed - hit_length
        if new_slots_needed <= 0:
            new_slots_needed = self._block_size  # at least one block for safety
        if (
            self._prefix_cache_available()
            and self.token_allocator.available_size() < new_slots_needed
        ):
            deficit = new_slots_needed - self.token_allocator.available_size()
            freed_arrays = self.prefix_cache.evict(deficit)
            for arr in freed_arrays:
                self.token_allocator.free(arr)
        new_locs = self.token_allocator.alloc(new_slots_needed)
        if new_locs is None:
            self.req_to_token_pool.free(state.req_pool_idx)
            state.req_pool_idx = -1
            if cache_node is not None:
                self.prefix_cache.dec_lock_ref(cache_node)
            return False

        # Build the full slot array: cached prefix slots + newly allocated slots.
        if hit_length > 0 and cached_indices is not None:
            all_locs = np.concatenate(
                [cached_indices[:hit_length].astype(np.int32), new_locs]
            )
        else:
            all_locs = new_locs
            hit_length = 0

        state.out_cache_loc = all_locs
        state.prefix_hit_length = hit_length
        state.prefix_cache_node = cache_node

        # Track prefix cache hit/total tokens.
        self.metrics.total_prefix_cache_hit_tokens += hit_length
        self.metrics.total_prefix_cache_total_tokens += int(state.prompt_ids.size)
        # Skip EXTEND for cached prefix tokens — but always re-run at least
        # the last token so we get logits to sample the first generated token.
        if hit_length >= prompt_len:
            state.extend_offset = max(0, prompt_len - 1)
        else:
            state.extend_offset = hit_length
        self.req_to_token_pool.req_to_token[state.req_pool_idx, : len(all_locs)] = (
            all_locs
        )
        state.seq_len = prompt_len
        return True

    def _free_request_resources(
        self, state: _RequestState, donate_to_cache: bool = False
    ) -> None:
        """Free KV cache slots + req pool slot.

        If ``donate_to_cache`` and prefix cache is enabled, the prompt prefix
        slots are donated to the cache instead of freed.  The remaining slots
        (generated tokens) are always freed.
        """
        if state.out_cache_loc is not None:
            hit = state.prefix_hit_length
            prompt_len = int(state.prompt_ids.size)

            keep_len = int(hit)
            if donate_to_cache and self._prefix_cache_available() and prompt_len > 0:
                # Donate only a block-aligned prefix to the cache so that cached
                # indices remain compatible with paged attention's (block, offset)
                # mapping.
                cache_len = (prompt_len // self._block_size) * self._block_size
                if cache_len > 0:
                    prompt_slots = state.out_cache_loc[:cache_len]
                    prefix_in_tree = self.prefix_cache.cache_finished_req(
                        PrefixCacheReq(
                            token_ids=state.prompt_ids[:cache_len].tolist(),
                            kv_indices=prompt_slots,
                        )
                    )
                    # The tree only stores the suffix beyond what it already
                    # had.  Slots in [hit : prefix_in_tree] were donated but
                    # the tree kept its existing values — free the duplicates
                    # to avoid leaking allocator capacity.
                    dup_end = min(prefix_in_tree, cache_len)
                    if dup_end > hit:
                        self.token_allocator.free(state.out_cache_loc[hit:dup_end])
                    keep_len = max(keep_len, cache_len)

            # Free everything except the prefix-cache-owned slots (hit) and
            # any donated-to-cache prefix (cache_len).
            free_slots = state.out_cache_loc[keep_len:]
            if free_slots.size > 0:
                self.token_allocator.free(free_slots)

            state.out_cache_loc = None

        # Unlock the prefix cache node if we locked it during admission.
        if state.prefix_cache_node is not None and self.prefix_cache is not None:
            self.prefix_cache.dec_lock_ref(state.prefix_cache_node)
            state.prefix_cache_node = None

        if state.req_pool_idx >= 0:
            if self._clear_request_state_on_free:
                clear_fn = getattr(self.worker_coordinator, "clear_request_state", None)
                if callable(clear_fn):
                    clear_fn([int(state.req_pool_idx)])
            self.req_to_token_pool.free(state.req_pool_idx)
            state.req_pool_idx = -1

    # -- Batch scheduling --

    def _make_batch(
        self,
        extend_states: list[_RequestState],
        decode_states: list[_RequestState],
        mixed: bool = False,
        attention_lane: int = -1,
    ) -> ScheduleBatch:
        return ScheduleBatch(
            extend_states,
            decode_states,
            block_size=self._block_size,
            chunked_prefill_size=self._chunked_prefill_size,
            paddings=self._paddings,
            requested_topk=self._distributed_sampled_local_topk(),
            mixed=mixed,
            attention_lane=attention_lane,
        )

    def _is_distributed_sampled_nkipy_model(self) -> bool:
        if self.runtime_config.execution_backend != "nkipy":
            return False
        model_id = str(self.runtime_config.model_id)
        return (
            model_id == "gpt-oss"
            or model_id.startswith("unsloth/gpt-oss-")
            or model_id == "qwen3-moe"
            or model_id.startswith("Qwen/Qwen3-")
        )

    def _is_dense_nkipy_model(self) -> bool:
        model_id = str(self.runtime_config.model_id)
        return self.runtime_config.execution_backend == "nkipy" and (
            model_id.startswith("Qwen/Qwen3-") and "-A" not in model_id
        )

    def _distributed_sampled_local_topk(self) -> int:
        if not self._is_distributed_sampled_nkipy_model():
            return 1
        return int(self.runtime_config.dense_local_topk)

    def _apply_prefill_budget(
        self,
        extend_states: list[_RequestState],
        decode_states: list[_RequestState],
    ) -> list[_RequestState]:
        """Limit extend states to fit within chunked_prefill_size token budget.

        In mixed mode, decode states consume 1 token each from the budget.
        The first request is always admitted (its chunk is capped by
        _build_batch_impl), but subsequent requests are only admitted if
        their chunk fits the remaining budget.
        """
        budget = self._chunked_prefill_size
        if self._is_mixed_chunk:
            budget -= len(decode_states)
            if budget <= 0:
                return []
        limited: list[_RequestState] = []
        for s in extend_states:
            if budget <= 0:
                break
            remaining = int(s.prompt_ids.size) - s.extend_offset
            chunk = min(remaining, self._chunked_prefill_size)
            if chunk > budget and limited:
                break
            limited.append(s)
            budget -= chunk
        return limited

    def _make_dp_attention_superstep(
        self,
        lane_states: dict[int, tuple[list[_RequestState], list[_RequestState]]],
        *,
        include_extend: bool,
        include_decode: bool,
        mixed: bool = False,
    ) -> DpAttentionSuperstepBatch | None:
        lane_batches: list[ScheduleBatch] = []
        for lane in sorted(lane_states):
            ext, dec = lane_states[lane]
            use_ext = ext if include_extend else []
            use_dec = dec if include_decode else []
            if not use_ext and not use_dec:
                continue
            lane_batches.append(
                self._make_batch(
                    use_ext,
                    use_dec,
                    mixed=mixed,
                    attention_lane=lane,
                )
            )
        if not lane_batches:
            return None
        return DpAttentionSuperstepBatch(
            lane_batches,
            num_lanes=self._attention_dp_degree,
            paddings=self._paddings,
        )

    def _get_next_batches(self) -> list[ScheduleBatch | DpAttentionSuperstepBatch]:
        """Build the list of ScheduleBatch objects for this step.

        Mixed mode: single combined batch (extend + decode in one EXTEND forward).
        Non-mixed: separate extend-only and decode-only batches (two forward calls).

        DP-attention: when `attention_dp_degree > 1`, requests are still
        grouped by lane for lane-local attention metadata, but the scheduler
        returns replica-local supersteps instead of per-lane forwards. MoE/FFN
        collectives must see one synchronized token layout across lanes.
        """
        extend_states = [s for s in self.running_batch if not s.extend_done]
        decode_states = [s for s in self.running_batch if s.extend_done]

        if extend_states and self._chunked_prefill_size > 0:
            extend_states = self._apply_prefill_budget(extend_states, decode_states)

        if self._attention_dp_degree <= 1:
            # Legacy single-lane path (Qwen3 / GPT-OSS): untouched.
            if self._is_mixed_chunk and extend_states and decode_states:
                return [self._make_batch(extend_states, decode_states, mixed=True)]
            batches: list[ScheduleBatch] = []
            if extend_states:
                batches.append(self._make_batch(extend_states, []))
            if decode_states:
                batches.append(self._make_batch([], decode_states))
            return batches

        # DP-attention superstep path. Group by `attention_lane`, then emit
        # one superstep per forward mode across all active lanes.
        lanes: dict[int, tuple[list[_RequestState], list[_RequestState]]] = {}
        for s in extend_states:
            lanes.setdefault(s.attention_lane, ([], []))[0].append(s)
        for s in decode_states:
            lanes.setdefault(s.attention_lane, ([], []))[1].append(s)
        out: list[ScheduleBatch | DpAttentionSuperstepBatch] = []
        if self._is_mixed_chunk and extend_states and decode_states:
            mixed_step = self._make_dp_attention_superstep(
                lanes,
                include_extend=True,
                include_decode=True,
                mixed=True,
            )
            if mixed_step is not None:
                out.append(mixed_step)
            return out

        extend_step = self._make_dp_attention_superstep(
            lanes,
            include_extend=True,
            include_decode=False,
        )
        if extend_step is not None:
            out.append(extend_step)
        decode_step = self._make_dp_attention_superstep(
            lanes,
            include_extend=False,
            include_decode=True,
        )
        if decode_step is not None:
            out.append(decode_step)
        return out

    # -- Forward + sampling --

    def _process_extend_output(
        self,
        states: list[_RequestState],
        query_start_loc: np.ndarray,
        next_token_ids: np.ndarray,
    ) -> None:
        """Process EXTEND results from pre-sampled token ids.

        For chunked prefill: if a request's chunk didn't reach the end of
        its prompt, advance extend_offset and skip sampling.
        """
        for i, state in enumerate(states):
            q_start = int(query_start_loc[i])
            q_end = int(query_start_loc[i + 1])
            chunk_len = q_end - q_start
            new_offset = state.extend_offset + chunk_len
            full_prompt_len = int(state.prompt_ids.size)

            self._prefill_tokens_since_last += chunk_len

            if new_offset < full_prompt_len:
                state.extend_offset = new_offset
                state.seq_len = new_offset
                continue

            state.extend_done = True
            state.seq_len = full_prompt_len
            if self._remaining_tokens(state) <= 0:
                continue

            token_id = int(next_token_ids[i])
            self._emit_token(state, token_id)
            state.seq_len = full_prompt_len + 1

    def _process_decode_output(
        self,
        states: list[_RequestState],
        next_token_ids: np.ndarray,
    ) -> None:
        """Process DECODE results from pre-sampled token ids.

        Output logprobs are handled by ``_process_logprobs_output`` from
        the model_runner's device-computed logprobs.
        """
        self._decode_tokens_since_last += len(states)

        for i, state in enumerate(states):
            token_id = int(next_token_ids[i])
            self._emit_token(state, token_id)
            state.seq_len += 1

    def _process_logprobs_output(
        self,
        states: list[_RequestState],
        forward_output: dict[str, Any],
    ) -> None:
        """Store device-computed logprobs into request states."""
        chosen = forward_output.get("chosen_logprobs")
        topk_vals = forward_output.get("topk_logprob_vals")
        topk_ids = forward_output.get("topk_logprob_ids")
        if chosen is None:
            return

        for i, state in enumerate(states):
            if not state.generated_ids:
                continue
            token_id = state.generated_ids[-1]
            if state.return_logprob:
                state.token_logprobs.append(
                    (
                        float(chosen[i]),
                        int(token_id),
                        self._logprob_token_text(state, token_id),
                    )
                )
                top_entries = None
                if state.top_logprobs_num > 0:
                    k = min(
                        state.top_logprobs_num,
                        int(topk_vals.shape[1]) if topk_vals is not None else 0,
                    )
                    top_entries = []
                    for j in range(k):
                        tid = int(topk_ids[i, j])
                        top_entries.append(
                            (
                                float(topk_vals[i, j]),
                                tid,
                                self._logprob_token_text(state, tid),
                            )
                        )
                state.top_logprobs.append(top_entries)

    # -- Admission + retirement --

    def _can_admit(self, state: _RequestState) -> bool:
        """Pre-check whether admitting this request is feasible.

        Budget: ``estimated_total ≤ free + evictable``.  Prefix-hit tokens
        are NOT subtracted from the estimate because they come from the
        evictable pool — locking them during allocation moves them to
        protected, so they can't also serve as reclaimable headroom.
        """
        prompt_len = int(state.prompt_ids.size)
        if prompt_len <= 0:
            prompt_len = 1
        self._cap_max_new_tokens_to_capacity(state.req, prompt_len=prompt_len)
        estimated_total = prompt_len + int(state.req.max_new_tokens)
        if estimated_total > self.req_to_token_pool.max_context_len:
            estimated_total = self.req_to_token_pool.max_context_len
        estimated_total = (
            (estimated_total + self._block_size - 1) // self._block_size
        ) * self._block_size
        # Budget: free allocator slots + evictable cache entries.
        # We intentionally do NOT subtract the prefix hit here because the
        # cached prefix tokens come from the evictable pool — locking them
        # during allocation moves them from evictable to protected, so they
        # can't also serve as reclaimable headroom.
        available = self.token_allocator.available_size()
        if self._prefix_cache_available():
            available += self.prefix_cache.evictable_size()
        if estimated_total > available:
            return False
        # Don't admit if running batch is at capacity.
        if len(self.running_batch) >= self._max_batch_size:
            return False
        return True

    def _admit_waiting_requests(self) -> None:
        """Move requests from waiting to running, allocating KV resources."""
        available_batch_slots = self._max_batch_size - len(self.running_batch)
        if available_batch_slots <= 0 or not self.waiting_queue:
            return

        # LPM: sort by longest prefix match when prefix cache is active.
        # Prefix matching is done at admission time (_handle_generate_cmd); only re-sort
        # when new requests were added since the last sort.
        if self._prefix_cache_available() and self._waiting_queue_dirty:
            self.waiting_queue.sort(key=lambda s: -s.prefix_hit_length)
            self._waiting_queue_dirty = False

        admitted: list[_RequestState] = []
        remaining: list[_RequestState] = []
        for state in self.waiting_queue:
            if len(admitted) >= available_batch_slots:
                remaining.append(state)
                continue
            # Pre-admission budget check.
            if not self._can_admit(state):
                remaining.append(state)
                break  # Back-pressure: stop trying.
            if self._allocate_request_resources(state):
                admitted.append(state)
            else:
                remaining.append(state)
                break  # Out of KV memory, stop trying.

        self.waiting_queue = (
            remaining + self.waiting_queue[len(admitted) + len(remaining) :]
        )
        self.running_batch.extend(admitted)

    def _retire_finished_requests(self) -> None:
        """Free KV slots and finalize finished requests."""
        next_running: list[_RequestState] = []
        for state in self.running_batch:
            if state.request_id not in self.requests_by_id:
                # Already finalized (e.g., via abort).
                self._free_request_resources(state)
                continue
            if self._state_is_finished(state):
                # Donate prompt KV slots to prefix cache for future reuse.
                self._free_request_resources(state, donate_to_cache=True)
                self._finalize_request_success(state)
                continue
            next_running.append(state)
        self.running_batch = next_running

    # -- Forward with TP dispatch --

    def _forward_step(self, batch: ForwardBatch) -> dict[str, Any]:
        """Dispatch forward to worker(s) and return forward output."""
        self._record_forward_batch_metrics(batch)
        req_id = self.worker_coordinator.dispatch_forward_step(batch)
        self.worker_coordinator.collect_forward_step(req_id)
        out = self.worker_coordinator.last_forward_output
        if out is None:
            raise RuntimeError("Missing forward output from TP worker coordinator")

        # Write IPC breakdown if profiling is active.
        if self._ipc_profile_writer is not None:
            ipc_prof = self.worker_coordinator.last_ipc_profile
            if ipc_prof is not None:
                self._ipc_profile_writer.write(
                    {
                        "step": self.metrics.scheduler_steps,
                        "ts": time.time(),
                        "mode": batch.forward_mode.value,
                        "batch_size": batch.batch_size,
                        "token_bucket": batch.token_bucket,
                        **ipc_prof,
                    }
                )

        return out

    def _record_forward_batch_metrics(self, batch: ForwardBatch) -> None:
        batch_size = int(batch.batch_size)
        if batch.forward_mode == ForwardMode.DECODE:
            self.metrics.max_decode_batch_size = max(
                int(self.metrics.max_decode_batch_size),
                batch_size,
            )
        elif batch.forward_mode == ForwardMode.EXTEND:
            self.metrics.max_extend_batch_size = max(
                int(self.metrics.max_extend_batch_size),
                batch_size,
            )

    # -- Timeout --

    def _check_request_timeouts(self) -> None:
        if self._request_timeout_s <= 0:
            return
        now = time.time()
        timeout = self._request_timeout_s

        def expired(s):
            return now - s.submitted_at >= timeout

        msg = f"Request timed out after {timeout}s"
        self.waiting_queue, c1 = self._filter_queue(self.waiting_queue, expired, msg)
        self.running_batch, c2 = self._filter_queue(
            self.running_batch, expired, msg, free_resources=True
        )
        self.metrics.total_timed_out += c1 + c2

    # -- Main step --

    def _run_single_batching_step(self) -> None:
        pw = self._profile_writer
        timer = StepTimer() if pw is not None else None

        self._admit_waiting_requests()

        if timer is not None:
            timer.mark("admit")

        if not self.running_batch:
            return

        batches = self._get_next_batches()

        if timer is not None:
            timer.mark("classify")

        for batch in batches:
            try:
                fb = batch.build_forward_batch()
                if timer is not None:
                    timer.mark("batch_build")

                forward_output = self._forward_step(fb)
                if timer is not None:
                    timer.mark("device_wait")

                batch.process_results(forward_output, self)
                self._flush_token_outputs()
                if timer is not None:
                    timer.mark("process_results")
            except Exception as exc:
                for state in batch.extend_states + batch.decode_states:
                    self._free_request_resources(state)
                    self._finalize_request_error(state, error=repr(exc), aborted=False)

        self._retire_finished_requests()

        if timer is not None and pw is not None:
            timer.mark("retire")
            # Determine batch mode and sizes from the last batch processed.
            last_batch = batches[-1] if batches else None
            fb_ref = last_batch._forward_batch if last_batch else None
            mode = "idle"
            batch_size = 0
            token_bucket = 0
            real_tokens = 0
            if fb_ref is not None:
                mode = fb_ref.forward_mode.value
                batch_size = fb_ref.batch_size
                token_bucket = fb_ref.token_bucket
                real_tokens = fb_ref.real_total_tokens

            durations = timer.elapsed()
            t_device = durations.get("t_device_wait", 0.0)
            t_total = durations.get("t_total", 0.0)
            overhead_pct = (
                round((t_total - t_device) / t_total * 100, 2) if t_total > 0 else 0.0
            )
            pw.write(
                {
                    "step": self.metrics.scheduler_steps,
                    "ts": time.time(),
                    "mode": mode,
                    "batch_size": batch_size,
                    "token_bucket": token_bucket,
                    "real_tokens": real_tokens,
                    "running_batch_size": len(self.running_batch),
                    "waiting_queue_size": len(self.waiting_queue),
                    "overhead_pct": overhead_pct,
                    **durations,
                }
            )

    # -- Overlap scheduling ---------------------------------------------------

    def _run_overlap_step(self, recv_fn) -> None:
        """Overlap scheduling: receive new requests during device execution.

        Between dispatch (non-blocking) and collect (blocking), the scheduler
        calls ``recv_fn`` to drain ZMQ messages.  This hides the cost of
        request handling (~52ms) behind the device wait (~36ms).

        ``recv_fn`` is a callable that drains ZMQ messages (non-blocking).
        """
        self._admit_waiting_requests()

        if not self.running_batch:
            return

        batches = self._get_next_batches()

        for batch in batches:
            try:
                fb = batch.build_forward_batch()
                self._record_forward_batch_metrics(fb)

                # Dispatch to workers (non-blocking, ~1ms).
                req_id = self.worker_coordinator.dispatch_forward_step(fb)

                # --- Device-wait window: recv while workers execute ---
                recv_fn()

                # Block until workers finish.
                self.worker_coordinator.collect_forward_step(req_id)
                out = self.worker_coordinator.last_forward_output
                if out is None:
                    raise RuntimeError(
                        "Missing forward output from TP worker coordinator"
                    )

                # Process results immediately (same as normal path).
                batch.process_results(out, self)
                self._flush_token_outputs()
            except Exception as exc:
                for state in batch.extend_states + batch.decode_states:
                    self._free_request_resources(state)
                    self._finalize_request_error(state, error=repr(exc), aborted=False)

        self._retire_finished_requests()

    def _update_throughput_metrics(self) -> None:
        if self.metrics.scheduler_steps % self._metrics_update_interval != 0:
            return
        now = time.time()
        decode_elapsed = now - self._last_decode_tic
        if decode_elapsed > 0:
            self.metrics.last_decode_throughput = (
                self._decode_tokens_since_last / decode_elapsed
            )
        self._decode_tokens_since_last = 0
        self._last_decode_tic = now
        prefill_elapsed = now - self._last_prefill_tic
        if prefill_elapsed > 0:
            self.metrics.last_prefill_throughput = (
                self._prefill_tokens_since_last / prefill_elapsed
            )
        self._prefill_tokens_since_last = 0
        self._last_prefill_tic = now

    def run_step(self) -> None:
        if self.paused:
            return
        self.metrics.scheduler_steps += 1

        self._check_request_timeouts()
        self._run_single_batching_step()
        self._update_throughput_metrics()


# ---------------------------------------------------------------------------
# EOS extraction helper
# ---------------------------------------------------------------------------


def _extract_hf_eos_token_ids(model_config: object) -> set[int]:
    """Extract eos_token_id from HF config.json for any model with an hf_model_id.

    Called after build_kv_metadata so the snapshot is already cached locally.
    Falls back to empty set when no HF model ID is configured (e.g. random-weight tests).
    """
    hf_model_id = getattr(model_config, "hf_model_id", None)
    if not hf_model_id:
        return set()

    import json
    from pathlib import Path

    revision = getattr(model_config, "hf_revision", None)
    local_only = getattr(model_config, "hf_local_files_only", True)

    snapshot_path = Path(str(hf_model_id)).expanduser()
    if snapshot_path.exists():
        if not snapshot_path.is_dir():
            raise RuntimeError(
                f"Local model source must be a directory, got {snapshot_path}"
            )
        config_path = str(snapshot_path / "config.json")
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=hf_model_id,
            filename="config.json",
            revision=revision,
            local_files_only=local_only,
        )
    with open(config_path, "r") as f:
        cfg = json.load(f)

    raw = cfg.get("eos_token_id")
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    return {int(t) for t in raw}


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------


def _build_scheduler_components(
    runtime_config: RuntimeConfig,
) -> tuple[
    SchedulerTokenizerService,
    SchedulerKVPoolStub,
    ReqToTokenPool,
    BaseTokenToKVPoolAllocator,
    WorkerCoordinator,
    BasePrefixCache | None,
    set[int],
]:
    """Build all scheduler components: tokenizer, KV pool stub, allocator, and workers."""
    validate_runtime_config(runtime_config)

    # Resolve model config to get KV metadata without loading full weights.
    from nkipy_serving.models.registry import resolve_model_spec

    spec = resolve_model_spec(runtime_config.model_id)
    model_config = spec.build_config(runtime_config)
    num_kv_heads, head_dim, num_layers, kv_dtype = spec.build_kv_metadata(model_config)

    block_size = runtime_config.kv_cache_block_size

    # Create lightweight KV pool stub (for slot allocation tracking, not device execution).
    kv_pool = SchedulerKVPoolStub(
        size=runtime_config.kv_pool_size,
        page_size=block_size,
        dtype=kv_dtype,
        layer_num=num_layers,
    )

    req_to_token_pool = ReqToTokenPool(
        size=runtime_config.max_requests,
        max_context_len=runtime_config.max_context_len,
    )

    token_allocator = PagedTokenToKVPoolAllocator(
        size=kv_pool.size,
        page_size=block_size,
        kvcache=kv_pool,
    )

    # Spawn worker(s): tp=1 gets 1 worker, tp=N gets N workers.
    # The scheduler is always a pure coordinator — never runs NRT.
    worker_coordinator = WorkerCoordinator(runtime_config)

    # Create prefix cache if enabled.
    prefix_cache = None
    if runtime_config.prefix_cache_enabled:
        prefix_cache = create_prefix_cache(
            cache_type=runtime_config.prefix_cache_type,
            page_size=runtime_config.prefix_cache_page_size,
        )

    manager = SchedulerTokenizerService(runtime_config)

    # Extract EOS token IDs from HF config.json (model-agnostic).
    # build_kv_metadata above already triggered snapshot_download, so the
    # snapshot is cached locally and hf_hub_download("config.json") is cheap.
    eos_token_ids = _extract_hf_eos_token_ids(model_config)

    return (
        manager,
        kv_pool,
        req_to_token_pool,
        token_allocator,
        worker_coordinator,
        prefix_cache,
        eos_token_ids,
    )


def run_scheduler_process(
    runtime_config_dict: dict,
    port_args_dict: dict,
    ready_writer,
) -> None:
    import zmq as _zmq

    runtime_config = RuntimeConfig(**runtime_config_dict)
    configure_runtime_environment(runtime_config)

    zmq_context: _zmq.Context | None = None
    worker_coordinator: WorkerCoordinator | None = None
    scheduler: _SchedulerCore | None = None
    poll_profile_writer: ProfileWriter | None = None
    try:
        (
            manager,
            kv_pool,
            req_to_token_pool,
            token_allocator,
            worker_coordinator,
            prefix_cache,
            eos_token_ids,
        ) = _build_scheduler_components(runtime_config)

        # Set up ZMQ sockets.
        zmq_context = _zmq.Context()
        recv_socket = zmq_context.socket(_zmq.PULL)
        recv_socket.bind(port_args_dict["scheduler_input_ipc_name"])
        # Send responses to the detokenizer process (which forwards to the
        # tokenizer manager after decoding token IDs to text).
        send_socket = zmq_context.socket(_zmq.PUSH)
        send_socket.connect(port_args_dict["detokenizer_ipc_name"])

        poller = _zmq.Poller()
        poller.register(recv_socket, _zmq.POLLIN)

        scheduler = _SchedulerCore(
            manager=manager,
            runtime_config=runtime_config,
            response_queue=send_socket,
            kv_pool=kv_pool,
            req_to_token_pool=req_to_token_pool,
            token_allocator=token_allocator,
            worker_coordinator=worker_coordinator,
            prefix_cache=prefix_cache,
            eos_token_ids=eos_token_ids,
        )

        # Signal readiness only after _SchedulerCore construction: its
        # invariant checks (for example DSV4 state sizing) may raise, and that
        # must surface as a parent init error, not a "ready" parent whose
        # scheduler died and whose RPCs hang forever.
        ready_writer.send(
            {
                "status": "ready",
                "variant_count": 0,
                "warmup_summary": {
                    "kv_pool_size": kv_pool.size,
                    "block_size": kv_pool.block_size,
                    "num_blocks": kv_pool.num_blocks,
                    "worker_startup": worker_coordinator.startup_summary(),
                },
            }
        )

        if PROFILING_ENABLED:
            poll_profile_writer = ProfileWriter("scheduler_poll")

        def _recv_and_handle() -> bool:
            """Drain all pending ZMQ messages. Returns True if any received."""
            had = False
            with suppress(_zmq.Again):
                while True:
                    msg = recv_socket.recv_pyobj(_zmq.NOBLOCK)
                    scheduler.handle_message(msg)
                    had = True
            return had

        # Buffer for messages received during the overlap device-wait window.
        # Only generate commands are safe to handle while a forward is in
        # flight; control commands (abort, flush, reload, shutdown) mutate
        # live batch state and must wait until after collect.
        _deferred_msgs: list = []

        def _recv_deferred() -> None:
            """Drain ZMQ into _deferred_msgs without handling."""
            with suppress(_zmq.Again):
                while True:
                    _deferred_msgs.append(recv_socket.recv_pyobj(_zmq.NOBLOCK))

        def _handle_deferred() -> None:
            """Handle all deferred messages after collect completes."""
            for msg in _deferred_msgs:
                scheduler.handle_message(msg)
            _deferred_msgs.clear()

        if scheduler._overlap_enabled:
            # Overlap event loop: recv during device-wait, handle after collect.
            while True:
                has_work = bool(scheduler.running_batch or scheduler.waiting_queue)
                if not has_work:
                    # Idle — block on poller, then recv+handle normally.
                    try:
                        poller.poll(timeout=50)
                    except KeyboardInterrupt:
                        return
                    _recv_and_handle()
                elif scheduler.paused:
                    # Paused with work — must still drain messages so resume/
                    # shutdown can be received. Poll with short timeout.
                    try:
                        poller.poll(timeout=10)
                    except KeyboardInterrupt:
                        return
                    _recv_and_handle()
                else:
                    # Active — run overlap step.
                    scheduler.metrics.scheduler_steps += 1
                    scheduler._check_request_timeouts()
                    scheduler._run_overlap_step(_recv_deferred)
                    _handle_deferred()
                    scheduler._update_throughput_metrics()

                if scheduler.shutdown_requested:
                    return
        else:
            # Normal event loop: recv between steps (original behavior).
            while True:
                has_work = bool(scheduler.running_batch or scheduler.waiting_queue)
                timeout_ms = 0 if has_work else 50

                if poll_profile_writer is not None:
                    _poll_t0 = time.perf_counter()

                try:
                    events = dict(poller.poll(timeout=timeout_ms))
                    if recv_socket in events:
                        _recv_and_handle()
                except KeyboardInterrupt:  # pragma: no cover
                    return

                if poll_profile_writer is not None:
                    _poll_dur = time.perf_counter() - _poll_t0
                    poll_profile_writer.write(
                        {
                            "step": scheduler.metrics.scheduler_steps,
                            "ts": time.time(),
                            "t_poll_and_handle": round(_poll_dur, 6),
                            "has_work": has_work,
                            "had_events": bool(events),
                        }
                    )

                if scheduler.shutdown_requested:
                    return

                if has_work or events:
                    scheduler.run_step()

    except KeyboardInterrupt:  # pragma: no cover - process shutdown path
        return
    except Exception as exc:  # pragma: no cover - process-level failure
        with suppress(EOFError, OSError):
            ready_writer.send({"status": "error", "error": repr(exc)})
    finally:
        # Flush profiling writers before process exit.
        for _pw in [
            getattr(scheduler, "_profile_writer", None)
            if scheduler is not None
            else None,
            getattr(scheduler, "_ipc_profile_writer", None)
            if scheduler is not None
            else None,
            poll_profile_writer,
        ]:
            if _pw is not None:
                with suppress(OSError, ValueError):
                    _pw.close()
        if worker_coordinator is not None:
            worker_coordinator.shutdown()
        if zmq_context is not None:
            with suppress(_zmq.ZMQError):
                zmq_context.destroy(linger=0)
