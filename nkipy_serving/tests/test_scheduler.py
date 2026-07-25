"""Unit tests for _SchedulerCore internals.

Tests scheduler logic (stop tokens, LPM scheduling, admission control,
metrics, abort propagation, logprobs, incremental detokenization) using
mock tokenizer and mock TP worker coordinator -- no HuggingFace model or
real hardware required.
"""

from __future__ import annotations

import queue
import unittest.mock
from types import SimpleNamespace
from typing import Any

import numpy as np

from nkipy_serving.batching.contracts import ForwardBatch
from nkipy_serving.config import RuntimeConfig
from nkipy_serving.managers.detokenizer_manager import DetokenizerManager
from nkipy_serving.managers.scheduler import (
    _extract_hf_eos_token_ids,
    _SchedulerCore,
    _slots_to_block_table,
)
from nkipy_serving.mem_cache.allocator import TokenToKVPoolAllocator
from nkipy_serving.mem_cache.memory_pool import ReqToTokenPool, SchedulerKVPoolStub
from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings

# ---------------------------------------------------------------------------
# Minimal runtime config for tests (no model, no HF, no real workers)
# ---------------------------------------------------------------------------

_TEST_VOCAB_SIZE = 128
_TEST_KV_POOL_SIZE = 256
_TEST_MAX_CONTEXT_LEN = 128
_TEST_BLOCK_SIZE = 4
_TEST_TOKEN_BUCKETS = (32, 128)
_TEST_REQUEST_BUCKETS = (1, 2, 4, 8)


def _test_runtime_config(**overrides) -> RuntimeConfig:
    defaults = dict(
        attention_backend="VanillaPagedAttention",
        paged_attn_impl="vanilla_paged_attention_kv_cache",
        execution_backend="numpy",
        kv_pool_size=_TEST_KV_POOL_SIZE,
        max_context_len=_TEST_MAX_CONTEXT_LEN,
        kv_cache_block_size=_TEST_BLOCK_SIZE,
        token_buckets=_TEST_TOKEN_BUCKETS,
        request_buckets=_TEST_REQUEST_BUCKETS,
        chunked_prefill_size=-1,
        request_timeout_s=0,
    )
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


def _write_dsv4_scheduler_config(tmp_path) -> str:
    (tmp_path / "config.json").write_text(
        """{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 3,
  "compress_ratios": [0, 4, 128],
  "sliding_window": 8,
  "head_dim": 16,
  "index_head_dim": 4
}
""",
        encoding="utf-8",
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Mock tokenizer manager
# ---------------------------------------------------------------------------


class _MockTokenizerManager:
    """Lightweight stand-in for TokenizerManager.

    Provides encode_prompt, decode_ids, decode_one_token using a simple
    mapping: each character maps to its ord() value (mod vocab_size).
    """

    def __init__(self, vocab_size: int = _TEST_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._proxy_mode = False

    def encode_prompt(self, prompt: str) -> np.ndarray:
        return np.asarray([ord(c) % self.vocab_size for c in prompt], dtype=np.int32)

    def decode_ids(self, token_ids: np.ndarray) -> str:
        arr = np.asarray(token_ids, dtype=np.int32)
        return "".join(chr(int(t) % self.vocab_size) for t in arr)

    def decode_one_token(self, token_id: int) -> str:
        return chr(int(token_id) % self.vocab_size)


# ---------------------------------------------------------------------------
# Mock TP worker coordinator
# ---------------------------------------------------------------------------


class _MockWorkerCoordinator:
    """Returns deterministic logits: argmax token = (first input token + 1) mod vocab_size."""

    def __init__(
        self, vocab_size: int = _TEST_VOCAB_SIZE, fixed_token: int | None = None
    ):
        self.vocab_size = vocab_size
        self._last_forward_output: dict[str, Any] | None = None
        self._fixed_token = fixed_token
        self._forward_call_count = 0
        self.flush_cache_calls = 0
        self.clear_request_state_calls: list[list[int]] = []
        self.checkpoint_request_state_calls: list[dict[str, Any]] = []
        self.restore_request_state_calls: list[str] = []
        self.reload_model_paths: list[str] = []

    def dispatch_forward_step(self, batch: ForwardBatch) -> str:
        self._forward_call_count += 1
        total_tokens = batch.total_tokens
        logits = np.zeros((total_tokens, self.vocab_size), dtype=np.float32)
        # Make every position's argmax = fixed_token if set, else = (input_id + 1) % vocab.
        for pos in range(int(batch.real_total_tokens)):
            if self._fixed_token is not None:
                chosen = self._fixed_token
            else:
                chosen = (int(batch.input_ids[pos]) + 1) % self.vocab_size
            logits[pos, chosen] = 10.0
        from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
        from nkipy_serving.sampling.logits_processor_np import NumpyLogitsProcessor

        proc = NumpyLogitsProcessor()
        sampling_batch = DeviceSamplingBatch.from_forward_batch(batch)
        lp_output = proc.forward(
            logits,
            batch.sample_mask,
            batch.query_start_loc,
            batch.batch_size,
            sampling_batch=sampling_batch,
            needs_logprobs=bool(batch.needs_logprobs),
            logprobs_k=int(batch.logprobs_k),
        )
        out = lp_output.to_shm_dict()
        # Synthesize next_token_ids from top1 (mirrors _combine_rank_outputs).
        if "next_token_ids" not in out and "top1_indices" in out:
            out["next_token_ids"] = out["top1_indices"].copy()
        self._last_forward_output = out
        return f"mock-req-{self._forward_call_count}"

    def collect_forward_step(self, req_id: str) -> None:
        pass  # logits already set in dispatch

    def poll_forward_step(self) -> bool:
        return True  # mock completes synchronously

    def collect_forward_step_result(self) -> None:
        pass  # output already set in dispatch

    @property
    def last_forward_output(self) -> dict[str, Any] | None:
        return self._last_forward_output

    def flush_cache(self) -> None:
        self.flush_cache_calls += 1

    def clear_request_state(self, owner_ids: list[int]) -> None:
        self.clear_request_state_calls.append([int(v) for v in owner_ids])

    def checkpoint_request_state(
        self,
        *,
        checkpoint_id: str | None = None,
        owner_id: int,
        seq_len: int,
        num_tokens: int,
    ) -> str:
        clean_id = str(
            checkpoint_id or f"mock-cp-{len(self.checkpoint_request_state_calls)}"
        )
        self.checkpoint_request_state_calls.append(
            {
                "checkpoint_id": clean_id,
                "owner_id": int(owner_id),
                "seq_len": int(seq_len),
                "num_tokens": int(num_tokens),
            }
        )
        return clean_id

    def restore_request_state(self, checkpoint_id: str) -> None:
        self.restore_request_state_calls.append(str(checkpoint_id))

    def reload_weights(self, model_path: str) -> None:
        self.reload_model_paths.append(str(model_path))

    def startup_summary(self) -> dict[str, object]:
        return {
            "total_workers": 1,
            "ready_workers": 1,
            "max_total_elapsed_s": 0.0,
            "mean_total_elapsed_s": 0.0,
            "slowest_ranks": [],
            "stage_max_elapsed_s": {},
        }


# ---------------------------------------------------------------------------
# Mock prefix cache (for LPM tests)
# ---------------------------------------------------------------------------


class _MockPrefixCache:
    """Prefix cache that returns configurable hit lengths per token list."""

    def __init__(
        self,
        hit_map: dict[tuple[int, ...], int] | None = None,
        evictable: int = 0,
        evict_returns: list | None = None,
    ):
        self._hit_map: dict[tuple[int, ...], int] = hit_map or {}
        self._lock_count: dict[int, int] = {}
        self._evictable = evictable
        self._evict_returns: list = evict_returns or []
        self.reset_calls = 0
        self.match_calls = 0
        self.cache_finished_calls = 0
        self.evict_calls = 0

    def match_prefix(self, key: list[int], **kwargs):
        self.match_calls += 1
        hit = self._hit_map.get(tuple(key), 0)
        return SimpleNamespace(
            device_indices=np.arange(hit, dtype=np.int32),
            last_device_node=None,
            last_host_node=id(self) if hit > 0 else None,
            host_hit_length=hit,
            payload=None,
        )

    def cache_finished_req(self, req, **kwargs):
        self.cache_finished_calls += 1
        return 0

    def cache_unfinished_req(self, req, **kwargs):
        return 0

    def evict(self, num_tokens: int):
        self.evict_calls += 1
        return self._evict_returns

    def inc_lock_ref(self, node) -> int:
        self._lock_count[node] = self._lock_count.get(node, 0) + 1
        return self._lock_count[node]

    def dec_lock_ref(self, node, swa_uuid_for_lock=None) -> int:
        self._lock_count[node] = max(0, self._lock_count.get(node, 0) - 1)
        return self._lock_count[node]

    def reset(self):
        self.reset_calls += 1
        self._hit_map.clear()
        self._lock_count.clear()

    def evictable_size(self) -> int:
        return self._evictable


# ---------------------------------------------------------------------------
# Fixture: build a _SchedulerCore with mocks
# ---------------------------------------------------------------------------


def _build_scheduler(
    *,
    runtime_config: RuntimeConfig | None = None,
    vocab_size: int = _TEST_VOCAB_SIZE,
    fixed_token: int | None = None,
    prefix_cache=None,
    eos_token_ids: set[int] | None = None,
) -> tuple[_SchedulerCore, queue.Queue]:
    """Create a _SchedulerCore backed by mocks and a queue.Queue for responses."""
    if runtime_config is None:
        runtime_config = _test_runtime_config()

    kv_pool = SchedulerKVPoolStub(
        size=runtime_config.kv_pool_size,
        page_size=runtime_config.kv_cache_block_size,
        dtype=np.float32,
        layer_num=1,
    )
    req_to_token_pool = ReqToTokenPool(
        size=runtime_config.max_requests,
        max_context_len=runtime_config.max_context_len,
    )
    token_allocator = TokenToKVPoolAllocator(
        size=kv_pool.size,
        kvcache=kv_pool,
    )
    response_q: queue.Queue = queue.Queue()

    manager = _MockTokenizerManager(vocab_size=vocab_size)
    tp_coordinator = _MockWorkerCoordinator(
        vocab_size=vocab_size, fixed_token=fixed_token
    )

    paddings = build_precompile_paddings(runtime_config)

    scheduler = _SchedulerCore(
        manager=manager,
        runtime_config=runtime_config,
        response_queue=response_q,
        kv_pool=kv_pool,
        req_to_token_pool=req_to_token_pool,
        token_allocator=token_allocator,
        worker_coordinator=tp_coordinator,
        prefix_cache=prefix_cache,
        paddings=paddings,
        eos_token_ids=eos_token_ids,
    )
    return scheduler, response_q


def _submit_generate(
    scheduler: _SchedulerCore,
    *,
    request_id: str = "req-1",
    prompt: str = "ABCD",
    input_ids: list[int] | None = None,
    max_new_tokens: int = 4,
    stop_token_ids: list[int] | None = None,
    no_stop_trim: bool = False,
    return_logprob: bool | None = None,
    logprob_start_len: int | None = None,
    top_logprobs_num: int | None = None,
    score: bool = False,
    stream: bool = False,
    stop: str | list[str] | None = None,
    ignore_eos: bool = False,
    temperature: float | None = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    seed: int | None = None,
) -> None:
    """Helper to submit a generate command to the scheduler."""
    req_payload: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "stream": stream,
        "score": score,
    }
    if input_ids is not None:
        req_payload["input_ids"] = input_ids
    else:
        req_payload["prompt"] = prompt
    if stop_token_ids is not None:
        req_payload["stop_token_ids"] = stop_token_ids
    if no_stop_trim:
        req_payload["no_stop_trim"] = True
    if return_logprob is not None:
        req_payload["return_logprob"] = return_logprob
    if logprob_start_len is not None:
        req_payload["logprob_start_len"] = logprob_start_len
    if top_logprobs_num is not None:
        req_payload["top_logprobs_num"] = top_logprobs_num
    if stop is not None:
        req_payload["stop"] = stop
    if ignore_eos:
        req_payload["ignore_eos"] = True
    if temperature is not None:
        req_payload["temperature"] = temperature
    if top_k is not None:
        req_payload["top_k"] = top_k
    if top_p is not None:
        req_payload["top_p"] = top_p
    if min_p is not None:
        req_payload["min_p"] = min_p
    if seed is not None:
        req_payload["seed"] = seed

    scheduler.handle_message(
        {"cmd": "generate", "request_id": request_id, "req": req_payload}
    )


def _make_test_detokenizer(vocab_size: int = _TEST_VOCAB_SIZE) -> DetokenizerManager:
    """Build a DetokenizerManager with a fake tokenizer matching _MockTokenizerManager."""
    from unittest.mock import patch

    with patch.object(DetokenizerManager, "__init__", lambda self, *a, **kw: None):
        dm = DetokenizerManager.__new__(DetokenizerManager)
    dm._states = {}

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=False):
            arr = np.asarray(ids, dtype=np.int32)
            return "".join(chr(int(t) % vocab_size) for t in arr)

    dm._tokenizer = _FakeTok()
    return dm


def _detok_responses(
    raw_messages: list[dict],
    detokenizer: DetokenizerManager,
) -> list[dict]:
    """Pipe raw scheduler messages through the detokenizer."""
    result: list[dict] = []
    for msg in raw_messages:
        result.extend(detokenizer.handle_message(msg))
    return result


def _drain_responses(
    response_q: queue.Queue,
    timeout: float = 0.1,
    detokenizer: DetokenizerManager | None = None,
) -> list[dict]:
    """Drain all responses from the queue, piped through a detokenizer."""
    raw: list[dict] = []
    while True:
        try:
            raw.append(response_q.get(timeout=timeout))
        except queue.Empty:
            break
    if detokenizer is None:
        detokenizer = _make_test_detokenizer()
    return _detok_responses(raw, detokenizer)


def test_build_forward_batch_marks_partial_prefill_rows_unsampled() -> None:
    runtime_config = _test_runtime_config(chunked_prefill_size=3)
    scheduler, _ = _build_scheduler(runtime_config=runtime_config)

    _submit_generate(
        scheduler, request_id="req-partial", prompt="ABCDE", max_new_tokens=1
    )

    scheduler._admit_waiting_requests()
    batches = scheduler._get_next_batches()
    assert len(batches) == 1

    fb = batches[0].build_forward_batch()

    assert np.array_equal(fb.sample_mask, np.asarray([False], dtype=np.bool_))


def test_build_forward_batch_sets_requested_topk_for_distributed_nkipy_models() -> None:
    for idx, model_id in enumerate(
        (
            "Qwen/Qwen3-0.6B",
            "qwen3-moe",
            "gpt-oss",
            "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "unsloth/gpt-oss-120b-BF16",
        ),
        start=1,
    ):
        runtime_config = _test_runtime_config(
            execution_backend="nkipy",
            model_id=model_id,
            dense_local_topk=4,
        )
        scheduler, _ = _build_scheduler(runtime_config=runtime_config)

        _submit_generate(
            scheduler,
            request_id=f"req-topk-{idx}",
            prompt="AB",
            max_new_tokens=1,
        )

        scheduler._admit_waiting_requests()
        batches = scheduler._get_next_batches()
        assert len(batches) == 1

        fb = batches[0].build_forward_batch()

        assert fb.requested_topk == 4


def test_build_forward_batch_enables_device_sampler_for_non_greedy_requests() -> None:
    runtime_config = _test_runtime_config(
        execution_backend="nkipy",
        model_id="Qwen/Qwen3-0.6B",
    )
    scheduler, _ = _build_scheduler(runtime_config=runtime_config)

    _submit_generate(
        scheduler,
        request_id="req-sample",
        prompt="AB",
        max_new_tokens=1,
        temperature=0.8,
        top_k=32,
        top_p=0.9,
        min_p=0.05,
        seed=123,
    )

    scheduler._admit_waiting_requests()
    batches = scheduler._get_next_batches()
    assert len(batches) == 1

    fb = batches[0].build_forward_batch()

    assert fb.use_full_sampler is True
    assert np.array_equal(fb.top_ks, np.asarray([32], dtype=np.int32))
    assert np.allclose(fb.top_ps, np.asarray([0.9], dtype=np.float32))
    assert np.allclose(fb.min_ps, np.asarray([0.05], dtype=np.float32))
    assert np.allclose(fb.temperatures, np.asarray([0.8], dtype=np.float32))
    assert 0.0 < float(fb.uniform_u[0]) < 1.0


def test_pure_decode_batch_uses_request_buckets() -> None:
    runtime_config = _test_runtime_config(
        execution_backend="nkipy",
        model_id="Qwen/Qwen3-0.6B",
        attention_backend="NKIBlockSparseFlashAttention",
        paged_attn_impl="nki_blocksparse_flash_attention",
        request_buckets=(1, 2, 4, 8),
        token_buckets=(128, 512),
    )
    scheduler, response_q = _build_scheduler(runtime_config=runtime_config)

    for idx in range(3):
        _submit_generate(
            scheduler,
            request_id=f"req-decode-{idx}",
            prompt="AB",
            max_new_tokens=4,
        )

    scheduler.run_step()
    _drain_responses(response_q)

    batches = scheduler._get_next_batches()
    assert len(batches) == 1

    fb = batches[0].build_forward_batch()

    assert fb.forward_mode.value == "decode"
    assert fb.batch_size == 3
    assert fb.real_total_tokens == 3
    assert fb.token_bucket == 4
    assert fb.input_ids.shape == (4,)
    assert fb.positions.shape == (4,)


def _run_until_done(
    scheduler: _SchedulerCore,
    response_q: queue.Queue,
    max_steps: int = 200,
) -> list[dict]:
    """Run scheduler steps until no requests remain, collecting responses."""
    dm = _make_test_detokenizer()
    all_responses: list[dict] = []
    for _ in range(max_steps):
        scheduler.run_step()
        all_responses.extend(_drain_responses(response_q, detokenizer=dm))
        if not scheduler.waiting_queue and not scheduler.running_batch:
            break
    all_responses.extend(_drain_responses(response_q, detokenizer=dm))
    return all_responses


def _run_until_done_overlap(
    scheduler: _SchedulerCore,
    response_q: queue.Queue,
    max_steps: int = 200,
) -> list[dict]:
    """Like _run_until_done but drives _run_overlap_step instead of run_step."""
    dm = _make_test_detokenizer()
    all_responses: list[dict] = []

    def _recv_noop() -> None:
        pass

    for _ in range(max_steps):
        scheduler.metrics.scheduler_steps += 1
        scheduler._run_overlap_step(_recv_noop)
        all_responses.extend(_drain_responses(response_q, detokenizer=dm))
        if not scheduler.waiting_queue and not scheduler.running_batch:
            break
    all_responses.extend(_drain_responses(response_q, detokenizer=dm))
    return all_responses


def _get_final_response(responses: list[dict], request_id: str = "req-1") -> dict:
    """Extract the final response for a given request_id."""
    for resp in responses:
        if resp.get("type") == "final" and resp.get("request_id") == request_id:
            return resp
    raise AssertionError(
        f"No final response for {request_id}. Got {len(responses)} messages: "
        + str([r.get("type") for r in responses])
    )


def test_deterministic_generation_token_ids_follow_worker_top1_sequence() -> None:
    """Runtime should append exactly the token ids selected by the worker."""
    scheduler, response_q = _build_scheduler()
    _submit_generate(
        scheduler,
        request_id="req-token-accuracy",
        prompt="ABCD",
        max_new_tokens=4,
    )
    responses = _run_until_done(scheduler, response_q)
    final = _get_final_response(responses, "req-token-accuracy")

    assert final["ok"] is True
    result = final["result"]
    assert result["prompt_ids"] == [65, 66, 67, 68]
    assert result["completion_ids"] == [69, 70, 71, 72]
    assert result["output_ids"] == [65, 66, 67, 68, 69, 70, 71, 72]
    assert result["completion_tokens"] == 4
    assert result["finish_reason"] == "length"


# ===========================================================================
# Tests: stop_token_ids
# ===========================================================================


class TestStopTokenIds:
    """Tests for stop_token_ids handling."""

    def test_stop_token_id_trim_modes(self):
        stop_token = 69
        cases = [
            ("trim-default", {}, 0, False),
            ("preserve", {"no_stop_trim": True}, 1, True),
        ]

        for request_id, kwargs, expected_completion_tokens, expect_stop_token in cases:
            scheduler, response_q = _build_scheduler()
            _submit_generate(
                scheduler,
                request_id=request_id,
                prompt="ABCD",
                max_new_tokens=10,
                stop_token_ids=[stop_token],
                **kwargs,
            )
            responses = _run_until_done(scheduler, response_q)
            final = _get_final_response(responses, request_id)

            assert final["ok"] is True
            result = final["result"]
            assert result["finish_reason"] == "stop"
            assert result["completion_tokens"] == expected_completion_tokens
            generated_part = result["output_ids"][4:]
            assert (stop_token in generated_part) is expect_stop_token


# ===========================================================================
# Tests: LPM scheduling
# ===========================================================================


class TestLPMScheduling:
    """Tests for Longest Prefix Match scheduling."""

    def test_lpm_admits_longest_prefix_first(self):
        """With prefix cache active, requests with longer prefix hits are admitted first."""
        # Prompt A has 10-token hit, prompt B has 2-token hit.
        prompt_a = "AAAAAAAAAA"  # encodes to [65]*10
        prompt_b = "BB"  # encodes to [66]*2

        ids_a = tuple(ord(c) % _TEST_VOCAB_SIZE for c in prompt_a)
        ids_b = tuple(ord(c) % _TEST_VOCAB_SIZE for c in prompt_b)

        prefix_cache = _MockPrefixCache(hit_map={ids_a: 10, ids_b: 2})

        scheduler, response_q = _build_scheduler(prefix_cache=prefix_cache)

        # Submit B first, then A.  Without LPM, B would be admitted first.
        _submit_generate(
            scheduler, request_id="req-b", prompt=prompt_b, max_new_tokens=1
        )
        _submit_generate(
            scheduler, request_id="req-a", prompt=prompt_a, max_new_tokens=1
        )

        # Before admission, waiting_queue should have B then A.
        assert len(scheduler.waiting_queue) == 2
        assert scheduler.waiting_queue[0].request_id == "req-b"
        assert scheduler.waiting_queue[1].request_id == "req-a"

        # With max_new_tokens=1, both requests complete in one step.
        # Verify req-a's final response arrives first, confirming it was
        # admitted (and processed) before req-b due to LPM sorting.
        scheduler.run_step()

        responses = _drain_responses(response_q)
        final_responses = [
            r for r in responses if r.get("type") == "final" and r.get("ok")
        ]
        assert len(final_responses) == 2

        # req-a (longer prefix hit) should be processed first.
        assert final_responses[0]["request_id"] == "req-a"
        assert final_responses[1]["request_id"] == "req-b"


class TestDSV4PrefixCachePolicy:
    def test_dsv4_request_state_disables_prefix_match_sort_and_donation(self):
        prompt_a = "AAAAAAAAAA"
        prompt_b = "BB"
        ids_a = tuple(ord(c) % _TEST_VOCAB_SIZE for c in prompt_a)
        ids_b = tuple(ord(c) % _TEST_VOCAB_SIZE for c in prompt_b)

        prefix_cache = _MockPrefixCache(hit_map={ids_a: 10, ids_b: 2})
        rc = _test_runtime_config(model_id="deepseek-v4", dsv4_state_size=128)
        scheduler, response_q = _build_scheduler(
            runtime_config=rc,
            prefix_cache=prefix_cache,
        )

        _submit_generate(
            scheduler, request_id="req-b", prompt=prompt_b, max_new_tokens=1
        )
        _submit_generate(
            scheduler, request_id="req-a", prompt=prompt_a, max_new_tokens=1
        )

        assert prefix_cache.match_calls == 0
        assert [s.prefix_hit_length for s in scheduler.waiting_queue] == [0, 0]

        scheduler.run_step()
        responses = _drain_responses(response_q)
        final_responses = [
            r for r in responses if r.get("type") == "final" and r.get("ok")
        ]

        assert [r["request_id"] for r in final_responses] == ["req-b", "req-a"]
        assert prefix_cache.cache_finished_calls == 0
        assert scheduler.worker_coordinator.clear_request_state_calls == [[0], [1]]


# ===========================================================================
# Tests: Admission control
# ===========================================================================


class TestAdmissionControl:
    """Tests for admission control."""

    def test_admission_considers_evictable_cache(self):
        """Request should be admitted when free space is low but evictable cache exists."""
        # Pool of 16 tokens, block_size=4.  Burn most slots so only 4 free.
        rc = _test_runtime_config(
            kv_pool_size=16, kv_cache_block_size=4, request_buckets=(1, 2, 4)
        )

        # First verify rejection WITHOUT evictable budget.
        cache_no_evict = _MockPrefixCache(evictable=0)
        sched1, _ = _build_scheduler(runtime_config=rc, prefix_cache=cache_no_evict)
        burn1 = sched1.token_allocator.alloc(12)
        assert burn1 is not None
        _submit_generate(sched1, request_id="req-no", prompt="ABCD", max_new_tokens=4)
        sched1.run_step()
        # 4 free < 8 needed, evictable=0 → rejected.
        assert len(sched1.waiting_queue) == 1

        # Now verify admission WITH evictable budget.
        cache_evict = _MockPrefixCache(evictable=12)
        sched2, _ = _build_scheduler(runtime_config=rc, prefix_cache=cache_evict)
        burn2 = sched2.token_allocator.alloc(12)
        assert burn2 is not None
        # evict() must return the actual burned slots so the allocator
        # can reclaim them during _allocate_request_resources.
        cache_evict._evict_returns = [burn2]
        _submit_generate(sched2, request_id="req-yes", prompt="ABCD", max_new_tokens=4)
        sched2.run_step()
        # 4 free + 12 evictable = 16 ≥ 8 → admitted.
        admitted = {s.request_id for s in sched2.running_batch}
        assert "req-yes" in admitted


# ===========================================================================
# Tests: Metrics
# ===========================================================================


class TestMetrics:
    """Tests for scheduler metrics."""

    def test_dp_attention_lane_metrics_visible_through_get_metrics(self):
        """ADP configs should expose admission/completion distribution by lane."""
        rc = _test_runtime_config(
            execution_backend="nkipy",
            tp_degree=1,
            ep_degree=4,
            replica_degree=1,
            attention_dp_degree=4,
        )
        scheduler, response_q = _build_scheduler(runtime_config=rc)

        for idx in range(5):
            _submit_generate(
                scheduler,
                request_id=f"req-lane-{idx}",
                prompt="AB",
                max_new_tokens=1,
            )

        pending = scheduler._lane_metrics_payload()
        assert pending["submitted"] == [2, 1, 1, 1]
        assert pending["waiting"] == [2, 1, 1, 1]
        assert pending["waiting_pressure"] == [8, 4, 4, 4]
        assert pending["next_lane_cursor"] == 1

        responses = _run_until_done(scheduler, response_q)
        finals = [r for r in responses if r.get("type") == "final"]
        assert len(finals) == 5

        scheduler.handle_message({"cmd": "get_metrics", "control_id": "lane-metrics"})
        metrics_resp = next(
            resp
            for resp in _drain_responses(response_q)
            if resp.get("control_id") == "lane-metrics"
        )
        lane_metrics = metrics_resp["metrics"]["attention_dp_lane_metrics"]
        worker_startup = metrics_resp["metrics"]["worker_startup"]

        assert lane_metrics["attention_dp_degree"] == 4
        assert (
            lane_metrics["routing_policy"]
            == "least_token_pressure_round_robin_tie_break"
        )
        assert lane_metrics["pressure_unit"] == "rounded_kv_token_slots"
        assert lane_metrics["submitted"] == [2, 1, 1, 1]
        assert lane_metrics["completed"] == [2, 1, 1, 1]
        assert lane_metrics["waiting"] == [0, 0, 0, 0]
        assert lane_metrics["running"] == [0, 0, 0, 0]
        assert lane_metrics["inflight"] == [0, 0, 0, 0]
        assert lane_metrics["inflight_pressure"] == [0, 0, 0, 0]
        assert worker_startup["ready_workers"] == 1
        assert worker_startup["stage_max_elapsed_s"] == {}

    def test_dsv4_dp_attention_lane_metrics_use_kv_byte_pressure(self, tmp_path):
        """DSV4 ADP routing should use model-specific KV/state byte estimates."""
        rc = _test_runtime_config(
            execution_backend="nkipy",
            model_id="deepseek-v4",
            hf_model_id=_write_dsv4_scheduler_config(tmp_path),
            tp_degree=1,
            ep_degree=2,
            replica_degree=1,
            attention_dp_degree=2,
            dsv4_state_size=128,
        )
        scheduler, _response_q = _build_scheduler(runtime_config=rc)
        assert scheduler._lane_pressure_model is not None

        for idx in range(3):
            _submit_generate(
                scheduler,
                request_id=f"dsv4-lane-{idx}",
                prompt="AB",
                max_new_tokens=1,
            )

        pending = scheduler._lane_metrics_payload()

        assert pending["routing_policy"] == (
            "least_dsv4_kv_byte_pressure_round_robin_tie_break"
        )
        assert pending["pressure_unit"] == "estimated_dsv4_kv_state_bytes"
        assert pending["submitted"] == [2, 1]
        assert pending["waiting"] == [2, 1]
        assert pending["waiting_pressure"] == [4496, 2248]
        assert pending["pressure_model"] == {
            "num_layers": 3,
            "layer_kind_counts": {"full": 1, "c4a": 1, "c128a": 1},
            "sliding_window": 8,
            "head_dim": 16,
            "index_head_dim": 4,
            "max_context_len": 128,
            "num_slots_per_layer": 257,
            "max_requests": 8,
            "estimated_bytes_per_full_context_request": 21024,
            "estimated_static_state_bytes_per_worker": 186720,
            "state_size": 128,
        }


# ===========================================================================
# Tests: Abort propagation
# ===========================================================================


class TestAbortPropagation:
    """Tests for abort propagation."""

    def test_abort_sends_finish_reason_abort(self):
        """Aborting a request should produce a response with finish_reason='abort'."""
        scheduler, response_q = _build_scheduler()

        # Submit a request with many tokens so it won't finish immediately.
        _submit_generate(
            scheduler, request_id="req-abort", prompt="AB", max_new_tokens=100
        )

        # Admit it.
        scheduler.run_step()

        # Now abort it.
        scheduler.handle_message(
            {"cmd": "abort", "request_id": "req-abort", "control_id": "ctrl-1"}
        )

        responses = _drain_responses(response_q)

        # Find the final response for the aborted request.
        abort_resp = None
        for resp in responses:
            if resp.get("type") == "final" and resp.get("request_id") == "req-abort":
                abort_resp = resp
                break

        assert abort_resp is not None, (
            f"No abort response found. Responses: {responses}"
        )
        assert abort_resp["ok"] is False
        assert abort_resp.get("aborted") is True
        assert abort_resp.get("finish_reason") == "abort"

        # The control response should also be present.
        ctrl_resp = None
        for resp in responses:
            if resp.get("control_id") == "ctrl-1":
                ctrl_resp = resp
                break
        assert ctrl_resp is not None
        assert ctrl_resp["ok"] is True


# ===========================================================================
# Tests: Reload / flush control
# ===========================================================================


class TestReloadAndFlushControl:
    def test_flush_cache_aborts_requests_and_resets_runtime_state(self):
        prefix_cache = _MockPrefixCache({(65, 66): 4})
        scheduler, response_q = _build_scheduler(prefix_cache=prefix_cache)

        _submit_generate(
            scheduler,
            request_id="req-flush",
            prompt="AB",
            max_new_tokens=16,
        )
        scheduler.run_step()

        scheduler.handle_message(
            {
                "cmd": "flush_cache",
                "control_id": "ctrl-flush",
                "abort_all_requests": True,
            }
        )

        responses = _drain_responses(response_q)
        abort_resp = next(
            resp
            for resp in responses
            if resp.get("type") == "final" and resp.get("request_id") == "req-flush"
        )
        ctrl_resp = next(
            resp for resp in responses if resp.get("control_id") == "ctrl-flush"
        )

        assert abort_resp["ok"] is False
        assert abort_resp["aborted"] is True
        assert ctrl_resp["ok"] is True
        assert ctrl_resp["aborted_count"] == 1
        assert scheduler.worker_coordinator.flush_cache_calls == 1
        assert prefix_cache.reset_calls == 1
        assert scheduler.waiting_queue == []
        assert scheduler.running_batch == []
        assert scheduler.requests_by_id == {}
        assert (
            scheduler.req_to_token_pool.available_size()
            == scheduler.req_to_token_pool.size
        )
        assert (
            scheduler.token_allocator.available_size() == scheduler.token_allocator.size
        )

    def test_reload_weights_calls_worker_reload_then_flush(self):
        prefix_cache = _MockPrefixCache()
        scheduler, response_q = _build_scheduler(prefix_cache=prefix_cache)

        scheduler.handle_message(
            {
                "cmd": "reload_weights_from_disk",
                "control_id": "ctrl-reload",
                "model_path": "/tmp/checkpoint-step-1",
            }
        )

        responses = _drain_responses(response_q)
        ctrl_resp = next(
            resp for resp in responses if resp.get("control_id") == "ctrl-reload"
        )

        assert ctrl_resp["ok"] is True
        assert ctrl_resp["aborted_count"] == 0
        assert scheduler.worker_coordinator.reload_model_paths == [
            "/tmp/checkpoint-step-1"
        ]
        assert scheduler.worker_coordinator.flush_cache_calls == 1
        assert prefix_cache.reset_calls == 1

    def test_dsv4_checkpoint_restore_control_restores_scheduler_state(self):
        rc = _test_runtime_config(model_id="deepseek-v4", dsv4_state_size=128)
        scheduler, response_q = _build_scheduler(runtime_config=rc)

        _submit_generate(
            scheduler,
            request_id="req-rollback",
            prompt="AB",
            max_new_tokens=4,
        )
        scheduler.run_step()
        _drain_responses(response_q)

        state = scheduler.requests_by_id["req-rollback"]
        before_generated = list(state.generated_ids)
        before_seq_len = int(state.seq_len)

        scheduler.handle_message(
            {
                "cmd": "checkpoint_request_state",
                "control_id": "ctrl-cp",
                "request_id": "req-rollback",
                "checkpoint_id": "cp-rollback",
                "num_tokens": 2,
            }
        )
        cp_resp = next(
            resp
            for resp in _drain_responses(response_q)
            if resp.get("control_id") == "ctrl-cp"
        )

        state.generated_ids.extend([42, 43])
        state.seq_len += 2
        scheduler._pending_token_outputs.append(
            {"request_id": "req-rollback", "token_id": 42, "stream": False}
        )
        scheduler.metrics.total_generated_tokens += 2
        scheduler._decode_tokens_since_last += 2

        scheduler.handle_message(
            {
                "cmd": "restore_request_state",
                "control_id": "ctrl-restore",
                "checkpoint_id": "cp-rollback",
            }
        )
        restore_resp = next(
            resp
            for resp in _drain_responses(response_q)
            if resp.get("control_id") == "ctrl-restore"
        )

        assert cp_resp["ok"] is True
        assert cp_resp["owner_id"] == 0
        assert cp_resp["seq_len"] == before_seq_len
        assert cp_resp["num_tokens"] == 2
        assert restore_resp["ok"] is True
        assert restore_resp["request_id"] == "req-rollback"
        assert state.generated_ids == before_generated
        assert state.seq_len == before_seq_len
        assert scheduler._pending_token_outputs == []
        assert scheduler.worker_coordinator.checkpoint_request_state_calls == [
            {
                "checkpoint_id": "cp-rollback",
                "owner_id": 0,
                "seq_len": before_seq_len,
                "num_tokens": 2,
            }
        ]
        assert scheduler.worker_coordinator.restore_request_state_calls == [
            "cp-rollback"
        ]


# ===========================================================================
# Tests: Mixed chunk
# ===========================================================================


class TestMixedChunk:
    """Tests for enable_mixed_chunk (extend+decode overlap in single batch)."""

    def test_mixed_chunk_budget_reserves_for_decode(self):
        """Decode states consume budget; extend should be limited accordingly."""
        rc = _test_runtime_config(
            chunked_prefill_size=4,
            enable_mixed_chunk=True,
        )
        scheduler, response_q = _build_scheduler(runtime_config=rc)

        # Submit req-1 "AB" and complete its extend.
        _submit_generate(scheduler, request_id="req-1", prompt="AB", max_new_tokens=10)
        scheduler.run_step()
        _drain_responses(response_q)

        # Now req-1 is decoding.  Submit req-2 with a long prompt (10 tokens).
        _submit_generate(
            scheduler, request_id="req-2", prompt="ABCDEFGHIJ", max_new_tokens=10
        )
        scheduler.run_step()
        responses_2 = _drain_responses(response_q)

        # req-2 should be in the running batch now, still extending.
        req2_state = scheduler.requests_by_id.get("req-2")
        assert req2_state is not None, (
            f"req-2 missing from requests_by_id. "
            f"running_batch={[s.request_id for s in scheduler.running_batch]}, "
            f"waiting_queue={[s.request_id for s in scheduler.waiting_queue]}, "
            f"responses={responses_2}"
        )
        # Budget = 4 - 1 (decode) = 3.  But _build_batch_impl caps chunk at
        # chunked_prefill_size (4), so the actual chunk is min(remaining, 4) = 4.
        # The budget reservation just determines which requests are *included*;
        # per-request chunk size is still bounded by chunked_prefill_size.
        assert req2_state.extend_offset == 4, (
            f"Expected extend_offset == 4, got {req2_state.extend_offset}"
        )
        assert not req2_state.extend_done, "req-2 should still be extending"


# ---------------------------------------------------------------------------
# EOS / ignore_eos tests
# ---------------------------------------------------------------------------


class TestEosHandling:
    """Verify that EOS token IDs are merged into stop_token_ids by default
    and skipped when ignore_eos=True."""

    def test_eos_handling_modes(self):
        cases = [
            ("req-eos", {42}, {}, 10, "stop", 0),
            ("req-ignore", {42}, {"ignore_eos": True}, 5, "length", 5),
            ("req-no-eos", None, {}, 5, "length", 5),
        ]

        for (
            request_id,
            eos_token_ids,
            kwargs,
            max_new_tokens,
            finish_reason,
            completion_tokens,
        ) in cases:
            scheduler, response_q = _build_scheduler(
                fixed_token=42,
                eos_token_ids=eos_token_ids,
            )
            _submit_generate(
                scheduler,
                request_id=request_id,
                prompt="AB",
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            responses = _run_until_done(scheduler, response_q)
            final = _get_final_response(responses, request_id)

            assert final["ok"] is True
            result = final["result"]
            assert result["finish_reason"] == finish_reason
            assert result["completion_tokens"] == completion_tokens


# ===========================================================================
# Tests: _extract_hf_eos_token_ids helper
# ===========================================================================


class TestExtractHfEosTokenIds:
    """Unit tests for _extract_hf_eos_token_ids helper."""

    def test_missing_hf_model_id_returns_empty(self):
        for config in (SimpleNamespace(), SimpleNamespace(hf_model_id=None)):
            assert _extract_hf_eos_token_ids(config) == set()

    def test_remote_config_eos_token_id_variants(self, tmp_path):
        config = SimpleNamespace(
            hf_model_id="fake/model",
            hf_revision=None,
            hf_local_files_only=True,
        )
        cases = [
            ('{"eos_token_id": 151643}', {151643}),
            ('{"eos_token_id": [151643, 151645]}', {151643, 151645}),
            ('{"vocab_size": 32000}', set()),
        ]

        for raw_config, expected in cases:
            config_path = tmp_path / "config.json"
            config_path.write_text(raw_config)
            with unittest.mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value=str(config_path),
            ):
                result = _extract_hf_eos_token_ids(config)

            assert result == expected

    def test_local_snapshot_path_reads_config_directly(self, tmp_path):
        """Local converted snapshots must not be passed to hf_hub_download."""
        (tmp_path / "config.json").write_text('{"eos_token_id": [1, 2]}')
        config = SimpleNamespace(
            hf_model_id=str(tmp_path),
            hf_revision=None,
            hf_local_files_only=True,
        )
        with unittest.mock.patch("huggingface_hub.hf_hub_download") as mock_download:
            result = _extract_hf_eos_token_ids(config)
        mock_download.assert_not_called()
        assert result == {1, 2}


# ===========================================================================
# Tests: max_context_len capping
# ===========================================================================


class TestMaxContextLenCap:
    """Generation must not exceed max_context_len KV slots."""

    def test_generation_length_bounds_respect_context_and_kv_capacity(self):
        cases = [
            dict(
                runtime_config=_test_runtime_config(
                    max_context_len=128, kv_pool_size=256
                ),
                prompt="ABCD",
                max_new_tokens=200,
                expected_finish="length",
                max_completion_tokens=128 - 4,
            ),
            dict(
                runtime_config=_test_runtime_config(
                    max_context_len=128, kv_pool_size=64
                ),
                prompt="ABCD",
                max_new_tokens=200,
                expected_finish="length",
                max_completion_tokens=64 - 4,
            ),
            dict(
                runtime_config=_test_runtime_config(
                    max_context_len=128, kv_pool_size=256
                ),
                prompt="AB",
                max_new_tokens=5,
                expected_finish=None,
                max_completion_tokens=5,
            ),
        ]

        for case in cases:
            scheduler, response_q = _build_scheduler(
                runtime_config=case["runtime_config"]
            )
            _submit_generate(
                scheduler,
                prompt=case["prompt"],
                max_new_tokens=case["max_new_tokens"],
            )
            responses = _run_until_done(scheduler, response_q)
            final = _get_final_response(responses)

            assert final["ok"] is True
            result = final["result"]
            if case["expected_finish"] is not None:
                assert result["finish_reason"] == case["expected_finish"]
                assert result["completion_tokens"] <= case["max_completion_tokens"]
            else:
                assert result["completion_tokens"] == case["max_completion_tokens"]


# ===========================================================================
# Tests: Input validation
# ===========================================================================


class TestInputValidation:
    """Requests with excessive input length are rejected at entry."""

    def test_oversized_inputs_are_rejected(self):
        cases = [
            (
                _test_runtime_config(max_context_len=16, kv_pool_size=64),
                "A" * 20,
                "exceeds or equals",
            ),
            (
                _test_runtime_config(
                    token_buckets=(32, 128),
                    max_context_len=256,
                    kv_pool_size=512,
                    chunked_prefill_size=-1,
                ),
                "A" * 200,
                "maximum token bucket",
            ),
        ]

        for runtime_config, prompt, expected_error in cases:
            scheduler, response_q = _build_scheduler(runtime_config=runtime_config)
            _submit_generate(scheduler, prompt=prompt, max_new_tokens=5)
            responses = _run_until_done(scheduler, response_q)
            final = _get_final_response(responses)

            assert final["ok"] is False
            assert expected_error in final["error"]
            assert len(scheduler.waiting_queue) == 0


# ===========================================================================
# Tests: Prefill budget
# ===========================================================================


class TestPrefillBudget:
    """Prefill budget must not over-admit extend requests."""

    def test_chunked_extend_block_tables_cover_current_chunk_context(self):
        """Second chunk must build block tables for the post-chunk sequence length."""
        rc = _test_runtime_config(
            token_buckets=(4, 8, 16, 32),
            max_context_len=32,
            kv_pool_size=64,
            kv_cache_block_size=4,
            chunked_prefill_size=4,
        )
        scheduler, response_q = _build_scheduler(runtime_config=rc)
        _submit_generate(
            scheduler,
            request_id="chunked",
            prompt="A" * 9,
            max_new_tokens=1,
        )

        scheduler.run_step()
        assert response_q.empty()
        assert len(scheduler.running_batch) == 1
        state = scheduler.running_batch[0]
        assert state.extend_offset == 4
        assert state.seq_len == 4

        batches = scheduler._get_next_batches()
        assert len(batches) == 1
        fb = batches[0].build_forward_batch()
        assert fb.query_start_loc.tolist() == [0, 4]
        assert fb.block_tables.shape == (1, 2)

    def test_chunked_extend_final_partial_chunk_uses_full_effective_context(self):
        """Final short chunk must still build block tables from full active context."""
        rc = _test_runtime_config(
            token_buckets=(4, 8, 16, 32),
            max_context_len=32,
            kv_pool_size=64,
            kv_cache_block_size=4,
            chunked_prefill_size=4,
        )
        scheduler, response_q = _build_scheduler(runtime_config=rc)
        _submit_generate(
            scheduler,
            request_id="chunked-final",
            prompt="A" * 10,
            max_new_tokens=1,
        )

        scheduler.run_step()
        scheduler.run_step()
        assert response_q.empty()
        assert len(scheduler.running_batch) == 1
        state = scheduler.running_batch[0]
        assert state.extend_offset == 8
        assert state.seq_len == 8

        batches = scheduler._get_next_batches()
        assert len(batches) == 1
        fb = batches[0].build_forward_batch()

        expected = _slots_to_block_table(
            state.out_cache_loc[:10],
            scheduler.runtime_config.kv_cache_block_size,
            3,
        )
        assert fb.query_start_loc.tolist() == [0, 2]
        assert fb.block_tables.shape == (1, 3)
        np.testing.assert_array_equal(fb.block_tables[0], expected)

    def test_mixed_chunk_second_extend_chunk_uses_effective_context_in_block_table(
        self,
    ):
        """Mixed decode-as-extend must still build extend row block tables from post-chunk context."""
        rc = _test_runtime_config(
            token_buckets=(4, 8, 16, 32),
            max_context_len=32,
            kv_pool_size=64,
            kv_cache_block_size=4,
            chunked_prefill_size=4,
            enable_mixed_chunk=True,
        )
        scheduler, response_q = _build_scheduler(runtime_config=rc)

        _submit_generate(
            scheduler,
            request_id="decode",
            prompt="AB",
            max_new_tokens=4,
        )
        scheduler.run_step()
        _drain_responses(response_q)

        _submit_generate(
            scheduler,
            request_id="extend",
            prompt="A" * 9,
            max_new_tokens=1,
        )
        scheduler.run_step()
        _drain_responses(response_q)

        batches = scheduler._get_next_batches()
        assert len(batches) == 1
        batch = batches[0]
        assert len(batch.extend_states) == 1
        assert len(batch.decode_states) == 1

        extend_state = batch.extend_states[0]
        decode_state = batch.decode_states[0]
        assert extend_state.request_id == "extend"
        assert extend_state.extend_offset == 4
        assert extend_state.seq_len == 4
        assert decode_state.request_id == "decode"
        assert decode_state.extend_done is True

        fb = batch.build_forward_batch()
        expected_extend = _slots_to_block_table(
            extend_state.out_cache_loc[:8],
            scheduler.runtime_config.kv_cache_block_size,
            2,
        )
        expected_decode = _slots_to_block_table(
            decode_state.out_cache_loc[: decode_state.seq_len],
            scheduler.runtime_config.kv_cache_block_size,
            2,
        )

        assert fb.query_start_loc.tolist() == [0, 4, 5]
        assert fb.block_tables.shape == (2, 2)
        np.testing.assert_array_equal(fb.block_tables[0], expected_extend)
        np.testing.assert_array_equal(fb.block_tables[1], expected_decode)


# ===========================================================================
# Tests: overlap scheduling
# ===========================================================================


class TestOverlapScheduling:
    """Overlap scheduling produces identical results to the normal path."""

    def test_basic_generation_matches_normal(self):
        """Same prompt+tokens should produce the same final text."""
        # Normal path.
        s1, q1 = _build_scheduler()
        _submit_generate(s1, prompt="ABCD", max_new_tokens=4)
        normal_resp = _run_until_done(s1, q1)
        normal_final = _get_final_response(normal_resp)

        # Overlap path.
        s2, q2 = _build_scheduler()
        _submit_generate(s2, prompt="ABCD", max_new_tokens=4)
        overlap_resp = _run_until_done_overlap(s2, q2)
        overlap_final = _get_final_response(overlap_resp)

        assert normal_final["result"]["text"] == overlap_final["result"]["text"]
        assert (
            normal_final["result"]["completion_ids"]
            == overlap_final["result"]["completion_ids"]
        )
