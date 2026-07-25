"""High-signal logprobs feature tests."""

import numpy as np
import pytest

from nkipy_serving.batching.contracts import ForwardBatch, ForwardMode
from nkipy_serving.runtime.worker_coordinator import (
    _compute_output_slot_size,
    _output_slot_read,
    _output_slot_write_ids_with_logprobs,
)
from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
from nkipy_serving.sampling.logits_processor import LogitsProcessorOutput


def _make_forward_batch(**overrides) -> ForwardBatch:
    kwargs = dict(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=np.zeros(2, dtype=np.int32),
        positions=np.zeros(2, dtype=np.int32),
        seq_lens=np.array([10, 10], dtype=np.int64),
        slot_mapping=np.zeros(2, dtype=np.int64),
        block_tables=np.zeros((2, 1), dtype=np.int64),
        query_start_loc=np.array([0, 1, 2], dtype=np.int64),
        sample_mask=np.ones(2, dtype=np.bool_),
    )
    kwargs.update(overrides)
    return ForwardBatch(**kwargs)


class TestDeviceSamplingBatchLogprobs:
    def test_from_forward_batch_carries_logprob_metadata(self):
        cases = [
            (
                {"needs_logprobs": True, "logprobs_k": 7, "use_full_sampler": True},
                True,
                7,
                True,
            ),
            ({}, False, 0, False),
        ]

        for overrides, needs_logprobs, logprobs_k, enabled in cases:
            fb = _make_forward_batch(**overrides)
            dsb = DeviceSamplingBatch.from_forward_batch(fb)
            assert dsb.needs_logprobs is needs_logprobs
            assert dsb.logprobs_k == logprobs_k
            assert dsb.enabled is enabled


class TestShmLogprobsSlotRoundtrip:
    def test_write_read_ids_with_logprobs(self):
        bs = 4
        k = 3
        total_workers = 1
        output_idx = 0
        generation = 42
        output_slot_size = _compute_output_slot_size(bs, max(k, 1))
        # Ensure enough space.
        total_buf_size = 4096 + 256 * total_workers + output_slot_size
        buf = bytearray(total_buf_size)
        mv = memoryview(buf)

        token_ids = np.array([10, 20, 30, 40], dtype=np.int32)
        chosen = np.array([-0.5, -1.0, -0.3, -2.0], dtype=np.float32)
        topk_vals = np.array(
            [
                [-0.1, -0.2, -0.3],
                [-0.4, -0.5, -0.6],
                [-0.01, -0.02, -0.03],
                [-1.0, -1.5, -2.0],
            ],
            dtype=np.float32,
        )
        topk_ids = np.array(
            [
                [5, 10, 15],
                [20, 25, 30],
                [35, 40, 45],
                [50, 55, 60],
            ],
            dtype=np.int32,
        )

        _output_slot_write_ids_with_logprobs(
            mv,
            total_workers,
            output_idx,
            output_slot_size,
            generation,
            token_ids,
            chosen,
            topk_vals,
            topk_ids,
        )

        result = _output_slot_read(
            mv,
            total_workers,
            output_idx,
            output_slot_size,
            generation,
        )

        assert "next_token_ids" in result
        assert "chosen_logprobs" in result
        assert "topk_logprob_vals" in result
        assert "topk_logprob_ids" in result
        np.testing.assert_array_equal(result["next_token_ids"], token_ids)
        np.testing.assert_allclose(result["chosen_logprobs"], chosen)
        np.testing.assert_allclose(result["topk_logprob_vals"], topk_vals)
        np.testing.assert_array_equal(result["topk_logprob_ids"], topk_ids)

    def test_stale_generation_returns_empty(self):
        total_workers = 1
        output_idx = 0
        output_slot_size = _compute_output_slot_size(4, 3)
        total_buf_size = 4096 + 256 * total_workers + output_slot_size
        buf = bytearray(total_buf_size)
        mv = memoryview(buf)

        _output_slot_write_ids_with_logprobs(
            mv,
            total_workers,
            output_idx,
            output_slot_size,
            10,
            np.zeros(4, dtype=np.int32),
            np.zeros(4, dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.int32),
        )

        # Read with wrong generation.
        result = _output_slot_read(mv, total_workers, output_idx, output_slot_size, 99)
        assert result == {}


class TestLogitsProcessorOutputSerialization:
    def test_to_shm_dict_contracts(self):
        with_logprobs = LogitsProcessorOutput(
            next_token_ids=np.array([1, 2, 3], dtype=np.int32),
            chosen_logprobs=np.array([-0.5, -1.0, -0.3], dtype=np.float32),
            topk_logprob_vals=np.array(
                [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=np.float32
            ),
            topk_logprob_ids=np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32),
        ).to_shm_dict()
        assert "next_token_ids" in with_logprobs
        assert "chosen_logprobs" in with_logprobs
        assert "topk_logprob_vals" in with_logprobs
        assert "topk_logprob_ids" in with_logprobs

        token_only = LogitsProcessorOutput(
            next_token_ids=np.array([1, 2, 3], dtype=np.int32),
        ).to_shm_dict()
        assert "next_token_ids" in token_only
        assert "chosen_logprobs" not in token_only

        greedy_top1 = LogitsProcessorOutput(
            top1_values=np.array([1.0, 2.0], dtype=np.float32),
            top1_indices=np.array([5, 6], dtype=np.int32),
        ).to_shm_dict(vocab_offset=100)
        assert "top1_values" in greedy_top1
        assert "top1_indices" in greedy_top1
        assert np.array_equal(
            greedy_top1["vocab_offset"], np.array([100], dtype=np.int32)
        )


class TestLogprobsKCapping:
    def _make_scheduler(self):
        """Build a minimal scheduler for testing."""
        from nkipy_serving.config import RuntimeConfig
        from nkipy_serving.managers.scheduler import (
            SchedulerTokenizerService,
            _SchedulerCore,
        )
        from nkipy_serving.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from nkipy_serving.mem_cache.memory_pool import (
            ReqToTokenPool,
            SchedulerKVPoolStub,
        )
        from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings

        config = RuntimeConfig(
            execution_backend="numpy",
            attention_backend="VanillaPagedAttention",
            paged_attn_impl="vanilla_paged_attention_kv_cache",
        )

        class FakeWorkerCoordinator:
            def dispatch_forward_step(self, batch):
                return "fake_req_id"

            def collect_forward_step(self, req_id):
                pass

            _rng = np.random.default_rng(0)
            last_forward_output = {
                "logits": _rng.standard_normal(
                    (1, config.prototype_vocab_size),
                    dtype=np.float32,
                ),
            }
            last_ipc_profile = None

        kv_pool = SchedulerKVPoolStub(
            size=config.kv_pool_size,
            page_size=config.kv_cache_block_size,
            dtype=np.float32,
            layer_num=1,
        )
        req_to_token_pool = ReqToTokenPool(
            size=config.max_requests,
            max_context_len=config.max_context_len,
        )
        token_allocator = PagedTokenToKVPoolAllocator(
            size=kv_pool.size,
            page_size=config.kv_cache_block_size,
            kvcache=kv_pool,
        )
        manager = SchedulerTokenizerService(config)
        import queue

        response_q = queue.Queue()
        paddings = build_precompile_paddings(config)

        scheduler = _SchedulerCore(
            manager=manager,
            runtime_config=config,
            response_queue=response_q,
            kv_pool=kv_pool,
            req_to_token_pool=req_to_token_pool,
            token_allocator=token_allocator,
            worker_coordinator=FakeWorkerCoordinator(),
            paddings=paddings,
        )
        return scheduler, response_q

    def test_logprobs_k_contracts(self):
        cases = [
            ("test_cap", 50, 20, 20),
            ("test_zero", 0, 0, 1),
            ("test_ok", 5, 5, 5),
        ]

        for (
            request_id,
            top_logprobs_num,
            expected_top_logprobs_num,
            expected_logprobs_k,
        ) in cases:
            scheduler, _ = self._make_scheduler()
            scheduler.handle_message(
                {
                    "cmd": "generate",
                    "request_id": request_id,
                    "req": {
                        "prompt": "hi",
                        "max_new_tokens": 1,
                        "return_logprob": True,
                        "top_logprobs_num": top_logprobs_num,
                    },
                }
            )

            state = scheduler.requests_by_id.get(request_id)
            assert state is not None
            assert state.return_logprob is True
            assert state.top_logprobs_num == expected_top_logprobs_num
            assert state.logprobs_k == expected_logprobs_k


class TestSchedulerLogprobsBatchFlags:
    def _make_scheduler(self):
        return TestLogprobsKCapping._make_scheduler(self)

    def test_logprobs_force_full_sampler_for_greedy_requests(self):
        cases = [
            (
                "r1",
                {"return_logprob": True, "top_logprobs_num": 3},
                True,
                3,
                True,
            ),
            ("r2", {}, False, 0, False),
        ]

        for (
            request_id,
            req_overrides,
            needs_logprobs,
            logprobs_k,
            use_full_sampler,
        ) in cases:
            scheduler, _ = self._make_scheduler()
            scheduler.handle_message(
                {
                    "cmd": "generate",
                    "request_id": request_id,
                    "req": {
                        "prompt": "hello",
                        "max_new_tokens": 1,
                        "temperature": 0.0,
                        **req_overrides,
                    },
                }
            )

            scheduler._admit_waiting_requests()
            batches = scheduler._get_next_batches()
            assert len(batches) >= 1
            fb = batches[0].build_forward_batch()
            assert fb.needs_logprobs is needs_logprobs
            assert fb.logprobs_k == logprobs_k
            assert fb.use_full_sampler is use_full_sampler


class TestOpenAILogprobsFormatting:
    """Tests that OpenAI serving helpers format logprobs correctly."""

    def test_build_logprobs_completions(self):
        from nkipy_serving.entrypoints.openai.serving_completions import _build_logprobs

        out = {
            "token_logprobs": [
                (-0.5, 10, "hello"),
                (-1.2, 20, " world"),
            ],
            "top_logprobs": [
                [(-0.5, 10, "hello"), (-0.8, 11, "hi"), (-1.0, 12, "hey")],
                [(-1.2, 20, " world"), (-1.5, 21, " there")],
            ],
        }
        lp = _build_logprobs(out)
        assert lp is not None
        assert lp.tokens == ["hello", " world"]
        assert lp.token_logprobs == [pytest.approx(-0.5), pytest.approx(-1.2)]
        assert len(lp.top_logprobs) == 2
        assert lp.top_logprobs[0] == {
            "hello": pytest.approx(-0.5),
            "hi": pytest.approx(-0.8),
            "hey": pytest.approx(-1.0),
        }
        assert lp.top_logprobs[1] == {
            " world": pytest.approx(-1.2),
            " there": pytest.approx(-1.5),
        }
        assert _build_logprobs({}) is None
        assert _build_logprobs({"token_logprobs": []}) is None

    def test_build_chat_logprobs(self):
        from nkipy_serving.entrypoints.openai.serving_chat import _build_chat_logprobs

        out = {
            "token_logprobs": [
                (-0.3, 5, "A"),
                (-0.7, 6, "B"),
            ],
            "top_logprobs": [
                [(-0.3, 5, "A"), (-0.9, 7, "C")],
                [(-0.7, 6, "B")],
            ],
        }
        cl = _build_chat_logprobs(out)
        assert cl is not None
        assert cl.content is not None
        assert len(cl.content) == 2
        assert cl.content[0].token == "A"
        assert cl.content[0].logprob == pytest.approx(-0.3)
        assert len(cl.content[0].top_logprobs) == 2
        assert cl.content[0].top_logprobs[0].token == "A"
        assert cl.content[1].token == "B"
        assert _build_chat_logprobs({}) is None


def _build_real_scheduler(vocab_size=256):
    """Build a real scheduler with numpy backend and deterministic logits worker."""
    import queue

    from nkipy_serving.config import RuntimeConfig
    from nkipy_serving.managers.scheduler import (
        SchedulerTokenizerService,
        _SchedulerCore,
    )
    from nkipy_serving.mem_cache.allocator import PagedTokenToKVPoolAllocator
    from nkipy_serving.mem_cache.memory_pool import ReqToTokenPool, SchedulerKVPoolStub
    from nkipy_serving.runtime.precompile_paddings import build_precompile_paddings

    config = RuntimeConfig(
        execution_backend="numpy",
        attention_backend="VanillaPagedAttention",
        paged_attn_impl="vanilla_paged_attention_kv_cache",
        prototype_vocab_size=vocab_size,
    )

    class RealLogitsWorker:
        """Returns sampled output via NumpyLogitsProcessor (matches real path)."""

        def __init__(self, vs):
            self._vocab_size = vs
            self._last = None
            self._count = 0

        def dispatch_forward_step(self, batch):
            self._count += 1
            total = batch.total_tokens
            logits = (
                np.random.RandomState(42 + self._count)
                .randn(total, self._vocab_size)
                .astype(np.float32)
            )
            # Make one token clearly dominant so greedy is deterministic.
            for pos in range(int(batch.real_total_tokens)):
                logits[pos, 7] = 20.0
            from nkipy_serving.sampling.device_batch import DeviceSamplingBatch
            from nkipy_serving.sampling.logits_processor_np import NumpyLogitsProcessor

            proc = NumpyLogitsProcessor()
            sb = DeviceSamplingBatch.from_forward_batch(batch)
            out = proc.forward(
                logits,
                batch.sample_mask,
                batch.query_start_loc,
                batch.batch_size,
                sampling_batch=sb,
                needs_logprobs=bool(batch.needs_logprobs),
                logprobs_k=int(batch.logprobs_k),
            ).to_shm_dict()
            if "next_token_ids" not in out and "top1_indices" in out:
                out["next_token_ids"] = out["top1_indices"].copy()
            self._last = out
            return f"req-{self._count}"

        def collect_forward_step(self, req_id):
            pass

        @property
        def last_forward_output(self):
            return self._last

        last_ipc_profile = None

    kv_pool = SchedulerKVPoolStub(
        size=config.kv_pool_size,
        page_size=config.kv_cache_block_size,
        dtype=np.float32,
        layer_num=1,
    )
    req_to_token_pool = ReqToTokenPool(
        size=config.max_requests,
        max_context_len=config.max_context_len,
    )
    token_allocator = PagedTokenToKVPoolAllocator(
        size=kv_pool.size,
        page_size=config.kv_cache_block_size,
        kvcache=kv_pool,
    )
    manager = SchedulerTokenizerService(config)
    response_q = queue.Queue()
    paddings = build_precompile_paddings(config)

    scheduler = _SchedulerCore(
        manager=manager,
        runtime_config=config,
        response_queue=response_q,
        kv_pool=kv_pool,
        req_to_token_pool=req_to_token_pool,
        token_allocator=token_allocator,
        worker_coordinator=RealLogitsWorker(vocab_size),
        paddings=paddings,
    )
    return scheduler, response_q


def _run_and_collect(scheduler, response_q, max_steps=30):
    from unittest.mock import patch

    from nkipy_serving.managers.detokenizer_manager import DetokenizerManager

    # Build a test detokenizer matching the scheduler's tokenizer.
    with patch.object(DetokenizerManager, "__init__", lambda self, *a, **kw: None):
        dm = DetokenizerManager.__new__(DetokenizerManager)
    dm._states = {}
    dm._tokenizer = scheduler.manager.tokenizer

    for _ in range(max_steps):
        scheduler.run_step()
    raw = []
    while not response_q.empty():
        raw.append(response_q.get_nowait())
    responses = []
    for msg in raw:
        responses.extend(dm.handle_message(msg))
    return responses


def _run_real_logprobs_request(
    request_id: str, req: dict[str, object]
) -> dict[str, object]:
    scheduler, response_q = _build_real_scheduler()
    scheduler.handle_message({"cmd": "generate", "request_id": request_id, "req": req})
    responses = _run_and_collect(scheduler, response_q)
    final = [
        r
        for r in responses
        if r.get("type") == "final" and r["request_id"] == request_id
    ]
    assert len(final) == 1 and final[0]["ok"]
    return final[0]["result"]


class TestSchedulerLogprobsRoundTrip:
    """Real end-to-end: request → scheduler → NumpyLogitsProcessor →
    scheduler receives logprobs via forward output → verifies numerical properties."""

    def test_requested_logprobs_roundtrip_contracts(self):
        result = _run_real_logprobs_request(
            "with-logprobs",
            {
                "prompt": "AB",
                "max_new_tokens": 4,
                "return_logprob": True,
                "top_logprobs_num": 5,
                "temperature": 0.0,
            },
        )
        assert len(result["token_logprobs"]) == result["completion_tokens"]
        assert len(result["top_logprobs"]) == result["completion_tokens"]
        for token_logprob, top_logprobs in zip(
            result["token_logprobs"], result["top_logprobs"], strict=True
        ):
            logprob, tid, _text = token_logprob
            assert logprob <= 0.0 + 1e-6
            assert tid == 7
            assert len(top_logprobs) <= 5
            assert all(entry[0] <= 0.0 + 1e-6 for entry in top_logprobs)
            sorted_vals = [entry[0] for entry in top_logprobs]
            assert sorted_vals == sorted(sorted_vals, reverse=True)
            np.testing.assert_allclose(logprob, top_logprobs[0][0], atol=1e-5)

    def test_no_logprobs_when_not_requested(self):
        result = _run_real_logprobs_request(
            "without-logprobs", {"prompt": "AB", "max_new_tokens": 1}
        )
        assert result.get("token_logprobs", []) == []
        assert result.get("top_logprobs", []) == []
