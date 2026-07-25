"""Fast unit tests for HTTP-server helpers and native route glue."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import nkipy_serving.entrypoints.http_server as http_server
from nkipy_serving.config import RuntimeConfig
from nkipy_serving.managers.io_struct import GenerateReqInput


@contextmanager
def _fake_global_state(tokenizer_manager, runtime_config=None):
    prev_state = http_server._global_state
    prev_init_task = http_server._init_task
    prev_init_error = http_server._init_error
    try:
        http_server._init_task = None
        http_server._init_error = None
        http_server.set_global_state(
            http_server._GlobalState(
                tokenizer_manager=tokenizer_manager,
                runtime_config=runtime_config
                or SimpleNamespace(
                    max_context_len=128,
                    execution_backend="numpy",
                    tokenizer_model_id="fake-tokenizer",
                    model_id="fake-model",
                ),
                variant_count=0,
                warmup_summary=None,
                process_group=None,
            )
        )
        yield
    finally:
        http_server.set_global_state(prev_state)
        http_server._init_task = prev_init_task
        http_server._init_error = prev_init_error


class _BatchManager:
    def __init__(self) -> None:
        self.call_count = 0
        self.completed: list[str | None] = []
        self.cancelled: list[str | None] = []

    async def generate_once(
        self, req: GenerateReqInput, request_id: str | None = None
    ) -> dict:
        self.call_count += 1
        if self.call_count == 1:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                self.cancelled.append(request_id)
                raise
            self.completed.append(request_id)
            return {"text": "ok"}
        raise RuntimeError("boom")


class _StreamManager:
    def __init__(self) -> None:
        self.seen_request_id: str | None = None

    async def generate_stream(
        self, req: GenerateReqInput, request_id: str | None = None
    ):
        self.seen_request_id = request_id
        yield {"text": "x", "token_id": 1, "finish_reason": None}


class _SuccessManager:
    def __init__(self) -> None:
        self.seen_request_id: str | None = None

    async def generate_once(
        self, req: GenerateReqInput, request_id: str | None = None
    ) -> dict:
        self.seen_request_id = request_id
        return {"text": "ok"}


class _RejectingManager:
    served_model_name = "fake-model"

    async def generate_once(
        self, req: GenerateReqInput, request_id: str | None = None
    ) -> dict:
        raise AssertionError("generate_once should not be called")

    async def get_scheduler_metrics(self) -> dict:
        return {"proxy_mode": False}


class _EmptyErrorManager:
    served_model_name = "fake-model"

    async def generate_once(
        self,
        req: GenerateReqInput,
        request_id: str | None = None,
    ) -> dict:
        raise AssertionError()

    async def get_scheduler_metrics(self) -> dict:
        return {"proxy_mode": False}


def test_run_native_batch_waits_for_sibling_tasks_before_raising():
    manager = _BatchManager()
    items = [
        {"req": GenerateReqInput(prompt="Hello"), "request_id": "req-ok"},
        {"req": GenerateReqInput(prompt="World"), "request_id": "req-fail"},
    ]

    with _fake_global_state(manager):
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(http_server._run_native_batch(items))

    assert manager.completed == ["req-ok"]
    assert manager.cancelled == []


def test_native_generate_reports_empty_exception_type():
    manager = _EmptyErrorManager()

    with _fake_global_state(manager):
        with TestClient(http_server.app) as client:
            resp = client.post(
                "/generate",
                json={"prompt": "Hello", "max_new_tokens": 1},
            )

    assert resp.status_code == 400
    assert resp.json()["error"].startswith("AssertionError:")


def test_native_generate_forwards_request_id_for_stream_and_once():
    stream_manager = _StreamManager()

    with _fake_global_state(stream_manager):
        with TestClient(http_server.app) as client:
            stream_resp = client.post(
                "/generate",
                json={
                    "prompt": "Hello",
                    "max_new_tokens": 1,
                    "stream": True,
                    "request_id": "stream-123",
                },
            )

    assert stream_resp.status_code == 200
    assert "data: [DONE]" in stream_resp.text
    assert stream_manager.seen_request_id == "stream-123"

    once_manager = _SuccessManager()
    with _fake_global_state(once_manager):
        with TestClient(http_server.app) as client:
            once_resp = client.post(
                "/generate",
                json={
                    "prompt": "Hello",
                    "max_new_tokens": 1,
                    "request_id": "req-123",
                },
            )

    assert once_resp.status_code == 200
    assert once_resp.json() == {
        "results": [
            {
                "request_id": "req-123",
                "index": 0,
                "sample_index": 0,
                "text": "ok",
            }
        ],
        "batch_size": 1,
        "n": 1,
    }
    assert once_manager.seen_request_id == "req-123"


def test_normalizers_reject_empty_inputs():
    with pytest.raises(ValueError, match="input_ids must contain at least one token"):
        http_server._normalize_input_ids_batch([])

    with pytest.raises(
        ValueError, match="input_ids rows must contain at least one token"
    ):
        http_server._normalize_input_ids_batch([[]])

    with pytest.raises(
        ValueError, match="text/prompt batch must contain at least one item"
    ):
        http_server._normalize_text_batch([])


def test_generate_rejects_empty_text_batches():
    manager = _RejectingManager()

    with _fake_global_state(manager):
        with TestClient(http_server.app) as client:
            for body in ({"text": []}, {"prompt": []}, {"prompts": []}):
                resp = client.post("/generate", json=body)
                assert resp.status_code == 400
                assert "must contain at least one item" in resp.json()["error"]


def test_generate_rejects_invalid_json_body():
    manager = _RejectingManager()

    with _fake_global_state(manager):
        with TestClient(http_server.app) as client:
            resp = client.post(
                "/generate",
                content=b"{not-json",
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 400
    assert resp.json()["error"] == "request body must be valid JSON"


def test_generate_rejects_non_object_json_body():
    manager = _RejectingManager()

    with _fake_global_state(manager):
        with TestClient(http_server.app) as client:
            resp = client.post("/generate", json=["hello"])

    assert resp.status_code == 400
    assert resp.json()["error"] == "request body must be a JSON object"


def test_generate_is_not_admitted_before_runtime_ready(monkeypatch):
    async def _noop_init(_server_args):
        return None

    prev_state = http_server._global_state
    prev_init_task = http_server._init_task
    prev_init_error = http_server._init_error
    monkeypatch.setattr(http_server, "_ensure_global_state_init_started", _noop_init)
    try:
        http_server.set_global_state(None)
        http_server._init_task = None
        http_server._init_error = None
        with TestClient(http_server.app) as client:
            resp = client.post(
                "/generate",
                json={"prompt": "Hello", "max_new_tokens": 1},
            )
    finally:
        http_server.set_global_state(prev_state)
        http_server._init_task = prev_init_task
        http_server._init_error = prev_init_error

    assert resp.status_code == 503
    assert resp.json()["error"] == "warming_up"


# ---- Tests moved from integration (no real server needed) ----


def test_metadata_endpoints():
    manager = _RejectingManager()
    with _fake_global_state(manager):
        with TestClient(http_server.app) as client:
            health = client.get("/health")
            version = client.get("/version")
            models = client.get("/v1/models")

    assert health.status_code == 200
    assert version.status_code == 200
    assert "version" in version.json()
    assert models.status_code == 200
    data = models.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "fake-model"


def test_unsupported_generation_adjacent_routes_return_501():
    manager = _RejectingManager()
    with _fake_global_state(manager):
        with TestClient(http_server.app) as client:
            for path in (
                "/v1/embeddings",
                "/pooling",
                "/classify",
                "/rerank",
                "/v1/score",
            ):
                resp = client.post(path, json={"input": "Hello"})
                assert resp.status_code == 501
                body = resp.json()
                assert body["object"] == "error"
                assert (
                    "not supported" in body["message"]
                    or "not implemented" in body["message"]
                )


def test_get_server_info_reports_dsv4_shape_and_lane_metadata():
    class _InfoManager:
        served_model_name = "deepseek-ai/DeepSeek-V4-Flash"

        async def get_scheduler_metrics(self) -> dict:
            return {"attention_dp_lane_metrics": {"attention_dp_degree": 16}}

        async def get_lane_metadata(self) -> dict:
            return {
                "attention_dp_degree": 16,
                "lane_routes": [{"lane": 0, "replica": 0}],
                "lane_metadata": {0: {"rank": 0, "attn_lane": 0}},
            }

    runtime = RuntimeConfig(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        hf_model_id="/checkpoints/DeepSeek-V4-Flash-neuron-fp8-noscale",
        attention_backend="Dsv4SparseAttention",
        paged_attn_impl="dsv4_sparse_attention",
        tp_degree=8,
        ep_degree=8,
        replica_degree=2,
        attention_dp_degree=16,
        dsv4_disable_mtp=True,
        request_buckets=(2,),
        token_buckets=(384,),
        kv_pool_size=384,
        dsv4_state_size=384,
    )

    with _fake_global_state(_InfoManager(), runtime):
        with TestClient(http_server.app) as client:
            resp = client.get("/get_server_info")

    assert resp.status_code == 200
    body = resp.json()
    assert body["runtime_config"]["model_id"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert body["dsv4_runtime"]["sharding"] == "TP8/EP8/R2/ADP16"
    assert body["dsv4_runtime"]["target_only"] is True
    assert body["dsv4_runtime"]["request_buckets"] == [2]
    assert body["dsv4_runtime"]["token_buckets"] == [384]
    assert body["dsv4_runtime"]["state_size"] == 384
    assert body["lane_metadata"]["attention_dp_degree"] == 16
    assert body["lane_metadata"]["lane_routes"] == [{"lane": 0, "replica": 0}]
