"""Qwen3 dense TP=8 serving integration test."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from device_server import (
    get_json,
    launch_server,
    load_model_id,
    parse_sse_events,
    post_json,
    post_stream,
    require_config,
    require_local_snapshot,
    wait_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.device_qwen3_dense]

_CONFIG = Path(__file__).resolve().parent / "runtime.tp8.qwen3.serving.test.json"
_PORT = 30100
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_HEALTH_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S", "600"))
_REQUEST_TIMEOUT_S = 120
_SERVED_MODEL_ID = load_model_id(_CONFIG, default="Qwen/Qwen3-0.6B")


def _get_json(path: str) -> dict[str, Any]:
    return get_json(_BASE_URL, path, timeout_s=_REQUEST_TIMEOUT_S)


def _post_json(
    path: str,
    body: dict[str, Any],
    *,
    expect_status: int = 200,
) -> dict[str, Any]:
    status, payload = post_json(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)
    assert status == expect_status, payload
    return payload


def _post_stream(path: str, body: dict[str, Any]) -> str:
    status, payload = post_stream(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)
    assert status == 200, payload
    return payload


@pytest.fixture(scope="module")
def qwen3_dense_server() -> dict[str, str]:
    require_config(_CONFIG)
    require_local_snapshot(_SERVED_MODEL_ID)

    with launch_server(config=_CONFIG, port=_PORT, terminate_timeout_s=10):
        ready, _init_error = wait_ready(_BASE_URL, timeout_s=_HEALTH_TIMEOUT_S)
        if not ready:
            pytest.fail(
                f"Qwen3 dense TP8 server failed to become ready within "
                f"{_HEALTH_TIMEOUT_S}s (port={_PORT}, config={_CONFIG})"
            )
        yield {"model": _SERVED_MODEL_ID}


def test_qwen3_dense_tp8_serving_end_to_end(
    qwen3_dense_server: dict[str, str],
) -> None:
    model = qwen3_dense_server["model"]

    info = _get_json("/get_server_info")
    assert info["variant_count"] == 0
    warmup = info["warmup_summary"]
    for key in ("kv_pool_size", "block_size", "num_blocks"):
        assert key in warmup

    models = _get_json("/v1/models")
    assert models["object"] == "list"
    assert len(models["data"]) == 1
    assert models["data"][0]["id"] == model

    version = _get_json("/version")
    tokenizer_info = _get_json("/tokenizer_info")
    assert "version" in version
    assert tokenizer_info["model_id"] == model
    assert tokenizer_info["vocab_size"] > 0

    comp = _post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": "Say hello in one sentence.",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": False,
        },
    )
    text = comp["choices"][0]["text"]
    assert len(text) > 0

    chat = _post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": False,
        },
    )
    msg = chat["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert len(msg["content"]) > 0

    usage = comp["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    info2 = _get_json("/get_server_info")
    metrics = info2.get("scheduler_metrics", {})
    assert metrics.get("total_completed", 0) >= 2
    assert metrics.get("total_generated_tokens", 0) > 0
    assert "last_decode_throughput" in metrics
    assert metrics["total_time_to_first_token_s"] > 0
    assert metrics.get("total_requests_with_ttft", 0) >= 2

    native = _post_json(
        "/generate",
        {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "max_new_tokens": 1,
            "temperature": 0.0,
            "return_logprob": True,
            "top_logprobs_num": 0,
        },
    )
    assert native["batch_size"] == 2
    assert native["n"] == 1
    assert len(native["results"]) == 2
    for item in native["results"]:
        assert len(item["completion_ids"]) == 1
        assert len(item["meta_info"]["output_token_logprobs"]) == 1

    toks = _post_json("/v1/tokenize", {"prompt": ["Hello world", "Hello"]})
    assert len(toks["tokens"]) == 2
    detok = _post_json(
        "/v1/detokenize",
        {"tokens": toks["tokens"], "skip_special_tokens": False},
    )
    assert len(detok["text"]) == 2

    error_body = _post_json(
        "/v1/embeddings",
        {"input": "hello"},
        expect_status=501,
    )
    assert "not supported" in error_body["message"]

    logprobs_comp = _post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": "The capital",
            "max_tokens": 3,
            "temperature": 0.0,
            "logprobs": 3,
            "stream": False,
        },
    )
    lp = logprobs_comp["choices"][0]["logprobs"]
    assert lp is not None
    assert len(lp["tokens"]) > 0
    assert len(lp["tokens"]) == len(lp["token_logprobs"])
    assert len(lp["tokens"]) == len(lp["top_logprobs"])
    for entries in lp["top_logprobs"]:
        if entries is not None:
            assert len(entries) <= 3

    stop_comp = _post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": "Say hello in one sentence.",
            "max_tokens": 20,
            "temperature": 0.0,
            "stop": ".",
            "stream": False,
        },
    )
    stop_text = stop_comp["choices"][0]["text"]
    stop_reason = stop_comp["choices"][0]["finish_reason"]
    assert "." not in stop_text
    assert stop_reason in ("stop", "length")

    sp_comp = _post_json(
        "/v1/completions",
        {
            "model": model,
            "prompt": "Hello",
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.1,
            "stream": False,
        },
    )
    assert len(sp_comp["choices"][0]["text"]) > 0

    sse_text = _post_stream(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": True,
        },
    )
    assert "data: [DONE]" in sse_text
    events = parse_sse_events(sse_text)
    assert len(events) >= 2
    first_delta = events[0]["choices"][0]["delta"]
    assert first_delta.get("role") == "assistant"
    assert events[0]["choices"][0]["finish_reason"] is None
    last = events[-1]
    assert last["choices"][0]["finish_reason"] in ("stop", "length")
