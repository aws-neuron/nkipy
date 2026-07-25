"""GPT-OSS TP8 device integration tests (opt-in).

Requires:
  - Trn2/Neuron runtime
  - local HF cache for unsloth/gpt-oss-120b-BF16
  - pytest flags: --run-integration --run-device-gpt-oss

TP8-only GPT-OSS should run without ``NEURON_LOGICAL_NC_CONFIG=1``. The fixture
clears that env var for the server subprocess so this suite can run in the same
parent test session as EP tests. If worker init still fails with a known Neuron
resource/runtime error, the suite skips and the supported EP16 device suite in
``test_gpt_oss_tp8_ep16_device.py`` remains the fallback validation path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from device_server import (
    launch_server,
    load_model_id,
    parse_sse_events,
    post_json,
    post_stream,
    require_config,
    require_local_snapshot,
    wait_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.device_gpt_oss]

_CONFIG = Path(__file__).resolve().parent / "runtime.tp8.gpt_oss.serving.test.json"
_PORT = 30108
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_READY_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S", "1800"))
_REQUEST_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_GPT_OSS_REQUEST_TIMEOUT_S", "300"))
_MODEL_CACHE_ID = "unsloth/gpt-oss-120b-BF16"
_UNSUPPORTED_INIT_ERROR_SNIPPETS = (
    "failed to allocate tensor",
    "nrt_resource",
    "worker crashed during init",
    "scheduler subprocess initialization failed",
)


def _http_post_json(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return post_json(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


def _http_post_stream(path: str, body: dict[str, Any]) -> tuple[int, str]:
    return post_stream(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


@pytest.fixture(scope="module")
def gpt_oss_server() -> dict[str, Any]:
    require_config(_CONFIG)
    require_local_snapshot(_MODEL_CACHE_ID)
    served_model = load_model_id(_CONFIG, default=_MODEL_CACHE_ID)
    env = {k: v for k, v in os.environ.items() if k != "NEURON_LOGICAL_NC_CONFIG"}

    with launch_server(config=_CONFIG, port=_PORT, env=env):
        ready, init_error = wait_ready(
            _BASE_URL,
            timeout_s=_READY_TIMEOUT_S,
            terminal_error_snippets=_UNSUPPORTED_INIT_ERROR_SNIPPETS,
        )
        if not ready and init_error is not None:
            pytest.skip(
                f"GPT-OSS TP8-only init is unsupported on this host: {init_error}"
            )
        if not ready:
            pytest.fail(
                f"GPT-OSS server failed to become ready within {_READY_TIMEOUT_S}s "
                f"(port={_PORT}, config={_CONFIG})"
            )
        yield {"model": served_model}


def test_gpt_oss_chat_completion_with_reasoning(
    gpt_oss_server: dict[str, Any],
) -> None:
    status, body = _http_post_json(
        "/v1/chat/completions",
        {
            "model": gpt_oss_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
            "separate_reasoning": True,
        },
    )
    assert status == 200, body
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str)
    assert len(message["content"]) > 0
    assert isinstance(message.get("reasoning_content"), str)
    assert len(message["reasoning_content"]) > 0


def test_gpt_oss_chat_stream_reasoning_before_content(
    gpt_oss_server: dict[str, Any],
) -> None:
    status, sse_text = _http_post_stream(
        "/v1/chat/completions",
        {
            "model": gpt_oss_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": True,
            "separate_reasoning": True,
        },
    )
    assert status == 200
    assert "data: [DONE]" in sse_text

    events = parse_sse_events(sse_text)
    assert len(events) >= 2
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"

    reasoning_idx = None
    content_idx = None
    content_parts: list[str] = []
    for i, event in enumerate(events):
        delta = event["choices"][0]["delta"]
        if reasoning_idx is None and "reasoning_content" in delta:
            reasoning_idx = i
        if "content" in delta:
            if content_idx is None:
                content_idx = i
            content_parts.append(str(delta["content"]))

    assert reasoning_idx is not None
    assert content_idx is not None
    assert reasoning_idx < content_idx
    assert len("".join(content_parts)) > 0


def test_gpt_oss_chat_completion_non_greedy_sampling(
    gpt_oss_server: dict[str, Any],
) -> None:
    status, body = _http_post_json(
        "/v1/chat/completions",
        {
            "model": gpt_oss_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 32,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "stream": False,
            "separate_reasoning": True,
        },
    )
    assert status == 200, body
    choice = body["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] in ("stop", "length")
    assert message["role"] == "assistant"
    assert isinstance(message.get("content"), str)
    assert "reasoning_content" in message
    assert body["usage"]["completion_tokens"] > 0


def test_gpt_oss_chat_logprobs(gpt_oss_server: dict[str, Any]) -> None:
    # Logprobs should succeed for GPT-OSS (device log_softmax + top-k).
    status, resp = _http_post_json(
        "/v1/chat/completions",
        {
            "model": gpt_oss_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 4,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 3,
            "stream": False,
        },
    )
    assert status == 200, f"Logprobs request failed: {resp}"
    choice = resp["choices"][0]
    assert choice.get("logprobs") is not None
    content_logprobs = choice["logprobs"]["content"]
    assert len(content_logprobs) > 0
    for entry in content_logprobs:
        assert entry["logprob"] <= 0.0 + 1e-6
        assert len(entry["top_logprobs"]) <= 3
