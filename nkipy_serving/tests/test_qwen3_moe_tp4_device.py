"""Qwen3 MoE TP4 device integration tests (opt-in).

Requires:
  - Trn2/Neuron runtime
  - local HF cache for Qwen/Qwen3-30B-A3B-Thinking-2507
  - pytest flags: --run-integration --run-device-qwen3-moe

TP4-only Qwen3 MoE should run without ``NEURON_LOGICAL_NC_CONFIG=1``. The
fixture clears that env var for the server subprocess so this suite can run in
the same parent test session as EP tests. If worker init still fails with a
known Neuron resource/runtime error, the suite skips instead of sitting on the
long readiness timeout.
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

pytestmark = [pytest.mark.integration, pytest.mark.device_qwen3_moe]

_CONFIG = Path(__file__).resolve().parent / "runtime.tp4.qwen3_moe.serving.test.json"
_PORT = 30109
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_READY_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_QWEN3_MOE_READY_TIMEOUT_S", "1800"))
_REQUEST_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_QWEN3_MOE_REQUEST_TIMEOUT_S", "300"))
_MODEL_CACHE_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
_UNSUPPORTED_INIT_ERROR_SNIPPETS = (
    "failed to allocate tensor",
    "failed to load model",
    "nrt_resource",
    "worker crashed during init",
    "scheduler subprocess initialization failed",
)


def _http_post_json(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return post_json(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


def _http_post_stream(path: str, body: dict[str, Any]) -> tuple[int, str]:
    return post_stream(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


@pytest.fixture(scope="module")
def qwen3_moe_server() -> dict[str, Any]:
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
                f"Qwen3 MoE TP4-only init is unsupported on this host: {init_error}"
            )
        if not ready:
            pytest.fail(
                f"Qwen3 MoE server failed to become ready within {_READY_TIMEOUT_S}s "
                f"(port={_PORT}, config={_CONFIG})"
            )
        yield {"model": served_model}


def test_qwen3_moe_chat_completion(qwen3_moe_server: dict[str, Any]) -> None:
    """Chat completion succeeds on the TP4 MoE device path."""
    status, body = _http_post_json(
        "/v1/chat/completions",
        {
            "model": qwen3_moe_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert status == 200, body
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str)
    assert len(message["content"]) > 0


def test_qwen3_moe_chat_stream_produces_content(
    qwen3_moe_server: dict[str, Any],
) -> None:
    status, sse_text = _http_post_stream(
        "/v1/chat/completions",
        {
            "model": qwen3_moe_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": True,
        },
    )
    assert status == 200
    assert "data: [DONE]" in sse_text

    events = parse_sse_events(sse_text)
    assert len(events) >= 2
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"

    content_parts: list[str] = []
    for event in events:
        delta = event["choices"][0]["delta"]
        if "content" in delta:
            content_parts.append(str(delta["content"]))

    full_content = "".join(content_parts)
    assert len(full_content) > 0


def test_qwen3_moe_chat_completion_non_greedy_sampling(
    qwen3_moe_server: dict[str, Any],
) -> None:
    status, body = _http_post_json(
        "/v1/chat/completions",
        {
            "model": qwen3_moe_server["model"],
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 32,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "stream": False,
            "seed": 123,
        },
    )
    assert status == 200, body
    choice = body["choices"][0]
    assert choice["finish_reason"] in ("stop", "length")
    message = choice["message"]
    assert isinstance(message["content"], str)
    assert len(message["content"]) > 0
