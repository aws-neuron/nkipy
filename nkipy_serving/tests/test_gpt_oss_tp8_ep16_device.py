"""GPT-OSS TP8+EP16 device integration tests (opt-in).

Expert parallelism serving contracts: tp_degree=8, ep_degree=16, total=128 cores.

Requires:
  - trn2.48xlarge (128 NeuronCores) with NEURON_LOGICAL_NC_CONFIG=1
  - local HF cache for unsloth/gpt-oss-120b-BF16
  - pytest flags: --run-integration --run-device-ep
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

pytestmark = [pytest.mark.integration, pytest.mark.device_ep]

_CONFIG = Path(__file__).resolve().parent / "runtime.tp8_ep16.gpt_oss.serving.test.json"
_PORT = 30110
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_READY_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_GPT_OSS_READY_TIMEOUT_S", "1800"))
_REQUEST_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_GPT_OSS_REQUEST_TIMEOUT_S", "300"))
_MODEL_CACHE_ID = "unsloth/gpt-oss-120b-BF16"


def _http_post_json(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return post_json(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


def _http_post_stream(path: str, body: dict[str, Any]) -> tuple[int, str]:
    return post_stream(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


@pytest.fixture(scope="module")
def gpt_oss_ep_server() -> dict[str, Any]:
    require_config(_CONFIG)
    require_local_snapshot(_MODEL_CACHE_ID)
    served_model = load_model_id(_CONFIG, default=_MODEL_CACHE_ID)

    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = "1"

    with launch_server(config=_CONFIG, port=_PORT, env=env):
        ready, init_error = wait_ready(
            _BASE_URL,
            timeout_s=_READY_TIMEOUT_S,
            any_500_is_terminal=True,
        )
        if not ready and init_error is not None:
            pytest.fail(
                "GPT-OSS EP server hit a terminal init error before readiness: "
                f"{init_error}"
            )
        if not ready:
            pytest.fail(
                f"GPT-OSS EP server failed to become ready within {_READY_TIMEOUT_S}s "
                f"(port={_PORT}, config={_CONFIG})"
            )
        yield {"model": served_model}


def test_gpt_oss_ep_chat_completion(gpt_oss_ep_server: dict[str, Any]) -> None:
    """Chat completion succeeds with expert parallelism."""
    status, body = _http_post_json(
        "/v1/chat/completions",
        {
            "model": gpt_oss_ep_server["model"],
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


def test_gpt_oss_ep_chat_stream(gpt_oss_ep_server: dict[str, Any]) -> None:
    """Streaming chat succeeds with expert parallelism."""
    status, sse_text = _http_post_stream(
        "/v1/chat/completions",
        {
            "model": gpt_oss_ep_server["model"],
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

    content_parts: list[str] = []
    for event in events:
        delta = event["choices"][0]["delta"]
        if "content" in delta:
            content_parts.append(str(delta["content"]))
    assert len("".join(content_parts)) > 0
