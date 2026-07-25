"""Qwen3 dense TP=8 live reload integration test."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from device_server import (
    launch_server,
    load_model_id,
    post_json,
    require_config,
    require_local_snapshot,
    wait_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.device_qwen3_dense]

_CONFIG = Path(__file__).resolve().parent / "runtime.tp8.qwen3.serving.test.json"
_PORT = 30112
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_HEALTH_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_QWEN3_DENSE_READY_TIMEOUT_S", "900"))
_REQUEST_TIMEOUT_S = 180
_SERVED_MODEL_ID = load_model_id(_CONFIG, default="Qwen/Qwen3-0.6B")


def _post_json(path: str, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    status, payload = post_json(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)
    return payload, status


def _post_json_expect_error(
    path: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    return _post_json(path, body)


def _single_generate_text(payload: dict[str, Any]) -> str:
    results = payload["results"]
    assert len(results) == 1
    text = results[0]["text"]
    assert isinstance(text, str)
    return text


@pytest.fixture(scope="module")
def qwen3_dense_reload_server() -> dict[str, str]:
    require_config(_CONFIG)
    snapshot_path = require_local_snapshot(_SERVED_MODEL_ID)

    with launch_server(config=_CONFIG, port=_PORT, terminate_timeout_s=20):
        ready, _init_error = wait_ready(_BASE_URL, timeout_s=_HEALTH_TIMEOUT_S)
        if not ready:
            pytest.fail(
                f"Qwen3 dense TP8 reload server failed to become ready within "
                f"{_HEALTH_TIMEOUT_S}s (port={_PORT}, config={_CONFIG})"
            )
        yield {"snapshot_path": snapshot_path}


def test_qwen3_dense_tp8_reload_weights_from_disk(
    qwen3_dense_reload_server: dict[str, str],
) -> None:
    before, status = _post_json(
        "/generate",
        {
            "prompt": "Reload validation prompt",
            "max_new_tokens": 4,
            "temperature": 0.0,
        },
    )
    assert status == 200
    before_text = _single_generate_text(before)
    assert len(before_text) > 0

    reload_result, status = _post_json(
        "/reload_weights_from_disk",
        {"model_path": qwen3_dense_reload_server["snapshot_path"]},
    )
    assert status == 200, reload_result
    assert reload_result["ok"] is True
    assert reload_result["success"] is True

    after, status = _post_json(
        "/generate",
        {
            "prompt": "Reload validation prompt",
            "max_new_tokens": 4,
            "temperature": 0.0,
        },
    )
    assert status == 200
    assert _single_generate_text(after) == before_text

    bad_result, status = _post_json_expect_error("/reload_weights_from_disk", {})
    assert status == 400
    assert "error" in bad_result

    bad_result, status = _post_json_expect_error(
        "/reload_weights_from_disk",
        {"model_path": "/nonexistent/nkipy-serving-reload-model"},
    )
    assert status == 400
    assert bad_result["ok"] is False
    assert bad_result["success"] is False
