"""DeepSeek-V4 TP8+EP8 multi-bucket device integration test (opt-in).

Requires:
  - trn2.48xlarge with NEURON_LOGICAL_NC_CONFIG=1
  - converted DeepSeek-V4 FP8 checkpoint
  - prepared TP8/EP8 per-rank weights matching the selected replica degree
  - pytest flags: --run-integration --run-device-dsv4
"""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from device_server import (
    get_json,
    launch_server,
    post_json,
    require_config,
    wait_ready,
)

from nkipy_serving.config import (
    RuntimeConfig,
    load_runtime_config,
    validate_runtime_config,
)

pytestmark = [pytest.mark.integration, pytest.mark.device_dsv4]

_DEFAULT_MULTI_BUCKET_CONFIG = (
    Path(__file__).resolve().parent
    / "runtime.tp8_ep8_r1.deepseek_v4.multi_bucket_4k.test.json"
)
_MULTI_BUCKET_CONFIG = Path(
    os.getenv("NKIPY_SERVING_DSV4_DEVICE_CONFIG", str(_DEFAULT_MULTI_BUCKET_CONFIG))
).expanduser()
_PORT = 30114
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_READY_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_DSV4_READY_TIMEOUT_S", "1800"))
_REQUEST_TIMEOUT_S = int(os.getenv("NKIPY_SERVING_DSV4_REQUEST_TIMEOUT_S", "300"))


def _is_path_like(value: str) -> bool:
    return value.startswith(("/", ".", "~"))


def _require_local_path(value: str | None, *, label: str, env_var: str) -> Path:
    if value is None:
        pytest.skip(f"{label} is not configured; set {env_var}")
    if not _is_path_like(value):
        pytest.skip(f"{label} must be a local path for this smoke; set {env_var}")
    return Path(value).expanduser()


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        pytest.skip(f"Missing {label}: {path}")


def _http_get_json(path: str) -> dict[str, Any]:
    return get_json(_BASE_URL, path, timeout_s=_REQUEST_TIMEOUT_S)


def _http_post_json(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return post_json(_BASE_URL, path, body, timeout_s=_REQUEST_TIMEOUT_S)


def _load_dsv4_runtime_config(config: Path = _MULTI_BUCKET_CONFIG) -> RuntimeConfig:
    return load_runtime_config(str(config))


def _dsv4_sharding_label(runtime_config: RuntimeConfig) -> str:
    return (
        f"TP{int(runtime_config.tp_degree)}"
        f"/EP{int(runtime_config.ep_degree)}"
        f"/R{int(runtime_config.replica_degree)}"
        f"/ADP{int(runtime_config.attention_dp_degree)}"
    )


def test_deepseek_v4_multi_bucket_config() -> None:
    runtime_config = _load_dsv4_runtime_config()

    assert len(runtime_config.request_buckets) > 1
    assert len(runtime_config.token_buckets) > 1
    assert list(runtime_config.request_buckets) == sorted(
        set(runtime_config.request_buckets)
    )
    assert list(runtime_config.token_buckets) == sorted(
        set(runtime_config.token_buckets)
    )
    validate_runtime_config(runtime_config)


@pytest.fixture(scope="module")
def dsv4_server() -> Iterator[dict[str, Any]]:
    require_config(_MULTI_BUCKET_CONFIG)
    runtime_config = _load_dsv4_runtime_config()
    checkpoint_dir = _require_local_path(
        runtime_config.hf_model_id,
        label="DeepSeek-V4 checkpoint",
        env_var="NKIPY_SERVING_HF_MODEL_ID",
    )
    tokenizer_dir = _require_local_path(
        runtime_config.tokenizer_model_id,
        label="DeepSeek-V4 tokenizer",
        env_var="NKIPY_SERVING_TOKENIZER_MODEL_ID",
    )
    prepared_weight_dir = _require_local_path(
        runtime_config.dsv4_prepared_weight_dir,
        label="DeepSeek-V4 prepared weight directory",
        env_var="NKIPY_SERVING_DSV4_PREPARED_WEIGHT_DIR",
    )
    _require_path(
        checkpoint_dir / "config.json",
        "DeepSeek-V4 checkpoint",
    )
    _require_path(
        tokenizer_dir,
        "DeepSeek-V4 tokenizer",
    )
    _require_path(
        prepared_weight_dir,
        "DeepSeek-V4 prepared weights",
    )
    validate_runtime_config(runtime_config)
    served_model = runtime_config.model_id

    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = "1"
    env.setdefault("NKIPY_SERVING_DSV4_WARMUP_TRACE", "0")

    with launch_server(
        config=_MULTI_BUCKET_CONFIG,
        port=_PORT,
        env=env,
        terminate_timeout_s=30,
    ):
        ready, init_error = wait_ready(
            _BASE_URL,
            timeout_s=_READY_TIMEOUT_S,
            any_500_is_terminal=True,
        )
        if not ready and init_error is not None:
            pytest.fail(
                "DeepSeek-V4 server hit a terminal init error before readiness: "
                f"{init_error}"
            )
        if not ready:
            pytest.fail(
                f"DeepSeek-V4 server failed to become ready within {_READY_TIMEOUT_S}s "
                f"(port={_PORT}, config={_MULTI_BUCKET_CONFIG})"
            )
        yield {
            "model": served_model,
            "request_buckets": [int(b) for b in runtime_config.request_buckets],
            "token_buckets": [int(b) for b in runtime_config.token_buckets],
            "max_requests": int(runtime_config.max_requests),
            "sharding": _dsv4_sharding_label(runtime_config),
        }


def test_deepseek_v4_serving_smoke(dsv4_server: dict[str, Any]) -> None:
    model = dsv4_server["model"]

    info = _http_get_json("/get_server_info")
    assert info["served_model_name"] == model
    assert info["dsv4_runtime"]["sharding"] == dsv4_server["sharding"]
    assert info["dsv4_runtime"]["target_only"] is True
    assert info["dsv4_runtime"]["request_buckets"] == dsv4_server["request_buckets"]
    assert info["dsv4_runtime"]["token_buckets"] == dsv4_server["token_buckets"]

    chat_status, chat = _http_post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert chat_status == 200, chat
    choice = chat["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    assert chat["usage"]["completion_tokens"] > 0

    native_status, native = _http_post_json(
        "/generate",
        {
            "input_ids": [100] * 5,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "return_logprob": True,
            "top_logprobs_num": 0,
        },
    )
    assert native_status == 200, native
    assert native["batch_size"] == 1
    result = native["results"][0]
    assert len(result["completion_ids"]) == 1
    assert len(result["meta_info"]["output_token_logprobs"]) == 1

    # Long prefill (> SWA window = 128 tokens) exercises the bucketed
    # cache-write tail-window path when the config's token buckets allow it.
    max_context = max(int(b) for b in dsv4_server["token_buckets"])
    if max_context >= 256:
        long_status, long_resp = _http_post_json(
            "/generate",
            {
                "input_ids": [101] * 200,
                "max_new_tokens": 4,
                "temperature": 0.0,
            },
        )
        assert long_status == 200, long_resp
        long_result = long_resp["results"][0]
        assert len(long_result["completion_ids"]) == 4

    if max_context >= 2048:
        bucket2k_status, bucket2k_resp = _http_post_json(
            "/generate",
            {
                "input_ids": [102] * 1500,
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        assert bucket2k_status == 200, bucket2k_resp
        bucket2k_result = bucket2k_resp["results"][0]
        assert len(bucket2k_result["completion_ids"]) == 1

    if max_context >= 4096:
        bucket4k_status, bucket4k_resp = _http_post_json(
            "/generate",
            {
                "input_ids": [103] * 3500,
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        assert bucket4k_status == 200, bucket4k_resp
        bucket4k_result = bucket4k_resp["results"][0]
        assert len(bucket4k_result["completion_ids"]) == 1


def test_deepseek_v4_batch_prefill_multi_bucket(dsv4_server: dict[str, Any]) -> None:
    """Drive one native bs>1 long-prefill request under multi-bucket config."""
    max_requests = int(dsv4_server["max_requests"])
    if max_requests <= 1:
        pytest.skip(
            "batch>1 prefill requires request_buckets with max_requests > 1 "
            f"(got max_requests={max_requests}); set NKIPY_SERVING_REQUEST_BUCKETS"
        )

    max_context = max(int(b) for b in dsv4_server["token_buckets"])
    if max_context < 256:
        pytest.skip(
            "long-prefill multi-bucket coverage requires a token bucket >= 256 "
            f"(got token_buckets={dsv4_server['token_buckets']})"
        )

    info = _http_get_json("/get_server_info")
    assert info["dsv4_runtime"]["request_buckets"] == dsv4_server["request_buckets"]
    assert info["dsv4_runtime"]["token_buckets"] == dsv4_server["token_buckets"]

    n_requests = 2
    per_request_tokens = 200
    required_tokens = n_requests * per_request_tokens
    if max_context < required_tokens:
        pytest.skip(
            "batch>1 prefill coverage requires token bucket large enough for "
            f"total prompt tokens={required_tokens}; got token_buckets="
            f"{dsv4_server['token_buckets']}"
        )

    input_ids = [
        [100 + request_idx] * per_request_tokens for request_idx in range(n_requests)
    ]
    status, payload = _http_post_json(
        "/generate",
        {
            "input_ids": input_ids,
            "max_new_tokens": 2,
            "temperature": 0.0,
        },
    )
    assert status == 200, payload
    assert payload["batch_size"] == n_requests
    assert len(payload["results"]) == n_requests
    for result in payload["results"]:
        assert len(result["completion_ids"]) >= 1
        assert isinstance(result.get("text", ""), str)


def test_deepseek_v4_batch_decode(dsv4_server: dict[str, Any]) -> None:
    """Drive batch>1 decode so the DP-attention reduce buffer exercises the
    promoted (max_requests-wide) geometry.

    Fires ``max_requests`` requests concurrently, each generating enough tokens
    for the scheduler to co-schedule at least one batch>1 decode step.
    With the compile_bsz->max_requests promotion, every such step reshapes the
    flat reduce buffer x[:max_requests*1]; the fix sizes that buffer correctly.
    """
    max_requests = int(dsv4_server["max_requests"])
    if max_requests <= 1:
        pytest.skip(
            "batch>1 decode requires request_buckets with max_requests > 1 "
            f"(got max_requests={max_requests}); set NKIPY_SERVING_REQUEST_BUCKETS"
        )

    n_requests = max_requests
    prompt_token_ids = [[120 + i] * 5 for i in range(n_requests)]
    metrics_before = _http_get_json("/get_server_info")["scheduler_metrics"]
    max_decode_before = int(metrics_before.get("max_decode_batch_size", 0))

    def _one(input_ids: list[int]) -> tuple[int, dict[str, Any]]:
        return _http_post_json(
            "/generate",
            {
                "input_ids": input_ids,
                "max_new_tokens": 2,
                "temperature": 0.0,
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as pool:
        responses = list(pool.map(_one, prompt_token_ids))

    for status, payload in responses:
        assert status == 200, payload
        result = payload["results"][0]
        assert len(result["completion_ids"]) >= 1
        assert isinstance(result.get("text", ""), str)
    metrics_after = _http_get_json("/get_server_info")["scheduler_metrics"]
    max_decode_after = int(metrics_after.get("max_decode_batch_size", 0))
    assert max_decode_after > max(1, max_decode_before), metrics_after

    # Sanity: a follow-up single request still works after batched decode (the
    # promoted-geometry NEFFs and the batch-1 path coexist).
    status, payload = _http_post_json(
        "/generate",
        {"input_ids": [100] * 5, "max_new_tokens": 2, "temperature": 0.0},
    )
    assert status == 200, payload
    assert payload["batch_size"] == 1
    assert len(payload["results"][0]["completion_ids"]) >= 1
