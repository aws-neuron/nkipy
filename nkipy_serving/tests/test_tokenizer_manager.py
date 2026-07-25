"""Unit tests for TokenizerManager pre-tokenization.

Verifies that the tokenizer manager tokenizes text prompts before sending
them to the scheduler, so the scheduler thread never blocks on tokenization.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from nkipy_serving.managers.io_struct import GenerateReqInput
from nkipy_serving.managers.tokenizer_manager import TokenizerManager

# ---------------------------------------------------------------------------
# Fake tokenizer (matches _MockTokenizerManager pattern from test_scheduler.py)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic tokenizer: each character maps to its ord() value."""

    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        return np.asarray([ord(c) for c in text], dtype=np.int32)

    def decode(
        self, token_ids: np.ndarray | list[int], skip_special_tokens: bool = False
    ) -> str:
        return "".join(chr(int(t)) for t in token_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer_manager() -> tuple[TokenizerManager, list[dict[str, Any]]]:
    """Build a TokenizerManager with a fake tokenizer and a captured send log.

    Returns (manager, sent_payloads) where sent_payloads collects every
    payload passed to _send_scheduler_payload.
    """
    sent: list[dict[str, Any]] = []

    # Patch HfTokenizer construction so we don't need a real model.
    with patch.object(TokenizerManager, "__init__", lambda self, *a, **kw: None):
        mgr = TokenizerManager.__new__(TokenizerManager)

    # Wire up the minimum attributes the proxy generation path needs.
    mgr.tokenizer = _FakeTokenizer()
    mgr._proxy_mode = True
    mgr._proxy_request_states = {}
    mgr._proxy_request_states_lock = asyncio.Lock()
    mgr._proxy_control_waiters = {}
    mgr._proxy_control_waiters_lock = asyncio.Lock()
    mgr._scheduler_timeout_s = 0.01
    mgr._http_profile_writer = None

    # Stub _ensure_proxy_response_router_started (no-op).
    mgr._ensure_proxy_response_router_started = AsyncMock()

    # Capture payloads sent to the scheduler.
    async def _capture_send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    setattr(mgr, "_send_scheduler_payload", _capture_send)

    return mgr, sent


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreTokenization:
    """TokenizerManager._start_proxy_generation pre-tokenizes text prompts."""

    def test_prompt_tokenization_contracts(self):
        cases = [
            (
                GenerateReqInput(prompt="ABC", max_new_tokens=4),
                {"prompt": "ABC", "input_ids": [ord("A"), ord("B"), ord("C")]},
            ),
            (
                GenerateReqInput(text="XY", max_new_tokens=2),
                {"text": "XY", "input_ids": [ord("X"), ord("Y")]},
            ),
            (
                GenerateReqInput(input_ids=[10, 20, 30], max_new_tokens=4),
                {"prompt": None, "text": None, "input_ids": [10, 20, 30]},
            ),
            (
                GenerateReqInput(prompt="AB", text="XYZ", max_new_tokens=1),
                {"prompt": "AB", "text": "XYZ", "input_ids": [ord("A"), ord("B")]},
            ),
            (
                GenerateReqInput(max_new_tokens=0),
                {"prompt": None, "input_ids": None},
            ),
        ]

        for req, expected in cases:
            mgr, sent = _make_tokenizer_manager()
            _run(mgr._start_proxy_generation(req))

            assert len(sent) == 1
            payload_req = sent[0]["req"]
            for key, value in expected.items():
                assert payload_req[key] == value

    def test_sampling_params_preserved(self):
        """Non-prompt fields are forwarded unchanged."""
        mgr, sent = _make_tokenizer_manager()
        req = GenerateReqInput(
            prompt="Hi",
            max_new_tokens=8,
            temperature=0.5,
            top_k=10,
            stream=True,
            stop=["end"],
        )

        _run(mgr._start_proxy_generation(req))

        payload_req = sent[0]["req"]
        assert payload_req["max_new_tokens"] == 8
        assert payload_req["temperature"] == 0.5
        assert payload_req["top_k"] == 10
        assert payload_req["stream"] is True
        assert payload_req["stop"] == ["end"]

    def test_original_req_not_mutated(self):
        """dataclasses.replace creates a copy; original req is untouched."""
        mgr, sent = _make_tokenizer_manager()
        req = GenerateReqInput(prompt="AB", max_new_tokens=1)

        _run(mgr._start_proxy_generation(req))

        # Original req still has prompt text.
        assert req.prompt == "AB"
        assert req.input_ids is None

    def test_registration_is_cleaned_up_when_send_is_cancelled(self):
        mgr, _ = _make_tokenizer_manager()

        async def _cancel_send(payload: dict[str, Any]) -> None:
            raise asyncio.CancelledError()

        setattr(mgr, "_send_scheduler_payload", _cancel_send)

        with pytest.raises(asyncio.CancelledError):
            _run(mgr._start_proxy_generation(GenerateReqInput(prompt="AB")))

        assert mgr._proxy_request_states == {}

    def test_control_waiter_is_cleaned_up_when_send_is_cancelled(self):
        mgr, _ = _make_tokenizer_manager()

        async def _cancel_send(payload: dict[str, Any]) -> None:
            raise asyncio.CancelledError()

        setattr(mgr, "_send_scheduler_payload", _cancel_send)

        with pytest.raises(asyncio.CancelledError):
            _run(mgr._request_scheduler_control("pause"))

        assert mgr._proxy_control_waiters == {}

    def test_control_waiter_is_cleaned_up_on_timeout(self):
        mgr, sent = _make_tokenizer_manager()
        mgr._scheduler_timeout_s = 0.001

        with pytest.raises(RuntimeError, match="Timed out waiting for scheduler pause"):
            _run(mgr._request_scheduler_control("pause"))

        assert len(sent) == 1
        assert mgr._proxy_control_waiters == {}

    def test_request_state_is_cleaned_up_when_generate_once_times_out(self):
        mgr, sent = _make_tokenizer_manager()
        mgr._scheduler_timeout_s = 0.001

        with pytest.raises(
            RuntimeError, match="Timed out waiting for scheduler response"
        ):
            _run(mgr.generate_once(GenerateReqInput(prompt="AB")))

        assert len(sent) == 1
        assert mgr._proxy_request_states == {}

    def test_request_state_is_cleaned_up_when_generate_stream_times_out(self):
        mgr, sent = _make_tokenizer_manager()
        mgr._scheduler_timeout_s = 0.001

        async def _consume_stream() -> None:
            async for _ in mgr.generate_stream(GenerateReqInput(prompt="AB")):
                pass

        with pytest.raises(
            RuntimeError, match="Timed out waiting for scheduler stream"
        ):
            _run(_consume_stream())

        assert len(sent) == 1
        assert mgr._proxy_request_states == {}
