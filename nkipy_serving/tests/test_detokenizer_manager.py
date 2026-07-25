"""Contract tests for DetokenizerManager."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from nkipy_serving.managers.detokenizer_manager import DetokenizerManager

_VOCAB_SIZE = 256


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        arr = np.asarray(ids, dtype=np.int32)
        return "".join(chr(int(t) % _VOCAB_SIZE) for t in arr)


def _make_detokenizer() -> DetokenizerManager:
    with patch.object(DetokenizerManager, "__init__", lambda self, *a, **kw: None):
        dm = DetokenizerManager.__new__(DetokenizerManager)
    dm._states = {}
    dm._tokenizer = _FakeTokenizer()
    return dm


def _finish_message(**overrides) -> dict:
    msg = {
        "type": "finish",
        "request_id": "r1",
        "generated_ids": [65, 66, 67],
        "prompt_ids": [],
        "finish_reason": "length",
        "stop_strs": [],
        "no_stop_trim": False,
        "first_scheduled_ts": 0.0,
        "first_token_ts": 0.0,
        "cached_tokens": 0,
        "metadata": {},
    }
    msg.update(overrides)
    return msg


def test_detokenizer_streaming_and_final_contracts():
    dm = _make_detokenizer()

    responses = dm.handle_message(
        {
            "type": "batch_tokens",
            "outputs": [{"request_id": "r1", "token_id": 65, "stream": True}],
        }
    )
    assert responses == [
        {"type": "token", "request_id": "r1", "token_id": 65, "text": "A"}
    ]
    assert (
        dm.handle_message(
            {
                "type": "batch_tokens",
                "outputs": [{"request_id": "r2", "token_id": 66, "stream": False}],
            }
        )
        == []
    )

    dm.handle_message(
        {
            "type": "batch_tokens",
            "outputs": [{"request_id": "r1", "token_id": 66, "stream": True}],
        }
    )
    assert "r1" in dm._states

    final = dm.handle_message(
        _finish_message(generated_ids=[65, 66], prompt_ids=[80, 81])
    )
    result = final[0]["result"]
    assert final[0]["type"] == "final"
    assert final[0]["ok"] is True
    assert result["text"] == "AB"
    assert result["token_texts"] == ["A", "B"]
    assert result["completion_ids"] == [65, 66]
    assert result["prompt_ids"] == [80, 81]
    assert result["completion_tokens"] == 2
    assert result["prompt_tokens"] == 2
    assert "r1" not in dm._states


def test_detokenizer_finish_stop_trim_and_logprob_text_contracts():
    dm = _make_detokenizer()

    trimmed = dm.handle_message(
        _finish_message(
            request_id="trimmed",
            finish_reason="stop",
            stop_strs=["B"],
            no_stop_trim=False,
        )
    )
    untrimmed = dm.handle_message(
        _finish_message(
            request_id="untrimmed",
            finish_reason="stop",
            stop_strs=["B"],
            no_stop_trim=True,
        )
    )
    assert trimmed[0]["result"]["text"] == "A"
    assert untrimmed[0]["result"]["text"] == "ABC"

    with_logprobs = dm.handle_message(
        _finish_message(
            request_id="logprobs",
            generated_ids=[65],
            logprob_data={
                "return_text_in_logprobs": True,
                "top_logprobs_num": 2,
                "logprob_start_len": -1,
                "token_logprobs": [(-0.5, 65, None)],
                "top_logprobs": [[(-0.5, 65, None), (-1.0, 66, None)]],
                "input_token_logprobs": None,
                "input_top_logprobs": None,
            },
        )
    )
    result = with_logprobs[0]["result"]
    assert result["token_logprobs"][0][2] == "A"
    assert result["top_logprobs"][0][0][2] == "A"
    assert result["top_logprobs"][0][1][2] == "B"


def test_detokenizer_passthrough_and_abort_cleanup():
    dm = _make_detokenizer()
    messages = [
        {"control_id": "abc", "ok": True, "metrics": {"total": 5}},
        {"type": "final", "request_id": "r0", "ok": False, "error": "boom"},
    ]
    for msg in messages:
        assert dm.handle_message(msg) == [msg]

    dm.handle_message(
        {
            "type": "batch_tokens",
            "outputs": [{"request_id": "r1", "token_id": 65, "stream": True}],
        }
    )
    assert "r1" in dm._states

    dm.handle_message(
        {
            "type": "final",
            "request_id": "r1",
            "ok": False,
            "error": "aborted",
        }
    )
    assert "r1" not in dm._states
