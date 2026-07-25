import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from nkipy_serving.conversation import generate_chat_conv
from nkipy_serving.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
)
from nkipy_serving.entrypoints.openai.serving_chat import OpenAIServingChat


class _FakeHFTokenizer:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return self.prompt


class _FailingHFTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise RuntimeError("bad chat template")


@dataclass
class _FakeTokenizerContainer:
    tokenizer: _FakeHFTokenizer


class _FakeTokenizerManager:
    def __init__(
        self,
        *,
        served_model_name: str,
        hf_tokenizer: _FakeHFTokenizer,
        once_result: dict[str, Any] | None = None,
        stream_results: list[dict[str, Any]] | None = None,
    ):
        self.served_model_name = served_model_name
        self.tokenizer = _FakeTokenizerContainer(tokenizer=hf_tokenizer)
        self.once_result = once_result or {
            "text": "",
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        self.stream_results = stream_results or []
        self.runtime_config = SimpleNamespace(max_context_len=1234)
        self.last_req = None
        self.last_request_id = None
        self.aborted_request_ids: list[str] = []

    async def generate_once(self, req, request_id: str | None = None):
        self.last_req = req
        self.last_request_id = request_id
        return self.once_result

    async def generate_stream(
        self, req, request_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.last_req = req
        self.last_request_id = request_id
        for item in self.stream_results:
            yield item

    async def abort_request(self, request_id: str) -> None:
        self.aborted_request_ids.append(request_id)


class _NoTokenizerAccessManager:
    served_model_name = "deepseek-ai/DeepSeek-V4-Flash"

    def __init__(self):
        self.last_req = None

    @property
    def tokenizer(self):
        raise AssertionError("DeepSeek-V4 chat should not require HF tokenizer access")

    async def generate_once(self, req, request_id: str | None = None):
        self.last_req = req
        return {
            "text": " Paris.",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 2,
        }

    async def generate_stream(
        self, req, request_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.last_req = req
        yield {
            "text": " Paris.",
            "finish_reason": None,
        }
        yield {
            "text": "",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 2,
        }

    async def abort_request(self, request_id: str) -> None:
        pass


async def _read_streaming_response(response: StreamingResponse) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


def _parse_chat_sse_events(sse_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in sse_text.strip().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_generate_chat_conv_propagates_chat_template_errors() -> None:
    with pytest.raises(RuntimeError, match="bad chat template"):
        generate_chat_conv(
            [{"role": "user", "content": "Hello"}],
            tokenizer=_FailingHFTokenizer(),
        )


def test_chat_completion_uses_harmony_template_and_splits_reasoning() -> None:
    fake_hf = _FakeHFTokenizer(prompt="PROMPT_CHAT")
    manager = _FakeTokenizerManager(
        served_model_name="unsloth/gpt-oss-120b-BF16",
        hf_tokenizer=fake_hf,
        once_result={
            "text": (
                "<|start|>assistant<|channel|>analysis<|message|>Thinking<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Paris<|return|>"
            ),
            "finish_reason": "stop",
            "prompt_tokens": 9,
            "completion_tokens": 5,
        },
    )
    serving = OpenAIServingChat(manager)
    request = ChatCompletionRequest(
        model="unsloth/gpt-oss-120b-BF16",
        messages=[ChatMessage(role="user", content="The capital of France is")],
        reasoning={"effort": "high"},
        chat_template_kwargs={"force_role": "assistant"},
        max_tokens=16,
        temperature=0.0,
        stream=False,
    )

    response = asyncio.run(serving.handle_request(request, None))
    assert isinstance(response, JSONResponse)
    data = json.loads(response.body.decode("utf-8"))

    assert request.reasoning_effort == "high"
    assert fake_hf.calls[0]["reasoning_effort"] == "high"
    assert fake_hf.calls[0]["force_role"] == "assistant"
    assert manager.last_req.prompt == "PROMPT_CHAT"
    assert manager.last_req.stop == "<|return|>"
    assert data["choices"][0]["message"]["content"] == "Paris"
    assert data["choices"][0]["message"]["reasoning_content"] == "Thinking"


def test_deepseek_v4_chat_completion_uses_native_prompt_format() -> None:
    manager = _NoTokenizerAccessManager()
    serving = OpenAIServingChat(manager)
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4-Flash",
        messages=[ChatMessage(role="user", content="The capital of France is")],
        max_tokens=32,
        temperature=0.0,
        stream=False,
    )

    response = asyncio.run(serving.handle_request(request, None))
    assert isinstance(response, JSONResponse)
    data = json.loads(response.body.decode("utf-8"))

    assert manager.last_req.prompt == (
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
        "<\uff5cUser\uff5c>The capital of France is"
        "<\uff5cAssistant\uff5c></think>"
    )
    assert data["choices"][0]["message"]["content"] == " Paris."


def test_deepseek_v4_chat_completion_thinking_prompt_variants() -> None:
    cases = [
        (
            {"chat_template_kwargs": {"enable_thinking": True}},
            "medium",
            {"enable_thinking": True},
            "<\uff5cAssistant\uff5c><think>",
        ),
        (
            {"reasoning": {"enabled": True, "effort": "high"}},
            "high",
            {"thinking": True, "enable_thinking": True},
            "<\uff5cAssistant\uff5c><think>",
        ),
        (
            {"reasoning_effort": "none"},
            "none",
            {"thinking": False, "enable_thinking": False},
            "<\uff5cAssistant\uff5c></think>",
        ),
    ]

    for request_kwargs, reasoning_effort, template_kwargs, prompt_suffix in cases:
        fake_hf = _FakeHFTokenizer(prompt="WRONG_GENERIC_TEMPLATE")
        manager = _FakeTokenizerManager(
            served_model_name="deepseek-ai/DeepSeek-V4-Flash",
            hf_tokenizer=fake_hf,
        )
        serving = OpenAIServingChat(manager)
        request = ChatCompletionRequest(
            model="deepseek-ai/DeepSeek-V4-Flash",
            messages=[ChatMessage(role="user", content="Think first.")],
            max_tokens=8,
            temperature=0.0,
            stream=False,
            **request_kwargs,
        )

        response = asyncio.run(serving.handle_request(request, None))

        assert isinstance(response, JSONResponse)
        assert fake_hf.calls == []
        assert request.reasoning_effort == reasoning_effort
        assert request.chat_template_kwargs == template_kwargs
        assert manager.last_req.prompt.endswith(prompt_suffix)


def test_chat_completion_omitted_max_tokens_uses_context_budget() -> None:
    fake_hf = _FakeHFTokenizer(prompt="PROMPT")
    manager = _FakeTokenizerManager(
        served_model_name="deepseek-ai/DeepSeek-V4-Flash",
        hf_tokenizer=fake_hf,
    )
    serving = OpenAIServingChat(manager)
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4-Flash",
        messages=[ChatMessage(role="user", content="The capital of France is")],
        stream=False,
    )

    response = asyncio.run(serving.handle_request(request, None))
    assert isinstance(response, JSONResponse)
    assert manager.last_req.max_new_tokens == 1234
    assert manager.last_req.temperature == 1.0
    assert manager.last_req.top_p == 1.0
    assert manager.last_req.top_k == -1
    assert manager.last_req.min_p == 0.0


def test_chat_stream_emits_reasoning_before_content() -> None:
    fake_hf = _FakeHFTokenizer(prompt="PROMPT_STREAM")
    manager = _FakeTokenizerManager(
        served_model_name="unsloth/gpt-oss-120b-BF16",
        hf_tokenizer=fake_hf,
        stream_results=[
            {
                "text": "<|start|>assistant<|channel|>analysis<|message|>Think",
                "finish_reason": None,
            },
            {
                "text": "ing<|end|><|start|>assistant<|channel|>final<|message|>Par",
                "finish_reason": None,
            },
            {"text": "is", "finish_reason": None},
            {
                "text": "",
                "finish_reason": "stop",
                "prompt_tokens": 7,
                "completion_tokens": 3,
            },
        ],
    )
    serving = OpenAIServingChat(manager)
    request = ChatCompletionRequest(
        model="unsloth/gpt-oss-120b-BF16",
        messages=[ChatMessage(role="user", content="The capital of France is")],
        max_tokens=32,
        temperature=0.0,
        stream=True,
        separate_reasoning=True,
    )

    response = asyncio.run(serving.handle_request(request, None))
    assert isinstance(response, StreamingResponse)
    sse_text = asyncio.run(_read_streaming_response(response))
    assert "data: [DONE]" in sse_text
    events = _parse_chat_sse_events(sse_text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"

    reasoning_idx = None
    content_idx = None
    for i, event in enumerate(events):
        delta = event["choices"][0]["delta"]
        if reasoning_idx is None and "reasoning_content" in delta:
            reasoning_idx = i
        if content_idx is None and "content" in delta:
            content_idx = i
    assert reasoning_idx is not None
    assert content_idx is not None
    assert reasoning_idx < content_idx
    assert events[-1]["choices"][0]["finish_reason"] in ("stop", "length")
