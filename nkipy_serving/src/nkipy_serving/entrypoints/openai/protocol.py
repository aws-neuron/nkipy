import time
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nkipy-serving"
    root: str | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: str | None = None
    code: int


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenizeRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    tokens: list[int] | list[list[int]]
    count: int | list[int]
    max_model_len: int


class DetokenizeRequest(BaseModel):
    model: str | None = None
    tokens: list[int] | list[list[int]]
    skip_special_tokens: bool = True


class DetokenizeResponse(BaseModel):
    text: str | list[str]


class TokenizerInfoResponse(BaseModel):
    tokenizer_class: str
    model_id: str
    vocab_size: int
    max_model_len: int


# ---------------------------------------------------------------------------
# Logprobs models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class TopLogprob(BaseModel):
    token: str
    logprob: float


class LogProbs(BaseModel):
    """Logprobs payload for /v1/completions."""

    tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)
    text_offset: list[int] = Field(default_factory=list)


class ChatCompletionTokenLogprob(BaseModel):
    """Single token logprob entry for /v1/chat/completions."""

    token: str
    logprob: float
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    """Logprobs wrapper for chat completion choices."""

    content: list[ChatCompletionTokenLogprob] | None = None


# ---------------------------------------------------------------------------
# Completion models
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    logprobs: int | None = None
    seed: int | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length", "abort"] | None = None
    logprobs: LogProbs | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Chat completion models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # OpenAI compatibility (subset):
    # - max_tokens: chat-completions output cap
    # - max_completion_tokens: output cap including reasoning + visible output
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = "medium"
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    seed: int | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    task: (
        Literal["action", "query", "authority", "domain", "title", "read_url"] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def normalize_reasoning_inputs(cls, values: Any):
        # OpenAI clients sometimes send reasoning params as:
        #   {"reasoning": {"effort": "high"}}
        # Mirror sglang's behavior by mapping it into reasoning_effort.
        if not isinstance(values, dict):
            return values
        r = values.get("reasoning")
        if isinstance(r, dict):
            effort = r.get("effort") or r.get("reasoning_effort")
            if effort in {"none", "low", "medium", "high", "max"}:
                values["reasoning_effort"] = effort
            enabled = (
                r.get("enabled")
                if r.get("enabled") is not None
                else r.get("enable", False)
            )
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
            if enabled:
                ctk = values.get("chat_template_kwargs")
                if not isinstance(ctk, dict):
                    ctk = {}
                ctk.setdefault("thinking", True)
                ctk.setdefault("enable_thinking", True)
                values["chat_template_kwargs"] = ctk
        if values.get("reasoning_effort") == "none":
            ctk = values.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
            ctk.setdefault("thinking", False)
            ctk.setdefault("enable_thinking", False)
            values["chat_template_kwargs"] = ctk
        return values


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "abort"] | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
