import time
import uuid

import orjson
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from nkipy_serving.conversation import (
    generate_chat_conv,
    generate_deepseek_v4_chat_conv,
)
from nkipy_serving.entrypoints.openai.gpt_oss_utils import (
    apply_harmony_stop,
    is_gpt_oss_model,
)
from nkipy_serving.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    TopLogprob,
    UsageInfo,
)
from nkipy_serving.entrypoints.openai.serving_base import OpenAIServingBase
from nkipy_serving.managers.io_struct import GenerateReqInput
from nkipy_serving.parser.reasoning_parser import GptOssReasoningParser


def _build_chat_logprobs(out: dict) -> ChoiceLogprobs | None:
    """Build OpenAI chat-format logprobs from scheduler result, or None."""
    raw_token_logprobs = out.get("token_logprobs")
    if not raw_token_logprobs:
        return None
    raw_top_logprobs = out.get("top_logprobs")
    content: list[ChatCompletionTokenLogprob] = []
    for i, entry in enumerate(raw_token_logprobs):
        logprob, _token_id, token_text = entry
        top: list[TopLogprob] = []
        if (
            raw_top_logprobs
            and i < len(raw_top_logprobs)
            and raw_top_logprobs[i] is not None
        ):
            for top_entry in raw_top_logprobs[i]:
                top.append(TopLogprob(token=top_entry[2], logprob=float(top_entry[0])))
        content.append(
            ChatCompletionTokenLogprob(
                token=token_text, logprob=float(logprob), top_logprobs=top
            )
        )
    return ChoiceLogprobs(content=content)


def _make_chat_chunk(model: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class OpenAIServingChat(OpenAIServingBase):
    async def handle_request(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        served_model = getattr(
            self.tokenizer_manager, "served_model_name", request.model
        )

        # GPT-OSS: use the checkpoint's Harmony chat_template.jinja (via transformers)
        # so we can return both `content` (final) and `reasoning_content` (analysis).
        if "DeepSeek-V4" in served_model:
            chat_template_kwargs = request.chat_template_kwargs or {}
            if messages and messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": ""})
            if request.task is not None:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        msg["task"] = request.task
                        break
                else:
                    raise ValueError(
                        "DeepSeek-V4 task requires at least one user message"
                    )
            effort_source = chat_template_kwargs.get(
                "reasoning_effort", request.reasoning_effort
            )
            dsv4_reasoning_effort = (
                effort_source if effort_source in {"max", "high"} else None
            )
            prompt = generate_deepseek_v4_chat_conv(
                messages,
                thinking=bool(
                    chat_template_kwargs.get("thinking")
                    or chat_template_kwargs.get("enable_thinking")
                ),
                reasoning_effort=dsv4_reasoning_effort,
            )
        elif is_gpt_oss_model(served_model):
            tokenizer = getattr(self.tokenizer_manager, "tokenizer", None)
            hf_tok = (
                getattr(tokenizer, "tokenizer", None) if tokenizer is not None else None
            )
            if hf_tok is None or not hasattr(hf_tok, "apply_chat_template"):
                prompt = generate_chat_conv(messages, tokenizer=hf_tok)
            else:
                prompt = hf_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=request.reasoning_effort,
                    **(request.chat_template_kwargs or {}),
                )
        else:
            tokenizer = getattr(self.tokenizer_manager, "tokenizer", None)
            hf_tok = (
                getattr(tokenizer, "tokenizer", None) if tokenizer is not None else None
            )
            prompt = generate_chat_conv(messages, tokenizer=hf_tok)

        # For chat, logprobs is a bool; top_logprobs is the requested top-k count.
        logprobs_k = int(request.top_logprobs or 0) if request.logprobs else 0

        stop = request.stop
        if is_gpt_oss_model(served_model):
            stop = apply_harmony_stop(stop)

        requested_max_tokens = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
        if requested_max_tokens is None:
            runtime_config = getattr(self.tokenizer_manager, "runtime_config", None)
            requested_max_tokens = int(
                getattr(runtime_config, "max_context_len", 0) or 4096
            )
        max_new_tokens = int(requested_max_tokens)

        req = GenerateReqInput(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            repetition_penalty=request.repetition_penalty,
            stream=request.stream,
            stop=stop,
            stop_token_ids=request.stop_token_ids,
            return_logprob=bool(request.logprobs),
            top_logprobs_num=logprobs_k,
            seed=request.seed,
            no_stop_trim=request.no_stop_trim,
            ignore_eos=request.ignore_eos,
        )

        if request.stream:
            gpt_oss_parser = (
                GptOssReasoningParser()
                if is_gpt_oss_model(served_model) and request.separate_reasoning
                else None
            )

            async def _stream():
                first = True
                async for out in self.tokenizer_manager.generate_stream(req):
                    if first:
                        first = False
                        # First chunk: role only
                        chunk = _make_chat_chunk(
                            request.model,
                            delta={"role": "assistant"},
                            finish_reason=None,
                        )
                        yield b"data: " + orjson.dumps(chunk) + b"\n\n"

                    if out["finish_reason"] is None:
                        # Middle chunks: content delta only
                        text = out["text"]
                        if text:
                            if gpt_oss_parser is not None:
                                parsed = gpt_oss_parser.parse_stream_chunk(str(text))
                                delta: dict = {}
                                if parsed.reasoning_text:
                                    delta["reasoning_content"] = parsed.reasoning_text
                                if parsed.normal_text:
                                    delta["content"] = parsed.normal_text
                                if delta:
                                    chunk = _make_chat_chunk(
                                        request.model, delta=delta, finish_reason=None
                                    )
                                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
                            else:
                                chunk = _make_chat_chunk(
                                    request.model,
                                    delta={"content": text},
                                    finish_reason=None,
                                )
                                yield b"data: " + orjson.dumps(chunk) + b"\n\n"
                    else:
                        # Final chunk: empty delta + finish_reason + optional usage
                        chunk = _make_chat_chunk(
                            request.model, delta={}, finish_reason=out["finish_reason"]
                        )
                        pt = int(out.get("prompt_tokens", 0))
                        ct = int(out.get("completion_tokens", 0))
                        if pt or ct:
                            chunk["usage"] = {
                                "prompt_tokens": pt,
                                "completion_tokens": ct,
                                "total_tokens": pt + ct,
                            }
                        yield b"data: " + orjson.dumps(chunk) + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        try:
            out = await self.tokenizer_manager.generate_once(req)
        except Exception as exc:
            return self._create_error_response(str(exc), code=500)

        reasoning_text = None
        if is_gpt_oss_model(served_model) and request.separate_reasoning:
            parsed = GptOssReasoningParser().parse_non_stream(str(out.get("text", "")))
            out["text"] = parsed.normal_text
            reasoning_text = parsed.reasoning_text or None

        prompt_tokens = int(out.get("prompt_tokens", 0))
        completion_tokens = int(out.get("completion_tokens", 0))
        resp = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=out["text"],
                        reasoning_content=reasoning_text,
                    ),
                    finish_reason=out["finish_reason"],
                    logprobs=_build_chat_logprobs(out),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return JSONResponse(content=resp.model_dump())
