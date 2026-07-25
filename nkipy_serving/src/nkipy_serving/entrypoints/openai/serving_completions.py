import time
import uuid

import orjson
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from nkipy_serving.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    LogProbs,
    UsageInfo,
)
from nkipy_serving.entrypoints.openai.serving_base import OpenAIServingBase
from nkipy_serving.managers.io_struct import GenerateReqInput


def _build_logprobs(out: dict) -> LogProbs | None:
    """Build OpenAI-format LogProbs from scheduler result, or None if absent."""
    raw_token_logprobs = out.get("token_logprobs")
    if not raw_token_logprobs:
        return None
    raw_top_logprobs = out.get("top_logprobs")
    lp = LogProbs()
    for entry in raw_token_logprobs:
        logprob, _token_id, token_text = entry
        lp.tokens.append(token_text)
        lp.token_logprobs.append(float(logprob))
        lp.text_offset.append(-1)
    if raw_top_logprobs:
        for entries in raw_top_logprobs:
            if entries is not None:
                lp.top_logprobs.append({e[2]: float(e[0]) for e in entries})
            else:
                lp.top_logprobs.append(None)
    return lp


class OpenAIServingCompletion(OpenAIServingBase):
    async def handle_request(self, request: CompletionRequest, raw_request: Request):
        req = GenerateReqInput(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            repetition_penalty=request.repetition_penalty,
            stream=request.stream,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            return_logprob=request.logprobs is not None,
            top_logprobs_num=int(request.logprobs or 0),
            seed=request.seed,
            no_stop_trim=request.no_stop_trim,
            ignore_eos=request.ignore_eos,
        )

        if request.stream:

            async def _stream():
                async for out in self.tokenizer_manager.generate_stream(req):
                    chunk = {
                        "id": f"cmpl-{uuid.uuid4().hex}",
                        "object": "text_completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "text": out["text"],
                                "finish_reason": out["finish_reason"],
                            }
                        ],
                    }
                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        try:
            out = await self.tokenizer_manager.generate_once(req)
        except Exception as exc:
            return self._create_error_response(str(exc), code=500)

        prompt_tokens = int(out.get("prompt_tokens", 0))
        completion_tokens = int(out.get("completion_tokens", 0))
        resp = CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            model=request.model,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text=out["text"],
                    finish_reason=out["finish_reason"],
                    logprobs=_build_logprobs(out),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return JSONResponse(content=resp.model_dump())
