import asyncio
import dataclasses
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from nkipy_serving.config import DSV4_ATTENTION_BACKEND, RuntimeConfig
from nkipy_serving.entrypoints.engine import _launch_subprocesses_or_threads
from nkipy_serving.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    TokenizeRequest,
    TokenizeResponse,
    TokenizerInfoResponse,
)
from nkipy_serving.entrypoints.openai.serving_chat import OpenAIServingChat
from nkipy_serving.entrypoints.openai.serving_completions import OpenAIServingCompletion
from nkipy_serving.managers.io_struct import GenerateReqInput
from nkipy_serving.managers.tokenizer_manager import TokenizerManager
from nkipy_serving.server_args import ServerArgs
from nkipy_serving.version import __version__

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager
    runtime_config: RuntimeConfig
    variant_count: int
    warmup_summary: dict[str, object] | None = None
    process_group: object | None = None


_global_state: _GlobalState | None = None
_init_task: asyncio.Task | None = None
_init_error: str | None = None
_init_lock = threading.Lock()


def set_global_state(global_state: _GlobalState | None):
    global _global_state
    _global_state = global_state


def _init_global_state(server_args: ServerArgs) -> None:
    init_result = _launch_subprocesses_or_threads(server_args)
    set_global_state(
        _GlobalState(
            tokenizer_manager=init_result.tokenizer_manager,
            runtime_config=init_result.runtime_config,
            variant_count=init_result.variant_count,
            warmup_summary=init_result.warmup_summary,
            process_group=init_result.process_group,
        )
    )


async def _ensure_global_state_init_started(server_args: ServerArgs) -> None:
    """Start blocking engine init in a background thread.

    This lets the HTTP server bind its port immediately; `/ready` becomes 200
    once init completes, while `/health` stays 200 as soon as the process is up.
    """
    global _init_task

    if _global_state is not None or _init_error is not None or _init_task is not None:
        return

    with _init_lock:
        if (
            _global_state is not None
            or _init_error is not None
            or _init_task is not None
        ):
            return

        async def _run_init():
            global _init_error
            try:
                await asyncio.to_thread(_init_global_state, server_args)
            except (
                Exception
            ) as exc:  # pragma: no cover - init failure is surfaced via /ready
                _init_error = repr(exc)
                logger.exception("HTTP server initialization failed")

        _init_task = asyncio.create_task(_run_init())


def _ready_guard(
    app_obj: FastAPI | None = None, *, openai: bool = False
) -> JSONResponse | None:
    """Return an error response if the server isn't ready.

    When *openai* is True, uses OpenAI-style error envelope and lazily creates
    serving helpers.  When False, returns a compact internal error.
    """
    if _init_error is not None:
        if openai:
            return JSONResponse(
                status_code=500,
                content={
                    "object": "error",
                    "message": f"Server initialization failed: {_init_error}",
                    "type": "server_error",
                    "code": 500,
                },
            )
        return JSONResponse(status_code=500, content={"error": _init_error})
    if _global_state is None:
        if openai:
            return JSONResponse(
                status_code=503,
                content={
                    "object": "error",
                    "message": "Server is warming up. Retry later or poll /ready.",
                    "type": "server_busy",
                    "code": 503,
                },
            )
        return JSONResponse(status_code=503, content={"error": "warming_up"})

    if (
        openai
        and app_obj is not None
        and getattr(app_obj.state, "openai_serving_completion", None) is None
    ):
        app_obj.state.openai_serving_completion = OpenAIServingCompletion(
            _global_state.tokenizer_manager
        )
        app_obj.state.openai_serving_chat = OpenAIServingChat(
            _global_state.tokenizer_manager
        )
    return None


def _runtime_config_payload(runtime_config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(runtime_config):
        return dataclasses.asdict(runtime_config)
    try:
        return dict(vars(runtime_config))
    except TypeError:
        return {}


def _dsv4_runtime_summary(runtime_config: Any) -> dict[str, Any] | None:
    cfg = _runtime_config_payload(runtime_config)
    model_id = str(cfg.get("model_id", ""))
    hf_model_id = str(cfg.get("hf_model_id", "") or "")
    is_dsv4 = (
        "DeepSeek-V4" in model_id
        or "deepseek-v4" in model_id.lower()
        or "DeepSeek-V4" in hf_model_id
        or "deepseek-v4" in hf_model_id.lower()
        or cfg.get("attention_backend") == DSV4_ATTENTION_BACKEND
    )
    if not is_dsv4:
        return None
    tp = int(cfg.get("tp_degree", 1))
    ep = int(cfg.get("ep_degree", 1))
    replica = int(cfg.get("replica_degree", 1))
    adp = int(cfg.get("attention_dp_degree", 1))
    return {
        "sharding": f"TP{tp}/EP{ep}/R{replica}/ADP{adp}",
        "tp_degree": tp,
        "ep_degree": ep,
        "replica_degree": replica,
        "attention_dp_degree": adp,
        "total_workers": int(cfg.get("total_workers", tp * ep * replica)),
        "attention_backend": cfg.get("attention_backend"),
        "paged_attn_impl": cfg.get("paged_attn_impl"),
        "target_only": bool(cfg.get("dsv4_disable_mtp", False)),
        "dp_attention_superstep": adp > 1,
        "request_buckets": list(cfg.get("request_buckets") or ()),
        "token_buckets": list(cfg.get("token_buckets") or ()),
        "kv_pool_size": int(cfg.get("kv_pool_size", 0)),
        "kv_cache_block_size": int(cfg.get("kv_cache_block_size", 0)),
        "max_context_len": int(cfg.get("max_context_len", 0)),
        "state_size": int(cfg.get("dsv4_state_size", 0)),
    }


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    server_args = getattr(fast_api_app.state, "server_args", None)
    if server_args is None:
        server_args = ServerArgs()

    # Start init in the background and return immediately so uvicorn can start
    # serving `/health` right away.
    await _ensure_global_state_init_started(server_args)

    try:
        yield
    finally:
        global _init_task, _init_error

        if _global_state is not None and _global_state.process_group is not None:
            _global_state.process_group.shutdown()
        set_global_state(None)
        fast_api_app.state.openai_serving_completion = None
        fast_api_app.state.openai_serving_chat = None
        _init_task = None
        _init_error = None


app = FastAPI(lifespan=lifespan)

# -- Middleware --

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "object": "error",
            "message": str(exc),
            "type": "invalid_request_error",
            "code": 400,
        },
    )


# -- Helper --


async def _validate_json_request(request: Request) -> dict[str, Any]:
    """Parse and return JSON body, raising 400 on failure."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _native_error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": str(message)},
    )


def _exception_error_message(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return message
    return f"{type(exc).__name__}: {exc!r}"


def _openai_error_response(
    message: str,
    *,
    status_code: int,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    err = ErrorResponse(
        message=str(message),
        type=error_type,
        code=int(status_code),
    )
    return JSONResponse(status_code=status_code, content=err.model_dump())


def _body_value(body: dict[str, Any], key: str, default: Any = None) -> Any:
    sampling_params = body.get("sampling_params")
    if isinstance(sampling_params, dict) and key in sampling_params:
        return sampling_params[key]
    return body.get(key, default)


def _normalize_text_batch(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if not value:
            raise ValueError("text/prompt batch must contain at least one item")
        return list(value)
    raise ValueError("text/prompt must be a string or list of strings")


def _normalize_input_ids_batch(value: Any) -> list[list[int]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("input_ids must be a list[int] or list[list[int]]")
    if not value:
        raise ValueError("input_ids must contain at least one token")
    if all(isinstance(item, int) for item in value):
        return [[int(item) for item in value]]
    if all(isinstance(item, list) for item in value):
        rows: list[list[int]] = []
        for row in value:
            if not row:
                raise ValueError("input_ids rows must contain at least one token")
            if not all(isinstance(token, int) for token in row):
                raise ValueError("input_ids rows must contain only integers")
            rows.append([int(token) for token in row])
        return rows
    raise ValueError("input_ids must be a list[int] or list[list[int]]")


def _expand_per_input_value(value: Any, batch_size: int, field_name: str) -> list[Any]:
    if isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError(f"{field_name} length must match batch size")
        return list(value)
    return [value for _ in range(batch_size)]


def _normalize_generate_items(
    body: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    text_value = body.get("text", body.get("prompt", body.get("prompts")))
    input_ids_value = body.get("input_ids")
    texts = _normalize_text_batch(text_value)
    input_ids_batch = _normalize_input_ids_batch(input_ids_value)
    if (texts is None) == (input_ids_batch is None):
        raise ValueError("Specify exactly one of text/prompt or input_ids")

    if texts is not None:
        base_inputs: list[dict[str, Any]] = [{"prompt": text} for text in texts]
    else:
        if input_ids_batch is None:
            raise RuntimeError("input_ids batch missing after input normalization")
        base_inputs = [{"input_ids": input_ids} for input_ids in input_ids_batch]

    batch_size = len(base_inputs)
    n = int(body.get("n", 1))
    if n <= 0:
        raise ValueError("n must be >= 1")

    metadata_list = _expand_per_input_value(
        body.get("metadata"), batch_size, "metadata"
    )
    request_ids_raw = body.get("request_ids", body.get("request_id"))
    if request_ids_raw is None:
        request_ids_list = [None for _ in range(batch_size)]
    elif isinstance(request_ids_raw, str):
        if batch_size != 1:
            raise ValueError("request_ids must be a list when batch size > 1")
        request_ids_list = [request_ids_raw]
    else:
        request_ids_list = _expand_per_input_value(
            request_ids_raw, batch_size, "request_ids"
        )

    max_new_tokens = int(
        _body_value(body, "max_new_tokens", _body_value(body, "max_tokens", 16))
    )
    stream = bool(body.get("stream", False))
    if stream and (batch_size != 1 or n != 1):
        raise ValueError("streaming is only supported for a single generate request")

    common_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": float(_body_value(body, "temperature", 1.0)),
        "top_p": float(_body_value(body, "top_p", 1.0)),
        "top_k": int(_body_value(body, "top_k", -1)),
        "min_p": float(_body_value(body, "min_p", 0.0)),
        "frequency_penalty": float(_body_value(body, "frequency_penalty", 0.0)),
        "presence_penalty": float(_body_value(body, "presence_penalty", 0.0)),
        "repetition_penalty": float(_body_value(body, "repetition_penalty", 1.0)),
        "stream": stream,
        "stop": _body_value(body, "stop"),
        "stop_token_ids": [
            int(token) for token in (_body_value(body, "stop_token_ids") or [])
        ]
        or None,
        "return_logprob": bool(body.get("return_logprob", False)),
        "logprob_start_len": int(body.get("logprob_start_len", -1)),
        "top_logprobs_num": int(body.get("top_logprobs_num", 0)),
        "return_text_in_logprobs": bool(body.get("return_text_in_logprobs", True)),
        "seed": int(body["seed"]) if body.get("seed") is not None else None,
        "no_stop_trim": bool(body.get("no_stop_trim", False)),
        "ignore_eos": bool(body.get("ignore_eos", False)),
    }

    items: list[dict[str, Any]] = []
    for index, base_input in enumerate(base_inputs):
        metadata = metadata_list[index]
        request_id = request_ids_list[index]
        for sample_index in range(n):
            sample_request_id = request_id
            if sample_request_id is not None and n > 1:
                sample_request_id = f"{sample_request_id}-{sample_index}"
            req = GenerateReqInput(
                metadata=metadata,
                **base_input,
                **common_kwargs,
            )
            items.append(
                {
                    "index": index,
                    "sample_index": sample_index,
                    "request_id": sample_request_id,
                    "req": req,
                }
            )
    return items, stream


def _format_generate_result(
    item: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "request_id": item["request_id"],
        "index": int(item["index"]),
        "sample_index": int(item["sample_index"]),
        **result,
    }


def _tokenizer_handle():
    global_state = _require_global_state()
    tok = getattr(global_state.tokenizer_manager, "tokenizer", None)
    if tok is None:
        raise RuntimeError("Tokenizer is unavailable")
    return tok


def _require_global_state() -> _GlobalState:
    if _global_state is None:
        raise RuntimeError("Server runtime is not initialized")
    return _global_state


async def _run_native_batch(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    global_state = _require_global_state()
    tasks = [
        asyncio.create_task(
            global_state.tokenizer_manager.generate_once(
                item["req"], request_id=item["request_id"]
            )
        )
        for item in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    first_error: BaseException | None = None
    final_results: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, BaseException):
            if first_error is None:
                first_error = result
            continue
        final_results.append(result)
    if first_error is not None:
        raise first_error
    return final_results


# -- Launch --


def launch(server_args: ServerArgs) -> None:
    app.state.server_args = server_args
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level_http or server_args.log_level,
        loop="uvloop",
        access_log=server_args.access_log,
        workers=server_args.workers,
    )


# =========================================================================
# Health & info routes
# =========================================================================


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.get("/ready")
async def ready() -> Response:
    """Readiness endpoint: 200 only after the runtime has finished warmup/init."""
    if _init_error is not None:
        return JSONResponse(
            status_code=500, content={"status": "error", "error": _init_error}
        )
    if _global_state is None:
        return JSONResponse(status_code=503, content={"status": "warming_up"})
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.get("/health_generate")
async def health_generate():
    """Probe health by running a minimal generate request."""
    try:
        if _global_state is None:
            return Response(status_code=503)
        req = GenerateReqInput(prompt="test", max_new_tokens=1, stream=False)
        await _global_state.tokenizer_manager.generate_once(req)
        return Response(status_code=200)
    except Exception:
        logger.debug("health_generate probe failed", exc_info=True)
        return Response(status_code=503)


@app.get("/get_server_info")
async def get_server_info():
    if (guard := _ready_guard()) is not None:
        return guard
    scheduler_metrics = None
    lane_metadata = None
    try:
        scheduler_metrics = (
            await _global_state.tokenizer_manager.get_scheduler_metrics()
        )
    except Exception as exc:  # pragma: no cover - info endpoint best effort
        scheduler_metrics = {"error": repr(exc)}
    get_lane_metadata = getattr(
        _global_state.tokenizer_manager, "get_lane_metadata", None
    )
    if callable(get_lane_metadata):
        try:
            lane_metadata = await get_lane_metadata()
        except Exception as exc:  # pragma: no cover - info endpoint best effort
            lane_metadata = {"error": repr(exc)}
    runtime_config = _runtime_config_payload(_global_state.runtime_config)
    return {
        "version": __version__,
        "served_model_name": _global_state.tokenizer_manager.served_model_name,
        "runtime_config": runtime_config,
        "dsv4_runtime": _dsv4_runtime_summary(_global_state.runtime_config),
        "variant_count": _global_state.variant_count,
        "warmup_summary": _global_state.warmup_summary,
        "scheduler_metrics": scheduler_metrics,
        "lane_metadata": lane_metadata,
    }


@app.get("/server_info")
async def server_info():
    return await get_server_info()


@app.get("/version")
async def show_version():
    return {"version": __version__}


@app.get("/get_model_info")
async def get_model_info():
    if (guard := _ready_guard()) is not None:
        return guard
    return {
        "model_id": _global_state.runtime_config.model_id,
        "is_generation": True,
    }


@app.get("/v4/lane_metadata")
async def v4_lane_metadata():
    """Per-rank DP-attention lane / group metadata (DeepSeek-V4 family).

    Returns `{"lane_metadata": {rank: {...}}, "lane_routes": [...],
    "tp_degree", "ep_degree", "replica_degree", "attention_dp_degree",
    "total_workers"}`. Returns an empty `lane_metadata` dict for models whose
    executor does not expose per-rank lane metadata (e.g. Qwen3, GPT-OSS).
    """
    if (guard := _ready_guard()) is not None:
        return guard
    return await _global_state.tokenizer_manager.get_lane_metadata()


@app.get("/tokenizer_info", response_class=JSONResponse)
async def tokenizer_info():
    if (guard := _ready_guard()) is not None:
        return guard
    tok = _tokenizer_handle()
    return JSONResponse(
        content=TokenizerInfoResponse(
            tokenizer_class=str(getattr(tok, "tokenizer_class", type(tok).__name__)),
            model_id=str(_global_state.runtime_config.tokenizer_model_id),
            vocab_size=int(tok.vocab_size),
            max_model_len=int(_global_state.runtime_config.max_context_len),
        ).model_dump()
    )


# =========================================================================
# Tokenization routes
# =========================================================================


@app.post("/v1/tokenize", response_class=JSONResponse)
async def tokenize(request: TokenizeRequest):
    if (guard := _ready_guard()) is not None:
        return guard
    tok = _tokenizer_handle()
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    tokens = [
        encoded.tolist()
        for encoded in tok.batch_encode(
            prompts, add_special_tokens=bool(request.add_special_tokens)
        )
    ]
    counts = [len(item) for item in tokens]
    response = TokenizeResponse(
        tokens=tokens[0] if len(tokens) == 1 else tokens,
        count=counts[0] if len(counts) == 1 else counts,
        max_model_len=int(_global_state.runtime_config.max_context_len),
    )
    return JSONResponse(content=response.model_dump())


@app.post("/v1/detokenize", response_class=JSONResponse)
async def detokenize(request: DetokenizeRequest):
    if (guard := _ready_guard()) is not None:
        return guard
    tok = _tokenizer_handle()
    batch_tokens = (
        request.tokens
        if request.tokens and isinstance(request.tokens[0], list)
        else [request.tokens]
    )
    texts = tok.batch_decode(
        batch_tokens,
        skip_special_tokens=bool(request.skip_special_tokens),
    )
    response = DetokenizeResponse(text=texts[0] if len(texts) == 1 else texts)
    return JSONResponse(content=response.model_dump())


# =========================================================================
# Native generate routes
# =========================================================================


@app.post("/generate")
async def generate(raw_request: Request):
    """Native generate endpoint."""
    if (guard := _ready_guard()) is not None:
        return guard
    try:
        body = await _validate_json_request(raw_request)
        items, stream = _normalize_generate_items(body)
    except ValueError as exc:
        return _native_error_response(str(exc))

    if stream:
        req = items[0]["req"]
        request_id = items[0]["request_id"]

        async def _stream():
            async for out in _global_state.tokenizer_manager.generate_stream(
                req,
                request_id=request_id,
            ):
                yield b"data: " + orjson.dumps(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    try:
        results = await _run_native_batch(items)
    except Exception as exc:
        return _native_error_response(_exception_error_message(exc))

    formatted = [
        _format_generate_result(item, result)
        for item, result in zip(items, results, strict=True)
    ]
    return JSONResponse(
        content={
            "results": formatted,
            "batch_size": len({int(item["index"]) for item in items}),
            "n": int(body.get("n", 1)),
        }
    )


# =========================================================================
# Control routes
# =========================================================================


@app.post("/abort_request")
async def abort_request(raw_request: Request):
    if (guard := _ready_guard()) is not None:
        return guard
    try:
        body = await _validate_json_request(raw_request)
    except ValueError as exc:
        return _native_error_response(str(exc))
    request_id = str(body.get("request_id", ""))
    if not request_id:
        return JSONResponse(
            status_code=400, content={"error": "request_id is required"}
        )
    await _global_state.tokenizer_manager.abort_request(request_id)
    return JSONResponse(content={"ok": True})


@app.post("/pause_generation")
async def pause_generation():
    if (guard := _ready_guard()) is not None:
        return guard
    await _global_state.tokenizer_manager.pause_generation()
    return JSONResponse(content={"ok": True})


@app.post("/continue_generation")
async def continue_generation():
    if (guard := _ready_guard()) is not None:
        return guard
    await _global_state.tokenizer_manager.continue_generation()
    return JSONResponse(content={"ok": True})


@app.post("/flush_cache")
async def flush_cache(raw_request: Request):
    if (guard := _ready_guard()) is not None:
        return guard
    try:
        body = await _validate_json_request(raw_request)
    except ValueError as exc:
        return _native_error_response(str(exc))
    result = await _global_state.tokenizer_manager.flush_cache(
        abort_all_requests=bool(body.get("abort_all_requests", True))
    )
    return JSONResponse(content={"ok": True, **result})


async def _reload_weights_from_disk_impl(raw_request: Request):
    if (guard := _ready_guard()) is not None:
        return guard
    try:
        body = await _validate_json_request(raw_request)
    except ValueError as exc:
        return _native_error_response(str(exc))
    model_path = str(body.get("model_path", "")).strip()
    if not model_path:
        return JSONResponse(
            status_code=400,
            content={"error": "model_path is required"},
        )
    try:
        result = await _global_state.tokenizer_manager.reload_weights_from_disk(
            model_path=model_path,
            abort_all_requests=bool(body.get("abort_all_requests", True)),
        )
    except Exception as exc:
        message = str(exc)
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "success": False,
                "error": message,
                "message": message,
            },
        )
    return JSONResponse(
        content={
            "ok": True,
            "success": True,
            "message": "weights reloaded",
            **result,
        }
    )


@app.post("/reload_weights_from_disk")
async def reload_weights_from_disk(raw_request: Request):
    return await _reload_weights_from_disk_impl(raw_request)


# =========================================================================
# OpenAI-compatible routes
# =========================================================================


@app.post("/v1/completions")
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    if (guard := _ready_guard(raw_request.app, openai=True)) is not None:
        return guard
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    if (guard := _ready_guard(raw_request.app, openai=True)) is not None:
        return guard
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.get("/v1/models", response_class=JSONResponse)
async def available_models(raw_request: Request):
    if (guard := _ready_guard(raw_request.app, openai=True)) is not None:
        return guard
    model_name = _global_state.tokenizer_manager.served_model_name
    return ModelList(data=[ModelCard(id=model_name, root=model_name)])


@app.get("/v1/models/{model:path}", response_class=JSONResponse)
async def get_model(model: str, raw_request: Request):
    """Single model lookup by name."""
    if (guard := _ready_guard(raw_request.app, openai=True)) is not None:
        return guard
    model_name = _global_state.tokenizer_manager.served_model_name
    if model != model_name:
        return JSONResponse(
            status_code=404,
            content={
                "object": "error",
                "message": f"Model not found: {model}",
                "type": "invalid_request_error",
                "code": 404,
            },
        )
    return ModelCard(id=model_name, root=model_name)


# =========================================================================
# Unsupported routes (501 Not Implemented)
# =========================================================================


def _register_unsupported_routes() -> None:
    """Register explicit 501 handlers for unsupported routes."""
    _NOT_IMPL = "not implemented."
    _NOT_GEN = "not supported by nkipy-serving generation models."
    unsupported_routes: dict[str, str] = {
        "/start_profile": _NOT_IMPL,
        "/stop_profile": _NOT_IMPL,
        "/start_trace": _NOT_IMPL,
        "/stop_trace": _NOT_IMPL,
        "/trace_status": _NOT_IMPL,
        "/release_memory_occupation": _NOT_IMPL,
        "/resume_memory_occupation": _NOT_IMPL,
        "/open_session": _NOT_IMPL,
        "/close_session": _NOT_IMPL,
        "/configure_logging": _NOT_IMPL,
        "/get_load": _NOT_IMPL,
        "/set_internal_state": _NOT_IMPL,
        "/v1/embeddings": _NOT_GEN,
        "/pooling": _NOT_GEN,
        "/classify": _NOT_GEN,
        "/v1/classify": _NOT_GEN,
        "/rerank": _NOT_GEN,
        "/v1/rerank": _NOT_GEN,
        "/v2/rerank": _NOT_GEN,
        "/v1/score": "Cross-encoder /v1/score is not supported.",
    }
    for path, msg in unsupported_routes.items():

        def _make(p: str = path, m: str = msg):
            async def _unsupported(request: Request):
                return _openai_error_response(
                    f"{m} Route: {p}",
                    status_code=501,
                    error_type="not_implemented_error",
                )

            return _unsupported

        app.add_api_route(path, _make(), methods=["GET", "POST", "PUT"])


_register_unsupported_routes()
