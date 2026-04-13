# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKIPy-extended vLLM server with P2P weight transfer endpoints.

Reuses vLLM's OpenAI-compatible server and adds NKIPy endpoints
for sleep/wake_up/P2P weight transfer with zero changes to vLLM.

Usage:
    python -m nkipy.vllm_plugin.server --model Qwen/Qwen3-30B-A3B \
        --tensor-parallel-size 8 --dtype bfloat16 ...

All standard vLLM CLI args are supported. Custom endpoints are mounted
alongside the standard /v1/* routes on the same FastAPI app.
"""

import asyncio
import logging
import os
from argparse import Namespace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Tracks sleep state in the API server process.
_nkipy_sleeping = False
_tok_embedding_cache = None  # cached raw bytes + headers for /tok_embedding


# --- Request schemas ---

class WakeUpRequest(BaseModel):
    peer_url: str | None = None


class PushWeightsRequest(BaseModel):
    per_rank: list[dict]
    chunk_start: int | None = None
    chunk_end: int | None = None
    is_last_chunk: bool = True


def _get_engine_core(app: FastAPI):
    """Get the engine core client from the app state."""
    return app.state.engine_client.engine_core


def register_nkipy_routes(app: FastAPI) -> None:
    """Add NKIPy P2P endpoints to the vLLM FastAPI app."""

    @app.get("/nkipy/health")
    async def nkipy_health():
        core = _get_engine_core(app)
        results = await core.collective_rpc_async("nkipy_health")
        return JSONResponse(results[0])

    @app.post("/nkipy/sleep")
    async def sleep():
        import time as _time
        global _nkipy_sleeping, _tok_embedding_cache
        t0 = _time.time()
        _nkipy_sleeping = True
        _tok_embedding_cache = None
        core = _get_engine_core(app)
        results = await core.collective_rpc_async("nkipy_sleep")
        result = results[0]
        result["server_total_s"] = round(_time.time() - t0, 4)
        logger.info("sleep server total: %.4fs", _time.time() - t0)
        return result

    @app.post("/nkipy/wake_up")
    async def wake_up(request: Request, req: WakeUpRequest = None):
        import time as _time
        client = f"{request.client.host}:{request.client.port}" if request.client else "unknown"
        logger.info('%s - "POST /nkipy/wake_up HTTP/1.1"', client)
        global _nkipy_sleeping
        t0 = _time.time()
        peer_url = req.peer_url if req else None
        core = _get_engine_core(app)
        results = await core.collective_rpc_async(
            "nkipy_wake_up", args=(peer_url,),
        )
        _nkipy_sleeping = False
        result = results[0]
        result["server_total_s"] = round(_time.time() - t0, 4)
        logger.info("wake_up server total: %.4fs", _time.time() - t0)
        return result

    @app.post("/nkipy/p2p_push_weights")
    async def p2p_push_weights(req: PushWeightsRequest):
        core = _get_engine_core(app)
        results = await core.collective_rpc_async(
            "nkipy_push_weights",
            args=(req.per_rank, req.chunk_start, req.chunk_end, req.is_last_chunk),
        )
        return results[0]

    @app.get("/nkipy/tok_embedding")
    async def tok_embedding():
        global _tok_embedding_cache
        if _tok_embedding_cache is None:
            # Populate cache on first request
            core = _get_engine_core(app)
            _tok_embedding_cache = (await core.collective_rpc_async("nkipy_get_tok_embedding"))[0]
        data = _tok_embedding_cache
        if data is None:
            raise HTTPException(503, "Token embedding not available")
        return Response(
            content=data["raw"],
            media_type="application/octet-stream",
            headers={
                "X-Shape": ",".join(str(d) for d in data["shape"]),
                "X-Dtype": data["dtype"],
            },
        )


async def run_server(args: Namespace) -> None:
    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        init_app_state,
    )

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)
        register_nkipy_routes(app)
        await init_app_state(engine_client, app.state, args)

        # Reject /v1/* requests while sleeping.
        global _nkipy_sleeping, _tok_embedding_cache
        if not os.environ.get("NKIPY_CHECKPOINT"):
            _nkipy_sleeping = True
        else:
            # Pre-cache tok_embedding so /nkipy/tok_embedding serves instantly
            core = _get_engine_core(app)
            _tok_embedding_cache = (await core.collective_rpc_async("nkipy_get_tok_embedding"))[0]
            logger.info("tok_embedding cached (%s)",
                        f"{len(_tok_embedding_cache['raw'])/1e6:.1f} MB" if _tok_embedding_cache else "None")

        # Wrap app with ASGI middleware to reject requests while sleeping.
        inner_app = app

        async def sleeping_guard(scope, receive, send):
            if (scope["type"] == "http"
                    and _nkipy_sleeping
                    and scope["path"].startswith("/v1/")):
                import json
                body = json.dumps({"error": "Engine is sleeping. "
                                   "Send POST /nkipy/wake_up to activate."}).encode()
                await send({"type": "http.response.start", "status": 503,
                            "headers": [[b"content-type", b"application/json"]]})
                await send({"type": "http.response.body", "body": body})
                return
            await inner_app(scope, receive, send)

        import uvicorn
        config = uvicorn.Config(
            app=sleeping_guard,
            host=args.host or "0.0.0.0",
            port=args.port or 8000,
            log_level=args.uvicorn_log_level or "info",
        )
        await uvicorn.Server(config).serve()


def _release_neuron_cores():
    """Best-effort release of Neuron cores via spike.reset() + nrt_close."""
    try:
        from spike import reset as spike_reset
        spike_reset()
    except Exception:
        pass
    # Belt-and-suspenders: call nrt_close directly in case spike singleton
    # was never created but NRT was initialised by another path.
    try:
        from spike._spike import nrt_close  # noqa: F401
        nrt_close()
    except Exception:
        pass


def main():
    import signal

    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    def _shutdown_handler(signum, frame):
        logger.info("Received signal %s, releasing Neuron cores...", signum)
        _release_neuron_cores()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    try:
        asyncio.run(run_server(args))
    finally:
        _release_neuron_cores()


if __name__ == "__main__":
    main()
