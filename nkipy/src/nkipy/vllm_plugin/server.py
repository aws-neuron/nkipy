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
_nkipy_transitioning = False  # guards against concurrent wake/sleep
_tok_embedding_cache = None  # cached raw bytes + headers for /tok_embedding


# --- Request schemas ---

class WakeUpRequest(BaseModel):
    peer_url: str | None = None



def _get_engine_core(app: FastAPI):
    """Get the engine core client from the app state."""
    return app.state.engine_client.engine_core


def register_nkipy_routes(app: FastAPI) -> None:
    """Add NKIPy P2P endpoints to the vLLM FastAPI app."""

    @app.get("/nkipy/health")
    async def nkipy_health():
        if _nkipy_transitioning:
            return JSONResponse({"status": "transitioning", "backend": "nkipy",
                                 "sleeping": _nkipy_sleeping})
        try:
            core = _get_engine_core(app)
            results = await asyncio.wait_for(
                core.collective_rpc_async("nkipy_health"), timeout=5.0
            )
            return JSONResponse(results[0])
        except (asyncio.TimeoutError, Exception) as e:
            return JSONResponse({"status": "degraded", "backend": "nkipy",
                                 "sleeping": _nkipy_sleeping,
                                 "error": str(e)[:200]})

    @app.post("/nkipy/sleep")
    async def sleep():
        import time as _time
        global _nkipy_sleeping, _nkipy_transitioning, _tok_embedding_cache
        if _nkipy_sleeping:
            return JSONResponse({"status": "already_sleeping"})
        if _nkipy_transitioning:
            raise HTTPException(409, "Engine is transitioning (wake/sleep in progress)")
        _nkipy_transitioning = True
        t0 = _time.time()
        _nkipy_sleeping = True
        _tok_embedding_cache = None
        try:
            core = _get_engine_core(app)
            results = await core.collective_rpc_async("nkipy_sleep")
            result = results[0]
            result["server_total_s"] = round(_time.time() - t0, 4)
            logger.info("sleep server total: %.4fs", _time.time() - t0)
            return result
        except Exception as e:
            logger.error("sleep failed: %s", e)
            _nkipy_sleeping = False
            return JSONResponse({"status": "error", "error": f"Sleep failed: {e}"},
                                status_code=500)
        finally:
            _nkipy_transitioning = False

    @app.post("/nkipy/wake_up")
    async def wake_up(request: Request, req: WakeUpRequest = None):
        import time as _time
        client = f"{request.client.host}:{request.client.port}" if request.client else "unknown"
        logger.info('%s - "POST /nkipy/wake_up HTTP/1.1"', client)
        global _nkipy_sleeping, _nkipy_transitioning
        if not _nkipy_sleeping:
            return JSONResponse({"status": "already_awake"})
        if _nkipy_transitioning:
            raise HTTPException(409, "Engine is transitioning (wake/sleep in progress)")
        _nkipy_transitioning = True
        t0 = _time.time()
        peer_url = req.peer_url if req else None
        try:
            core = _get_engine_core(app)
            results = await core.collective_rpc_async(
                "nkipy_wake_up", args=(peer_url,),
            )
            result = results[0]
            result["server_total_s"] = round(_time.time() - t0, 4)
            if result.get("status") == "error":
                err_msg = result.get("error", "unknown error")
                logger.error("wake_up failed: %s", err_msg)
                return JSONResponse(result, status_code=503)
            _nkipy_sleeping = False
            logger.info("wake_up server total: %.4fs", _time.time() - t0)
            return result
        except Exception as e:
            logger.error("wake_up RPC failed: %s", e)
            return JSONResponse({"status": "error", "error": f"Wake-up failed: {e}"},
                                status_code=500)
        finally:
            _nkipy_transitioning = False

    @app.post("/nkipy/push")
    async def push(request: Request):
        body = await request.json()
        core = _get_engine_core(app)
        results = await core.collective_rpc_async(
            "nkipy_push", args=(body["per_rank"],),
        )
        return JSONResponse(results[0])

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
            # Pre-cache tok_embedding so /nkipy/tok_embedding serves instantly.
            # With hidden-dim TP sharding, each rank stores
            # vocab_size * (hidden_size / tp) — check the per-shard size.
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(args.model)
            tp = getattr(args, "tensor_parallel_size", 1) or 1
            emb_bytes = hf_cfg.vocab_size * (hf_cfg.hidden_size // tp) * 2  # bf16
            if emb_bytes > 0:
                logger.info("tok_embedding shard too large for pre-cache (%.0f MB), will serve lazily",
                            emb_bytes / 1e6)
            else:
                core = _get_engine_core(app)
                try:
                    _tok_embedding_cache = (await core.collective_rpc_async(
                        "nkipy_get_tok_embedding", timeout=30))[0]
                    logger.info("tok_embedding cached (%s)",
                                f"{len(_tok_embedding_cache['raw'])/1e6:.1f} MB" if _tok_embedding_cache else "None")
                except Exception as e:
                    logger.warning("tok_embedding pre-cache failed: %s", e)
                    _tok_embedding_cache = None

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
    """Best-effort release of Neuron cores and RDMA resources."""
    from .cleanup_utils import release_neuron_cores_and_rdma
    release_neuron_cores_and_rdma()


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
