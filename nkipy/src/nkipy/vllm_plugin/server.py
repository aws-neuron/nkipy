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
import gc
import logging
import time
from argparse import Namespace

import numpy as np
import torch
import torch.distributed as dist
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _fetch_tok_embedding_nkipy(peer_url: str):
    """Like fetch_tok_embedding but uses /nkipy/tok_embedding path."""
    import requests as _req
    if dist.get_rank() == 0:
        base = peer_url.rstrip("/")
        resp = _req.get(f"{base}/nkipy/tok_embedding")
        resp.raise_for_status()
        shape = tuple(int(d) for d in resp.headers["X-Shape"].split(","))
        dtype_str = resp.headers["X-Dtype"]
        torch_dtype = getattr(torch, dtype_str.replace("torch.", ""), None)
        if torch_dtype is not None:
            tok_embedding = (
                torch.frombuffer(bytearray(resp.content), dtype=torch_dtype)
                .reshape(shape).clone()
            )
        else:
            tok_embedding = torch.from_numpy(
                np.frombuffer(resp.content, dtype=np.dtype(dtype_str))
                .reshape(shape).copy()
            )
    else:
        tok_embedding = None
    obj_list = [tok_embedding]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


# --- Request schemas ---

class WakeUpRequest(BaseModel):
    peer_url: str | None = None


class PushWeightsRequest(BaseModel):
    per_rank: list[dict]
    chunk_start: int | None = None
    chunk_end: int | None = None
    is_last_chunk: bool = True


# --- P2P state (populated after model load) ---

class _P2PState:
    model = None          # NKIPy model instance
    weight_server = None  # WeightServer for /nkipy/weight_info
    sleeping = False
    kernel_cache = None   # {name: (neff_path, cache_key)}
    lock = None           # asyncio.Lock
    _model_runner = None  # NKIPyModelRunner ref (to update model after wake)

_p2p = _P2PState()

_KERNEL_NAMES = (
    "kernel_cte", "kernel_tkg",
    "kernel_cte_greedy_sampling", "kernel_tkg_greedy_sampling",
)


def _cache_kernels_and_release(model):
    """Save kernel NEFF paths, free device resources, release cores."""
    from spike import get_spike_singleton, reset as spike_reset
    from nkipy.runtime.device_kernel import _LOADED_KERNELS

    _p2p.kernel_cache = {}
    for name in _KERNEL_NAMES:
        kernel = getattr(model, name, None)
        if kernel is not None:
            _p2p.kernel_cache[name] = (kernel.neff_path, kernel.cache_key)

    spike = get_spike_singleton()
    for layer in model.layer_tensors:
        for t in layer.values():
            spike.free_tensor(t.tensor_ref)
    for t in (model.norm_weight, model.lm_head_weight):
        if t is not None:
            spike.free_tensor(t.tensor_ref)
    for name in _KERNEL_NAMES:
        kernel = getattr(model, name, None)
        if kernel is not None:
            spike.unload_model(kernel.model_ref)

    _LOADED_KERNELS.clear()
    gc.collect()
    spike_reset()


def register_nkipy_routes(app: FastAPI) -> None:
    """Add NKIPy P2P endpoints to the vLLM FastAPI app."""

    @app.get("/nkipy/health")
    async def nkipy_health():
        return JSONResponse({
            "status": "ok",
            "backend": "nkipy",
            "sleeping": _p2p.sleeping,
        })

    @app.post("/nkipy/sleep")
    async def sleep():
        if _p2p.lock is None:
            _p2p.lock = asyncio.Lock()
        async with _p2p.lock:
            if _p2p.sleeping:
                return {"status": "already_sleeping"}

            loop = asyncio.get_event_loop()
            def _sleep():
                from nkipy.p2p import rank_endpoint
                rank_endpoint.dereg_async()
                _cache_kernels_and_release(_p2p.model)
                _p2p.model = None
                _p2p.sleeping = True
                if _p2p.weight_server is not None:
                    _p2p.weight_server.cleanup()
                    _p2p.weight_server = None
            await loop.run_in_executor(None, _sleep)
            return {"status": "sleeping"}

    @app.post("/nkipy/wake_up")
    async def wake_up(req: WakeUpRequest = None):
        if _p2p.lock is None:
            _p2p.lock = asyncio.Lock()
        async with _p2p.lock:
            if not _p2p.sleeping:
                return {"status": "already_awake"}

            peer_url = req.peer_url if req else None
            loop = asyncio.get_event_loop()

            def _wake_up():
                from spike import get_spike_singleton
                from nkipy.runtime import DeviceKernel
                from nkipy.p2p import (
                    WeightServer, preregister_weights,
                )

                # Broadcast peer_url to all workers
                obj_list = [peer_url]
                dist.broadcast_object_list(obj_list, src=0)
                actual_peer = obj_list[0]

                dist.barrier()
                get_spike_singleton()
                dist.barrier()

                # Rebuild model with empty tensors
                # (model class stored on _p2p by the model runner)
                model = _p2p._model_class(
                    config=_p2p._config, skip_kernels=True
                )
                model._allocate_empty_tensors()

                if actual_peer:
                    from nkipy.p2p import receive_from_peer, rank_endpoint as _rank_ep, collect_weight_buffers
                    bufs = collect_weight_buffers(model)
                    receive_from_peer(_rank_ep, bufs, actual_peer,
                                      push_endpoint="/nkipy/p2p_push_weights")

                # Reload kernels from cached NEFFs
                for name, (neff_path, cache_key) in _p2p.kernel_cache.items():
                    setattr(model, name, DeviceKernel.load_with_cache_key(
                        neff_path, cache_key, name=name
                    ))
                dist.barrier()

                if model.tok_embedding is None and actual_peer:
                    model.tok_embedding = _fetch_tok_embedding_nkipy(actual_peer)

                _p2p.model = model
                _p2p.sleeping = False
                preregister_weights(model)
                _p2p.weight_server = WeightServer(model)

                # Update model runner so execute_model uses the new model
                if _p2p._model_runner is not None:
                    _p2p._model_runner._nkipy_model = model
                    _p2p._model_runner.model = model

            await loop.run_in_executor(None, _wake_up)
            return {"status": "awake"}

    @app.get("/nkipy/weight_info")
    async def weight_info():
        if _p2p.weight_server is None:
            raise HTTPException(503, "Weight server not initialized")
        return _p2p.weight_server.get_weight_info()

    @app.post("/nkipy/p2p_push_weights")
    async def p2p_push_weights(req: PushWeightsRequest):
        if _p2p.model is None:
            raise HTTPException(503, "Model not loaded")

        loop = asyncio.get_event_loop()
        def _push():
            from nkipy.p2p import push_weights_to_peer
            per_rank_info = [
                (r["remote_metadata"], r["remote_descs"])
                for r in req.per_rank
            ]
            push_weights_to_peer(
                _p2p.model, per_rank_info,
                chunk_start=req.chunk_start,
                chunk_end=req.chunk_end,
                is_last_chunk=req.is_last_chunk,
            )
        await loop.run_in_executor(None, _push)
        return {"status": "done"}

    @app.get("/nkipy/tok_embedding")
    async def tok_embedding():
        if _p2p.model is None or _p2p.model.tok_embedding is None:
            raise HTTPException(503, "Token embedding not available")
        emb = _p2p.model.tok_embedding
        if isinstance(emb, torch.Tensor):
            t = emb.cpu().contiguous()
            raw = t.view(torch.uint8).numpy().tobytes()
            shape, dtype = t.shape, str(t.dtype)
        else:
            data = np.ascontiguousarray(emb)
            raw = data.tobytes()
            shape, dtype = data.shape, str(data.dtype)
        return Response(
            content=raw,
            media_type="application/octet-stream",
            headers={
                "X-Shape": ",".join(str(d) for d in shape),
                "X-Dtype": dtype,
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

        import uvicorn
        config = uvicorn.Config(
            app=app,
            host=args.host or "0.0.0.0",
            port=args.port or 8000,
            log_level=args.uvicorn_log_level or "info",
        )
        await uvicorn.Server(config).serve()


def main():
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
