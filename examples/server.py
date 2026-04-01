"""
FastAPI server for model inference, with vLLM-style API endpoints.
Supports both Qwen3 and Llama3 architectures via --arch flag.

Usage:
    # With local checkpoint (Engine A):
    torchrun --nproc_per_node=8 server.py --arch qwen3 --model Qwen/Qwen3-30B-A3B --checkpoint /path/to/shards

    # Without checkpoint, P2P from peer (Engine B — starts sleeping, activate with /wake_up):
    torchrun --nproc_per_node=8 server.py --arch qwen3 --model Qwen/Qwen3-30B-A3B

Endpoints:
    POST /completions  - Generate text completions
    POST /sleep        - Offload model from device
    POST /wake_up      - Reload model onto device

Example curl:
    curl -X POST http://localhost:8000/completions \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "The capital of France is", "max_tokens": 32}'
"""

import argparse
import asyncio
import gc
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from p2p_weight_transfer import (
    WeightServer,
    fetch_tok_embedding,
    push_weights_to_peer,
    receive_weights,
)

# Add models/ to sys.path so common.* imports work
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
sys.path.insert(0, _models_dir)

from common.model import load_model, _warmup
from common.utils import print_log

# These will be set after parsing --arch
ModelClass = None  # Qwen3Model or Llama3Model
EOS_TOKEN_IDS = set()

# --- Command codes broadcast from rank 0 to all workers ---
CMD_GENERATE = 0
CMD_SLEEP = 1
CMD_WAKEUP = 2
CMD_P2P_PUSH_WEIGHTS = 3

# Kernel attribute names used for caching and lifecycle management
_KERNEL_NAMES = ("kernel_cte", "kernel_tkg", "kernel_cte_greedy_sampling", "kernel_tkg_greedy_sampling")


# --- Request / Response schemas (vLLM-compatible subset) ---

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = "Hello"
    max_tokens: int = Field(default=16, ge=1)
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class PushWeightsRequest(BaseModel):
    per_rank: list[dict]  # [{remote_metadata: str, remote_descs: str}, ...] per rank


# --- Global state ---

def setup_arch(arch: str):
    """Set global ModelClass and EOS_TOKEN_IDS for the chosen architecture."""
    global ModelClass, EOS_TOKEN_IDS

    arch_dir = os.path.join(_models_dir, arch)
    sys.path.insert(0, arch_dir)

    if arch == "qwen3":
        from qwen3 import Qwen3Model, EOS_TOKEN_IDS as eos
        ModelClass = Qwen3Model
        EOS_TOKEN_IDS = eos
    elif arch == "llama3":
        from llama3 import Llama3Model, EOS_TOKEN_IDS as eos
        ModelClass = Llama3Model
        EOS_TOKEN_IDS = eos
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.args = None
        self.weights = None
        self.config = None
        self.sleeping = False
        self.kernel_cache = None  # {name: (neff_path, cache_key)} saved during sleep
        self.weight_server = None  # WeightServer for /weight_info (rank 0 only)
        self._lock = asyncio.Lock()

    def is_ready(self):
        return self.model is not None and not self.sleeping


state = ModelState()


# --- Distributed helpers ---

def broadcast_cmd(cmd: int):
    t = torch.tensor([cmd], dtype=torch.int64)
    dist.broadcast(t, src=0)


def recv_cmd() -> int:
    t = torch.tensor([0], dtype=torch.int64)
    dist.broadcast(t, src=0)
    return t.item()


def broadcast_input_ids(input_ids_np=None):
    if dist.get_rank() == 0:
        t = torch.from_numpy(input_ids_np.astype(np.int64))
    else:
        t = torch.zeros(1, state.config.context_len, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return t.numpy()


def broadcast_max_tokens(max_tokens=None):
    if dist.get_rank() == 0:
        t = torch.tensor([max_tokens], dtype=torch.int64)
    else:
        t = torch.tensor([0], dtype=torch.int64)
    dist.broadcast(t, src=0)
    return t.item()


# --- Model helpers ---

def pad_input_ids(input_ids, context_len, pad_token_id=0):
    seq_len = input_ids.shape[1]
    if seq_len >= context_len:
        return input_ids[:, :context_len]
    pad_width = context_len - seq_len
    padding = np.full((input_ids.shape[0], pad_width), pad_token_id, dtype=input_ids.dtype)
    return np.concatenate([padding, input_ids], axis=1)


def run_generate(padded_ids, max_tokens):
    state.model.config.max_new_tokens = max_tokens
    tokens = []
    for tok_tensor in state.model.generate(padded_ids):
        tok_id = tok_tensor[0].tolist()[-1]
        if tok_id in EOS_TOKEN_IDS:
            break
        tokens.append(tok_id)
    return tokens


# --- Device resource lifecycle ---

def _cache_kernels_and_release(model):
    """Save kernel NEFF paths to state.kernel_cache, free all device resources, and release cores."""
    from spike import get_spike_singleton, reset as spike_reset
    from nkipy.runtime.device_kernel import _LOADED_KERNELS

    state.kernel_cache = {}
    for name in _KERNEL_NAMES:
        kernel = getattr(model, name, None)
        if kernel is not None:
            state.kernel_cache[name] = (kernel.neff_path, kernel.cache_key)

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


def run_sleep():
    t0 = time.time()
    _cache_kernels_and_release(state.model)
    state.model = None
    dist.barrier()
    state.sleeping = True
    print_log(f"Sleep completed in {time.time() - t0:.2f}s")


def run_wake_up(peer_url=None):
    from spike import get_spike_singleton
    from nkipy.runtime import DeviceKernel

    # Broadcast peer_url from rank 0 to all workers
    obj_list = [peer_url]
    dist.broadcast_object_list(obj_list, src=0)
    peer_url = obj_list[0]

    print_log("Waking up the model ...")
    dist.barrier()
    t0 = time.time()

    get_spike_singleton()
    dist.barrier()
    print_log(f"--> get_spike_singleton() in {time.time() - t0:.2f}s")

    # Rebuild model tensors on device
    if state.weights is not None:
        model = ModelClass(state.weights, state.config, skip_kernels=True)
    else:
        model = ModelClass(config=state.config, skip_kernels=True)
        model._allocate_empty_tensors()

    # P2P weight transfer from peer (all ranks participate)
    if peer_url:
        receive_weights(model, peer_url)

    # Reload kernels from cached NEFFs
    t_neff = time.time()
    for name, (neff_path, cache_key) in state.kernel_cache.items():
        setattr(model, name, DeviceKernel.load_with_cache_key(neff_path, cache_key, name=name))
    dist.barrier()
    print_log(f"--> Kernel reload from NEFF completed in {time.time() - t_neff:.2f}s")

    # Fetch tok_embedding from peer if needed
    if model.tok_embedding is None and peer_url:
        model.tok_embedding = fetch_tok_embedding(peer_url)

    # Warmup
    warmup_ids = pad_input_ids(
        state.tokenizer("Hello", return_tensors="np")["input_ids"],
        state.config.context_len,
        state.tokenizer.pad_token_id or 0,
    )
    _warmup(model, warmup_ids)

    state.model = model
    state.sleeping = False
    print_log(f"Wake up completed in {time.time() - t0:.2f}s")


# --- Request handling ---

def generate_text(prompt: str, max_tokens: int):
    model_inputs = state.tokenizer(prompt, return_tensors="np")
    input_ids = model_inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    if prompt_len > state.config.context_len:
        raise ValueError(
            f"Prompt length {prompt_len} exceeds compiled context_len {state.config.context_len}"
        )

    padded_ids = pad_input_ids(input_ids, state.config.context_len, state.tokenizer.pad_token_id or 0)

    broadcast_cmd(CMD_GENERATE)
    broadcast_input_ids(padded_ids)
    max_tokens = broadcast_max_tokens(max_tokens)

    tokens = run_generate(padded_ids, max_tokens)
    text = state.tokenizer.decode(tokens, skip_special_tokens=True)
    return text, prompt_len, len(tokens)


# --- Worker loop (non-rank-0) ---

def worker_loop():
    while True:
        cmd = recv_cmd()
        if cmd == CMD_GENERATE:
            padded_ids = broadcast_input_ids()
            max_tokens = broadcast_max_tokens()
            run_generate(padded_ids, max_tokens)
        elif cmd == CMD_SLEEP:
            run_sleep()
        elif cmd == CMD_WAKEUP:
            run_wake_up()
        elif cmd == CMD_P2P_PUSH_WEIGHTS:
            push_weights_to_peer(state.model)


# --- FastAPI app ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print_log("Server ready")
    yield
    print_log("Shutting down")


app = FastAPI(lifespan=lifespan)


@app.post("/completions")
async def completions(req: CompletionRequest):
    async with state._lock:
        if not state.is_ready():
            raise HTTPException(status_code=503, detail="Model is sleeping. Call /wake_up first.")

        loop = asyncio.get_event_loop()
        text, prompt_len, comp_len = await loop.run_in_executor(
            None, generate_text, req.prompt, req.max_tokens
        )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=state.args.model,
        choices=[CompletionChoice(text=text, finish_reason="stop" if comp_len < req.max_tokens else "length")],
        usage=CompletionUsage(prompt_tokens=prompt_len, completion_tokens=comp_len, total_tokens=prompt_len + comp_len),
    )


@app.post("/sleep")
async def sleep():
    async with state._lock:
        if state.sleeping:
            return {"status": "already_sleeping"}

        loop = asyncio.get_event_loop()
        def _sleep():
            broadcast_cmd(CMD_SLEEP)
            run_sleep()
            if state.weight_server is not None:
                state.weight_server.cleanup()
                state.weight_server = None
        await loop.run_in_executor(None, _sleep)
        return {"status": "sleeping"}


class WakeUpRequest(BaseModel):
    peer_url: str | None = None  # URL of active peer engine for P2P weight transfer


@app.post("/wake_up")
async def wake_up(req: WakeUpRequest = None):
    async with state._lock:
        if not state.sleeping:
            return {"status": "already_awake"}

        peer_url = req.peer_url if req else None

        loop = asyncio.get_event_loop()
        def _wake_up():
            broadcast_cmd(CMD_WAKEUP)
            run_wake_up(peer_url)
            if state.model is not None:
                state.weight_server = WeightServer(state.model)
        await loop.run_in_executor(None, _wake_up)
        return {"status": "awake"}


# --- P2P weight transfer endpoints (active engine serves these) ---

@app.get("/weight_info")
async def weight_info():
    if state.weight_server is None:
        raise HTTPException(status_code=503, detail="Weight server not initialized")
    return state.weight_server.get_weight_info()


@app.post("/p2p_push_weights")
async def p2p_push_weights(req: PushWeightsRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    def _push():
        per_rank_info = [(r["remote_metadata"], r["remote_descs"]) for r in req.per_rank]
        broadcast_cmd(CMD_P2P_PUSH_WEIGHTS)
        push_weights_to_peer(state.model, per_rank_info)
    await loop.run_in_executor(None, _push)
    return {"status": "done"}


@app.get("/tok_embedding")
async def tok_embedding():
    """Serve the token embedding table as raw bytes."""
    from fastapi.responses import Response
    if state.model is None or state.model.tok_embedding is None:
        raise HTTPException(status_code=503, detail="Token embedding not available")
    emb = state.model.tok_embedding
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
        headers={"X-Shape": ",".join(str(d) for d in shape), "X-Dtype": dtype},
    )


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["qwen3", "llama3"], required=True,
                        help="Model architecture to serve.")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name, e.g. Qwen/Qwen3-30B-A3B")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to pre-sharded safetensors directory. "
                             "If omitted, weights are loaded from peer via --p2p-peer-url.")
    parser.add_argument("--context-len", type=int, default=64,
                        help="Fixed context length for kernel compilation.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--neuron-port", type=int, default=61239)
    parser.add_argument("--core-offset", type=int, default=0,
                        help="Neuron core offset. Rank i uses core (core_offset + i).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_arch(args.arch)
    state.args = args

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{args.neuron_port}"
    dist.init_process_group()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank() + args.core_offset)

    model, tokenizer, weights, config, _ = load_model(ModelClass, args)
    state.tokenizer = tokenizer
    state.weights = weights
    state.config = config

    if weights is not None:
        state.model = model
    else:
        # Kernels compiled, cache NEFFs, release device for other standby engines
        _cache_kernels_and_release(model)
        state.model = None
        state.sleeping = True
        print_log("Kernels compiled and cached, device released, waiting for /wake_up")

    dist.barrier()

    if dist.get_rank() == 0 and state.model is not None:
        state.weight_server = WeightServer(model)

    if dist.get_rank() == 0:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        worker_loop()
