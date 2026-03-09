"""
FastAPI server for model inference, with vLLM-style API endpoints.
Supports both Qwen3 and Llama3 architectures via --arch flag.

Usage:
    torchrun --nproc_per_node=8 server.py --arch qwen3 --model Qwen/Qwen3-30B-A3B --checkpoint /kaena/qwen3_shards_30B_A3B_TP8
    torchrun --nproc_per_node=8 server.py --arch llama3 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --checkpoint ./tmp_tinyllama_TP8

Endpoints:
    POST /completions  - Generate text completions
    POST /sleep        - Offload model from device
    POST /wake_up      - Reload model onto device

Example curl:
    curl -X POST http://localhost:8000/completions \
      -H "Content-Type: application/json" \
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

# Add models/ to sys.path so common.* imports work
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
sys.path.insert(0, _models_dir)

from common.model import load_model
from common.utils import print_log

# These will be set after parsing --arch
ModelClass = None  # Qwen3Model or Llama3Model
EOS_TOKEN_IDS = set()

# --- Command codes broadcast from rank 0 to all workers ---
CMD_GENERATE = 0
CMD_SLEEP = 1
CMD_WAKEUP = 2


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


# --- Global state ---

def setup_arch(arch: str):
    """Set global ModelClass and EOS_TOKEN_IDS for the chosen architecture."""
    global ModelClass, EOS_TOKEN_IDS

    # Add arch-specific dir to sys.path for kernel imports
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
        self.kernel_neff_paths = None  # saved during sleep for fast wake_up
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


def run_sleep():
    from spike import get_spike_singleton, reset as spike_reset
    from nkipy.runtime.device_kernel import _LOADED_KERNELS

    t0 = time.time()

    model = state.model
    spike = get_spike_singleton()

    # Save kernel NEFF paths for fast reload on wake_up
    state.kernel_neff_paths = {
        name: getattr(kernel, "neff_path", None)
        for name, kernel in [
            ("kernel_cte", model.kernel_cte),
            ("kernel_tkg", model.kernel_tkg),
            ("kernel_cte_greedy_sampling", model.kernel_cte_greedy_sampling),
            ("kernel_tkg_greedy_sampling", model.kernel_tkg_greedy_sampling),
        ]
        if kernel is not None
    }

    for layer in model.layer_tensors:
        for t in layer.values():
            spike.free_tensor(t.tensor_ref)
    for t in (model.norm_weight, model.lm_head_weight):
        if t is not None:
            spike.free_tensor(t.tensor_ref)

    for kernel in (model.kernel_cte, model.kernel_tkg,
                   model.kernel_cte_greedy_sampling, model.kernel_tkg_greedy_sampling):
        if kernel is not None:
            spike.unload_model(kernel.model_ref)
    _LOADED_KERNELS.clear()

    state.model = None
    gc.collect()

    # Release Neuron cores (calls nrt_close) so another engine can use them
    spike_reset()

    state.sleeping = True
    print_log(f"Sleep completed in {time.time() - t0:.2f}s")


def _load_kernels_from_neff(neff_paths):
    from nkipy.runtime import DeviceKernel

    distributed = dist.is_initialized() and dist.get_world_size() > 1
    kernels = {}
    for name, neff_path in neff_paths.items():
        if distributed:
            dist.barrier()
            kernels[name] = DeviceKernel.load_from_neff(
                neff_path,
                name=name,
                cc_enabled=True,
                rank_id=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
        else:
            kernels[name] = DeviceKernel.load_from_neff(neff_path, name=name)
    return kernels


def run_wake_up():
    from spike import get_spike_singleton

    t0 = time.time()

    # Re-acquire Neuron cores (lazy nrt_init via get_spike_singleton)
    get_spike_singleton()

    # Build model but skip kernel compilation — we'll inject cached kernels
    model = ModelClass(state.weights, state.config, skip_kernels=True)

    # Reload kernels directly from saved NEFF paths
    t_neff = time.time()
    kernels = _load_kernels_from_neff(state.kernel_neff_paths)
    print_log(f"--> _load_kernels_from_neff completed in {time.time() - t_neff:.2f}s")
    model.kernel_cte = kernels.get("kernel_cte")
    model.kernel_tkg = kernels.get("kernel_tkg")
    model.kernel_cte_greedy_sampling = kernels.get("kernel_cte_greedy_sampling")
    model.kernel_tkg_greedy_sampling = kernels.get("kernel_tkg_greedy_sampling")

    # Warmup
    dummy_ids = pad_input_ids(
        state.tokenizer("Hello", return_tensors="np")["input_ids"],
        state.config.context_len,
        state.tokenizer.pad_token_id or 0,
    )
    for i, _ in enumerate(model.generate(dummy_ids)):
        if i == 1:
            break
    state.model = model
    state.sleeping = False
    print_log(f"Wake up completed in {time.time() - t0:.2f}s")


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


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print_log("Server ready")
    yield
    print_log("Shutting down")


app = FastAPI(lifespan=lifespan)


# --- Endpoints ---

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
        await loop.run_in_executor(None, _sleep)
        return {"status": "sleeping"}


@app.post("/wake_up")
async def wake_up():
    async with state._lock:
        if not state.sleeping:
            return {"status": "already_awake"}

        loop = asyncio.get_event_loop()
        def _wake_up():
            broadcast_cmd(CMD_WAKEUP)
            run_wake_up()
        await loop.run_in_executor(None, _wake_up)
        return {"status": "awake"}


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["qwen3", "llama3"], required=True,
                        help="Model architecture to serve.")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name, e.g. Qwen/Qwen3-30B-A3B or TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to pre-sharded safetensors directory.")
    parser.add_argument("--context-len", type=int, default=32,
                        help="Fixed context length for kernel compilation. Prompts are padded to this.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--neuron-port", type=int, default=61239)
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
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(dist.get_rank())

    # All ranks load model together (kernels may have collectives)
    model, tokenizer, weights, config, _ = load_model(ModelClass, args)
    state.model = model
    state.tokenizer = tokenizer
    state.weights = weights
    state.config = config
    dist.barrier()

    if dist.get_rank() == 0:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        worker_loop()
