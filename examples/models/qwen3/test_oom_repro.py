"""Reproducer / regression test for the nkigen-lite SBUF OOM when compiling the
fused Qwen3-30B-A3B transformer layer (prefill / cte_layer).

Runs single-process (no torchrun, no real checkpoint): mocks torch.distributed
to a TP=4 world and builds synthetic weights with the same per-rank shapes a
TP=4 shard has, then compiles `transformer_layer` on the nkigen-lite backend.

Before the fix this fails with neuronx-cc "Out of memory in sbuf" →
RuntimeError("Pass pipeline failed"). After the fix it compiles cleanly.

Run directly:
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python test_oom_repro.py
or as a test:
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 pytest test_oom_repro.py -v
"""

import os
from unittest import mock

import numpy as np
import torch.distributed as dist

# The real shards are built for TP=4, so the kernel must see world_size=4 to
# compute matching head splits (n_local_heads = 32/4 = 8 → qkv width 1280).
WORLD_SIZE = 4
mock.patch.object(dist, "is_initialized", lambda: True).start()
mock.patch.object(dist, "get_world_size", lambda *a, **k: WORLD_SIZE).start()
mock.patch.object(dist, "get_rank", lambda *a, **k: 0).start()

from config import Config  # noqa: E402
from kernels.transformer_layer import transformer_layer  # noqa: E402
from nkipy.core.compile import CompilationTarget  # noqa: E402
from nkipy.core.trace import NKIPyKernel  # noqa: E402
from nkipy.runtime import DeviceKernel, DeviceTensor  # noqa: E402

# Qwen3-30B-A3B dims (global; the kernel divides head counts by world_size).
HIDDEN = 2048
HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 4
N_EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 192  # 768 / TP4 (already per-rank in the shard)
CONTEXT_LEN = 8  # short prompt
DTYPE = np.dtype("float16")  # stand-in for bf16 weights

# Per-rank weight shapes (from a real shard_0.safetensors):
QKV_OUT = 1280  # (8 q + 1 k + 1 v) * 128
O_IN = 1024  # 8 local heads * 128


def _dt(shape, name=None):
    return DeviceTensor.from_numpy(np.zeros(shape, dtype=DTYPE), name)


def compile_cte_layer(backend="nkigen-lite"):
    """Compile the prefill transformer layer on the given backend.

    Raises the compiler error (e.g. the SBUF OOM) if compilation fails.
    """
    cfg = Config(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        num_layers=1,
        num_experts_per_tok=TOP_K,
        num_experts=N_EXPERTS,
        context_len=CONTEXT_LEN,
        max_new_tokens=4,
        intermediate_size=INTERMEDIATE,
    )

    x = _dt((1, CONTEXT_LEN, HIDDEN), "x_context")
    n_local_kv_heads = max(1, NUM_KV_HEADS // WORLD_SIZE)
    cache = np.zeros((1, cfg.max_seq_len, n_local_kv_heads, HEAD_DIM), dtype=DTYPE)

    kernel_fn = transformer_layer
    if backend != "hlo":
        kernel_fn = NKIPyKernel.trace(transformer_layer, backend=backend)

    # Compile only (the OOM was a compile-time failure).  Call _trace_and_compile
    # directly rather than compile_and_load: the latter would, after a successful
    # compile, go through the SPMD broadcast + device load, which need a real
    # torch.distributed group and a Neuron core that this single-process repro
    # doesn't have.  A returned NEFF path proves compilation succeeded.
    neff_path, _ = DeviceKernel._trace_and_compile(
        kernel_fn,
        "cte_layer_repro",
        (),
        dict(
            x=x,
            start_pos=None,
            qkv_weight=_dt((HIDDEN, QKV_OUT)),
            o_weight=_dt((O_IN, HIDDEN)),
            input_weight=_dt((HIDDEN,)),
            q_norm_weight=_dt((HEAD_DIM,)),
            k_norm_weight=_dt((HEAD_DIM,)),
            post_attention_weight=_dt((HIDDEN,)),
            router_weight=_dt((HIDDEN, N_EXPERTS)),
            gate_up_weight=_dt((N_EXPERTS, HIDDEN, 2 * INTERMEDIATE)),
            down_weight=_dt((N_EXPERTS, INTERMEDIATE, HIDDEN)),
            cache_k=DeviceTensor.from_numpy(cache, "cache_k"),
            cache_v=DeviceTensor.from_numpy(cache, "cache_v"),
            configs=cfg,
        ),
        additional_compiler_args=cfg.additional_compiler_args_nkipy,
        use_cached_if_exists=False,
        build_dir="./build_repro",
        target=CompilationTarget.DEFAULT,
    )
    assert neff_path is not None and os.path.exists(neff_path), (
        f"compilation did not produce a NEFF: {neff_path}"
    )
    return neff_path


def test_cte_layer_compiles_nkigen_lite():
    """The fused transformer layer should compile on nkigen-lite without OOM."""
    compile_cte_layer(backend="nkigen-lite")


if __name__ == "__main__":
    backend = os.environ.get("REPRO_BACKEND", "nkigen-lite")
    print(f"[repro] compiling transformer_layer on backend={backend} ...", flush=True)
    compile_cte_layer(backend=backend)
    print("[repro] COMPILE OK", flush=True)
