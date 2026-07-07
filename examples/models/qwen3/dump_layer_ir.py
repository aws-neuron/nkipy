#!/usr/bin/env python3
"""Dump the full generated tensor_ir and nki_ir for one fused Qwen3-30B-A3B
MoE transformer layer (prefill), TP=4 per-rank shapes.

Companion to dump_nki_ir.py (per building-block pattern) — this one is the WHOLE
fused layer, so it captures the fully-unrolled MoE expert loop
(`for b: for t: for e in top_k`), which the per-pattern dumps miss and which the
performance ledger flags as the dominant 30B bottleneck.

The MoE loop unrolls to B*L*top_k feed-forward invocations at trace time, so
this is run at a SHORT context length (default L=8) to stay tractable; op counts
scale ~linearly in L. Set QWEN3_DUMP_SEQ to change it.

    QWEN3_BACKEND=nkigen-lite uv run python dump_layer_ir.py
"""

import os
from collections import Counter
from unittest import mock

import numpy as np
import torch.distributed as dist

WORLD_SIZE = 4
mock.patch.object(dist, "is_initialized", lambda: True).start()
mock.patch.object(dist, "get_world_size", lambda *a, **k: WORLD_SIZE).start()
mock.patch.object(dist, "get_rank", lambda *a, **k: 0).start()

from config import Config  # noqa: E402
from kernels.transformer_layer import transformer_layer  # noqa: E402

from nkipy.core.trace import NKIPyKernel  # noqa: E402
from nkigen_lite.tensor_ir.passes import lower_to_nki  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "nki_ir_dumps")
DT = np.dtype("float32")

# Qwen3-30B-A3B per-rank (TP=4) dims, matching test_oom_repro.py.
HIDDEN = 2048
HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 4
N_EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 192
QKV_OUT = 1280
O_IN = 1024
SEQ = int(os.environ.get("QWEN3_DUMP_SEQ", "8"))


def _z(shape):
    return np.zeros(shape, dtype=DT)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = Config(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        num_layers=1,
        num_experts_per_tok=TOP_K,
        num_experts=N_EXPERTS,
        context_len=SEQ,
        max_new_tokens=4,
        intermediate_size=INTERMEDIATE,
    )
    n_local_kv = max(1, NUM_KV_HEADS // WORLD_SIZE)
    cache = _z((1, cfg.max_seq_len, n_local_kv, HEAD_DIM))

    arrays = dict(
        x=_z((1, SEQ, HIDDEN)),
        start_pos=None,
        qkv_weight=_z((HIDDEN, QKV_OUT)),
        o_weight=_z((O_IN, HIDDEN)),
        input_weight=_z((HIDDEN,)),
        q_norm_weight=_z((HEAD_DIM,)),
        k_norm_weight=_z((HEAD_DIM,)),
        post_attention_weight=_z((HIDDEN,)),
        router_weight=_z((HIDDEN, N_EXPERTS)),
        gate_up_weight=_z((N_EXPERTS, HIDDEN, 2 * INTERMEDIATE)),
        down_weight=_z((N_EXPERTS, INTERMEDIATE, HIDDEN)),
        cache_k=cache,
        cache_v=cache.copy(),
        configs=cfg,
    )

    print(f"Tracing fused MoE transformer layer (L={SEQ}, top_k={TOP_K}) ...")
    k_snap = NKIPyKernel.trace(transformer_layer, backend="nkigen-lite")
    tg = k_snap.specialize(**arrays)._graph
    tensor_dump = tg.dump()
    tensor_hist = Counter(op.opcode for op in tg.ops)
    print(f"  tensor_ir ops: {len(tg.ops)}")

    kernel = NKIPyKernel.trace(transformer_layer, backend="nkigen-lite")
    tensor_graph = kernel.specialize(**arrays)._graph
    print("Lowering tensor_ir -> nki_ir ...")
    nki = lower_to_nki(tensor_graph)
    nki_hist = Counter(o.opcode for o in nki.ops)
    print(f"  nki_ir ops: {len(nki.ops)}  (expansion {len(nki.ops)/len(tg.ops):.1f}x)")

    ti_path = os.path.join(OUT_DIR, "transformer_layer.tensor_ir")
    with open(ti_path, "w") as f:
        f.write(f"# Fused Qwen3-30B-A3B MoE transformer layer (TP=4, L={SEQ}) "
                f"— tensor_ir\n")
        f.write(f"# total tensor_ir ops: {len(tg.ops)}\n\n")
        f.write("=" * 70 + "\n## opcode histogram\n" + "=" * 70 + "\n")
        for op, n in tensor_hist.most_common():
            f.write(f"  {op:24s} {n}\n")
        f.write("\n" + "=" * 70 + "\n## tensor_ir (full)\n" + "=" * 70 + "\n")
        f.write(tensor_dump + "\n")
    print(f"  wrote {ti_path}")

    nki_path = os.path.join(OUT_DIR, "transformer_layer.nki")
    with open(nki_path, "w") as f:
        f.write(f"# Fused Qwen3-30B-A3B MoE transformer layer (TP=4, L={SEQ}) "
                f"— nki_ir\n")
        f.write(f"# tensor_ir ops: {len(tg.ops)}   nki_ir ops: {len(nki.ops)}"
                f"   (expansion {len(nki.ops)/len(tg.ops):.1f}x)\n")
        f.write(f"# NOTE: MoE loop = B*L*top_k = 1*{SEQ}*{TOP_K} feed-forward "
                f"invocations, fully unrolled; scales ~linearly in L.\n\n")
        f.write("=" * 70 + "\n## nki_ir opcode histogram\n" + "=" * 70 + "\n")
        for op, n in nki_hist.most_common():
            f.write(f"  {op:24s} {n}\n")
        f.write("\n" + "=" * 70 + "\n## nki_ir (full)\n" + "=" * 70 + "\n")
        f.write(nki.dump() + "\n")
    print(f"  wrote {nki_path}")


if __name__ == "__main__":
    main()
