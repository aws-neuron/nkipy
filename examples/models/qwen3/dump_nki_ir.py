#!/usr/bin/env python3
"""Dump the lowered nki_ir for each Qwen3 building-block pattern, for inspection.

Traces each kernel, lowers tensor_ir -> nki_ir, and writes per pattern into
nki_ir_dumps/<name>.nki: the tensor_ir graph, an nki opcode histogram, and the
full nki_ir text dump. Lets us examine exactly what instructions each pattern
expands into (and where the HBM round-trips / scaffolding are).

This mirrors qwen3_embedding/dump_nki_ir.py but uses the Qwen3-30B-A3B MoE
building blocks and per-rank (TP=4) shapes. torch.distributed is mocked to a
TP=4 world (single process) so the head splits and intermediate sizes match a
real shard, exactly like test_oom_repro.py.

    uv run python dump_nki_ir.py
"""

import os
from collections import Counter
from unittest import mock

import numpy as np
import torch.distributed as dist

# The kernels divide head counts / hidden by world_size, so the dump must see
# the same TP=4 world the real shards are built for (see test_oom_repro.py).
WORLD_SIZE = 4
mock.patch.object(dist, "is_initialized", lambda: True).start()
mock.patch.object(dist, "get_world_size", lambda *a, **k: WORLD_SIZE).start()
mock.patch.object(dist, "get_rank", lambda *a, **k: 0).start()

from kernels.rmsnorm import rmsnorm_kernel  # noqa: E402
from kernels.softmax import softmax_kernel  # noqa: E402
from kernels.feedforward import feedforward_kernel, silu_kernel_  # noqa: E402
from kernels.rope import apply_rotary_emb_kernel  # noqa: E402

from nkipy.core.trace import NKIPyKernel  # noqa: E402
from nkigen_lite.tensor_ir.passes import lower_to_nki  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "nki_ir_dumps")
DT = np.dtype("float32")  # f32 so both backends share one numerics path
NORM_EPS = 1e-6

# Qwen3-30B-A3B dims (global; kernels divide head counts / intermediate by TP).
HIDDEN = 2048
HEAD_DIM = 128
NUM_HEADS = 32          # -> 8 local q heads at TP=4
NUM_KV_HEADS = 4        # -> 1 local kv head at TP=4
INTER = 768 // WORLD_SIZE  # per-rank expert intermediate = 192
SEQ = 128

N_LOCAL_HEADS = NUM_HEADS // WORLD_SIZE        # 8
N_LOCAL_KV_HEADS = max(1, NUM_KV_HEADS // WORLD_SIZE)  # 1

# Shapes for standalone building blocks.
HID = (1, SEQ, HIDDEN)                         # hidden states for norm / ffn
SCORE = (1, N_LOCAL_HEADS, SEQ, SEQ)           # attention scores for softmax
QSHAPE = (1, SEQ, N_LOCAL_HEADS, HEAD_DIM)     # query post-reshape (BSHD)
KSHAPE = (1, SEQ, N_LOCAL_KV_HEADS, HEAD_DIM)  # key post-reshape (BSHD)


def matmul_gup(x, w):
    return np.matmul(x, w)


def dump(name, fn, arrays):
    k = NKIPyKernel.trace(fn, backend="nkigen-lite")
    ir = k.specialize(**arrays)
    tensor_graph = ir._graph
    # lower_to_nki mutates the tensor graph in place (canonicalize/decompose),
    # so snapshot its dump from a fresh trace for the "tensor_ir" section.
    k2 = NKIPyKernel.trace(fn, backend="nkigen-lite")
    tg = k2.specialize(**arrays)._graph
    tensor_dump = tg.dump()

    nki = lower_to_nki(tensor_graph)
    hist = Counter(o.opcode for o in nki.ops)

    path = os.path.join(OUT_DIR, f"{name}.nki")
    with open(path, "w") as f:
        f.write(f"# Pattern: {name}\n")
        f.write(f"# inputs: " + ", ".join(
            f"{key}={getattr(v, 'shape', v)}" for key, v in arrays.items()) + "\n")
        f.write(f"# nki_ir total ops: {len(nki.ops)}\n\n")
        f.write("=" * 70 + "\n## tensor_ir (after trace)\n" + "=" * 70 + "\n")
        f.write(tensor_dump + "\n\n")
        f.write("=" * 70 + "\n## nki_ir opcode histogram\n" + "=" * 70 + "\n")
        for op, n in hist.most_common():
            f.write(f"  {op:24s} {n}\n")
        f.write("\n" + "=" * 70 + "\n## nki_ir (full)\n" + "=" * 70 + "\n")
        f.write(nki.dump() + "\n")
    print(f"  wrote {path}  ({len(nki.ops)} nki ops)")


def main():
    rng = np.random.default_rng(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    dump("rmsnorm", rmsnorm_kernel,
         {"x": rng.standard_normal(HID).astype(DT),
          "weight": rng.standard_normal(HIDDEN).astype(DT), "eps": NORM_EPS})

    dump("softmax", softmax_kernel,
         {"x": rng.standard_normal(SCORE).astype(DT)})

    dump("silu", silu_kernel_,
         {"x": rng.standard_normal((1, SEQ, INTER)).astype(DT)})

    # Single-expert feed-forward (one expert's gate_up / down shard).
    dump("feedforward", feedforward_kernel,
         {"x": rng.standard_normal(HID).astype(DT),
          "gate_up_weight": rng.standard_normal((HIDDEN, 2 * INTER)).astype(DT),
          "down_weight": rng.standard_normal((INTER, HIDDEN)).astype(DT)})

    # QKV projection matmul (hidden -> per-rank qkv width).
    qkv_out = (N_LOCAL_HEADS + 2 * N_LOCAL_KV_HEADS) * HEAD_DIM
    dump("matmul_qkv", matmul_gup,
         {"x": rng.standard_normal(HID).astype(DT),
          "w": rng.standard_normal((HIDDEN, qkv_out)).astype(DT)})

    # RoPE applied to q/k (post-reshape BSHD).
    half = HEAD_DIM // 2
    dump("rope", apply_rotary_emb_kernel,
         {"xq": rng.standard_normal(QSHAPE).astype(DT),
          "xk": rng.standard_normal(KSHAPE).astype(DT),
          "freqs_cos": rng.standard_normal((SEQ, half)).astype(DT),
          "freqs_sin": rng.standard_normal((SEQ, half)).astype(DT)})

    print(f"\nDumps in {OUT_DIR}/")


if __name__ == "__main__":
    main()
