#!/usr/bin/env python3
"""Dump the lowered nki_ir for each building-block pattern, for inspection.

Traces each kernel, lowers tensor_ir -> nki_ir, and writes per pattern into
nki_ir_dumps/<name>.nki: the tensor_ir graph, an nki opcode histogram, and the
full nki_ir text dump. Lets us examine exactly what instructions each pattern
expands into (and where the HBM round-trips / scaffolding are).

    uv run python dump_nki_ir.py
"""

import os
from collections import Counter

import numpy as np

from config import Qwen3Config
from kernels.rmsnorm import rmsnorm
from kernels.softmax import softmax
from kernels.ffn import feedforward_kernel

from nkipy.core.trace import NKIPyKernel
from nkigen_lite.tensor_ir.passes import lower_to_nki

OUT_DIR = os.path.join(os.path.dirname(__file__), "nki_ir_dumps")
DT = np.dtype("float32")

H = 1024
INTER = 3072
HEADS = 16
SEQ = 128
SCORE = (1, HEADS, SEQ, SEQ)
HID = (1, SEQ, H)


def layernorm(x, weight, bias, eps: float):
    cd = np.float32
    x = x.astype(cd)
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def matmul_gup(x, w):
    return np.matmul(x, w)


def swiglu_act(g, u):
    return (g * (1 / (1 + np.exp(-g)))) * u


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
            f"{k}={getattr(v,'shape',v)}" for k, v in arrays.items()) + "\n")
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
    cfg = Qwen3Config()
    eps = cfg.rms_norm_eps

    dump("rmsnorm", rmsnorm,
         {"x": rng.standard_normal(HID).astype(DT),
          "weight": rng.standard_normal(H).astype(DT), "eps": eps})
    dump("softmax", softmax, {"x": rng.standard_normal(SCORE).astype(DT)})
    dump("layernorm", layernorm,
         {"x": rng.standard_normal(HID).astype(DT),
          "weight": rng.standard_normal(H).astype(DT),
          "bias": rng.standard_normal(H).astype(DT), "eps": eps})
    dump("feedforward", feedforward_kernel,
         {"x": rng.standard_normal(HID).astype(DT),
          "gate_up_weight": rng.standard_normal((H, 2 * INTER)).astype(DT),
          "down_weight": rng.standard_normal((INTER, H)).astype(DT),
          "gate_up_bias": np.zeros(2 * INTER, dtype=DT),
          "down_bias": np.zeros(H, dtype=DT)})
    dump("matmul_gup", matmul_gup,
         {"x": rng.standard_normal(HID).astype(DT),
          "w": rng.standard_normal((H, 2 * INTER)).astype(DT)})
    dump("swiglu_act", swiglu_act,
         {"g": rng.standard_normal((1, SEQ, INTER)).astype(DT),
          "u": rng.standard_normal((1, SEQ, INTER)).astype(DT)})

    print(f"\nDumps in {OUT_DIR}/")


if __name__ == "__main__":
    main()
