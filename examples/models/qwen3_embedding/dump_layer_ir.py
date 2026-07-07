#!/usr/bin/env python3
"""Dump the full generated tensor_ir and nki_ir for one fused Qwen3 layer.

Traces the fused transformer-layer kernel (the body that runs x28 in the
0.6B embedding model), lowers tensor_ir -> nki_ir, and writes into
nki_ir_dumps/transformer_layer.{tensor_ir,nki}:
  - the tensor_ir graph after trace,
  - an nki opcode histogram,
  - the full lowered nki_ir text.

Companion to dump_nki_ir.py (per building-block pattern) and profile_layer.py
(per-op attribution). This one is the whole layer, so it shows the actual
inter-op HBM boundaries that fusion would remove.

    QWEN3_BACKEND=nkigen-lite uv run python dump_layer_ir.py
"""

import os
from collections import Counter

import numpy as np

from config import get_config
from kernels.transformer_layer import transformer_layer_kernel
from kernels.rope import compute_qwen3_cos_sin

from nkipy.core.trace import NKIPyKernel
from nkigen_lite.tensor_ir.passes import lower_to_nki

OUT_DIR = os.path.join(os.path.dirname(__file__), "nki_ir_dumps")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = get_config("0.6b")
    dt = config.dtype

    cos, sin = compute_qwen3_cos_sin(
        max_model_len=config.max_model_len,
        head_dim=config.head_dim,
        theta=config.rope_theta,
    )
    qkv_size = (
        config.num_attention_heads + 2 * config.num_key_value_heads
    ) * config.head_dim
    H = config.hidden_size
    inter = config.intermediate_size

    arrays = dict(
        hidden_states=np.empty(
            (config.max_batch_size, config.max_model_len, H), dtype=dt
        ),
        input_layernorm_weight=np.empty(H, dtype=dt),
        qkv_weight=np.empty((H, qkv_size), dtype=dt),
        o_weight=np.empty((config.num_attention_heads * config.head_dim, H), dtype=dt),
        q_norm_weight=np.empty(config.head_dim, dtype=dt),
        k_norm_weight=np.empty(config.head_dim, dtype=dt),
        cos=cos.astype(np.float32),
        sin=sin.astype(np.float32),
        post_attention_layernorm_weight=np.empty(H, dtype=dt),
        gate_up_weight=np.empty((H, 2 * inter), dtype=dt),
        down_weight=np.empty((inter, H), dtype=dt),
        gate_up_bias=np.zeros(2 * inter, dtype=dt),
        down_bias=np.zeros(H, dtype=dt),
        config=config,
        compute_dtype=dt,
    )

    # lower_to_nki mutates the tensor graph in place, so snapshot the tensor_ir
    # dump from a fresh trace before lowering the other.
    k_snap = NKIPyKernel.trace(transformer_layer_kernel, backend="nkigen-lite")
    tg = k_snap.specialize(**arrays)._graph
    tensor_dump = tg.dump()
    tensor_hist = Counter(op.opcode for op in tg.ops)

    kernel = NKIPyKernel.trace(transformer_layer_kernel, backend="nkigen-lite")
    tensor_graph = kernel.specialize(**arrays)._graph
    nki = lower_to_nki(tensor_graph)
    nki_hist = Counter(o.opcode for o in nki.ops)

    ti_path = os.path.join(OUT_DIR, "transformer_layer.tensor_ir")
    with open(ti_path, "w") as f:
        f.write("# Fused Qwen3-Embedding-0.6B transformer layer — tensor_ir\n")
        f.write(f"# total tensor_ir ops: {len(tg.ops)}\n\n")
        f.write("=" * 70 + "\n## opcode histogram\n" + "=" * 70 + "\n")
        for op, n in tensor_hist.most_common():
            f.write(f"  {op:24s} {n}\n")
        f.write("\n" + "=" * 70 + "\n## tensor_ir (full)\n" + "=" * 70 + "\n")
        f.write(tensor_dump + "\n")
    print(f"  wrote {ti_path}  ({len(tg.ops)} tensor_ir ops)")

    nki_path = os.path.join(OUT_DIR, "transformer_layer.nki")
    with open(nki_path, "w") as f:
        f.write("# Fused Qwen3-Embedding-0.6B transformer layer — nki_ir\n")
        f.write(f"# tensor_ir ops: {len(tg.ops)}   nki_ir ops: {len(nki.ops)}"
                f"   (expansion {len(nki.ops) / len(tg.ops):.1f}x)\n\n")
        f.write("=" * 70 + "\n## nki_ir opcode histogram\n" + "=" * 70 + "\n")
        for op, n in nki_hist.most_common():
            f.write(f"  {op:24s} {n}\n")
        f.write("\n" + "=" * 70 + "\n## nki_ir (full)\n" + "=" * 70 + "\n")
        f.write(nki.dump() + "\n")
    print(f"  wrote {nki_path}  ({len(nki.ops)} nki ops)")


if __name__ == "__main__":
    main()
