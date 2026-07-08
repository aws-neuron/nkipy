#!/usr/bin/env python3
"""Static per-op profiler for the nkigen-lite lowering of one fused Qwen3-30B-A3B
MoE transformer layer (TP=4). Mirrors qwen3_embedding/profile_layer.py but for
the MoE layer, whose fully-unrolled expert loop dominates.

Attributes emitted nki ops + DMAs back to the high-level tensor_ir op that
produced them. Run at a short context (QWEN3_DUMP_SEQ, default 8) since the MoE
loop unrolls B*L*top_k feed-forward calls; counts scale ~linearly in L.

    QWEN3_BACKEND=nkigen-lite uv run python profile_layer.py
"""

import os
from collections import Counter, defaultdict
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

DT = np.dtype("float32")
HIDDEN, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS = 2048, 128, 32, 4
N_EXPERTS, TOP_K, INTERMEDIATE = 128, 8, 192
QKV_OUT, O_IN = 1280, 1024
SEQ = int(os.environ.get("QWEN3_DUMP_SEQ", "8"))


def _z(shape):
    return np.zeros(shape, dtype=DT)


def main():
    cfg = Config(
        hidden_size=HIDDEN, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS, num_layers=1, num_experts_per_tok=TOP_K,
        num_experts=N_EXPERTS, context_len=SEQ, max_new_tokens=4,
        intermediate_size=INTERMEDIATE,
    )
    n_local_kv = max(1, NUM_KV_HEADS // WORLD_SIZE)
    cache = _z((1, cfg.max_seq_len, n_local_kv, HEAD_DIM))
    arrays = dict(
        x=_z((1, SEQ, HIDDEN)), start_pos=None,
        qkv_weight=_z((HIDDEN, QKV_OUT)), o_weight=_z((O_IN, HIDDEN)),
        input_weight=_z((HIDDEN,)), q_norm_weight=_z((HEAD_DIM,)),
        k_norm_weight=_z((HEAD_DIM,)), post_attention_weight=_z((HIDDEN,)),
        router_weight=_z((HIDDEN, N_EXPERTS)),
        gate_up_weight=_z((N_EXPERTS, HIDDEN, 2 * INTERMEDIATE)),
        down_weight=_z((N_EXPERTS, INTERMEDIATE, HIDDEN)),
        cache_k=cache, cache_v=cache.copy(), configs=cfg,
    )

    print(f"Tracing fused MoE layer (L={SEQ}, top_k={TOP_K}) ...")
    kernel = NKIPyKernel.trace(transformer_layer, backend="nkigen-lite")
    tensor_graph = kernel.specialize(**arrays)._graph
    n_tensor = len(tensor_graph.ops)
    print(f"tensor_ir: {n_tensor} ops")
    for op, n in Counter(o.opcode for o in tensor_graph.ops).most_common(12):
        print(f"  {op:20s} {n}")

    from nkigen_lite.tensor_ir.passes import lower_to_nki
    from nkigen_lite.tensor_ir.passes.basic import direct_lower as dl

    attribution = defaultdict(lambda: {"nki_ops": 0, "dmas": 0, "calls": 0})

    def _instrument(orig, label):
        """Wrap an emitter to attribute the nki ops/DMAs it appends to ``label``."""
        def wrapped(nb, *a, **kw):
            before = len(nb.graph.ops)
            r = orig(nb, *a, **kw)
            added = nb.graph.ops[before:]
            attribution[label]["nki_ops"] += len(added)
            attribution[label]["dmas"] += sum(
                1 for o in added if o.opcode in ("dma_copy", "dma_copy_indirect")
            )
            attribution[label]["calls"] += 1
            return r
        return wrapped

    # Elementwise ops are dispatched by a direct module-global call in
    # ``lower_graph`` (``_emit_elementwise_op(nb, op, hbm_map)``), so reassigning
    # the module attribute intercepts them.
    dl._emit_elementwise_op = _instrument(dl._emit_elementwise_op, "elementwise")

    # Every other op dispatches through the ``_OP_EMITTERS`` table keyed by
    # opcode. The table holds direct function references captured at import, so
    # patching module attributes would miss them — wrap the entries in place.
    # Keying the label by opcode auto-covers reduce/matmul/transpose/reshape/
    # slice/concat/broadcast_to/iota/topk/gather/scatter/collectives.
    for opcode in list(dl._OP_EMITTERS):
        dl._OP_EMITTERS[opcode] = _instrument(dl._OP_EMITTERS[opcode], opcode)

    nki = lower_to_nki(tensor_graph)
    n_nki = len(nki.ops)
    print(f"\nnki_ir: {n_nki} ops  (expansion {n_nki / n_tensor:.1f}x)")
    for op, n in Counter(o.opcode for o in nki.ops).most_common(14):
        print(f"  {op:20s} {n}")

    print("\n" + "=" * 60)
    print(f"Attribution (L={SEQ}; MoE = B*L*top_k = {SEQ * TOP_K} expert calls)")
    print("=" * 60)
    print(f"  {'op':14s} {'calls':>6s} {'nki_ops':>9s} {'dmas':>7s} {'nki/call':>9s}")
    for opcode, st in sorted(attribution.items(), key=lambda kv: -kv[1]["nki_ops"]):
        per = st["nki_ops"] / st["calls"] if st["calls"] else 0
        print(f"  {opcode:14s} {st['calls']:6d} {st['nki_ops']:9d} "
              f"{st['dmas']:7d} {per:9.1f}")

    n_ind = sum(1 for o in nki.ops if o.opcode == "dma_copy_indirect")
    print(f"\n  dma_copy_indirect (expert weight gather): {n_ind}")
    print(f"  ~ per expert call: {n_ind / (SEQ * TOP_K):.1f}")


if __name__ == "__main__":
    main()
