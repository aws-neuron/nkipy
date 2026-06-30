#!/usr/bin/env python3
"""Static per-op profiler for the nkigen-lite lowering of one fused Qwen3 layer.

Traces the fused transformer-layer kernel, lowers tensor_ir -> nki_ir, and
reports what each high-level op expands into: nki_ir op counts, HBM DMA traffic
(loads/stores + bytes), matmul FLOPs, and the per-op expansion factor.

This tests the "every tensor_ir op does its own HBM load->compute->store"
hypothesis for the ~118x gap vs HLO.

Usage:
    uv run python profile_layer.py
"""

import numpy as np
from collections import Counter, defaultdict

from config import get_config
from kernels.transformer_layer import transformer_layer_kernel
from kernels.rope import compute_qwen3_cos_sin

from nkipy.core.trace import NKIPyKernel
from nkigen_lite.core import _DTYPE_BYTES
from nkigen_lite.nki_ir.ir import MemorySpace


def _tile_bytes(v):
    t = v.type
    return t.num_elements * _DTYPE_BYTES[t.dtype]


def _dma_direction_and_bytes(op):
    """Return (direction, bytes) for a dma_copy / dma_copy_indirect op.

    nki_ir dma_copy(dst, src): inputs = [dst, src, ...]. The op carries an
    explicit "direction" attr ("load" = HBM->SBUF, "store" = SBUF->HBM).
    Bytes moved equal the on-chip (SBUF) tile size.
    """
    direction = op.attrs.get("direction")
    dst_v, src_v = op.inputs[0], op.inputs[1]
    if direction is None:
        direction = "store" if dst_v.type.memory == MemorySpace.HBM else "load"
    # On-chip tile is the non-HBM side.
    onchip = src_v if src_v.type.memory != MemorySpace.HBM else dst_v
    return direction, _tile_bytes(onchip)


def main():
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

    hidden = np.empty(
        (config.max_batch_size, config.max_model_len, config.hidden_size), dtype=dt
    )

    print("Tracing fused transformer layer (nkigen-lite)...")
    kernel = NKIPyKernel.trace(transformer_layer_kernel, backend="nkigen-lite")
    ir = kernel.specialize(
        hidden_states=hidden,
        input_layernorm_weight=np.empty(config.hidden_size, dtype=dt),
        qkv_weight=np.empty((config.hidden_size, qkv_size), dtype=dt),
        o_weight=np.empty(
            (config.num_attention_heads * config.head_dim, config.hidden_size), dtype=dt
        ),
        q_norm_weight=np.empty(config.head_dim, dtype=dt),
        k_norm_weight=np.empty(config.head_dim, dtype=dt),
        cos=cos.astype(np.float32),
        sin=sin.astype(np.float32),
        post_attention_layernorm_weight=np.empty(config.hidden_size, dtype=dt),
        gate_up_weight=np.empty(
            (config.hidden_size, 2 * config.intermediate_size), dtype=dt
        ),
        down_weight=np.empty((config.intermediate_size, config.hidden_size), dtype=dt),
        gate_up_bias=np.zeros(2 * config.intermediate_size, dtype=dt),
        down_bias=np.zeros(config.hidden_size, dtype=dt),
        config=config,
        compute_dtype=dt,
    )

    tensor_graph = ir._graph
    n_tensor_ops = len(tensor_graph.ops)
    tensor_op_counts = Counter(op.opcode for op in tensor_graph.ops)

    print(f"\ntensor_ir graph: {n_tensor_ops} ops")
    for opcode, n in tensor_op_counts.most_common():
        print(f"  {opcode:24s} {n}")

    # Lower to nki_ir, attributing emitted nki ops back to the high-level
    # tensor_ir op that produced them. We wrap the per-op emit dispatch
    # functions in direct_lower and snapshot the nki graph size around each.
    print("\nLowering tensor_ir -> nki_ir ...")
    from nkigen_lite.tensor_ir.passes import lower_to_nki
    from nkigen_lite.tensor_ir.passes.basic import direct_lower as dl

    # attribution[opcode] -> {"nki_ops": int, "dmas": int, "calls": int}
    attribution = defaultdict(lambda: {"nki_ops": 0, "dmas": 0, "calls": 0})

    def _wrap(fn_name, opcode_getter):
        orig = getattr(dl, fn_name)

        def wrapped(nb, *a, **kw):
            before = len(nb.graph.ops)
            before_dma = sum(
                1 for o in nb.graph.ops if o.opcode in ("dma_copy", "dma_copy_indirect")
            )
            r = orig(nb, *a, **kw)
            added = nb.graph.ops[before:]
            opcode = opcode_getter(a)
            attribution[opcode]["nki_ops"] += len(added)
            attribution[opcode]["dmas"] += sum(
                1 for o in added if o.opcode in ("dma_copy", "dma_copy_indirect")
            )
            attribution[opcode]["calls"] += 1
            return r

        setattr(dl, fn_name, wrapped)

    # elementwise segments take a list of ops; label by the set of opcodes
    _orig_ew = dl._emit_elementwise_segment

    def _wrapped_ew(nb, ops, *a, **kw):
        before = len(nb.graph.ops)
        r = _orig_ew(nb, ops, *a, **kw)
        added = nb.graph.ops[before:]
        attribution["elementwise"]["nki_ops"] += len(added)
        attribution["elementwise"]["dmas"] += sum(
            1 for o in added if o.opcode in ("dma_copy", "dma_copy_indirect")
        )
        attribution["elementwise"]["calls"] += 1
        return r

    dl._emit_elementwise_segment = _wrapped_ew
    _wrap("_emit_reduce_op", lambda a: "reduce")
    _wrap("_emit_matmul_op", lambda a: "matmul")
    _wrap("_emit_transpose_op", lambda a: "transpose")
    _wrap("_emit_reshape_op", lambda a: "reshape")
    _wrap("_emit_slice_op", lambda a: "slice")
    _wrap("_emit_concat_op", lambda a: "concat")
    _wrap("_emit_broadcast_op", lambda a: "broadcast_to")

    nki_graph = lower_to_nki(tensor_graph)
    n_nki_ops = len(nki_graph.ops)
    nki_op_counts = Counter(op.opcode for op in nki_graph.ops)

    print(f"\nnki_ir graph: {n_nki_ops} ops  (expansion {n_nki_ops / n_tensor_ops:.1f}x)")
    for opcode, n in nki_op_counts.most_common():
        print(f"  {opcode:24s} {n}")

    # DMA / HBM traffic analysis
    dma_opcodes = {"dma_copy", "dma_copy_indirect"}
    n_load = n_store = 0
    bytes_load = bytes_store = 0
    for op in nki_graph.ops:
        if op.opcode in dma_opcodes:
            direction, nbytes = _dma_direction_and_bytes(op)
            if direction == "load":
                n_load += 1
                bytes_load += nbytes
            else:
                n_store += 1
                bytes_store += nbytes

    # Matmul FLOPs (2*M*N*K per matmul). nki_ir matmul: stationary[K,M], moving[K,N] -> [M,N]
    total_matmul_flops = 0
    n_matmul = 0
    for op in nki_graph.ops:
        if op.opcode == "matmul":
            n_matmul += 1
            stat, mov = op.inputs[0], op.inputs[1]
            K, M = stat.type.shape[-2], stat.type.shape[-1]
            N = mov.type.shape[-1]
            total_matmul_flops += 2 * M * N * K

    # alloc accounting
    n_alloc = nki_op_counts.get("alloc", 0)
    hbm_alloc_bytes = sum(
        _tile_bytes(op.results[0])
        for op in nki_graph.ops
        if op.opcode == "alloc" and op.results[0].type.memory == MemorySpace.HBM
    )

    print("\n" + "=" * 60)
    print("Attribution: which tensor_ir op expands into what")
    print("=" * 60)
    print(f"  {'op':14s} {'calls':>6s} {'nki_ops':>9s} {'dmas':>7s} {'nki/call':>9s}")
    rows = sorted(attribution.items(), key=lambda kv: -kv[1]["nki_ops"])
    for opcode, st in rows:
        per = st["nki_ops"] / st["calls"] if st["calls"] else 0
        print(
            f"  {opcode:14s} {st['calls']:6d} {st['nki_ops']:9d} "
            f"{st['dmas']:7d} {per:9.1f}"
        )

    # Per-shape breakdown of the two dominant categories, computed directly
    # from the tensor_ir graph (cheap, exact for unrolled loops). The driver of
    # the explosion is the batch product prod(out_shape[:-2]): the broadcast and
    # elementwise lowerings emit a load+store DMA pair per batch element.
    print("\n" + "=" * 60)
    print("broadcast_to shapes (driver = batch product prod(out[:-2]))")
    print("=" * 60)
    bc = defaultdict(int)
    for op in tensor_graph.ops:
        if op.opcode == "broadcast_to":
            bc[(op.inputs[0].type.shape, op.results[0].type.shape)] += 1
    for (ishape, oshape), n in sorted(
        bc.items(), key=lambda kv: -np.prod(kv[0][1][:-2] or (1,)) * kv[1]
    ):
        nbatch = int(np.prod(oshape[:-2])) if len(oshape) > 2 else 1
        print(f"  {n}x  {ishape} -> {oshape}   n_batch={nbatch}")

    print("\n" + "=" * 60)
    print("HBM DMA traffic (per single layer)")
    print("=" * 60)
    print(f"  loads:  {n_load:5d}   {bytes_load / 1e6:8.2f} MB")
    print(f"  stores: {n_store:5d}   {bytes_store / 1e6:8.2f} MB")
    print(f"  total:  {n_load + n_store:5d}   {(bytes_load + bytes_store) / 1e6:8.2f} MB")
    print(f"  HBM scratch allocs: {hbm_alloc_bytes / 1e6:.2f} MB")

    print("\n" + "=" * 60)
    print("Compute")
    print("=" * 60)
    print(f"  matmuls: {n_matmul}")
    print(f"  matmul FLOPs: {total_matmul_flops / 1e9:.3f} GFLOP")

    # Roofline-ish: 28 layers, what does the DMA volume imply?
    n_layers = config.num_hidden_layers
    total_dma_gb = (bytes_load + bytes_store) * n_layers / 1e9
    print("\n" + "=" * 60)
    print(f"Whole model estimate ({n_layers} layers)")
    print("=" * 60)
    print(f"  total HBM DMA: {total_dma_gb:.2f} GB")
    print(f"  total matmul:  {total_matmul_flops * n_layers / 1e9:.2f} GFLOP")
    # trn2 HBM ~ 1.3 TB/s/core-ish; just give the implied floor at a few BWs
    for bw in (0.5, 1.0, 2.0):
        print(f"    DMA-bound floor @ {bw:.1f} TB/s: {total_dma_gb / (bw*1000) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
