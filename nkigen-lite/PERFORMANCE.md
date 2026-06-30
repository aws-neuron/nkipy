# nkigen-lite Performance Tracker

Living ledger of the nkigen-lite-vs-HLO performance gap and the work to close
it. Append a row to **Progress log** for every change that moves the numbers;
keep the **Backlog** ordered by expected impact.

For overall backend state (correctness, coverage, known limitations) see
[STATUS.md](STATUS.md).

---

## Benchmark of record

**Qwen3-Embedding-0.6B**, `examples/models/qwen3_embedding/example_retrieval.py
--benchmark`, input `(1, 128)`, LNC=2, trn2, 3 warmup + 10 timed iterations,
clean `build/` per backend. Numerics gate: `--compare` min cosine > 0.99 vs
HuggingFace (currently 0.9997).

```bash
cd examples/models/qwen3_embedding && rm -rf build/
QWEN3_BACKEND=hlo         uv run python example_retrieval.py --benchmark
rm -rf build/
QWEN3_BACKEND=nkigen-lite uv run python example_retrieval.py --benchmark
```

| Backend | Mean latency | Throughput | vs HLO |
|---------|-------------:|-----------:|-------:|
| HLO (reference) | 8.57 ms | 116.7 inf/s | 1.0x |
| nkigen-lite — original baseline | 1007.20 ms | 0.99 inf/s | 118x |
| nkigen-lite — current | **174.07 ms** | **5.74 inf/s** | **20x** |

Runs are very stable (std < 0.2 ms), so deltas below noise are meaningful.

### Secondary benchmark — Qwen3-30B-A3B (generative, MoE, TP=4)

`examples/models/qwen3/evaluate.py --benchmark`, TP=4 on trn2, checkpoint
`/home/ubuntu/models/Qwen3-30B-A3B-TP4`, `-n 64`, 1 warmup + 3 runs, clean
`build/` per backend. Both backends produce correct text ("...Paris. The
capital of the United Kingdom is London...").

```bash
cd examples/models/qwen3 && rm -rf build/
QWEN3_BACKEND=<hlo|nkigen-lite> uv run torchrun --nproc-per-node 4 \
    evaluate.py --benchmark -n 64 \
    --checkpoint /home/ubuntu/models/Qwen3-30B-A3B-TP4 \
    --model Qwen/Qwen3-30B-A3B --benchmark-warmup 1 --benchmark-runs 3
```

| Backend | TTFT | Decode p50 | Throughput | vs HLO |
|---------|-----:|-----------:|-----------:|-------:|
| HLO (reference) | 62.4 ms | 25.9 ms | 38.1 tok/s | 1.0x |
| nkigen-lite — original baseline | — | — | ~0.23 tok/s | ~165x |
| nkigen-lite — current | 7296 ms | 2614 ms | 0.38 tok/s | ~100x |

The lowering fixes carry over (~0.23 → 0.38 tok/s, ~1.65x), but the 30B gap is
still ~100x because its dominant cost is the **fully-unrolled 64-expert MoE
loop**, which the embedding model doesn't have and which these
collapse-onto-partition fixes don't touch. That loop is the main 30B lever and
is not yet addressed.

---

## Root cause (why the gap exists)

Profiled with `examples/models/qwen3_embedding/profile_layer.py`: trace one
fused transformer layer, lower tensor_ir → nki_ir, count emitted nki ops + HBM
DMAs, and attribute them back to each high-level op.

- One layer originally expanded **396 tensor_ir ops → ~82.6k nki_ir ops
  (208x)**, with ~29k DMA copies but only ~152 MB of HBM traffic. The
  whole-model DMA-bandwidth floor is ~4 ms vs ~1007 ms measured.
- **Conclusion: the gap is per-instruction / serialization overhead, not
  bandwidth.** neuronx-cc (HLO) does global scheduling + fusion; nkigen-lite's
  `direct_lower` gives each op its own load→compute→store and, worse, iterates
  leading ("batch") dims one element at a time, underusing the 128-lane
  partition.

### Per-op cost, original baseline (one fused layer)

| Source op | nki ops | DMAs | nki/call |
|-----------|--------:|-----:|---------:|
| elementwise | 28,535 | 11,356 | 560 |
| broadcast_to | 25,402 | 13,940 | 1,104 |
| concat | 2,301 | 1,534 | 767 |
| matmul (real compute) | 2,184 | 600 | 364 |
| slice | 1,551 | 1,034 | 172 |
| transpose | 534 | 354 | 107 |
| reduce | 290 | 116 | 48 |

Re-run `profile_layer.py` after any lowering change to refresh these counts.

---

## Progress log

### 2026-06-30 — concat + slice on last axis: collapse leading dims onto partition
- **Change:** `direct_lower_memory.py:emit_concat` / `emit_slice` gained fast
  paths for rank>=3 ops on the last axis. Shared helpers `_collapse_to_2d`
  (zero-copy HBM view folding leading dims onto the partition) and
  `_emit_2d_window_copy` (tiled full-height column-window copy) reduce both to
  a 2D last-axis assembly/extraction: slice → one window; concat → one window
  per input. Non-last-axis ops fall back to the original per-tile path.
- **Effect:** per-layer `concat` **2,301 → 915 nki ops (2.5x)**; `slice`
  **1,551 → 177 (8.8x)**.
- **End-to-end:** **264.73 ms → 174.07 ms (1.52x)**, 31x → 20x vs HLO.
- **Numerics:** unchanged, min cosine 0.9997 vs HF.
- **Tests:** `test_direct_lower_memory.py::TestCollapsedLastAxis` (4 interp + HW
  cases: 4D concat, uneven-width concat, low/high 4D slice).

### 2026-06-30 — elementwise segments: collapse leading dims onto partition
- **Change:** `direct_lower.py:_emit_elementwise_segment` now has a fast path
  (`_try_emit_collapsed_ew` + `_collapse_ew_shape`) for rank>=3 segments. When
  every value collapses cleanly to the rep's `(prod(shape[:-1]), shape[-1])`
  form (full, or free/partition size-1 for broadcast), it reshapes each HBM
  buffer to 2D via a zero-copy view and runs a single 2D tiled loop, reusing
  `_emit_ew_tile` (which already handles size-1 broadcast via `emit_binary_op`).
  `_emit_ew_tile` gained a `shape_override` arg so `constant` ops size their
  tile to the collapsed 2D shape. Non-collapsible segments fall back to the
  original per-tile path.
- **Effect:** per-layer `elementwise` **28,535 → 5,604 nki ops (5.1x)**; total
  layer nki ops ~55k → ~32k.
- **End-to-end:** **487.79 ms → 264.73 ms (1.84x)**, 57x → 31x vs HLO.
- **Numerics:** unchanged, min cosine 0.9997 vs HF.
- **Tests:** `test_lower_to_nki.py::TestCollapsedElementwise` (7 interp + HW
  cases: full 4D mul, free-size-1 broadcast, 4D constant chain, sub+exp).

### 2026-06-30 — broadcast_to lowering: collapse to (L, B, T)
- **Change:** `direct_lower_broadcast.py:emit_broadcast_to` now collapses a
  single contiguous broadcast run into a canonical `(L, B, T)` form via
  zero-copy HBM views (`_collapse_broadcast` / `_emit_collapsed_broadcast`) and
  picks a partition-packed strategy: `T==1` F-broadcast, `L==1` stride-0
  P-broadcast, otherwise load-once / store-B-times. Multi-run broadcasts fall
  back to the original per-tile loop.
- **Effect:** per-layer `broadcast_to` **25,402 → 715 nki ops (35x)**; total
  layer nki ops ~82.6k → ~55k; layer DMAs 28,936 → 15,322.
- **End-to-end:** **1007.20 ms → 487.79 ms (2.06x)**, 118x → 57x vs HLO.
- **Numerics:** unchanged, min cosine 0.9997 vs HF.
- **Tests:** `test_direct_lower_broadcast.py::TestEmitBroadcastCollapsed` (8 new
  HW cases covering all three collapsed paths + scalar fan-out).

---

## Backlog (ordered by expected impact)

After the broadcast + elementwise + concat/slice fixes, the per-layer top
offenders are now `elementwise` (5,604 over 51 calls, ~110/call), `matmul`
(2,184), `concat` (915), `broadcast_to` (715), `transpose` (534), `reduce`
(290), `slice` (177).

1. **Keep intermediates in SBUF across consecutive ops** — avoid the per-op HBM
   round-trip for chained elementwise/broadcast/reduce. This is now the biggest
   structural lever: each of the ~51 elementwise segments still loads its
   inputs from and stores its result to HBM even when the next op consumes it
   immediately. Higher effort (cross-op scheduling) but the largest remaining
   win.
2. **Fuse broadcasts into their elementwise consumer** — most broadcasts only
   feed an `add`/`mul`; `tensor_scalar` / `tensor_tensor` can broadcast inline
   without materializing through HBM. (Partly subsumed by #1.)
3. **matmul** (~2.2k ops/layer, ~364/call) — revisit the tiling once
   data-movement is handled.
4. **transpose** (~0.5k) and **reduce** (~0.3k) — lower priority.
5. **`_emit_hbm_copy` final output copy** — still loops `prod(shape[:-2])`
   per-batch DMAs; matters when a collapsible op's result is a graph output
   (surfaced in standalone slice profiling, minor inside the fused layer).

---

## Methodology notes

- `profile_layer.py` is a **static** analysis (op + DMA counts from the lowered
  graph) — fast, deterministic, no hardware needed. Use it to predict and
  attribute, then confirm wall-clock with the benchmark of record.
- Always `rm -rf build/` between backend switches and after lowering changes;
  a stale kernel cache has produced wrong output that looked like a numerical
  bug.
- Validate lowering changes cheaply against the numpy interpreter
  (`nkigen_lite.nki_ir.interpret.run`) before spending a HW compile, then add a
  `@pytest.mark.hw` regression test for the on-device path.
