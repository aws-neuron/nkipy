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
| HLO (reference) | 8.56 ms | 116.8 inf/s | 1.0x |
| nkigen-lite — original baseline | 1007.20 ms | 0.99 inf/s | 118x |
| nkigen-lite — current | **166.75 ms** | **6.00 inf/s** | **19.5x** |

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
| HLO (reference) | 57.3 ms | 19.5 ms | 50.3 tok/s | 1.0x |
| nkigen-lite — original baseline | — | — | ~0.23 tok/s | ~220x |
| nkigen-lite — pre-KV-cache-fix | 7296 ms | 2614 ms | 0.38 tok/s | ~132x |
| nkigen-lite — KV-cache fix | 3344 ms | 1632 ms | 0.61 tok/s | ~82x |
| nkigen-lite — current (+M=1 matmul) | **3269 ms** | **1620 ms** | **0.62 tok/s** | **~81x** |

(HLO row re-measured on the same box; slightly faster than the earlier 38.1
tok/s reading, so the vs-HLO ratios shifted accordingly.)

The KV-cache-collapse fix cut TTFT 2.2x by removing the per-sequence-row unroll
of the `cache[:, :seq]=x` update. The M=1 matmul fix (below) shaved another ~2%
by dropping the systolic transposes from the per-token MoE GEMMs. The remaining
~81x gap is dominated by the **fully-unrolled 64-expert MoE loop** — but the BIR
shows the cost is *per-op HBM scaffolding* (alloc/dma/dealloc per op), not the
compute or the matmul structure (HLO keeps the same 64-dot/64-gather structure,
just ~4x denser per pair). Closing it needs **cross-op SBUF residency**, the
systemic lever at the top of the backlog — not more per-op fast paths.

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

### 2026-06-30 — matmul: transpose-free stationary load for M=1 (matrix-vector)
- **Change:** `direct_lower_matmul.py` (`emit_matmul` + `lower_matmul`) added an
  M==1 fast path. The stationary operand A is `(1, K)`; the tensor engine wants
  it as `(K, 1)` (K on partition), which the generic path builds with a
  per-K-tile `dma_transpose`. But `(1,K)->(K,1)` is a pure row-major reinterpret
  (the K elements are contiguous in HBM either way), so we view A's HBM buffer as
  `(K,1)` (zero-copy) and DMA each K-tile straight into a `(k_size,1)` stationary
  tile — no transpose. This is exactly how HLO lowers the per-token MoE expert
  dots (`mhlo.dot` with `contract_dims=[0]`, zero transposes). Non-batched A only.
- **Effect:** MoE expert GEMM `(1,2048)@(2048,384)` **172 -> 125 nki ops**, 16
  transposes -> 0; isolated MoE-expert feed-forward **136 -> 125 BIR
  instructions** (now *below* HLO's 143). Fused 30B prefill layer 71,324 ->
  67,996 nki ops.
- **End-to-end 30B (TP=4):** TTFT 3344 -> 3269 ms, decode p50 1632 -> 1620 ms,
  0.61 -> 0.62 tok/s. Modest e2e (transposes were a small fraction of the layer);
  the win is per-op quality + it generalizes to every matrix-vector matmul
  (decode attention, router, MoE). Output verified correct.
- **Numerics:** exact/within tol on device across MoE shapes, K-tiled, remainder.
- **Tests:** `test_direct_lower_matmul.py::TestMatrixVector` (5 HW cases).
- **Takeaway:** per-op matmul lowering now matches HLO; the residual 30B gap is
  HBM-round-trip scaffolding (SBUF residency), not matmul structure.

### 2026-06-30 — KV-cache update: collapse non-last-axis slice/concat/copy onto partition
- **Change:** `direct_lower_memory.py` `emit_slice` / `emit_concat` gained fast
  paths for a rank>=3 slice/concat on a **non-last** axis (all other axes kept
  full), and `direct_lower.py:_emit_hbm_copy` collapses a full rank>=3 copy to
  2D. New shared helpers `_emit_2d_rows_copy` (partition = row axis) and
  `_emit_axis_window_copy` (fold the operated axis onto the partition via an
  `(L, A, T)` view). The KV-cache update `cache[:, :seq] = x` lowers to a
  concat on axis 1 of `(1, 4096, 1, nkv, D)`-shaped tensors and its read-back to
  a slice on the same axis; both used to tile `shape[-2]` (= 1 kv-head lane) and
  unroll the 4096 sequence rows one DMA at a time (~12k ops each). Folding the
  sequence axis onto the 128-lane partition tiles it in ~32 steps.
- **Effect (fused prefill layer):** **168,658 → 71,324 nki ops (2.4x)**; the
  KV-cache concat/slice/copy (~74k ops, ~44% of the layer) drops to a few
  hundred. Python lowering time 2.8 → 1.0s; neuronx-cc compile went from
  >10 min to ~2 min (build dir 650M → 307M).
- **End-to-end 30B (TP=4):** **TTFT 7296 → 3344 ms (2.2x)**, decode p50
  2614 → 1632 ms (1.6x), throughput 0.38 → 0.61 tok/s (1.6x), ~82x vs HLO.
  Output verified correct ("...Paris. ...London...").
- **Numerics:** exact (0.0 abs err vs numpy) across concat/slice on axis 1,
  mid-axis offsets, and L>1 outer batches.
- **Tests:** `test_lowering_issues.py::test_perf2_kv_cache_concat_slice_no_seq_unroll`
  (DMA-count guard). Existing `test_direct_lower_memory.py` unchanged (760 pass).

### 2026-06-30 — elementwise segmenter: split on a second collapsed extent (RoPE)
- **Change:** `direct_lower.py:_segment_ops` now breaks an elementwise group
  when an op would introduce a second distinct non-1 collapsed `(P, F)` extent
  (new `_collapsed_pf` helper). The layout solver gives RoPE's q tensor
  `(1,128,8,64)` and k tensor `(1,128,1,64)` the same `(p_dims, f_dims)`, so
  consecutive q/k elementwise ops were merged into one segment whose ops
  collapse to two partition extents (prod 1024 vs 128). The fast collapse path
  (`_try_emit_collapsed_ew`) then bailed and the generic fallback put the
  size-1-ish axis on the partition and unrolled seq=128 one element at a time
  (one merged segment alone = 4,864 ops). Splitting keeps each group cleanly
  collapsible.
- **Effect (pattern-level):** standalone RoPE q/k `(1,128,{8,1},128)` **8,351 →
  1,754 nki ops (4.8x)**; the worst segment 4,864 → ~100 ops. Scales with
  prefill sequence length (the unrolled axis). Other building blocks unchanged.
- **Numerics:** exact (0.0 abs err vs numpy, interpreter).
- **Tests:** `test_lowering_issues.py::test_perf1_mixed_collapse_elementwise_no_blowup`
  (compute-op count guard: 128 → ≤8). New dumper:
  `examples/models/qwen3/dump_nki_ir.py` + `nki_ir_dumps/` (qwen3 MoE building
  blocks, TP=4 per-rank shapes).
- **End-to-end embedding:** **174.07 → 166.75 ms (1.04x)**, 20x → 19.5x vs HLO
  (HLO 8.56 ms). Numerics gate unchanged (min cosine 0.9997). The embedding
  layer has only one RoPE op at seq=128, so the e2e move is modest; the win
  scales with prefill length and matters most for 30B TTFT.

### 2026-06-30 — reduce over trailing axis: collapse leading dims onto partition
- **Change:** `direct_lower_reduce.py:_emit_f_reduce_inline` gained a fast path
  (`_try_emit_collapsed_f_reduce`) for rank>=3 F-reduces over a trailing-suffix
  axis with keepdims: view the input as `(prod(leading), reduced)` 2-D, tile
  the partition at 128, reduce the free axis in one `tensor_reduce_arith` per
  tile. The old path used only `shape[-2]` partition lanes (16 for the 4-D
  attention tensor) and iterated the other leading dims one row at a time
  (~640 ops/reduce). Mirrors how HLO's BIR lowers softmax (reduce → reshape to
  2-D). Wide reduced rows (> SBUF budget) fall back to the general path.
- **Effect (pattern-level):** standalone softmax `(1,16,128,128)` **2,644 →
  1,080 nki ops (2.4x)**, device **1.07 → 0.70 ms (1.54x)**; a 4-D last-axis
  reduce **961 → 179 ops (5.4x)**.
- **End-to-end embedding:** ~unchanged (174 ms). Reduce was already only ~290
  nki ops/layer (~2%) in this model — softmax is reduce-dominated in isolation
  but a small slice of the embedding layer, which is still elementwise- and
  matmul-bound. This fix matters for reduce/softmax-heavy models and is correct
  + verified regardless.
- **Numerics:** unchanged, min cosine 0.9997 vs HF.
- **Tests:** `test_lower_to_nki.py::TestCollapsedReduce` (5 interp + HW cases).
- **Reference:** compared against HLO BIR (`penguin.py`): identical 8-op
  softmax compiles to **18 instructions/core** on HLO vs our 1,080 nki ops —
  HLO additionally fuses the whole max/sub/exp/sum/div chain on-chip (no HBM
  round-trips between ops), which is the remaining SBUF-residency lever.

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

## Per-pattern device comparison (profile_patterns.py)

Building-block kernels timed in isolation on device, both backends (trn2, f32,
batch=1 seq=128, Qwen3-Embedding-0.6B shapes). `nki_ops` is the lowered
nkigen-lite instruction count; wall-clock tracks it closely (~0.4 µs/op).

```bash
cd examples/models/qwen3_embedding
rm -rf build_pat/ && QWEN3_BACKEND=hlo         uv run python profile_patterns.py
rm -rf build_pat/ && QWEN3_BACKEND=nkigen-lite uv run python profile_patterns.py
```

| pattern | HLO ms | nkigen-lite ms | slowdown | nki ops |
|---------|-------:|---------------:|---------:|--------:|
| swiglu_act (pure elementwise) | 0.132 | 0.117 | **0.9x** ✅ | 24 |
| rmsnorm | 0.128 | 0.156 | 1.2x | 112 |
| layernorm | 0.137 | 0.242 | 1.7x | 206 |
| softmax (after reduce fix) | 0.119 | 0.696 | 5.9x | 1,080 |
| matmul_gup | 0.172 | 0.708 | 4.1x | 1,049 |
| feedforward | 0.210 | 1.111 | 5.3x | 1,624 |

Slowdown tracks nki-op count → instruction-issue bound. `swiglu_act` at parity
proves the collapse approach reaches HLO when an op lowers to few instructions.
Remaining gaps: softmax (HBM round-trips between the chained ops — SBUF
residency, #1 below) and matmul (tiling emits ~1k ops for one GEMM, #3).

## Backlog (ordered by expected impact)

After the broadcast + elementwise + concat/slice + reduce fixes, the per-layer
top offenders are now `elementwise` (5,604 over 51 calls, ~110/call), `matmul`
(2,184), `concat` (915), `broadcast_to` (715), `transpose` (534), `reduce`
(302), `slice` (177).

1. **Keep intermediates in SBUF across consecutive ops** — avoid the per-op HBM
   round-trip for chained elementwise/broadcast/reduce. This is now the biggest
   structural lever: each of the ~51 elementwise segments still loads its
   inputs from and stores its result to HBM even when the next op consumes it
   immediately. Higher effort (cross-op scheduling) but the largest remaining
   win.
2. **Fuse broadcasts into their elementwise consumer** — most broadcasts only
   feed an `add`/`mul`; `tensor_scalar` / `tensor_tensor` can broadcast inline
   without materializing through HBM. (Partly subsumed by #1.)
3. **matmul** (~2.2k ops/layer, ~364/call) — per-pattern profiling shows one
   `(128,1024)@(1024,6144)` GEMM lowers to ~1,049 nki ops and runs 4.1x slower
   than HLO. The tiling over-emits load/copy instructions; high aggregate impact
   (6 matmuls/layer).
4. **transpose** (~0.5k) — lower priority. **reduce** is done (collapse fix).
5. **`_emit_hbm_copy` final output copy** — still loops `prod(shape[:-2])`
   per-batch DMAs; matters when a collapsible op's result is a graph output
   (surfaced in standalone slice profiling, minor inside the fused layer).

---

## Methodology notes

- `profile_layer.py` is a **static** analysis (op + DMA counts from the lowered
  graph) — fast, deterministic, no hardware needed. Use it to predict and
  attribute, then confirm wall-clock with the benchmark of record.
  `profile_layer.py --device` times one fused layer in isolation (×28 ≈ the
  full model, confirming the layer is ~100% of runtime).
- `profile_patterns.py` times building-block kernels (rmsnorm/softmax/ffn/…) per
  backend with nki-op counts — use it to localize a gap to one pattern.
- For an HLO reference, compile the same kernel on the HLO backend with a
  `build_dir`: the `SaveTemps` pipeline writes `penguin.py` (BIR op graph) and
  `log-neuron-cc.txt` (per-core instruction counts, e.g. "instructions=18") into
  the build dir. Diff op counts and layouts against the nkigen-lite lowering.
- Always `rm -rf build/` between backend switches and after lowering changes;
  a stale kernel cache has produced wrong output that looked like a numerical
  bug.
- Validate lowering changes cheaply against the numpy interpreter
  (`nkigen_lite.nki_ir.interpret.run`) before spending a HW compile, then add a
  `@pytest.mark.hw` regression test for the on-device path.
