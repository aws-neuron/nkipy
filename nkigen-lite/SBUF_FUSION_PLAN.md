# SBUF Fusion Plan

Goal: reduce HBM traffic in the basic lowering flow by keeping more
intermediates in SBUF — dead-store elimination, view-composition of
data-movement ops, and real op fusion (matmul epilogue, reduce + elementwise).

Context: see `REVIEW_basic_lowering.md` for the design review this plan grew
out of, and `PERFORMANCE.md` for the qwen3 layer profile that motivates it
(broadcast/elementwise DMA pairs dominate; per-op HBM round-trips).

Benchmark: `examples/models/qwen3/profile_layer.py` — re-run after each phase
and record op counts / timing in the Progress log below.

---

## Phase 0 — Stop paying for traffic we don't need

Cheap, independent, no restructuring.

- [x] **Dead-store elimination in elementwise segments.**
  `lower_graph` computes escape analysis against the segmentation: an
  elementwise result gets an HBM buffer + store only if consumed by another
  segment or a graph output. Modest on the qwen3 MoE layer (−184 DMAs, −5%
  of elementwise DMAs): segments average ~2 ops and most results escape into
  reshapes/matmuls. Will compound once Phase 2 grows the groups.
- [x] **Dtype-aware tile sizes for elementwise.**
  Segments now tile to their widest dtype (`_segment_dtype`) instead of the
  F32 default. One shared dtype per segment keeps per-value slice offsets
  aligned with the rep loop. No effect on the (all-F32) qwen3 profile; pays
  on bf16 models.

## Phase 0.5 — Profile-driven hotspot fixes (added after baseline)

The baseline attribution showed three hotspots the original plan under-ranked;
fixed before Phase 1:

- [x] **Partition-packed wide-row gather_rows.** The MoE expert-weight gather
  (M=1, W=786432) emitted ~212 single-lane column-window indirect DMAs per
  expert. Packed path: view the (N, W) table as (N*128, W/128), expand the
  dynamic index to idx*128 + lane (iota), fetch the row as one
  partition-packed indirect DMA. Cost model keeps tall gathers (embedding)
  on the generic path. 20416 → 1920 gather ops.
- [x] **Splat-fill constant concat inputs.** The router-weight assembly
  concats ~130 scalar constants per token; each constant was materialized in
  HBM and copied in. Now `emit_concat` memsets an SBUF tile and stores it
  straight into the output window (per-concat tile cache; dead constants
  pruned from elementwise segments). 7330 → 6739 concat ops **and**
  elementwise 10270 → 3274 (the constants' emission was attributed there).
- [x] Remaining transpose cost investigated: (1,4096,8,128) head/seq swaps
  are pure DMA remapping already near-minimal; scores transpose (0,1,3,2)
  misses the passthrough path because I*J=524288 exceeds the SBUF free
  budget. ~2.9k ops (6.7%) — not worth further effort before Phase 2.

## Phase 0.75 — Segment-first layout (done)

Design decision recorded 2026-07-05: layouts are **per-segment** decisions
made by each emitter, not per-value graph properties. Every segment boundary
round-trips through HBM, which is layout-agnostic (row-major), so a consumer
segment can load a value in any layout regardless of how the producer stored
it — there is no global layout problem until cross-segment SBUF residency
exists (explicitly deferred below).

- [x] **Delete the global layout solver** (`layout_solver.py` / `solve_graph`,
  a five-phase propagation pass whose output only `_segment_ops` and
  `emit_reduce` consumed — elementwise emission already ignored it in favor
  of canonical row-major). Kept the useful parts in `passes/layout.py`:
  the `Layout` (I/P/F) dataclass, `get_matmul_layouts` (the tensor-engine
  hard constraint), and `default_layout` (scored I|P|F split).
- [x] **Layout-free segmentation.** `_segment_ops` groups on collapsed-(P,F)
  shape compatibility only; the layout-flip break (which could only split
  groups spuriously, since emission never read the solved layouts) is gone.
- [x] **Reduce classifies axes locally** via `default_layout` of its input.
- [x] `lower_graph(graph)` / `lower_to_nki(graph)` no longer take or thread
  a `layouts` dict.

## Phase 1 — Fuse data movement via HBM views

Generalize `hbm_map` from name → buffer to name → buffer + access
descriptor, so pure data-movement ops become metadata that consumers' DMA
loads compose through (the view-reshape rebind is the existing prototype).

- [x] **broadcast_to as fold** (done differently than planned — no view
  descriptor needed). A `broadcast_to` feeding only elementwise binary ops is
  pure waste: the vector/scalar engine broadcasts a size-1 *free* dim natively
  (`tensor_scalar_arith`) and a partition-1 operand is fanned by the load, so
  the consumer can read the un-broadcast source directly. New tensor_ir
  peephole `passes/fold_broadcast.py` rewires collapse-safe broadcasts
  (source right-aligned + collapsed to 2D has P∈{1,rep_P}, F∈{1,rep_F}) to
  their source and drops the broadcast. Runs *before* decompose, so
  `div(a, broadcast_to(b))` → `div(a,b)` → `mul(a, reciprocal(b))` — this also
  removes the extra HBM round-trip that decompose.py flags as the source of a
  residual ~1/65536 floor-divide precision error. Middle broadcasts (GQA head
  expansion `(1,8,1,64)→(1,8,8,64)`) are *not* collapse-safe and stay
  materialized. qwen3 MoE: broadcast_to 105 calls/1438 ops → 7 calls/700 ops;
  total 42805 → 41633 nki ops (−2.7%). All 7 remaining are legit GQA middles.
  **Reverted for now** — `passes/fold_broadcast.py` and its test removed and
  dropped from the pipeline; revisit alongside the SBUF-first rework. Every
  `broadcast_to` is again materialized to HBM.
- [x] **Shared "materialize a tile from a viewed value" load helper.**
  `load_input_tile` / `store_output_tile` (+ the shared `canonical_layout`)
  in `direct_lower_utils.py` are now the single load/store path for
  elementwise emission. `_emit_ew_tile` previously carried two near-identical
  load branches (normal load vs. slice-view load-with-offset) inlined; both
  now call `load_input_tile`, which composes an optional per-dim `offsets`
  descriptor (the slice's `starts`) into the `DimSlice`s. The rep shape and
  tile-sizes no longer thread through `_emit_ew_tile` — each value tiles in its
  own canonical row-major layout mapped onto the rep loop, so the helper needs
  only `(shape, dtype, seg_dtype, indices, rep_layout, offsets)`. Behavior-
  neutral: qwen3 MoE stays at 39968 nki ops / 10601 dma_copy (same DMAs, one
  code path). The access-descriptor seam this creates is what a future
  transpose/reshape view composes through.
- [ ] **slice as view** (contiguous, static starts): implemented (d109c16) then
  **reverted** — every `slice` now materializes to its own HBM buffer via
  `emit_slice`, and elementwise consumers read that buffer normally. The
  offset-fold version (`slice_views` + `_resolve` in `lower_graph`, a
  `row_off`/`col_off` composed into `_emit_ew_tile`'s loads) carried a
  correctness bug in the 2D-window contiguity check and added lazy-materialization
  machinery whose only payoff was a −1.5% op count on qwen3 MoE (41633 → 40993
  nki ops). Removed for simplicity; revisit as an access-descriptor composed
  through the elementwise load path if the op/buffer savings prove to matter.
- [x] **compose chained transposes** (done ahead of the "transpose as view"
  item; higher payoff on the qwen3 graph). `transpose(transpose(x, p1), p2)` is
  a single `transpose(x, compose)` with `compose[i] = p1[p2[i]]` — permuting
  twice is permuting once. The chain materializes the intermediate through HBM
  (load → remap → store → reload); the composed form skips it, and DCE drops the
  inner transpose's whole materialization when it has no other consumer. New
  tensor_ir peephole `passes/fold_transpose.py`, runs before decompose beside
  `fold_broadcast`. The qwen3 attention path chains `(0,2,1,3)` then `(0,1,3,2)`
  on a (1,8,4096,128) tensor to feed QK^T — one of the two ~16 MB attention
  transposes folds away. qwen3 MoE: transpose 2864 → 2096 nki ops; total
  40993 → 39968 (−1025, −2.5%); dma_copy 11113 → 10601.
  **Reverted for now** — `passes/fold_transpose.py` and its test removed and
  dropped from the pipeline; revisit alongside the SBUF-first transpose rework.
- [ ] **batch-only transpose as view** (no P↔F swap): pure coordinate
  remapping on the consumer's slices, as `emit_transpose` already proves. (The
  two remaining qwen3 transposes both swap P↔F, so this needs a new benchmark
  to motivate — deferred until a batch-only-transpose hotspot shows up.)
- [ ] Re-run `profile_layer.py`; record op counts before/after.

## Phase 2 — Segments own the loop, ops are tile callbacks

Structural change enabling real fusion: the segment/group owns the tile loop
nest and all boundary DMA; each op contributes an SBUF-tiles → SBUF-tiles
callback (`emit_binary_op` / `emit_unary_op` already have this shape).
Fusion generalizes from "consecutive elementwise" to "ops that can share one
tile loop."

Prerequisite infrastructure:

- [ ] **Per-group SBUF budget accounting.** Sum actual per-partition bytes of
  a group's live tiles; split the group when over budget. Replaces the
  `max_free_elems = capacity/4` heuristic (which also lets small groups use
  bigger tiles).
- [ ] **Single fusion-compatibility predicate**, replacing the duplicated
  `_segment_ops` / `_collapse_ew_shape` logic, so grouping and emission
  provably agree. Segment-first and anchor-driven: the anchor op (matmul,
  reduce) dictates the group's tile layout; the predicate asks whether a
  follower can join the anchor's loop in that layout.
- [ ] **Layout at fusion boundaries.** A matmul epilogue tile arrives in the
  matmul output layout (P=M, F=N via `get_matmul_layouts`), not the canonical
  row-major layout the elementwise tile machinery assumes — epilogue
  callbacks must operate in the producer's tile layout. When a candidate
  follower needs the value in a different layout, the resolution is "break
  the segment" (spill through HBM), recorded explicitly rather than fused
  wrong.

Fusions, in order of expected payoff (re-profile after Phase 1 to confirm):

- [ ] **Matmul epilogue fusion**: apply the consumer elementwise chain (bias,
  activation, residual, cast) to the PSUM→SBUF tile before the store.
- [ ] **Reduce + elementwise fusion** (RMSNorm, softmax): reduce anchor keeps
  the source tile resident, follow-up elementwise applies to the same tile —
  one load, one store instead of two round-trips.

## Explicitly deferred

- Cross-segment SBUF residency (graph-level SBUF cache with liveness and
  spilling) — a real scheduler; revisit only if post-Phase-2 profiles show
  it matters. This is also the point where a *global* layout pass becomes
  meaningful: once a value stays SBUF-resident between segments, the
  producer's stored layout constrains the consumer, and layout conflicts
  become real decisions (insert a conversion vs. break residency). Build it
  then as constraint propagation with explicit conversion points — not
  first-assignment-wins.
- Loop constructs in the IR (unrolled emission is a scalability ceiling, not
  a bandwidth cost).

---

## Progress log

| Date | Change | qwen3 layer ops | Notes |
|------|--------|-----------------|-------|
| 2026-07-05 | Baseline (plan created) | 80332 nki (17.3x), 23017 dma_copy | L=8; top: gather_rows 20416, matmul 11504, elementwise 10486 (3685 dmas), concat 7330 |
| 2026-07-05 | Phase 0: dead-store elim + dtype-aware ew tiles | 79932 nki (17.2x), 22833 dma_copy | elementwise 10270 (3501 dmas); all-F32 model so dtype-aware tiling is a no-op here |
| 2026-07-05 | Phase 0.5: packed gather + splat concat | 42805 nki (9.2x), 11577 dma_copy | gather_rows 20416→1920, elementwise →3274, concat →6739; indirect DMAs 6720→128 |
| 2026-07-05 | HW benchmark, 30B MoE TP=4, n=64 | TTFT 3280 ms (was 4596), decode p50 2333 ms (was 2500) | 1.40x TTFT, 1.07x decode vs same-day pre-branch baseline (964b5ff), same box, fresh builds. NOTE: the repo's benchmark_report.json (07-01: TTFT 3309 / p50 1625) predates the basic-lowering-cleanup merge and is NOT a valid before-point — decode regressed ~1.5x at that merge, independent of this branch. Decode profile (L=1): 17984→14761 nki ops, but decode's hot gather/concat fixes barely apply per-token; matmul+transpose dominate. |
| 2026-07-05 | Phase 1: fold broadcast_to into elementwise consumers | 41633 nki (9.0x), 11369 dma_copy | broadcast_to 105 calls/1438 ops → 7 calls/700 ops (only GQA middles remain); elementwise unchanged (native F/P broadcast). Fold runs pre-decompose, also fixing the div-denominator precision round-trip. |
| 2026-07-06 | Phase 1: slice as view (offset fold + JIT materialize) | 40993 nki (8.8x), 11113 dma_copy | slice attribution 1149 ops/714 dmas → 0 (folded into EW consumers or materialized off the profiled wrapper); elementwise 3240→3264 (composed offset loads). Foldable slices: static-start, stride-1. All 695 tensor_ir tests + 11 new slice-view tests (7 interp, 4 HW) pass. |
| 2026-07-06 | Phase 1: compose chained transposes | 39968 nki (8.6x), 10601 dma_copy | transpose 2864→2096 ops (1056 dmas); one of the two 4096-wide attention transposes folds away (`(0,2,1,3)`∘`(0,1,3,2)` → `(0,2,3,1)`). New `fold_transpose.py` peephole. 6 interp + 2 HW fold tests pass; transpose+lowering HW suites (35 tests) green on trn2. |
| 2026-07-06 | Phase 1: shared load/store tile helper | 39968 nki (8.6x), 10601 dma_copy | Refactor only — `load_input_tile`/`store_output_tile`/`canonical_layout` in `direct_lower_utils.py` are the single elementwise load/store path (was two inlined branches in `_emit_ew_tile`); slice-view offset is now an optional `offsets` descriptor. Op count unchanged (behavior-neutral). All 702 tensor_ir tests pass (1 flaky HW-compile contention, green in isolation). |
