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

## Phase 1 — Fuse data movement via HBM views

Generalize `hbm_map` from name → buffer to name → buffer + access
descriptor, so pure data-movement ops become metadata that consumers' DMA
loads compose through (the view-reshape rebind is the existing prototype).

- [ ] **Shared "materialize a tile from a viewed value" load helper.**
  All emitters load through one helper that composes the access descriptor
  into the `DimSlice`s. This is also the 2D-(P,F)-normalization consolidation
  from the review — collapse-to-2D becomes the normal form, not a per-op
  fast path.
- [ ] **slice as view** (contiguous, static starts): base-offset added to the
  consumer's slices; no load-compute-store sequence emitted.
- [ ] **broadcast_to as view**: stride-0 `DimSlice` on broadcast axes (the DMA
  engine already supports this — see `broadcast_partition` and the collapsed
  broadcast path).
- [ ] **batch-only transpose as view** (no P↔F swap): pure coordinate
  remapping on the consumer's slices, as `emit_transpose` already proves.
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
  provably agree.

Fusions, in order of expected payoff (re-profile after Phase 1 to confirm):

- [ ] **Matmul epilogue fusion**: apply the consumer elementwise chain (bias,
  activation, residual, cast) to the PSUM→SBUF tile before the store.
- [ ] **Reduce + elementwise fusion** (RMSNorm, softmax): reduce anchor keeps
  the source tile resident, follow-up elementwise applies to the same tile —
  one load, one store instead of two round-trips.

## Explicitly deferred

- Cross-segment SBUF residency (graph-level SBUF cache with liveness and
  spilling) — a real scheduler; revisit only if post-Phase-2 profiles show
  it matters.
- Loop constructs in the IR (unrolled emission is a scalability ceiling, not
  a bandwidth cost).

---

## Progress log

| Date | Change | qwen3 layer ops | Notes |
|------|--------|-----------------|-------|
| 2026-07-05 | Baseline (plan created) | 80332 nki (17.3x), 23017 dma_copy | L=8; top: gather_rows 20416, matmul 11504, elementwise 10486 (3685 dmas), concat 7330 |
| 2026-07-05 | Phase 0: dead-store elim + dtype-aware ew tiles | 79932 nki (17.2x), 22833 dma_copy | elementwise 10270 (3501 dmas); all-F32 model so dtype-aware tiling is a no-op here |
| 2026-07-05 | Phase 0.5: packed gather + splat concat | 42805 nki (9.2x), 11577 dma_copy | gather_rows 20416→1920, elementwise →3274, concat →6739; indirect DMAs 6720→128 |
