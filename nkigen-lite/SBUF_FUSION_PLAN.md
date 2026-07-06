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

- [ ] **Dead-store elimination in elementwise segments.**
  `lower_graph` pre-allocates an HBM buffer for every op result and
  `_emit_ew_tile` stores every result, including segment-internal
  intermediates nobody reads from HBM. Compute liveness against the
  segmentation (consumed by a later segment or a graph output?); skip the
  HBM allocation and the store for segment-internal values.
- [ ] **Dtype-aware tile sizes for elementwise.**
  Elementwise never passes `dtype` to `compute_tile_sizes`, so bf16 segments
  tile at the conservative F32 budget — half the achievable tile width.

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
| 2026-07-05 | Baseline (plan created) | — | fill in from profile_layer.py |
