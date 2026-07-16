# Tile Schedule / Scratch / Body-Split Plan

Goal: separate the three concerns currently braided in every direct-lowering
loop body — **tiling**, **allocation**, and **codegen** — without fighting the
intrinsic coupling (tile shape is a scheduling output, so alloc stays fused to
tiling; only the *artificial* alloc↔codegen coupling is removed).

Context: `src/nkigen_lite/tensor_ir/passes/basic/` — every emitter loop-nest is
the same three-part object spelled differently: per-dim tile sizes, an
iteration over `ceildiv(extent, tile)`, and a clamped slice + extent at each
index. Reduce (`direct_lower_reduce.py`) already reinvented a general N-D tile
schedule privately (`_nested` + `loop_dims` + `build_slices` + `clamped_extent`).
This plan promotes that to a shared type and re-points the other emitters at it.

## Target layering

| Layer | Knows about | Never touches |
|---|---|---|
| `TileSchedule` | shapes, tile sizes, indices | `nb`, ops, dtypes-as-buffers |
| `Scratch` | `nb.alloc`, memory spaces | tiling policy, op semantics |
| emitter body | `nb.<compute>`, op semantics | `ceildiv`/`range`/`min`, raw `nb.alloc` |

Tiling→alloc and tiling→codegen stay coupled through the `TileIndex` passed into
the body — the honest, intrinsic coupling. Deallocs remain the existing
post-pass (`insert_deallocs.py`); `Scratch` governs allocation only.

## Fit assessment (done during design)

- **General reduce `_nested`** — recursive N-D nest; *defines* the abstraction.
- **elementwise / hbm-copy / broadcast** — regular 2D `(P,F)`; `TileSchedule.pf`.
- **topk / gather / scatter** — outer row tiling fits a schedule; inner body is
  data-dependent, stays hand-written. Adopt `Scratch` + body-split, not
  schedule-driven iteration.
- **reshape rectangle-splitting** (`flat_range_to_src_chunks`,
  `prefix_row_segments`) — this is *layout remapping*, NOT tiling. Must stay its
  own helper; forcing it into `TileSchedule` would be the design's failure mode.

---

## Step 1 — Extract `TileSchedule`  ✅ DONE

- [x] New module `direct_lower_schedule.py` with `TileSchedule` + `TileIndex`.
  - `TileIndex.slices(shape, tile_sizes)` ≡ old `build_slices`.
  - `TileIndex.extent(dims, shape, tile_sizes)` ≡ old `clamped_extent`.
  - `TileSchedule.__iter__` generalizes reduce's `_nested` recursion.
  - `TileSchedule.pf(P, F, dtype)` + `.pf_tiles()` ≡ old `iter_pf_tiles` policy.
- [x] Equivalence test `test_direct_lower_schedule.py` pins TileSchedule against
  inlined copies of the three old free functions (65 cases). Green.
- [x] Re-pointed all 7 `iter_pf_tiles` callers (direct_lower, broadcast×3,
  gather, memory×2) at `TileSchedule.pf(...).pf_tiles()`.
- [x] Re-pointed reduce: collapsed-f fast path, `_emit_f_reduce_inline`, and the
  three doubly-nested emitters (gpsimd p-reduce, matmul p-reduce, mixed). The
  load-bearing `if accum_loops: ... else: ...` branch (else path skips the
  partial+combine step) preserved exactly.
- [x] Deleted `iter_pf_tiles` / `build_slices` / `clamped_extent` from
  `direct_lower_utils.py` — no remaining callers.
- [x] `uv run pytest tests/tensor_ir -n auto`: **794 passed, 1 xfailed**.

Note: elementwise (`direct_lower_elementwise.py`) still hand-rolls its 2D loop
with `_free_tile` (power-of-two free tile, capped at 512) — a *different* tile
policy than `pf` (`max_free_elems`). Left as-is: folding it onto TileSchedule
needs a `.free_pow2` constructor and would change generated IR, so it belongs in
Step 3 (body-split), not this equivalence-preserving step.

## Step 2 — Introduce `Scratch`  ✅ DONE

- [x] `Scratch` handle wrapping `nb` in `direct_lower_alloc.py`:
  `.sbuf(shape, dtype)`, `.hbm(shape, dtype)`, and
  `.load(src, slices, shape, dtype)` — the last fuses the ubiquitous
  `nb.dma_copy(nb.alloc(...), src, slices)` idiom (57 sites).
- [x] Threaded ONE `Scratch` per graph from `lower_graph` through the emitter
  dispatch. Every dispatched emitter now takes a trailing `scratch`; standalone
  `lower_*` wrappers construct a local one (public `emit_*` take `scratch=None`
  and default-construct, so the wrappers didn't need touching).
- [x] Migrated all HBM scratch sites through `Scratch.hbm` — the single audit
  choke-point for scratch HBM (the OOM lever from the qwen3 notes): collective×2
  (direct_lower), topk×2 (`_topk_scan`, `cand_*`), topk `_first_cols` /
  `_overlay_columns`, reshape `_emit_reshape_diff_f`, `broadcast_partition`.
- [x] `broadcast_partition` (buried in the `emit_binary_op` compute chain)
  self-constructs a local `Scratch` rather than threading through every arith
  path — equivalent, since `Scratch` is a stateless `nb` wrapper and the audit
  point is `Scratch.hbm` regardless.
- [x] Added `test_direct_lower_alloc.py` (4 cases: space correctness + load ==
  alloc+dma equivalence).
- [x] `uv run pytest tests/ -n auto`: **856 passed, 1 xfailed**. No new ruff
  F401s (removed the `MemorySpace`/`ceildiv` imports that went dead).

Deferred (not blocking): a handful of multi-line `nb.dma_copy(nb.alloc(...))`
loads in broadcast's `_emit_collapsed_broadcast` and topk's `_topk_scan` were
left un-collapsed — they still route SBUF through the builder directly rather
than `scratch.sbuf`. Equivalence-preserving; fold in during Step 3 body-split.
Bare non-load SBUF allocs (accumulators, `ones`, scale constants, PSUM) were
intentionally left as `nb.alloc`/`nb.constant`: they are compute outputs, not
scratch, so they don't belong to the allocation-audit surface.

## Step 3 — Body-split emitters (opportunistic)  ✅ DONE

- [x] Added `TileSchedule.free_pow2(P, F, free_max)` — the elementwise tile
  policy (partition at 128, free = largest power-of-two ≤ min(F, 512)). Now the
  single home of that policy (formerly `_free_tile`).
- [x] Folded the elementwise emitter's hand-rolled 2D loop onto
  `free_pow2(...).pf_tiles()`; deleted `_free_tile` and the dead
  `ceildiv`/`PARTITION_MAX` imports. The emitter was already plan-once
  (`load_plan`/`store_2d`) + `for tile: _emit_ew_tile`, so this completes its
  body-split shape.
- [x] Collapsed the 3 straggler `nb.dma_copy(nb.alloc(...))` loads in
  broadcast's `_emit_collapsed_broadcast` to `scratch.load` (incl. the stride-0
  fan-out and the 3D middle-broadcast slice) + the ones-multiply dst to
  `scratch.sbuf`.
- [x] Equivalence test `test_free_pow2_matches_old_elementwise_loop` (pins
  `free_pow2` against the old `_free_tile` + loop).
- [x] `uv run pytest tests/ -n auto`: **869 passed, 1 xfailed**. No new F401s.

Intentionally NOT migrated (matches the Step-3 fit assessment): the remaining
`range(ceildiv(...))` loops in gather (indirect-DMA, data-dependent), iota
(produce, not load), memory's flat-range chunking, and the generic
batch-unrolled broadcast/hbm-copy paths. These build per-dim slices via batch
`unravel` + start-offsets that the flat `tile_sizes` model doesn't express
without changing generated IR — bespoke iteration is correct there. Reduce and
elementwise (the regular tile-loop emitters) are the ones that belong on
`TileSchedule`, and both now are.

---

## Progress log

- 2026-07-15: Branch `lowering-tile-schedule` created; plan written.
- 2026-07-15: Step 1 complete. `TileSchedule`/`TileIndex` extracted; all
  `iter_pf_tiles` callers + all five reduce loop-nests migrated; three old free
  functions deleted; equivalence test added. Full tensor_ir suite green (794
  passed).
- 2026-07-15: Step 2 complete. `Scratch` added and threaded per-graph through
  every emitter; all HBM scratch + 57 load idioms routed through it; unit test
  added. Full suite green (856 passed).
- 2026-07-15: Step 3 complete. `free_pow2` constructor added; elementwise folded
  onto it; straggler broadcast loads collapsed; equivalence test added. Full
  suite green (869 passed). **All three steps done** — tiling, allocation, and
  codegen are now separated in the regular tile-loop emitters (reduce,
  elementwise), with `Scratch` the allocation seam across all emitters.
