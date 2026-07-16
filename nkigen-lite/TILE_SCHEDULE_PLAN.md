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

## Step 2 — Introduce `Scratch`

- [ ] `Scratch` handle wrapping `nb`: `.sbuf(shape, dtype)`, `.hbm(shape, dtype)`,
  `.load(hbm_2d, slices, dtype)` (fuses the ubiquitous
  `nb.dma_copy(nb.alloc(...), src, slices)` idiom).
- [ ] Migrate the 11 scattered `nb.alloc(..., HBM)` scratch sites (topk×6,
  reduce, transpose, collective×2, memory, utils `broadcast_partition`).
- [ ] Tests green.

## Step 3 — Body-split emitters (opportunistic)

- [ ] Reshape each regular emitter into plan-once + `for tile in sched: body(...)`.
- [ ] Lowest urgency; do while touching each emitter.

---

## Progress log

- 2026-07-15: Branch `lowering-tile-schedule` created; plan written.
- 2026-07-15: Step 1 complete. `TileSchedule`/`TileIndex` extracted; all
  `iter_pf_tiles` callers + all five reduce loop-nests migrated; three old free
  functions deleted; equivalence test added. Full tensor_ir suite green (794
  passed). Next: Step 2 (`Scratch`).
