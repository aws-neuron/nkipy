# Slice-as-view lowering for nkigen-lite

Status: **DONE** (commit `d109c16`, branch `nkigen-lite-fusion`). A
static-start, stride-1 `slice` no longer emits a load→store copy: it becomes a
zero-copy *offset view* that its elementwise consumer folds into its own DMA
loads. Part of the SBUF-fusion Phase 1 work (see
[../SBUF_FUSION_PLAN.md](../SBUF_FUSION_PLAN.md)); companion to the reshape and
`broadcast_to` view folds.

qwen3 MoE layer (L=8): **41633 → 40993 nki ops** (−1.5%), `dma_copy`
11369 → 11113. The `slice` attribution category drops to **zero** — every
foldable slice either composes into a consumer's load or is materialized off
the profiled path.

## The idea

A `slice` with static (compile-time) starts and unit stride is just a per-axis
window of its source. It preserves rank, so a consumer that already loads its
inputs through per-dimension `DimSlice`s reads exactly the sliced data by
**adding the slice's `starts` into its own load offsets**. No intermediate HBM
buffer, no load-compute-store round-trip.

```
gate, up = split(x, 2, axis=0)   # x: (384,) -> two (192,) slices
y = silu(gate) * up
```

Previously: `x` → copy `x[0:192]` to a buffer → copy `x[192:384]` to a buffer →
load both → compute. Now: the `mul`/`silu` segment loads `x[0:192]` and
`x[192:384]` **directly from `x`** by offsetting its tile loads by 0 and 192.
The two slice buffers and their copies vanish.

This is the MoE hot path — the gate/up `(384,)→(192,)` split runs ×128 in one
profiled layer — plus the top-k index slices, and it's why `slice` was 1149
nki ops / 714 DMAs before this change.

## How it works (`direct_lower.py`)

Foldable slices are lowered **lazily**, with just-in-time materialization as
the safety net:

1. **`_is_view_slice(op)`** — a slice is foldable iff every stride is 1.
   Non-unit strides can't compose as a pure offset (the consumer would also
   have to carry the stride), so those keep the existing `emit_slice` copy path.

2. **`slice_views`** (in `lower_graph`) — `{result_name: slice_op}` for every
   foldable slice. Their results get **no HBM allocation** and **no emission**
   in the segment loop.

3. **Elementwise consumer composes the offset.** `_emit_elementwise_segment`
   builds `slice_srcs = {name: (source_buffer, starts)}` for any slice-view it
   reads. In `_emit_ew_tile`, a slice-view input loads from `source_buffer`
   with each per-dim tile offset shifted by `starts[d]`:

   ```python
   slices = hbm_slices(val_shape, val_layout, val_tile_sizes, indices, rep_layout)
   slices = [DimSlice(s.offset + st, s.size, s.stride)
             for s, st in zip(slices, starts)]
   tile = nb.dma_copy(dst, source_buffer, slices)
   ```

   The tile layout is computed from the *slice's own* (output) shape, so it
   stays aligned with the segment's rep loop; only the base offset moves.

4. **`_resolve(name)`** — every *other* consumer materializes the slice once,
   on demand: it allocates the buffer, runs the normal `emit_slice` copy, and
   caches the result in `hbm_map`. It recurses through chained slice-views
   (a slice of a slice). Callers that trigger it:
   - any non-elementwise op reading the slice (reshape, matmul, reduce,
     transpose, concat, gather/scatter, collective) — resolved before the
     emitter runs, since those read `hbm_map[name]` directly;
   - a slice that is itself a **graph output**;
   - the **rank≥3 collapse path** (`_try_emit_collapsed_ew`), which cannot
     compose (see below).

## Why the collapse path materializes instead of composing

The rank≥3 elementwise fast path reshapes each input's HBM buffer to a 2D
`(P, F)` view with `nb.view(...)` before tiling. **KB refuses `.view()` on a
sliced tile** — verified on device:

```
ValueError: view() only works on full tile views.
For sliced views, use rearrange() instead.
```

So an offset view can't be reshaped-then-loaded there. `_try_emit_collapsed_ew`
therefore calls `_resolve` on any slice-view input **after** its bail checks
(so a non-collapsible segment still falls through to the generic path, which
*can* compose). The generic per-tile path is the only place the offset is
folded; everywhere else the slice is materialized. This is a correctness
boundary, not a heuristic.

## Validation

Interpreter alone is **not sufficient**: it treats views as plain numpy arrays,
so it never hits the `.view()`-on-a-sliced-tile error — that only shows up on
device. Tests live in
`nkigen-lite/tests/tensor_ir/test_lower_to_nki.py::TestSliceAsView`
(7 interpreter + 4 hardware) and cover:

- the gate/up split feeding `silu`/`mul` (the hot fold);
- a partition-axis slice `(256,128)→(192,128)` across multiple partition tiles;
- a mid-tensor last-axis column window;
- a slice as a **graph output** (materialize path);
- a slice feeding a **reshape** (materialize path);
- a **strided** slice (stays on the copy path);
- a **chained** slice-of-slice (inner materializes, outer composes).

Before designing this, the three needed KB view mechanisms — 1D offset HBM
sliced view → 2D tile, chained sub-indexing, and partition-axis offset views —
were each confirmed to compile and execute correctly on Trainium.

Full `nkigen-lite/tests/tensor_ir/` suite (695 tests) stays green.

## Reproduce the op counts

```bash
QWEN3_BACKEND=nkigen-lite uv run python examples/models/qwen3/profile_layer.py
```
