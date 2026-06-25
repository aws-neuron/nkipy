# Scatter-family design for nkigen-lite

Status: **IMPLEMENTED.** Companion to the gather-family work in commits
`e3b5c93` / `8ed7979`. This document scoped the *scatter* half of the indexing
gap; the design below is now built. Summary of what landed:

- `scatter_rows` / `gather_rows` tensor_ir primitives → indirect-DMA store/load
  (`dma_copy_indirect`), with partition tiling and (M,1) U32 indices.
- Frontend `scatter_along_axis`, `put_along_axis`, `scatter_strided` (all
  registered for nkigen-lite), normalizing onto `scatter_rows`.
- A row-gather fast path in dynamic `take` (axis 0) via `gather_rows`, so tall
  tables (embedding (128256, 2048)) no longer transpose-and-OOM.
- HW tests: `nkigen-lite/tests/tensor_ir/test_scatter.py` (scatter_rows +
  gather_rows, incl. N>128 / M>128 / duplicate indices).

Unblocked on hardware: `test_put_along_axis(_scalar_value)`,
`test_slice_assignment(_indeterministic)`, `test_step_slicing_assignment`,
`test_rotary_embed`, and the `embedding_dynamo` kernel.

Known still-failing (NOT scatter, out of scope):
- `test_view_assignment_semantics` — pre-existing view-aliasing bug; the test
  `pytest.skip`s on hlo, so there is no hlo behaviour to match.
- `llama_decoder_dynamo` — OOMs in the **LM-head matmul** (`(1,2048) @
  (2048,128256)` → a 128256-wide free dim), reproducible with a 3-line matmul
  and no indexing. A matmul free-dim tiling limit, unrelated to the scatter
  family.

The original design discussion follows.

## Motivation

After the gather family closed, the remaining nkigen-lite indexing xfails are
all scatter:

| Op (frontend) | Tests blocked | Frontend entry |
|---|---|---|
| `scatter_along_axis` | `test_slice_assignment` (3), `test_slice_assignment_indeterministic` (1), + `rotary_embed`'s `x_out[...] = ...` (2) | `__setitem__` with a tensor index → `_do_scatter_indexing` |
| `put_along_axis` | `test_put_along_axis` (3), `test_put_along_axis_scalar_value` (3) | `np.put_along_axis` direct |
| `scatter_strided` | `test_step_slicing_assignment` (1) | `__setitem__` with a `step>1` slice → `_do_scatter_strided_assignment` |
| (view-aliasing bug) | `test_view_assignment_semantics` (1) | `__setitem__` static slice → `dynamic_update_slice` (already registered) |

`dynamic_update_slice` is the *only* scatter-like op already on nkigen-lite
(slice + concat reconstruction, no hardware scatter). All others raise
`NotImplementedError` (auto-xfail).

## The load-bearing question — and the answer

The gather family mapped 1:1 onto `nisa.gather`, which worked on hardware the
first time. **Scatter has no such clean primitive.** The only candidate is
`dma_copy_indirect` (store direction). I validated it directly; findings:

1. **It is row-indexed, not flat-element-indexed.** The KB API computes
   `dst_indirect_max_index = dst.tile.shape[0]`, i.e. the index vector selects
   *rows* of the destination: `dst[index[r], :] = src[r, :]`. It is the exact
   mirror of `gather`, not a general `np.put(flat, idx, vals)`. The numpy
   interpreter in `nki_ir/interpret.py` (which uses `np.put` on a flattened
   buffer) is therefore **more permissive than the hardware** and would pass a
   flat-scatter test that cannot actually lower. Do not trust the interpreter
   alone here.

2. **It does not currently lower.** Both a flat-scatter and a row-scatter
   hardware attempt failed at:
   ```
   emit_to_kb.py:610: cannot align DMA operands: source has N elements but
   destination has M; ensure source and destination tile shapes have matching
   element counts
   ```
   The nkigen-lite emit path (`emit_to_kb.py` ~598-610) calls
   `nisa.dma_copy_indirect(dst=…, src=…, dst_index=…)` but the operand is not
   set up with the `vector_offset_coeff=1` / `prepare_operand` access pattern
   that the KB `isa.dma_copy_indirect` (isa.py:373) needs for the indirect
   addressing — so the DMA tiler treats it as a plain copy and rejects the
   element-count mismatch.

3. **It is entirely unproven.** `grep` finds zero uses of `dma_copy_indirect`
   in nkigen-lite lowering or tests. It has a Builder method, an interpreter
   case, and an emit case, but the chain has never executed end-to-end.

**Conclusion:** unlike gather, the scatter primitive needs a *primitive-level
fix first*. The KB integration in `emit_to_kb` must be corrected (and validated
on hardware) before any frontend op can rely on it. This is increment 0 below
and is the principal risk.

## BIR evidence: how the (proven) XLA scatter actually lowers

To de-risk increment 0, I compiled real scatter kernels through `neuronx-cc`
(`--pipeline compile SaveTemps`, plus `--enable-dge`) and read the Backend IR
(`sg00/bir.json`). This is the *proven* HLO path, so its BIR is the reference
for what nkigen-lite must reproduce.

A `put_along_axis(b, idx, vals, axis=1)` with **runtime** `idx` (kernel input,
not a constant) lowers to two BIR instructions, both tagged
`op_name: hlo__scatter_op11`:

1. `I-34` `DMACopy`: full base copy `a → output0`, plain affine addresses,
   `access_shape: [128]` (the 8×16 operand, **flattened to 1-D**).
2. `I-61` `DMACopy` — the scatter itself:
   ```
   ins:  [ values (float32, 24 elems), indices (int32, 24 elems) ]
   outs: [ output0  addrs:[{"kind":"IndirectArgId","arg_id":1}]  access_shape:[128] ]
   dge_type: <set>
   ```
   The destination address is **`IndirectArgId` over operand `arg_id 1` (the
   index tensor)** — runtime-computed addresses. This *is* indirect-DMA scatter,
   the BIR realization of `nisa.dma_copy_indirect` store direction, routed
   through the Dynamic Gather Engine (`dge_type`).

Control: with **constant** indices, XLA const-folds the scatter to static
`DMACopy`s with no `IndirectArgId`. The indirect form only appears for genuinely
dynamic indices, and `--enable-dge` must be present.

### Two corrections to the assumptions above

1. **Indirect-DMA scatter is real and hardware-proven.** The failure in
   "It does not currently lower" is therefore a *wiring bug in nkigen-lite's
   `emit_to_kb`*, not a missing hardware capability. The concrete fix target:
   emit a `DMACopy` whose destination address is an `IndirectArgId` over the
   index operand (the KB `prepare_operand(..., vector_offset_coeff=1)` path),
   **and add `--enable-dge` to the nkigen-lite compile args** (check
   `compile.py` / CompileOptions; the gather path does not need it but indirect
   scatter does).

2. **The BIR addressing is flat/linearized, not strictly row-indexed.** XLA
   flattened (8,16)→[128] and scattered 24 elements by flat index — matching the
   HLO `put_along_axis` strides trick. So a **flat-element** indirect scatter is
   achievable at the BIR level, even though the nki Builder wrapper currently
   constrains the index to the outer dim (`dst_indirect_max_index =
   dst.tile.shape[0]`). Re-examine whether that wrapper constraint — not the
   hardware — is the real blocker; a flat scatter may be expressible directly.

### Gather and scatter are NOT symmetric in hardware

Compiling the working nki `gather_along_axis` and reading its BIR shows it uses
a dedicated **`Gather`** opcode (SBUF-resident `nisa.gather`) plus plain
`DMACopy` staging — **no `IndirectArgId`**. Gather has an on-chip compute-engine
primitive; scatter goes through indirect-address DMA to HBM. Do not assume a
scatter emitter can be a mirror image of `_emit_gather_along_axis_op` — the
mechanisms differ.

## Increment plan

### Increment 0 — make `dma_copy_indirect` store actually work (PREREQUISITE) — **PROVEN ON HARDWARE**

Pure nki_ir-level task; no frontend, no tensor_ir. **A working prototype now
exists** (`emit_to_kb.py` store branch) and produces correct results on
Trainium. The fix and the gotchas found while proving it:

**The fix.** The old store emit called the *low-level*
`nisa.dma_copy_indirect(dst=full_tile, src=tile, dst_index=index)` directly,
passing the full HBM `dst` tile — the DMA tiler then saw `src.size != dst.size`
and rejected it. The working approach uses the **canonical indexed-view idiom**
(the same `.ap(vector_offset=)` already used for the broadcast/AP op in this
file), then a plain `dma_copy` which auto-routes to the indirect path:

```python
free = prod(dst.shape[1:])
m_rows = src.shape[0]                       # number of scattered rows = index length
dst_view = dst.ap([[free, m_rows], [1, free]], vector_offset=index)
nisa.dma_copy(dst=dst_view, src=src)        # routes to dma_copy_indirect
```

Key insight: the **view shape must match `src` (M scattered rows × free), not
the full dst (N rows)**. `.ap` derives the row stride (`free`) and the bound
(`indirect_max_index = dst.shape[0] = N`) from the full dst tile, while the
index tile supplies one row selector per scattered row.

**Gotchas discovered (must hold for any caller):**
- **Index tile must be 2-D `(M, 1)`, not 1-D `(M,)`.** A 1-D SBUF index tile
  fails compilation: `'nisa.bind_memloc' op 1D tensors are not supported in
  SBUF`. The tensor_ir lowering must allocate the index as `(M, 1)`.
- **`vector_offset` partition stride must equal the free-dim size** (the `.ap`
  validation enforces `pattern[0][0] == prod(shape[1:])`).
- `--enable-dge` was **not** required in this prototype (default DGE selection
  sufficed for SWDGE). Re-confirm when wiring through `compile.py`; the XLA path
  used it but the KB path auto-selects.

**Still open before declaring increment 0 done:**
- Generalize to N>128 / M>128 (partition tiling), as the gather emitter does.
- Characterize duplicate-index semantics on hardware (last-write-wins vs
  undefined). `test_slice_assignment_indeterministic` only needs *some* valid
  source to win, so either is acceptable — but characterize, don't assume.
- Add a permanent hardware test analogous to `tests/tensor_ir/test_gather.py`
  (the prototype was validated ad hoc: scattering rows into an (N,W) buffer,
  verified against numpy — correct).

**Status:** exit criterion (a row-scatter nki_ir graph matches numpy on
hardware) is **met** by the prototype. The fallback below is therefore unlikely
to be needed, but kept for the N>128 / duplicate-index edge cases.

> Note: this is a **row**-scatter (`dst[index[r], :] = src[r, :]`), the mirror
> of `gather_along_axis`. The XLA BIR showed a *flat* element scatter; the KB
> `.ap(vector_offset=)` path is row-granular. For `scatter_along_axis` along the
> free axis, the frontend must transpose so the scattered axis becomes the
> partition/row axis (inverse of the gather wrappers) — or a flat reshape per
> the put_along_axis strides trick. Decide in increment 1.

### Increment 1 — `scatter_along_axis` tensor_ir op + lowering

This is the highest-value op (unblocks the `__setitem__` tensor-index path and
the most tests). Semantics (numpy `put_along_axis` along one axis, assign):
`out = x.copy(); out[..., idx[..., i], ...] = vals[..., i, ...]`.

- **tensor_ir Builder** `scatter_along_axis(data, idx, updates)` — 2-D
  per-partition form mirroring `gather_along_axis`: `out[p, idx[p,i]] =
  updates[p,i]`, `out` shape == `data` shape. Validate ranks/partition match,
  U32 idx.
- **Interpreter** in `tensor_ir/ir.py`: per-partition `np.put_along_axis` (or an
  explicit loop) — but write it to match the *hardware* row/column semantics
  proven in increment 0, not a looser numpy model.
- **Lowering** `_emit_scatter_along_axis_op` in `direct_lower.py`, templated on
  `_emit_gather_along_axis_op`: copy `data` HBM→result HBM (the unchanged base),
  load updates+idx to SBUF, scatter into the result via the increment-0
  primitive, with PARTITION_MAX chunking.
- **Frontend** `_nkigen_lite_impls.scatter_along_axis(arr, indices, values,
  axis)`: the same transpose-to-free-axis + flatten-to-(P,F) normalization used
  by `take_along_axis`, then the 2-D op, then reshape/transpose back. Register
  in `_register_nkigen_lite.py`.

Note the axis convention: the `gather`/`scatter` hardware primitives index the
**free** axis per partition. The `__setitem__` path scatters along the
*partition-ish* axis (`a[:, t, :]`), so the frontend must transpose so the
scattered axis becomes free — same move as gather, just inverted.

### Increment 2 — `put_along_axis`

`np.put_along_axis(a, idx, vals, axis)` is `scatter_along_axis` with numpy's
broadcasting of `vals` and support for a scalar `vals`. Once increment 1 exists
this is a thin frontend wrapper:
- materialize scalar `values` via `full(idx.shape, v)`;
- broadcast `values` to `idx` shape;
- delegate to `scatter_along_axis`.
Handles `axis=None` (flatten) like the gather wrappers.

### Increment 3 — `scatter_strided`

`a[::s, ::s] = b`. The frontend already lowers the strided slice to an explicit
per-dim index list (`_do_scatter_strided_assignment` →
`scatter_strided(self, value, scatter_indices_per_dim)`), and HLO expands it to
the cartesian product of positions. For nkigen-lite the cleanest route is
**not** the indirect DMA at all: strided assignment is static (indices known at
trace time), so reuse the existing slice/concat machinery —
`dynamic_update_slice`-style — to write the strided positions without any
hardware scatter. Lowest risk; do last.

## Fallback if increment 0 fails

If `dma_copy_indirect` store cannot be made to lower on hardware, scatter can
still be implemented without a hardware scatter, at a cost:

- **iota + select fallback** for `scatter_along_axis`: for each of the F target
  columns, compare a broadcast index tile against `iota` and `select` the update
  vs the original (O(F) selects per tile, O(F²) work). Correct and uses only
  proven primitives (`iota`, `affine_select`/`select`, already exercised), but
  slow for wide axes.
- **slice/concat** for the static cases (`scatter_strided`, contiguous
  assignment) — already how `dynamic_update_slice` works; no new primitive
  needed.

This guarantees the tests can pass even if the indirect DMA never works; the
indirect path is purely a performance upgrade.

## Explicitly out of scope

`test_view_assignment_semantics` is a **pre-existing correctness bug** (verified
by stashing the gather changes: it fails identically without them). It is about
*view aliasing* — `view = a[0:5,:]; view[1:3,2:4] = b` not propagating back to
`a` — and lives in the frontend `__getitem__`/`__setitem__` parent-link logic
(`tensor.py` docstring already notes "Mutations through views are NOT
tracked"), not in any scatter primitive. Track it separately.

## Risk summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| `dma_copy_indirect` store never lowers | medium | iota+select / slice-concat fallback (correctness preserved, perf lost) |
| Interpreter over-permissive vs hardware | high (confirmed) | every scatter increment must have a hardware test, not just interpreter |
| Duplicate-index semantics differ from numpy | medium | `indeterministic` test only needs *a* valid winner; characterize on HW in inc 0 |
| Partition tiling for scattered axis | low | same pattern as gather, already proven |

## Sequencing

`inc 0 (primitive, HW-validated)` → `inc 1 (scatter_along_axis)` → `inc 2
(put_along_axis)` → `inc 3 (scatter_strided, static/no-primitive)`. Each
increment ships with `nkigen-lite/tests/...` hardware coverage before flipping
the corresponding parent-suite xfails. Do **not** flip any xfail on interpreter
evidence alone.
