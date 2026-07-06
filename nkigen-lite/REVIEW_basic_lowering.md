# Review: `tensor_ir/passes/basic` direct lowering

Review of the basic lowering flow (2026-07-05). Findings ordered by priority;
check items off as they're addressed. Repro snippets are included so each fix
can start from a failing case.

## Confirmed bugs

### 1. [ ] Transpose passthrough-partition path crashes on unaligned collapsed B

`direct_lower_transpose.py:227` (`_lower_transpose_passthrough_partition`) and
the duplicated code in `emit_transpose` (`direct_lower_transpose.py:464`).

The fast path assumes each 128-row tile of the collapsed leading dim `B` is a
single source rectangle: `(src_slices, _), = src_chunks`. When `B` collapses
from multiple dims and 128 doesn't divide at a leading-dim boundary,
`flat_range_to_src_chunks` returns several chunks and the unpack raises.

Repro (through the full pipeline):

```python
x = b.add_input('x', (2, 100, 8, 64), DType.F32)
y = b.transpose(x, (0, 1, 3, 2))   # collapses to B=200, I=8, J=64
# lower_graph → ValueError: too many values to unpack (expected 1)
```

Fix direction: loop over all chunks (splitting the on-chip tile per chunk),
mirroring `_prefix_row_segments` in `direct_lower_memory.py`, which solved the
same problem for reshape.

### 2. [ ] Comparison/bitwise ops don't support free-dim broadcast

`direct_lower_utils.py:445-470` (`emit_binary_op`). The compare and bitwise
branches only handle partition broadcast (`broadcast_partition` when the
partition extent is smaller); a free-size-1 operand falls through unchanged and
`tensor_tensor_compare` rejects it.

```python
z = b.greater(x_128x64, y_128x1)
# → ValueError: tensor_tensor_compare: shapes must match, got (128, 64) vs (128, 1)
```

This matters because `_collapse_ew_shape` explicitly admits `F == 1` operands
("per-row scalar broadcast") and relies on `emit_binary_op` to align them —
which works for `add`/`mul` but not `greater`/`equal`/etc.

Fix direction: materialize the free-axis broadcast before comparing (e.g. the
ones-multiply `tensor_scalar_arith` pattern used in `_lower_f_broadcast`).

### 3. [ ] P-reduce emits SBUF/PSUM tiles exceeding hardware limits when F is wide

Both P-reduce strategies take the free extent at full width with no cap:

- `_emit_p_reduce_inline` (`direct_lower_reduce.py:988`):
  `inp_tile_sizes[d] = inp_shape[d]` for all f_dims. A `(256, 131072)` f32 sum
  over axis 0 lowers without complaint but `graph.verify()` reports
  `SBUF tile 524288 bytes exceeds per-partition capacity 180224`.
  Same pattern in `lower_p_reduce_gpsimd` and the mixed-reduce paths.
- `lower_p_reduce_matmul` (`direct_lower_reduce.py:715`): allocates
  `psum = nb.alloc((1, f_extent), ...)` unconditioned — `f_extent = 1024`
  already yields `PSUM free size 1024 exceeds max 512`.

Repro:

```python
x = b.add_input('x', (256, 131072), DType.F32)
y = b.reduce(x, axis=(0,), kind='sum', keepdims=True)
# lower_graph succeeds; graph.verify() → SBUF per-partition overflow
```

Fix direction: tile F at `max_free_elems(dtype)` (gpsimd path) /
`PSUM_FREE_MAX` (matmul path) with an outer F loop; the elementwise path's
`compute_tile_sizes` already shows the pattern.

### 4. [ ] `_first_cols` reads out of bounds for k in 9..15 (any k>8 not a multiple of 8)

`direct_lower.py:1061-1070`. The scratch is hard-coded `(P, 8)` but callers
pass `kp`-wide tiles (`kp = ceil(k/8)*8`). For `k = 12`: tile is `(P, 16)`,
only columns 0–7 reach the scratch, then `DimSlice(0, 12)` reads 12 columns
from an 8-wide HBM buffer. The builder does NOT validate slice bounds (checked:
an OOB `dma_copy` is accepted silently), so this produces garbage in columns
8–11 rather than an error.

Fix: size the scratch `(P, tile.type.shape[1])` and copy the full width.
Note: there is currently **zero topk test coverage** — add tests for
k = 3, 8, 12, 20 (and the F > TOPK_FREE_MAX chunked-merge path).

## Design / robustness

### 5. [ ] Dedup `lower_*` vs `emit_*` — divergence is already happening

Nearly every module carries two copies of the same tiling logic: a standalone
`lower_*` building its own graph (used only by unit tests) and an `emit_*`
used by the orchestrator. Confirmed drift:

- `emit_reduce` has the collapsed-F fast path (`_try_emit_collapsed_f_reduce`);
  `lower_f_reduce` doesn't.
- `emit_slice` supports strides + the axis-window fast paths; `lower_slice`
  raises on strides.
- `emit_reshape` has the flat-copy overflow guard; `lower_reshape` doesn't.
- `direct_lower_elementwise.py` is entirely legacy — nothing outside its own
  test imports it.

Fix direction: make `lower_*` a thin wrapper (create Builder + HBM inputs,
call `emit_*`, set outputs); delete `lower_elementwise`. This would have
prevented at least two of the divergences above.

### 6. [ ] `where` lowering is numerically unsafe for masked-softmax patterns

`direct_lower.py:554-575`. `cond*x + (1-cond)*y` produces NaN whenever the
*unselected* branch is `inf`/`-inf`/NaN (`0 * inf = NaN`). Attention masks
(`where(mask, scores, -inf)`) are exactly this shape. A select-style primitive
or the compare-XOR-select pattern used in `_emit_floor` (integer masks, no
0*inf) would be safe.

### 7. [ ] `_emit_topk_op` doesn't tile P

`direct_lower.py:831` uses `P` straight from the source shape in `(P, 8)`
allocs; anything over 128 rows overflows the partition. `gather_along_axis`
right next to it tiles P at `PARTITION_MAX`; topk should do the same.

### 8. [ ] `_emit_iota_op` doesn't cap tile_f

`direct_lower.py:749`: `tile_f = shape[-1]` with no `max_free_elems` guard —
a vocab-wide iota `(1, 128256)` would blow the per-partition SBUF budget.
One-line fix, same as elsewhere.

### 9. [ ] `_split_on_layout_conflict` condition possibly wrong (or unnecessary)

`direct_lower.py:312-313`: a conflict is flagged only when *both* p_dims *and*
f_dims differ. An input whose partition assignment alone differs (same f_dims)
passes as compatible and is then loaded with a canonical layout that may
disagree with its producer's. Since HBM is addressed logically this may be
intentionally safe — but then the whole check may be unnecessary; if not, the
condition should be `or`. Decide and document either way.

## Minor

### 10. [ ] Dead code: `flat_range_to_src_slices`

`direct_lower_utils.py:72` — superseded by `flat_range_to_src_chunks`,
referenced only in that function's docstring. Delete.

### 11. [ ] Undefined-name annotation `NisaReduceOp`

`direct_lower_reduce.py:202` annotates `reduce_nki_op: NisaReduceOp` — only
importable as `nki_ir.NisaReduceOp`. Harmless under
`from __future__ import annotations`, but breaks runtime introspection.

### 12. [ ] Transpose fast-path uses its own SBUF budget constant

`direct_lower_transpose.py:186-188`: hard-coded `49152 // bytes_per_elem`
instead of `max_free_elems`. Two notions of "fits in SBUF" in one package will
eventually disagree; unify on `max_free_elems`.

### 13. [ ] `emit_matmul` assumes F32 output dtype

The PSUM→SBUF copy is always F32 regardless of the result dtype; a bf16 matmul
result would be a silent dtype mismatch at the final `dma_copy`. Fine if matmul
results are F32 by construction — add an assert to make the contract explicit.
