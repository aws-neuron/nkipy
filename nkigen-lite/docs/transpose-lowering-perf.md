# Transpose lowering performance for nkigen-lite

Status: **OPEN / NOT YET FIXED.** This documents a known performance cliff in
the transpose lowering (`tensor_ir/passes/basic/direct_lower_transpose.py`) and
the two hardware constraints that make the clean fix non-trivial. It is the
remaining bottleneck for the 1152-channel Qwen3-VL conv3d case, which stays
skipped via the `out_channels >= 512` guard in
`tests/unit/test_tensor_api.py::test_conv3d`.

Companion to the reshape fix in commit `2f8f706` (the *other* Qwen conv3d
bottleneck — the im2col weight reshape — which is now fast). Reshape and
transpose are the two data-movement ops conv im2col leans on; only reshape has
been optimized.

## Symptom

The conv im2col weight transpose moves the input-channel axis `Ci` from
position 1 to last so the weight flattens in the same `(kernel-position, Ci)`
order as the im2col columns:

```python
# _conv_nd in nkipy/.../_nkigen_lite_impls.py
perm = (0,) + tuple(range(2, 2 + n)) + (1,)   # (Co, Ci, *K) -> (Co, *K, Ci)
w_t = b.transpose(w, perm)
```

For the Qwen3-VL case `w = (1152, 3, 2, 16, 16)`, `perm = (0, 2, 3, 4, 1)`,
output `(1152, 2, 16, 16, 3)`:

| op | nki_ops | lower time |
|----|---------|-----------|
| weight transpose `(1152,3,2,16,16)->(1152,2,16,16,3)` | **258,048** | ~1.5 s |

(For comparison, the weight *reshape* that follows used to be ~1.88 M ops and
is now 36 ops after `2f8f706`. The transpose is now the dominant cost; the full
Qwen conv3d lowering exceeds 200 s and is skipped.)

## Root cause

`lower_transpose_dma` / `emit_transpose` only perform the on-chip P↔F swap on
the **last two** dims. Every other dim is treated as a batch position and
iterated one tile at a time:

```
out_shape   = (1152, 2, 16, 16, 3)
out_batch   = (1152, 2, 16)        # 36,864 batch positions
per-tile     (P, F) = (16, 3)      # one tiny transpose each
=> 36,864 transposes x ~7 ops = ~258k ops
```

Because `Ci` (the moved axis) and the spatial block are far apart in the source
index order, the framework can't fold them into a larger tile, so the partition
axis is only ever 16 wide and the free axis 3 wide — a near-worst-case use of
the 128-wide partition.

## What a fix looks like (and why it is non-trivial)

Two independent ideas each help, and were prototyped and verified **in the
interpreter** — but both ran into hardware constraints that the interpreter does
*not* enforce. Caveat for whoever picks this up: **validate on hardware (the
`nkigen-lite/tests/tensor_ir/test_direct_lower_transpose.py` suite runs on
device), not just the numpy interpreter.** The interpreter accepted both
prototypes below; the MLIR verifier / KB rejected them.

### Idea 1 — collapse adjacent in-order axis runs

A run of source axes that stay adjacent and in order under `perm` moves as one
contiguous block (row-major), so it can be merged into a single axis:

```
(1152, 3, 2, 16, 16) perm (0,2,3,4,1)   ->   (1152, 3, 512) perm (0,2,1)
```

This is a row-major no-op (verified correct across many perms) and reduces the
batch iteration from `1152*2*16` to `1152`. Reusable sketch:

```python
def _collapse_perm(in_shape, perm):
    groups = [[perm[0]]]
    for j in range(1, len(perm)):
        if perm[j] == perm[j-1] + 1:
            groups[-1].append(perm[j])
        else:
            groups.append([perm[j]])
    src_order = sorted(groups, key=lambda g: g[0])
    collapsed_in = tuple(math.prod(in_shape[a] for a in g) for g in src_order)
    pos = {tuple(g): i for i, g in enumerate(src_order)}
    collapsed_perm = tuple(pos[tuple(g)] for g in groups)
    return collapsed_in, collapsed_perm
```

### Idea 2 — fold leading passthrough dims into the partition, one N-D `dma_transpose`

After collapse, the Qwen transpose is `(Co, Ci, S) -> (Co, S, Ci)` with `Co` a
leading passthrough dim. Keeping `Co` as the partition and doing a single 3D
`dma_transpose` over a `(P_co, Ci, S)` tile drops the whole transpose to **~45
ops** in the interpreter.

## The two blocking hardware constraints

1. **`dma_transpose` supports only specific permutations.** The MLIR verifier
   (surfaced via `emit_to_kb`) allows exactly:

   ```
   2D = [1, 0]
   3D = [2, 1, 0]
   4D = [3, 1, 2, 0]
   ```

   The "keep partition, swap the trailing two" perm `[0, 2, 1]` that Idea 2
   needs is **not** in that set, so the one-shot 3D transpose is illegal on
   device even though the interpreter runs it. Any real fix must express the
   swap using one of the legal perms (or fall back to the tensor-engine
   `A.T @ I` path in `lower_transpose_te`).

2. **Partial slicing of a *merged* axis needs original-rank decomposition.**
   `emit_transpose` receives HBM tensors at the **original** rank (rank-5 for
   the Qwen weight). After collapsing to `(Co, Ci, 512)`, tiling the merged
   `512 = 2*16*16` axis at 128 produces a range like `[0:128]` that is **not** a
   single rectangle over `(2, 16, 16)`. So either:
   - the HBM tensors must be re-declared at collapsed rank (there is no in-place
     HBM reshape today — would need plumbing through `hbm_map` /
     `_emit_transpose_op`), or
   - each partial tile must be expanded back to original-rank rectangles with
     `flat_range_to_src_chunks` (from `direct_lower_utils`), emitting possibly
     several DMAs per tile.

   `flat_range_to_src_chunks` already exists and does exactly this (it backs the
   reshape fix), so this half is mechanical — but it has to be wired into the
   slice generation for both the load and the store.

## Suggested approach for the fix

1. Add `_collapse_perm` (Idea 1) as a canonicalization at the top of
   `emit_transpose` / `lower_transpose_dma`. Low risk, immediate ~8x on Qwen
   even with the existing per-tile emitter.
2. For the trailing P↔F swap, keep using the legal 2D `dma_transpose([1,0])`
   (already what the per-tile path does) or the TE matmul path — do **not** rely
   on `[0,2,1]`.
3. Generate load/store slices at original rank via `flat_range_to_src_chunks`
   so merged-axis tiles that straddle original-axis boundaries stay correct.
4. Optionally fold leading passthrough dims into the partition to widen the
   partition axis (the big constant-factor win), but only once 1–3 are correct
   and hardware-verified.

## Validation checklist

- `nkigen-lite/tests/tensor_ir/test_direct_lower_transpose.py` — **runs on
  device**; the source of truth. The interpreter is necessary but not
  sufficient (it accepted both rejected prototypes above).
- End-to-end: unskip the Qwen parametrization in
  `tests/unit/test_tensor_api.py::test_conv3d` (currently guarded by
  `out_channels >= 512`) and confirm it lowers in seconds and matches PyTorch.
- Regression: the conv2d / conv3d on-device tests already enabled in `2f8f706`
  / `ecaba84` must stay green.
