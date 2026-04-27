# Qwen3 Layer Debug: NISA Lowering Bugs

## Overview

`buggy.mlir` is the NISA-lowered output of `kernel.py` (a full Qwen3 transformer
decoder layer). It fails BIR verification and produces incorrect results due to
three categories of bugs in the linalg-to-nisa / simplify-linalg passes.

`fix_rope_vector_partition.mlir` contains the corrected version (simulation passes).

```
source ../run.sh qwen3_layer/buggy.mlir                    # BIR verification failed
source ../run.sh qwen3_layer/fix_rope_vector_partition.mlir # SIMULATION PASSED
```

---

## Bug 1: Q/K/V reshape — wrong column interleaving

**Symptom**: Silent numerical corruption (no compilation error).

**Location in MLIR**: Lines 196-234 — `(256,256) shared_hbm → (2,2,128,128) hbm/shared_hbm`

**What happens**: The pass emits a column-by-column 128x2 → 2x128 DMA transpose,
iterating `arg13 = 0..128` and loading 2 adjacent columns per iteration. It maps
`d0 ∈ [0,2)` to the head dimension. But adjacent columns in the `(BS, hidden_size)`
layout are **not** different heads — they are adjacent `head_dim` values within the
**same** head. Head 0 occupies cols `[0, 128)`, head 1 occupies cols `[128, 256)`.

```
# Python: (256,256) → (2,128,2,128) → transpose(0,2,1,3) → (2,2,128,128)
# Column j maps to: head = j // head_dim, hd = j % head_dim
# Adjacent cols j, j+1 are both head 0 (for j < 127) — NOT different heads!
```

**Fix**: Replace the 128x2 transpose with 128x128 block copies per `(batch, head)`:
```mlir
scf.for %batch = %c0 to %c2 step %c1 {
  scf.for %head = %c0 to %c2 step %c1 {
    %row = arith.muli %batch, %c128
    %col = arith.muli %head, %c128
    // Load 128x128 block: rows [batch*128, (batch+1)*128), cols [head*128, (head+1)*128)
    dma_copy(src=Q_proj[%row+d0, %col+d1] → sbuf_tmp[d0, d1], tile <128|128>)
    dma_copy(sbuf_tmp → Q_mh[%batch, %head, d0, d1], tile <128|128>)
  }
}
```

**Pass to fix**: `simplify-linalg` or `linalg-to-nisa` — the reshape
`(BS, hidden) → (batch, seq, heads, hd)` + transpose `(0,2,1,3)` lowering.

---

## Bug 2: RoPE subtract/add on multi-partition SBUF

**Symptom**: BIR verification failure:
```
Invalid access of 1 partitions starting at partition 1
Opcode: TensorTensor
```

**Location in MLIR**: Lines 265-273, 304-312, 348-356, 387-395 — the
`q_rot0 = q0*cos - q1*sin` (and similar) final subtract/add loops.

**What happens**: The pass lowers `q0*cos - q1*sin` as three separate loops:
1. Loop over 4 heads: `mem_24[i] = q0_slice * cos` → 4x128x64 SBUF
2. Loop over 4 heads: `mem_25[i] = q1_slice * sin` → 4x128x64 SBUF
3. Loop over 4 heads: `mem_26[i] = mem_24[i] - mem_25[i]` ← **BUG**

Loop 3 uses `tensor_tensor_arith` (`engine=vector`) reading from `mem_24[%arg12+d0, ...]`.
The vector engine processes all 128 SBUF partitions simultaneously and **cannot**
selectively address partition N of a multi-partition tensor. When `%arg12=1`, BIR
verification rejects the access to partition 1.

A simpler staging-DMA fix (copy each partition to a 1-partition temp) was tried but
causes **SBUF OOM** because all three 4-partition tensors must be live simultaneously.

**Fix**: Fuse the three loops into one, computing multiply and subtract in 1-partition
temps within a single iteration:
```mlir
scf.for %i = %c0 to %c4 step %c1 {
  tmp_a = q0_slice[i] * cos   // 1x128x64 sbuf — safe for vector engine
  tmp_b = q1_slice[i] * sin   // 1x128x64 sbuf
  result = tmp_a - tmp_b      // 1x128x64 sbuf — both operands are 1-partition
  dma_copy(result → output[%i, ...])
}
```

**Pass to fix**: `linalg-to-nisa` — when lowering element-wise binary ops on tensors
tiled with a small partition dim (e.g. `BH=4`), the pass should detect that both
operands of the subtract share the same loop structure and fuse them, avoiding
multi-partition SBUF intermediates.

---

## Bug 3: Head-concat reshape — DMA transpose OOB

**Symptom**: BIR verification failure:
```
Access pattern out of bounds on instruction ... Pattern: [[32768,128],[1,1],[1,1],[1,2]]
```

**Location in MLIR**: Lines 519-534 — `(4,128,128) shared_hbm → (2,128,2,128) sbuf`
via `mem_49`, then `(2,128,2,128) → (128,2,2,128)` via second transpose.

**What happens**: The first DMA transpose uses tile `<128|2>` on `memref<2x128x2x128xf32, sbuf>`.
This writes 128 elements into dimension 0 (size 2) — **out of bounds**. The entire
two-step transpose through `mem_49` is structurally wrong.

**Fix**: Skip `mem_49`. Directly create `mem_51` by transposing each 128x128 head block:
```mlir
// mem_51[hd, head, batch, seq] = mem_48[batch*2+head, seq, hd]
scf.for %bh = %c0 to %c4 step %c1 {
  tmp = dma_copy(mem_48[%bh, d0, d1])       // 128x128 sbuf: (seq, hd)
  %batch = divui %bh, %c2
  %head  = remui %bh, %c2
  dma_transpose(tmp → mem_51[d0, %head, %batch, d1], perm=[1,0])  // (hd, seq)
}
```

Note: `mem_51` dim1 = head, dim2 = batch (matching the downstream matmul's
`stationary[d0, %arg14, %arg12, d1]` where arg14 = reduction tile = head,
arg12 = row tile = batch).

**Pass to fix**: `simplify-linalg` or `linalg-to-nisa` — the inverse reshape
`(BH, seq, hd) → (batch, seq, heads, hd) → (BS, hidden)` lowering. Same root
cause as Bug 1 (incorrect head/column interleaving in the transpose).

---

## Root Cause Summary

| Bug | Pass stage | Root cause | Compilation error? |
|-----|-----------|------------|-------------------|
| 1 | reshape lowering | Adjacent cols treated as different heads | No (silent corruption) |
| 2 | elementwise fusion | Vector op on multi-partition SBUF | Yes (BIR verification) |
| 3 | reshape lowering | Same as Bug 1, inverse direction | Yes (BIR verification) |

Bugs 1 and 3 share the same root cause: the reshape `(BS, hidden) ↔ (batch, head, seq, hd)`
is lowered with a column-by-column transpose that conflates the head and head_dim dimensions.
The fix for both is to tile at the head granularity (128x128 blocks) rather than column
granularity (128x2 strips).

---

## Proposed Compiler Pass Fixes

### Fix A: `SimplifyLinalg.cpp` — `decomposeHighRankTranspose()` (Bugs 1 & 3)

**Root cause in the pass**: `decomposeHighRankTranspose` handles `[0,2,1,3]` on
`(2,128,2,128)` by looping over identity dims {0(batch=2), 3(hd=128)} and doing a
2D `linalg.transpose [1,0]` on the swapped pair {1(seq=128), 2(heads=2)}.

This produces 128x2 → 2x128 inner transposes, which get collapsed back through
`expand_shape → subview → collapse_shape` when `getBaseAndOffsets` in linalg-to-nisa
traces to the flat (256,256) base. The collapse_shape strides are lost — `d1*128 + hd`
becomes `hd + d1`, making adjacent columns look like different heads.

**Proposed change** (in `decomposeHighRankTranspose`, ~line 131):

When one of the two swapped dims is small (size << the other), move it to the outer
loop instead of transposing it. This converts the inner operation from a transpose to
a plain copy, avoiding the problematic stride through collapse_shape entirely.

```cpp
// After identifying d0, d1 as the two swapped dims:
int64_t sizeD0 = srcShape[d0], sizeD1 = srcShape[d1];

// If one swapped dim is much smaller than the other, loop over the small
// dim instead of transposing.  This makes the inner operation a copy
// (both remaining dims are identity), which avoids non-unit-stride
// collapse_shape tile maps that getBaseAndOffsets cannot preserve.
//
// Example: [0,2,1,3] on (2,128,2,128)
//   Before: loop batch(2) * hd(128) = 256 iters, inner 128x2 transpose
//   After:  loop batch(2) * head(2) = 4 iters, inner 128x128 copy
constexpr int64_t kSmallDimThreshold = 16;  // heads are typically 2-32
int64_t smallSwapDim = -1;
if (sizeD0 < sizeD1 && sizeD0 <= kSmallDimThreshold) {
  smallSwapDim = d0;   // move d0 to outer loops
} else if (sizeD1 < sizeD0 && sizeD1 <= kSmallDimThreshold) {
  smallSwapDim = d1;   // move d1 to outer loops
}

if (smallSwapDim >= 0) {
  // Treat the small swapped dim as an identity dim (loop over it).
  // The remaining inner dims are all identity → emit copy, not transpose.
  identityDims.push_back(smallSwapDim);
  // Remove from swappedDims so we don't try to transpose it
  swappedDims.erase(std::remove(swappedDims.begin(), swappedDims.end(),
                                smallSwapDim), swappedDims.end());
  // ... continue to loop-nest creation below ...
  // Inner operation becomes memref.copy (or linalg.copy) instead of
  // linalg.transpose, since only 1 non-identity dim remains.
}
```

The key insight: for `[0,2,1,3]` on `(batch, seq, heads, hd)` where `heads=2`:
- Loop over batch, **head**: 2 x 2 = 4 iterations
- Inner: copy of `(seq=128, hd=128)` block — 128x128 DMA copy, no transpose
- Access: `src[batch*128+d0, head*128+d1]` — correct strides, no collapse_shape

For the inverse (Bug 3, `(BH,seq,hd) → (batch,heads,seq,hd)`), the same optimization
applies: loop over the small head dim, inner is 128x128 transpose (seq,hd)→(hd,seq).

The dst subview offset computation needs adjustment: for a swapped dim that's now
looped, the loop IV goes to `perm[dim]` position in the dst (not `dim` position),
since this dim's src→dst mapping differs from identity dims.

### Fix B: Tiling / linalg-to-nisa — fuse RoPE elementwise chain (Bug 2)

**Root cause**: The tiling pass creates three separate loops for `q0*cos - q1*sin`:
1. `tmp_a[i] = q0[i] * cos` → writes 4-partition SBUF
2. `tmp_b[i] = q1[i] * sin` → writes 4-partition SBUF
3. `result[i] = tmp_a[i] - tmp_b[i]` → reads 4-partition SBUF with vector engine (BUG)

The vector engine cannot address individual partitions of a multi-partition SBUF tensor.

**Proposed fix** (two options):

**Option 1 — Tiling pass**: Detect element-wise chains like `sub(mul(a,b), mul(c,d))`
where all four operands share the same partition-dim tiling. Fuse into a single tiled
loop that keeps all intermediates at 1-partition granularity:

```
for i = 0..n_partitions:
  tmp_a = a[i] * b[i]    // 1-partition temp
  tmp_b = c[i] * d[i]    // 1-partition temp
  out[i] = tmp_a - tmp_b  // 1-partition vector op — safe
```

This is the approach used in the working fix. The key: the multiply and subtract
share the same outer loop and the intermediates never grow beyond 1 partition.

**Option 2 — linalg-to-nisa verifier/rewriter**: Add a post-lowering check that
flags any `tensor_tensor_arith` (`engine=vector`) whose SBUF operand has a
loop-varying partition index. When detected, insert DMA staging copies to
1-partition temps. Note: this is less preferred since it increases SBUF pressure
and may cause OOM (as tested — the staging approach failed with SBUF OOM for this
kernel).
