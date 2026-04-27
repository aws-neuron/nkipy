---
name: debug_nisa_ir
description: Debug NISA MLIR that fails BIRSim. Creates a debug case under tests/debug/ with buggy.mlir, kernel.py, iterative fixes, and a README proposing compiler pass changes.
user-invocable: true
---

## Usage

`/debug_nisa_ir <bug_name> [kernel.py path] [buggy NISA MLIR path or inline]`

- `bug_name`: Short snake_case name for the debug case (e.g., `rope_partition_oob`)
- `kernel.py path`: Path to the Python source that was fed into `nkipy_opt`. If omitted, ask the user.
- `buggy NISA MLIR`: Path to the `.mlir` file that `nkipy_opt` produced, or the user may paste it inline. If omitted, ask the user.

## Instructions

You are debugging a NISA-level MLIR kernel that `nkipy_opt` generated but that fails BIRSim verification or produces incorrect numerical results. Follow this systematic workflow.

### Step 1: Set up the debug case directory

Create `tests/debug/<bug_name>/` with:

```
tests/debug/<bug_name>/
  kernel.py       # Copy of the input Python kernel
  buggy.mlir      # The failing NISA MLIR from nkipy_opt
  README.md       # Will be populated in Step 6
```

Copy the user-provided `kernel.py` and `buggy.mlir` into this directory. Ensure `kernel.py` contains a function whose name matches the `sym_name` in the MLIR (this is required by `run_sim.py`).

### Step 2: Reproduce the failure

Run the buggy MLIR through BIRSim:

```bash
cd tests/debug && source ./run.sh <bug_name>/buggy.mlir
```

Record the exact error output. Common failure modes:
- **BIR verification error**: `Invalid access of N partitions starting at partition M` or `Access pattern out of bounds`
- **BIRSim runtime error**: `NCC_ISIM*` errors (e.g., uninitialized PSUM read)
- **Numerical mismatch**: `SIMULATION FAILED (max_diff=...)` -- BIRSim runs but output doesn't match kernel.py

### Step 3: Analyze the bug

Read the MLIR carefully and identify the root cause. Common patterns:

1. **Multi-partition SBUF with vector engine**: `tensor_tensor_arith` (engine=vector) reading from a loop-indexed partition of a multi-partition SBUF tensor. The vector engine processes all 128 partitions simultaneously and cannot address partition N selectively.

2. **Wrong reshape/transpose lowering**: Column-by-column transposes that conflate head and head_dim dimensions. Often manifests as `<128|2>` tile on a dim of size 2 (OOB), or silent numerical corruption.

3. **Missing accumulate flags**: Matmul K-loops without `psum_accumulate_flags`, causing PSUM overwrite instead of accumulate.

4. **SBUF OOM**: Too many live SBUF tensors. Check if intermediates can be fused or freed earlier.

Focus on understanding:
- Which MLIR lines are problematic (cite line numbers)
- What the pass *intended* to generate vs what it actually generated
- Why the hardware rejects it (BIR rules violated)

### Step 4: Create iterative fixes

For each fix attempt, create a new MLIR file:

```
fix_<number>_<what_was_fixed>.mlir
```

For example:
- `fix_01_fuse_rope_elementwise.mlir`
- `fix_02_reshape_head_granularity.mlir`

Edit the MLIR by hand to correct the identified issue. Then run:

```bash
cd tests/debug && source ./run.sh <bug_name>/fix_01_<description>.mlir
```

If it still fails, analyze the new error, create another fix file, and iterate. Keep each attempt as a separate file so the progression is visible.

### Step 5: Verify the final fix

The last `fix_*.mlir` should produce:

```
BIRSim PASSED
SIMULATION PASSED
```

Confirm that the numerical output matches `kernel.py` within tolerance (atol=1e-2, rtol=1e-2).

### Step 6: Write the README

Create `tests/debug/<bug_name>/README.md` documenting:

1. **Overview**: One paragraph summarizing what `buggy.mlir` is (which kernel, what it does) and what goes wrong.

2. **How to reproduce**: The exact `source ../run.sh` commands for buggy and fixed versions.

3. **Bug analysis**: For each bug found:
   - **Symptom**: The exact error message
   - **Location in MLIR**: Line numbers and what the code does
   - **What happens**: Why the hardware rejects it or produces wrong results
   - **Fix**: What was changed in the MLIR (with code snippets)

4. **Root cause summary**: Table mapping each bug to the compiler pass responsible and whether it causes a compilation error or silent corruption.

5. **Proposed compiler pass fixes**: For each bug, describe:
   - Which pass to fix (e.g., `simplify-linalg`, `linalg-to-nisa`, tiling)
   - The root cause *in the pass* (not just the MLIR symptom)
   - A concrete proposed change (pseudocode or description of the algorithm change)

Use the format from existing debug cases (see `tests/debug/qwen3_layer/README.md` for reference).

### Tips

- The debug harness (`run.sh` / `run_sim.py`) automatically sets up the NKI environment, generates random inputs (seed=42), compiles to NEFF with BIRSim, and compares against `kernel.py`.
- Artifacts (NEFF, BIR) are written to `artifacts_<stem>/` next to each MLIR file (git-ignored).
- When editing MLIR, keep changes minimal and targeted. Change only the ops/loops related to the bug.
- If you're unsure which pass generated a problematic pattern, check the pass pipeline in `nkipy_opt` or ask the user.
