# NISA MLIR Debug (`nisa_mlir_debug`)

Tools for debugging pre-compiled NISA-level MLIR kernels by running them through BIRSim and comparing against NumPy references.

## Quick Start

```bash
# From this directory:
source ./run.sh <mlir_file>
```

This will:
1. Set up the NKI environment and PYTHONPATH
2. Parse the MLIR to extract function signature (shapes, dtypes)
3. Generate deterministic random inputs (seed=42)
4. Compile the MLIR to NEFF with BIRSim enabled (`target=trn2`)
5. Compare BIRSim output against a NumPy reference in `kernel.py`

## Files

| File | Description |
|------|-------------|
| `run.sh` | Bash entry point — sets up env, resolves paths, calls `run_sim.py` |
| `run_sim.py` | Core harness — parses MLIR, compiles to NEFF, runs BIRSim, compares output |

## Adding a Test Case

Each test case lives in its own subdirectory alongside the MLIR file(s) it tests.

Required structure:
```
my_bug_fix/
├── kernel.py       # NumPy reference: must define a function matching the MLIR func name
├── buggy.mlir      # (optional) original broken MLIR
├── fixed.mlir      # corrected MLIR
└── README.md       # description of the bug and fix
```

### `kernel.py` contract
- Must contain a function whose name matches the `sym_name` in the MLIR
- Takes the same number of numpy arrays as the MLIR function's inputs
- Returns a single numpy array matching the MLIR function's output shape/dtype

### Running a test case

```bash
source ./run.sh my_bug_fix/fixed.mlir
```

Artifacts (NEFF, BIR, BIRSim outputs) are written to `artifacts_<stem>/` next to the MLIR file. These directories are git-ignored.

## Output

A successful run prints:

```
BIRSim output: shape=(256, 256), dtype=float32
  range: [-1.2345, 1.2345]
  mean:  0.0012

BIRSim PASSED

--- Running numpy reference from kernel.py ---
  Max difference:  1.95e-02
  Mean difference: 3.40e-05
  Match: True

SIMULATION PASSED
```

Failures exit non-zero with either:
- **`NCC_ISIM*` errors** — BIRSim detected an issue (e.g., uninitialized PSUM read)
- **`SIMULATION FAILED`** — BIRSim output doesn't match NumPy within tolerance (`atol=1e-2, rtol=1e-2`)

## Existing Test Cases

| Directory | Issue |
|-----------|-------|
| `psum_accumulate_flags_fix/` | Missing `psum_accumulate_flags` on matmul K-loops, unreleased SBUF, wrong elementwise op |
