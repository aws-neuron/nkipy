# CLAUDE.md

> **Keep this file up to date.** After any change to the codebase (new passes, renamed files, pipeline changes, new test patterns), update the relevant sections of this file so future sessions start with accurate context.

## Project Overview

NKIPyKernelGen is a compiler that traces NumPy functions and lowers them to NISA (Neuron Instruction Set Architecture) for AWS Neuron hardware. Users write kernels in Python with `@trace` and `knob.knob()` annotations, and the compiler handles tiling, memory placement, layout legalization, and NISA lowering.

### Git Commit Policy

- Do not add a `Co-Authored-By` line in commit messages.

### Design Philosophy

- **Fast prototyping.** Prioritize speed of development. Keep code simple, ship quickly, iterate.
- **Composable passes.** Each pass does one thing well and produces valid IR. The IR after any pass should be functional and simulatable (before NISA lowering: via LLVM JIT; after: via BIR simulation).
- **Testable at every stage.** Pass tests verify individual transformations with FileCheck. E2E tests verify numerical correctness through the full pipeline.

## Two-Package Architecture

The compilation flow spans two packages:

1. **NKIPyKernelGen** (this package): Python tracing + C++ MLIR passes via `nkipy-opt`. Takes Python → NISA IR.
2. **private-nki-staging** (at `$NKI_COMPILER_PATH`): NISA dialect definitions and backend passes via `nki-opt-pipeline`. Takes NISA IR → BIR binary.

NKIPyKernelGen depends on the NISA dialect from private-nki-staging. The `nkipy-opt` binary links against its NISA dialect libraries. After running `source scripts/setup_nki.sh`, the `$NKI_COMPILER_PATH` environment variable points to the private-nki-staging source tree and `$NKI_BUILD_DIR` points to its build directory. Build private-nki-staging first, then build NKIPyKernelGen.

### private-nki-staging backend pipeline (`nki-opt-pipeline`)

Runs after NKIPyKernelGen's pipeline, on NISA dialect IR:

| Pass | Description |
|------|-------------|
| canonicalize + cse | Fixed-point simplification before NISA transforms |
| nisa-unroll | Unroll NISA ops within tile loops |
| rotate-numbering | Assign rotation slot numbers |
| nisa-post-unroll-canonicalization | NISA-specific cleanup after unrolling |
| linear-scan-allocation | Assign physical SBUF/PSUM addresses |
| instruction-scheduling | Reorder ops for latency hiding (ASAP, critical-path, DMA-prefetch heuristics) |
| mlir-to-bir | Emit BIR binary for neuron-cc |

Defined in `private-nki-staging/lib/Pipelines/CompilationPipelines.cpp`.

### private-nki-staging key files

**NISA dialect definitions** (TableGen):
| File | What it defines |
|------|-----------------|
| `include/nki/Dialect/Nisa/IR/Nisa.td` | Main dialect definition (`Nisa_Dialect`), base op class (`Nisa_Op`) |
| `build/include/nki/Dialect/Nisa/IR/NisaOps.td` | All 68 NISA ops (generated from `.td.j2` template). Key ops: `DmaCopy`, `TensorTensorArith`, `Matmul`, `Activation`, `Reciprocal`, `Memset`, `Rand` |
| `build/include/nki/Dialect/Nisa/IR/NisaAttrs.td` | All enum attrs (generated). Key enums: `MemSpaceKind` (PSUM/SBUF/HBM/SHAREDHBM), `EngineKind` (TENSOR/SCALAR/VECTOR/DMA/SYNC), `ActivationFunctionKind` (30+ activations), `ArithOpKind`, `TargetKind` (TRN1/TRN2/TRN3) |
| `include/nki/Dialect/Nisa/IR/NisaAllocationOps.td` | `AllocOp` (with num_slots for multi-buffering), `ReleaseOp`, `BindMemLocOp`, `RotateOp` |
| `include/nki/Dialect/Nisa/IR/NisaTypes.td` | `SpecialRegisterType`, `Float4E2M1FnX2Type` |
| `include/nki/Dialect/Nisa/IR/NisaSynchronizationOps.td` | `CoreBarrierOp` for multi-core sync |

**Op interfaces** (used by scheduler and allocator):
| File | Interface |
|------|-----------|
| `include/.../NisaMemDepOpInterface.td` | `getReadMemAccesses()` / `getWriteMemAccesses()` — memory dependency tracking |
| `include/.../NisaLatencyOpInterface.td` | `computeLatencyNs()` — per-op latency for scheduling |

**Pass implementations** (C++):
| File | Pass |
|------|------|
| `lib/Dialect/Nisa/Transforms/NisaUnroll.cpp` | Unroll SCF loops with known trip counts |
| `lib/Dialect/Nisa/Transforms/RotateNumbering.cpp` | Assign slot numbers to rotate ops |
| `lib/Dialect/Nisa/Transforms/LinearScanAllocation.cpp` | Physical SBUF/PSUM address assignment (34K lines) |
| `lib/Dialect/Nisa/Transforms/InstructionScheduling.cpp` | List scheduling with configurable heuristics |
| `lib/Dialect/Nisa/Transforms/MLIRToBIR.cpp` | Emit BIR JSON for neuron-cc |
| `include/nki/Dialect/Nisa/Transforms/Passes.td` | All pass definitions |

## Build & Environment Setup

### Environment setup

Before building or running tests, activate your Python virtual environment and set the required environment variables:

```bash
source <your-venv>/bin/activate
export NKI_BUILD_DIR=<path-to-private-nki-staging>/build
export NKI_SOURCE_DIR=<path-to-private-nki-staging>
```

> **Note:** `scripts/setup_nki.sh` exists but has hardcoded paths that may not match your local setup. Set the variables above manually or edit the script for your environment.

### Install / rebuild

`pip install -e .` builds the C++ passes (`nkipy-opt`) via CMake under the hood — no manual CMake invocation needed:

```bash
pip install -e .          # builds C++ and installs Python package in editable mode
```

If you only changed Python code (no C++ pass changes), the editable install means changes take effect immediately — no rebuild needed.

If `private-nki-staging` (sibling repo) changes, rebuild it first:
```bash
cd <path-to-private-nki-staging> && nki-rebuild
```

### Run tests

```bash
python3 -m pytest tests/passes/ -v      # pass-level tests
python3 -m pytest tests/e2e/ -v         # end-to-end tests
python3 -m pytest tests/unit/ -v        # unit tests
```

## Compilation Pipeline (nkipy-opt)

Defined in `nkigen/transforms/nkipy_opt.py` → `apply_complete_knob_pipeline()`.
C++ pass implementations in `mlir/lib/Transforms/`.

### Phase 0: Arithmetic Preparation
| # | Pass | Source | What it does |
|---|------|--------|--------------|
| 1 | `remove-redundant-zero-fill` | RemoveRedundantZeroFill.cpp | Remove `linalg.fill(0)` ops whose only users are matmul-like ops. NISA matmul auto-zeros PSUM, so the fill is redundant. Must run before tiling to prevent the fill from becoming a `nisa.memset` with out-of-bounds access. |
| 2 | `prepare-arithmetic` | PrepareArithmetic.cpp | Rewrite `linalg.div(A,B)` → `linalg.mul(A, linalg.reciprocal(B))`. NISA has no divide. Runs before tiling so reciprocal gets tiled normally. |

### Phase 1: Layout Inference, Partition Dim Canonicalization, and Tiling (on tensor IR)
| # | Pass | Source | What it does |
|---|------|--------|--------------|
| 3 | `infer-layout` | InferLayout.cpp | Auto-infer tile_size, mem_space, partition_dim, and reduction_tile for all linalg ops. Seeds from user annotations, then matmul hardware rules, then elementwise fallback defaults. Uses bidirectional BFS propagation with tile-size divisibility conflict checks. Return values get SharedHbm; intermediates default to SBUF. |
| 4 | `canonicalize-partition-dim` | CanonicalizePartitionDim.cpp | Insert `linalg.transpose` ops at boundaries where `partition_dim != 0`, so all downstream ops see `partition_dim=0`. Annotates inserted transposes with tile_size/mem_space for tiling. |
| 5 | `assign-linalg-op-ids` | AssignLinalgOpIds.cpp | Stamp unique `nkipy.op_id` on each linalg op (including transposes from step 3) so transform dialect can match individual instances. |
| 6 | `knob-driven-tiling` | KnobDrivenTiling.cpp | Read `nkipy.annotate` ops and emit a `transform.named_sequence @__transform_main` that tiles each linalg op according to its tile_size/reduction_tile. Supports arbitrary-rank tensors and `linalg.transpose`. |
| 7 | `apply-and-strip-transforms` | ApplyAndStripTransforms.cpp | Fused pass: runs `@__transform_main` (same semantics as upstream `--transform-interpreter`) and then erases the transform module (NamedSequenceOps + `transform.with_named_sequence` attr). Leaves the IR free of any transform-dialect ops, which is what the Python linalg→NISA phase (parsed via upstream MLIR bindings) needs. |
| 8 | `canonicalize-loop-step` | CanonicalizeLoopStep.cpp | Rewrite `scf.for %i = 0 to N step S` → `scf.for %i = 0 to N/S step 1` with `%orig = %i * S`. Simplifies downstream index math. |

### Phase 2: Bufferization
| # | Pass | Source | What it does |
|---|------|--------|--------------|
| 9 | `one-shot-bufferize` | (upstream MLIR) | Convert tensor IR to memref IR (tensor.extract_slice → memref.subview, etc.) |
| 10 | `canonicalize` | (upstream MLIR) | Fold and simplify memref operations |

### Phase 3: Memory Space Annotation + Reshape Canonicalization
| # | Pass | Source | What it does |
|---|------|--------|--------------|
| 11 | `eliminate-uninitialized-copies` | EliminateUninitializedCopies.cpp | Remove `memref.copy` from never-written allocs (e.g. PSUM accumulator init — matmul zeros PSUM via `psum_zero_region`). |
| 12 | `canonicalize` | (upstream MLIR) | Clean up dead subview chains from eliminated copies |
| 13 | `annotate-memory-space` | AnnotateMemorySpace.cpp | Read `nkipy.annotate` ops, apply NISA memory space attrs (`#nisa.mem<sbuf>`, `#nisa.mem<psum>`, `#nisa.mem<sharedhbm>`) to memref types, mark function args as SharedHbm, then erase all `nkipy.annotate` ops. |
| 14 | `canonicalize-reshape` | CanonicalizeReshape.cpp | Classify `expand_shape`/`collapse_shape` by mem_space and partition_dim. HBM reshapes and SBUF non-pdim reshapes stay as views. SBUF partition dim splits get alloc+copy (NISA has no modulo). Returned expand_shape views of func args and direct returns of func args get alloc+copy (NISA needs separate output allocations). |
| 15 | `eliminate-same-memspace-copy` | EliminateSameMemSpaceCopy.cpp | Eliminate redundant copies within the same memory space (e.g. SBUF→SBUF when data is already in SBUF from a previous op). Rewires uses to read the source directly. |
| 16 | `canonicalize` | (upstream MLIR) | DCE dead allocs from eliminated copies |

### Phase 4: NISA Lowering
| # | Pass | Source | What it does |
|---|------|--------|--------------|
| 17 | `legalize-layout` | LegalizeLayout.cpp | Transform SBUF allocs from R-D to (R+2)-D physical layout (`[partTile, numBlocks..., freeTile]`), rewrite subviews to (R+2)-D indexing, collapse everything to 2D for NISA compute. Reconstruct linalg named ops (AddOp, SubOp, ReciprocalOp, etc.) with 2D operands. |
| 18 | `canonicalize` | (upstream MLIR) | Fold collapse_shape chains, simplify affine maps |
| 19 | `simplify-linalg` | SimplifyLinalg.cpp | Decompose high-rank transposes to loops of 2D, collapse >2D SBUF transpose to 2D, canonicalize trivial-broadcast generics to named ops. Runs before insert-spill-reload so any SBUF temps it creates are accounted for in spill/reload memory budgeting. |
| 20 | `insert-spill-reload` | InsertSpillReload.cpp | Analyze per-partition SBUF memory pressure and insert spill (SBUF→HBM) and reload (HBM→SBUF) `memref.copy` ops when capacity is exceeded. Uses Belady's MIN heuristic. Per-partition size = `total_size / shape[0]` on physical layout. |
| 21 | `insert-memref-dealloc` | InsertMemRefDealloc.cpp | Insert `memref.dealloc` for SBUF/PSUM allocs at the end of their enclosing scope (loop body or function). These become `nisa.release` after lowering. Skips HBM/SharedHBM (externally managed). |
| 22 | `linalg-to-nisa` | LinalgToNisa.cpp | Pattern-match linalg/memref ops to NISA: `linalg.add/sub/mul` → `nisa.tensor_tensor_arith`; `linalg.matmul` → `nisa.matmul`; `memref.copy(HBM↔SBUF)` → `nisa.dma_copy`; `linalg.exp` → `nisa.activation(op=exp)`; scalar broadcast ops → `nisa.tensor_scalar_arith`; `linalg.reciprocal` → `nisa.reciprocal`; `linalg.transpose` → `nisa.dma_transpose` or copy (for trivial reshapes); `memref.dealloc` → `nisa.release`. |
| 23 | `cse` | (upstream MLIR) | Common subexpression elimination |
| 24 | `canonicalize` | (upstream MLIR) | DCE unused subviews and cleanup |

### Phase 5: Custom Op Resolution + Finalization
| # | Pass | Source | What it does |
|---|------|--------|--------------|
| 25 | `resolve-custom-ops` | ResolveCustomOps.cpp | Inline stashed NISA custom op bodies at each call site. The backend pipeline doesn't support cross-function calls, so custom op functions are cloned into the caller via IRMapping. |
| 26 | `prepare-for-nki` | PrepareForNki.cpp | Strip transform module, remove internal `nkipy.*` attributes, add `nisa.target` hardware attribute. |

## Key Source Files

### Python (tracing & frontend)
| File | Role |
|------|------|
| `nkigen/trace.py` | `@trace` decorator — traces NumPy functions to MLIR |
| `nkigen/knob.py` | `knob.knob()` API — annotate tensors with mem_space, tile_size, reduction_tile |
| `nkigen/op_vtable.py` | NumPy op → MLIR lowering table |
| `nkigen/traced_array.py` | TracedArray wrapping MLIR SSA values with NumPy-like interface |
| `nkigen/transforms/nkipy_opt.py` | Pipeline orchestration — shells out to `nkipy-opt` binary |

### C++ (MLIR dialect & passes)
| File | Role |
|------|------|
| `mlir/include/nkipy/Dialect/NkipyOps.td` | `nkipy.annotate` op (target, mem_space, partition_dim, tile_size, reduction_tile) |
| `mlir/lib/Transforms/*.cpp` | All pass implementations (see pipeline table) |
| `mlir/lib/Transforms/OpClassification.cpp` | Shared helpers: classify linalg ops as unary/binary elementwise, matmul, etc. |
| `mlir/lib/Transforms/IRHelpers.cpp` | Shared IR utilities: constant extraction, memref helpers |
| `mlir/lib/Transforms/InlineNkipyReference.cpp` | Inline reference bodies into post-bufferization IR |

## Test Structure

```
tests/
├── conftest.py                    # Root conftest: centralizes sys.path setup
├── harness.py                     # Unified test harness: run_kernel_test(), Mode flags
├── passes/                        # Per-pass unit tests
│   ├── pass_utils.py              # Shared: run_passes(), compile_through_passes(), run_filecheck()
│   ├── annotate_memory_space/     # Memory space annotation tests
│   ├── canonicalize_loop_step/    # Loop step normalization tests
│   ├── canonicalize_partition_dim/# Partition dim canonicalization tests
│   ├── eliminate_same_memspace_copy/ # Same-memspace copy elimination tests
│   ├── eliminate_uninitialized_copies/ # Uninitialized copy elimination tests
│   ├── infer_layout/              # Layout inference propagation tests
│   ├── insert_spill_reload/       # SBUF spill/reload tests
│   ├── knob_driven_tiling/        # Tiling tests (2D + 3D)
│   ├── legalize_layout/           # Layout legalization tests (2D + 3D)
│   ├── linalg_to_nisa/            # NISA lowering tests
│   ├── prepare_arithmetic/        # Div-to-reciprocal rewrite tests
│   ├── resolve_custom_ops/        # Custom op inlining tests
│   └── ...                        # One directory per pass, each with test_*.py
├── e2e/                           # End-to-end: trace → NISA → BIR simulation / HW execution
│   ├── test_3d_elementwise.py     # 3D tensor tests for rank-R generalization
│   ├── test_attention.py          # Multi-head attention
│   ├── test_auto_layout.py        # Auto-inferred layouts (no user annotations)
│   ├── test_custom_op.py          # Custom NISA op inlining
│   ├── test_feedforward.py        # Feedforward network (matmul + SiLU + split)
│   ├── test_head_deconcat.py      # Head deconcatenation
│   ├── test_matmul_add.py         # Matmul + add fusion patterns
│   ├── test_multi_output.py       # Multiple output tensors
│   ├── test_partition_dim.py      # partition_dim != 0 with canonicalize-partition-dim
│   ├── test_qwen3_layer.py        # Qwen3 transformer layer
│   ├── test_reduce.py             # Reduce sum/mean operations
│   ├── test_rmsnorm.py            # RMS normalization
│   ├── test_rope.py               # Rotary positional embedding
│   ├── test_sigmoid.py            # Sigmoid, exp, scalar arithmetic, reciprocal
│   └── ...
└── unit/                          # Python-level unit tests
```

### Test modes (from `harness.py`)
- `Mode.LLVM` — LLVM JIT execution, compare to NumPy (requires `stop_after`)
- `Mode.HW` — Full pipeline → run on Trainium hardware (auto-skips if no device)
- `Mode.STRING_CHECK` — Assert compiled IR contains/excludes specific strings
- `Mode.FILECHECK` — Run LLVM FileCheck on compiled IR

### Test artifacts
When `request` fixture is passed to `run_kernel_test`, intermediate IR is dumped to `tests/<dir>/outputs/<test_name>/`. Each pass produces a numbered file (e.g., `01_assign_linalg_op_ids.mlir`, `15_legalize_layout.mlir`).

Pass tests explicitly set `dump_dir` via their `utils.py` helpers, typically saving to an `outputs/` directory next to the test file.

## Debugging

### `--dump-ir`: Dump IR from any test (recommended)

Add `--dump-ir` to any pytest invocation to save intermediate MLIR after every compiler pass:

```bash
python3 -m pytest tests/e2e/test_rope.py::test_rope --dump-ir -v -s
```

Output:
```
[dump-ir] IR will be saved to: /tmp/nkipy_dump_ir_abc123
[dump-ir] 27 IR files saved to: /tmp/nkipy_dump_ir_abc123
  00_input.mlir  (3,532 bytes)
  01_remove-redundant-zero-fill.mlir  (3,532 bytes)
  02_prepare-arithmetic.mlir  (3,533 bytes)
  ...
  22_linalg-to-nisa.mlir  (18,044 bytes)
  26_prepare-for-nki.mlir  (11,260 bytes)
```

If a test passes `request` to `run_kernel_test`, IR goes to `tests/<dir>/outputs/<test_name>/` instead of `/tmp`.

**Auto-dump on failure:** Even without `--dump-ir`, if compilation crashes, the harness automatically re-runs pass-by-pass into a temp directory and prints the path so you always get IR context.

### Inspect intermediate IR directly

```bash
# Run single pass directly:
nkipy-opt --legalize-layout input.mlir

# Chain passes:
nkipy-opt --annotate-memory-space --legalize-layout input.mlir

# See IR after every pass:
nkipy-opt --mlir-print-ir-after-all --legalize-layout input.mlir 2>&1
```

### From Python

```python
from nkigen.transforms.nkipy_opt import apply_complete_knob_pipeline

# Dump all intermediate files:
apply_complete_knob_pipeline(mlir_str, dump_dir="debug_outputs/")

# Stop at a specific pass:
apply_complete_knob_pipeline(mlir_str, stop_after="legalize-layout", dump_dir="debug_outputs/")

# Stop after pass N (1-indexed):
apply_complete_knob_pipeline(mlir_str, stop_after=10)  # stop after bufferize (canonicalize)

# For repeated passes like canonicalize, use "name:N" for the Nth occurrence:
apply_complete_knob_pipeline(mlir_str, stop_after="canonicalize:3")  # 3rd canonicalize (pass 16)
```

### `tests/debug/`: NISA MLIR debug cases

The `tests/debug/` directory contains standalone NISA MLIR test cases for debugging BIRSim failures. Each subdirectory is a self-contained repro with:

- `kernel.py` — NumPy reference implementation (used to compare BIRSim output)
- `buggy.mlir` — the broken NISA IR
- `fix_*.mlir` — proposed fixes (one or more iterations)
- `README.md` — (optional) root cause analysis and proposed compiler changes

Run a debug case:
```bash
cd tests/debug
source ./run.sh bmm/buggy.mlir            # run buggy version
source ./run.sh bmm/fix_3d_dma_indices.mlir  # run fixed version
```

This compiles the MLIR to NEFF, runs BIRSim, and compares against `kernel.py`. Artifacts (NEFF, BIR JSON, logs) are saved to `artifacts_<mlir_stem>/` next to the MLIR file.
