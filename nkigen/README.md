# nkigen

A Python-to-NISA MLIR compiler for AWS Trainium. Write kernels as ordinary
NumPy functions, annotate placement and tiling with `knob`, and `nkigen`
traces them to linalg MLIR and lowers them through a pipeline of MLIR passes
to the NISA/NKI dialect for the Neuron compiler.

`nkigen` is one member of the **NKIPy** monorepo (a uv workspace), alongside
`nkipy` (the NumPy-like front end) and `spike` (NRT runtime bindings). It
provides the NumPy-trace → NISA lowering path: it handles tiling, memory
placement (SBUF/PSUM/HBM), layout legalization, spill/reload, and NISA
lowering on the way to a NEFF.

## Features

- **`@trace` decorator** — trace a Python function that uses NumPy operations
  into linalg MLIR.
- **`knob` API** — annotate tensors with loop tiling (`.tile_op`) and memory
  placement / physical layout (`.layout`), plus sibling-loop fusion
  (`knob.fuse`). Similar in spirit to OpenMP pragmas: hints, not rewrites.
- **Layout inference** — the `infer-layout` pass propagates tile sizes and
  memory spaces to unannotated intermediate ops, so you only annotate what you
  care about (or nothing at all — see `tests/e2e/test_auto_layout.py`).
- **MLIR pass pipeline** — a 26-stage pipeline from linalg-on-tensors to the
  NISA dialect (tiling, bufferization, layout legalization, spill/reload, NISA
  lowering). Inspectable after any pass.
- **Custom NISA ops** — drop precompiled NISA kernels into a traced function
  with `CustomOp`; they are inlined at lowering time.

## Setup

`nkigen` builds a small MLIR dialect + C++ passes (the `nkipy-opt` tool and the
`_mlir` Python bindings) against a pre-built LLVM/MLIR, and uses the `nki` and
`neuronx-cc` wheels for NISA lowering and compilation. Requirements:

- Python ≥ 3.10
- A pre-built LLVM/MLIR install matching the bundled MLIR Python bindings
- A C++ toolchain (defaults to `clang-22` / `clang++-22`)
- The Neuron wheels: `nki>=0.1.0`, `neuronx-cc` (installed from the Neuron
  package index)

The setup script wires up the environment (venv, LLVM/MLIR paths, Neuron index,
`PYTHONPATH`, and adds `nkipy-opt` to `PATH`):

```bash
# Wire up venv + LLVM/MLIR + Neuron deps, then build & install editable.
source scripts/setup_nki.sh
pip install -e .
```

`pip install -e .` runs CMake/Ninja under the hood to build `nkipy-opt` and the
`_mlir` bindings — no manual CMake invocation needed. Pure-Python changes take
effect immediately (editable install); changes to the C++ passes require a
rebuild (`pip install -e .` again).

Override knobs honored by `scripts/setup_nki.sh` / `setup.py`:

| Variable | Purpose |
|---|---|
| `LLVM_INSTALL_PREFIX` / `LLVM_BUILD_DIR` | Where the pre-built LLVM/MLIR lives |
| `NKIPY_VENV` | venv location (created/activated on first run) |
| `CC` / `CXX` | C/C++ compilers (default `clang-22` / `clang++-22`) |
| `BUILD_WITH` | `ninja` (default) or `make` |
| `NUM_THREADS` | parallelism for the C++ build |

## Quick Start

Trace a NumPy function, annotate with `knob`, and emit MLIR:

```python
import numpy as np
from nkigen import trace, knob

@trace(input_specs=[((128, 256), "f32")])
def add_scalar(x):
    result = x + 2.0
    # .tile_op = loop tile (one entry per linalg iterator)
    # .layout  = memory placement of the produced value
    knob.knob(result).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
    return result

# Trace to linalg MLIR (tensor IR, before any passes):
module = add_scalar.to_mlir()
print(module)

# Run the full compilation pipeline down to the NISA dialect:
from nkigen.transforms.nkipy_opt import apply_complete_knob_pipeline
nisa_ir = apply_complete_knob_pipeline(str(module), target="trn2")
print(nisa_ir)
```

A matmul + bias example showing matmul's 3-dim iteration-space tile and an SBUF
intermediate:

```python
@trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32"), ((256, 256), "f32")])
def matmul_add(a, b, bias):
    c = np.matmul(a, b)
    # matmul A[M,K]@B[K,N] -> C[M,N] has three iterators: [M_t, N_t, K_t]
    knob.knob(c).tile_op(tile_size=[128, 128, 128]).layout(mem_space="Sbuf")
    result = c + bias
    knob.knob(result).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
    return result
```

## The `knob` API

`knob.knob(tensor)` returns a chainable builder. Both methods are
side-effecting (they emit `nkipy.layout` / `nkipy.tile_op` ops) and return
`self`, so they chain:

```python
knob.knob(x).tile_op(tile_size=[64, 64]).layout(mem_space="Sbuf")
```

- **`.tile_op(tile_size=[...])`** — the loop tile for the op *producing* this
  tensor. One entry per linalg iterator, in iterator order (no reordering):
  - Elementwise: matches the output rank.
  - Reduction (e.g. `np.sum(x[M,N], axis=-1)`): matches the *input* rank
    `[M_t, N_t]`; the compiler knows which axis reduces.
  - Matmul (`A[M,K] @ B[K,N] -> C[M,N]`): three dims `[M_t, N_t, K_t]` (the
    last is the reduction/K tile).
- **`.layout(mem_space=..., partition_dim=..., tile_size=...)`** — memory
  placement and physical tile of the produced value.
  - `mem_space` ∈ `{"Hbm", "Psum", "Sbuf", "SharedHbm"}`.
  - `partition_dim` must be `< rank` (NISA requires partition dim 0; the
    `canonicalize-partition-dim` pass inserts transposes to make it so).
  - `tile_size` here is the value-shape placement tile. If omitted, the
    `infer-layout` pass projects it from `.tile_op`.

You don't have to annotate every tensor — `infer-layout` propagates tiling and
memory spaces to unannotated intermediates. Return values default to
`SharedHbm`; intermediates default to `Sbuf`.

`knob.fuse(a, b, ...)` hints that the `scf.for` loops producing the given
tensors should be fused into a single loop (each must already carry a matching
`.tile_op`).

## The nkipy dialect

The frontend lowers each `knob` call to one op in a tiny dialect
(`mlir/include/nkipy/Dialect/NkipyOps.td`). These ops ride along on the IR and
are consumed (and erased) by the passes that implement that knob. There are
exactly three knobs, plus an indexed-read op:

| Op | Emitted by | Carries |
|----|-----------|---------|
| `nkipy.tile_op` | `.tile_op(...)` | `loop_tile_size` — one entry per linalg iterator |
| `nkipy.layout` | `.layout(...)` | `mem_space`, `partition_dim`, `tile_size` (value-shape tile) |
| `nkipy.fuse_op` | `knob.fuse(...)` | the set of tensors whose loops to fuse |
| `nkipy.gather` | indexed reads | source / indices / output (lowers to `nisa.dma_copy_indirect`) |

The **Knob** column in the pipeline tables below maps each pass to the knob it
implements — `tile_op`, `layout`, `fuse_op`, or — for shared
hardware-rewrite / bufferization / cleanup passes.

## Compilation Pipeline

Defined in `nkigen/transforms/nkipy_opt.py` → `apply_complete_knob_pipeline()`.
Passes 1–25 are C++ (run via the `nkipy-opt` binary); pass 26 is Python (uses
the `nki` wheel's MLIR bindings). C++ pass sources live in
`mlir/lib/Transforms/`. Each phase below is a group of subpasses that run in
the listed order.

### Phase 0 — Linalg rewrites for NISA constraints

The `canonicalize-linalg-for-nisa` pass group: rewrites that adapt the program
to NISA hardware before tiling. Stop after it by name
(`stop_after="canonicalize-linalg-for-nisa"`).

| # | Pass | Knob | What it does |
|---|------|------|--------------|
| 1 | `remove-redundant-zero-fill` | — | Drop `linalg.fill(0)` feeding matmul (NISA matmul auto-zeros PSUM) |
| 2 | `prepare-arithmetic` | — | `linalg.div(A,B)` → `mul(A, reciprocal(B))` (NISA has no divide) |
| 3 | `decompose-batch-matmul` | — | `linalg.batch_matmul` → `scf.for` + 2D `linalg.matmul` |

### Phase 1 — Layout inference, partition-dim canonicalization, tiling, fusion

The `tile_op` / `layout` / `fuse_op` knobs are all realized here, on tensor IR.

| # | Pass | Knob | What it does |
|---|------|------|--------------|
| 4 | `infer-layout` | tile_op + layout | Infer tile_size / mem_space / partition_dim for unannotated ops |
| 5 | `canonicalize-partition-dim` | layout | Insert `linalg.transpose` so `partition_dim=0` everywhere |
| 6 | `assign-linalg-op-ids` | tile_op | Stamp a unique `nkipy.op_id` on each linalg op |
| 7 | `knob-driven-tiling` | tile_op | Emit transform-dialect IR that tiles each op per its tile sizes |
| 8 | `apply-and-strip-transforms` | tile_op | Run `@__transform_main`, then erase the transform module |
| 9 | `knob-driven-fusion` | fuse_op | Fuse sibling `scf.for` loops sharing a `nkipy.fuse_op` |
| 10 | `canonicalize-loop-step` | tile_op | Normalize `scf.for` steps to 1 |

### Phase 2 — Bufferization

| # | Pass | Knob | What it does |
|---|------|------|--------------|
| 11 | `one-shot-bufferize` | — | Tensor IR → memref IR (upstream MLIR) |
| 12 | `canonicalize` | — | Fold/simplify memref ops |

### Phase 3 — Memory-space annotation + reshape canonicalization

| # | Pass | Knob | What it does |
|---|------|------|--------------|
| 13 | `eliminate-uninitialized-copies` | — | Drop `memref.copy` from never-written allocs (e.g. PSUM init) |
| 14 | `canonicalize` | — | Clean up dead subview chains |
| 15 | `annotate-memory-space` | layout | Apply NISA mem-space attrs; mark func args SharedHbm; erase `nkipy.layout` |
| 16 | `canonicalize-reshape` | layout | Classify `expand`/`collapse_shape` by mem_space + partition_dim (view vs alloc+copy) |
| 17 | `eliminate-same-memspace-copy` | — | Remove redundant same-space copies (e.g. SBUF→SBUF) |
| 18 | `canonicalize` | — | DCE dead allocs from eliminated copies |

### Phase 4 — Layout legalization, spill/reload, memref finalization

| # | Pass | Knob | What it does |
|---|------|------|--------------|
| 19 | `legalize-layout` | layout | Transform SBUF allocs to physical multi-D layout (partition dim ≤ 128) |
| 20 | `canonicalize` | — | Fold collapse_shape chains, simplify affine maps |
| 21 | `simplify-linalg` | — | Decompose high-rank transposes to 2D; trivial-broadcast generics → named ops |
| 22 | `insert-spill-reload` | layout | Insert SBUF↔HBM spill/reload on memory pressure (Belady's MIN) |
| 23 | `insert-memref-dealloc` | — | Insert `memref.dealloc` at each alloc's scope end |
| 24 | `cse` | — | Common subexpression elimination |
| 25 | `canonicalize` | — | DCE unused subviews and final cleanup |

### Phase 5 — NISA lowering (Python)

| # | Pass | Knob | What it does |
|---|------|------|--------------|
| 26 | `py:linalg-to-nisa` | — | Lower linalg/memref/scf to the NISA dialect via the `nki` bindings (see below) |

The final pass pattern-matches `linalg.add/sub/mul` → `nisa.tensor_tensor_arith`,
`linalg.matmul` → `nisa.matmul`, `memref.copy(HBM↔SBUF)` → `nisa.dma_copy`,
`linalg.exp` → `nisa.activation(op=exp)`, scalar-broadcast ops →
`nisa.tensor_scalar_arith`, `linalg.reciprocal` → `nisa.reciprocal`, transposes
→ `nisa.dma_transpose`, `memref.dealloc` → `nisa.release`. It also inlines
custom-op bodies and adds the `nisa.target` attribute.

> This single Python pass subsumes what used to be three separate C++ passes
> (`linalg-to-nisa`, `resolve-custom-ops`, `prepare-for-nki`). NISA lowering
> happens in Python because `nkipy-opt` does not link the NISA dialect.

## Inspecting Intermediate IR

### From Python

```python
from nkigen.transforms.nkipy_opt import apply_complete_knob_pipeline

# Dump every intermediate file:
apply_complete_knob_pipeline(mlir_str, dump_dir="debug_outputs/")

# Stop after a named pass:
apply_complete_knob_pipeline(mlir_str, stop_after="legalize-layout", dump_dir="debug_outputs/")

# Stop after pass N (1-indexed, on the flattened pipeline):
apply_complete_knob_pipeline(mlir_str, stop_after=11)  # after one-shot-bufferize

# For repeated passes like canonicalize, use "name:N" for the Nth occurrence:
apply_complete_knob_pipeline(mlir_str, stop_after="canonicalize:3")
```

`stop_after` also accepts a pass-group name (e.g. `"canonicalize-linalg-for-nisa"`),
which resolves to the group's last member.

### From a test (`--dump-ir`)

Any test can dump the IR after every pass:

```bash
pytest tests/e2e/test_rope.py::test_rope --dump-ir -v -s
```

This writes numbered files (`00_input.mlir`, `01_remove-redundant-zero-fill.mlir`,
…, `26_linalg-to-nisa.mlir`) to a temp dir (printed to the console). If a
compilation crashes, the harness auto-dumps pass-by-pass and prints the path.

### With `nkipy-opt` directly

```bash
nkipy-opt --legalize-layout input.mlir                       # one pass
nkipy-opt --annotate-memory-space --legalize-layout in.mlir  # chain passes
nkipy-opt --mlir-print-ir-after-all --legalize-layout in.mlir 2>&1
```

## Project Structure

```
nkigen/                       # this workspace member (under the NKIPy monorepo)
├── nkigen/                   # Python package
│   ├── trace.py              #   @trace decorator (NumPy fn -> linalg MLIR)
│   ├── knob.py               #   knob API (tile_op / layout / fuse)
│   ├── op_vtable.py          #   NumPy op -> MLIR lowering table
│   ├── traced_array.py       #   TracedArray (NumPy-like over MLIR SSA values)
│   ├── builder.py            #   IR construction layer
│   ├── control_flow.py       #   fori_loop
│   ├── custom_op.py          #   CustomOp (inline precompiled NISA kernels)
│   ├── execution.py          #   verify_against_numpy (LLVM JIT vs NumPy)
│   └── transforms/
│       ├── nkipy_opt.py      #   pipeline orchestration (apply_complete_knob_pipeline)
│       └── linalg_to_nisa_py.py  # Python NISA lowering (pass 26)
├── mlir/                     # nkipy dialect + C++ passes
│   ├── include/nkipy/        #   dialect + pass TableGen (.td)
│   ├── lib/Transforms/       #   pass implementations
│   └── tools/nkipy-opt/      #   the nkipy-opt driver
├── tests/                    # passes/ (FileCheck), e2e/, unit/  — see tests/README.md
├── scripts/                  # setup_nki.sh (environment wiring)
├── setup.py                  # CMake-driven build of mlir/ -> nkigen/_mlir
└── pyproject.toml
```

## Public API

```python
from nkigen import trace, knob          # core
from nkigen import TracedArray, CustomOp # array wrapper, custom NISA ops
from nkigen import verify_against_numpy  # LLVM-JIT-vs-NumPy checker
from nkigen.apis import fori_loop        # bounded for-loop in traced code
```

Supported NumPy surface (see `nkigen/op_vtable.py`): elementwise ufuncs
(`add`, `subtract`, `multiply`, `divide`, `exp`, `log`, `sqrt`, `square`,
`reciprocal`, `negative`, `abs`, trig, `tanh`, `maximum`/`minimum`, comparisons,
bitwise/logical, `matmul`, …) and array functions (`np.matmul`, `np.transpose`,
`np.reshape`, `np.sum`/`prod`/`max`/`min`/`mean`/`std`, `np.concatenate`,
`np.split`, `np.expand_dims`, `np.broadcast_to`, `np.where`, `np.take`).

## Testing

```bash
source scripts/setup_nki.sh   # set up the environment first

pytest tests/passes/          # per-pass FileCheck tests
pytest tests/e2e/             # end-to-end (auto-skips when no Trainium device)
pytest tests/unit/            # Python-level unit tests

pytest tests/e2e/test_rope.py -v       # a single file
pytest -k test_add_2d                  # by name substring
```

Test modes (from `tests/harness.py`): `Mode.LLVM` (LLVM JIT vs NumPy — requires
`stop_after`), `Mode.HW` (run on Trainium, auto-skips without a device),
`Mode.STRING_CHECK` (assert substrings in the compiled IR), `Mode.FILECHECK`
(run LLVM FileCheck). See `tests/README.md` for the full harness reference.

## Contributing & License

`nkigen` is part of the NKIPy monorepo. See the repo-root `CONTRIBUTING.md` for
contribution guidelines and `CODE_OF_CONDUCT.md` for community standards.
Licensed under Apache License 2.0 (repo-root `LICENSE`).
