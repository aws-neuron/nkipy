
# NKIPy KernelGen

A Python-to-NISA MLIR compiler for Trainium. Trace NumPy functions, annotate
with knobs, and lower through a pipeline of MLIR passes to NKI/NISA dialect.

## Features

- **`@trace` decorator** — trace Python functions with NumPy operations into linalg MLIR
- **`knob()` API** — annotate tensors with tiling, memory placement, and partitioning hints (similar to OpenMP pragmas)
- **MLIR pass pipeline** — prepare-arithmetic, assign-op-ids, infer-layout, knob-driven-tiling, legalize-layout, linalg-to-nisa, and more
- **Compiler Explorer** — interactive web UI for inspecting IR at each compilation stage

## Setup

NKIPyKernelGen depends on LLVM/MLIR. Install LLVM with MLIR support first.

```bash
# Activate your Python virtual environment
source <your-venv>/bin/activate

# Install the Python package (editable) — builds C++ passes via CMake
pip install -e .
```

## Quick Start

```python
import numpy as np
from nkipy_kernelgen import trace, knob

@trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
def matmul_add(A, B):
    C = np.matmul(A, B)
    knob.knob(C, mem_space="Sbuf", tile_size=[128, 128, 128])
    result = np.exp(C)
    knob.knob(result, mem_space="Sbuf", tile_size=[128, 128])
    return result
```

The `knob()` API injects `nkipy.annotate` ops into the IR to guide tiling and
buffer placement. The `infer-layout` pass propagates annotations to unannotated
intermediate ops.

## Project Structure

```
NKIPyKernelGen/
├── nkipy_kernelgen/           # Python package (tracer, knob API, transforms)
├── mlir/                      # MLIR dialects (NKIPy) and C++ passes
│   └── lib/Transforms/        # C++ pass implementations (phases 0–4)
├── compiler_explorer/         # Compiler Explorer web UI integration
├── examples/                  # Example kernels and usage patterns
└── tests/                     # Test suite
    ├── unit/                  #   Unit tests for ops and execution engine
    ├── passes/                #   MLIR pass tests (tiling, layout, etc.)
    ├── e2e/                   #   End-to-end compilation tests
    └── python/                #   Python-level pass and rewrite tests (lit-style)
```

## MLIR Passes

The compilation pipeline runs 24 passes in 5 phases. Key passes:

| Phase | Key Passes | Description |
|-------|------------|-------------|
| 0. Arithmetic prep | `remove-redundant-zero-fill`, `prepare-arithmetic` | Remove redundant zero fills, rewrite div → mul+reciprocal |
| 1. Layout & tiling | `infer-layout`, `canonicalize-partition-dim`, `knob-driven-tiling`, `apply-and-strip-transforms` | Infer tile sizes/mem spaces, canonicalize partition dims, tile via transform dialect |
| 2. Bufferization | `one-shot-bufferize` | Convert tensor IR to memref IR |
| 3. Memory | `annotate-memory-space`, `canonicalize-reshape`, `eliminate-same-memspace-copy` | Apply NISA memory spaces, canonicalize reshapes, eliminate redundant copies |
| 4. Layout & spill | `legalize-layout`, `simplify-linalg`, `insert-spill-reload`, `insert-memref-dealloc` | Physical layout legalization, SBUF spill/reload, dealloc insertion |
| 5. NISA lowering | `py:linalg-to-nisa` | Lower linalg/memref to NISA ops (Python, via `nki` wheel) |

See `CLAUDE.md` for the full 24-pass pipeline with detailed descriptions.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test category
python -m pytest tests/passes/infer_layout/ -v
python -m pytest tests/e2e/ -v
python -m pytest tests/unit/ -v
```

## Compiler Explorer

Interactive web UI for inspecting IR at each compilation stage:

```bash
cd compiler_explorer
./setup.sh        # Clone Compiler Explorer and start server
# Open http://localhost:10240, select NKIPy MLIR as the compiler
```

Use `--stop=N` to view IR after each pass, `--sim` for BIR simulation
verification, `--hw` for hardware compilation to NEFF.
