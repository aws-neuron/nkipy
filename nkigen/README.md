
# NKIPy KernelGen

A Python-to-NISA MLIR compiler for Trainium. Trace NumPy functions, annotate
with knobs, and lower through a pipeline of MLIR passes to NKI/NISA dialect.

- https://quip-amazon.com/UYybAmfN2E2k/Design-Doc-NKIPy-KernelGen-in-MLIR

## Features

- **`@trace` decorator** — trace Python functions with NumPy operations into linalg MLIR
- **`knob()` API** — annotate tensors with tiling, memory placement, and partitioning hints (similar to OpenMP pragmas)
- **MLIR pass pipeline** — prepare-arithmetic, assign-op-ids, infer-layout, knob-driven-tiling, legalize-layout, linalg-to-nisa, and more
- **Compiler Explorer** — interactive web UI for inspecting IR at each compilation stage

## Setup

NKIPyKernelGen depends on LLVM/MLIR and the NKI compiler (NISA dialect). Build
them first by following the
[NKI build instructions](https://github.com/aws-neuron/private-nki-staging/blob/main/docs/BUILDING.md).

Then set up this project:

```bash
# Set up the NKI development environment (venv, LLVM/MLIR paths, PYTHONPATH)
source scripts/setup_nki.sh

# Install the Python package (editable)
pip install -e .
```

## Quick Start

```python
import numpy as np
from nkigen import trace, knob

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
├── nkigen/                    # Python package (tracer, knob API, transforms)
├── mlir/                      # MLIR dialects and C++ passes
│   └── lib/Transforms/        # Pass implementations
├── tests/                     # Test suite
│   ├── unit/                  #   Unit tests for ops and execution engine
│   ├── passes/                #   MLIR pass tests (tiling, layout, etc.)
│   └── e2e/                   #   End-to-end compilation tests
└── scripts/                   # Environment setup and build scripts
```

## MLIR Passes

The compilation pipeline runs these passes in order:

| Pass | Description |
|------|-------------|
| `prepare-arithmetic` | Normalize arithmetic (e.g. `divf by N` → `mulf by 1/N`) |
| `assign-linalg-op-ids` | Tag each linalg op with a unique `nkipy.op_id` |
| `infer-layout` | Propagate tile_size/mem_space to unannotated ops (elementwise chains, reduce chains) |
| `knob-driven-tiling` | Generate transform dialect sequences from knob annotations |
| `transform-interpreter` | Apply the generated tiling transforms |
| `legalize-layout` | Insert 4D physical layout transformations |
| `linalg-to-nisa` | Lower linalg ops to NISA dialect |
| `prepare-for-nki` | Final preparation for NKI backend |

## Testing

```bash
# Set up environment first
source scripts/setup_nki.sh

# Run all tests
cd tests && python -m pytest . -v

# Run a specific test category
python -m pytest passes/infer_layout/ -v
python -m pytest e2e/ -v
python -m pytest unit/ -v
```

