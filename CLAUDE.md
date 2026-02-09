# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NKIPy is an experimental NumPy-like tensor-level programming layer for AWS Trainium hardware, built on top of NKI (Neuron Kernel Interface). It is a prototype/alpha project, not an official AWS product.

The repository is a **uv workspace monorepo** with two packages:
- **nkipy** (`nkipy/`) — Core tracing, compilation, and tensor operations
- **spike** (`spike/`) — C++ runtime (nanobind bindings to libnrt) for device execution

## Common Commands

```bash
# Install all packages (creates .venv)
uv sync

# Install with all dependency groups (test, docs, examples)
uv sync --all-groups

# Run tests
uv run pytest                                        # all tests
uv run pytest tests/unit/test_tensor_api.py          # single file
uv run pytest tests/unit/test_tensor_api.py::test_fn # single test
uv run pytest -x                                     # stop on first failure

# Lint and format
uv run ruff check .            # lint (E, F, I rules)
uv run ruff check . --fix      # auto-fix lint issues
uv run ruff format .           # format code
uv run ruff format --check .   # check formatting without changes

# Type checking
uv run mypy nkipy/

# Build wheels
uv build --package nkipy
uv build --package spike

# Build docs
uv run make -C docs html
```

## Architecture

### Execution Modes

Kernels are written as plain NumPy functions and can run in three modes:
1. **Pure NumPy** — Direct CPU execution, no decorator needed
2. **Simulation** — `@simulate_jit` traces computation and simulates via HLO
3. **Baremetal** — `@baremetal_jit` compiles to NEFF via neuronx-cc and runs on Trainium hardware via Spike

### NKIPy Core (`nkipy/src/nkipy/`)

- **`core/ops/`** — Operation registry using the `Op` dispatcher pattern. Each op registers backend implementations (currently `hlo` and `cpu`) via `@op.impl('backend')`. Operations are organized by category: `binary.py`, `unary.py`, `reduce.py`, `transform.py`, `indexing.py`, `creation.py`, `conv.py`, `nn.py`, `collectives.py`, `linalg.py`.
- **`core/ops/_registry.py`** — Global backend state management and the `Op` class that dispatches to the correct implementation based on the active backend.
- **`core/tensor.py`** — `NKIPyTensorRef`, the main tensor type during tracing. Implements `TensorArithmeticMixin` for operator overloading.
- **`core/trace.py`** — Tracing engine that captures computation graphs.
- **`core/compile.py`** — HLO lowering and compilation via neuronx-cc.
- **`core/backend/hlo.py`** — HLO backend interface.
- **`runtime/decorators.py`** — `@simulate_jit` and `@baremetal_jit` decorators.
- **`runtime/execute.py`** — Execution backends (simulation and device).
- **`runtime/baremetal_executor.py`** — Hardware execution via Spike.
- **`distributed/`** — Multi-device collective operations.
- **`third_party/`** — Generated protobuf files for XLA HLO representation. Built during `nkipy` package build via custom `setup.py` that compiles `.proto` files.

### Spike Runtime (`spike/`)

C++ runtime with nanobind Python bindings. Architecture layers:
```
Python (SpikeTensor, SpikeModel) → spike_singleton.py (lifecycle)
    → C++ nanobind bindings → libnrt (Neuron Runtime)
```

- Built with scikit-build-core + CMake
- Singleton pattern: 1 Python process → 1 Spike instance → 1 libnrt instance
- Provides `spike-run` CLI command

### Testing

Tests are in `tests/` (nkipy) and `spike/tests/` (spike). `conftest.py` configures `NEURON_RT_VISIBLE_CORES` isolation for pytest-xdist parallel execution on hardware.

## Code Style

- **Python**: Ruff (line-length 88, double quotes, isort via `I` rule)
- **C++**: clang-format
- Pre-commit hooks enforce both via `.pre-commit-config.yaml`
- Protobuf files in `nkipy/third_party/` are generated — do not edit manually
