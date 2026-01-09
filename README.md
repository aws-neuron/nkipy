# NKIPy

This repository contains packages, examples, and docs for NKIPy.

## Packages

### NKIPy

NKIPy empowers researchers to quickly prototype ideas from one kernel to the
full ML stack on Trainium through a kernel-driven approach. It supports
NumPy-like syntax, uses the Neuron compiler to lower kernels to NKI or binary
while incorporating directives to control compilation, and includes an agile
runtime to support kernel execution.

### Spike Runtime

Spike provides a modern, efficient Python interface to AWS Neuron Runtime (NRT) through optimized C++ bindings. It enables direct execution of compiled NEFF models and tensor management on AWS Neuron devices with minimal overhead.

## Installation

### Quick Start with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies and virtual environments automatically.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/aws-neuron/nkipy.git
cd nkipy

# Install all packages in editable mode
uv sync
```

This will:
- Create a `.venv` virtual environment
- Install `nkipy`, `spike`, and all dependencies (including `neuronx-cc` from the Neuron repository)
- Install development tools (pytest, ruff, mypy)

### Running Commands

```bash
# Run Python with the workspace environment
uv run python -c "import nkipy; import spike"

# Run a small NKIPy kernel
uv run python examples/playground/simple_nkipy_kernel.py 
```

### Alternative: pip Installation

If you prefer pip, see the [Installation Guide](./docs/installation.md) for detailed instructions.

## Building Wheels

To build distribution wheels:

```bash
# Build nkipy
uv build --package nkipy

# Build spike
uv build --package spike
```

Wheels will be created in the `dist/` directory.

## Documentation

For more information, please refer to the detailed documentation:

- [Installation Guide](./docs/installation.md)
- [Quickstart](./docs/quickstart.md)
- [Tutorials](./docs/tutorials/index.md)
- [Spike README](./spike/README.md)
