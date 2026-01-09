# Installation

This guide covers how to install NKIPy and its dependencies on an AWS Trainium instance.

## Prerequisites

NKIPy requires a Trainium instance with the Neuron Driver and Runtime installed.

### Install Neuron Driver and Runtime

If you are using a **Neuron Multi-Framework DLAMI**, the driver and runtime are already installed. You can skip to the next section.

Otherwise, follow the [Neuron Setup Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/torch-neuronx.html#setup-torch-neuronx) up to the "**Install Drivers and Tools**" section for your OS. Note that NKIPy does not require PyTorch, but it supports Torch tensors if available.

## Installation with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. It's the recommended way to install NKIPy.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and Install

```bash
git clone https://github.com/aws-neuron/nkipy.git
cd nkipy

# Install all packages (creates .venv automatically)
uv sync
```

This single command will:
- Create a `.venv` virtual environment
- Install `nkipy` and `spike` in editable mode
- Install all dependencies including `neuronx-cc` from the Neuron repository
- Install development tools (pytest, ruff, mypy)

### Running Commands

```bash
# Run Python with the workspace environment
uv run python examples/playground/simple_nkipy_kernel.py 

# Start an interactive Python session
uv run python
```

### Building Wheels

```bash
# Build nkipy
uv build --package nkipy

# Build spike
uv build --package spike

# Output will be in dist/
ls dist/
```

---

## Alternative: Installation with pip

If you prefer using pip instead of uv, follow these instructions.

### Create a Virtual Environment

```bash
python3.10 -m venv nkipy_venv
source nkipy_venv/bin/activate
```

### Configure Neuron Repository

NKIPy depends on the Neuron Compiler (`neuronx-cc`). Configure pip to use the Neuron repository:

```bash
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
```

### Install Packages

```bash
# Install both packages in editable mode
pip install -e nkipy -e spike

# Or install without editable mode
pip install nkipy spike
```

### Building Wheels with pip

```bash
# Install build tool
pip install build

# Build wheel packages
python -m build nkipy --wheel --outdir dist
python -m build spike --wheel --outdir dist

# Install the built wheels
pip install dist/*.whl
```

---

## Verifying Installation

After installation, verify that everything works:

```bash
# With uv
uv run python -c "import nkipy; import spike; print('Installation successful!')"

# With pip (after activating venv)
python -c "import nkipy; import spike; print('Installation successful!')"
```

## Troubleshooting

### neuronx-cc not found

If you see an error about `neuronx-cc` not being found:

**With uv:** The Neuron repository is already configured in `pyproject.toml`. Try running `uv sync --refresh`.

**With pip:** Make sure you've configured the extra index URL:
```bash
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
```

### spike build fails

The `spike` package contains C++ code that requires compilation. Ensure you have:

- CMake installed (`apt install cmake`)
- A C++ compiler (gcc/clang)
- Python development headers (`apt install python3-dev`)
- Neuron Runtime (libnrt) installed
