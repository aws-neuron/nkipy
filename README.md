# NKIPy

**NKIPy is an experimental project that provides a NumPy-like tensor-level programming layer on top of [NKI (Neuron Kernel Interface)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html).** It enables developers to write kernels for rapid prototyping of their ML programs at good performance, while abstracting away low-level hardware details and tiling strategies from developers.

**NKIPy is designed for AWS Trainium and depends on components of the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/index.html) to function. While NKIPy uses Neuron SDK components, it is not an official part of the Neuron SDK.** It requires the [Neuron Compiler](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/index.html) and [Neuron Runtime](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/index.html) to compile and execute kernels. NKIPy currently lowers tensor operations to HLO and then calls Neuron Compiler (`neuronx-cc`) to generate NKI code or executables as outputs.

**This project is at a prototype/alpha level and is not intended for customers to use in production deployments.** You should expect bugs, incomplete features, and breaking API changes. There is no guarantee of API stability, ongoing maintenance, or future support at this time. We welcome you to experiment with NKIPy, and we appreciate feedback, bug reports, and contributions via GitHub Issues and Pull Requests.


## Frequently Asked Questions

**Q: Is NKIPy an official AWS product?**  
No. NKIPy is an experimental research project and is not part of the AWS Neuron SDK or any official AWS product offering.

**Q: Can I use NKIPy in production?**  
While there is nothing stopping you from using it in production, NKIPy is a prototype intended for experimentation and rapid prototyping only. You should expect incomplete features and breaking API changes without notice. We are not providing time critical support for any production issues for this project.

**Q: Will NKIPy be officially supported or maintained?**  
There are no plans or commitments for official support, ongoing maintenance, or API stability. Use at your own risk.

**Q: What is the relationship between NKIPy and the Neuron SDK?**  
NKIPy depends on Neuron SDK components (Neuron Compiler and Neuron Runtime) to function, but it is a separate experimental project and not part of the official SDK.

**Q: How can I contribute or report issues?**  
We welcome feedback, bug reports, and contributions through GitHub Issues and Pull Requests.

**Q: Who should use NKIPy?**  
Researchers and developers who want to experiment with NumPy style kernel development on AWS Trainium and are comfortable working with unstable, experimental software, including self-solving issues using the open source codebase.


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

# For additional features (examples, docs, testing):
uv sync --all-groups
```

This will:
- Create a `.venv` virtual environment
- Install `nkipy`, `spike`, and all dependencies (including `neuronx-cc` from the Neuron repository)

The `--all-groups` flag additionally installs:
- **test**: pytest, ruff, mypy for testing and linting
- **examples**: torch, transformers, ipython for running examples
- **docs**: sphinx and related tools for building documentation

### Running Commands

**Note:** Activate the environment with `source .venv/bin/activate`, or use `uv run python your_script.py` to run without activation.

```bash
# Run Python with the workspace environment
source .venv/bin/activate
python -c "import nkipy; import spike"

# Run a small NKIPy kernel
python examples/playground/simple_nkipy_kernel.py 
```

### Building Wheels

To build distribution wheels:

```bash
# Build nkipy
uv build --package nkipy

# Build spike
uv build --package spike
```

Wheels will be created in the `dist/` directory.

### Alternative: pip Installation

If you prefer pip, see the [Installation Guide](./docs/installation.md) for detailed instructions.

## Basic Usage

NKIPy kernels look like NumPy functions and can run in three modes.

### 1. Pure NumPy (CPU)

Write and run kernels directly as NumPy code:

```python
import numpy as np

def softmax_kernel(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_x

# Run on CPU with NumPy
x = np.random.rand(2, 128).astype(np.float32)
result = softmax_kernel(x)
```

### 2. Simulation Mode

Use the `@simulate_jit` decorator to trace and simulate execution:

```python
import numpy as np
from nkipy.runtime import simulate_jit

@simulate_jit
def softmax_kernel(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_x

# Automatically traced and simulated
x = np.random.rand(2, 128).astype(np.float32)
result = softmax_kernel(x)
```

### 3. Trainium Hardware

Use the `@baremetal_jit` decorator to compile and run on Trainium:

```python
import numpy as np
from nkipy.runtime import baremetal_jit

@baremetal_jit
def softmax_kernel(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    sum_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_x

# Compiled and executed on Trainium hardware
x = np.random.rand(2, 128).astype(np.float32)
result = softmax_kernel(x)
```

The `@baremetal_jit` decorator compiles the kernel and executes it on Trainium hardware.

## Documentation

For more information, please refer to the detailed documentation:

- [Installation Guide](./docs/installation.md)
- [Quickstart](./docs/quickstart.md)
- [Tutorials](./docs/tutorials/index.md)
- [Spike README](./spike/README.md)
