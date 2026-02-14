# Quickstart Guide

Welcome to NKIPy! This guide will help you get started quickly.

## Installation

For installation instructions, see the [README](https://github.com/aws-neuron/nkipy/blob/main/README.md#installation) or the detailed [Installation Guide](installation.md).

Quick install with uv:
```bash
git clone https://github.com/aws-neuron/nkipy.git
cd nkipy
uv sync
```

## Basic Usage

NKIPy kernels can run in two modes:

1. **CPU (Direct)** - Call kernels directly as NumPy code for prototyping
2. **Trainium Hardware** - Use `@baremetal_jit` to compile and run on Trainium

For complete code examples, see the [Basic Usage section in the README](https://github.com/aws-neuron/nkipy/blob/main/README.md#basic-usage).

## Learning Path

### 1. Start with Tutorials

The best way to learn NKIPy is through hands-on tutorials:

- **[Simple Tutorial](tutorials/01_simple.ipynb)** - Learn the basics with a softmax kernel
  - Define a NKIPy kernel
  - Run on CPU and on Trainium hardware
  - Understand tracing and compilation

- **[NKIPy to NKI](tutorials/02_nkipy_to_nki.ipynb)** - Learn how NKIPy lowers to NKI code

- **[MLP Tutorial](tutorials/03_mlp.ipynb)** - Build a multi-layer perceptron

See all tutorials in the [Tutorials Index](tutorials/index.md).

### 2. Explore Examples

Check out the `examples/` directory for more complex use cases:

- `examples/playground/simple_nkipy_kernel.py` - Simple standalone example
- `examples/models/qwen3_embedding/` - Full model implementation

### 3. Dive Deeper

Once you're comfortable with the basics:

- [User Guide](user_guide/tracing_architecture.md) - Understand NKIPy's architecture
- [API Reference](api/index.md) - Detailed API documentation
- [Developer Guide](dev_guide/extending_language.md) - Contribute to NKIPy

## Key Concepts

- **Tracing**: NKIPy traces your NumPy-like code to build a computation graph
- **Compilation**: The traced graph is lowered and compiled to binary 
- **Execution**: Compiled kernels run on Trainium hardware 

## Getting Help

- Report bugs via [GitHub Issues](https://github.com/aws-neuron/nkipy/issues)
- See the [README FAQ](https://github.com/aws-neuron/nkipy/blob/main/README.md#frequently-asked-questions) for common questions
