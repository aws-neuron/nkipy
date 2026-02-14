# NKIPy Documentation

Welcome to the NKIPy documentation! NKIPy provides a NumPy-like tensor-level programming layer on top of [NKI (Neuron Kernel Interface)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) for AWS Trainium.

For an overview of the project, installation instructions, and basic usage examples, see the [README](https://github.com/aws-neuron/nkipy/blob/main/README.md).

## Key Features

- **NumPy-like API**: Write kernels using familiar NumPy syntax with Python control flow
- **HLO Lowering**: Tensor operations are traced and lowered to HLO for compilation
- **Neuron Compiler Integration**: Direct integration with neuronx-cc for generating NKI or executables

## Runtime

For kernel execution, NKIPy uses **Spike**, a lightweight Pythonic runtime layer for AWS Neuron. See the Spike README for details on the runtime architecture and API.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Learning NKIPy

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/indexing_slicing_reference
user_guide/tracing_architecture
```

```{toctree}
:maxdepth: 2
:caption: Developing NKIPy

dev_guide/extending_language
dev_guide/testing
dev_guide/building_docs
api/index
```
