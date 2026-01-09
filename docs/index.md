# NKIPy

NKIPy is an experimental project that provides a NumPy-like tensor-level programming layer on top of NKI (Neuron Kernel Interface). It enables developers to write kernels for rapid prototyping of their ML programs at good performance, while abstracting away low-level hardware details and tiling strategies from developers.

NKIPy is designed for AWS Trainium and depends on components of the AWS Neuron SDK to function. While NKIPy uses Neuron SDK components, it is not an official part of the Neuron SDK. It requires the Neuron Compiler and Neuron Runtime to compile and execute kernels. NKIPy currently lowers tensor operations to HLO and then calls Neuron Compiler (neuronx-cc) to generate NKI code or executables as outputs.

```{warning}
This project is at a prototype/alpha level and is not intended for customers to use in production deployments. You should expect bugs, incomplete features, and breaking API changes. There is no guarantee of API stability, ongoing maintenance, or future support at this time. We welcome you to experiment with NKIPy, and we appreciate feedback, bug reports, and contributions via GitHub Issues and Pull Requests.
```

## Key Features

- **NumPy-like API**: Write kernels using familiar NumPy syntax with Python control flow
- **HLO Lowering**: Tensor operations are traced and lowered to HLO for compilation
- **Neuron Compiler Integration**: Direct integration with neuronx-cc for generating NKI or executables

## Runtime

For kernel execution, NKIPy uses **Spike**, a lightweight Pythonic runtime layer for AWS Neuron. See the Spike README for details on the runtime architecture and API.

## Frequently Asked Questions

**Q: Is NKIPy an official AWS product?**  
No. NKIPy is an experimental research project and is not part of the AWS Neuron SDK or any official AWS product offering.

**Q: Can I use NKIPy in production?**  
While there is nothing stopping you from using it in production, NKIPy is a prototype intended for experimentation and rapid prototyping only. You should expect bugs, incomplete features, and breaking API changes without notice. We are not providing time critical support for any production issues for this project.

**Q: Will NKIPy be officially supported or maintained?**  
There are no plans or commitments for official support, ongoing maintenance, or API stability. Use at your own risk.

**Q: What is the relationship between NKIPy and the Neuron SDK?**  
NKIPy depends on Neuron SDK components (Neuron Compiler and Neuron Runtime) to function, but it is a separate experimental project and not part of the official SDK.

**Q: How can I contribute or report issues?**  
We welcome feedback, bug reports, and contributions through GitHub Issues and Pull Requests.

**Q: Who should use NKIPy?**  
Researchers and developers who want to experiment with NumPy style kernel development on AWS Trainium and are comfortable working with unstable, experimental software, including self-solving issues using the open source codebase.

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
dev_guide/known_issues
api/index
