# NKIPy Tracing Architecture

## Overview

NKIPy provides a Python-based interface for writing ML kernels that compile to AWS Neuron hardware. The tracing system translates tensor operations into HLO (High Level Operations) suitable for compilation to Neuron Executable File Format (NEFF).

Users write normal-looking Python/NumPy code, but during tracing, operations generate HLO instead of computing values.

## How Tracing Works

NKIPy uses a **shallow layer design** where operations dispatch directly to HLO construction. The tracing flow is straightforward:

```
User Function (Python/NumPy)
    ↓
Tracing Context Activated
    ↓
Arguments Replaced with Tensor Wrappers
    ↓
Operations Intercepted & Recorded as HLO
    ↓
HLO Module Generated
    ↓ (compilation)
Neuron Compiler (neuronx-cc)
    ↓
NEFF Binary
```

### Tensor Wrappers

When tracing begins, input arrays are replaced with special tensor wrapper objects that:

- Intercept NumPy operations via `__array_function__` and `__array_ufunc__` protocols
- Dispatch operations directly to HLO builders
- Track metadata (shape, dtype, source location)
- Generate HLO instead of computing values

This allows users to write familiar NumPy code while NKIPy captures the computation graph.

### Operation Dispatch

Operations are dispatched through a simple registry pattern:

1. NumPy function called on tensor wrapper (e.g., `np.add(a, b)`)
2. Wrapper's `__array_function__` intercepts the call
3. Registry looks up the corresponding HLO builder
4. HLO operation is emitted to the module

NKIPy also provides additional operations beyond NumPy (e.g., `ops.matmul`, `ops.conv2d`) that follow the same dispatch pattern.

## Supported Operations

| Category           | Example Operations            |
| ------------------ | ----------------------------- |
| Elementwise Unary  | abs, exp, log, sin, cos       |
| Elementwise Binary | add, multiply, subtract       |
| Comparison         | equal, less, greater          |
| Reduction          | sum, max, min, mean           |
| Shape              | transpose, reshape, broadcast |
| Indexing           | take, gather, scatter         |
| Array Creation     | zeros, ones, full             |
| Linear Algebra     | matmul, conv2d, conv3d        |
| Distributed        | all_reduce, all_gather        |

See the [Operations Reference](../api/ops.md) for a complete list.

## Special Features

### IO Aliasing

NKIPy supports in-place modification semantics for mutable parameters:

```python
def kernel(x: tensor[mutable, float32, (N,)]):
    x[:] = x + 1  # In-place modification
    return x      # Returns aliased input
```

The compiler respects aliasing information for memory optimization.

### Source Location Tracking

Python source locations (filename, line number) are embedded in HLO operations for debugging. This information is preserved through compilation and used for error reporting.

### IO Naming

Parameters and outputs can be named for stable NEFF interfaces:

```python
@trace
def kernel(input_tensor):
    ...
    return output_tensor
```

Names are preserved through compilation and used in NEFF for input/output identification.
