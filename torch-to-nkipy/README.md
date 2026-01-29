# Torch-to-NKIPy

Run your PyTorch models on AWS Trainium accelerators with just a few lines of code.

## What is Torch-to-NKIPy?

Torch-to-NKIPy is a PyTorch backend that bridges the gap between your existing PyTorch code and AWS Trainium hardware. Instead of rewriting your models, you simply add `torch.compile(backend="nkipy")` and your code runs on Trainium.

**Why use Torch-to-NKIPy?**

- **Zero model rewrites** - Your existing PyTorch models work as-is
- **Familiar APIs** - Use standard PyTorch patterns you already know
- **Flexible execution** - Mix CPU and Trainium operations in the same model
- **Custom kernels** - Inject optimized NKI kernels when you need extra performance
- **Distributed ready** - Built-in support for multi-device tensor parallelism

## Features

- **Seamless Integration**: Drop-in replacement for PyTorch's default backend via `torch.compile(backend="nkipy")`
- **Fine-grained Control**: Mark subgraphs for selective compilation and control CPU/Trainium execution boundaries
- **Custom NKI Kernels**: Inject optimized custom NKI (Neuron Kernel Interface) kernels for enhanced performance
- **Comprehensive Op Support**: Extensive ATen operation registry with 50+ operations including:
  - Tensor operations (add, mul, matmul, etc.)
  - Activation functions (relu, gelu, sigmoid, etc.)
  - Normalization (layer_norm, softmax, etc.)
  - Indexing and slicing operations
  - Collective operations for distributed training
- **Automatic Tensor Management**: Handles runtime tensor placement and data movement between CPU and Trainium devices
- **Custom Device Registration**: C++ extension for PyTorch device integration with Neuron Runtime (NRT)

## Prerequisites

Before getting started, make sure you have:

- **Hardware**: AWS Trn1 or Trn2 instance
- **Python**: Version 3.10
- **PyTorch**: Version 2.8.0
- **NKIPy**: The NKIPy package must be installed

## Installation

From the root of the nkipy repository, install torch-to-nkipy and its dependencies:

```bash
uv sync --group torch-to-nkipy
```

## Quick Start

Here's the simplest way to get started. This example compiles and runs a small MLP on Trainium:

```python
import torch
import torch.nn as nn
from torch_to_nkipy import init_nkipy_backend

# Step 1: Initialize the backend
init_nkipy_backend()

# Step 2: Define your model with the compile decorator
@torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))

# Step 3: Move model and input to the nkipy device
model = MLP().to("nkipy")
input_tensor = torch.randn(2, 4).to("nkipy")

# Step 4: Run inference
with torch.no_grad():
    output = model(input_tensor)

# Step 5: Move result back to CPU to print
print(output.cpu())
```

**What happens behind the scenes:**
1. `init_nkipy_backend()` registers the nkipy backend with PyTorch
2. `@torch.compile` captures your model's computation graph
3. `.to("nkipy")` moves tensors to Trainium device memory
4. The first forward pass compiles the graph to Neuron-optimized code
5. Subsequent calls use the cached compiled kernel

## Examples

We provide several examples in the `examples/` directory:

### Basic Examples

| Example | Description |
|---------|-------------|
| [example_hybrid_exe.py](examples/basics/example_hybrid_exe.py) | Hybrid CPU + Trainium execution |
| [example_dist.py](examples/basics/example_dist.py) | Distributed training with collective operations |
| [example_nki_op.py](examples/basics/example_nki_op.py) | Custom NKI kernel integration |

### HuggingFace Qwen Examples

| Example | Description |
|---------|-------------|
| [inference_single.py](examples/hf_qwen/inference_single.py) | Single-device Qwen inference |
| [inference_distributed.py](examples/hf_qwen/inference_distributed.py) | Distributed inference with tensor parallelism |

## Configuration

The `init_nkipy_backend()` function accepts several configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nkipy_cache` | `"./nkipy_cache"` | Directory to store compiled kernels for reuse |
| `log_level` | `logging.INFO` | Logging verbosity level |
| `rank` | `None` | Process rank (auto-detected if None) |
| `world_size` | `None` | Total processes (auto-detected if None) |
| `core_offset` | `0` | Offset for visible_cores assignment |
| `additional_compiler_args` | `""` | Extra flags to pass to the Neuron compiler |

### Rank Auto-Detection

When `rank` or `world_size` is `None`, the backend automatically detects values using this priority:

1. **torch.distributed** - If `dist.is_initialized()`, uses `dist.get_rank()` / `dist.get_world_size()`
2. **LOCAL_RANK env** - For torchrun and similar launchers
3. **RANK / WORLD_SIZE env** - Fallback for other launchers
4. **Default** - rank=0, world_size=1 for single process

This means you can simplify distributed setup:

```python
# Old way (explicit)
torch.distributed.init_process_group("nkipy")
init_nkipy_backend(rank=dist.get_rank(), world_size=dist.get_world_size())

# New way (auto-detect)
torch.distributed.init_process_group("nkipy")
init_nkipy_backend()  # Automatically gets rank from dist.get_rank()
```

### Core Offset

The `core_offset` parameter is useful for multi-node setups where visible cores start at different indices:

```python
# visible_cores = [rank + core_offset]
init_nkipy_backend(core_offset=4)  # Uses core rank+4 instead of rank
```

**Example with custom configuration:**
```python
init_nkipy_backend(
    nkipy_cache="./my_cache",
    log_level=logging.DEBUG,
    rank=0,
    world_size=4,
    core_offset=0,
)
```

## Supported Operations

The package includes implementations for 50+ PyTorch operations:

**Arithmetic:** add, sub, mul, div, pow, abs, reciprocal, sqrt, rsqrt

**Linear Algebra:** matmul, addmm, embedding

**Activations:** relu, gelu, sigmoid, hardtanh, leaky_relu, log_softmax, softmax

**Normalization:** native_layer_norm, linalg_vector_norm

**Reductions:** sum, mean, max, min, any, argmax, argmin

**Indexing/Slicing:** index, select, slice, gather, index_select, squeeze

**Shape Operations:** expand, repeat, cat, split_with_sizes, view, permute

**Tensor Creation:** full, arange, scalar_tensor, empty_permuted

**Logical:** where, clamp, ne, bitwise_not

**Collectives:** all_reduce, all_gather (for distributed training)

For the complete list, see the `nkipy_builder/aten_op_registry/` directory.

## Architecture Overview

Torch-to-NKIPy consists of several components that work together:

```
Your PyTorch Code
       |
       v
+------------------+
|  torch.compile   |  <- Captures computation graph
+------------------+
       |
       v
+------------------+
|  NKIPy Backend   |  <- Converts graph to NKIPy AST
+------------------+
       |
       v
+------------------+
|  ATen Op Registry|  <- Maps PyTorch ops to NKIPy
+------------------+
       |
       v
+------------------+
|  Neuron Compiler |  <- Produces optimized NEFF
+------------------+
       |
       v
+------------------+
|  Neuron Runtime  |  <- Executes on Trainium
+------------------+
```

**Key Components:**

1. **Backend** (`backend/`): PyTorch Dynamo backend registration and graph compilation
2. **NKIPy Builder** (`nkipy_builder/`): Converts FX graphs to NKIPy AST and generates executable kernels
3. **ATen Op Registry** (`nkipy_builder/aten_op_registry/`): Maps PyTorch ATen operations to NKIPy implementations
4. **Device Module** (`device/`): Custom device registration and distributed backend support
5. **Runtime** (`runtime/`): Execution runtime for compiled kernels
6. **C++ Extension** (`csrc/`): Native PyTorch device registration and tensor allocation

## FAQ

### How do I move tensors between CPU and Trainium?

Use the standard PyTorch `.to()` method or `.cpu()`:

```python
# CPU to Trainium
x_device = x_cpu.to("nkipy")

# Trainium to CPU
x_cpu = x_device.cpu()
# or
x_cpu = x_device.to("cpu")
```

### What models are supported?

Any PyTorch model that uses supported operations can run on Trainium. We've tested extensively with:
- Custom MLPs and transformers
- HuggingFace Qwen models
- Models using standard PyTorch layers (Linear, LayerNorm, etc.)

### How do I debug compilation issues?

1. **Check the cache directory** - Look in `nkipy_cache/` for generated code and compiler logs
2. **Enable debug logging** - Set `log_level=logging.DEBUG` in `init_nkipy_backend()`
3. **Look for unsupported ops** - If an operation isn't in the registry, it will fall back to CPU

### Can I use custom NKI kernels?

Yes! Use the `NKIOpRegistry.register()` decorator to integrate custom NKI kernels. See [example_nki_op.py](examples/basics/example_nki_op.py) for a complete example.

### How does distributed training work?

Torch-to-NKIPy supports PyTorch's distributed API with the `"nkipy"` backend:

```python
torch.distributed.init_process_group("nkipy")
init_nkipy_backend()  # Auto-detects rank/world_size from dist
```

Then use standard collective operations like `dist.all_reduce()` in your model.

You can also pass rank/world_size explicitly if needed:
```python
init_nkipy_backend(rank=dist.get_rank(), world_size=dist.get_world_size())
```

### What's the first-run compilation overhead?

The first forward pass compiles your model to Neuron code, which takes some time. Subsequent runs use cached compiled kernels. To warm up:

```python
# Warm-up run
with torch.no_grad():
    _ = model(dummy_input)

# Now run at full speed
output = model(real_input)
```

## Project Structure

```
torch-to-nkipy/
├── src/torch_to_nkipy/           # Python package
│   ├── backend/                   # Dynamo backend implementation
│   ├── device/                    # Device module and distributed support
│   ├── nkipy_builder/            # FX graph to NKIPy conversion
│   │   └── aten_op_registry/     # ATen operation implementations
│   ├── runtime/                   # Execution runtime
│   └── utils/                     # Utility functions
├── csrc/                          # C++ extension source
│   ├── nkipy_device.cpp/h        # Device registration
│   ├── nkipy_tensor.cpp/h        # Tensor implementation
│   ├── nkipy_tensor_allocator.cpp/h  # Memory allocation
│   ├── python_bindings.cpp        # PyBind11 bindings
│   └── torch_register.cpp/h       # PyTorch integration
├── examples/                      # Example scripts
│   ├── basics/                    # Basic usage examples
│   └── hf_qwen/                   # HuggingFace Qwen examples
├── tests/                         # Test suite
├── setup.py                       # Build configuration
├── pyproject.toml                 # Package metadata
└── MANIFEST.in                    # Source distribution manifest
```

## Getting Help

- **Report issues**: Open an issue on GitHub
- **Check examples**: The `examples/` directory has working code for common use cases
- **Enable debug logging**: Set `log_level=logging.DEBUG` for detailed output
