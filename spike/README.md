# Spike

## Overview

Spike is a Pythonic runtime layer for AWS Neuron that lifts `libnrt` (Neuron Runtime Library) into a clean Python interface. It provides:

- **Pythonic user experience**: Core runtime concepts (tensors, models, execution) with idiomatic Python patterns for resource management, data movement, and error handling. Neuron Runtime concepts that are deprecated, hard to understand, or not commonly used are abstracted away.
- **Simple codebase**: Under 1,500 lines of C++ and Python code with clear layers, making it easy for the community to learn, maintain, and extend.
- **Minimal performance overhead**: Performance equivalent to C++, achieved through zero-copy buffer protocol and efficient nanobind C++ bindings.

## Installation

### Prerequisites

- Python 3.10 or higher
- AWS Neuron Runtime (NRT) library installed
- CMake 3.15 or higher
- C++17 compatible compiler

### From Source

```bash
pip install -e .
```

## Terminology

| Term                              | Definition                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------- |
| **libnrt**                        | Neuron Runtime Library - C API for Neuron hardware interaction                      |
| **NEFF**                          | Neuron Executable File Format - compiled model binary                               |
| **NeuronCore**                    | Individual compute unit on Neuron hardware. Each instance has multiple NeuronCores  |
| **HBM**                           | High Bandwidth Memory - device memory accessible by NeuronCores                     |
| **Collective Communication (CC)** | Multi-device operations like all-reduce, all-gather                                 |
| **nanobind**                      | C++/Python binding library used for Spike's bindings                                |
| **RAII**                          | Resource Acquisition Is Initialization - C++ pattern for automatic resource cleanup |

## Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application                       │
│              (Running NKIPy or NKI kernels)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Spike Python API                        │
│                 SpikeModel, SpikeTensor                     │
│                  Spike (nanobind Python)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Spike C++                             │
│                   Spike (nanobind C++)                      │
│      NrtRuntime, NrtModel, NrtTensor, NrtTensorSet          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 libnrt (Neuron Runtime Library)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Neuron Hardware                          │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure and Components

```
spike/
├── src/
│   ├── include/*.h        # C++ headers
│   ├── *.cpp              # C++ implementation
│   └── spike/*.py         # Python package
├── tests/                 # Test suite
├── CMakeLists.txt         # Build configuration
└── pyproject.toml         # Package metadata
```

| Component      | Layer  | File(s)               | Purpose                                      |
| -------------- | ------ | --------------------- | -------------------------------------------- |
| `NrtRuntime`   | C++    | `nrt_wrapper.cpp/h`   | RAII wrapper for libnrt init/shutdown        |
| `NrtTensor`    | C++    | `tensor.cpp/h`        | RAII wrapper for device tensors              |
| `NrtTensorSet` | C++    | `tensor_set.cpp/h`    | Internal tensor grouping (hidden from users) |
| `NrtModel`     | C++    | `model.cpp/h`         | RAII wrapper for loaded NEFF models          |
| `Spike`        | C++    | `spike.cpp/h`         | Main runtime interface, coordinates all ops  |
|                | C++    | `python_bindings.cpp` | The nanobind Python binding in C++           |
|                | Python | `_spike.pyi`          | Auto-generated Spike Python interfaces       |
| `SpikeTensor`  | Python | `spike_tensor.py`     | High-level tensor with actions               |
| `SpikeModel`   | Python | `spike_model.py`      | High-level model class with actions          |

### Memory Management

Spike provides Pythonic memory management - tensors and models keep the runtime alive as long as they exist, so you don't need to worry about cleanup order. Device memory is automatically freed when tensors are garbage collected by Python. For explicit control, use `spike.free_tensor(tensor)` or `spike.unload_model(model)`.

### Execution Model

**1 Python process → 1 Spike singleton → 1 libNRT**

Spike follows a simple execution model where each Python process has a single Spike singleton instance, which manages a single `libnrt` runtime. This design ensures clean resource management and predictable behavior. The singleton is created lazily on first use (e.g., when creating a `SpikeTensor` or `SpikeModel`) and coordinates all operations including model loading, tensor allocation, and execution.

The singleton pattern prevents accidental creation of multiple NRT instances, which would cause errors. Use `spike.configure()` before first use to set visible cores, and `spike.reset()` to close and allow reconfiguration.

#### Default Mode: Single NeuronCore per Process

Default Model: 1 Python process → 1 Spike instance → 1 libNRT → **1 NeuronCore**

In the default mode, the Spike instance manages a single NeuronCore. All models and tensors are allocated to this core (core_id=0 by default), and execution happens sequentially on this core. This is the simplest and most common usage pattern, suitable for most applications.

Workloads requiring collective communication can use multiple Python processes in this mode.


#### Advanced Mode: Multiple NeuronCore per Process

1 Python process → 1 Spike instance → 1 libNRT → **N NeuronCore**

Spike also supports **N NeuronCores** within a single Python process and a single Spike instance.
In this advanced mode, users can specify the core ID for the tensors and models. 
Then, they can use Python threading to launch multiple `execute()` on different cores.
This can achieve running models in parallel on different core, either for parallel testing or running workloads with collective communication.
To support concurrent execution, The `execute()` method releases Python's GIL.

## Python API

### Low-level API: Spike, NrtModel, and NrtTensor

The low-level API provides direct access to the Spike runtime interface. The `Spike` instance manages the runtime lifecycle and coordinates all operations. Models and tensors (`NrtModel` and `NrtTensor`) are Python objects returned by model loading and tensor allocation operations.

In the low-level API, all actions are performed through the `Spike` instance. These actions have relatively straightforward mappings to the underlying `libnrt` C++ APIs. The models and tensors are essentially data objects that point to on-device models and tensors.

**Note:** The `Spike` class is internal and not exported from the `spike` package. For most use cases, use `SpikeTensor` and `SpikeModel` which manage the singleton automatically. For advanced/testing use cases, access via `from spike._spike import Spike`.

```python
from spike._spike import Spike  # Internal class

# Query hardware (static method, no instance needed)
Spike.get_visible_neuron_core_count() -> int

# Lifecycle (RAII: constructor initializes NRT)
spike = Spike()  # Initializes NRT automatically
del spike        # releases spike, will close NRT if all associated tensor/models are deleted
spike.close()    # Explicitly closes NRT

# Models
model: NrtModel = spike.load_model(
    neff_file: str,
    core_id: int = 0,         # Target NeuronCore
    cc_enabled: bool = False, # Enable collective communication
    rank_id: int = 0,         # Rank in CC group
    world_size: int = 1       # Total ranks
)
spike.unload_model(model)
spike.get_tensor_info(model) -> ModelTensorInfo

# Tensors
tensor: NrtTensor = spike.allocate_tensor(size: int, core_id: int = 0, name: str = None)
sliced_tensor: NrtTensor = spike.slice_from_tensor(source, offset=0, size=0, name=None)
spike.free_tensor(tensor)

# Tensor I/O
spike.tensor_write(tensor, data: bytes, offset=0)
spike.tensor_read(tensor, offset=0, size=0) -> bytes
spike.tensor_write_from_pybuffer(tensor, buffer, offset=0)  # Zero-copy from numpy
spike.tensor_read_to_pybuffer(tensor, buffer, offset=0, size=0)

# Execution (releases GIL for CC support)
spike.execute(model, inputs: Dict[str, NrtTensor], outputs: Dict[str, NrtTensor],
           ntff_name: str = None, save_trace: bool = False)

# Benchmarking
spike.benchmark(model, inputs, outputs, warmup_iterations=1, benchmark_iterations=1) -> BenchmarkResult
```

### High-Level API: SpikeTensor and SpikeModel

High-level tensor with NumPy integration and automatic cleanup. High-level model wrapper with validation and automatic output allocation.

The `SpikeTensor` and `SpikeModel` classes use the Spike singleton from `spike_singleton.py` for runtime initialization and cleanup.

### Singleton Configuration

The singleton can be configured before first use with `spike.configure()`, and reset with `spike.reset()`:

```python
import spike

# Optional: Configure visible NeuronCores before first spike operation
spike.configure(visible_cores=[0, 1])  # Use cores 0 and 1

# Use SpikeTensor/SpikeModel - singleton is created lazily on first use
tensor = SpikeTensor(...)
model = SpikeModel.load_from_neff("model.neff")

# To switch to different cores, reset and reconfigure
del tensor, model  # (Optional) Release resources first
spike.reset()      # Close the runtime
spike.configure(visible_cores=[2, 3])  # Configure new cores
tensor2 = SpikeTensor(...)  # New runtime with cores 2, 3
```

**API:**
- `spike.configure(visible_cores=None)` - Configure before first use. Raises `RuntimeError` if runtime already active.
- `spike.reset()` - Close the runtime. All existing `SpikeTensor`/`SpikeModel` objects become invalid.
- `spike.get_spike_singleton()` - Get the singleton instance (recommend only for internal use).

```python
from spike import SpikeModel, SpikeTensor
import numpy as np

# Load a NEFF model (optionally specify core_id, cc_enabled, rank_id, world_size)
model = SpikeModel.load_from_neff("path/to/model.neff")

# Create input tensors (optionally specify core_id to match model's core)
input_data = np.random.randn(1, 10).astype(np.float32)
input_tensor = SpikeTensor.from_numpy(input_data, name="input")

# Create output tensor
output_shape = (1, 10)
output_data = np.zeros(output_shape, dtype=np.float32)
output_tensor = SpikeTensor.from_numpy(output_data, name="output")

# Execute model
model(
    inputs={"input": input_tensor},
    outputs={"output": output_tensor} # optional
)

# Benchmark model execution
results = model.benchmark(
    inputs={"input": input_tensor},
    outputs={"output": output_tensor}, # optional
    warmup_iter=5,
    benchmark_iter=100
)

print(f"Mean execution time: {results.mean_ms:.2f} ms")
print(f"Min: {results.min_ms:.2f} ms, Max: {results.max_ms:.2f} ms")

# Get results
result = output_tensor.numpy()
print(result)
```

## Error Handling

Spike provides a clean exception hierarchy:

```
SpikeRuntimeError (base, inherits RuntimeError)
├── SpikeError      # Spike-level errors (e.g., using freed tensor)
└── NrtError        # libnrt errors with preserved status code
```

```python
from spike import SpikeError, NrtError

try:
    model = spike.load_model("nonexistent.neff")
except NrtError as e:
    # "NRT Error NRT_FAILURE(1): Failed to load model"
    ...

try:
    spike.execute(model, {"x": freed_tensor}, outputs)
except SpikeError as e:
    # "Spike Error: Tensor 'x' is freed..."
    ...
```

## Limitations & Future Work

### Current Limitations

- **FP8 dtypes**: `float8_e4m3`/`float8_e5m2` are reported as `int8` by `libnrt`; Spike includes workarounds.

### Future Work

- [ ] **Improved benchmarking**: Integrate libnrt inspect APIs for device-side latency measurement
- [ ] **Kernel switch overhead**: Add option to measure kernel switching costs
- [ ] **NKI debugger integration**: Support for debugging NKI kernels via Spike
