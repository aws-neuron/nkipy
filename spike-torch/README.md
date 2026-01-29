# spike-torch

PyTorch device integration for the spike runtime on AWS Neuron.

## Overview

spike-torch provides PyTorch device registration using spike as the runtime. It enables:

- `torch.device("nkipy")` for creating tensors on Neuron devices
- `tensor.to("nkipy")` for moving tensors to/from Neuron devices
- `model.nkipy()` for moving modules to Neuron devices

## Installation

```bash
cd nkipy/spike-torch
pip install -e .
```

## Usage

```python
import torch
import spike_torch  # Registers the nkipy device and initializes NRT

# Check device availability
print(f"NKIPy devices available: {spike_torch.device_count()}")

# Create tensor on nkipy device
x = torch.randn(10, 10, device="nkipy")

# Move tensor to nkipy device
cpu_tensor = torch.randn(10, 10)
nkipy_tensor = cpu_tensor.to("nkipy")

# Move back to CPU
back_to_cpu = nkipy_tensor.cpu()
```

## Important Limitations

### Synchronous Operations

All spike-torch operations are **synchronous**:
- `synchronize()` is a no-op - operations already complete before returning
- `non_blocking` parameter in `.to()` and `copy_()` is accepted but ignored
- There are no CUDA-style streams or async execution

### Copy Requirements

All copy operations require **contiguous** tensors:

| Operation | Requirement |
|-----------|-------------|
| CPU → nkipy | Source must be contiguous |
| nkipy → CPU | Source must be contiguous |
| nkipy → nkipy | Both must be contiguous, **same Neuron core** |

```python
# Non-contiguous tensor (e.g., transposed) cannot copy directly
x = torch.randn(10, 10, device="nkipy")
y = x.t()      # View, not contiguous
y.cpu()        # ERROR

# Workaround: make contiguous on CPU
x.cpu().t().contiguous()
```

### Device-to-Device Copy

Cross-core copies are not supported. Use CPU as intermediate:
```python
x_core0 = torch.randn(10, device="nkipy:0")
x_core1 = x_core0.cpu().to("nkipy:1")  # Via CPU
```

## Support Matrix

### Supported Operations

| Category | Operations | Notes |
|----------|-----------|-------|
| **Tensor Creation** | `torch.empty()`, `torch.zeros()`, `torch.ones()`, `torch.randn()` | Creates tensors directly on nkipy device |
| **Copy (Contiguous)** | `tensor.to("nkipy")`, `tensor.cpu()`, `tensor.copy_()`, `tensor.clone()` | CPU↔nkipy and nkipy↔nkipy copies |
| **View Operations** | `view()`, `reshape()`, `transpose()`, `t()`, `permute()` | Zero-copy, shares underlying storage |
| **Slicing** | `tensor[:]`, `tensor[a:b]`, `tensor[..., idx]` | Creates view into original tensor |
| **Shape Operations** | `squeeze()`, `unsqueeze()`, `as_strided()` | Zero-copy shape manipulation |
| **Resize** | `resize_()`, `resize_as_()` | In-place resize, preserves existing data |
| **Properties** | `shape`, `stride()`, `numel()`, `dim()`, `is_contiguous()` | Tensor metadata access |
| **Device Management** | `device_count()`, `set_device()`, `current_device()`, `is_available()` | Multi-device support |
| **Memory Management** | `empty_cache()`, `get_cached_blocks()` | Caching allocator control |
| **Gradient** | `requires_grad`, `detach()` | Gradient tracking (computation on CPU) |

### Not Supported

| Category | Examples | Reason |
|----------|----------|--------|
| **Computation Ops** | `add()`, `mul()`, `matmul()`, `relu()`, `softmax()` | Falls back to CPU with warning |
| **Non-Contiguous Copy** | Copy of transposed/permuted tensors to CPU | Requires `contiguous()` first |
| **In-Place Arithmetic** | `add_()`, `mul_()`, `fill_()` | No on-device compute support |
| **Reductions** | `sum()`, `mean()`, `max()`, `min()` | Falls back to CPU |
| **Indexing with Tensors** | `tensor[indices]` where indices is a tensor | Advanced indexing not supported |

### Limitations

- **Strided tensors**: Non-contiguous tensors (e.g., after `transpose()`) cannot be copied directly to CPU. Use `.contiguous()` first, but note this is also not yet supported on device.
- **Computation**: spike-torch is designed for **data movement only**. All compute operations should be executed through compiled NKI kernels via the nkipy backend, not through PyTorch eager ops.
- **dtype conversion on device**: dtype changes during copy require CPU roundtrip.

## Architecture

```
torch-to-nkipy (FX graph capture & transformation)
       |
       v
spike-torch (PyTorch device registration)
       |
       v
  spike (NRT runtime wrapper)
       |
       v
  libnrt (Neuron Runtime)
```

spike-torch handles PyTorch's device registration, tensor allocation, and dispatch
while delegating actual NRT operations to the spike module. Computation is expected
to go through compiled NKI kernels, not through eager PyTorch operations.

## API Reference

```python
import spike_torch

# Device management
spike_torch.device_count()      # Number of available devices
spike_torch.current_device()    # Current device index
spike_torch.set_device(0)       # Set current device
spike_torch.is_available()      # Check if devices available

# Memory management
spike_torch.empty_cache()       # Release cached memory
spike_torch.get_cached_blocks() # Number of cached blocks

# Runtime/model execution
# Use the `spike` package directly (SpikeModel, SpikeTensor, profiling, etc.).
```
