# Spiky (Experimental) is Spike Yet torch stuff

Spiky is the user-facing runtime for [NKIPy](../../README.md) on AWS Neuron hardware. It provides a `torch.compile` backend that compiles PyTorch models to NKIPy IR and executes them on Trainium/Inferentia via a C++ execution engine.

## Features

- **`torch.compile` backend** — Register `"nkipy"` as a backend and compile standard PyTorch models with one line
- **Dynamic shapes** — Bucket-based compilation with JIT compilation for new sizes at runtime
- **Pipelined execution** — Overlapped H2D transfers and NRT execution via unified double-buffering
- **Device-side padding** — Inputs padded on-device to bucket size, outputs unpadded automatically
- **Per-function options** — Control pipelining, padding, layout, and output behavior per compiled function
- **Memory pool** — Unified device memory pool with stats, cache clearing, and trimming
- **NTFF profiling** — Generate Neuron trace files for profiling via `save_ntff` option
- **Distributed** — Collective communication support with auto-detected rank/world_size from `torch.distributed`
- **Custom device** — `"nkipy"` registered as a PyTorch device; tensors move with `.to("nkipy")`

## Quickstart

```bash
cd nkipy/
uv pip install -e experimental/spiky/spiky
```

```python
import torch
import torch.nn as nn
from spiky.torch import init_nkipy_backend

init_nkipy_backend(nkipy_cache="./nkipy_cache")

model = nn.Linear(32, 64, bias=False)
compiled = torch.compile(model, backend="nkipy")

with torch.no_grad():
    out = compiled(torch.randn(1, 32))
```

## Usage Examples

### Dynamic shapes

```python
model = nn.Linear(32, 64, bias=False)
compiled = torch.compile(model, backend="nkipy", dynamic=True)

x = torch.randn(4, 32)
torch._dynamo.maybe_mark_dynamic(x, 0)  # mark batch dim as dynamic

with torch.no_grad():
    out1 = compiled(x)
    out2 = compiled(torch.randn(16, 32))  # different size, bucket auto-selected
```

### Per-function options

```python
compiled = torch.compile(model, backend="nkipy", options={
    "pipelined": True,
    "pad_on_device": True,
    "output_layout": "padded",
    "input_layout": "padded",
    "keep_outputs_on_device": True,
})

with torch.no_grad():
    out = compiled(x)
```

### Memory and stats

```python
import spiky

stats = spiky.get_stats()
print(stats.total_executions, stats.bucket_hits)

mem = spiky.get_memory_stats()
print(f"Used: {mem['used_bytes']}, Cached: {mem['cached_bytes']}")

spiky.trim_memory_pool(target_bytes=0)
spiky.reset_stats()
```

### Distributed

```python
import torch.distributed as dist
from spiky.torch import init_nkipy_backend

dist.init_process_group("gloo")
init_nkipy_backend()  # rank, world_size auto-detected from dist

model = nn.Linear(32, 64, bias=False)
compiled = torch.compile(model, backend="nkipy", fullgraph=True)

with torch.no_grad():
    out = compiled(torch.randn(1, 32))
```

## Requirements

- Python >= 3.10
- AWS Neuron hardware (Trainium / Inferentia2)
- `neuronxcc` compiler
- PyTorch with Neuron support

## Running Tests

```bash
cd nkipy/

# All spiky tests
uv run pytest experimental/spiky/spiky/tests/ -v

# Smoke tests only
uv run pytest experimental/spiky/spiky/tests/ -m smoke -v
```
