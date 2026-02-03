# torch-to-nkipy Tests

This directory contains tests for the `torch-to-nkipy` package.

## Test Structure

```
tests/
├── ops/                      # ATen operator tests (88+ tests)
├── conftest.py               # Pytest fixtures
├── base.py                   # NKIPyTestBase class for op tests
├── test_backend.py           # Backend initialization tests
├── test_runtime.py           # Runtime compile/execute tests
├── test_integration.py       # End-to-end torch.compile tests
└── test_distributed.py       # Distributed execution tests
```

## Running Tests

### Prerequisites

Ensure you have Neuron hardware available and the environment set up:

```bash
cd torch-to-nkipy
```

### Non-Distributed Tests

Run all non-distributed tests:

```bash
uv run pytest tests/ --ignore=tests/ops -v
```

Run specific test modules:

```bash
# Backend tests
uv run pytest tests/test_backend.py -v

# Runtime tests
uv run pytest tests/test_runtime.py -v

# Integration tests
uv run pytest tests/test_integration.py -v
```

### Distributed Tests

Distributed tests require `torchrun` with multiple processes:

```bash
# Run with 2 processes
uv run torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py -v

# Run with 4 processes
uv run torchrun --nproc_per_node=4 -m pytest tests/test_distributed.py -v
```

### ATen Operator Tests

Run all operator tests:

```bash
uv run pytest tests/ops/ -v
```

Run a specific operator test:

```bash
uv run pytest tests/ops/test_aten_add.py -v
```

## Test Categories

### test_backend.py

Tests for backend initialization and configuration:
- `init_nkipy_backend()` / `reset_nkipy_backend()`
- Auto-detection of rank and world_size from environment
- Configuration immutability and parameter storage

### test_runtime.py

Tests for the runtime compilation and execution pipeline:
- `TensorSpec`, `IOSpecs`, `NeuronExecutable` dataclasses
- `compile_model()`, `load_model()` caching behavior
- `run_neff_model()` execution
- Parallel compilation context

### test_integration.py

End-to-end integration tests:
- `torch.compile` with nkipy backend
- Multiple data types (fp32, fp16, bf16)
- NTFF profiling output
- Backward compatibility aliases
- Device tensor operations

### test_distributed.py

Distributed execution tests:
- `torch.distributed` with nkipy backend
- `all_reduce` collective operations
- Multi-rank MLP models
- Rank and world_size configuration

## Writing New Tests

### Using NKIPyTestBase

For operator tests, extend `NKIPyTestBase`:

```python
from base import NKIPyTestBase

class TestMyOp(NKIPyTestBase):
    def test_my_op(self):
        def test_func(x):
            return torch.ops.aten.my_op(x)

        x = torch.randn(16, 32)
        self.run_test_on_host(test_func, (x,))
        self.run_test_on_device(test_func, (x,))
```

### Using isolated_backend Fixture

For tests that need to control backend initialization:

```python
def test_my_backend_test(isolated_backend):
    isolated_backend["init"](rank=0, world_size=1)

    config = isolated_backend["get_config"]()
    assert config.rank == 0

    # Cleanup is automatic
```

### Writing Distributed Tests

```python
class TestMyDistributed:
    @classmethod
    def setup_class(cls):
        setup_distributed()

    @classmethod
    def teardown_class(cls):
        cleanup_distributed()

    def test_my_collective(self):
        @torch.compile(backend="nkipy", fullgraph=True)
        def my_func(x):
            y = x * 2
            return dist.all_reduce(y, op=dist.ReduceOp.SUM)

        x = torch.randn(4, 4, device="nkipy")
        result = my_func(x)
```
