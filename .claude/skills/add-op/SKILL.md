---
name: add-op
description: Add a new operation to the NKIPy ops registry with hlo and cpu backend implementations
disable-model-invocation: true
argument-hint: "[op-name] [category: unary|binary|reduce|transform|indexing|creation|conv|nn|collectives|linalg]"
---

# Add a New NKIPy Operation

Add the operation `$0` to the `$1` category file.

## Steps

### 1. Identify the correct category file

Operations live in `nkipy/src/nkipy/core/ops/<category>.py`. Choose from:

| Category | File | Examples |
|----------|------|----------|
| unary | `unary.py` | exp, log, sqrt, abs, neg |
| binary | `binary.py` | add, mul, sub, div, pow |
| reduce | `reduce.py` | sum, mean, max, min, prod |
| transform | `transform.py` | reshape, transpose, concat, split |
| indexing | `indexing.py` | getitem, setitem, slice |
| creation | `creation.py` | zeros, ones, arange, full |
| conv | `conv.py` | conv2d operations |
| nn | `nn.py` | relu, softmax, gelu |
| collectives | `collectives.py` | all_reduce, all_gather |
| linalg | `linalg.py` | matmul, dot |

### 2. Create the Op instance and register implementations

In the chosen category file, follow this pattern:

```python
from nkipy.core.backend.hlo import get_hlo_context
from nkipy.core.ops._registry import Op

# Create the dispatcher
my_op = Op('my_op')

@my_op.impl('hlo')
def _my_op_hlo(arg1, arg2, ...):
    """HLO tracing implementation."""
    ctx = get_hlo_context()
    # Build HLO operation using ctx
    # Return NKIPyTensorRef
    ...

@my_op.impl('cpu')
def _my_op_cpu(arg1, arg2, ...):
    """CPU eager implementation using NumPy."""
    import numpy as np
    # Return numpy array
    ...
```

Key rules:
- The `hlo` impl builds HLO IR nodes via the trace context (`get_hlo_context()` from `nkipy.core.backend.hlo`)
- The `cpu` impl uses pure NumPy and returns numpy arrays
- Both implementations must accept the same signature and produce equivalent results
- Both backends are **required** — every op needs `hlo` and `cpu`

### 3. Export from `__init__.py`

Add the new op to `nkipy/src/nkipy/core/ops/__init__.py`:
- Import it from the category module
- Add it to the `__all__` list if one exists
- Follow the existing grouping/ordering in the imports

### 4. Wire into tensor API (if applicable)

If the op should be callable as a method on tensors (e.g., `tensor.sum()`), or via `np.*` dispatch:
- Check `nkipy/src/nkipy/core/tensor.py` for `TensorArithmeticMixin`
- Check `nkipy/src/nkipy/core/tensor_apis.py` for high-level API wrappers
- Check `nkipy/src/nkipy/core/_numpy_dispatch.py` for NumPy protocol integration

### 5. Add tests

Create or extend tests in `tests/unit/`. Follow existing patterns:
- Test both CPU and HLO backends
- Test edge cases (empty tensors, broadcasting, dtype promotion)
- Run with: `uv run pytest tests/unit/test_tensor_api.py -k "test_my_op" -v`

### 6. Verify

```bash
uv run ruff check nkipy/src/nkipy/core/ops/
uv run pytest tests/unit/ -k "$0" -v
```
