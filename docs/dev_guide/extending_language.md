# Extending NKIPy Operations

This guide explains how to add new operations to NKIPy.

## Overview

NKIPy operations are defined in `nkipy/src/nkipy/core/ops/`. Each operation is created using the `Op` class from `_registry.py`, which provides a simple dispatch mechanism to route operations to backend implementations.

## Adding a New Operation

### Step 1: Choose the Right Module

Operations are organized by category:

| Module | Operations |
|--------|------------|
| `binary.py` | add, subtract, multiply, divide, comparisons |
| `unary.py` | abs, exp, log, sqrt, sin, cos, etc. |
| `reduce.py` | sum, max, min, mean |
| `transform.py` | reshape, transpose, concatenate, split |
| `creation.py` | zeros, full, zeros_like, ones_like |
| `indexing.py` | take, where, static_slice |
| `linalg.py` | matmul |
| `nn.py` | softmax, topk, rms_norm |
| `conv.py` | conv2d, conv3d |
| `collectives.py` | all_reduce, all_gather |

### Step 2: Define the Operation

Create an `Op` instance and register backend implementations:

```python
from nkipy.core.ops._registry import Op

# Create the operation
my_op = Op("my_op")

# Register HLO backend implementation
@my_op.impl("hlo")
def _my_op_hlo(x, y, some_param=None):
    from nkipy.core.backend.hlo import get_hlo_context
    from nkipy.core.tensor import NKIPyTensorRef
    
    ctx = get_hlo_context()
    
    # Convert NKIPyTensorRef to backend tensor
    if isinstance(x, NKIPyTensorRef):
        x = x.backend_tensor
    if isinstance(y, NKIPyTensorRef):
        y = y.backend_tensor
    
    # Build the HLO operation
    result_tensor = ctx.build_op(
        "hlo_op_name",
        [x, y],
        output_shape,
        output_dtype,
        {"attribute": value}
    )
    
    return NKIPyTensorRef(result_tensor)
```

### Step 3: Export the Operation

Add the operation to `__init__.py`:

```python
# In nkipy/src/nkipy/core/ops/__init__.py
from nkipy.core.ops.my_module import my_op

__all__ = [
    # ... existing ops ...
    "my_op",
]
```
