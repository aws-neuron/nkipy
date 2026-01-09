# NumPy API Compatibility

NKIPy provides NumPy-compatible APIs that allow you to use familiar NumPy functions
with NKIPy tensors. When you call a NumPy function on an NKIPy tensor, it automatically
dispatches to the corresponding NKIPy operation.

## Usage Example

```python
import numpy as np
from nkipy.core import ops

# Inside a traced kernel, you can use NumPy functions directly:
# result = np.add(tensor_a, tensor_b)  # Dispatches to ops.add
# result = np.matmul(tensor_a, tensor_b)  # Dispatches to ops.matmul
```

## Supported NumPy Functions

### Binary Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.add` | `ops.add` |
| `np.bitwise_and` | `ops.bitwise_and` |
| `np.bitwise_or` | `ops.bitwise_or` |
| `np.bitwise_xor` | `ops.bitwise_xor` |
| `np.divide` | `ops.divide` |
| `np.maximum` | `ops.maximum` |
| `np.minimum` | `ops.minimum` |
| `np.multiply` | `ops.multiply` |
| `np.power` | `ops.power` |
| `np.subtract` | `ops.subtract` |

### Comparison Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.equal` | `ops.equal` |
| `np.greater` | `ops.greater` |
| `np.greater_equal` | `ops.greater_equal` |
| `np.less` | `ops.less` |
| `np.less_equal` | `ops.less_equal` |
| `np.logical_and` | `ops.logical_and` |
| `np.logical_or` | `ops.logical_or` |
| `np.logical_xor` | `ops.logical_xor` |
| `np.not_equal` | `ops.not_equal` |

### Unary Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.abs` | `ops.abs` |
| `np.arctan` | `ops.arctan` |
| `np.bitwise_not` | `ops.bitwise_not` |
| `np.ceil` | `ops.ceil` |
| `np.cos` | `ops.cos` |
| `np.exp` | `ops.exp` |
| `np.floor` | `ops.floor` |
| `np.invert` | `ops.invert` |
| `np.log` | `ops.log` |
| `np.logical_not` | `ops.logical_not` |
| `np.negative` | `ops.negative` |
| `np.rint` | `ops.rint` |
| `np.sign` | `ops.sign` |
| `np.sin` | `ops.sin` |
| `np.sqrt` | `ops.sqrt` |
| `np.square` | `ops.square` |
| `np.tan` | `ops.tan` |
| `np.tanh` | `ops.tanh` |
| `np.trunc` | `ops.trunc` |

### Reduction Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.any` | `ops.any` |
| `np.max` | `ops.max` |
| `np.mean` | `ops.mean` |
| `np.min` | `ops.min` |
| `np.sum` | `ops.sum` |

### Linear Algebra

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.matmul` | `ops.matmul` |

### Transform Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.concatenate` | `ops.concatenate` |
| `np.copy` | `ops.copy` |
| `np.expand_dims` | `ops.expand_dims` |
| `np.repeat` | `ops.repeat` |
| `np.reshape` | `ops.reshape` |
| `np.split` | `ops.split` |
| `np.transpose` | `ops.transpose` |

### Creation Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.empty_like` | `ops.empty_like` |
| `np.full_like` | `ops.full_like` |
| `np.zeros_like` | `ops.zeros_like` |

### Indexing Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.put_along_axis` | `ops.put_along_axis` |
| `np.take` | `ops.take` |
| `np.take_along_axis` | `ops.take_along_axis` |
| `np.where` | `ops.where` |

### Broadcast and Copy Operations

| NumPy Function | NKIPy Operation |
|----------------|-----------------|
| `np.broadcast_to` | `ops.broadcast_to` |
| `np.copyto` | `ops.copyto` |

## Unsupported NumPy Operations

The following NumPy operations are **not supported** due to hardware limitations:

| NumPy Function | Reason |
|----------------|--------|
| `np.arctan2` | not supported on hardware |
| `np.floor_divide` | not supported |
| `np.positive` | not supported, use np.copy for "y = +x" operation |
| `np.round` | not a supported activation |

### Workarounds

For some unsupported operations, workarounds exist:

- **`np.mod` / `np.remainder`**: Use `a - b * np.floor(a/b)`
- **`np.positive`**: Use `np.copy(x)` for the `y = +x` operation
