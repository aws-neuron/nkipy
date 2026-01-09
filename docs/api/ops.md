# Operations Reference

This page documents all operations supported by NKIPy, organized by category.

## Overview

NKIPy operations are traced and lowered to HLO for compilation by the Neuron Compiler.

## Categories

- [Binary Operations](#binary-operations)
- [Indexing Operations](#indexing-operations)
- [Reduction Operations](#reduction-operations)
- [Creation Operations](#creation-operations)
- [Neural Network Operations](#neural-network-operations)
- [Transform Operations](#transform-operations)
- [Unary Operations](#unary-operations)
- [Convolution Operations](#convolution-operations)
- [Linear Algebra Operations](#linear-algebra-operations)
- [Collective Operations](#collective-operations)

## Binary Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `add` | hlo | `np.add` |
| `bitwise_and` | hlo | `np.bitwise_and` |
| `bitwise_or` | hlo | `np.bitwise_or` |
| `bitwise_xor` | hlo | `np.bitwise_xor` |
| `divide` | hlo | `np.divide` |
| `equal` | hlo | `np.equal` |
| `greater` | hlo | `np.greater` |
| `greater_equal` | hlo | `np.greater_equal` |
| `less` | hlo | `np.less` |
| `less_equal` | hlo | `np.less_equal` |
| `logical_and` | hlo | `np.logical_and` |
| `logical_or` | hlo | `np.logical_or` |
| `logical_xor` | hlo | `np.logical_xor` |
| `maximum` | hlo | `np.maximum` |
| `minimum` | hlo | `np.minimum` |
| `multiply` | hlo | `np.multiply` |
| `not_equal` | hlo | `np.not_equal` |
| `power` | hlo | `np.power` |
| `subtract` | hlo | `np.subtract` |

## Indexing Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `dynamic_update_slice` | hlo | — |
| `put_along_axis` | hlo | `np.put_along_axis` |
| `scatter_strided` | hlo | — |
| `static_slice` | hlo | — |
| `take` | hlo | `np.take` |
| `take_along_axis` | hlo | `np.take_along_axis` |
| `where` | hlo | `np.where` |

## Reduction Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `any` | hlo | `np.any` |
| `max` | hlo | `np.max` |
| `mean` | hlo | `np.mean` |
| `min` | hlo | `np.min` |
| `sum` | hlo | `np.sum` |

## Creation Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `empty_like` | cpu, hlo | `np.empty_like` |
| `full` | cpu, hlo | — |
| `full_like` | cpu, hlo | `np.full_like` |
| `ones_like` | cpu, hlo | — |
| `zeros` | cpu, hlo | — |
| `zeros_like` | cpu, hlo | `np.zeros_like` |

## Neural Network Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `rms_norm` | — | — |
| `softmax` | — | — |
| `topk` | cpu, hlo | — |

## Transform Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `astype` | hlo | — |
| `broadcast_to` | hlo | `np.broadcast_to` |
| `concatenate` | hlo | `np.concatenate` |
| `copy` | hlo | `np.copy` |
| `copyto` | — | `np.copyto` |
| `expand_dims` | hlo | `np.expand_dims` |
| `repeat` | hlo | `np.repeat` |
| `reshape` | hlo | `np.reshape` |
| `split` | hlo | `np.split` |
| `transpose` | hlo | `np.transpose` |

## Unary Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `abs` | hlo | `np.abs` |
| `arctan` | hlo | `np.arctan` |
| `bitwise_not` | hlo | `np.bitwise_not` |
| `ceil` | hlo | `np.ceil` |
| `cos` | hlo | `np.cos` |
| `exp` | hlo | `np.exp` |
| `floor` | hlo | `np.floor` |
| `invert` | hlo | `np.invert` |
| `log` | hlo | `np.log` |
| `logical_not` | hlo | `np.logical_not` |
| `negative` | hlo | `np.negative` |
| `rint` | hlo | `np.rint` |
| `sign` | hlo | `np.sign` |
| `sin` | hlo | `np.sin` |
| `sqrt` | hlo | `np.sqrt` |
| `square` | hlo | `np.square` |
| `tan` | hlo | `np.tan` |
| `tanh` | hlo | `np.tanh` |
| `trunc` | hlo | `np.trunc` |

## Convolution Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `conv2d` | cpu, hlo | — |
| `conv3d` | cpu, hlo | — |

## Linear Algebra Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `matmul` | hlo | `np.matmul` |

## Collective Operations

| Operation | Backend(s) | NumPy Equivalent |
|-----------|-----------|------------------|
| `all_gather` | cpu, hlo | — |
| `all_reduce` | cpu, hlo | — |
| `all_to_all` | cpu, hlo | — |
| `reduce_scatter` | cpu, hlo | — |

## API Reference

```{eval-rst}
.. automodule:: nkipy.core.ops
   :members:
   :undoc-members:
   :show-inheritance:
```