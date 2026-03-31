# Distributed Execution

NKIPy supports multi-device execution with collective communication (CC)
through `DeviceKernel.compile_and_load`. This guide covers the three
execution patterns and when to use each.

## Execution Patterns

### 1. SPMD (default)

When `torch.distributed` is initialized and `is_spmd=True` (the default),
rank 0 traces and compiles the kernel, then broadcasts the NEFF path to all
workers. All ranks load the same NEFF with CC enabled.

```python
import torch.distributed as dist

dist.init_process_group(...)

kernel = DeviceKernel.compile_and_load(my_kernel, input_a, input_b)
```

Use this when every rank runs the **same kernel** with the **same input shapes**.

### 2. MPMD (`is_spmd=False`)

Set `is_spmd=False` so every rank traces and compiles independently. This is
required when different ranks run different kernels or different input shapes.

```python
# With torch.distributed (CC auto-detected)
kernel = DeviceKernel.compile_and_load(
    my_kernel, input_a, input_b,
    is_spmd=False,
)

# Without torch.distributed (explicit CC)
kernel = DeviceKernel.compile_and_load(
    my_kernel, input_a, input_b,
    is_spmd=False,
    cc_enabled=True,
    rank_id=my_rank,
    world_size=total_workers,
)
```

### 3. No CC (single device or explicit opt-out)

Without `torch.distributed` and without explicit CC parameters, the kernel
loads for single-device execution. You can also pass `cc_enabled=False` to
explicitly disable CC even when `torch.distributed` is active.

```python
# Single device (no torch.distributed)
kernel = DeviceKernel.compile_and_load(my_kernel, input_a)

# Opt out of CC in a distributed setting
kernel = DeviceKernel.compile_and_load(my_kernel, input_a, cc_enabled=False)
```

## Parameter Reference

| Parameter    | Controls           | Values                                        |
|--------------|--------------------|-----------------------------------------------|
| `is_spmd`    | Compilation        | `True` = rank-0 broadcast, `False` = all rank |
| `cc_enabled` | CC at load time    | `None` = auto, `True` = on, `False` = off     |
| `rank_id`    | Rank for CC load   | `None` = auto from dist, or explicit `int`    |
| `world_size` | World size for CC  | `None` = auto from dist, or explicit `int`    |

## Comparison

| Setting                 | SPMD (default)          | MPMD                 | No CC         |
|-------------------------|-------------------------|----------------------|---------------|
| `is_spmd`               | `True`                  | `False`              | Either        |
| `cc_enabled`            | `None` (auto)           | `None`/`True`        | `False`/`None`|
| `torch.distributed`     | Required                | Optional             | N/A           |
| Compilation             | Rank 0 only + broadcast | Every rank           | Every rank    |
| Barrier                 | Yes                     | No                   | No            |
| Use case                | Same kernel, all ranks  | Per-rank kernels     | Single device |

## Build Directory Isolation

In MPMD mode (`is_spmd=False`), the build directory is automatically
namespaced by rank (e.g. `build_dir/rank_0/`, `build_dir/rank_1/`) to
prevent concurrent writes when different ranks produce the same content hash.
The rank is taken from the explicit `rank_id` parameter, or auto-detected
from `torch.distributed` when available.

## Caching

Compiled NEFFs are cached in memory by a content hash of the HLO and compiler
arguments. The cache key is the same regardless of CC mode, so a kernel
compiled once can be reused across calls. Pass `use_cached_if_exists=False` to
force recompilation.
