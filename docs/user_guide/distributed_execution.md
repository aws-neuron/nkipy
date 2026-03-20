# Distributed Execution

NKIPy supports multi-device execution with collective communication (CC)
through `DeviceKernel.compile_and_load`. This guide covers the three
execution patterns and when to use each.

## Execution Patterns

### 1. Auto-detected SPMD (default with torch.distributed)

When `torch.distributed` is initialized and no explicit CC parameters are
passed, NKIPy automatically enters **SPMD mode**: rank 0 traces and compiles
the kernel, then broadcasts the NEFF path to all workers. All ranks load the
same NEFF with CC enabled.

```python
import torch.distributed as dist

# torch.distributed must be initialized before compile_and_load
dist.init_process_group(...)

kernel = DeviceKernel.compile_and_load(my_kernel, input_a, input_b)
```

This is the simplest path — no extra parameters needed. Use this when every
rank runs the **same kernel** with the **same input shapes**.

### 2. Explicit CC (MPMD)

Pass `cc_enabled=True` with `rank_id` and `world_size` to enable CC while
letting **every rank trace and compile independently**. This is required for
MPMD workloads where different ranks run different kernels or different input
shapes.

```python
kernel = DeviceKernel.compile_and_load(
    my_kernel,
    input_a,
    input_b,
    cc_enabled=True,
    rank_id=my_rank,
    world_size=total_workers,
)
```

This also works for runtimes that manage their own ranks without
`torch.distributed`.

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

## Comparison

| Parameter           | Auto SPMD               | Explicit CC (MPMD)   | No CC         |
|---------------------|-------------------------|----------------------|---------------|
| `cc_enabled`        | `None` (default)        | `True`               | `False`/`None`|
| `torch.distributed` | Required                | Optional             | N/A           |
| Compilation         | Rank 0 only + broadcast | Every rank           | Every rank    |
| Barrier             | Yes                     | No                   | No            |
| Use case            | Same kernel, all ranks  | Per-rank kernels     | Single device |

## Build Directory Isolation

In explicit CC (MPMD) mode, each rank compiles independently. When `rank_id`
is provided, the build directory is automatically namespaced by rank
(e.g. `build_dir/rank_0/`, `build_dir/rank_1/`) to prevent concurrent writes
when different ranks produce the same content hash. This is transparent —
no extra configuration needed.

## Caching

Compiled NEFFs are cached in memory by a content hash of the HLO and compiler
arguments. The cache key is the same regardless of CC mode, so a kernel
compiled once can be reused across calls. Pass `use_cached_if_exists=False` to
force recompilation.
