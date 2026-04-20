# Cache Tensor Separation Optimization

## Date: 2026-04-16

## Problem

After implementing `allocate_uninitialized()` for weight tensors, we achieved:
- **Rank 0**: spike_reset 15s → 7.67s (52% improvement) ✅
- **Ranks 1-31**: spike_reset ~17-18s (NO improvement) ❌

## Root Cause Analysis

The original `_allocate_empty_tensors()` allocated tensors in this order:
```python
for each layer (60 layers):
    allocate weight1 (uninitialized)  # No write
    allocate weight2 (uninitialized)  # No write
    ...
    allocate cache_k (from_numpy(zeros))  # Write zeros! 
    allocate cache_v (from_numpy(zeros))  # Write zeros!
```

With TP=32 and 60 layers:
- 60 layers × 2 cache tensors × 32 ranks = **3,840 cache tensor writes**
- Each write interleaved with uninitialized weight allocations

**Hypothesis**: Interleaving uninitialized allocations with zero-writes causes device memory fragmentation, making NRT cleanup (spike_reset) slow.

## Why Cache Tensors Need Zeros

Cache tensors (cache_k, cache_v) **must** be initialized to zeros because:
1. They store KV cache state per request
2. They are NOT transferred via P2P (unlike weights)
3. Fresh instance needs clean cache state

So we cannot use `allocate_uninitialized()` for cache tensors.

## Solution: Separate Allocation Phases

Instead of interleaving, separate into two phases:

**Phase 1**: Allocate ALL weight tensors (uninitialized)
```python
for each layer:
    allocate all weight tensors (uninitialized)
```

**Phase 2**: Allocate ALL cache tensors (zeros)
```python
for each layer:
    allocate cache_k (zeros)
    allocate cache_v (zeros)
```

This keeps device memory regions contiguous:
- Weight region: uninitialized, filled by RDMA
- Cache region: zero-initialized

## Implementation

### nkipy/src/nkipy/vllm_plugin/models/qwen3.py

```python
# Phase 1: Weight tensors (uninitialized)
self.layer_tensors = []
for lid in range(cfg.num_layers):
    layer = {}
    for wk, prefix in LAYER_WEIGHT_KEYS:
        layer[wk] = DeviceTensor.allocate_uninitialized(
            shapes[wk], cfg.dtype, f"{prefix}_L{lid}")
    self.layer_tensors.append(layer)

# Phase 2: Cache tensors (zeros)
for lid in range(cfg.num_layers):
    layer = self.layer_tensors[lid]
    layer["cache_k"] = DeviceTensor.from_numpy(cache_k, f"cache_k_L{lid}")
    layer["cache_v"] = DeviceTensor.from_numpy(cache_v, f"cache_v_L{lid}")
```

Same changes applied to `models/llama.py`.

## Expected Impact

**Hypothesis**: By keeping weight and cache allocations separate, NRT can:
1. Handle uninitialized region efficiently (no fragmentation)
2. Handle zero-initialized region as a contiguous block
3. Clean up faster during `spike_reset()` (less fragmented state)

**Expected Results**:
- Ranks 1-31 should now match Rank 0 performance
- spike_reset should drop from 17-18s → ~2-7s
- All ranks should benefit uniformly

## Why Rank 0 Was Already Fast

**Theory**: Rank 0 may have fewer tensors due to MoE expert distribution, so interleaving had less impact. Or Rank 0 has special NRT handling as coordinator.

Once we eliminate interleaving, all ranks should see similar performance regardless of tensor count.

## Testing

Test procedure:
1. Start receiver (sleep mode)
2. Wake up via P2P from server
3. Serve 30 requests (trigger GC)
4. Sleep
5. Check spike_reset_s across all 32 ranks

**Success Criteria**:
- All ranks: spike_reset_s < 5s (ideally ~2s)
- Uniform performance (no outlier ranks)
- Total sleep latency < 6s

## Files Modified

1. `nkipy/src/nkipy/vllm_plugin/models/qwen3.py`
   - Separated weight and cache allocation in `_allocate_empty_tensors()`

2. `nkipy/src/nkipy/vllm_plugin/models/llama.py`
   - Same separation for Llama models

## Fallback Plan

If this doesn't work, next investigation areas:
1. Profile what NRT does during spike_reset on slow ranks
2. Check if cache tensor size is the limiting factor
3. Consider alternative cache allocation methods (NRT-specific APIs)
4. Test with smaller TP (e.g., TP=8) to isolate rank-specific behavior

## References

- Previous fix: `allocate_uninitialized()` eliminated first write (zeros → RDMA)
- TEST_RESULTS_FINAL.md: Shows rank 0 improvement, ranks 1-31 still slow
- OPTIMIZATION_SUMMARY.md: Complete history of sleep latency optimization
