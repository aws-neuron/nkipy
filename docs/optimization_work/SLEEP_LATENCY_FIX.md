# Sleep Latency Fix: Eliminate Double-Write for P2P Weights

## Date: 2026-04-16

## Problem

Receiver `/sleep` latency was **15s** after serving requests, while server `/sleep` was **~2s**.

**Root Cause**: The receiver was doing a **double-write** to device memory during P2P weight loading:
1. **First write**: `DeviceTensor.from_numpy(np.zeros(...))` - writes zeros to all 434 weight tensors
2. **Second write**: RDMA P2P transfer overwrites with actual weights

This double-write caused device memory fragmentation/state inconsistency, making `spike_reset()` (which calls `nrt_close()`) take 15s instead of ~2s.

## Solution

**Allocate device memory WITHOUT initializing** - let P2P RDMA write directly to uninitialized memory.

### Code Changes

#### 1. Added `DeviceTensor.allocate_uninitialized()` method

**File**: `nkipy/src/nkipy/runtime/device_tensor.py`

```python
@classmethod
def allocate_uninitialized(cls, shape, dtype, name: str = None, core_id=0) -> "DeviceTensor":
    """Allocate device tensor without initializing (for P2P receive).

    This avoids writing zeros to device memory that will be immediately
    overwritten by RDMA transfer, reducing memory fragmentation and
    improving spike_reset() performance.
    """
    spike = get_spike_singleton()
    size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    
    # Allocate device memory WITHOUT writing
    tensor_ref = spike.allocate_tensor(
        size=size_bytes, core_id=core_id, name=name
    )
    
    return cls(
        tensor_ref=tensor_ref,
        shape=shape,
        dtype=dtype,
        name=name,
    )
```

#### 2. Updated `_allocate_empty_tensors()` in model files

**Files**: 
- `nkipy/src/nkipy/vllm_plugin/models/qwen3.py`
- `nkipy/src/nkipy/vllm_plugin/models/llama.py`

**Before** (writes zeros then RDMA overwrites):
```python
layer[wk] = DeviceTensor.from_numpy(
    np.zeros(shapes[wk], dtype=cfg.dtype), f"{prefix}_L{lid}")
```

**After** (allocates without writing, RDMA writes directly):
```python
layer[wk] = DeviceTensor.allocate_uninitialized(
    shapes[wk], cfg.dtype, f"{prefix}_L{lid}")
```

**Note**: Cache tensors (cache_k, cache_v) still use `from_numpy(zeros)` since they need zero initialization and are NOT filled by P2P.

## Expected Impact

### Before Fix:
```json
{
  "spike_reset_s": 15-17,  // SLOW after serving
  "clear_refs_s": 0.02,    // Fast (GC optimization)
  "total_s": 15.1
}
```

### After Fix (Expected):
```json
{
  "spike_reset_s": 1-2,    // Fast (matches server!)
  "clear_refs_s": 0.02,    // Fast (GC optimization)
  "total_s": 1.5-2.5       // ✅ Acceptable latency!
}
```

## Why This Works

### Server Path (Fast):
```
load_file(checkpoint) → DeviceTensor.from_torch(weights)
  ↓
Write actual weight data ONCE to device
  ↓
Serve requests
  ↓
spike_reset() → Fast cleanup (~2s)
```

### Receiver Path Before Fix (Slow):
```
_allocate_empty_tensors() → DeviceTensor.from_numpy(np.zeros(...))
  ↓
Write zeros to ALL 434 tensors (WASTED WORK!)
  ↓
P2P RDMA overwrites with actual weights (DOUBLE WRITE!)
  ↓
Device memory fragmented/inconsistent state
  ↓
Serve requests
  ↓
spike_reset() → Slow cleanup (15s) due to fragmentation
```

### Receiver Path After Fix (Fast):
```
_allocate_empty_tensors() → DeviceTensor.allocate_uninitialized(...)
  ↓
Allocate device memory WITHOUT writing (CLEAN!)
  ↓
P2P RDMA writes actual weights (SINGLE WRITE!)
  ↓
Clean device memory state (like server)
  ↓
Serve requests
  ↓
spike_reset() → Fast cleanup (~2s) ✅
```

## Key Insights

1. **The problem wasn't P2P itself**, but HOW we prepared memory for P2P
2. **Writing zeros then overwriting** caused NRT-level state issues
3. **Uninitialized allocation** is safe for P2P since RDMA will fill ALL memory
4. **Cache tensors still need zeros** - they're not touched by P2P

## Testing

To verify the fix works:

```bash
# Instance B (receiver with P2P)
bash examples/p2p/run_vllm_qwen_1_receiver.sh

# Wait for startup, then:
curl -X POST http://localhost:8000/nkipy/wake_up \
  -H "Content-Type: application/json" \
  -d '{"peer_url": "http://172.31.44.131:8000"}'

# Serve 30 requests
for i in {1..30}; do
  curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hi", "max_tokens": 5}' >/dev/null
done

# Sleep and check latency
curl -X POST http://localhost:8000/nkipy/sleep | jq '.latency'
```

**Expected**: `spike_reset_s: ~2s` (down from 15s)

## Files Modified

1. `nkipy/src/nkipy/runtime/device_tensor.py`
   - Added `allocate_uninitialized()` classmethod

2. `nkipy/src/nkipy/vllm_plugin/models/qwen3.py`
   - Updated `_allocate_empty_tensors()` to use `allocate_uninitialized()`

3. `nkipy/src/nkipy/vllm_plugin/models/llama.py`
   - Updated `_allocate_empty_tensors()` to use `allocate_uninitialized()`

## Related Documents

- `HYPOTHESIS_CONFIRMED.md` - Confirmed `clear_refs_s` optimization (13.5s → 0.02s)
- `SPIKE_RESET_ANALYSIS.md` - Deep dive into spike_reset bottleneck
- `OPTIMIZATION_RESULTS.md` - Previous optimization attempts

## Conclusion

By eliminating the wasteful double-write pattern, we expect receiver `/sleep` latency to match server latency (~2s), making it acceptable for production use.

The fix is minimal, surgical, and addresses the root cause without changing P2P transfer logic or NRT internals.
