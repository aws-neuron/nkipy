# spike_reset() Latency Analysis

## Problem Statement

After serving inference requests, `spike_reset()` takes 15-17 seconds during `/sleep`, making the overall sleep latency unacceptable (~15s total).

## Test Results

### Test 1: Immediate Sleep (No Serving)
```json
{
  "spike_reset_s": 0.6,  // FAST
  "clear_refs_s": 13.5,  // Python reference cleanup
  "total_s": 14.1
}
```

### Test 2: Sleep After Serving (30 requests)
```json
{
  "spike_reset_s": 15-17,  // VERY SLOW ❌
  "clear_refs_s": 0.02,    // Fast (GC already ran)
  "total_s": 15.1
}
```

##Key Observation

**`spike_reset()` latency depends on prior serving activity:**
- **Without serving**: 0.6s (fast)
- **After serving**: 15-17s (slow)

The slowdown is NOT in Python code - it's in the native `_Spike.close()` → `nrt_close()` path.

## Root Cause Hypothesis

`spike_reset()` calls `nrt_close()` (Neuron Runtime), which must:
1. Tear down execution contexts
2. Deallocate device memory
3. Unload kernels from NeuronCores
4. Flush any pending operations

**After serving requests:**
- Kernels have executed on device
- Device memory contains activations, intermediates
- NRT has internal state from 30 inference runs
- More cleanup is required → 15s instead of 0.6s

**Without serving:**
- Clean device state
- Minimal NRT state
- Fast teardown → 0.6s

## Why is this a Problem?

The original observation said "server sleep is 2s" but current tests show:
- Server sleep (TP=32, after serving): **ALSO ~15s** (curl still running after 90s)
- This suggests the 2s observation was from a different setup (TP=8? Different code version?)

## Current Code Flow

```python
def nkipy_sleep():
    # 1. Clear endpoint descriptors (0.001s)
    rank_endpoint.xfer_descs = []
    
    # 2. Check cache (0s)
    
    # 3. Pre-GC (NEW - testing)
    gc.collect()
    
    # 4. spike_reset() ← 15s BOTTLENECK
    spike_reset()  # Calls _runtime.close() → nrt_close()
    
    # 5. Clear Python refs (0.02s after serving, 13s immediate)
    model = None
```

## Attempted Optimizations

### Attempt 1: Reorder to clear refs first ❌
```python
model = None  # Clear Python refs first
gc.collect()
spike_reset()  # Hope this is faster with refs cleared
```

**Result**: Hung during wake_up with shared memory deadlock
**Reason**: Unclear, but changing order broke something

### Attempt 2: Pre-GC before spike_reset (testing)
```python
gc.collect()  # Cleanup before spike
spike_reset()
model = None
```

**Status**: Currently testing, but unlikely to help since spike_reset slowness is in NRT, not Python

## The Real Question

**Why does NRT cleanup take 15s after serving but only 0.6s without?**

Possible causes:
1. **Device memory fragmentation**: After serving, memory is fragmented and takes longer to deallocate
2. **Kernel unloading**: NRT must unload kernels from cores, and this is slower when they've been executed
3. **Execution context teardown**: NRT maintains execution contexts that need cleanup
4. **Synchronization**: NRT waits for all cores to finish/sync before closing (15s for TP=32)
5. **Resource leaks**: Something isn't being cleaned up properly during serving

## Next Steps to Debug

### 1. Check if TP size matters
Test with TP=8 to see if 15s scales with number of workers:
- TP=8: expect ~4s?
- TP=32: seeing 15s
- Suggests per-worker overhead of ~0.5s

### 2. Check NRT logs
Look for NRT-level errors or warnings during `nrt_close()`:
```bash
grep -i "nrt.*close\|nrt.*error" /tmp/receiver_test2.log
```

### 3. Profile spike_reset internals
Add timing inside spike C++ code to see where the 15s is spent:
- `nrt_close()` entry
- Individual cleanup steps
- Per-core cleanup

### 4. Compare with server
Confirm server also has 15s sleep after serving (current test shows it does)

### 5. Test with fewer inference requests
Does 1 request cause 15s sleep? Or does it scale with number of requests?

### 6. Check for resource leaks
Are there resources not being freed during serving that accumulate?

## Potential Solutions

### Solution 1: Asynchronous spike_reset ⚠️
Start spike_reset in background, return immediately, complete cleanup later.
**Risk**: Next wake_up might conflict with pending cleanup

### Solution 2: Partial reset 🤔
Instead of full `nrt_close()`, do lighter-weight cleanup.
**Challenge**: May not actually free resources

### Solution 3: Keep NRT alive, reset model only ✅
Don't call spike_reset() - just clear model and re-allocate on wake_up.
**Benefit**: Avoids NRT teardown/reinit overhead
**Challenge**: May leak device memory over multiple sleep/wake cycles

### Solution 4: NRT-level optimization 🔧
Fix the underlying NRT slowness (requires Neuron SDK changes)

### Solution 5: Accept the latency 🤷
If production usage is rare sleep (every hours), 15s may be acceptable.

## Recommendation

**Immediate**: Confirm that server also has ~15s sleep after serving (test in progress)

**If server is also slow**: This is expected NRT behavior with TP=32, not a receiver-specific bug. The "2s server sleep" observation was likely from different setup.

**If server is fast**: There's a difference between server and receiver code paths that we need to identify.

**Long-term**: Consider Solution 3 (keep NRT alive) if frequent sleep/wake cycles are needed in production.

## Status

- ✅ Confirmed `clear_refs_s` can be fast (0.02s) after serving
- ❌ spike_reset is still 15s bottleneck
- 🔄 Testing if server has same issue (curl running 90s+)
- ⏳ Need to determine if this is acceptable for production use case
