# Critical Fix: Receiver MR Deregistration

## Date: 2026-04-20

## Problem Discovered During Manual Testing

When testing the hypothesis, the receiver showed:
- `spike_reset_s: 18.66s` ❌ SLOW (expected < 2s)
- `endpoint_clear_s: 28.03s` ❌ VERY SLOW (synchronous dereg in destructor)
- `dereg_waited: False` (conditional wait didn't trigger)
- **Total sleep: 47 seconds!**

Server side (sender) had fast sleep (~few seconds) because it called `dereg_async()`.

## Root Cause

**The receiver never called `dereg_async()` after P2P weight transfer!**

### Code Analysis

**Sender side** (`push_to_peer`):
```python
# Line 246-247
if is_last_chunk and not pre_registered:
    ep.dereg_async()  # ✓ Called on sender
```

**Receiver side** (`receive_from_peer`):
```python
# Line 182-186 (BEFORE FIX)
elapsed = time.time() - t0
throughput_gbps = (total_bytes * 8) / elapsed / 1e9 if elapsed > 0 else 0
logger.info("Rank %d: P2P receive complete — %d bufs, %.2f MB, %.2fs, %.2f Gbps", ...)
# ❌ NO dereg_async() call!
# Function returns with MRs still active
```

### Impact

When receiver calls `/sleep`:
1. MRs are still active (435 MRs registered during receive)
2. Conditional wait checks `_dereg_thread` → `None` (never started)
3. `dereg_waited = False` (no thread to wait for)
4. `spike_reset()` called with active MRs → **18.66s** (60x slowdown)
5. Endpoint cleared → destructor runs → synchronous dereg → **28.03s**
6. **Total: 47s sleep latency instead of ~1s!**

## Solution

Added `dereg_async()` call at the end of `receive_from_peer()`:

```python
# Line 187-192 (AFTER FIX)
# Deregister MRs asynchronously after receive completes (on-demand registration only)
if not ep.registered and len(ep.xfer_descs) > 0:
    ep.dereg_async()
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Rank %d: Started async MR deregistration (%d MRs)", 
                   dist.get_rank(), len(ep.xfer_descs))
```

### Logic
- Only call if using on-demand registration (`not ep.registered`)
- Only if MRs were actually registered (`len(ep.xfer_descs) > 0`)
- Starts background thread to deregister ~435 MRs (~20-25s)
- Logs the start of deregistration for debugging

## Expected Results After Fix

### Receiver /sleep (Early - 5s after wake):
- `dereg_waited: True` (waits for dereg thread)
- `dereg_wait_s: ~15-20s` (remaining time for 435 MRs)
- `spike_reset_s: < 2s` ✅ (fast after wait)
- `endpoint_clear_s: < 0.01s` ✅ (no MRs to dereg)
- **Total: ~15-20s** (acceptable for early sleep)

### Receiver /sleep (Late - 35s after wake):
- `dereg_waited: False` (dereg already complete)
- `dereg_wait_s: 0s`
- `spike_reset_s: < 2s` ✅ (fast, no MRs active)
- `endpoint_clear_s: < 0.01s` ✅ (no MRs to dereg)
- **Total: ~1s** ✅✅✅ (hypothesis confirmed!)

### Server /sleep:
- No change needed (already has fast sleep)
- Already calls `dereg_async()` in `push_to_peer()`

## Why This Was Missed

The previous optimization work focused on:
1. **Server-side** MR registration optimization
2. **Sleep endpoint** conditional wait logic
3. But **never tested receiver-side sleep latency** after P2P transfer!

The hypothesis tests were designed to catch exactly this issue.

## Files Modified

- `nkipy/src/nkipy/p2p/transfer.py` - Added `dereg_async()` to `receive_from_peer()`

## Testing

### Before Fix:
```json
{
  "spike_reset_s": 18.66,
  "endpoint_clear_s": 28.03,
  "total_s": 47.22,
  "dereg_waited": false
}
```

### After Fix (Expected):
**Early sleep (5s):**
```json
{
  "dereg_wait_s": 15-20,
  "dereg_waited": true,
  "spike_reset_s": < 2,
  "endpoint_clear_s": < 0.01,
  "total_s": 15-21
}
```

**Late sleep (35s):**
```json
{
  "dereg_wait_s": 0,
  "dereg_waited": false,
  "spike_reset_s": < 2,
  "endpoint_clear_s": < 0.01,
  "total_s": < 2
}
```

## Deployment Status

✅ **Fixed on Instance A** (current instance)  
✅ **Synced to Instance B**  
⏳ **Ready for re-test**

## Action Required

**Re-run the manual test** with fresh engines to verify the fix:
1. Restart both engines (to load updated code)
2. Run `bash test_hypothesis_manual.sh`
3. Verify both tests show `spike_reset_s < 2s`

## Lesson Learned

**Both sender AND receiver must call `dereg_async()` after on-demand RDMA registration!**
- Sender: after `push_to_peer()` completes
- Receiver: after `receive_from_peer()` completes

Without this, the receiver's sleep will be **47s instead of 1s** - a 47x slowdown!
