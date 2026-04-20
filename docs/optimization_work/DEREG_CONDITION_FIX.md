# Fix: dereg_async() Condition Logic

## Date: 2026-04-20 (Second Fix)

## Problem

After adding `dereg_async()` call to `receive_from_peer()`, the log message `"Started async MR deregistration"` never appeared, and receiver still showed slow sleep (46.5s).

## Root Cause

**Wrong condition logic:**
```python
# WRONG - This was always False!
if not ep.registered and len(ep.xfer_descs) > 0:
    ep.dereg_async()
```

### Why It Failed

1. **After chunked registration**: `ep.xfer_descs` is populated with MRs
2. **`ep.registered` property**: Returns `bool(self.xfer_descs)`
3. **Condition evaluation**: `if not True and True` → `False`
4. **Result**: `dereg_async()` never called!

### The Logic Error

The property `ep.registered` checks if MRs are currently registered:
```python
@property
def registered(self) -> bool:
    return bool(self.xfer_descs)
```

So after chunked registration in `receive_from_peer()`:
- `ep.xfer_descs = [435 MR descriptors]` ← Populated
- `ep.registered = True` ← Property returns True
- `not ep.registered = False` ← Condition fails!

## Solution

### Approach 1: Check Pre-registration Mode

Track whether we're in pre-registration mode at function start:

```python
pre_registered = ep.registered and len(ep.xfer_descs) == len(buffers)

if pre_registered:
    # Single-shot transfer with pre-registered MRs
    # ... transfer ...
    return  # Don't dereg - MRs persist across transfers

# Chunked registration path (on-demand)
# ... chunks ...
# After all chunks complete:
ep.dereg_async()  # Always called for chunked path
```

### Approach 2: Unconditional Call (Current)

Since we only reach the end if NOT pre-registered (early return above), simply call unconditionally:

```python
# At end of receive_from_peer(), after throughput log:
# Deregister MRs asynchronously after on-demand receive completes
# (We only reach here if not pre-registered, see early return above)
ep.dereg_async()
if not dist.is_initialized() or dist.get_rank() == 0:
    logger.info("Rank %d: Started async MR deregistration (%d MRs)", 
               dist.get_rank(), len(ep.xfer_descs))
```

## Code Flow Clarity

### Pre-Registered Path
```python
if pre_registered:
    # ... single-shot transfer ...
    logger.info("P2P receive (pre-registered)")
    return  # ← EXIT early, no dereg
```

### On-Demand Path  
```python
# Chunked registration
for chunk in chunks:
    ep.register(buffers[chunk])
    # ... transfer ...
    ep.reregister(next_chunk)  # or keep MRs

# After all chunks
logger.info("P2P receive complete")
ep.dereg_async()  # ← Always called
logger.info("Started async MR deregistration")
```

## Files Modified

- `nkipy/src/nkipy/p2p/transfer.py`:
  - Line 102: Track `pre_registered` upfront
  - Line 131: Added comment explaining no dereg for pre-registered
  - Line 135: Clarified "chunked registration path (on-demand)"
  - Line 187-191: Simplified dereg call (unconditional)

## Testing

### Before Fix (Wrong Condition):
```
# No log message appeared
Rank 0: P2P receive complete — 435 bufs, 15360.00 MB, 8.23s, 14.93 Gbps
# ❌ dereg_async() not called
```

### After Fix (Correct Logic):
```
# Should see:
Rank 0: P2P receive complete — 435 bufs, 15360.00 MB, 8.23s, 14.93 Gbps
Rank 0: Started async MR deregistration (435 MRs)  ← NEW!
```

## Expected /sleep Results After Fix

Now that `dereg_async()` is actually called:

**Receiver early sleep (5s):**
```json
{
  "dereg_waited": true,
  "dereg_wait_s": 15-20,
  "spike_reset_s": < 2,
  "total_s": 15-21
}
```

**Receiver late sleep (35s):**
```json
{
  "dereg_waited": false,
  "dereg_wait_s": 0,
  "spike_reset_s": < 2,
  "total_s": < 2
}
```

## Deployment

✅ **Instance A** - Fixed  
✅ **Instance B** - Synced  
⚠️ **Engines must restart** to load new code

## Lesson Learned

**Don't confuse "currently has MRs" with "using pre-registration mode"!**

The `ep.registered` property tells you if MRs are currently registered, not whether you're in pre-registration mode. After on-demand chunked registration, `ep.registered` is True because MRs were just registered!

Better to:
1. Track pre-registration mode explicitly at function start
2. Use early return for pre-registered path
3. Unconditional dereg for on-demand path (we only reach it if not pre-registered)
