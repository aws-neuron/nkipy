# P2P Architecture Constraints

## MR Registration Strategy

### Server (Sender) Side
- **Can use either**:
  - `NKIPY_PREREGISTER_MRS=1`: Pre-register all 435 MRs at startup (~2s overhead)
  - `NKIPY_PREREGISTER_MRS=0`: On-demand registration during push (~2s per transfer)

- **With Pre-registration**:
  - MRs registered once at init
  - Single-shot P2P transfer (no chunking)
  - No dereg after transfer (MRs persist)
  - Fast /wake_up, but slow /sleep if not deregistered

- **With On-demand**:
  - MRs registered during push_to_peer()
  - Calls `dereg_async()` after transfer
  - +2s overhead per transfer
  - But enables fast /sleep (~0.2-1s)

### Receiver Side
- **CONSTRAINT: Receivers NEVER pre-register MRs**
  - Stated in `how_to_optimize.md`: "Don't pre-register MRs at the receiver side"
  - Reason: Receiver starts sleeping with no model loaded
  - No buffers to register until /wake_up is called

- **Always uses on-demand chunked registration**:
  1. /wake_up called with peer_url
  2. Model allocated (empty tensors)
  3. Buffers collected
  4. P2P receive starts → registers MRs in chunks
  5. Transfer completes
  6. **MUST call `dereg_async()`** ← Critical!

## Code Implications

### `push_to_peer()` (Server)
```python
pre_registered = ep.registered and len(ep.xfer_descs) == len(buffers)

if pre_registered:
    # Single-shot transfer
    # ... transfer ...
    # No dereg - MRs persist
    return

# On-demand path
# ... register + transfer ...
if is_last_chunk and not pre_registered:
    ep.dereg_async()  # Cleanup
```

### `receive_from_peer()` (Receiver)  
```python
# Receivers NEVER pre-register
# Always use chunked on-demand registration

for chunk in chunks:
    # Register chunk
    # Transfer chunk
    # Reregister next chunk

# After all chunks complete
ep.dereg_async()  # Always called! ← Critical
```

## Why This Matters

### Without Receiver dereg_async():
- 435 MRs remain active after P2P transfer
- /sleep called → spike_reset() with active MRs → **18.5s** (60x slow!)
- Endpoint cleared → destructor runs → synchronous dereg → **27.6s**
- **Total: 46.5s sleep instead of 1s!**

### With Receiver dereg_async():
- Background thread deregisters 435 MRs over ~20-25s
- Early /sleep (5s) → waits for dereg → spike_reset in ~2s → total ~20s
- Late /sleep (35s) → dereg done → spike_reset in ~2s → **total ~1s** ✅

## Current Implementation (Fixed)

### Server Side
- `push_to_peer()` line 246:
  ```python
  if is_last_chunk and not pre_registered:
      ep.dereg_async()
  ```

### Receiver Side  
- `receive_from_peer()` line 188:
  ```python
  # Deregister MRs asynchronously after on-demand receive completes
  ep.dereg_async()
  logger.info("Rank %d: Started async MR deregistration (%d MRs)", ...)
  ```

## Production Recommendation

**Use `NKIPY_PREREGISTER_MRS=0` (on-demand) for both**:
- Server: +2s on /wake_up, but enables fast /sleep
- Receiver: Always on-demand (no choice)
- Both call `dereg_async()` after transfer
- Both achieve ~1s sleep latency (late sleep)

## Constraints Summary

| Component | Pre-register? | dereg_async()? | Rationale |
|-----------|---------------|----------------|-----------|
| **Server** | Optional | Yes (if on-demand) | Can pre-register at startup |
| **Receiver** | **NO** | **YES (always)** | Starts sleeping, no buffers yet |

**Critical**: Receivers must ALWAYS call `dereg_async()` after `receive_from_peer()` completes, regardless of any other settings!
