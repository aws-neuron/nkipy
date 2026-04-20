# Hypothesis Test Plan: RDMA vs NRT Write Path

## Date: 2026-04-16

## Hypothesis

**The slow spike_reset() is caused by NRT not tracking RDMA-written memory properly.**

### Theory

**Server (2s sleep):**
- Writes go through NRT APIs (`tensor_write_from_pybuffer`)
- NRT tracks all writes and memory state
- During `nrt_close()`, NRT knows exactly what to clean up
- Fast cleanup: ~2s

**Receiver (18s sleep):**
- Memory allocated via NRT (`allocate_tensor`)
- **RDMA hardware writes directly to device HBM** (bypassing NRT)
- NRT allocation table shows "allocated but never written"
- During `nrt_close()`, NRT must validate/scan memory regions
- Slow cleanup: ~18s

### Key Evidence

1. Rank 0 improved to 7.17s (has less data, less RDMA writes)
2. Ranks 1-31 still ~18s (more data, more RDMA writes)
3. Server consistently ~2s regardless of data size
4. The pattern correlates with RDMA transfer size, not tensor count

## Test Design

### Test 1: Baseline Measurements (Already Have)

**Server sleep after serving:**
```
Expected: ~2s (confirmed from previous tests)
```

**Receiver sleep after RDMA + serving:**
```
Current: 7.67s (rank 0), 18s (ranks 1-31)
```

### Test 2: Force NRT to Acknowledge RDMA Writes

**Method**: After RDMA transfer, read back a sample of tensors via NRT API.

This forces NRT to:
1. Access the device memory
2. Discover it's been modified
3. Update its internal tracking

**Implementation**:
```python
# In worker.py nkipy_wake_up(), after P2P transfer:
def nkipy_wake_up(self, peer_url: str | None = None) -> dict:
    # ... existing code ...
    
    # After RDMA transfer
    receive_from_peer(rank_endpoint, bufs, actual_peer, ...)
    t_p2p = _time.time()
    
    # NEW: Force NRT to acknowledge RDMA writes
    if actual_peer:
        _acknowledge_rdma_writes(model)
    t_ack = _time.time()
    
    # ... rest of code ...

def _acknowledge_rdma_writes(model):
    """Read back a sample of weight tensors to force NRT tracking update."""
    import gc
    sample_count = 0
    max_samples = 10  # Don't read everything, just enough
    
    for lid, layer in enumerate(model.layer_tensors):
        if sample_count >= max_samples:
            break
        # Read first weight tensor in layer (just metadata check)
        for key, tensor in layer.items():
            if key not in ("cache_k", "cache_v"):
                # Read one byte to force NRT to check memory
                _ = tensor.numpy().flat[0]
                sample_count += 1
                break
    gc.collect()
```

**Expected Results**:

**If hypothesis is CORRECT:**
- Reading back tensors will cause spike_reset to be fast (~2-5s)
- All ranks should improve uniformly
- The "ack" overhead should be minimal (<1s)

**If hypothesis is WRONG:**
- Reading back tensors won't help
- spike_reset will still be slow (~18s)
- Need to investigate other causes

### Test 3: Extreme Case - Read All Tensors

If Test 2 shows improvement, try reading ALL tensors to see maximum effect:

```python
def _acknowledge_all_rdma_writes(model):
    """Read back all weight tensors."""
    for lid, layer in enumerate(model.layer_tensors):
        for key, tensor in layer.items():
            if key not in ("cache_k", "cache_v"):
                _ = tensor.numpy().flat[0]  # Touch memory
```

This will be slower during wake_up but might give fastest sleep.

### Test 4: Alternative - Memory Fence/Sync

Check if UCCL or Neuron has explicit memory fence/sync APIs:

```python
# Check UCCL docs for:
ucc_barrier()  # Wait for all RDMA writes to complete
neuron_memory_fence()  # If such API exists
```

## Metrics to Track

For each test variant, measure:

1. **wake_up latency**:
   - `p2p_transfer_s`: RDMA transfer time
   - `ack_writes_s`: Time to acknowledge writes (new)
   - `total_s`: Total wake_up time

2. **sleep latency**:
   - `spike_reset_s`: NRT cleanup time (main target)
   - `total_s`: Total sleep time

3. **Per-rank variance**:
   - Does the fix make all ranks uniform?
   - Or does rank 0 still differ?

## Success Criteria

**Hypothesis CONFIRMED if:**
- Acknowledging RDMA writes reduces spike_reset from 18s → <5s
- All ranks benefit uniformly
- The overhead is acceptable (<2s added to wake_up)

**Alternative paths if hypothesis is WRONG:**
1. Profile NRT internals to see what actually takes 18s
2. Check if it's GC-related after all (though clear_refs_s is fast)
3. Investigate if cache tensor writes are the real bottleneck
4. Contact Neuron team for NRT optimization guidance

## Implementation Priority

1. **Start with Test 2** (sample read-back) - minimal overhead
2. **If promising**, try Test 3 (full read-back) - measure overhead
3. **If neither helps**, profile NRT or investigate alternatives

## Next Steps

1. Implement `_acknowledge_rdma_writes()` in worker.py
2. Test on Instance B with cache separation fix already in place
3. Compare sleep latency before/after acknowledgement
4. Document results and decide on permanent solution

## Risk Assessment

**Low risk:**
- Reading back tensors is a normal NRT operation
- Won't corrupt data (read-only)
- Can A/B test easily (with/without acknowledgement)

**Potential downsides:**
- Adds overhead to wake_up (~1-2s estimated)
- May not fix the problem (then we wasted time)

**Mitigation:**
- Start with minimal sampling (10 tensors)
- Make it configurable via env var
- Can disable if it doesn't help
