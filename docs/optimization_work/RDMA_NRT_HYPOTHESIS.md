# RDMA vs NRT Memory Tracking Hypothesis

## Date: 2026-04-16

## The Problem

**Current state:**
- Server `/sleep`: ~2s (all ranks)
- Receiver `/sleep`: 7.67s (rank 0), 18s (ranks 1-31)

**Previous optimizations tried:**
1. ✅ `allocate_uninitialized()` - eliminated double-write
   - Result: Rank 0 improved 52% (15s → 7.67s)
   - Result: Ranks 1-31 NO improvement (still ~18s)

2. ✅ Cache tensor separation - reduced fragmentation
   - Status: Implemented but not tested yet (environment issues)

**Why the big rank disparity?**

## The Hypothesis

**Root cause: NRT doesn't properly track RDMA-written memory.**

### Memory Write Paths

**Server (Fast path - 2s cleanup):**
```
1. allocate_tensor()           → NRT tracks allocation
2. tensor_write_from_pybuffer() → NRT tracks write
3. Model execution             → NRT tracks usage
4. spike_reset() / nrt_close() → NRT knows exact state
   └─ Fast cleanup: ~2s
```

**Receiver (Slow path - 18s cleanup):**
```
1. allocate_tensor()           → NRT tracks allocation  
2. RDMA hardware writes        → **BYPASSES NRT!**
   └─ Direct HBM write via EFA
   └─ NRT allocation table: "allocated but unwritten"
3. Model execution             → Uses RDMA-written data
4. spike_reset() / nrt_close() → NRT sees "dirty" state
   └─ Expensive validation/scan: ~18s
```

### Why This Explains Everything

1. **Rank 0 is faster (7.67s):**
   - Fewer expert weights in MoE model
   - Less RDMA-written memory
   - Less NRT confusion

2. **Ranks 1-31 are slow (18s):**
   - More expert weights
   - More RDMA-written memory
   - More NRT validation required

3. **Server is always fast (2s):**
   - No RDMA, all writes go through NRT
   - NRT has perfect tracking

4. **Correlates with data size, not tensor count:**
   - allocate_uninitialized helped (fewer operations)
   - But RDMA volume still matters

## The Test

### Implementation

Added `_acknowledge_rdma_writes()` in `worker.py`:

```python
def _acknowledge_rdma_writes(model):
    """Force NRT to acknowledge RDMA-written memory."""
    # Read back a sample of weight tensors
    for layer in model.layer_tensors[:10]:
        for key, tensor in layer.items():
            if key not in ("cache_k", "cache_v"):
                _ = tensor.numpy().flat[0]  # Touch memory
                break
```

**Called after RDMA transfer in `nkipy_wake_up()`:**
```python
receive_from_peer(...)      # RDMA writes
_acknowledge_rdma_writes()  # Force NRT to see writes
```

### What This Does

1. **Reads back tensors via NRT API** (`tensor_read_to_pybuffer`)
2. **Forces NRT to access device memory**
3. **NRT discovers memory has been modified**
4. **NRT updates internal tracking structures**
5. **spike_reset() now has correct state** → fast cleanup

### Configuration

Control via environment variable:
```bash
export NKIPY_RDMA_ACK_SAMPLES=10  # Default: sample 10 tensors
export NKIPY_RDMA_ACK_SAMPLES=0   # Disable (for A/B testing)
export NKIPY_RDMA_ACK_SAMPLES=60  # Read all layers (if helpful)
```

## Expected Results

### If Hypothesis is CORRECT ✅

**wake_up latency:**
- `rdma_ack_s`: +0.5-2s (reading sample)
- `total_s`: Slightly higher but acceptable

**sleep latency:**
- `spike_reset_s`: 18s → **2-5s** (all ranks)
- Uniform performance across all ranks
- Total sleep: ~3-6s (acceptable!)

**Trade-off:**
- Pay ~1s during wake_up (infrequent)
- Save ~13s during sleep (frequent in autoscaling)
- Net win for serverless use case

### If Hypothesis is WRONG ❌

**wake_up latency:**
- `rdma_ack_s`: +0.5-2s (wasted overhead)

**sleep latency:**
- `spike_reset_s`: Still ~18s (no improvement)
- Need to investigate other causes

**Next steps if wrong:**
1. Profile NRT internals to see what takes 18s
2. Check if it's truly cache tensors (though unlikely)
3. Contact Neuron team for NRT guidance
4. Consider keeping NRT alive between cycles

## Testing Procedure

### 1. Test with acknowledgement enabled (default)

```bash
# On Instance B
ssh ubuntu@172.31.40.200
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate

# Clean start
pkill -9 VLLM; sleep 3

# Start receiver
bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver.log 2>&1 &
sleep 60  # Wait for startup

# Test cycle
curl -X POST http://localhost:8000/nkipy/wake_up \
    -H "Content-Type: application/json" \
    -d '{"peer_url": "http://172.31.44.131:8000"}' | python3 -m json.tool

# Serve 30 requests
for i in $(seq 1 30); do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hello", "max_tokens": 5}' \
        > /dev/null
done

sleep 5  # GC

# Sleep and check results
curl -X POST http://localhost:8000/nkipy/sleep | python3 -m json.tool

# Check all ranks
grep 'sleep latency breakdown' /tmp/receiver.log | tail -32
```

### 2. Test with acknowledgement disabled (A/B test)

```bash
# Set env var to disable
export NKIPY_RDMA_ACK_SAMPLES=0

# Restart receiver with env var
pkill -9 VLLM; sleep 3
bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver_no_ack.log 2>&1 &

# Run same test cycle
# Compare results
```

### 3. Compare results

**Metrics to compare:**

| Metric | With Ack | Without Ack | Improvement |
|--------|---------|-------------|-------------|
| wake_up.rdma_ack_s | ~1s | 0s | Cost |
| wake_up.total_s | ~25s | ~24s | Acceptable |
| sleep.spike_reset_s (rank 0) | **?** | 7.67s | **Target: <5s** |
| sleep.spike_reset_s (rank 1-31) | **?** | 18s | **Target: <5s** |
| sleep.total_s (all ranks) | **?** | 18s | **Target: <6s** |

**Success criteria:**
- spike_reset_s < 5s for ALL ranks
- Uniform performance (no outliers)
- Total sleep < 6s

## Implementation Status

**Files modified:**
1. `nkipy/src/nkipy/vllm_plugin/worker.py`:
   - Added `_acknowledge_rdma_writes()` method
   - Integrated into `nkipy_wake_up()` after P2P transfer
   - Added `rdma_ack_s` to latency reporting

**Combined with previous optimizations:**
- `allocate_uninitialized()` (device_tensor.py, models/*.py)
- Cache tensor separation (models/qwen3.py, models/llama.py)
- Clear refs before spike_reset (worker.py)

**Ready to test:** Yes, pending environment stability

## Fallback Plan

If acknowledgement helps but overhead is too high:

1. **Optimize sample size:**
   - Try reading just 1 tensor per rank
   - Or just on rank 0 (if that broadcasts state)

2. **Lazy acknowledgement:**
   - Skip if going to serve immediately
   - Only do if sleep is imminent

3. **Batch operations:**
   - Read multiple tensors in parallel
   - Use async I/O if available

## Alternative Solutions (if hypothesis is wrong)

1. **NRT memory fence API:**
   - Check if NRT/Neuron has explicit sync API
   - `neuron_memory_barrier()` or similar

2. **UCCL completion:**
   - Ensure UCCL RDMA operations are fully completed
   - Add explicit barrier after P2P

3. **NRT optimization:**
   - Work with Neuron team to optimize RDMA path
   - Maybe NRT needs a "mark_external_write()" API

4. **Keep NRT alive:**
   - Don't call spike_reset() at all
   - Reuse same NRT instance across wake/sleep cycles
   - Just clear model references

## Conclusion

This hypothesis is testable, low-risk, and addresses the core mystery:
**Why does server sleep in 2s while receiver takes 18s?**

The RDMA bypass theory explains:
- ✅ Why rank 0 improved but others didn't
- ✅ Why server is always fast
- ✅ Why it correlates with data size
- ✅ Why previous optimizations had limited effect

**Next step:** Test when environment is stable and confirm/refute hypothesis.
