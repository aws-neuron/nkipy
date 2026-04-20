# Current Test Status - RDMA Acknowledgement Hypothesis

## Date: 2026-04-16 22:30

## Implementation Complete ✅

**Files modified:**
1. `nkipy/src/nkipy/vllm_plugin/worker.py`:
   - Added `_acknowledge_rdma_writes()` method
   - Integrated into `nkipy_wake_up()` after RDMA transfer
   - Added `rdma_ack_s` timing to latency reporting

2. Previous optimizations also in place:
   - `allocate_uninitialized()` (device_tensor.py, models/*.py)
   - Cache tensor separation (models/qwen3.py, models/llama.py)
   - Clear refs before spike_reset (worker.py)

**Configuration:**
- `NKIPY_RDMA_ACK_SAMPLES=10` (default, reads 10 tensors)
- Can be disabled with `NKIPY_RDMA_ACK_SAMPLES=0` for A/B testing

## Testing Blocked ❌

**Issue:** vLLM shared memory broadcast timeout during wake_up

**Symptoms:**
```
INFO [shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds.
This typically happens when some processes are hanging or doing some time-consuming work
```

**Impact:**
- wake_up hangs indefinitely
- Cannot complete hypothesis test
- Not related to our changes (vLLM internal issue)

**Environment state:**
- Instance B receiver starts successfully
- P2P listeners active ("Waiting to accept incoming connection...")
- Hangs during wake_up collective operations

## Root Cause Analysis

**Shared memory broadcast** is vLLM's mechanism for distributing data across worker processes (TP=32).

**Possible causes:**
1. **Worker process deadlock** - some ranks waiting, others not responding
2. **Shared memory exhaustion** - previous runs left stale segments
3. **Network issue** - collective ops timing out
4. **NRT initialization race** - some ranks slower to init

**Why it's intermittent:**
- Earlier test (TEST_RESULTS_FINAL.md) completed successfully
- Recent attempts all hang
- Suggests resource exhaustion or state corruption

## Workarounds Attempted

1. ✅ Clean neuron cores (`pkill -9 VLLM`)
2. ✅ Clean shared memory (`ipcrm`)
3. ✅ Restart receiver
4. ❌ Still hangs during wake_up

## Recommendations

### Option 1: Debug vLLM Shared Memory (Immediate)

```bash
# On Instance B
# Check shared memory usage
ipcs -m

# Check for zombie processes
ps aux | grep VLLM

# Check vLLM logs for all ranks
grep "EngineCore" /tmp/receiver.log | grep -v "Waiting to accept"

# Try reducing TP to isolate
export TP=8  # Instead of 32
bash examples/p2p/run_vllm_qwen_1_receiver.sh
```

### Option 2: Test Without P2P (Validate Hypothesis Indirectly)

Test the acknowledgement concept without RDMA:

```python
# In a simple test script
from nkipy.runtime import DeviceTensor
import numpy as np

# Allocate uninitialized
tensor = DeviceTensor.allocate_uninitialized((1000,), np.float32)

# Write via external means (simulate RDMA)
# ... RDMA write would happen here ...

# Acknowledge by reading back
_ = tensor.numpy().flat[0]

# Now spike_reset should be fast
```

### Option 3: Wait for Environment Stabilization

- The test previously worked (TEST_RESULTS_FINAL.md was successful)
- Environment may need cleanup/restart
- Could try again later or on fresh instances

### Option 4: Test on Smaller TP

Reduce from TP=32 to TP=8 to:
- Reduce collective operation complexity
- Lower shared memory requirements  
- Easier to debug if issues persist

```bash
# Modify run script
export TP=8
bash examples/p2p/run_vllm_qwen_1_receiver.sh
```

## Expected Test Results (When Unblocked)

**If hypothesis is CORRECT:**

wake_up latency:
```json
{
  "p2p_transfer_s": 15-20,
  "rdma_ack_s": 0.5-2.0,  ← NEW overhead
  "total_s": 20-25
}
```

sleep latency (ALL ranks):
```json
{
  "spike_reset_s": 2-5,  ← Down from 18s!
  "total_s": 3-6
}
```

**If hypothesis is WRONG:**

sleep latency:
```json
{
  "spike_reset_s": 17-18,  ← Still slow
  "total_s": 18-19
}
```

## Next Steps

1. **Immediate:** Investigate shared memory broadcast issue
   - Check if it's a vLLM bug or environment issue
   - Try reducing TP or restarting instances

2. **Once environment is stable:**
   - Run hypothesis test (`/tmp/test_hypothesis.sh` on Instance B)
   - Compare with/without acknowledgement (set `NKIPY_RDMA_ACK_SAMPLES=0`)
   - Analyze results and confirm/refute hypothesis

3. **If hypothesis is confirmed:**
   - Optimize sample count (maybe 1-5 tensors is enough)
   - Add configuration option
   - Update documentation
   - Create PR with all optimizations

4. **If hypothesis is refuted:**
   - Profile NRT internals to find real bottleneck
   - Contact Neuron team for guidance
   - Consider alternative approaches (keep NRT alive, etc.)

## Files Ready for Testing

1. `test_hypothesis.sh` - automated test script
2. `HYPOTHESIS_TEST_PLAN.md` - detailed test procedure
3. `RDMA_NRT_HYPOTHESIS.md` - complete explanation
4. `worker.py` - implementation with acknowledgement

**Status:** Implementation complete, waiting for stable test environment.
