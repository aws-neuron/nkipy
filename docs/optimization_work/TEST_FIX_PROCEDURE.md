# Test Procedure for Sleep Latency Fix

## Fix Applied

**Problem**: Receiver was writing zeros then overwriting via RDMA (double-write), causing 15s `spike_reset()` latency.

**Solution**: Use `DeviceTensor.allocate_uninitialized()` - allocate without writing, let RDMA write directly (single-write).

**Files Changed**:
- `nkipy/src/nkipy/runtime/device_tensor.py` - Added `allocate_uninitialized()` method
- `nkipy/src/nkipy/vllm_plugin/models/qwen3.py` - Use uninitialized allocation
- `nkipy/src/nkipy/vllm_plugin/models/llama.py` - Use uninitialized allocation

## Test Procedure

### Prerequisites

1. **Clean environment**: Restart both instances or ensure Neuron cores are free
2. **Synced code**: Instance B has the latest code with fix

### Step 1: Start Server (Instance A)

```bash
# Instance A: 172.31.44.131
cd /home/ubuntu/vllm-nkipy/nkipy
killall -9 python3; sleep 5
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_1.sh > /tmp/server.log 2>&1 &

# Wait for startup (check logs)
tail -f /tmp/server.log
# Wait for: "Application startup complete"

# Verify
curl http://localhost:8000/nkipy/health
```

### Step 2: Start Receiver (Instance B)

```bash
# Instance B: 172.31.40.200
cd /home/ubuntu/vllm-nkipy/nkipy
killall -9 python3; sleep 5
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver.log 2>&1 &

# Wait for startup
tail -f /tmp/receiver.log
# Wait for: "Application startup complete"

# Verify
curl http://localhost:8000/nkipy/health
```

### Step 3: Run End-to-End Test (Instance B)

```bash
# Copy test script (if not already there)
# scp /tmp/test_final_fix.sh ubuntu@172.31.40.200:/tmp/

# Run the test
bash /tmp/test_final_fix.sh
```

OR run manually:

```bash
# 3.1: Wake up (P2P transfer)
echo "=== Wake up ==="
curl -X POST http://localhost:8000/nkipy/wake_up \
    -H "Content-Type: application/json" \
    -d '{"peer_url": "http://172.31.44.131:8000"}' \
    | python3 -m json.tool | tee wake_result.json

# 3.2: Serve 30 requests
echo "=== Serving 30 requests ==="
for i in {1..30}; do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hi", "max_tokens": 5}' \
        > /dev/null && echo -n "." || echo -n "E"
    sleep 0.2
done
echo ""

# 3.3: Wait for GC
sleep 5

# 3.4: Sleep and measure
echo "=== Sleep (with fix) ==="
curl -X POST http://localhost:8000/nkipy/sleep \
    -H "Content-Type: application/json" \
    | python3 -m json.tool | tee sleep_result.json
```

## Expected Results

### Before Fix (Baseline - from previous tests):
```json
{
  "status": "sleeping",
  "latency": {
    "endpoint_clear_s": 0.0019,
    "cache_check_s": 0.0,
    "pre_gc_s": 0.001,
    "spike_reset_s": 15.0945,  ← SLOW
    "clear_refs_s": 0.0217,
    "total_s": 15.1181  ← UNACCEPTABLE
  }
}
```

### After Fix (Expected):
```json
{
  "status": "sleeping",
  "latency": {
    "endpoint_clear_s": 0.001,
    "cache_check_s": 0.0,
    "pre_gc_s": 0.001,
    "spike_reset_s": 1.5-2.5,  ← FAST! (down from 15s)
    "clear_refs_s": 0.02,
    "total_s": 1.6-2.6  ← ACCEPTABLE! ✅
  }
}
```

## Success Criteria

✅ **Fix is successful if**:
- `spike_reset_s` < 5s (ideally 1.5-2.5s)
- `total_s` < 5s (ideally 2-3s)
- Latency matches server-side sleep (~2s)

⚠️ **Partial success if**:
- `spike_reset_s` 5-10s (improved but not optimal)
- May need additional investigation

❌ **Fix failed if**:
- `spike_reset_s` still >10s (no improvement)

## What to Check

### 1. Verify fix is applied

Check receiver logs during wake_up for any errors about `allocate_uninitialized`:

```bash
grep -i "allocate_uninitialized\|AttributeError" /tmp/receiver.log
```

Should see NO errors. If you see `AttributeError: 'DeviceTensor' object has no attribute 'allocate_uninitialized'`, the code wasn't synced properly.

### 2. Verify P2P transfer works

Wake_up should complete successfully:
```json
{
  "status": "awake",
  "latency": {
    "p2p_transfer_s": 4-6,  // P2P should still work
    "total_s": 20-30
  }
}
```

### 3. Verify inference works

Completions should return valid text, not errors.

### 4. Check detailed logs

```bash
# On Instance B, check for sleep latency breakdown
grep "sleep latency breakdown" /tmp/receiver.log | tail -1
```

All ranks should show similar improvements.

## Comparison: Before vs After

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| **spike_reset_s** | 15-17s | 1.5-2.5s | **85-90% faster** |
| **clear_refs_s** | 0.02s | 0.02s | (already optimized) |
| **total_s** | 15s | 2-3s | **80-85% faster** |

## Troubleshooting

### If spike_reset_s is still slow:

1. **Check code sync**: Verify Instance B has the updated files
   ```bash
   grep -n "allocate_uninitialized" /home/ubuntu/vllm-nkipy/nkipy/nkipy/src/nkipy/runtime/device_tensor.py
   # Should show the new method around line 50-75
   ```

2. **Check for AttributeError**: If `allocate_uninitialized` doesn't exist, code wasn't synced

3. **Verify it's using the new path**: Add debug logging to confirm `allocate_uninitialized` is being called

4. **Check if there are other issues**: Look for NRT errors or warnings in logs

### If wake_up fails:

1. **Server not running**: Check Instance A
2. **Network issue**: Verify connectivity between instances
3. **Neuron cores busy**: Restart receiver

### If inference fails:

1. **Uninitialized memory issue**: RDMA may not have filled all memory
   - This would indicate a bug in the fix
   - Check for NaN outputs or crashes

2. **Model corruption**: Weights may not have transferred correctly

## Alternative: Test Immediate Sleep (Sanity Check)

To verify we didn't break immediate sleep:

```bash
# After wake_up, sleep immediately (no serving)
curl -X POST http://localhost:8000/nkipy/wake_up \
    -d '{"peer_url": "http://172.31.44.131:8000"}' && \
curl -X POST http://localhost:8000/nkipy/sleep | python3 -m json.tool
```

**Expected**: Should still work, `clear_refs_s` may be higher (~13s) but that's okay for immediate sleep.

## Files to Save for Analysis

After running the test, save these files:

```bash
# On Instance B
cp /tmp/receiver.log ~/receiver_with_fix.log
cp wake_result.json ~/wake_with_fix.json
cp sleep_result.json ~/sleep_with_fix.json
```

Then you can compare with baseline results to confirm improvement.

## Summary

This fix eliminates the double-write pattern that caused device memory fragmentation. By writing weights only once (via RDMA to uninitialized memory), we expect receiver sleep latency to drop from **15s to ~2s**, matching server performance.

The test procedure above will confirm whether the fix works as expected.
