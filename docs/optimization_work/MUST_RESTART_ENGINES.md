# CRITICAL: Engines Must Be Restarted to Load New Code!

## Current Situation

**Code has been synced**, but **engines are still running old code**.

### Evidence from Latest Test

**Server (sender)**: Fast sleep ✅
```json
{
  "spike_reset_s": 0.18,
  "endpoint_clear_s": 0.0,
  "total_s": 0.64
}
```

**Receiver**: STILL SLOW ❌ (same as before fix)
```json
{
  "spike_reset_s": 18.52,
  "endpoint_clear_s": 27.58,
  "total_s": 46.53,
  "dereg_waited": false
}
```

This proves the receiver is **still running old code** where `receive_from_peer()` doesn't call `dereg_async()`.

## Why Code Sync Alone Isn't Enough

Python modules are loaded when the process starts:
1. Engine starts → loads `transfer.py` into memory
2. Code file updated on disk → engine still uses old in-memory code
3. **Engine must restart** to load new code from disk

## How to Restart Engines

### Current Instance (A) - Server

```bash
# Kill server
fuser -k 8000/tcp
pkill -9 -f "vllm_plugin.server.*port 8000"
sleep 3

# Restart with new code
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_1.sh > server.log 2>&1 &
tail -f server.log  # Wait for "Application startup complete"
```

### Instance B (or port 8001) - Receiver

```bash
# Kill receiver
fuser -k 8001/tcp
pkill -9 -f "vllm_plugin.server.*port 8001"
sleep 3

# Restart with new code
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_2.sh > receiver.log 2>&1 &
tail -f receiver.log  # Wait for "Application startup complete"
```

## Verify New Code Is Loaded

After restarting receiver and running a P2P transfer, check logs:

```bash
# Look for the new log message in receiver logs
grep "Started async MR deregistration" receiver.log
```

**Should see**:
```
Rank 0: Started async MR deregistration (435 MRs)
```

If you DON'T see this message, the old code is still running!

## Expected Results After Restart

### After P2P Transfer (with new code):

**Receiver logs should show**:
```
Rank 0: P2P receive complete — 435 bufs, 15360.00 MB, 8.23s, 14.93 Gbps
Rank 0: Started async MR deregistration (435 MRs)  ← NEW!
```

### Receiver /sleep (early - 5s):
```json
{
  "dereg_waited": true,        ← Changed from false
  "dereg_wait_s": 15-20,       ← NEW: waiting time
  "spike_reset_s": < 2,        ← Changed from 18.5s!
  "endpoint_clear_s": < 0.01,  ← Changed from 27.6s!
  "total_s": 15-21             ← Changed from 46.5s!
}
```

### Receiver /sleep (late - 35s):
```json
{
  "dereg_waited": false,
  "dereg_wait_s": 0,
  "spike_reset_s": < 2,        ← Changed from 18.5s!
  "endpoint_clear_s": < 0.01,  ← Changed from 27.6s!
  "total_s": < 2               ← Changed from 46.5s! ✅✅✅
}
```

## Quick Restart Command

**Kill both engines and restart:**
```bash
# Kill all
fuser -k 8000/tcp 8001/tcp
pkill -9 -f vllm_plugin.server
sleep 5

# Restart server (Terminal 1)
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_1.sh > server.log 2>&1 &

# Restart receiver (Terminal 2)
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_2.sh > receiver.log 2>&1 &

# Wait for both to be ready (~2-3 min each)
tail -f server.log
tail -f receiver.log

# Run test (Terminal 3)
bash test_hypothesis_manual.sh
```

## Summary

✅ **Code synced** to Instance B  
❌ **Engines NOT restarted** - still running old code  
⏳ **Action required**: Restart both engines to load new code  

The fix is correct and deployed, but **Python only loads code at process startup**. After restart, you should see fast sleep on the receiver side!
