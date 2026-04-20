# Sleep Latency Hypothesis Test Plan

## Hypothesis

**The receiver's `/sleep` latency (14s) is much higher than the server's (2s) because:**
- Server has been serving inference requests → Python GC has run naturally
- Receiver calls `/sleep` immediately after `/wake_up` → 434 fresh tensor references need cleanup (13.5s)

**If the receiver serves requests before sleeping, the latency should match the server's ~2s.**

## Test Scripts Created

### 1. `test_wake_serve_sleep.sh` - Single Realistic Test
```bash
bash examples/p2p/test_wake_serve_sleep.sh [peer_url] [engine_url] [num_requests]
```

Runs: wake_up → serve N requests → sleep

### 2. `compare_sleep_scenarios.sh` - Full Comparison
```bash
bash examples/p2p/compare_sleep_scenarios.sh [peer_url] [engine_url]
```

Automatically runs both tests and compares results.

### 3. `simple_test.sh` - Manual Step-by-Step
```bash
# On Instance A (server)
bash examples/p2p/simple_test.sh start-server
# Wait for "Application startup complete"

# On Instance B (receiver) - SSH there first
cd /home/ubuntu/vllm-nkipy/nkipy

# Test 1: Immediate sleep
bash examples/p2p/simple_test.sh start-receiver
# Wait for ready
bash examples/p2p/simple_test.sh test-immediate

# Test 2: Sleep after serving
bash examples/p2p/simple_test.sh start-receiver
# Wait for ready
bash examples/p2p/simple_test.sh test-after-serving

# Compare results
bash examples/p2p/simple_test.sh compare
```

## Expected Results

### Test 1: Immediate Sleep (Current Behavior)
```json
{
  "status": "sleeping",
  "latency": {
    "endpoint_clear_s": 0.0005,
    "cache_check_s": 0.0,
    "spike_reset_s": 0.6,
    "clear_refs_s": 13.5,  ← BOTTLENECK
    "total_s": 14.1
  }
}
```

### Test 2: Sleep After Serving (Hypothesis)
```json
{
  "status": "sleeping",
  "latency": {
    "endpoint_clear_s": 0.0005,
    "cache_check_s": 0.0,
    "spike_reset_s": 0.6,
    "clear_refs_s": 0.8,   ← Should be much faster!
    "total_s": 1.4-2.0     ← Matches server side!
  }
}
```

**Key metric**: `clear_refs_s` should drop from **13.5s → <1s**

## Why This Matters

### If Hypothesis is CONFIRMED:
✅ **No code changes needed!** The current implementation is already optimal.
- The 14s sleep is artificial (only happens in immediate wake→sleep tests)
- In production, engines will serve requests before sleeping
- Natural GC during serving prepares model for fast teardown

### If Hypothesis is NOT CONFIRMED:
❌ **Code optimization needed:**
- The model cleanup is inherently slow
- Need to investigate: explicit `gc.collect()`, clearing individual tensors, etc.
- May require architectural changes to model lifecycle

## Root Cause Analysis

### Why is the server fast (<2s)?
1. Model has been serving requests for some time
2. Python GC runs periodically during normal operation
3. Temporary objects created during inference are cleaned up
4. Reference counts are normalized
5. When `/sleep` is called, most cleanup is already done

### Why is the receiver slow (14s)?
1. Fresh model just reconstructed in `/wake_up`
2. 434 weight tensors are brand new from Python's perspective
3. No GC cycles have run on them yet
4. Setting `self.model_runner._nkipy_model = None` triggers immediate cleanup of all fresh references
5. Python's reference counting + tensor destructors take 13.5s

### The Key Insight
The server's fast sleep isn't due to different code—it's due to **when** the code runs.
The model's "age" in Python's memory system matters.

## Current Status

**Infrastructure Issues**: Neuron cores not releasing properly, preventing server startup.

**Next Steps**:
1. Clean restart of both instances
2. Run `simple_test.sh` following manual steps
3. Compare sleep latencies between immediate and after-serving scenarios
4. Document results in `OPTIMIZATION_RESULTS.md`

## Quick Test Commands (When Ready)

```bash
# On Instance B (receiver), after both engines are running:

# Test immediate sleep
curl -X POST http://localhost:8000/nkipy/wake_up -H "Content-Type: application/json" \
  -d '{"peer_url": "http://172.31.44.131:8000"}' | python3 -m json.tool

curl -X POST http://localhost:8000/nkipy/sleep -H "Content-Type: application/json" \
  | python3 -m json.tool | grep -A 10 latency

# Restart receiver, then test after serving
# ... wake_up again ...
for i in {1..30}; do 
  curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hi", "max_tokens": 5}' >/dev/null
done
sleep 3

curl -X POST http://localhost:8000/nkipy/sleep -H "Content-Type: application/json" \
  | python3 -m json.tool | grep -A 10 latency
```

Compare the `clear_refs_s` values!
