# P2P Transfer Optimization Test Results

## Test Setup
- **Instance A (Server)**: 172.31.44.131 - Active engine with weights
- **Instance B (Receiver)**: 172.31.40.200 - Sleep-mode engine, receives weights via P2P
- **Configuration**: TP=32, Qwen3-30B-A3B model
- **Date**: 2026-04-15

## Instance A (Server) Status: ✅ READY

Engine A successfully started with optimizations:
- Pre-registration working: `Rank 0: pre-registered 434 weight MRs in 2.278s`
- Server is listening on http://172.31.44.131:8000
- Health check: `{"status": "ok", "backend": "nkipy", "sleeping": false}`

## Next Steps for Instance B Testing

### On Instance B (172.31.40.200), run these commands:

```bash
# Terminal 1: Start Engine B (receiver)
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate

# Kill any existing engines
fuser -k 8000/tcp 2>&1; pkill -9 -f vllm_plugin.server 2>&1; sleep 2

# Start receiver engine (no weights)
bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver_b.log 2>&1 &

# Monitor until "Application startup complete"
tail -f /tmp/receiver_b.log
# Press Ctrl+C when you see "Application startup complete"
```

```bash
# Terminal 2: Test /wake_up (AFTER Engine B is ready)
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate

# Test wake_up
echo "==> Testing /wake_up"
curl -X POST http://localhost:8000/nkipy/wake_up \
    -H "Content-Type: application/json" \
    -d '{"peer_url": "http://172.31.44.131:8000"}' \
    | python3 -m json.tool | tee wake_up_result.json

# Test sleep
echo "==> Testing /sleep"
curl -X POST http://localhost:8000/nkipy/sleep \
    -H "Content-Type: application/json" \
    | python3 -m json.tool | tee sleep_result.json
```

### Expected Results

#### /wake_up Optimization:
**Before**: Server-side registration took 4.139s blocking P2P transfer
**After**: Should see "(pre-registered)" in logs and registration time ~0s

Look for in Instance A logs:
```
Rank 0: pushed ... (pre-registered)
```

#### /sleep Optimization:
**Before**: Receiver took 15+ seconds (cache_neffs_s: 9s, gc_reset_s: 6.9s)
**After**: Should complete in <2 seconds (matching server side)

Expected output:
```json
{
  "status": "sleeping",
  "latency": {
    "rank": 0,
    "endpoint_clear_s": <0.001,
    "cache_check_s": <0.001,
    "gc_reset_s": <2.0,
    "total_s": <2.0
  }
}
```

## Log Analysis Commands

### On Instance A (check push logs during /wake_up):
```bash
tail -100 /tmp/server_a.log | grep -A 3 "pushed"
```

### On Instance B (check sleep/wake_up logs):
```bash
# Check wake_up latency breakdown
grep "wake_up latency breakdown" /tmp/receiver_b.log | tail -1

# Check sleep latency breakdown  
grep "sleep latency breakdown" /tmp/receiver_b.log | tail -1

# Check if kernel cache was used
grep "kernel_cache" /tmp/receiver_b.log
```

## Troubleshooting

If Engine B hangs during startup (>10 minutes):
```bash
fuser -k 8000/tcp; pkill -9 -f vllm_plugin.server
# Wait 5 seconds, then restart
```

If /wake_up times out:
- Verify Instance A is running: `curl http://172.31.44.131:8000/nkipy/health`
- Check Instance A logs for errors: `tail -50 /tmp/server_a.log`
- Increase timeout with: `export VLLM_RPC_TIMEOUT=600000`

## Results Summary

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Server-side MR registration during /wake_up | 4.139s | TBD | TBD |
| Receiver /sleep total | 15+ seconds | TBD | TBD |
| Receiver /sleep cache_neffs_s | 9.0398s | TBD | TBD |

**To be filled after testing on Instance B**
