# End-to-End Hypothesis Test Instructions

## Hypothesis
**"With proper deregistration timing, /sleep can achieve ~0.2-1s latency"**

## Test Setup (Two Instances)

- **Instance A (172.31.44.131)**: Server with checkpoint - STARTED ✓
- **Instance B (172.31.40.200)**: Receiver, starts sleeping

## Current Status

✅ **Instance A server is starting** (takes ~2-3 minutes)
- Started at: $(date +%H:%M:%S)
- Log: `/home/ubuntu/vllm-nkipy/nkipy/server_a.log`

## Next Steps

### Option 1: Run test from Instance B (SSH)

```bash
# SSH to Instance B
ssh ubuntu@172.31.40.200

# Navigate to repo
cd /home/ubuntu/vllm-nkipy/nkipy

# Start receiver (if not already running)
source .venv/bin/activate
bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver_b.log 2>&1 &

# Wait for receiver to start (~2-3 minutes)
# Check with: curl -s http://localhost:8000/health

# Run the hypothesis test
./test_e2e_dereg_hypothesis.sh
```

### Option 2: Run test remotely from Instance A

```bash
# On Instance A (current instance)
# Wait for server to be ready first
curl -s http://localhost:8000/health  # Should return OK

# Then run remote test script
ssh ubuntu@172.31.40.200 'cd /home/ubuntu/vllm-nkipy/nkipy && bash run_remote_hypothesis_test.sh'
```

## Expected Results

The test will run two scenarios:

### Test 1: Early Sleep (5s after /wake_up)
**Expected**:
- `dereg_waited: True` (waits for MR deregistration to complete)
- `dereg_wait_s: ~15-20s` (waits for 435 MRs to dereg)
- `spike_reset_s: < 2s` (fast after waiting)

### Test 2: Late Sleep (35s after /wake_up)  
**Expected**:
- `dereg_waited: False` (dereg already complete)
- `dereg_wait_s: 0s` (no wait needed)
- `spike_reset_s: < 2s` (fast because MRs already deregistered)

## Success Criteria

✓ **Hypothesis CONFIRMED** if:
- Both tests show `spike_reset_s < 2s`
- Early test: achieves fast spike_reset by waiting
- Late test: achieves fast spike_reset without waiting

## Troubleshooting

### Check Server Status (Instance A)
```bash
# On Instance A
curl -s http://localhost:8000/health
tail -f server_a.log
```

### Check Receiver Status (Instance B)
```bash
# On Instance B
ssh ubuntu@172.31.40.200 "curl -s http://localhost:8000/health"
ssh ubuntu@172.31.40.200 "tail -50 /home/ubuntu/vllm-nkipy/nkipy/receiver_b.log"
```

### Kill and Restart Engines
```bash
# On either instance
fuser -k 8000/tcp
pkill -9 -f vllm_plugin.server
# Then restart with appropriate script
```

## Files Created

- `test_e2e_dereg_hypothesis.sh` - Main test script (run from Instance B)
- `run_full_hypothesis_test.sh` - Full automated setup + test
- `run_remote_hypothesis_test.sh` - Helper for remote execution
- `HYPOTHESIS_VERIFICATION.md` - Verification document with baseline results
