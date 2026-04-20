#!/usr/bin/env bash
# Single-instance hypothesis test with TP=8
# Tests MR deregistration timing without needing Instance B
set -euo pipefail

WEIGHTS=~/zhuangw/nkipy/examples/models/qwen3/tmp_Qwen3-30b-a3b_TP32
TP=32

echo "========================================"
echo "Single-Instance MR Deregistration Test"
echo "========================================"
echo "Testing on Instance A only with TP=$TP"
echo ""
echo "Hypothesis: With proper deregistration timing, /sleep achieves ~0.2-1s latency"
echo ""

# Check checkpoint
if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: Checkpoint not found at $WEIGHTS"
    exit 1
fi

# Function to wait for engine
wait_for_engine() {
    local url=$1
    local name=$2
    local max_wait=${3:-180}

    echo "Waiting for $name at $url..."
    for i in $(seq 1 $max_wait); do
        if curl -s "$url/health" > /dev/null 2>&1; then
            echo "✓ $name ready after ${i}s"
            return 0
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "  ... still waiting ($i/${max_wait}s)"
        fi
        sleep 1
    done
    echo "✗ $name failed to start after ${max_wait}s"
    return 1
}

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    fuser -k 8001/tcp 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    pkill -9 -f "vllm_plugin.server.*port 8001" 2>/dev/null || true
    pkill -9 -f "vllm_plugin.server.*port 8000" 2>/dev/null || true
    sleep 2
}
trap cleanup EXIT

# Initial cleanup
cleanup
echo ""

# Step 1: Start SERVER with checkpoint on port 8001
echo "========================================"
echo "Step 1: Starting Server (port 8001)"
echo "========================================"

export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
export NKIPY_CHECKPOINT=$WEIGHTS
export OMP_NUM_THREADS=1
export NKIPY_SKIP_CTE=1
export VLLM_RPC_TIMEOUT=600000
export NKIPY_PREREGISTER_MRS=0

source .venv/bin/activate
python -m nkipy.vllm_plugin.server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size $TP \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8001 \
    > server_tp8.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

if ! wait_for_engine "http://localhost:8001" "Server" 180; then
    echo "Server log tail:"
    tail -50 server_tp8.log
    exit 1
fi

echo ""

# Step 2: Start RECEIVER without checkpoint on port 8000
echo "========================================"
echo "Step 2: Starting Receiver (port 8000)"
echo "========================================"

unset NKIPY_CHECKPOINT  # Receiver starts in sleeping mode

python -m nkipy.vllm_plugin.server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size $TP \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8000 \
    > receiver_tp8.log 2>&1 &

RECEIVER_PID=$!
echo "Receiver PID: $RECEIVER_PID"

if ! wait_for_engine "http://localhost:8000" "Receiver" 180; then
    echo "Receiver log tail:"
    tail -50 receiver_tp8.log
    exit 1
fi

echo ""

# Function to run a test scenario
run_test() {
    local sleep_delay=$1
    local test_name=$2

    echo "========================================"
    echo "Test: $test_name"
    echo "Sleep delay: ${sleep_delay}s after /wake_up"
    echo "========================================"

    # Wake up receiver from server
    echo "[$(date +%H:%M:%S)] Calling /wake_up on receiver..."
    wake_result=$(curl -s -X POST "http://localhost:8000/nkipy/wake_up" \
        -H "Content-Type: application/json" \
        -d '{"peer_url": "http://localhost:8001"}')

    wake_status=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
        echo "ERROR: Wake up failed with status: $wake_status"
        echo "$wake_result" | python3 -m json.tool
        return 1
    fi

    wake_total=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")
    p2p_time=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('p2p_transfer_s', 0))" 2>/dev/null || echo "0")

    echo "[$(date +%H:%M:%S)] Wake up completed in ${wake_total}s (P2P: ${p2p_time}s)"
    echo "  → dereg_async() started deregistering MRs in background (~20-25s)"

    # Wait specified delay
    echo ""
    echo "[$(date +%H:%M:%S)] Waiting ${sleep_delay}s before /sleep..."
    sleep "$sleep_delay"

    # Call /sleep
    echo ""
    echo "[$(date +%H:%M:%S)] Calling /sleep on receiver..."
    sleep_result=$(curl -s -X POST "http://localhost:8000/nkipy/sleep" \
        -H "Content-Type: application/json")

    sleep_status=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$sleep_status" != "sleeping" ]; then
        echo "ERROR: Sleep failed with status: $sleep_status"
        echo "$sleep_result" | python3 -m json.tool
        return 1
    fi

    # Extract timing metrics
    dereg_wait=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('dereg_wait_s', 0))" 2>/dev/null || echo "0")
    dereg_waited=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('dereg_waited', False))" 2>/dev/null || echo "False")
    spike_reset=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('spike_reset_s', 0))" 2>/dev/null || echo "0")
    sleep_total=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")

    echo "[$(date +%H:%M:%S)] Sleep completed"
    echo ""
    echo "========================================"
    echo "Results: $test_name"
    echo "========================================"
    echo "  dereg_wait_s:    ${dereg_wait}s"
    echo "  dereg_waited:    ${dereg_waited}"
    echo "  spike_reset_s:   ${spike_reset}s"
    echo "  total_s:         ${sleep_total}s"
    echo ""

    # Validate
    echo "Validation:"
    if (( $(echo "$sleep_delay < 10" | bc -l) )); then
        # Early sleep
        if [ "$dereg_waited" = "True" ]; then
            echo "  ✓ Early sleep waited for dereg as expected"
        else
            echo "  ⚠ Early sleep did NOT wait (dereg completed faster than expected)"
        fi

        if (( $(echo "$spike_reset < 2" | bc -l) )); then
            echo "  ✓ spike_reset FAST (<2s) after dereg"
            echo ""
            echo "  🎉 Early sleep test PASSED!"
        else
            echo "  ✗ spike_reset SLOW (${spike_reset}s) despite waiting"
        fi
    else
        # Late sleep
        if [ "$dereg_waited" = "False" ]; then
            echo "  ✓ Late sleep did NOT wait (dereg already complete)"
        else
            echo "  ⚠ Late sleep still waited (dereg took longer than expected)"
        fi

        if (( $(echo "$spike_reset < 2" | bc -l) )); then
            echo "  ✓ spike_reset FAST (<2s) without waiting"
            echo ""
            echo "  🎉 Late sleep test PASSED!"
        else
            echo "  ✗ spike_reset SLOW (${spike_reset}s) despite 30s+ delay"
        fi
    fi
    echo ""
}

# Run tests
echo "========================================"
echo "Running Hypothesis Tests"
echo "========================================"
echo ""

# Test 1: Early sleep (5s delay)
run_test 5 "Early Sleep (5s after wake)"

# Wait before next test
echo "Waiting 10s before next test..."
sleep 10

# Test 2: Late sleep (35s delay)
run_test 35 "Late Sleep (35s after wake)"

# Final summary
echo ""
echo "========================================"
echo "Hypothesis Test Complete"
echo "========================================"
echo ""
echo "Hypothesis: 'With proper deregistration timing, /sleep achieves ~0.2-1s latency'"
echo ""
echo "Results:"
echo "  - Early sleep: Should show dereg_waited=True, spike_reset < 2s"
echo "  - Late sleep:  Should show dereg_waited=False, spike_reset < 2s"
echo ""
echo "If BOTH tests show spike_reset < 2s:"
echo "  ✅ Hypothesis CONFIRMED!"
echo ""
echo "Logs:"
echo "  Server:   server_tp8.log"
echo "  Receiver: receiver_tp8.log"
echo "========================================"
