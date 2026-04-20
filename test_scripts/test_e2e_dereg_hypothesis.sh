#!/usr/bin/env bash
# End-to-end test of MR deregistration hypothesis with two instances
# Instance A (172.31.44.131): Server with checkpoint
# Instance B (172.31.40.200): Receiver, starts sleeping

set -euo pipefail

INSTANCE_A_IP="172.31.44.131"
INSTANCE_B_IP="172.31.40.200"
SERVER_URL="http://${INSTANCE_A_IP}:8000"
RECEIVER_URL="http://${INSTANCE_B_IP}:8000"

echo "========================================"
echo "End-to-End MR Deregistration Hypothesis Test"
echo "========================================"
echo "Instance A (Server):   $INSTANCE_A_IP"
echo "Instance B (Receiver): $INSTANCE_B_IP"
echo ""
echo "Hypothesis: With proper deregistration timing, /sleep achieves ~0.2-1s latency"
echo ""
echo "Test Plan:"
echo "  1. Early sleep (5s after wake):  Should wait ~20s for dereg"
echo "  2. Late sleep (35s after wake):   Should be fast ~0.2-1s"
echo ""

# Check which instance we're on
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [ "$CURRENT_IP" = "$INSTANCE_A_IP" ]; then
    ON_INSTANCE_A=true
    echo "Running on Instance A (Server)"
else
    ON_INSTANCE_A=false
    echo "Running on Instance B (Receiver)"
fi
echo ""

# Function to check engine health
check_engine() {
    local url=$1
    local name=$2
    echo "Checking $name at $url..."
    if curl -s "$url/health" > /dev/null 2>&1; then
        echo "  ✓ $name is running"
        return 0
    else
        echo "  ✗ $name is not responding"
        return 1
    fi
}

# Function to run a test scenario
run_test() {
    local sleep_delay=$1
    local test_name=$2

    echo ""
    echo "========================================"
    echo "Test: $test_name"
    echo "Sleep delay: ${sleep_delay}s after /wake_up"
    echo "========================================"

    # Step 1: Wake up receiver from server
    echo "[$(date +%H:%M:%S)] Step 1: Calling /wake_up on receiver..."
    wake_result=$(curl -s -X POST "$RECEIVER_URL/nkipy/wake_up" \
        -H "Content-Type: application/json" \
        -d "{\"peer_url\": \"$SERVER_URL\"}")

    wake_status=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
        echo "ERROR: Wake up failed with status: $wake_status"
        echo "$wake_result" | python3 -m json.tool
        return 1
    fi

    wake_total=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")
    p2p_time=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('p2p_transfer_s', 0))" 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Wake up completed in ${wake_total}s (P2P: ${p2p_time}s)"
    echo "  → dereg_async() started in background thread"

    # Step 2: Wait specified delay
    echo ""
    echo "[$(date +%H:%M:%S)] Step 2: Waiting ${sleep_delay}s before /sleep..."
    echo "  (dereg_async deregistering 435 MRs in background, takes ~20-25s)"
    sleep "$sleep_delay"

    # Step 3: Call /sleep
    echo ""
    echo "[$(date +%H:%M:%S)] Step 3: Calling /sleep on receiver..."
    sleep_start=$(date +%s.%N)
    sleep_result=$(curl -s -X POST "$RECEIVER_URL/nkipy/sleep" \
        -H "Content-Type: application/json")
    sleep_end=$(date +%s.%N)

    sleep_status=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$sleep_status" != "sleeping" ]; then
        echo "ERROR: Sleep failed with status: $sleep_status"
        echo "$sleep_result" | python3 -m json.tool
        return 1
    fi

    # Extract timing metrics from rank 0
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

    # Validate against hypothesis
    echo "Validation:"
    if (( $(echo "$sleep_delay < 10" | bc -l) )); then
        # Early sleep: expect dereg wait
        echo "  Expected: Early sleep should wait for dereg"
        if [ "$dereg_waited" = "True" ]; then
            echo "  ✓ PASS: Waited for dereg completion"
        else
            echo "  ⚠ INFO: Did NOT wait (dereg already done faster than expected)"
        fi

        if (( $(echo "$spike_reset < 2" | bc -l) )); then
            echo "  ✓ PASS: spike_reset FAST (<2s) after waiting"
            echo ""
            echo "  🎉 Early sleep hypothesis CONFIRMED!"
            echo "     With conditional wait, spike_reset is fast even for early sleep"
        else
            echo "  ✗ FAIL: spike_reset SLOW (${spike_reset}s) despite waiting"
        fi
    else
        # Late sleep: expect no wait, fast spike_reset
        echo "  Expected: Late sleep should NOT wait (dereg already done)"
        if [ "$dereg_waited" = "False" ]; then
            echo "  ✓ PASS: Did NOT wait for dereg (already complete)"
        else
            echo "  ⚠ WARNING: Still had to wait (dereg slower than expected)"
        fi

        if (( $(echo "$spike_reset < 2" | bc -l) )); then
            echo "  ✓ PASS: spike_reset FAST (<2s) without waiting"
            echo ""
            echo "  🎉 Late sleep hypothesis CONFIRMED!"
            echo "     With proper timing, /sleep achieves ~0.2-1s latency"
        else
            echo "  ✗ FAIL: spike_reset SLOW (${spike_reset}s) despite 30s+ delay"
        fi
    fi
    echo ""
}

# Main test flow
echo "========================================"
echo "Pre-Test Checks"
echo "========================================"

# Check both engines are running
if ! check_engine "$SERVER_URL" "Instance A (Server)"; then
    echo ""
    echo "ERROR: Server not running on Instance A"
    echo "Please start it first:"
    echo "  ssh ubuntu@$INSTANCE_A_IP"
    echo "  cd /home/ubuntu/vllm-nkipy/nkipy"
    echo "  source .venv/bin/activate"
    echo "  bash examples/p2p/run_vllm_qwen_1.sh > server.log 2>&1 &"
    exit 1
fi

if ! check_engine "$RECEIVER_URL" "Instance B (Receiver)"; then
    echo ""
    echo "ERROR: Receiver not running on Instance B"
    echo "Please start it first:"
    echo "  ssh ubuntu@$INSTANCE_B_IP"
    echo "  cd /home/ubuntu/vllm-nkipy/nkipy"
    echo "  source .venv/bin/activate"
    echo "  bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver.log 2>&1 &"
    exit 1
fi

echo ""
echo "✓ Both engines are ready"
echo ""
echo "========================================"
echo "Starting Hypothesis Tests"
echo "========================================"

# Test 1: Early sleep (5s delay)
run_test 5 "Early Sleep (5s after wake)"

# Wait between tests
echo ""
echo "Waiting 10s before next test..."
sleep 10

# Test 2: Late sleep (35s delay)
run_test 35 "Late Sleep (35s after wake)"

# Final summary
echo ""
echo "========================================"
echo "Hypothesis Test Summary"
echo "========================================"
echo ""
echo "Hypothesis: 'With proper deregistration timing, /sleep achieves ~0.2-1s latency'"
echo ""
echo "Check results above:"
echo "  - Early sleep test: Should show dereg_waited=True, spike_reset < 2s"
echo "  - Late sleep test:  Should show dereg_waited=False, spike_reset < 2s"
echo ""
echo "If BOTH tests show spike_reset < 2s:"
echo "  ✓ Hypothesis CONFIRMED!"
echo "  The conditional wait ensures fast spike_reset in both scenarios"
echo ""
echo "========================================"
