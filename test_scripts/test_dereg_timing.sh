#!/usr/bin/env bash
# Test hypothesis: spike_reset() latency depends on RDMA MR deregistration timing
# Expected results:
#   - Early /sleep (0-10s after /wake): ~20s (waits for dereg)
#   - Late /sleep (30s+ after /wake):   ~0.2-1s (dereg already complete)

set -euo pipefail

PEER_URL=${1:-http://172.31.44.131:8000}
ENGINE_URL=${2:-http://localhost:8000}

echo "========================================"
echo "MR Deregistration Timing Test"
echo "========================================"
echo "Peer URL: $PEER_URL"
echo "Engine URL: $ENGINE_URL"
echo ""
echo "Hypothesis: /sleep latency depends on dereg completion"
echo "  - Early sleep (<10s):   waits for dereg (~20s total)"
echo "  - Late sleep (>30s):    dereg done (~0.2-1s total)"
echo ""

run_test() {
    local sleep_delay=$1
    local test_name=$2

    echo "========================================"
    echo "Test: $test_name"
    echo "Sleep delay: ${sleep_delay}s after /wake_up"
    echo "========================================"

    # Wake up
    echo "[$(date +%H:%M:%S)] Calling /wake_up..."
    wake_result=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
        -H "Content-Type: application/json" \
        -d "{\"peer_url\": \"$PEER_URL\"}")

    wake_status=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
        echo "ERROR: Wake up failed with status: $wake_status"
        return 1
    fi

    wake_total=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Wake up completed in ${wake_total}s"

    # Wait specified delay
    echo "[$(date +%H:%M:%S)] Waiting ${sleep_delay}s before /sleep..."
    sleep "$sleep_delay"

    # Sleep
    echo "[$(date +%H:%M:%S)] Calling /sleep..."
    sleep_start=$(date +%s.%N)
    sleep_result=$(curl -s -X POST "$ENGINE_URL/nkipy/sleep" \
        -H "Content-Type: application/json")
    sleep_end=$(date +%s.%N)

    sleep_status=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$sleep_status" != "sleeping" ]; then
        echo "ERROR: Sleep failed with status: $sleep_status"
        return 1
    fi

    # Extract timing metrics
    dereg_wait=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('dereg_wait_s', 0))" 2>/dev/null || echo "0")
    dereg_waited=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('dereg_waited', False))" 2>/dev/null || echo "False")
    spike_reset=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('spike_reset_s', 0))" 2>/dev/null || echo "0")
    sleep_total=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")

    echo "[$(date +%H:%M:%S)] Sleep completed"
    echo ""
    echo "Results:"
    echo "  dereg_wait_s:    ${dereg_wait}s"
    echo "  dereg_waited:    ${dereg_waited}"
    echo "  spike_reset_s:   ${spike_reset}s"
    echo "  total_s:         ${sleep_total}s"
    echo ""

    # Validate against hypothesis
    if (( $(echo "$sleep_delay < 10" | bc -l) )); then
        # Early sleep: expect dereg wait
        if [ "$dereg_waited" = "True" ]; then
            echo "✓ PASS: Early sleep waited for dereg as expected"
        else
            echo "⚠ WARNING: Early sleep did NOT wait for dereg (dereg may have completed already)"
        fi
        if (( $(echo "$spike_reset < 2" | bc -l) )); then
            echo "✓ PASS: spike_reset fast (<2s) after dereg wait"
        else
            echo "✗ FAIL: spike_reset slow (${spike_reset}s) despite waiting"
        fi
    else
        # Late sleep: expect no wait, fast spike_reset
        if [ "$dereg_waited" = "False" ]; then
            echo "✓ PASS: Late sleep did NOT wait (dereg already done)"
        else
            echo "⚠ WARNING: Late sleep still had to wait for dereg"
        fi
        if (( $(echo "$spike_reset < 2" | bc -l) )); then
            echo "✓ PASS: spike_reset fast (<2s) without waiting"
        else
            echo "✗ FAIL: spike_reset slow (${spike_reset}s) despite 30s+ delay"
        fi
    fi
    echo ""
}

# Test 1: Early sleep (5s delay)
# Expected: dereg_waited=True, spike_reset < 2s after wait
run_test 5 "Early Sleep (5s delay)"

# Wait a moment between tests
echo "Waiting 5s before next test..."
sleep 5
echo ""

# Test 2: Late sleep (35s delay)
# Expected: dereg_waited=False, spike_reset < 2s
run_test 35 "Late Sleep (35s delay)"

echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Both tests should show spike_reset < 2s:"
echo "  - Early sleep: achieves this by waiting for dereg"
echo "  - Late sleep: achieves this because dereg is already done"
echo ""
echo "If both tests pass, the hypothesis is confirmed:"
echo "With proper deregistration timing, /sleep achieves ~0.2-1s latency"
echo "========================================"
