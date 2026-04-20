#!/usr/bin/env bash
# Test MR deregistration via the API endpoints
# This mimics the actual production flow: wake_up (registers MRs) → dereg_async → sleep
set -euo pipefail

ENGINE_URL=${1:-http://localhost:8000}
PEER_URL=${2:-}

echo "========================================"
echo "MR Deregistration → Sleep Test (via API)"
echo "========================================"
echo "Engine URL: $ENGINE_URL"
echo "Peer URL: ${PEER_URL:-none (will load from checkpoint)}"
echo ""
echo "Test flow:"
echo "  1. Wake up engine (registers MRs if PREREGISTER=1)"
echo "  2. If peer provided, dereg_async is called automatically"
echo "  3. Wait for dereg to complete (if applicable)"
echo "  4. Call /sleep and measure spike_reset latency"
echo ""

# Check if engine is responding
if ! curl -s "$ENGINE_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Engine not responding at $ENGINE_URL"
    echo "Please start the engine first with one of:"
    echo "  - NKIPY_PREREGISTER_MRS=0 (on-demand, no MRs)"
    echo "  - NKIPY_PREREGISTER_MRS=1 (pre-register MRs)"
    exit 1
fi

echo "Engine is responding."
echo ""

# Test with different wait times
test_sleep_latency() {
    local wait_time=$1
    local description=$2

    echo "----------------------------------------"
    echo "Test: $description"
    echo "Wait time before /sleep: ${wait_time}s"
    echo "----------------------------------------"

    # Wake up
    echo "[$(date +%H:%M:%S)] Calling /wake_up..."
    if [ -n "$PEER_URL" ]; then
        wake_result=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
            -H "Content-Type: application/json" \
            -d "{\"peer_url\": \"$PEER_URL\"}")
    else
        wake_result=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
            -H "Content-Type: application/json" \
            -d "{}")
    fi

    wake_status=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
        echo "ERROR: Wake up failed with status: $wake_status"
        return 1
    fi

    wake_total=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Wake up completed in ${wake_total}s"

    # Wait specified time
    if [ "$wait_time" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] Waiting ${wait_time}s for dereg to complete..."
        sleep "$wait_time"
    fi

    # Sleep
    echo "[$(date +%H:%M:%S)] Calling /sleep..."
    sleep_result=$(curl -s -X POST "$ENGINE_URL/nkipy/sleep" \
        -H "Content-Type: application/json")

    sleep_status=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    if [ "$sleep_status" != "sleeping" ]; then
        echo "ERROR: Sleep failed with status: $sleep_status"
        echo "$sleep_result" | python3 -m json.tool
        return 1
    fi

    # Extract timing
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

    # Validate
    if (( $(echo "$spike_reset < 2" | bc -l) )); then
        echo "✓ PASS: spike_reset FAST (<2s)"
        if [ "$dereg_waited" = "True" ]; then
            echo "  → Achieved via waiting for dereg completion"
        else
            echo "  → Dereg already complete (or no MRs)"
        fi
    else
        echo "✗ FAIL: spike_reset SLOW (${spike_reset}s)"
        if [ "$dereg_waited" = "True" ]; then
            echo "  → Waited for dereg but still slow"
        else
            echo "  → MRs may still be active"
        fi
    fi
    echo ""
}

# Run test
if [ -n "$PEER_URL" ]; then
    echo "Testing with P2P transfer (MRs will be registered and deregistered)..."
    echo ""
    test_sleep_latency 35 "Late sleep (35s after wake - dereg should be complete)"
else
    echo "Testing without P2P (loading from checkpoint)..."
    echo ""
    test_sleep_latency 0 "Sleep immediately after wake (no P2P, no MRs)"
fi

echo "========================================"
echo "Test Complete"
echo "========================================"
echo ""
echo "Hypothesis verification:"
echo "  If spike_reset < 2s → Hypothesis CONFIRMED"
echo "  'With proper deregistration timing, /sleep achieves ~0.2-1s latency'"
echo "========================================"
