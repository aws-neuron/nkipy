#!/usr/bin/env bash
# Simpler test: Verify spike_reset timing WITHOUT P2P transfer
# This tests the conditional wait logic even without RDMA MRs
set -euo pipefail

ENGINE_URL=${1:-http://localhost:8000}

echo "========================================"
echo "Single-Server Sleep Latency Test"
echo "========================================"
echo "Engine URL: $ENGINE_URL"
echo ""
echo "This test verifies /sleep latency when NO P2P transfer occurs."
echo "Expected: spike_reset should be fast (~0.2-1s) since no MRs to wait for."
echo ""

# Test 1: Wake up without peer (no P2P)
echo "Test 1: Wake up without P2P transfer..."
echo "[$(date +%H:%M:%S)] Calling /wake_up (no peer_url)..."

wake_result=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{}")

wake_status=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
    echo "ERROR: Wake up failed with status: $wake_status"
    exit 1
fi

wake_total=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")
echo "[$(date +%H:%M:%S)] Wake up completed in ${wake_total}s"
echo ""

# Test 2: Sleep immediately (no MRs, so no dereg wait needed)
echo "Test 2: Sleep immediately after wake..."
echo "[$(date +%H:%M:%S)] Calling /sleep..."

sleep_result=$(curl -s -X POST "$ENGINE_URL/nkipy/sleep" \
    -H "Content-Type: application/json")

sleep_status=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
if [ "$sleep_status" != "sleeping" ]; then
    echo "ERROR: Sleep failed with status: $sleep_status"
    exit 1
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

# Validate
if [ "$dereg_waited" = "False" ]; then
    echo "✓ PASS: No dereg wait (no P2P transfer occurred)"
else
    echo "⚠ WARNING: Unexpected dereg wait when no P2P occurred"
fi

if (( $(echo "$spike_reset < 2" | bc -l) )); then
    echo "✓ PASS: spike_reset fast (<2s) without MRs"
else
    echo "✗ FAIL: spike_reset slow (${spike_reset}s) - unexpected!"
fi

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Without P2P transfer:"
echo "  - No RDMA MRs registered"
echo "  - No dereg_async() called"
echo "  - dereg_waited should be False"
echo "  - spike_reset should be fast (<2s)"
echo ""
echo "This establishes baseline performance before testing"
echo "the full P2P + dereg timing hypothesis."
echo "========================================"
