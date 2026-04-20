#!/usr/bin/env bash
# Automated test script for Instance B (Receiver)
# Run this on Instance B (172.31.40.200)
set -euo pipefail

INSTANCE_A_URL="http://172.31.44.131:8000"
LOG_FILE="/tmp/receiver_b.log"

echo "======================================"
echo "P2P Transfer Test - Instance B"
echo "======================================"

# Step 1: Kill existing engines
echo ""
echo "[1/6] Cleaning up existing engines..."
fuser -k 8000/tcp 2>&1 || true
pkill -9 -f vllm_plugin.server 2>&1 || true
sleep 3

# Step 2: Verify Instance A is ready
echo ""
echo "[2/6] Checking Instance A status..."
if curl -s -f "$INSTANCE_A_URL/nkipy/health" > /dev/null; then
    echo "✓ Instance A is ready"
    curl -s "$INSTANCE_A_URL/nkipy/health" | python3 -m json.tool
else
    echo "✗ Instance A is not responding at $INSTANCE_A_URL"
    echo "Please start Instance A first!"
    exit 1
fi

# Step 3: Start Engine B
echo ""
echo "[3/6] Starting Engine B (receiver, no weights)..."
cd /home/ubuntu/vllm-nkipy/nkipy
source .venv/bin/activate

nohup bash examples/p2p/run_vllm_qwen_1_receiver.sh > "$LOG_FILE" 2>&1 &
ENGINE_PID=$!
echo "Engine B started with PID: $ENGINE_PID"

# Step 4: Wait for startup
echo ""
echo "[4/6] Waiting for Engine B startup (this may take 2-3 minutes)..."
timeout 300 bash -c "
    while ! grep -q 'Application startup complete' $LOG_FILE 2>/dev/null; do
        sleep 5
        echo -n '.'
    done
" || {
    echo ""
    echo "✗ Engine B startup timed out or failed"
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE"
    exit 1
}

echo ""
echo "✓ Engine B is ready"

# Verify health
sleep 2
if curl -s http://localhost:8000/nkipy/health | python3 -m json.tool | grep -q "sleeping.*true"; then
    echo "✓ Engine B is in sleep mode (as expected)"
else
    echo "✗ Engine B health check failed"
    curl -s http://localhost:8000/nkipy/health
    exit 1
fi

# Step 5: Test /wake_up
echo ""
echo "[5/6] Testing /wake_up..."
echo "Sending wake_up request to Instance A ($INSTANCE_A_URL)..."

START_TIME=$(date +%s)
curl -X POST http://localhost:8000/nkipy/wake_up \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$INSTANCE_A_URL\"}" \
    2>/dev/null | python3 -m json.tool | tee /tmp/wake_up_result.json
END_TIME=$(date +%s)

WAKE_ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "✓ /wake_up completed in ${WAKE_ELAPSED}s (wall time)"

# Extract latency from response
WAKE_TOTAL=$(python3 -c "import json; print(json.load(open('/tmp/wake_up_result.json'))['latency']['total_s'])" 2>/dev/null || echo "N/A")
echo "  Total latency reported: ${WAKE_TOTAL}s"

# Step 6: Test /sleep
echo ""
echo "[6/6] Testing /sleep..."

START_TIME=$(date +%s)
curl -X POST http://localhost:8000/nkipy/sleep \
    -H "Content-Type: application/json" \
    2>/dev/null | python3 -m json.tool | tee /tmp/sleep_result.json
END_TIME=$(date +%s)

SLEEP_ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "✓ /sleep completed in ${SLEEP_ELAPSED}s (wall time)"

# Extract latency from response
SLEEP_TOTAL=$(python3 -c "import json; print(json.load(open('/tmp/sleep_result.json'))['latency']['total_s'])" 2>/dev/null || echo "N/A")
echo "  Total latency reported: ${SLEEP_TOTAL}s"

# Summary
echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "Wake-up latency: ${WAKE_TOTAL}s"
echo "Sleep latency:   ${SLEEP_TOTAL}s"
echo ""
echo "Detailed results saved to:"
echo "  - /tmp/wake_up_result.json"
echo "  - /tmp/sleep_result.json"
echo "  - $LOG_FILE"
echo ""
echo "To check Instance A push logs:"
echo "  ssh ubuntu@172.31.44.131 'tail -100 /tmp/server_a.log | grep pushed'"
echo ""
echo "✓ All tests completed!"
