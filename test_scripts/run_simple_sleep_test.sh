#!/usr/bin/env bash
# Simplest test: Start server with checkpoint (awake), then test sleep latency
set -euo pipefail

WEIGHTS=~/models/qwen3/tmp_Qwen3-30b-a3b_TP32

echo "========================================"
echo "Simple Sleep Latency Test"
echo "========================================"
echo "Start server with checkpoint (awake), test /sleep without prior P2P transfer"
echo ""

# Check checkpoint
if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: Checkpoint not found at $WEIGHTS"
    exit 1
fi

# Start server WITH checkpoint (starts awake)
echo "Starting server with checkpoint (awake mode)..."
export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
export NKIPY_CHECKPOINT=$WEIGHTS
export OMP_NUM_THREADS=1
export NKIPY_SKIP_CTE=1
export VLLM_RPC_TIMEOUT=600000

uv run python -m nkipy.vllm_plugin.server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 32 \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8000 \
    > /tmp/server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Cleanup
cleanup() {
    echo ""
    echo "Cleaning up server..."
    kill $SERVER_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to initialize (this takes ~2-3 minutes)..."
for i in {1..300}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "ERROR: Server failed to start after 300s"
        echo "Last 100 lines of log:"
        tail -100 /tmp/server.log
        exit 1
    fi
    sleep 1
    if [ $((i % 30)) -eq 0 ]; then
        echo "  ... still waiting ($i/300s)"
    fi
done

# Give it a moment to stabilize
sleep 2

# Now test /sleep
echo ""
echo "Testing /sleep endpoint..."
echo "[$(date +%H:%M:%S)] Calling /sleep..."

sleep_result=$(curl -s -X POST "http://localhost:8000/nkipy/sleep" \
    -H "Content-Type: application/json")

sleep_status=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")

if [ "$sleep_status" != "sleeping" ]; then
    echo "ERROR: Sleep failed with status: $sleep_status"
    echo "Full response:"
    echo "$sleep_result" | python3 -m json.tool
    exit 1
fi

echo "[$(date +%H:%M:%S)] Sleep completed"
echo ""

# Extract and display timing
dereg_wait=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('dereg_wait_s', 0))" 2>/dev/null || echo "0")
dereg_waited=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('dereg_waited', False))" 2>/dev/null || echo "False")
spike_reset=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('spike_reset_s', 0))" 2>/dev/null || echo "0")
sleep_total=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 0))" 2>/dev/null || echo "0")

# Get breakdown from rank 0
echo "========================================"
echo "Sleep Latency Results (Rank 0)"
echo "========================================"
echo "  dereg_wait_s:    ${dereg_wait}s"
echo "  dereg_waited:    ${dereg_waited}"
echo "  spike_reset_s:   ${spike_reset}s"
echo "  total_s:         ${sleep_total}s"
echo ""

# Validate
echo "Validation:"
if [ "$dereg_waited" = "False" ]; then
    echo "  ✓ No dereg wait (no prior P2P transfer)"
else
    echo "  ⚠ Unexpected: dereg_waited=$dereg_waited"
fi

if (( $(echo "$spike_reset < 2" | bc -l) )); then
    echo "  ✓ spike_reset FAST (<2s)"
    echo ""
    echo "SUCCESS: Baseline confirmed - spike_reset is fast without RDMA MRs"
else
    echo "  ✗ spike_reset SLOW (${spike_reset}s)"
    echo ""
    echo "UNEXPECTED: spike_reset should be fast without RDMA transfers"
    echo "This establishes the baseline before testing P2P scenarios."
fi

echo ""
echo "========================================"
echo "Next Steps"
echo "========================================"
echo "This test establishes baseline /sleep performance without P2P."
echo "To test the full hypothesis, run with a peer instance:"
echo "  1. Start receiver on peer: ./examples/p2p/run_vllm_qwen_1_receiver.sh"
echo "  2. Run timing test: ./test_dereg_timing.sh <peer-url>"
echo "========================================"
