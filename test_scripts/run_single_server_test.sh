#!/usr/bin/env bash
# Start single server and test sleep latency without P2P
set -euo pipefail

WEIGHTS=~/zhuangw/nkipy/examples/models/qwen3/tmp_Qwen3-30b-a3b_TP32

echo "========================================"
echo "Single-Server Sleep Test"
echo "========================================"
echo "Starting server in sleeping mode, then testing wake/sleep cycle"
echo ""

# Check checkpoint
if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: Checkpoint not found at $WEIGHTS"
    exit 1
fi

# Start server in sleeping mode
# Note: Server starts sleeping when NKIPY_CHECKPOINT is NOT set
echo "Starting server (port 8000, sleeping mode)..."
export VLLM_PLUGINS=nkipy
export VLLM_USE_V1=1
# Do NOT set NKIPY_CHECKPOINT - server will start in sleeping mode
unset NKIPY_CHECKPOINT
export OMP_NUM_THREADS=1
export NKIPY_SKIP_CTE=1
export VLLM_RPC_TIMEOUT=600000
export NKIPY_PREREGISTER_MRS=0
# Store checkpoint path for later use in /wake_up
CHECKPOINT_PATH=$WEIGHTS

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

# Wait for server
echo "Waiting for server to initialize..."
for i in {1..120}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server failed to start after 120s"
        echo "Last 50 lines of log:"
        tail -50 /tmp/server.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "Running test..."
./test_sleep_no_p2p.sh http://localhost:8000

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo "Check /tmp/server.log for detailed server logs"
