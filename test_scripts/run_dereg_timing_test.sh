#!/usr/bin/env bash
# Full end-to-end test of MR deregistration timing hypothesis
set -euo pipefail

WEIGHTS=~/models/qwen3/tmp_Qwen3-30b-a3b_TP32

echo "========================================"
echo "MR Deregistration Timing Hypothesis Test"
echo "========================================"
echo "Testing: /sleep latency depends on RDMA MR dereg completion"
echo ""
echo "Expected results:"
echo "  - Early /sleep (<10s): waits ~20s for dereg"
echo "  - Late /sleep (>30s):  fast ~0.2-1s (dereg done)"
echo ""

# Check if checkpoint exists
if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: Checkpoint not found at $WEIGHTS"
    exit 1
fi

# Step 1: Start receiver (engine that will provide weights)
echo "Step 1: Starting receiver server (port 8001)..."
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
    --port 8001 \
    > /tmp/receiver.log 2>&1 &

RECEIVER_PID=$!
echo "Receiver PID: $RECEIVER_PID"

# Wait for receiver to be ready
echo "Waiting for receiver to initialize..."
for i in {1..60}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "Receiver ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Receiver failed to start after 60s"
        kill $RECEIVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Step 2: Start main server (will wake up from receiver)
echo ""
echo "Step 2: Starting main server (port 8000)..."
# Use NKIPY_PREREGISTER_MRS=0 to enable on-demand registration and dereg_async
export NKIPY_PREREGISTER_MRS=0

uv run python -m nkipy.vllm_plugin.server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 32 \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --enforce-eager \
    --dtype bfloat16 \
    --port 8000 \
    --sleeping \
    > /tmp/main.log 2>&1 &

MAIN_PID=$!
echo "Main server PID: $MAIN_PID"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up servers..."
    kill $MAIN_PID 2>/dev/null || true
    kill $RECEIVER_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# Wait for main server to be ready (sleeping)
echo "Waiting for main server to initialize (sleeping mode)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Main server ready (sleeping)!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Main server failed to start after 60s"
        exit 1
    fi
    sleep 1
done

# Step 3: Run the timing tests
echo ""
echo "Step 3: Running dereg timing tests..."
echo ""

./test_dereg_timing.sh http://localhost:8001

# Step 4: Show logs if there were errors
echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo ""
echo "Check /tmp/main.log and /tmp/receiver.log for detailed server logs"
echo ""
echo "Key metrics to verify:"
echo "  1. Early sleep: dereg_waited=True, spike_reset < 2s"
echo "  2. Late sleep:  dereg_waited=False, spike_reset < 2s"
echo ""
echo "If both conditions met, hypothesis CONFIRMED:"
echo "With proper deregistration timing, /sleep achieves ~0.2-1s latency"
echo "========================================"
