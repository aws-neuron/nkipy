#!/usr/bin/env bash
# Script to run on Instance B - starts receiver and runs hypothesis test
set -euo pipefail

echo "========================================"
echo "Remote Hypothesis Test Runner"
echo "Instance B (172.31.40.200)"
echo "========================================"
echo ""

cd /home/ubuntu/vllm-nkipy/nkipy

# Check if receiver is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Receiver already running"
else
    echo "Starting receiver..."
    source .venv/bin/activate
    bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver_b.log 2>&1 &
    RECEIVER_PID=$!
    echo "  Receiver PID: $RECEIVER_PID"

    # Wait for receiver to be ready
    echo "  Waiting for receiver to initialize (2-3 minutes)..."
    for i in {1..300}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  ✓ Receiver ready!"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "  ✗ Receiver failed to start after 300s"
            echo "  Last 50 lines of log:"
            tail -50 receiver_b.log
            exit 1
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "    ... still waiting ($i/300s)"
        fi
        sleep 1
    done
fi

echo ""
echo "Running hypothesis test..."
echo ""

# Run the test
./test_e2e_dereg_hypothesis.sh

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo "Receiver log: /home/ubuntu/vllm-nkipy/nkipy/receiver_b.log"
