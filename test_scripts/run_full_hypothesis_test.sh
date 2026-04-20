#!/usr/bin/env bash
# Complete setup and test runner for MR deregistration hypothesis
set -euo pipefail

INSTANCE_A_IP="172.31.44.131"
INSTANCE_B_IP="172.31.40.200"

echo "========================================"
echo "Full MR Deregistration Hypothesis Test"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Check if engines are running on both instances"
echo "  2. Start engines if needed (in background)"
echo "  3. Run the hypothesis test"
echo ""

# Detect which instance we're on
CURRENT_IP=$(hostname -I | awk '{print $1}')
if [ "$CURRENT_IP" = "$INSTANCE_A_IP" ]; then
    ON_INSTANCE="A"
    echo "Running on Instance A (Server)"
elif [ "$CURRENT_IP" = "$INSTANCE_B_IP" ]; then
    ON_INSTANCE="B"
    echo "Running on Instance B (Receiver)"
else
    echo "ERROR: Unknown instance IP: $CURRENT_IP"
    echo "Expected: $INSTANCE_A_IP (A) or $INSTANCE_B_IP (B)"
    exit 1
fi
echo ""

# Function to check if engine is running
is_engine_running() {
    curl -s http://localhost:8000/health > /dev/null 2>&1
}

# Function to start engine
start_engine() {
    local script=$1
    local log=$2

    echo "Starting engine: $script"
    source .venv/bin/activate
    bash "$script" > "$log" 2>&1 &
    local pid=$!
    echo "  Engine PID: $pid"
    echo "  Log file: $log"

    # Wait for engine to be ready
    echo "  Waiting for engine to initialize..."
    for i in {1..300}; do
        if is_engine_running; then
            echo "  ✓ Engine ready!"
            return 0
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "    ... still waiting ($i/300s)"
        fi
        sleep 1
    done

    echo "  ✗ Engine failed to start after 300s"
    echo "  Last 50 lines of log:"
    tail -50 "$log"
    return 1
}

# Step 1: Check/start local engine
echo "========================================"
echo "Step 1: Local Engine Setup"
echo "========================================"

if is_engine_running; then
    echo "✓ Local engine is already running"
else
    echo "Local engine not running, starting it..."

    if [ "$ON_INSTANCE" = "A" ]; then
        start_engine "examples/p2p/run_vllm_qwen_1.sh" "server.log"
    else
        start_engine "examples/p2p/run_vllm_qwen_1_receiver.sh" "receiver.log"
    fi
fi

echo ""

# Step 2: Check remote engine
echo "========================================"
echo "Step 2: Remote Engine Check"
echo "========================================"

if [ "$ON_INSTANCE" = "A" ]; then
    REMOTE_IP=$INSTANCE_B_IP
    REMOTE_NAME="Instance B (Receiver)"
else
    REMOTE_IP=$INSTANCE_A_IP
    REMOTE_NAME="Instance A (Server)"
fi

echo "Checking $REMOTE_NAME at http://$REMOTE_IP:8000..."
if curl -s "http://$REMOTE_IP:8000/health" > /dev/null 2>&1; then
    echo "✓ Remote engine is running"
else
    echo "✗ Remote engine is not responding"
    echo ""
    echo "Please start the remote engine manually:"
    echo "  ssh ubuntu@$REMOTE_IP"
    echo "  cd /home/ubuntu/vllm-nkipy/nkipy"
    if [ "$ON_INSTANCE" = "A" ]; then
        echo "  bash examples/p2p/run_vllm_qwen_1_receiver.sh > receiver.log 2>&1 &"
    else
        echo "  bash examples/p2p/run_vllm_qwen_1.sh > server.log 2>&1 &"
    fi
    echo ""
    read -p "Press Enter when remote engine is ready, or Ctrl-C to cancel..."
fi

echo ""

# Step 3: Run the hypothesis test
echo "========================================"
echo "Step 3: Running Hypothesis Test"
echo "========================================"
echo ""

if [ "$ON_INSTANCE" = "A" ]; then
    echo "NOTE: Test must run from Instance B (Receiver) to measure its /sleep latency"
    echo ""
    echo "Please run this command on Instance B:"
    echo "  ssh ubuntu@$INSTANCE_B_IP"
    echo "  cd /home/ubuntu/vllm-nkipy/nkipy"
    echo "  ./test_e2e_dereg_hypothesis.sh"
    exit 0
fi

# Run the test (only on Instance B)
./test_e2e_dereg_hypothesis.sh

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo ""
echo "Check the output above for results."
echo "Log files:"
echo "  - Instance A: /home/ubuntu/vllm-nkipy/nkipy/server.log"
echo "  - Instance B: /home/ubuntu/vllm-nkipy/nkipy/receiver.log"
