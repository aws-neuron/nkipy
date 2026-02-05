#!/bin/bash
# Test script for Qwen3-30B-A3B on Trainium
# Usage: bash test.sh

set -e

echo "=========================================="
echo "Qwen3-30B-A3B Test Script"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf build/ 2>/dev/null || true
echo "✓ Cache cleaned"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

WEIGHTS_PATH="./tmp_qwen3-30b-a3b"
TP_DEGREE=4 # Tensor parallelism

if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python tensor_preparation.py --model-name Qwen/Qwen3-30B-A3B --world-size "$TP_DEGREE" --head-dim 128 --output-dir="$WEIGHTS_PATH"
    echo "✓ Weights prepared"
else
    echo "✓ Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run example
echo ""
echo "[3/3] Running Qwen3 inference..."
echo "=========================================="

# Enable async to improve performance
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
# export NEURON_LOGICAL_NC_CONFIG=1
torchrun --nproc-per-node "$TP_DEGREE" qwen3.py -n 500 --checkpoint "$WEIGHTS_PATH" --model Qwen/Qwen3-30B-A3B

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
