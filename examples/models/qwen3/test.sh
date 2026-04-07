#!/bin/bash
# Test script for Qwen3-30B-A3B on Trainium
# Usage: bash test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Qwen3-30B-A3B Test Script"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf "$SCRIPT_DIR/build/" 2>/dev/null || true
echo "✓ Cache cleaned"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

MODEL_NAME="Qwen3-30b-a3b"
TP_DEGREE=32 # Tensor parallelism
WEIGHTS_PATH="$SCRIPT_DIR/tmp_${MODEL_NAME}_TP${TP_DEGREE}"

if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    (cd "$EXAMPLES_DIR" && python -m models.qwen3.tensor_preparation --model-name Qwen/${MODEL_NAME} --world-size "$TP_DEGREE" --head-dim 128 --output-dir="$WEIGHTS_PATH")
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
(cd "$EXAMPLES_DIR" && torchrun --nproc-per-node "$TP_DEGREE" -m models.qwen3.qwen3 -n 500 --checkpoint "$WEIGHTS_PATH" --model Qwen/${MODEL_NAME})

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
