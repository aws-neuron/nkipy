#!/bin/bash
# Test script for Llama3 on Trainium
# Usage: bash test.sh

set -e

echo "=========================================="
echo "Llama3 Test Script"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf build/ 2>/dev/null || true
echo "✓ Cache cleaned"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

MODEL_NAME="TinyLlama-1.1B-Chat-v1.0"
WEIGHTS_PATH="./tmp_tinyllama_TP8"
TP_DEGREE=8 # Tensor parallelism

if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python tensor_preparation.py --model-name TinyLlama/${MODEL_NAME} --world-size "$TP_DEGREE" --head-dim 64 --output-dir="$WEIGHTS_PATH"
    echo "✓ Weights prepared"
else
    echo "✓ Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run example
echo ""
echo "[3/3] Running Llama3 inference..."
echo "=========================================="

# Enable async to improve performance
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
torchrun --nproc-per-node "$TP_DEGREE" llama3.py -n 500 --checkpoint "$WEIGHTS_PATH" --model TinyLlama/${MODEL_NAME}

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
