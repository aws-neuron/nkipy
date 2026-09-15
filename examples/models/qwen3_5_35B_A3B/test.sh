#!/bin/bash
# Test script for Qwen3.5-35B-A3B on Trainium
# Usage: bash test.sh

set -e

echo "=========================================="
echo "Qwen3.5-35B-A3B Test Script"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf build/ 2>/dev/null || true
echo "Done"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

WEIGHTS_PATH="./qwen3_5_shards"
TP_DEGREE=4

if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python tensor_preparation.py --model-name Qwen/Qwen3.5-35B-A3B --world-size "$TP_DEGREE" --head-dim 256 --output-dir="$WEIGHTS_PATH"
    echo "Done"
else
    echo "Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run inference
echo ""
echo "[3/3] Running Qwen3.5 inference..."
echo "=========================================="

export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=-1
export MALLOC_MMAP_THRESHOLD_=131072
torchrun --nproc-per-node "$TP_DEGREE" qwen3_5.py -n 256 --checkpoint "$WEIGHTS_PATH" --model Qwen/Qwen3.5-35B-A3B "what is capital city of Austria?"

echo ""
echo "=========================================="
echo "Test passed!"
echo "=========================================="
