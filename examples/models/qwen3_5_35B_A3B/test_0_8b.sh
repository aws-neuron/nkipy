#!/bin/bash
# Test script for Qwen3.5-0.8B on Trainium (TP=1, no sharding)
# Usage: bash test_0_8b.sh

set -e

echo "=========================================="
echo "Qwen3.5-0.8B Test Script (TP=1)"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf build_0_8b/ 2>/dev/null || true
echo "Done"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

WEIGHTS_PATH="./qwen3_5_0_8b_shards"

if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python test_0_8b.py --prepare
    echo "Done"
else
    echo "Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run inference + validate against HF
echo ""
echo "[3/3] Running Qwen3.5-0.8B inference (TP=1)..."
echo "=========================================="

export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=-1
export MALLOC_MMAP_THRESHOLD_=131072

torchrun --nproc-per-node 1 test_0_8b.py -- --run -n 32 "who are you"

echo ""
echo "=========================================="
echo "Test passed!"
echo "=========================================="
