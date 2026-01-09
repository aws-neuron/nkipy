#!/bin/bash
# Test script for qwen3-embedding
# Usage: bash test.sh

set -e  # Exit on error

echo "=========================================="
echo "Qwen3 Embedding Test Script"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf /tmp/build/qwen3_* 2>/dev/null || true
echo "✓ Cache cleaned"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."
WEIGHTS_PATH="tmp_qwen3_weights/qwen3_weights.safetensors"

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python prepare_weights.py
    echo "✓ Weights prepared"
else
    echo "✓ Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run example retrieval test
echo ""
echo "[3/3] Running example retrieval test..."
echo "=========================================="
python example_retrieval.py

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
