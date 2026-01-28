#!/bin/bash
# Test script for Qwen3-Embedding on Trainium
# Usage: bash test.sh [0.6b|8b]

set -e

MODEL_SIZE="${1:-0.6b}"

echo "=========================================="
echo "Qwen3 Embedding Test Script"
echo "Model: $MODEL_SIZE"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf build/ 2>/dev/null || true
echo "✓ Cache cleaned"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

if [ "$MODEL_SIZE" = "8b" ]; then
    WEIGHTS_PATH="tmp_qwen3_weights_8b/qwen3_weights.safetensors"
else
    WEIGHTS_PATH="tmp_qwen3_weights/qwen3_weights.safetensors"
fi

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python prepare_weights.py --model-size "$MODEL_SIZE"
    echo "✓ Weights prepared"
else
    echo "✓ Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run example
echo ""
echo "[3/3] Running retrieval example..."
echo "=========================================="
python example_retrieval.py --model-size "$MODEL_SIZE" --compare

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
