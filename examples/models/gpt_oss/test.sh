#!/bin/bash
# Test script for gpt-oss-20b on Trainium
# Usage: bash test.sh

set -e

echo "=========================================="
echo "gpt-oss-20b Test Script"
echo "=========================================="

# Step 1: Clean compilation cache
echo ""
echo "[1/3] Cleaning compilation cache..."
rm -rf build/ 2>/dev/null || true
echo "✓ Cache cleaned"

# Step 2: Check and prepare weights
echo ""
echo "[2/3] Checking weights..."

MODEL="${MODEL:-openai/gpt-oss-20b}"
WEIGHTS_PATH="${WEIGHTS_PATH:-./tmp_gpt-oss-20b}"
TP_DEGREE="${TP_DEGREE:-4}" # Tensor parallelism

if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Converting (dequantizing MXFP4 to bf16)..."
    python tensor_preparation.py --model-name "$MODEL" --world-size "$TP_DEGREE" --output-dir="$WEIGHTS_PATH"
    echo "✓ Weights prepared"
else
    echo "✓ Weights found at $WEIGHTS_PATH"
fi

# Step 3: Run example
echo ""
echo "[3/3] Running gpt-oss inference..."
echo "=========================================="

# Decode drives NRT's explicit async execution queue directly (see the
# TKG_MAX_INFLIGHT window in gpt_oss.py) to overlap host dispatch with device
# compute, so no runtime async env var is needed.
torchrun --nproc-per-node "$TP_DEGREE" gpt_oss.py -n 500 --checkpoint "$WEIGHTS_PATH" --model "$MODEL"

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
