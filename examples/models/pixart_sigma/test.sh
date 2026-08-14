#!/bin/bash
# Smoke test for PixArt-Sigma on Trainium
# Usage: bash test.sh
set -e

echo "=========================================="
echo "PixArt-Sigma Test Script"
echo "=========================================="

MODEL="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
WEIGHTS_PATH="./tmp_pixart_sigma"
SAMPLE_SIZE=128   # 1024px / 8 (VAE downsample); use 64 for 512px

# Step 1: Clean compilation cache
echo ""
echo "[1/4] Cleaning compilation cache..."
rm -rf build/ 2>/dev/null || true
echo "OK cache cleaned"

# Step 2: Prepare weights
echo ""
echo "[2/4] Checking weights..."
if [ ! -d "$WEIGHTS_PATH" ]; then
    echo "Weights not found. Downloading and converting..."
    python tensor_preparation.py --model-name "$MODEL" --output-dir "$WEIGHTS_PATH"
    echo "OK weights prepared"
else
    echo "OK weights found at $WEIGHTS_PATH"
fi

# Step 3: Validate a single forward against the diffusers baseline
echo ""
echo "[3/4] Validating denoiser vs diffusers baseline..."
python evaluate.py --validate --model "$MODEL" --checkpoint "$WEIGHTS_PATH" \
    --sample-size "$SAMPLE_SIZE"

# Step 4: Generate one image end-to-end
echo ""
echo "[4/4] Generating an image..."
echo "=========================================="
python pixart.py "a small robot reading a book in a cozy library, warm light" \
    --model "$MODEL" --checkpoint "$WEIGHTS_PATH" \
    --sample-size "$SAMPLE_SIZE" --steps 20 --output output.png

echo ""
echo "=========================================="
echo "OK Test passed! Image written to output.png"
echo "=========================================="
