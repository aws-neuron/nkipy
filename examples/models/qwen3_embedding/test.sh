#!/bin/bash
# Test script for Qwen3-Embedding on Trainium
# Usage: bash test.sh [--model-size 0.6b|8b] [--profile [--no-scalene]]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_SIZE="0.6b"
PROFILE=""
NO_SCALENE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --profile) PROFILE=1; shift ;;
        --no-scalene) NO_SCALENE=1; shift ;;
        *) shift ;;
    esac
done

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

# Ensure NRT inspect is disabled (conflicts with SystemTraceSession)
unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_OUTPUT_DIR 2>/dev/null || true

# Step 3: Run
echo ""
if [ -n "$PROFILE" ]; then
    echo "[3/3] Running profiled inference..."
    echo "=========================================="

    PROFILE_ARGS="--model-size $MODEL_SIZE --num-iterations 100"

    if [ -z "$NO_SCALENE" ]; then
        # Scalene CPU + device tracing
        cd "$SCRIPT_DIR"
        scalene run --off --profile-all --profile-only model.py --cpu-only \
            -o "$SCRIPT_DIR/scalene_profile.json" \
            "$SCRIPT_DIR/profile.py" --- $PROFILE_ARGS

        # scalene_profile.json may land in the model dir due to chdir
        if [ -f "$SCRIPT_DIR/scalene_profile.json" ]; then
            cd "$REPO_DIR"
            uv run python -m nkipy.tools.profiler \
                "$SCRIPT_DIR/scalene_profile.json" \
                "$SCRIPT_DIR/kernel_profile.json" \
                "$SCRIPT_DIR/merged_profile.json"
        fi
        echo ""
        echo "View: scalene view $SCRIPT_DIR/merged_profile.json"
    else
        # Device tracing only
        cd "$SCRIPT_DIR"
        python profile.py --no-scalene $PROFILE_ARGS

        cd "$REPO_DIR"
        uv run python -m nkipy.tools.profiler --kernel-only \
            "$SCRIPT_DIR/kernel_profile.json" \
            "$SCRIPT_DIR/merged_profile.json"
        echo ""
        echo "View: scalene view $SCRIPT_DIR/merged_profile.json"
    fi
else
    echo "[3/3] Running retrieval example..."
    echo "=========================================="
    python example_retrieval.py --model-size "$MODEL_SIZE" --compare
fi

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
