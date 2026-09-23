#!/bin/bash
# Test script for Qwen3-30B-A3B on Trainium
# Usage: bash test.sh [--profile [--no-scalene]]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PROFILE=""
NO_SCALENE=""
for arg in "$@"; do
    case $arg in
        --profile) PROFILE=1 ;;
        --no-scalene) NO_SCALENE=1 ;;
    esac
done

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

# Ensure NRT inspect is disabled (conflicts with SystemTraceSession)
unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_OUTPUT_DIR 2>/dev/null || true

# Step 3: Run
echo ""
if [ -n "$PROFILE" ]; then
    echo "[3/3] Running profiled inference (TP=$TP_DEGREE)..."
    echo "=========================================="

    if [ -z "$NO_SCALENE" ]; then
        # Scalene on rank 0 + device tracing
        scalene run --off --profile-all --cpu-only \
            -o "$SCRIPT_DIR/scalene_profile.json" --- \
            "$SCRIPT_DIR/launch_distributed_worker.py" \
            --nproc-per-node "$TP_DEGREE" \
            "$SCRIPT_DIR/profile.py" \
            --output-dir "$SCRIPT_DIR" \
            --checkpoint "$WEIGHTS_PATH" --model Qwen/Qwen3-30B-A3B || true

        # Merge CPU + device profiles
        cd "$REPO_DIR"
        if [ -f "$SCRIPT_DIR/scalene_profile.json" ]; then
            uv run python -m nkipy.tools.profiler \
                "$SCRIPT_DIR/scalene_profile.json" \
                "$SCRIPT_DIR/kernel_profile.json" \
                "$SCRIPT_DIR/merged_profile.json"
        else
            uv run python -m nkipy.tools.profiler --kernel-only \
                "$SCRIPT_DIR/kernel_profile.json" \
                "$SCRIPT_DIR/merged_profile.json"
        fi
        echo ""
        echo "View: scalene view $SCRIPT_DIR/merged_profile.json"
    else
        # Device tracing only, no scalene
        export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
        torchrun --nproc-per-node "$TP_DEGREE" \
            "$SCRIPT_DIR/profile.py" \
            --output-dir "$SCRIPT_DIR" --no-scalene \
            --checkpoint "$WEIGHTS_PATH" --model Qwen/Qwen3-30B-A3B

        cd "$REPO_DIR"
        uv run python -m nkipy.tools.profiler --kernel-only \
            "$SCRIPT_DIR/kernel_profile.json" \
            "$SCRIPT_DIR/merged_profile.json"
        echo ""
        echo "View: scalene view $SCRIPT_DIR/merged_profile.json"
    fi
else
    echo "[3/3] Running Qwen3 inference..."
    echo "=========================================="

    # Enable async to improve performance
    export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16
    torchrun --nproc-per-node "$TP_DEGREE" qwen3.py -n 500 --checkpoint "$WEIGHTS_PATH" --model Qwen/Qwen3-30B-A3B
fi

echo ""
echo "=========================================="
echo "✓ Test passed!"
echo "=========================================="
