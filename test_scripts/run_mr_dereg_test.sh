#!/usr/bin/env bash
# Test MR registration → deregistration → spike_reset cycle
# This verifies the hypothesis on a SINGLE instance without P2P transfer
set -euo pipefail

WEIGHTS=~/models/qwen3/tmp_Qwen3-30b-a3b_TP32
TP=32

echo "========================================"
echo "MR Deregistration Cycle Test"
echo "========================================"
echo "Testing: spike_reset latency after MR deregistration"
echo ""
echo "Scenarios to test:"
echo "  1. NKIPY_PREREGISTER_MRS=0 (on-demand, no MRs to dereg)"
echo "  2. NKIPY_PREREGISTER_MRS=1 (pre-register MRs, then dereg)"
echo ""

# Check checkpoint
if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: Checkpoint not found at $WEIGHTS"
    exit 1
fi

run_test() {
    local preregister=$1
    local test_name=$2

    echo "========================================"
    echo "Test: $test_name"
    echo "NKIPY_PREREGISTER_MRS=$preregister"
    echo "========================================"

    export VLLM_PLUGINS=nkipy
    export VLLM_USE_V1=1
    export NKIPY_CHECKPOINT=$WEIGHTS
    export OMP_NUM_THREADS=1
    export NKIPY_SKIP_CTE=1
    export VLLM_RPC_TIMEOUT=600000
    export NKIPY_PREREGISTER_MRS=$preregister

    # Run with torchrun for proper distributed setup
    uv run torchrun \
        --nproc_per_node=$TP \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        ./test_mr_dereg_cycle.py 2>&1 | tee /tmp/mr_dereg_test_${preregister}.log

    echo ""
    echo "Test completed. Log saved to /tmp/mr_dereg_test_${preregister}.log"
    echo ""
}

# Test 1: On-demand registration (no MRs, baseline)
echo "Starting Test 1: On-demand MR registration (baseline)..."
run_test 0 "On-demand registration (no pre-registered MRs)"

echo ""
echo "Waiting 5s before next test..."
sleep 5
echo ""

# Test 2: Pre-registration (MRs registered, then deregistered)
echo "Starting Test 2: Pre-registered MRs + deregistration..."
run_test 1 "Pre-registered MRs with dereg_async()"

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "Check the logs above for results:"
echo ""
echo "Test 1 (PREREGISTER=0): Should show fast spike_reset (<2s)"
echo "  - No MRs registered"
echo "  - Baseline performance"
echo ""
echo "Test 2 (PREREGISTER=1): Should show fast spike_reset (<2s)"
echo "  - 435 MRs registered at init"
echo "  - dereg_async() called and completed"
echo "  - spike_reset after dereg should be fast"
echo ""
echo "If BOTH tests show spike_reset < 2s:"
echo "  ✓ Hypothesis CONFIRMED:"
echo "    'With proper deregistration timing, /sleep achieves ~0.2-1s'"
echo ""
echo "========================================"
