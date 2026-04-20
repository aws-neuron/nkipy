#!/usr/bin/env bash
# Realistic test: wake_up -> serve requests -> sleep
# This mimics production usage where engine serves traffic before sleeping
set -euo pipefail

PEER_URL=${1:-http://172.31.44.131:8000}
ENGINE_URL=${2:-http://localhost:8000}
NUM_REQUESTS=${3:-20}

echo "========================================"
echo "Realistic P2P Test: Wake -> Serve -> Sleep"
echo "========================================"
echo "Peer URL: $PEER_URL"
echo "Engine URL: $ENGINE_URL"
echo "Inference requests: $NUM_REQUESTS"
echo ""

# Step 1: Wake up and receive weights
echo "==> Step 1: Waking up engine (receiving weights from peer)..."
wake_result=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$PEER_URL\"}")

echo "$wake_result" | python3 -m json.tool
wake_status=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")

if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
    echo "ERROR: Wake up failed with status: $wake_status"
    exit 1
fi

wake_total=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
echo "✓ Wake up completed in ${wake_total}s"
echo ""

# Step 2: Wait a moment for engine to stabilize
echo "==> Step 2: Waiting 2s for engine to stabilize..."
sleep 2
echo ""

# Step 3: Send inference requests to trigger GC cycles
echo "==> Step 3: Sending $NUM_REQUESTS inference requests..."
echo "This allows Python GC to run naturally, mimicking production usage"
echo ""

success_count=0
failed_count=0

for i in $(seq 1 $NUM_REQUESTS); do
    result=$(curl -s -X POST "$ENGINE_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-30B-A3B",
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
            "temperature": 0.0
        }' 2>&1)

    if echo "$result" | grep -q '"choices"'; then
        success_count=$((success_count + 1))
        echo -n "."
    else
        failed_count=$((failed_count + 1))
        echo -n "E"
    fi

    # Brief pause between requests
    sleep 0.1
done

echo ""
echo "✓ Completed $NUM_REQUESTS requests (success: $success_count, failed: $failed_count)"
echo ""

# Step 4: Wait for GC to settle
echo "==> Step 4: Waiting 3s for GC to settle..."
sleep 3
echo ""

# Step 5: Sleep the engine
echo "==> Step 5: Putting engine to sleep..."
sleep_result=$(curl -s -X POST "$ENGINE_URL/nkipy/sleep" \
    -H "Content-Type: application/json")

echo "$sleep_result" | python3 -m json.tool
sleep_total=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
clear_refs=$(echo "$sleep_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('clear_refs_s', 'N/A'))" 2>/dev/null || echo "N/A")

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"
echo "Wake up total:       ${wake_total}s"
echo "Inference requests:  $success_count successful"
echo "Sleep total:         ${sleep_total}s"
echo "  - clear_refs_s:    ${clear_refs}s"
echo ""
echo "Hypothesis: After serving requests, sleep latency should match"
echo "            server-side (~2s) instead of immediate sleep (~14s)"
echo "========================================"
