#!/usr/bin/env bash
# Compare sleep latency: immediate vs after serving requests
set -euo pipefail

PEER_URL=${1:-http://172.31.44.131:8000}
ENGINE_URL=${2:-http://localhost:8000}

echo "========================================"
echo "Sleep Latency Comparison Test"
echo "========================================"
echo "Testing hypothesis: Sleep after serving should be faster"
echo ""

# Helper function to restart receiver engine
restart_engine() {
    echo "Restarting receiver engine..."
    fuser -k 8000/tcp 2>/dev/null || true
    pkill -9 -f vllm_plugin.server 2>/dev/null || true
    sleep 3

    cd /home/ubuntu/vllm-nkipy/nkipy
    bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver_test.log 2>&1 &

    echo "Waiting for engine to start..."
    for i in {1..60}; do
        if grep -q "Application startup complete" /tmp/receiver_test.log 2>/dev/null; then
            echo "✓ Engine started"
            sleep 2
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    echo "ERROR: Engine failed to start within 2 minutes"
    return 1
}

# Test 1: Immediate sleep (current baseline)
echo "========================================"
echo "Test 1: Immediate Sleep (baseline)"
echo "========================================"

restart_engine || exit 1

echo ""
echo "Waking up..."
wake1=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$PEER_URL\"}")

wake1_total=$(echo "$wake1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
echo "Wake up completed: ${wake1_total}s"

echo ""
echo "Sleeping immediately (no inference requests)..."
sleep1=$(curl -s -X POST "$ENGINE_URL/nkipy/sleep" \
    -H "Content-Type: application/json")

sleep1_total=$(echo "$sleep1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
sleep1_clear=$(echo "$sleep1" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('clear_refs_s', 'N/A'))" 2>/dev/null || echo "N/A")

echo "Sleep total: ${sleep1_total}s"
echo "  - clear_refs_s: ${sleep1_clear}s"

# Test 2: Sleep after serving requests
echo ""
echo "========================================"
echo "Test 2: Sleep After Serving (realistic)"
echo "========================================"

restart_engine || exit 1

echo ""
echo "Waking up..."
wake2=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$PEER_URL\"}")

wake2_total=$(echo "$wake2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
echo "Wake up completed: ${wake2_total}s"

echo ""
echo "Sending 30 inference requests..."
success=0
for i in $(seq 1 30); do
    result=$(curl -s -X POST "$ENGINE_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-30B-A3B",
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
            "temperature": 0.0
        }' 2>&1)

    if echo "$result" | grep -q '"choices"'; then
        success=$((success + 1))
        echo -n "."
    else
        echo -n "E"
    fi
    sleep 0.1
done
echo ""
echo "✓ Completed 30 requests ($success successful)"

echo ""
echo "Waiting 3s for GC to settle..."
sleep 3

echo ""
echo "Sleeping after serving..."
sleep2=$(curl -s -X POST "$ENGINE_URL/nkipy/sleep" \
    -H "Content-Type: application/json")

sleep2_total=$(echo "$sleep2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
sleep2_clear=$(echo "$sleep2" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('clear_refs_s', 'N/A'))" 2>/dev/null || echo "N/A")

echo "Sleep total: ${sleep2_total}s"
echo "  - clear_refs_s: ${sleep2_clear}s"

# Summary
echo ""
echo "========================================"
echo "Comparison Results"
echo "========================================"
printf "%-25s %10s %15s\n" "Scenario" "Sleep Total" "clear_refs_s"
printf "%-25s %10s %15s\n" "------------------------" "----------" "-------------"
printf "%-25s %10s %15s\n" "Test 1: Immediate sleep" "${sleep1_total}s" "${sleep1_clear}s"
printf "%-25s %10s %15s\n" "Test 2: After serving" "${sleep2_total}s" "${sleep2_clear}s"
echo ""

# Calculate improvement
if [ "$sleep1_total" != "N/A" ] && [ "$sleep2_total" != "N/A" ]; then
    improvement=$(python3 -c "print(f'{(float('$sleep1_total') - float('$sleep2_total')):.2f}')" 2>/dev/null || echo "N/A")
    echo "Improvement: ${improvement}s faster after serving requests"
    echo ""

    if (( $(echo "$sleep2_total < $sleep1_total" | bc -l 2>/dev/null || echo 0) )); then
        echo "✓ HYPOTHESIS CONFIRMED: Sleep is faster after serving requests"
        echo "  This matches server-side behavior where GC runs naturally"
    else
        echo "✗ HYPOTHESIS NOT CONFIRMED: Sleep latency unchanged"
    fi
else
    echo "Unable to calculate improvement (parse error)"
fi

echo "========================================"
