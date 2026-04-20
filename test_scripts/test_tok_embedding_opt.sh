#!/usr/bin/env bash
# Test tok_embedding P2P transfer optimization
set -euo pipefail

PEER_URL=${1:-http://172.31.44.131:8000}
ENGINE_URL=${2:-http://localhost:8000}

echo "=========================================="
echo "Testing tok_embedding P2P Optimization"
echo "=========================================="
echo "Peer: $PEER_URL"
echo "Engine: $ENGINE_URL"
echo ""

# Baseline: Current wake_up latency
echo "==> Baseline: Wake up and measure latency..."
wake_result=$(curl -s -X POST "$ENGINE_URL/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$PEER_URL\"}")

echo "$wake_result" | python3 -m json.tool

# Extract key metrics
total_s=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('total_s', 'N/A'))" 2>/dev/null || echo "N/A")
tok_embedding_s=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('tok_embedding_s', 'N/A'))" 2>/dev/null || echo "N/A")
p2p_transfer_s=$(echo "$wake_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('latency', {}).get('p2p_transfer_s', 'N/A'))" 2>/dev/null || echo "N/A")

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo "Total wake_up:       ${total_s}s"
echo "P2P transfer:        ${p2p_transfer_s}s"
echo "tok_embedding:       ${tok_embedding_s}s"
echo ""
echo "Expected after optimization:"
echo "  - tok_embedding_s should drop from ~2.5s to ~0.2s"
echo "  - Total should improve from ~40s to ~38s"
echo "=========================================="

# Test inference to verify correctness
echo ""
echo "==> Testing inference to verify correctness..."
result=$(curl -s -X POST "$ENGINE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-30B-A3B",
        "prompt": "Hello, my name is",
        "max_tokens": 5,
        "temperature": 0.0
    }')

if echo "$result" | grep -q '"choices"'; then
    generated_text=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERROR")
    echo "✓ Inference successful!"
    echo "  Generated: $generated_text"
else
    echo "✗ Inference FAILED!"
    echo "$result"
    exit 1
fi

echo ""
echo "Test complete!"
