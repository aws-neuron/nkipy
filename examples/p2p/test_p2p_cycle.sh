#!/usr/bin/env bash
# P2P weight transfer cycle test for vLLM+NKIPy plugin.
#
# Runs: wake B → infer B → sleep B → wake B → infer B
#
# Usage:
#   test_p2p_cycle.sh --model MODEL [--engine-a URL] [--engine-b URL]
#
# Examples:
#   ./test_p2p_cycle.sh --model Qwen/Qwen3-30B-A3B
#   ./test_p2p_cycle.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   ./test_p2p_cycle.sh --model Qwen/Qwen3-30B-A3B \
#       --engine-a http://172.31.44.131:8000 --engine-b http://172.31.40.200:8000
set -euo pipefail

ENGINE_A="http://localhost:8000"
ENGINE_B="http://localhost:8001"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2;;
        --engine-a) ENGINE_A="$2"; shift 2;;
        --engine-b) ENGINE_B="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "${MODEL:-}" ]]; then
    echo "Usage: $0 --model MODEL [--engine-a URL] [--engine-b URL]"
    exit 1
fi

echo "=== Step 1: Wake Engine B (P2P from Engine A) ==="
curl -sf -X POST "$ENGINE_B/nkipy/wake_up" \
  -H "Content-Type: application/json" \
  -d "{\"peer_url\": \"$ENGINE_A\"}"
echo

echo "=== Step 2: Completion on Engine B ==="
curl -sf -X POST "$ENGINE_B/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}"
echo

echo "=== Step 3: Sleep Engine B ==="
curl -sf -X POST "$ENGINE_B/nkipy/sleep"
echo

echo "=== Step 4: Wake Engine B again (second cycle) ==="
curl -sf -X POST "$ENGINE_B/nkipy/wake_up" \
  -H "Content-Type: application/json" \
  -d "{\"peer_url\": \"$ENGINE_A\"}"
echo

echo "=== Step 5: Completion on Engine B (after second wake) ==="
curl -sf -X POST "$ENGINE_B/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}"
echo

echo "=== PASS ==="
