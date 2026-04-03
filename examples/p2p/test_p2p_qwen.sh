#!/usr/bin/env bash
# P2P weight transfer test for Qwen3.
#
# Prerequisites:
#   - Engine A running: bash run_qwen_1.sh   (port 8000, cores 0-7)
#   - Engine B running: bash run_qwen_2.sh   (port 8001, cores 16-23)
#
# Engine B starts sleeping (no checkpoint). This script:
#   1. Wakes Engine B via P2P from Engine A
#   2. Runs a completion on Engine B
#   3. Puts Engine B to sleep
#   4. Wakes Engine B again via P2P (tests the second wake_up cycle)
#   5. Runs a completion on Engine B again

set -euo pipefail

ENGINE_A="http://localhost:8000"
ENGINE_B="http://localhost:8001"

echo "=== Step 1: Wake Engine B (P2P from Engine A) ==="
curl -s -X POST "$ENGINE_B/wake_up" \
  -H "Content-Type: application/json" \
  -d "{\"peer_url\": \"$ENGINE_A\"}"
echo

echo "=== Step 2: Completion on Engine B ==="
curl -s -X POST "$ENGINE_B/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
echo

echo "=== Step 3: Sleep Engine B ==="
curl -s -X POST "$ENGINE_B/sleep"
echo

echo "=== Step 4: Wake Engine B again (P2P, second cycle) ==="
curl -s -X POST "$ENGINE_B/wake_up" \
  -H "Content-Type: application/json" \
  -d "{\"peer_url\": \"$ENGINE_A\"}"
echo

echo "=== Step 5: Completion on Engine B (after second wake) ==="
curl -s -X POST "$ENGINE_B/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
echo

echo "=== PASS ==="
