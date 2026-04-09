#!/usr/bin/env bash
# P2P weight transfer test for TinyLlama via vLLM+NKIPy plugin.
#
# Prerequisites:
#   - Engine A running: bash run_vllm_tinyllama_1.sh   (port 8000, cores 0-7)
#   - Engine B running: bash run_vllm_tinyllama_2.sh   (port 8001, cores 8-15)
#
# Engine B starts sleeping (no checkpoint). This script:
#   1. Wakes Engine B via P2P from Engine A
#   2. Runs a completion on Engine B
#   3. Puts Engine B to sleep
#   4. Wakes Engine B again (second cycle)
#   5. Runs a completion again

set -euo pipefail

ENGINE_A="http://localhost:8000"
ENGINE_B="http://localhost:8001"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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


curl -sf -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}"


curl -sf -X POST "http://localhost:8001/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}"


curl -sf -X POST "http://localhost:8001/nkipy/wake_up" \
  -H "Content-Type: application/json" \
  -d "{\"peer_url\": \"http://localhost:8000\"}"


curl -sf -X POST "http://localhost:8001/nkipy/sleep"