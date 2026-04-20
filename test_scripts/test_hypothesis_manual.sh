#!/usr/bin/env bash
# Manual hypothesis test with timing measurements
# Prerequisites: Engine A (8000) and Engine B (8001) already running

set -euo pipefail

ENGINE_A="http://localhost:8000"
ENGINE_B="http://localhost:8001"

echo "========================================"
echo "MR Deregistration Hypothesis Test"
echo "========================================"
echo "Engine A (Server):   $ENGINE_A"
echo "Engine B (Receiver): $ENGINE_B"
echo ""

# Test 1: Early sleep
echo "========================================"
echo "Test 1: Early Sleep (5s after wake)"
echo "========================================"
echo ""

echo "[$(date +%H:%M:%S)] Waking Engine B..."
curl -s -X POST "$ENGINE_B/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$ENGINE_A\"}" | python3 -m json.tool

echo ""
echo "[$(date +%H:%M:%S)] Waiting 5 seconds..."
sleep 5

echo "[$(date +%H:%M:%S)] Sleeping Engine B..."
curl -s -X POST "$ENGINE_B/nkipy/sleep" \
    -H "Content-Type: application/json" | python3 -m json.tool

echo ""
echo "----------------------------------------"
echo ""

# Test 2: Late sleep
echo "========================================"
echo "Test 2: Late Sleep (35s after wake)"
echo "========================================"
echo ""

echo "[$(date +%H:%M:%S)] Waking Engine B..."
curl -s -X POST "$ENGINE_B/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$ENGINE_A\"}" | python3 -m json.tool

echo ""
echo "[$(date +%H:%M:%S)] Waiting 35 seconds..."
sleep 35

echo "[$(date +%H:%M:%S)] Sleeping Engine B..."
curl -s -X POST "$ENGINE_B/nkipy/sleep" \
    -H "Content-Type: application/json" | python3 -m json.tool

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
echo ""
echo "Check the results above:"
echo "  Test 1: Should show dereg_waited=true, spike_reset_s < 2s"
echo "  Test 2: Should show dereg_waited=false, spike_reset_s < 2s"
echo ""
echo "If BOTH show spike_reset < 2s → Hypothesis CONFIRMED!"
echo "========================================"
