#!/bin/bash
# Test cache tensor separation optimization
# This tests if separating weight allocation from cache allocation reduces spike_reset latency

set -e
cd /home/ubuntu/vllm-nkipy/nkipy

echo "========================================"
echo "Cache Separation Optimization Test"
echo "Test on Instance B (172.31.40.200)"
echo "========================================"

# Clean environment
echo "=== Cleaning environment ==="
ssh ubuntu@172.31.40.200 "pkill -9 VLLM 2>&1 || true; pkill -9 python3 2>&1 || true"
sleep 5

# Start receiver
echo "=== Starting receiver ==="
ssh ubuntu@172.31.40.200 "cd /home/ubuntu/vllm-nkipy/nkipy && source .venv/bin/activate && bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver.log 2>&1 &"

echo "Waiting for startup (60s max)..."
for i in {1..12}; do
    if ssh ubuntu@172.31.40.200 "grep -q 'Application startup complete' /tmp/receiver.log 2>/dev/null"; then
        echo "✓ Receiver started"
        break
    fi
    sleep 5
done
sleep 3

# Wake up with P2P
echo "=== Wake up (P2P from server) ==="
ssh ubuntu@172.31.40.200 "curl -s -X POST http://localhost:8000/nkipy/wake_up -H 'Content-Type: application/json' -d '{\"peer_url\": \"http://172.31.44.131:8000\"}' | python3 -m json.tool | tee /tmp/wake_result.json"

wake_status=$(ssh ubuntu@172.31.40.200 "python3 -c \"import json; print(json.load(open('/tmp/wake_result.json')).get('status', 'unknown'))\" 2>/dev/null" || echo "failed")
if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
    echo "❌ Wake up failed: $wake_status"
    exit 1
fi
echo "✓ Wake up completed"

# Serve 30 requests to trigger GC
echo "=== Serving 30 requests ==="
success=0
for i in $(seq 1 30); do
    if ssh ubuntu@172.31.40.200 "curl -s -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{\"model\": \"Qwen/Qwen3-30B-A3B\", \"prompt\": \"Hello\", \"max_tokens\": 5, \"temperature\": 0}' > /dev/null 2>&1"; then
        success=$((success + 1))
        echo -n "."
    else
        echo -n "E"
    fi
    sleep 0.2
done
echo ""
echo "✓ Served $success/30 requests"

# Wait for GC
echo "=== Waiting for GC (5s) ==="
sleep 5

# Sleep - this is where we test the optimization
echo "=== Testing /sleep ==="
ssh ubuntu@172.31.40.200 "curl -s -X POST http://localhost:8000/nkipy/sleep -H 'Content-Type: application/json' | python3 -m json.tool | tee /tmp/sleep_result.json"

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"

ssh ubuntu@172.31.40.200 'python3 << "EOF"
import json
try:
    with open("/tmp/sleep_result.json") as f:
        result = json.load(f)

    lat = result.get("latency", {})
    rank = lat.get("rank", 0)
    spike_reset = lat.get("spike_reset_s", "N/A")
    total = lat.get("total_s", "N/A")

    print(f"Rank {rank} sleep latency:")
    print(f"  spike_reset_s: {spike_reset}")
    print(f"  total_s:       {total}")
    print("")
    print("Expected results:")
    print("  Before fix: spike_reset ~15-18s")
    print("  Target:     spike_reset ~2s")
    print("")

    if spike_reset != "N/A":
        spike_val = float(spike_reset)
        if spike_val < 3:
            print(f"✅ SUCCESS! spike_reset = {spike_val}s (target achieved!)")
        elif spike_val < 8:
            print(f"⚠️  PARTIAL: spike_reset = {spike_val}s (better than 15s, not quite 2s)")
        else:
            print(f"❌ INSUFFICIENT: spike_reset = {spike_val}s (still too slow)")

    print("")
    print("Check all ranks with:")
    print("  ssh ubuntu@172.31.40.200 \"grep 'sleep latency breakdown' /tmp/receiver.log | tail -32\"")

except Exception as e:
    print(f"Error: {e}")
EOF
'

echo "========================================"
