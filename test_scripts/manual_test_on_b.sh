#!/bin/bash
# Manual test to run DIRECTLY on Instance B (172.31.40.200)
# This tests the sleep latency fix end-to-end

set -e
cd /home/ubuntu/vllm-nkipy/nkipy

echo "========================================"
echo "P2P Sleep Latency Fix - Manual Test"
echo "Run this script ON INSTANCE B"
echo "========================================"
echo ""

echo "=== Step 1: Clean environment ==="
pkill -9 VLLM 2>&1 || true
pkill -9 python3 2>&1 || true
sleep 5

echo ""
echo "=== Step 2: Start receiver ==="
source /home/ubuntu/vllm-nkipy/nkipy/.venv/bin/activate
bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver.log 2>&1 &

echo "Waiting for startup..."
for i in {1..60}; do
    if grep -q "Application startup complete" /tmp/receiver.log 2>/dev/null; then
        echo "✓ Receiver started"
        break
    fi
    sleep 5
done
sleep 3

echo ""
echo "=== Step 3: Wake up (P2P from server) ==="
curl -X POST http://localhost:8000/nkipy/wake_up \
    -H "Content-Type: application/json" \
    -d '{"peer_url": "http://172.31.44.131:8000"}' \
    | python3 -m json.tool | tee /tmp/wake_result.json

wake_status=$(python3 -c "import json; print(json.load(open('/tmp/wake_result.json')).get('status', 'unknown'))" 2>/dev/null || echo "failed")
if [ "$wake_status" != "awake" ] && [ "$wake_status" != "already_awake" ]; then
    echo "❌ Wake up failed!"
    exit 1
fi

echo ""
echo "✓ Wake up completed"

echo ""
echo "=== Step 4: Serve 30 requests ==="
success=0
for i in $(seq 1 30); do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hello", "max_tokens": 5, "temperature": 0}' \
        > /dev/null 2>&1 && success=$((success + 1)) && echo -n "." || echo -n "E"
    sleep 0.2
done
echo ""
echo "✓ Served $success/30 requests"

echo ""
echo "=== Step 5: Wait for GC ==="
sleep 5

echo ""
echo "=== Step 6: Sleep (TEST THE FIX!) ==="
curl -X POST http://localhost:8000/nkipy/sleep \
    -H "Content-Type: application/json" \
    | python3 -m json.tool | tee /tmp/sleep_result.json

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"

python3 << 'EOF'
import json
try:
    with open('/tmp/sleep_result.json') as f:
        result = json.load(f)

    lat = result.get('latency', {})
    spike_reset = lat.get('spike_reset_s', 'N/A')
    clear_refs = lat.get('clear_refs_s', 'N/A')
    total = lat.get('total_s', 'N/A')

    print(f"Sleep latency breakdown:")
    print(f"  - spike_reset_s: {spike_reset}s")
    print(f"  - clear_refs_s:  {clear_refs}s")
    print(f"  - Total:         {total}s")
    print("")
    print("========================================"
    print("VERDICT")
    print("========================================")
    print(f"Before fix: spike_reset ~15s, total ~15s")
    print(f"After fix:  spike_reset {spike_reset}s, total {total}s")
    print("")

    if spike_reset != 'N/A':
        spike_val = float(spike_reset)
        if spike_val < 5:
            print(f"✅ FIX SUCCESSFUL!")
            print(f"   spike_reset dropped from 15s to {spike_val}s")
            print(f"   That's {((15-spike_val)/15*100):.0f}% faster!")
        elif spike_val < 10:
            print(f"⚠️  PARTIAL IMPROVEMENT")
            print(f"   spike_reset is {spike_val}s (better than 15s, but not optimal)")
        else:
            print(f"❌ NO IMPROVEMENT")
            print(f"   spike_reset is still {spike_val}s")
    else:
        print("❌ Could not parse results")

except Exception as e:
    print(f"Error parsing results: {e}")
EOF

echo "========================================"
echo ""
echo "To check detailed logs:"
echo "  grep 'sleep latency breakdown' /tmp/receiver.log | tail -5"
