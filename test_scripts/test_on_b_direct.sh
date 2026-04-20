#!/bin/bash
# Run this DIRECTLY on Instance B
cd /home/ubuntu/vllm-nkipy/nkipy

echo "=== Wake up ==="
curl -s -X POST http://localhost:8000/nkipy/wake_up \
    -H "Content-Type: application/json" \
    -d '{"peer_url": "http://172.31.44.131:8000"}' | python3 -m json.tool

echo ""
echo "=== Serve 30 requests ==="
for i in $(seq 1 30); do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hello", "max_tokens": 5, "temperature": 0}' \
        > /dev/null && echo -n "." || echo -n "E"
    sleep 0.2
done

echo ""
echo "=== Wait for GC ==="
sleep 5

echo "=== Sleep ==="
curl -s -X POST http://localhost:8000/nkipy/sleep \
    -H "Content-Type: application/json" | python3 -m json.tool

echo ""
echo "=== Check all ranks ==="
grep 'sleep latency breakdown' /tmp/receiver.log | tail -32 | while read line; do
    rank=$(echo "$line" | grep -oP "rank \K[0-9]+")
    spike=$(echo "$line" | grep -oP "'spike_reset_s': \K[0-9.]+")
    total=$(echo "$line" | grep -oP "'total_s': \K[0-9.]+")
    echo "Rank $rank: spike_reset=${spike}s, total=${total}s"
done
