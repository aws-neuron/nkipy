#!/usr/bin/env bash
# Simple manual test - run each step one at a time
set -euo pipefail

cd /home/ubuntu/vllm-nkipy/nkipy

case "${1:-help}" in
    start-server)
        echo "Starting server on Instance A..."
        fuser -k 8000/tcp 2>&1 || true
        killall -9 python3 2>&1 || true
        sleep 5
        source .venv/bin/activate
        bash examples/p2p/run_vllm_qwen_1.sh > /tmp/server_a.log 2>&1 &
        echo "Server starting... Monitor with: tail -f /tmp/server_a.log"
        ;;

    start-receiver)
        echo "Starting receiver on Instance B..."
        ssh ubuntu@172.31.40.200 'cd /home/ubuntu/vllm-nkipy/nkipy && fuser -k 8000/tcp 2>&1; killall -9 python3 2>&1; sleep 5; source .venv/bin/activate && bash examples/p2p/run_vllm_qwen_1_receiver.sh > /tmp/receiver_b.log 2>&1 &'
        echo "Receiver starting... Check with: ssh ubuntu@172.31.40.200 tail -f /tmp/receiver_b.log"
        ;;

    test-immediate)
        echo "Test 1: Immediate sleep (no serving)"
        echo "Wake up..."
        curl -s -X POST http://localhost:8000/nkipy/wake_up \
            -H "Content-Type: application/json" \
            -d '{"peer_url": "http://172.31.44.131:8000"}' \
            | python3 -m json.tool | tee /tmp/wake1.json

        echo ""
        echo "Sleep immediately..."
        curl -s -X POST http://localhost:8000/nkipy/sleep \
            -H "Content-Type: application/json" \
            | python3 -m json.tool | tee /tmp/sleep1.json
        ;;

    test-after-serving)
        echo "Test 2: Sleep after serving requests"
        echo "Wake up..."
        curl -s -X POST http://localhost:8000/nkipy/wake_up \
            -H "Content-Type: application/json" \
            -d '{"peer_url": "http://172.31.44.131:8000"}' \
            | python3 -m json.tool | tee /tmp/wake2.json

        echo ""
        echo "Sending 30 inference requests..."
        for i in $(seq 1 30); do
            curl -s -X POST http://localhost:8000/v1/completions \
                -H "Content-Type: application/json" \
                -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hello!", "max_tokens": 5}' >/dev/null 2>&1 && echo -n "." || echo -n "E"
            sleep 0.1
        done
        echo ""

        echo "Waiting 3s for GC..."
        sleep 3

        echo "Sleep after serving..."
        curl -s -X POST http://localhost:8000/nkipy/sleep \
            -H "Content-Type: application/json" \
            | python3 -m json.tool | tee /tmp/sleep2.json
        ;;

    compare)
        echo "Comparison Results"
        echo "=================="
        python3 << 'EOF'
import json
try:
    with open('/tmp/sleep1.json') as f:
        s1 = json.load(f)
    with open('/tmp/sleep2.json') as f:
        s2 = json.load(f)

    s1_total = s1.get('latency', {}).get('total_s', 'N/A')
    s1_clear = s1.get('latency', {}).get('clear_refs_s', 'N/A')
    s2_total = s2.get('latency', {}).get('total_s', 'N/A')
    s2_clear = s2.get('latency', {}).get('clear_refs_s', 'N/A')

    print(f"Test 1 (immediate):     total={s1_total}s  clear_refs={s1_clear}s")
    print(f"Test 2 (after serving): total={s2_total}s  clear_refs={s2_clear}s")

    if s1_total != 'N/A' and s2_total != 'N/A':
        improvement = float(s1_total) - float(s2_total)
        print(f"\nImprovement: {improvement:.2f}s faster after serving")
        if improvement > 5:
            print("✓ HYPOTHESIS CONFIRMED!")
        else:
            print("✗ No significant improvement")
except Exception as e:
    print(f"Error: {e}")
EOF
        ;;

    *)
        echo "Usage: $0 {start-server|start-receiver|test-immediate|test-after-serving|compare}"
        echo ""
        echo "Run on Instance A (server):"
        echo "  1. bash simple_test.sh start-server"
        echo "  2. Wait for 'Application startup complete'"
        echo ""
        echo "Run on Instance B (receiver):"
        echo "  3. bash simple_test.sh start-receiver"
        echo "  4. Wait for 'Application startup complete'"
        echo "  5. bash simple_test.sh test-immediate"
        echo "  6. Restart receiver: bash simple_test.sh start-receiver"
        echo "  7. bash simple_test.sh test-after-serving"
        echo "  8. bash simple_test.sh compare"
        ;;
esac
