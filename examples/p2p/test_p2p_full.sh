#!/usr/bin/env bash
# Full P2P weight transfer test for vLLM+NKIPy plugin.
#
# Steps:
#   1. Health check (A awake, B sleeping)
#   2. Baseline inference on A
#   3. Wake B via P2P from A
#   4. Inference on B, compare output with A
#   5. Sleep B, wait for RDMA deregistration
#   6. Second wake/infer/sleep cycle
#   7. Non-blocking push (inference on A during transfer to B)
#
# Usage:
#   test_p2p_full.sh --model MODEL --engine-a URL --engine-b URL
#                    [--rdma-wait SECONDS]
#
# Examples:
#   ./test_p2p_full.sh --model /home/ubuntu/models/llama-3-70b \
#       --engine-a http://172.31.44.131:8000 --engine-b http://172.31.40.200:8000
#
#   ./test_p2p_full.sh --model Qwen/Qwen3-30B-A3B \
#       --engine-a http://172.31.44.131:8000 --engine-b http://172.31.40.200:8000
set -euo pipefail

RDMA_WAIT=60

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2";     shift 2;;
        --engine-a)  ENGINE_A="$2";  shift 2;;
        --engine-b)  ENGINE_B="$2";  shift 2;;
        --rdma-wait) RDMA_WAIT="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "${MODEL:-}" || -z "${ENGINE_A:-}" || -z "${ENGINE_B:-}" ]]; then
    echo "Usage: $0 --model MODEL --engine-a URL --engine-b URL [--rdma-wait SECONDS]"
    exit 1
fi

PROMPT="The capital of France is"

echo "=============================================="
echo "  P2P Weight Transfer Test"
echo "  Model: $MODEL"
echo "  A: $ENGINE_A    B: $ENGINE_B"
echo "=============================================="
echo ""

# --- Step 1: Health check ---
echo "=== Step 1: Health Check ==="
a_health=$(curl -s "$ENGINE_A/nkipy/health")
b_health=$(curl -s "$ENGINE_B/nkipy/health")
echo "  A: $a_health"
echo "  B: $b_health"
a_sleep=$(echo "$a_health" | python3 -c "import sys,json; print(json.load(sys.stdin)['sleeping'])")
b_sleep=$(echo "$b_health" | python3 -c "import sys,json; print(json.load(sys.stdin)['sleeping'])")
if [ "$a_sleep" != "False" ]; then echo "ERROR: A must be awake"; exit 1; fi
if [ "$b_sleep" != "True" ]; then echo "ERROR: B must be sleeping"; exit 1; fi
echo ""

# --- Step 2: Baseline inference on A ---
echo "=== Step 2: Baseline Inference on A ==="
a_resp=$(curl -s "$ENGINE_A/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 20, \"temperature\": 0}")
a_text=$(echo "$a_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])")
echo "  A output: \"$a_text\""
echo ""

# --- Step 3: Wake B via P2P from A ---
echo "=== Step 3: Wake B (P2P from A) ==="
wake1=$(curl -s -X POST "$ENGINE_B/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$ENGINE_A\"}")
echo "$wake1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
l=d.get('latency',{})
print(f'  Status: {d[\"status\"]}')
print(f'  Total: {l.get(\"total_s\",\"?\")}s  P2P: {l.get(\"p2p_transfer_s\",\"?\")}s  Gloo: {l.get(\"gloo_init_s\",\"?\")}s  NRT: {l.get(\"nrt_init_s\",\"?\")}s')
"
echo ""

# --- Step 4: Inference on B and compare ---
echo "=== Step 4: Inference on B ==="
b_resp=$(curl -s "$ENGINE_B/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 20, \"temperature\": 0}")
b_text=$(echo "$b_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])")
echo "  B output: \"$b_text\""
if [ "$a_text" = "$b_text" ]; then
    echo "  MATCH: Outputs are identical"
else
    echo "  MISMATCH: A=\"$a_text\" B=\"$b_text\""
fi
echo ""

# --- Step 5: Sleep B ---
echo "=== Step 5: Sleep B ==="
sleep1=$(curl -s -X POST "$ENGINE_B/nkipy/sleep")
echo "$sleep1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
l=d.get('latency',{})
print(f'  Status: {d[\"status\"]}  Total: {l.get(\"total_s\",\"?\")}s')
"
echo "  Waiting ${RDMA_WAIT}s for RDMA deregistration..."
sleep "$RDMA_WAIT"
echo ""

# --- Step 6: Second wake/infer/sleep cycle ---
echo "=== Step 6: Second Wake Cycle ==="
wake2=$(curl -s -X POST "$ENGINE_B/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$ENGINE_A\"}")
echo "$wake2" | python3 -c "
import sys,json
d=json.load(sys.stdin)
l=d.get('latency',{})
print(f'  Wake: {l.get(\"total_s\",\"?\")}s  P2P: {l.get(\"p2p_transfer_s\",\"?\")}s')
"

b_resp2=$(curl -s "$ENGINE_B/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 20, \"temperature\": 0}")
b_text2=$(echo "$b_resp2" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])")
echo "  B output (cycle 2): \"$b_text2\""
if [ "$a_text" = "$b_text2" ]; then
    echo "  MATCH: Outputs identical across cycles"
else
    echo "  MISMATCH"
fi

sleep2=$(curl -s -X POST "$ENGINE_B/nkipy/sleep")
echo "$sleep2" | python3 -c "
import sys,json; d=json.load(sys.stdin); l=d.get('latency',{})
print(f'  Sleep: {l.get(\"total_s\",\"?\")}s')
"
echo "  Waiting ${RDMA_WAIT}s for RDMA deregistration..."
sleep "$RDMA_WAIT"
echo ""

# --- Step 7: Non-blocking push (inference on A during transfer) ---
echo "=== Step 7: Non-blocking Push Test ==="
echo "  Starting background inference loop on A..."
LOG_DIR=/tmp/nkipy_p2p_full
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/inference.log"

(
    i=0
    while true; do
        t_start=$(date +%s%N)
        resp=$(curl -s "$ENGINE_A/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 10, \"temperature\": 0}")
        t_end=$(date +%s%N)
        latency_ms=$(( (t_end - t_start) / 1000000 ))
        text=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERROR")
        echo "$(date +%H:%M:%S.%3N) req=$i latency=${latency_ms}ms" >> "$LOG_DIR/inference.log"
        i=$((i+1))
        sleep 0.5
    done
) &
INFER_PID=$!

echo "  Collecting baseline (8s)..."
sleep 8

echo "  Triggering wake_up on B..."
t_wake_start=$(date +%s%N)
wake3=$(curl -s -X POST "$ENGINE_B/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$ENGINE_A\"}")
t_wake_end=$(date +%s%N)
wake_ms=$(( (t_wake_end - t_wake_start) / 1000000 ))
echo "$wake3" | python3 -c "
import sys,json
d=json.load(sys.stdin)
l=d.get('latency',{})
print(f'  Wake: {l.get(\"total_s\",\"?\")}s  P2P: {l.get(\"p2p_transfer_s\",\"?\")}s  wall: ${wake_ms}ms')
" 2>/dev/null

echo "  Collecting post-transfer (8s)..."
sleep 8

kill $INFER_PID 2>/dev/null
wait $INFER_PID 2>/dev/null

b_resp3=$(curl -s "$ENGINE_B/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 20, \"temperature\": 0}")
b_text3=$(echo "$b_resp3" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])")
echo "  B output (non-blocking): \"$b_text3\""
if [ "$a_text" = "$b_text3" ]; then
    echo "  MATCH"
else
    echo "  MISMATCH"
fi

python3 -c "
import re, statistics
lines = open('$LOG_DIR/inference.log').readlines()
times = [int(re.search(r'latency=(\d+)ms', l).group(1)) for l in lines if 'latency=' in l]
if not times:
    print('No data'); exit()
split = min(6, len(times))
baseline = times[:split]
rest = times[split:]
print(f'  Total requests: {len(times)}')
if baseline: print(f'  Baseline: avg={statistics.mean(baseline):.0f}ms  min={min(baseline)}ms  max={max(baseline)}ms')
if rest: print(f'  During/after push: avg={statistics.mean(rest):.0f}ms  min={min(rest)}ms  max={max(rest)}ms')
stall_threshold = max(baseline) * 2 if baseline else 5000
stalls = [t for t in times if t > stall_threshold]
if stalls: print(f'  WARNING: {len(stalls)} stall(s) > {stall_threshold}ms: {stalls}')
else: print(f'  No stalls (threshold: {stall_threshold}ms)')
"

curl -s -X POST "$ENGINE_B/nkipy/sleep" > /dev/null

echo ""
echo "=== ALL TESTS COMPLETE ==="
