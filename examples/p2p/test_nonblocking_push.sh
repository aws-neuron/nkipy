#!/usr/bin/env bash
# Test non-blocking P2P push: serve inference on A while pushing weights to B.
#
# IMPORTANT: engine-a must use A's real IP, not localhost.
# B's workers resolve peer_url on their machine — localhost would
# point back to B, causing a deadlock.
#
# Usage:
#   test_nonblocking_push.sh --model MODEL --engine-a URL --engine-b URL
#
# Example:
#   ./test_nonblocking_push.sh --model Qwen/Qwen3-30B-A3B \
#       --engine-a http://172.31.44.131:8000 --engine-b http://172.31.40.200:8000
set -uo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2;;
        --engine-a) ENGINE_A="$2"; shift 2;;
        --engine-b) ENGINE_B="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "${MODEL:-}" || -z "${ENGINE_A:-}" || -z "${ENGINE_B:-}" ]]; then
    echo "Usage: $0 --model MODEL --engine-a URL --engine-b URL"
    exit 1
fi

LOG_DIR=/tmp/nkipy_nonblocking_push
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/inference.log"

echo "=============================================="
echo "  Non-blocking P2P Push Test"
echo "  Model: $MODEL"
echo "  A: $ENGINE_A serves inference while pushing"
echo "  B: $ENGINE_B receives weights"
echo "=============================================="
echo ""

# Verify A is awake and B is sleeping
a_health=$(curl -s "$ENGINE_A/nkipy/health")
b_health=$(curl -s "$ENGINE_B/nkipy/health")
a_sleep=$(echo "$a_health" | python3 -c "import sys,json; print(json.load(sys.stdin)['sleeping'])")
b_sleep=$(echo "$b_health" | python3 -c "import sys,json; print(json.load(sys.stdin)['sleeping'])")
echo "Pre-check: A sleeping=$a_sleep  B sleeping=$b_sleep"
if [ "$a_sleep" != "False" ]; then echo "ERROR: A must be awake"; exit 1; fi
if [ "$b_sleep" != "True" ]; then echo "ERROR: B must be sleeping"; exit 1; fi
echo ""

# Start background inference loop on A
echo "Starting background inference loop on A..."
(
    i=0
    while true; do
        t_start=$(date +%s%N)
        resp=$(curl -s "$ENGINE_A/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0}")
        t_end=$(date +%s%N)
        latency_ms=$(( (t_end - t_start) / 1000000 ))
        text=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERROR")
        echo "$(date +%H:%M:%S.%3N) req=$i latency=${latency_ms}ms output=\"$text\"" >> "$LOG_DIR/inference.log"
        i=$((i+1))
        sleep 0.5
    done
) &
INFER_PID=$!
echo "  Inference loop PID=$INFER_PID"
echo ""

echo "Collecting baseline inference latency (8s)..."
sleep 8

echo ""
echo ">>> Triggering wake_up on B (A pushes weights non-blocking) <<<"
echo ">>> peer_url=$ENGINE_A <<<"
t_wake_start=$(date +%s%N)
wake_result=$(curl -s -X POST "$ENGINE_B/nkipy/wake_up" \
    -H "Content-Type: application/json" \
    -d "{\"peer_url\": \"$ENGINE_A\"}")
t_wake_end=$(date +%s%N)
wake_ms=$(( (t_wake_end - t_wake_start) / 1000000 ))
echo "$wake_result" | python3 -c "
import sys,json
d=json.load(sys.stdin)
l=d.get('latency',{})
print(f'  wake_up completed: total={l.get(\"total_s\",\"?\")}s p2p={l.get(\"p2p_transfer_s\",\"?\")}s')
print(f'  gloo_init={l.get(\"gloo_init_s\",\"?\")}s nrt_init={l.get(\"nrt_init_s\",\"?\")}s')
" 2>/dev/null
echo "  wall time: ${wake_ms}ms"
echo ""

echo "Collecting post-transfer inference latency (8s)..."
sleep 8

kill $INFER_PID 2>/dev/null
wait $INFER_PID 2>/dev/null

echo ""
echo "Verifying B inference after transfer..."
b_infer=$(curl -s "$ENGINE_B/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0}")
b_text=$(echo "$b_infer" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
echo "  B output: \"$b_text\""

echo ""
echo "=============================================="
echo "  Inference Latency Analysis"
echo "=============================================="
python3 -c "
import re, statistics

lines = open('$LOG_DIR/inference.log').readlines()
times = []
timestamps = []
for line in lines:
    m = re.search(r'latency=(\d+)ms', line)
    ts = re.match(r'(\S+)', line)
    if m:
        times.append(int(m.group(1)))
        timestamps.append(ts.group(1) if ts else '?')

if not times:
    print('No data')
    exit()

split = min(6, len(times))
baseline = times[:split]
rest = times[split:]

during_split = min(20, len(rest))
during = rest[:during_split]
after = rest[during_split:]

print(f'Total requests: {len(times)}')
print()
if baseline:
    print(f'Baseline (before push):  n={len(baseline)}  min={min(baseline)}ms  max={max(baseline)}ms  avg={statistics.mean(baseline):.0f}ms')
if during:
    print(f'During push:             n={len(during)}  min={min(during)}ms  max={max(during)}ms  avg={statistics.mean(during):.0f}ms')
if after:
    print(f'After push:              n={len(after)}  min={min(after)}ms  max={max(after)}ms  avg={statistics.mean(after):.0f}ms')

if baseline:
    avg_baseline = statistics.mean(baseline)
    stall_threshold = max(max(baseline) * 2, avg_baseline + 500)
else:
    stall_threshold = 5000
stalls = [(i, t, timestamps[i]) for i, t in enumerate(times) if t > stall_threshold]
if stalls:
    print(f'\nWARNING: {len(stalls)} request(s) exceeded {stall_threshold:.0f}ms:')
    for idx, lat, ts in stalls:
        phase = 'baseline' if idx < split else ('during' if idx < split + during_split else 'after')
        print(f'  req {idx} ({phase}) at {ts}: {lat}ms')
else:
    print(f'\nNo stalls detected (threshold: {stall_threshold:.0f}ms)')
"
echo ""
echo "Raw log:"
cat "$LOG_DIR/inference.log"
