#!/usr/bin/env bash
# Scalability test: start sleeping receiver engines sequentially and measure resources
set -uo pipefail

VENV=/home/ubuntu/vllm-nkipy/nkipy/.venv
LOG_DIR=/tmp/scalability_test
RESULTS_FILE=$LOG_DIR/results.csv
MAX_ENGINES=${1:-50}
START_PORT=8000
MODEL="Qwen/Qwen3-30B-A3B"
TP=32

mkdir -p "$LOG_DIR"

source "$VENV/bin/activate"

capture_metrics() {
    local n=$1
    local mem_used_gb=$(free -g | awk '/Mem:/{print $3}')
    local mem_avail_gb=$(free -g | awk '/Mem:/{print $7}')
    local proc_count=$(ps aux | grep -E 'VLLM|nkipy' | grep -v grep | wc -l)
    local shm_count=$(ls /dev/shm/ 2>/dev/null | wc -l)
    local shm_used=$(df /dev/shm | awk 'NR==2{print $3}')
    local tcp_ports=$(ss -tln | wc -l)
    local fd_total=$(cat /proc/sys/fs/file-nr | awk '{print $1}')
    echo "$n,$mem_used_gb,$mem_avail_gb,$proc_count,$shm_count,$shm_used,$tcp_ports,$fd_total"
}

echo "engines,mem_used_gb,mem_avail_gb,proc_count,shm_files,shm_used_kb,tcp_ports,fd_total" > "$RESULTS_FILE"

echo "$(date '+%H:%M:%S') Baseline metrics:"
capture_metrics 0 | tee -a "$RESULTS_FILE"

for i in $(seq 1 "$MAX_ENGINES"); do
    port=$((START_PORT + i - 1))
    echo ""
    echo "$(date '+%H:%M:%S') Starting engine $i on port $port..."

    HF_HUB_OFFLINE=1 \
    VLLM_PLUGINS=nkipy \
    VLLM_USE_V1=1 \
    NKIPY_SKIP_CTE=1 \
    OMP_NUM_THREADS=1 \
    VLLM_RPC_TIMEOUT=600000 \
    VLLM_SLEEP_WHEN_IDLE=1 \
    nohup python3 -m nkipy.vllm_plugin.server \
        --model "$MODEL" \
        --tensor-parallel-size $TP \
        --max-model-len 128 \
        --max-num-seqs 1 \
        --enforce-eager \
        --dtype bfloat16 \
        --port "$port" \
        > "$LOG_DIR/engine_${port}.log" 2>&1 &

    engine_pid=$!
    echo "$(date '+%H:%M:%S')   PID=$engine_pid, waiting for port $port..."

    # Wait for the engine to become healthy (up to 5 min)
    timeout=300
    elapsed=0
    ready=false
    while [ $elapsed -lt $timeout ]; do
        if health=$(timeout 3 curl -s "http://localhost:$port/nkipy/health" 2>/dev/null); then
            if echo "$health" | grep -q '"sleeping":true'; then
                ready=true
                break
            fi
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    if [ "$ready" = true ]; then
        metrics=$(capture_metrics "$i")
        echo "$metrics" >> "$RESULTS_FILE"
        echo "$(date '+%H:%M:%S')   Engine $i READY (${elapsed}s). Metrics: $metrics"
    else
        echo "$(date '+%H:%M:%S')   Engine $i FAILED after ${timeout}s"
        # Capture metrics even on failure to understand the bottleneck
        metrics=$(capture_metrics "$i")
        echo "${i}_FAIL,$metrics" >> "$RESULTS_FILE"
        echo "$(date '+%H:%M:%S')   Failure metrics: $metrics"
        echo "$(date '+%H:%M:%S')   Last 10 lines of log:"
        tail -10 "$LOG_DIR/engine_${port}.log" 2>/dev/null
        echo "$(date '+%H:%M:%S') Stopping test at engine $i."
        break
    fi
done

echo ""
echo "=== FINAL RESULTS ==="
column -t -s',' "$RESULTS_FILE"
echo ""
echo "Results saved to $RESULTS_FILE"
