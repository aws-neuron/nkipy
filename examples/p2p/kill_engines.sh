#!/usr/bin/env bash
# Gracefully stop NKIPy engines and release NeuronCores.
#
# After kill -9, the neuron kernel module retains stale core allocations
# that cannot be reclaimed without reloading the module (or rebooting).
# This script avoids that by using SIGTERM first, which allows the engine
# to run its cleanup handlers (spike_reset, RDMA deregistration).
#
# Usage:
#   kill_engines.sh              # stop all nkipy engines
#   kill_engines.sh --port 8000  # stop only the engine on port 8000
#   kill_engines.sh --force      # SIGKILL + device reset (last resort)
set -euo pipefail

PORT=""
FORCE=0
TIMEOUT=15

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)  PORT="$2"; shift 2;;
        --force) FORCE=1; shift;;
        --timeout) TIMEOUT="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

find_engine_pids() {
    if [[ -n "$PORT" ]]; then
        pgrep -f "nkipy.vllm_plugin.server.*--port $PORT" 2>/dev/null || true
    else
        pgrep -f "nkipy.vllm_plugin.server" 2>/dev/null || true
    fi
}

PIDS=$(find_engine_pids)
if [[ -z "$PIDS" ]]; then
    echo "No NKIPy engines running."
    exit 0
fi

echo "Found engine PIDs: $PIDS"

if [[ $FORCE -eq 1 ]]; then
    echo "Force killing (SIGKILL)..."
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
    # Kill any orphan workers
    pkill -9 -f "nkipy.vllm_plugin" 2>/dev/null || true
    sleep 1
    # Reset neuron devices to reclaim stale allocations
    echo "Resetting neuron devices..."
    for i in $(seq 0 15); do
        sudo sh -c "echo 1 > /sys/devices/virtual/neuron_device/neuron$i/reset" 2>/dev/null || true
    done
    echo "Done. WARNING: Cores may still be stuck until module reload or reboot."
    echo "Run 'neuron-ls' to verify device state."
    exit 0
fi

# Graceful shutdown: SIGTERM → wait → verify
echo "Sending SIGTERM..."
echo "$PIDS" | xargs kill -TERM 2>/dev/null || true

echo "Waiting up to ${TIMEOUT}s for graceful shutdown..."
for i in $(seq 1 "$TIMEOUT"); do
    REMAINING=$(find_engine_pids)
    if [[ -z "$REMAINING" ]]; then
        echo "All engines stopped gracefully."
        exit 0
    fi
    sleep 1
done

# Still running — escalate
REMAINING=$(find_engine_pids)
if [[ -n "$REMAINING" ]]; then
    echo "WARNING: Engines still running after ${TIMEOUT}s, sending SIGKILL..."
    echo "$REMAINING" | xargs kill -9 2>/dev/null || true
    sleep 2
    pkill -9 -f "nkipy.vllm_plugin" 2>/dev/null || true
    echo "Resetting neuron devices (cores may be in stale state)..."
    for i in $(seq 0 15); do
        sudo sh -c "echo 1 > /sys/devices/virtual/neuron_device/neuron$i/reset" 2>/dev/null || true
    done
    echo "Done. Some cores may remain stuck until module reload or reboot."
fi
