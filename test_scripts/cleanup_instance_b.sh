#!/usr/bin/env bash
# Clean up Instance B completely before running tests
set -euo pipefail

echo "========================================"
echo "Cleaning Up Instance B"
echo "========================================"

# Kill all vllm processes
echo "Killing vLLM processes..."
fuser -k 8000/tcp 2>/dev/null || true
pkill -9 -f vllm_plugin.server 2>/dev/null || true
pkill -9 -f nkipy 2>/dev/null || true

# Wait for processes to die
sleep 3

# Clean up any leftover Python processes
echo "Cleaning up Python processes..."
pkill -9 python3 2>/dev/null || true
pkill -9 python 2>/dev/null || true

# Wait for full cleanup
sleep 2

echo "✓ Cleanup complete"
echo ""
echo "Instance B is ready for a fresh start"
