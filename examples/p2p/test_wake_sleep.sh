#!/usr/bin/env bash
# Test script for /wake_up and /sleep endpoints
# Usage: ./test_wake_sleep.sh [wake|sleep] [peer_url]
set -euo pipefail

ACTION=${1:-wake}
PEER_URL=${2:-http://172.31.44.131:8000}

if [ "$ACTION" = "wake" ]; then
    echo "==> Testing /wake_up with peer_url=$PEER_URL"
    curl -X POST http://localhost:8000/nkipy/wake_up \
        -H "Content-Type: application/json" \
        -d "{\"peer_url\": \"$PEER_URL\"}" \
        | python3 -m json.tool
elif [ "$ACTION" = "sleep" ]; then
    echo "==> Testing /sleep"
    curl -X POST http://localhost:8000/nkipy/sleep \
        -H "Content-Type: application/json" \
        | python3 -m json.tool
else
    echo "Usage: $0 [wake|sleep] [peer_url]"
    exit 1
fi
