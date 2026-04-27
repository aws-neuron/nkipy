#!/bin/bash
# Rebuild NKIPyKernelGen (C++ passes and Python package).
set -e

# Derive repo root from script location: scripts/ -> build_nkipykernelgen/ -> skills/ -> .claude/ -> repo root
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

cd "$REPO_ROOT"

echo "=== Rebuilding NKIPyKernelGen ==="
pip install -e . 2>&1 | tail -5
echo "=== Build complete ==="
