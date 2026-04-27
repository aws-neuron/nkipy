#!/bin/bash
# Usage: source ./run.sh <mlir_file>
#
# Runs BIRSim on a pre-compiled NISA MLIR file and compares against
# a NumPy reference (kernel.py in the same directory as the MLIR file).
#
# Examples:
#   source ./run.sh matmul_add_sbuf_oom/buggy.mlir
#   source ./run.sh psum_accumulate_flags_fix/fixed.mlir

MLIR_FILE="${1:?Usage: source ./run.sh <mlir_file>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NKIPY_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source NKI environment
source "$NKIPY_ROOT/scripts/setup_nki.sh"

# Add NKIPyKernelGen and e2e test utils to PYTHONPATH
export PYTHONPATH="$NKIPY_ROOT:$NKIPY_ROOT/tests/e2e:$PYTHONPATH"

# Resolve to absolute path if relative
if [[ "$MLIR_FILE" != /* ]]; then
    MLIR_FILE="$SCRIPT_DIR/$MLIR_FILE"
fi

if [ ! -f "$MLIR_FILE" ]; then
    echo "Error: File not found: $MLIR_FILE"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "Running BIRSim on: $MLIR_FILE"
echo "================================================"

python3 "$SCRIPT_DIR/run_sim.py" "$MLIR_FILE"
