#!/bin/bash
# Run NKIPyKernelGen tests with proper environment setup.
# Usage: run_tests.sh [scope]
#   scope: all (default), passes, e2e, or a specific path like passes/infer_layout

SCOPE="${1:-all}"
RESULTS_FILE="/tmp/nkipykernelgen_test_results.txt"

# Derive repo root from script location: scripts/ -> run_nkipykernelgen_tests/ -> skills/ -> .claude/ -> repo root
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

cd "$REPO_ROOT"

# Run tests, capturing full output to file
echo "=== Running tests (scope: $SCOPE) ==="
echo "Results will be saved to: $RESULTS_FILE"

case "$SCOPE" in
  all)
    python -m pytest tests/ -v --tb=short 2>&1 | tee "$RESULTS_FILE"
    ;;
  passes)
    python -m pytest tests/passes/ -v --tb=short 2>&1 | tee "$RESULTS_FILE"
    ;;
  e2e)
    python -m pytest tests/e2e/ -v --tb=short 2>&1 | tee "$RESULTS_FILE"
    ;;
  *)
    python -m pytest "tests/$SCOPE" -v --tb=short 2>&1 | tee "$RESULTS_FILE"
    ;;
esac
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=== Full results saved to: $RESULTS_FILE ==="
exit $EXIT_CODE
