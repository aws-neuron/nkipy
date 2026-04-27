#!/bin/bash
# Wrapper script for Compiler Explorer to invoke nkipy_compiler.py
#
# Compiler Explorer passes:
#   $1 = input file path
#   Additional args passed via compiler options in CE UI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NKIPY_ROOT="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include nkipy_kernelgen
export PYTHONPATH="$NKIPY_ROOT:$PYTHONPATH"

# Run the compiler
exec python3 "$SCRIPT_DIR/nkipy_compiler.py" "$@"
