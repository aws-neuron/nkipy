#!/bin/bash
# Setup script for Compiler Explorer with NKIPy integration
#
# Usage: ./setup.sh [example_file.py]
#   example_file.py - Optional path to a Python file to use as the default example
#
# This script:
# 1. Clones Compiler Explorer if not present
# 2. Installs dependencies
# 3. Configures NKIPy as a compiler backend (disables all other compilers)
# 4. Starts the server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NKIPY_ROOT="$(dirname "$SCRIPT_DIR")"
CE_DIR="$SCRIPT_DIR/compiler-explorer"

# Convert example file to absolute path (needed since we cd later)
if [ -n "$1" ]; then
    EXAMPLE_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
else
    EXAMPLE_FILE="$SCRIPT_DIR/config/example.nkipy"
fi

echo "=== NKIPy Compiler Explorer Setup ==="
echo "Script directory: $SCRIPT_DIR"
echo "NKIPy root: $NKIPY_ROOT"
echo "Example file: $EXAMPLE_FILE"

# If user provided a file path, make sure it exists
if [ -n "$1" ] && [ ! -f "$EXAMPLE_FILE" ]; then
    echo "Error: File not found: $EXAMPLE_FILE"
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required. Please install Node.js 20+."
    echo "  Ubuntu: sudo apt install nodejs npm"
    echo "  Or use nvm: https://github.com/nvm-sh/nvm"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "Error: Node.js 18+ required. Found: $(node -v)"
    exit 1
fi

echo "Node.js version: $(node -v)"

# Clone Compiler Explorer if not present
if [ ! -d "$CE_DIR" ]; then
    echo ""
    echo "=== Cloning Compiler Explorer ==="
    git clone --depth 1 https://github.com/compiler-explorer/compiler-explorer.git "$CE_DIR"
fi

cd "$CE_DIR"

# Install dependencies
echo ""
echo "=== Installing dependencies ==="
npm install

# Create config directory and clean up default configs
echo ""
echo "=== Configuring for NKIPy only ==="
mkdir -p etc/config
mkdir -p examples/python

# Remove all default config files to avoid warnings
rm -f etc/config/*.amazon.properties etc/config/*.defaults.properties 2>/dev/null || true

# Copy example file
if [ -f "$EXAMPLE_FILE" ]; then
    cp "$EXAMPLE_FILE" examples/python/default.py
    echo "Copied example from: $EXAMPLE_FILE"
else
    # Create default example
    cat > examples/python/default.py << 'PYEXAMPLE'
import numpy as np
from nkipy_kernelgen import trace, knob

M, N, K = 256, 256, 256
matmul_tile = [128, 128, 128]
add_tile = [128, 128]

@trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
def matmul_add_kernel(a, b, bias):
    c = np.matmul(a, b)
    knob.knob(c, mem_space="Sbuf", tile_size=matmul_tile)
    result = c + bias
    knob.knob(result, mem_space="SharedHbm", tile_size=add_tile)
    return result
PYEXAMPLE
    echo "Created default matmul_add example"
fi

# Update the wrapper path in config and copy
WRAPPER_PATH="$SCRIPT_DIR/nkipy_ce_wrapper.sh"
chmod +x "$WRAPPER_PATH"
chmod +x "$SCRIPT_DIR/nkipy_compiler.py"

# Generate minimal CE config (Python only)
cat > etc/config/compiler-explorer.local.properties << EOF
# Minimal Compiler Explorer config - only NKIPy
languages=python
defaultLanguage=python
noRemoteFetch=true

# Increase compile timeout for --sim/--hw modes (NEFF compilation + execution)
# Default is 7500ms which is too short
compileTimeoutMs=600000
EOF

# Generate Python config with NKIPy only
# Read the example file and escape for Java properties format (newlines -> \n)
EXAMPLE_CONTENT=$(cat "$EXAMPLE_FILE" | sed 's/\\/\\\\/g' | sed ':a;N;$!ba;s/\n/\\n/g')

cat > etc/config/python.local.properties << EOF
# NKIPy-only Python config
compilers=nkipy
defaultCompiler=nkipy
defaultSource=$EXAMPLE_CONTENT

compiler.nkipy.exe=$WRAPPER_PATH
compiler.nkipy.name=NKIPy MLIR
compiler.nkipy.supportsBinary=false
compiler.nkipy.supportsExecute=false
compiler.nkipy.notification=Compiles Python+NumPy to NISA MLIR for Neuron hardware
EOF

# Create sponsors.yaml with required format
echo "levels: []" > etc/config/sponsors.yaml

echo ""
echo "=== Configuration complete ==="
echo ""
echo "Wrapper script: $WRAPPER_PATH"
echo "Example: examples/python/default.py"
echo ""
echo "=== Starting Compiler Explorer ==="
echo "Access at: http://localhost:10240"
echo ""
echo "Compiler options (add in the options box):"
echo "  --stop=0           Trace only (initial MLIR before any passes)"
echo "  --stop=7           Stop after tiling (apply-and-strip-transforms)"
echo "  --stop=10          Stop after bufferization"
echo "  --stop=16          Stop after memory space annotation + cleanup"
echo "  --stop=24          Full compilation (same as omitting --stop)"
echo "  --sim              Run simulation (BIR sim or LLVM JIT with --stop)"
echo "  --raw              Clean MLIR without source annotations"
echo ""

# Start the server
npm run dev
